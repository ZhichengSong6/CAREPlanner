#!/usr/bin/env python3
"""P2: reuse P0/P1 update/validation functions; change only boundary eikonal .1 -> 0."""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import protocol as p2
old = p2.old


def save(model, optimizer, scaler, args, cache, step, validation, streams,
         parent, control, references, mode, baseline, *, final=False):
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, {'global': streams[0].hexdigest(), 'boundary': streams[1].hexdigest()})
    error = [None]
    if dist.get_rank() == 0:
        try:
            matched = ('MATCH' if records == control['sample_stream_sha256_by_rank'] else 'MISMATCH')
            if final and mode == 'pilot' and matched != 'MATCH':
                raise ValueError('Full P2 sample streams differ; refusing successful final.pt')
            state = dict(format=p2.FORMAT, arm='P2', mode=mode, pilot_updates=step, step=step,
                parent_updates=50000, total_updates=50000+step, parent_sha256=old.V1_SHA,
                initialization='V1_weights_only_fresh_Adam',
                optimizer_policy='fresh_Adam_constant_lr_no_scheduler_no_clipping', args=vars(args),
                cache_manifest_sha256=cache.identity, out_dim=9, frozen_parameters=0,
                architecture=parent['architecture'], output_layout=parent['output_layout'],
                model_state=model.module.state_dict(), optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict(), sample_stream_sha256_by_rank=records,
                validation=validation, source_sha256=old.source_fingerprints(),
                p2_source_sha256=p2.fingerprints(), boundary_weights=p2.WEIGHTS,
                reference_sha256=references, completed=final,
                pair_sample_streams=matched if mode == 'pilot' and final else
                    'NOT_COMPARABLE_2_VS_2000' if mode == 'smoke' else 'PREFIX_ONLY')
            if final:
                p2.assert_p2(state, control, cache.identity, references, require_pilot=mode == 'pilot')
            baseline.atomic_save(state, Path(args.output) / ('final.pt' if final else 'latest.pt'))
        except Exception as exc:
            error[0] = repr(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'pilot'), required=True)
    opt = parser.parse_args()
    reference = opt.reference_root.resolve()
    if not torch.cuda.is_available() or 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('Use one Slurm allocation and torchrun with 4 GPUs')
    rank_local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank_local)
    device = torch.device('cuda', rank_local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = old.module('train_pilot', p2.LEGACY)
    baseline = old.module('train', old.SCRATCH)
    obj = old.module('objective', old.SCRATCH)
    bw = p2.weights()
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 4:
            raise ValueError('Exactly four ranks required')
        cache, control, other, references = p2.load_references(reference)
        del other
        out = reference / ('P2_smoke' if opt.mode == 'smoke' else 'P2')
        args = p2.make_args(control, out, opt.mode)
        if opt.mode == 'pilot':
            smoke, _ = p2.load_saved(reference / 'P2_smoke/final.pt')
            p2.assert_p2(smoke, control, cache.identity, references, require_pilot=False)
            if smoke['mode'] != 'smoke':
                raise ValueError('P2 smoke gate is not a smoke artifact')
            del smoke
        baseline.synchronized_check(not out.exists(), 'Output exists; no resume or overwrite', device)
        if rank == 0:
            out.mkdir()
        dist.barrier()
        baseline.seed_everything(args.seed)
        artifact_root = Path(args.artifact_root)
        model, parent = old.load_v1(artifact_root / old.V1_REL, device, trainable=True)
        if not all(p.requires_grad for p in model.parameters()) or sum(p.numel() for p in model.parameters()) != 1133705:
            raise ValueError('Architecture or trainable parameter count changed')
        if old.sha256(old.URDF) != cache.manifest['urdf_sha256'] or old.sha256(old.URDF) != parent['metadata']['urdf_sha256']:
            raise ValueError('URDF changed')
        api = baseline.load_repo_api(old.REPO)
        dataset = api['VisibilityQ0Dataset'](str(artifact_root / old.DATA_REL), val_count=1000, seed=0)
        cache.verify_dataset(dataset, artifact_root / old.DATA_REL)
        oracle = api['PinocchioFOVOracle'](urdf_path=str(old.URDF), joint_names=api['DEFAULT_JOINT_NAMES'],
            sensor_frames=api['DEFAULT_SENSOR_FRAMES'], horizontal_fov_deg=50., vertical_fov_deg=66., z_min=.2, z_max=.7, delta=.01)
        ddp = DDP(model, device_ids=[rank_local], broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp == 'fp16', init_scale=1024.)
        streams = (hashlib.sha256(), hashlib.sha256())
        bcounts = torch.full((16,), args.boundary_count//16, device=device, dtype=torch.float32)
        if rank == 0:
            old.write_json(out / 'run.json', dict(status='RUNNING', arm='P2', mode=opt.mode, args=vars(args),
                boundary_weights=p2.WEIGHTS, parent_sha256=old.V1_SHA, reference_sha256=references,
                cache_manifest_sha256=cache.identity, source_sha256=old.source_fingerprints(),
                p2_source_sha256=p2.fingerprints(), optimizer='fresh Adam, constant lr, no scheduler/clipping'))
        baseline.log(f'[pilot] P2 parent=V1@50000 extra_updates={args.steps} mode={opt.mode} '
                     f'uniform_pairs={args.global_batch_x*args.batch_q} boundary_queries={args.boundary_count} lr={args.lr}')
        baseline.log('[boundary_weights]', p2.WEIGHTS, 'global objective unchanged')
        val = engine.validate(ddp, cache, api, dataset, oracle, args, device, baseline, obj, bw)
        if rank == 0:
            old.write_json(out / 'initial_validation.json', val)
        for step in range(1, args.steps+1):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            baseline.seed_everything(old.seed_for_update(args.seed, step))
            uniform, counts = baseline.prepare_batch(api, dataset, oracle, args, device, 'train')
            for batch in uniform:
                streams[0].update(batch[0].detach().cpu().numpy().tobytes())
            ids = cache.draw_indices(args.seed, step, args.boundary_count, rank, world)
            streams[1].update(ids.tobytes())
            boundary = cache.tensors('train', ids, device)
            result = engine.perform_update(ddp, optimizer, scaler, uniform, counts, boundary, bcounts,
                obj.LossWeights(), bw, p2.alpha(step, args.boundary_warmup), args, baseline, obj)
            del uniform, boundary
            torch.cuda.synchronize(device)
            result.update(update=step, total_updates=50000+step, arm='P2', seconds=time.perf_counter()-started,
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                h = result['boundary']['heads']
                baseline.log(f"[train] P2 update={step}/{args.steps} global_loss={result['global']['loss']:.5f} "
                    f"alpha={result['boundary_alpha']:.4f} S6_absf={h['s6']['abs_value']:.4f} "
                    f"S6_norm={h['s6']['gradient_norm']:.3f} S7_absf={h['s7']['abs_value']:.4f} "
                    f"S7_norm={h['s7']['gradient_norm']:.3f} seconds={result['seconds']:.2f}")
                if rank == 0:
                    with (out / 'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(result, allow_nan=False)+'\n')
            if step == 1 or step % args.val_every == 0 or step == args.steps:
                val = engine.validate(ddp, cache, api, dataset, oracle, args, device, baseline, obj, bw)
                h = val['boundary_val']['heads']
                baseline.log('[val] P2 update=' + str(step) + ' ' + ' '.join(
                    f"{s}:absf={h[s]['abs_value']:.4f},norm={h[s]['gradient_norm']:.3f},cos={h[s]['normal_cosine']:.4f}"
                    for s in ('s6','s7')))
                if rank == 0:
                    with (out / 'validation.jsonl').open('a') as f:
                        f.write(json.dumps({'update':step, **val}, allow_nan=False)+'\n')
                save(ddp, optimizer, scaler, args, cache, step, val, streams, parent, control,
                     references, opt.mode, baseline)
        save(ddp, optimizer, scaler, args, cache, args.steps, val, streams, parent, control,
             references, opt.mode, baseline, final=True)
        if rank == 0:
            run = json.loads((out/'run.json').read_text())
            run.update(status='COMPLETE', final_sha256=old.sha256(out/'final.pt'),
                       successful_updates=args.steps, pair_sample_streams='MATCH' if opt.mode == 'pilot' else 'NOT_COMPARABLE_2_VS_2000')
            old.write_json(out/'run.json', run)
        baseline.log(f'[done] p2_training_complete mode={opt.mode} successful_updates={args.steps} output={out}')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
