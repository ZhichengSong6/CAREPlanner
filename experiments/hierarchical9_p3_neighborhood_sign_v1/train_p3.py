#!/usr/bin/env python3
"""P3 from V1: unchanged uniform/anchor streams plus online actual-sign neighborhoods."""
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

import p3_protocol as p3
import neighborhood as side
import update
old, p2 = p3.old, p3.p2


def save(model, optimizer, scaler, args, cache, step, validation, streams, extra,
         totals, parent, controls, refs, mode, checks, baseline, final=False):
    records = [None]*dist.get_world_size()
    neighborhood_records = [None]*dist.get_world_size()
    dist.all_gather_object(records, {'global': streams[0].hexdigest(), 'boundary': streams[1].hexdigest()})
    dist.all_gather_object(neighborhood_records, {'proposal': extra[0].hexdigest(), 'labeled': extra[1].hexdigest()})
    error = [None]
    if dist.get_rank() == 0:
        try:
            matched = records == controls['P0']['sample_stream_sha256_by_rank']
            if final and mode == 'pilot' and not matched:
                raise ValueError('P3 original query/anchor streams differ; no successful final saved')
            state = dict(format=p3.FORMAT, arm='P3', mode=mode, pilot_updates=step, step=step,
                parent_updates=50000, total_updates=50000+step, parent_sha256=old.V1_SHA,
                initialization='V1_weights_only_fresh_Adam',
                optimizer_policy='fresh_Adam_constant_lr_no_scheduler_no_clipping', args=vars(args),
                cache_manifest_sha256=cache.identity, out_dim=9, frozen_parameters=0,
                architecture=parent['architecture'], output_layout=parent['output_layout'],
                model_state=model.module.state_dict(), optimizer_state=optimizer.state_dict(),
                scaler_state=scaler.state_dict(), sample_stream_sha256_by_rank=records,
                neighborhood_stream_sha256_by_rank=neighborhood_records,
                neighborhood_totals=totals, neighborhood_config=p3.NEIGHBOR,
                validation=validation, source_sha256=old.source_fingerprints(),
                p2_source_sha256=p2.fingerprints(), p3_source_sha256=p3.fingerprints(),
                boundary_weights=p2.WEIGHTS, reference_sha256=refs, completed=final,
                preflight=checks, pair_sample_streams=('MATCH' if matched else 'MISMATCH')
                    if mode == 'pilot' and final else
                    'NOT_COMPARABLE_2_VS_2000' if mode == 'smoke' else 'PREFIX_ONLY',
                added_supervision_note='Neighbors are additional labeled queries, not matched supervision/compute budget')
            if final:
                p3.assert_p3(state, controls['P0'], cache.identity, refs, require_pilot=mode == 'pilot')
            baseline.atomic_save(state, Path(args.output)/('final.pt' if final else 'latest.pt'))
        except Exception as exc:
            error[0] = repr(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def validate(ddp, cache, api, dataset, oracle, pair, args, device, baseline, obj, engine):
    result = engine.validate(ddp, cache, api, dataset, oracle, args, device, baseline, obj, p2.weights())
    for split in ('train', 'val'):
        result['neighborhood_'+split] = update.evaluate_neighbors(ddp.module, cache, split, pair, device, args.seed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke', 'pilot'), required=True)
    opt = parser.parse_args()
    root = opt.reference_root.resolve()
    if not torch.cuda.is_available() or 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('Use Slurm and torchrun with four allocated GPUs')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = old.module('train_pilot', p2.LEGACY)
    baseline = old.module('train', old.SCRATCH)
    obj = old.module('objective', old.SCRATCH)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 4:
            raise ValueError('Exactly four ranks required')
        cache, controls, refs = p3.load_references(root)
        out = root/('P3_smoke' if opt.mode == 'smoke' else 'P3')
        args = p3.make_args(controls['P0'], out, opt.mode)
        if opt.mode == 'pilot':
            p3.check_smoke(root, controls['P0'], cache.identity, refs)
        baseline.synchronized_check(not out.exists(), 'No overwrite/resume of P3 output', device)
        if rank == 0:
            out.mkdir()
        dist.barrier()
        baseline.seed_everything(args.seed)
        artifact = Path(args.artifact_root)
        model, parent = old.load_v1(artifact/old.V1_REL, device, trainable=True)
        if not all(p.requires_grad for p in model.parameters()) or sum(p.numel() for p in model.parameters()) != 1133705:
            raise ValueError('Unexpected architecture or frozen parameters')
        if old.sha256(old.URDF) != cache.manifest['urdf_sha256'] or old.sha256(old.URDF) != parent['metadata']['urdf_sha256']:
            raise ValueError('URDF changed')
        api = baseline.load_repo_api(old.REPO)
        dataset = api['VisibilityQ0Dataset'](str(artifact/old.DATA_REL), val_count=1000, seed=0)
        cache.verify_dataset(dataset, artifact/old.DATA_REL)
        oracle = api['PinocchioFOVOracle'](str(old.URDF), api['DEFAULT_JOINT_NAMES'],
                    api['DEFAULT_SENSOR_FRAMES'], 50., 66., .2, .7, .01)
        geometric = old.module('oracle', old.AUDIT).SensorOracle(old.URDF, device,
                    api['DEFAULT_JOINT_NAMES'], api['DEFAULT_SENSOR_FRAMES'])
        pair = side.PairwiseFOV(geometric)
        checks = old.module('audit', old.AUDIT).preflight(dataset, geometric, {}, device)
        checks['pairwise_neighbor_FOV'] = pair.verify(cache, device)
        baseline.log('[preflight]', json.dumps(checks))
        lo, hi = dataset.q_limits(device)
        ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=args.lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp == 'fp16', init_scale=1024.)
        streams = (hashlib.sha256(), hashlib.sha256())
        extra = (hashlib.sha256(), hashlib.sha256())
        bcounts = torch.full((16,), args.boundary_count//16, device=device, dtype=torch.float32)
        totals = {k: 0 for k in ('attempted', 'valid', 'out_of_limits', 'ambiguous', 'offset_sign_disagrees')}
        if rank == 0:
            old.write_json(out/'run.json', dict(status='RUNNING', arm='P3', mode=opt.mode, args=vars(args),
                boundary_weights=p2.WEIGHTS, neighborhood_config=p3.NEIGHBOR, parent_sha256=old.V1_SHA,
                reference_sha256=refs, cache_manifest_sha256=cache.identity,
                source_sha256=old.source_fingerprints(), p2_source_sha256=p2.fingerprints(),
                p3_source_sha256=p3.fingerprints(), preflight=checks))
        baseline.log(f'[pilot] P3 parent=V1@50000 extra_updates={args.steps} mode={opt.mode} '
                     f'uniform_pairs=400000 boundary_queries=4096 neighbor_attempts=8192 lr={args.lr}')
        baseline.log('[objective] unchanged P2 anchors', p2.WEIGHTS, 'added neighborhood', p3.NEIGHBOR)
        val = validate(ddp, cache, api, dataset, oracle, pair, args, device, baseline, obj, engine)
        if rank == 0:
            old.write_json(out/'initial_validation.json', val)
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
            radii = side.radii_for_update(args.seed, step, args.boundary_count, rank, world)
            neighbors = side.build_queries(boundary, radii, pair, lo, hi)
            side.hash_queries(*extra, ids, neighbors)
            ncounts = side.valid_counts(neighbors)
            dist.all_reduce(ncounts)
            if not ncounts.sum().item():
                raise RuntimeError('No unambiguous in-limit neighborhood labels in global batch')
            result = update.perform_update(ddp, optimizer, scaler, uniform, counts, boundary, bcounts,
                neighbors, ncounts, obj.LossWeights(), p2.weights(), p2.alpha(step, args.boundary_warmup),
                args, baseline, obj)
            del uniform, boundary, neighbors
            for key in totals:
                totals[key] += int(result['neighborhood'][key])
            torch.cuda.synchronize(device)
            result.update(update=step, total_updates=50000+step, arm='P3',
                seconds=time.perf_counter()-started,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                ns = result['neighborhood']
                baseline.log(f"[train] P3 update={step}/{args.steps} global_loss={result['global']['loss']:.5f} "
                    f"alpha={result['boundary_alpha']:.4f} side_loss={ns['loss']:.5f} "
                    f"side_valid={int(ns['valid'])}/{int(ns['attempted'])} seconds={result['seconds']:.2f}")
                if rank == 0:
                    with (out/'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(result, allow_nan=False)+'\n')
            if step == 1 or step % args.val_every == 0 or step == args.steps:
                val = validate(ddp, cache, api, dataset, oracle, pair, args, device, baseline, obj, engine)
                h, n = val['boundary_val']['heads'], val['neighborhood_val']['heads']
                baseline.log('[val] P3 update='+str(step)+' '+' '.join(
                    f"{s}:absf={h[s]['abs_value']:.4f},cos={h[s]['normal_cosine']:.4f},"
                    f"pos_recall={n[s]['positive_recall']},neg_recall={n[s]['negative_recall']}"
                    for s in ('s0', 's1', 's6', 's7')))
                if rank == 0:
                    with (out/'validation.jsonl').open('a') as f:
                        f.write(json.dumps({'update': step, **val}, allow_nan=False)+'\n')
                save(ddp, optimizer, scaler, args, cache, step, val, streams, extra, totals,
                     parent, controls, refs, opt.mode, checks, baseline)
        save(ddp, optimizer, scaler, args, cache, args.steps, val, streams, extra, totals,
             parent, controls, refs, opt.mode, checks, baseline, final=True)
        if rank == 0:
            run = json.loads((out/'run.json').read_text())
            run.update(status='COMPLETE', final_sha256=old.sha256(out/'final.pt'), successful_updates=args.steps,
                       neighborhood_totals=totals,
                       pair_sample_streams='MATCH' if opt.mode == 'pilot' else 'NOT_COMPARABLE_2_VS_2000')
            old.write_json(out/'run.json', run)
        baseline.log(f'[done] p3_training_complete mode={opt.mode} successful_updates={args.steps} output={out}')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
