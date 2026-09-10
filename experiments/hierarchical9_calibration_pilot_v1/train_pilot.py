#!/usr/bin/env python3
"""Matched 4-GPU P0/P1 pilot. Same V1 weights; fresh Adam; all weights trainable.

Both arms see the SAME 400k uniform pairs per update. Both evaluate/backpropagate
boundary graphs; P0 multiplies their loss by exactly zero. P1 adds ramped boundary
supervision. This is an added-supervision ablation, NOT an equal-label-budget claim.
"""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import (REPO, SCRATCH, SCRIPTS, URDF, V1_REL, V1_SHA, DATA_REL, PILOT_FORMAT,
                    BoundaryCache, load_v1, module, setup_paths, source_fingerprints,
                    seed_for_update, write_json, sha256)
from calibration import BoundaryWeights, boundary_loss, summary, ramp_weight


def perform_update(model, optimizer, scaler, uniform, counts, boundary, boundary_counts,
                   base_weights, correction_weights, alpha, args, baseline, obj):
    """Single optimizer step after all global+boundary micros; synchronize exactly once."""
    device = counts.device
    world = dist.get_world_size()
    for attempt in range(args.max_amp_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        global_stats = torch.zeros((9, len(obj.STAT_NAMES)), device=device, dtype=torch.float64)
        bstats = torch.zeros((16, 6), device=device, dtype=torch.float64)
        model.train()
        for batch in uniform:
            # Final synchronization is deliberately at the last boundary microbatch.
            with model.no_sync():
                with baseline.amp_context(args.amp):
                    loss, st = obj.loss_for_microbatch(model, *batch, counts, base_weights,
                                                       world_size=world, training=True)
                baseline.synchronized_check(torch.isfinite(loss) & torch.isfinite(st).all(),
                                            'Nonfinite uniform loss', device)
                scaler.scale(loss).backward()
            global_stats += st
        n = len(boundary[0])
        for begin in range(0, n, args.boundary_microbatch):
            end = min(begin + args.boundary_microbatch, n)
            context = model.no_sync() if end < n else contextlib.nullcontext()
            with context:
                loss, st = boundary_loss(model, *(v[begin:end] for v in boundary), boundary_counts,
                                         correction_weights, world_size=world, training=True)
                baseline.synchronized_check(torch.isfinite(loss) & torch.isfinite(st).all(),
                                            'Nonfinite FP32 boundary loss', device)
                scaler.scale(alpha * loss).backward()
            bstats += st
        scaler.unscale_(optimizer)
        good = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().int()
        dist.all_reduce(good, op=dist.ReduceOp.MIN)
        if good.item():
            parameter_grad_norm = math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None))
            scaler.step(optimizer)
            scaler.update()
            break
        if args.amp != 'fp16' or attempt == args.max_amp_retries:
            raise RuntimeError('Nonfinite parameter gradients; no successful update saved')
        scaler.update(new_scale=scaler.get_scale()/2)
        baseline.log(f'[amp-retry] exact same uniform+boundary samples; retry={attempt+1}')
    dist.all_reduce(global_stats)
    dist.all_reduce(bstats)
    if not torch.equal(global_stats[:, 0].float(), counts) or not torch.equal(bstats[:, 0].float(), boundary_counts):
        raise RuntimeError('Global normalization/count mismatch')
    return {'global': obj.summarize(global_stats, base_weights),
            'boundary': summary(bstats, correction_weights), 'boundary_alpha': alpha,
            'amp_retries': attempt, 'grad_scale': scaler.get_scale(), 'parameter_grad_norm': parameter_grad_norm}


def boundary_eval(model, cache, split, device, weights, rank, world):
    model.eval()
    n = len(cache.arrays[split]['s'])
    ids = np.arange(rank, n, world)
    all_counts = np.bincount(2*cache.arrays[split]['s'] + cache.arrays[split]['kind'], minlength=16)
    denom = torch.tensor(all_counts, device=device, dtype=torch.float32)
    stats = torch.zeros((16, 6), device=device, dtype=torch.float64)
    for start in range(0, len(ids), 256):
        batch = cache.tensors(split, ids[start:start+256], device)
        _, st = boundary_loss(model, *batch, denom, weights, training=False)
        stats += st
    dist.all_reduce(stats)
    return summary(stats, weights)


def validate(model, cache, api, dataset, oracle, args, device, baseline, obj, bw):
    valargs = argparse.Namespace(**vars(args)); valargs.amp = 'off'
    batches, counts = baseline.prepare_batch(api, dataset, oracle, valargs, device, 'val')
    result = {'global': baseline.run_prepared_batch(model, batches, counts, obj.LossWeights(), valargs),
              'boundary_val': boundary_eval(model.module, cache, 'val', device, bw, dist.get_rank(), dist.get_world_size()),
              'boundary_train': boundary_eval(model.module, cache, 'train', device, bw, dist.get_rank(), dist.get_world_size())}
    return result


def save(model, optimizer, scaler, args, cache, step, validation, streams, parent, baseline, final=False):
    record = {'global': streams[0].hexdigest(), 'boundary': streams[1].hexdigest()}
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    error = [None]
    if dist.get_rank() == 0:
        try:
            state = {'format': PILOT_FORMAT, 'arm': args.arm, 'pilot_updates': step, 'step': step,
                'parent_updates': 50000, 'total_updates': 50000+step, 'parent_sha256': V1_SHA,
                'initialization': 'V1_weights_only_fresh_Adam', 'optimizer_policy': 'fresh_Adam_constant_lr_no_scheduler_no_clipping',
                'args': vars(args), 'cache_manifest_sha256': cache.identity, 'out_dim': 9,
                'architecture': parent['architecture'], 'output_layout': parent['output_layout'],
                'model_state': model.module.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'scaler_state': scaler.state_dict(), 'sample_stream_sha256_by_rank': records,
                'validation': validation, 'source_sha256': source_fingerprints(),
                'boundary_weights': asdict(BoundaryWeights()),
                'completed': final, 'frozen_parameters': 0}
            baseline.atomic_save(state, Path(args.output) / ('final.pt' if final else 'latest.pt'))
        except Exception as exc:
            error[0] = repr(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', required=True, choices=('P0', 'P1'))
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--global-batch-x', type=int, default=4000)
    p.add_argument('--batch-q', type=int, default=100)
    p.add_argument('--microbatch-x', type=int, default=250)
    p.add_argument('--decode-x-chunk', type=int, default=64)
    p.add_argument('--boundary-count', type=int, default=4096)
    p.add_argument('--boundary-microbatch', type=int, default=256)
    p.add_argument('--boundary-warmup', type=int, default=500)
    p.add_argument('--val-global-batch-x', type=int, default=512)
    p.add_argument('--val-batch-q', type=int, default=100)
    p.add_argument('--val-microbatch-x', type=int, default=128)
    p.add_argument('--val-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--amp', choices=('fp16', 'bf16', 'off'), default='fp16')
    p.add_argument('--max-amp-retries', type=int, default=16)
    args = p.parse_args()
    for name in ('steps','global_batch_x','batch_q','microbatch_x','decode_x_chunk','boundary_count',
                 'boundary_microbatch','boundary_warmup','val_global_batch_x','val_batch_q','val_microbatch_x','val_every','log_every'):
        if getattr(args, name) <= 0: p.error(name+' must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0 or args.seed < 0 or args.max_amp_retries < 0:
        p.error('Invalid lr/seed/retries')
    if not torch.cuda.is_available() or 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('Use torch.distributed.run with 4 allocated GPUs')
    rank_local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank_local)
    device = torch.device('cuda', rank_local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    setup_paths()
    baseline = module('train', SCRATCH)
    obj = module('objective', SCRATCH)
    bw = BoundaryWeights()
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 4 or args.global_batch_x % world or args.val_global_batch_x % world or args.boundary_count % (16*world):
            raise ValueError('Expected 4 ranks and divisible global/stratum batches')
        out = Path(args.output).resolve(); args.output = str(out)
        baseline.synchronized_check(not out.exists() or not any(out.iterdir()), 'Refusing nonempty pilot output; no implicit resume', device)
        if rank == 0: out.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        baseline.seed_everything(args.seed)
        root = Path(args.artifact_root).resolve()
        model, parent = load_v1(root / V1_REL, device, trainable=True)
        cache = BoundaryCache(Path(args.cache))
        if cache.manifest['urdf_sha256'] != sha256(URDF) or parent['metadata']['urdf_sha256'] != sha256(URDF):
            raise ValueError('URDF mismatch')
        api = baseline.load_repo_api(REPO)
        dataset = api['VisibilityQ0Dataset'](str(root / DATA_REL), val_count=1000, seed=0)
        cache.verify_dataset(dataset, root / DATA_REL)
        oracle = api['PinocchioFOVOracle'](urdf_path=str(URDF), joint_names=api['DEFAULT_JOINT_NAMES'],
            sensor_frames=api['DEFAULT_SENSOR_FRAMES'], horizontal_fov_deg=50., vertical_fov_deg=66., z_min=.2, z_max=.7, delta=.01)
        ddp = DDP(model, device_ids=[rank_local], broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=args.lr)
        if args.amp == 'bf16' and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 not supported')
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp == 'fp16', init_scale=1024.)
        streams = (hashlib.sha256(), hashlib.sha256())
        bcounts = torch.full((16,), args.boundary_count//16, device=device, dtype=torch.float32)
        if rank == 0:
            write_json(out / 'run.json', {'status': 'RUNNING', 'args': vars(args), 'parent_sha256': V1_SHA,
                'cache_manifest_sha256': cache.identity, 'source_sha256': source_fingerprints(),
                'boundary_weights': asdict(bw), 'optimizer': 'fresh Adam, constant lr; NOT optimizer resume'})
        baseline.log(f'[pilot] {args.arm} parent=V1@50000 extra_updates={args.steps} all_trainable=1133705 '
                     f'uniform_pairs={args.global_batch_x*args.batch_q} boundary_queries={args.boundary_count} lr={args.lr}')
        val = validate(ddp, cache, api, dataset, oracle, args, device, baseline, obj, bw)
        if rank == 0:
            write_json(out / 'initial_validation.json', val)
        for step in range(1, args.steps+1):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            baseline.seed_everything(seed_for_update(args.seed, step))
            uniform, counts = baseline.prepare_batch(api, dataset, oracle, args, device, 'train')
            for batch in uniform:
                streams[0].update(batch[0].detach().cpu().numpy().tobytes())
            ids = cache.draw_indices(args.seed, step, args.boundary_count, rank, world)
            streams[1].update(ids.tobytes())
            boundary = cache.tensors('train', ids, device)
            result = perform_update(ddp, optimizer, scaler, uniform, counts, boundary, bcounts,
                obj.LossWeights(), bw, ramp_weight(args.arm, step, args.boundary_warmup), args, baseline, obj)
            del uniform, boundary
            torch.cuda.synchronize(device)
            result.update(update=step, total_updates=50000+step, arm=args.arm,
                          seconds=time.perf_counter()-started,
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                h = result['boundary']['heads']
                baseline.log(f"[train] {args.arm} update={step}/{args.steps} global_loss={result['global']['loss']:.5f} "
                    f"alpha={result['boundary_alpha']:.4f} S6_absf={h['s6']['abs_value']:.4f} "
                    f"S6_norm={h['s6']['gradient_norm']:.3f} S7_absf={h['s7']['abs_value']:.4f} "
                    f"S7_norm={h['s7']['gradient_norm']:.3f} seconds={result['seconds']:.2f}")
                if rank == 0:
                    with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(result, allow_nan=False)+'\n')
            if step == 1 or step % args.val_every == 0 or step == args.steps:
                val = validate(ddp, cache, api, dataset, oracle, args, device, baseline, obj, bw)
                h = val['boundary_val']['heads']
                baseline.log(f"[val] {args.arm} update={step} S6_absf={h['s6']['abs_value']:.4f} "
                             f"S6_norm={h['s6']['gradient_norm']:.3f} S6_cos={h['s6']['normal_cosine']:.4f} "
                             f"S7_absf={h['s7']['abs_value']:.4f} S7_norm={h['s7']['gradient_norm']:.3f}")
                if rank == 0:
                    with (out / 'validation.jsonl').open('a') as f: f.write(json.dumps({'update':step, **val}, allow_nan=False)+'\n')
                save(ddp, optimizer, scaler, args, cache, step, val, streams, parent, baseline)
        save(ddp, optimizer, scaler, args, cache, args.steps, val, streams, parent, baseline, final=True)
        if rank == 0:
            run = json.loads((out/'run.json').read_text()); run.update(status='COMPLETE', final_sha256=sha256(out/'final.pt'))
            write_json(out/'run.json', run)
        baseline.log(f'[done] calibration_pilot_complete arm={args.arm} successful_updates={args.steps} final={out}/final.pt')
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
