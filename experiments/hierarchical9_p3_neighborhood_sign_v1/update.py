"""One Adam update for uniform + unchanged P2 anchors + neighborhood inequalities.

Reuse the old global and boundary objectives. Only add a third accumulation stream;
there is no second optimizer step. DDP synchronizes at the last neighborhood micro.
"""
from __future__ import annotations
import contextlib
import math

import torch
import torch.distributed as dist

from p3_protocol import p2, NeighborhoodConfig
import neighborhood as side


def perform_update(model, optimizer, scaler, uniform, counts, boundary, boundary_counts,
                   neighbors, neighbor_counts, base_weights, correction_weights,
                   alpha, args, baseline, obj, config=NeighborhoodConfig()):
    device, world = counts.device, dist.get_world_size()
    if len(neighbors['inputs']) == 0:
        raise ValueError('All ranks must have attempted neighborhood rows for DDP synchronization')
    for attempt in range(args.max_amp_retries+1):
        optimizer.zero_grad(set_to_none=True)
        gs = torch.zeros((9, len(obj.STAT_NAMES)), device=device, dtype=torch.float64)
        bs = torch.zeros((16, 6), device=device, dtype=torch.float64)
        ns = torch.zeros((32, len(side.COLUMNS)), device=device, dtype=torch.float64)
        model.train()
        for batch in uniform:
            with model.no_sync(), torch.enable_grad():
                with baseline.amp_context(args.amp):
                    loss, stats = obj.loss_for_microbatch(model, *batch, counts, base_weights,
                                                         world_size=world, training=True)
                baseline.synchronized_check(torch.isfinite(loss) & torch.isfinite(stats).all(),
                                             'Nonfinite uniform loss', device)
                scaler.scale(loss).backward()
            gs += stats
        for start in range(0, len(boundary[0]), args.boundary_microbatch):
            end = min(start+args.boundary_microbatch, len(boundary[0]))
            with model.no_sync(), torch.enable_grad():
                loss, stats = p2.calibration.boundary_loss(model, *(v[start:end] for v in boundary),
                    boundary_counts, correction_weights, world_size=world, training=True)
                baseline.synchronized_check(torch.isfinite(loss) & torch.isfinite(stats).all(),
                                             'Nonfinite unchanged P2 anchor loss', device)
                scaler.scale(alpha*loss).backward()
            bs += stats
        n = len(neighbors['inputs'])
        for start in range(0, n, config.microbatch):
            end = min(start+config.microbatch, n)
            context = model.no_sync() if end < n else contextlib.nullcontext()
            with context, torch.enable_grad():
                loss, stats = side.sign_loss(model, side.slice_batch(neighbors, start, end),
                                            neighbor_counts, world_size=world, config=config)
                baseline.synchronized_check(torch.isfinite(loss) & torch.isfinite(stats).all(),
                                             'Nonfinite FP32 neighborhood loss', device)
                scaler.scale(alpha*loss).backward()
            ns += stats
        scaler.unscale_(optimizer)
        good = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().int()
        dist.all_reduce(good, op=dist.ReduceOp.MIN)
        if good.item():
            gn = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                               for p in model.parameters() if p.grad is not None))
            scaler.step(optimizer)
            scaler.update()
            break
        if args.amp != 'fp16' or attempt == args.max_amp_retries:
            raise RuntimeError('Nonfinite parameter gradients; no successful update')
        scaler.update(new_scale=scaler.get_scale()/2)
        baseline.log(f'[amp-retry] SAME uniform/anchor/neighborhood rows; retry={attempt+1}')
    for stats in (gs, bs, ns):
        dist.all_reduce(stats)
    if (not torch.equal(gs[:, 0].float(), counts)
            or not torch.equal(bs[:, 0].float(), boundary_counts)
            or not torch.equal(ns[:, 2].float(), neighbor_counts)):
        raise RuntimeError('Global normalization/count mismatch')
    result = dict(global_=obj.summarize(gs, base_weights), boundary=p2.calibration.summary(bs, correction_weights),
                  neighborhood=side.summary(ns, config), boundary_alpha=alpha, amp_retries=attempt,
                  grad_scale=scaler.get_scale(), parameter_grad_norm=gn)
    result['global'] = result.pop('global_')
    result['combined_loss'] = result['global']['loss'] + alpha*(result['boundary']['loss']+result['neighborhood']['loss'])
    return result


def evaluate_neighbors(model, cache, split, oracle, device, seed, config=NeighborhoodConfig()):
    """Fixed radial monitoring on train/val independently; never used as training rows."""
    import numpy as np
    rank, world = dist.get_rank(), dist.get_world_size()
    n = len(cache.arrays[split]['s'])
    indices = np.arange(rank, n, world)
    radii = side.radii_for_update(seed, 0, n, config=config)
    lo, hi = (torch.tensor(a, device=device) for a in (cache.lo, cache.hi))
    stats = torch.zeros((32, len(side.COLUMNS)), device=device, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            ids = indices[start:start+128]
            b = cache.tensors(split, ids, device)
            nb = side.build_queries(b, radii[ids], oracle, lo, hi, config)
            _, st = side.sign_loss(model, nb, torch.ones(32, device=device), config=config)
            if not torch.isfinite(st).all():
                raise RuntimeError('Nonfinite neighborhood validation')
            stats += st
    dist.all_reduce(stats)
    result = side.summary(stats, config)
    result['sampling'] = 'fixed radius per cache anchor; seed/step=0 stream; this split only'
    return result
