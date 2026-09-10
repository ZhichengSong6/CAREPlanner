"""Boundary-only correction. No legacy q0-bank distance targets on these rows."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BoundaryWeights:
    zero: float = 5.0
    normal: float = .1
    eikonal: float = .1

    def validate(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Boundary weights must be finite and nonnegative')


# 16 rows: (S0 bank, S0 offbank, ..., S7 offbank).
# columns: count, f^2, 1-cos, |norm-1|, |f|, norm

def boundary_loss(model, inputs, sensors, normals, groups, global_counts, weights,
                  *, world_size=1, training=True):
    weights.validate()
    n = len(inputs)
    if inputs.shape != (n, 10) or normals.shape != (n, 7) or sensors.shape != (n,) or groups.shape != (n,):
        raise ValueError('Invalid boundary tensor shapes')
    if global_counts.shape != (16,) or torch.any(global_counts <= 0):
        raise ValueError('All 16 sensor/origin strata need global supervision')
    if torch.any((sensors < 0) | (sensors > 7) | (groups // 2 != sensors)):
        raise ValueError('Sensor/group mismatch')
    with torch.enable_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
        q = inputs[:, 3:].detach().float().clone().requires_grad_(True)
        pred = model(torch.cat((inputs[:, :3].detach().float(), q), 1)).float()
        if pred.shape != (n, 9):
            raise ValueError('Expected [union,S0,...,S7]')
        value = pred.gather(1, (sensors + 1)[:, None])[:, 0]
        grad = torch.autograd.grad(value.sum(), q, create_graph=training, retain_graph=True)[0]
        norm = grad.norm(dim=-1)
        # Raw full-q gradient: inactive components are penalized, not hidden by a mask.
        direction = 1 - F.cosine_similarity(grad, normals.float(), dim=-1, eps=1e-6)
        zero, eik = value.square(), (norm - 1).abs()
        per_row = weights.zero * zero + weights.normal * direction + weights.eikonal * eik
        sums = torch.zeros(16, device=inputs.device).index_add(0, groups, per_row)
        # Equal sensor weight, then equal bank/offbank weight. No mean-of-microbatch-means.
        loss = (sums / global_counts.float()).sum() / 16
        loss = loss + (pred * 0).sum()  # Keep all DDP parameters graph-connected, including union.
        with torch.no_grad():
            rows = torch.stack((torch.ones_like(value), zero, direction, eik, value.abs(), norm), 1)
            stats = torch.zeros((16, 6), device=inputs.device, dtype=torch.float64)
            stats.index_add_(0, groups, rows.double())
    return loss * world_size, stats


def summary(stats, weights):
    means = stats.detach().cpu().double() / stats[:, 0:1].detach().cpu().double().clamp_min(1)
    heads = {}
    for s in range(8):
        rows = stats[2*s:2*s+2].detach().cpu().double()
        count = int(rows[:, 0].sum())
        m = means[2*s:2*s+2].mean(0)
        heads[f's{s}'] = {'count': count, 'abs_value': float(m[4]), 'normal_cosine': float(1-m[2]),
                          'gradient_norm': float(m[5]), 'norm_error': float(m[3])}
    components = means[:, 1:4].mean(0)
    return {'loss': float(components @ torch.tensor([weights.zero, weights.normal, weights.eikonal], dtype=torch.float64)),
            'zero_mse': float(components[0]), 'normal_loss': float(components[1]),
            'eikonal_l1': float(components[2]), 'heads': heads}


def ramp_weight(arm, update, warmup):
    if arm not in ('P0', 'P1') or update < 0 or warmup < 1:
        raise ValueError('Invalid arm/update/warmup')
    return 0.0 if arm == 'P0' else min(1.0, update / warmup)
