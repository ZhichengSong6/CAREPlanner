"""Refreshed normal-offset queries; geometry labels are NOT inferred from the offset.

Only an actual-FOV sign margin is added. No distance regression, unit-norm loss,
union labels, oracle gradient supervision, model-dependent mining or clipping.
"""
from __future__ import annotations
import math

import numpy as np
import torch

from p3_protocol import NeighborhoodConfig

COLUMNS = ('attempted', 'in_limits', 'valid', 'ambiguous', 'tp', 'tn', 'fp', 'fn',
           'margin_violations', 'hinge_squared', 'offset_sign_disagrees', 'abs_value')


class PairwiseFOV:
    """Same upstream FK and six conservative planes; x differs for EACH query."""
    def __init__(self, base):
        self.base = base

    @torch.no_grad()
    def margins(self, inputs, sensors):
        if inputs.ndim != 2 or inputs.shape[1] != 10 or sensors.shape != (len(inputs),):
            raise ValueError('Expected matched [x,q] and sensor arrays')
        if torch.any((sensors < 0) | (sensors > 7)) or not torch.isfinite(inputs).all():
            raise ValueError('Invalid geometry query')
        b = self.base
        with torch.autocast(inputs.device.type, enabled=False):
            inputs = inputs.float()
            result = torch.empty(len(inputs), device=inputs.device, dtype=torch.float32)
            plane = torch.empty(len(inputs), device=inputs.device, dtype=torch.long)
            ax, ay = math.tan(math.radians(b.hfov)/2), math.tan(math.radians(b.vfov)/2)
            nx, ny = math.sqrt(1+ax*ax), math.sqrt(1+ay*ay)
            for s in range(8):
                ids = torch.where(sensors == s)[0]
                if not len(ids):
                    continue
                inp = inputs[ids]
                T = b.fk(b.specs[s], inp[:, 3:])
                diff = inp[:, :3] - T[:, :3, 3]
                local = torch.einsum('nji,nj->ni', T[:, :3, :3], diff)
                px, py, pz = local.unbind(-1)
                planes = torch.stack(((px+pz*ax)/nx, (-px+pz*ax)/nx,
                                      (py+pz*ay)/ny, (-py+pz*ay)/ny,
                                      pz-b.zmin, b.zmax-pz), -1) - b.delta
                g, active = planes.min(-1)
                result[ids], plane[ids] = g, active
            if not torch.isfinite(result).all():
                raise RuntimeError('Nonfinite actual FOV label')
            return result, plane

    def verify(self, cache, device):
        """Mixed-x batches checked against scalar planes AND original upstream oracle."""
        rows, sensor = [], []
        d = cache.arrays['val']
        for s in range(8):
            pool = np.flatnonzero(d['s'] == s)[:2]
            if not len(pool):
                raise ValueError('Missing sensor in verification cache')
            for i in pool:
                x = torch.tensor(d['x'][i], device=device)
                qs = torch.tensor(d['q'][i][None] + np.array([-.02, 0, .02], np.float32)[:, None]
                                  * d['normal'][i], device=device)
                self.base.verify_against_upstream(x, qs)
                for q in qs:
                    rows.append(torch.cat((x, q)))
                    sensor.append(s)
        inp = torch.stack(rows)
        ss = torch.tensor(sensor, device=device)
        g, plane = self.margins(inp, ss)
        err = 0.
        for i, s in enumerate(sensor):
            planes = self.base.planes(inp[i, :3], inp[i:i+1, 3:], s)[0]
            err = max(err, abs(float(g[i] - planes.min())))
            if plane[i] != planes.argmin():
                raise RuntimeError('Pairwise FOV active plane mismatch')
        if err > 2e-6:
            raise RuntimeError(f'Pairwise FOV mismatch: {err}')
        return dict(status='PASS', queries=len(rows), max_abs_error_m=err,
                    actual_sign_from='upstream FK and conservative six-plane margins')


def radii_for_update(seed, step, count, rank=0, world=1, config=NeighborhoodConfig()):
    if seed < 0 or step < 0 or count <= 0 or count % world or not 0 <= rank < world:
        raise ValueError('Invalid neighbor sample partition')
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, config.rng_tag]))
    r = rng.uniform(config.radius_min, config.radius_max, count).astype(np.float32)
    n = count // world
    return r[rank*n:(rank+1)*n]


def build_queries(boundary, radii, oracle, lo, hi, config=NeighborhoodConfig()):
    inputs, sensor, normals, anchor_group = boundary
    if inputs.shape != (len(sensor), 10) or normals.shape != (len(sensor), 7):
        raise ValueError('Invalid matched anchors')
    device = inputs.device
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        r = torch.as_tensor(radii, device=device, dtype=torch.float32)
        if r.shape != sensor.shape or not torch.isfinite(r).all() or torch.any(r <= 0):
            raise ValueError('Radius must be finite and positive for each anchor')
        offset = (r[:, None] * torch.tensor([-1., 1.], device=device)).reshape(-1)
        inp = inputs.detach().float().repeat_interleave(2, 0)
        inp[:, 3:] += offset[:, None] * normals.detach().float().repeat_interleave(2, 0)
        ss = sensor.repeat_interleave(2)
        group = 2*anchor_group.repeat_interleave(2) + torch.arange(2, device=device).repeat(len(r))
        inside = ((inp[:, 3:] >= lo) & (inp[:, 3:] <= hi)).all(1)
        g = torch.zeros(len(inp), device=device)
        plane = torch.full((len(inp),), -1, device=device, dtype=torch.long)
        if inside.any():
            g[inside], plane[inside] = oracle.margins(inp[inside], ss[inside])
        if not torch.isfinite(g).all():
            raise RuntimeError('Nonfinite geometry; never silently relabel or drop')
        ambiguous = inside & (g.abs() <= config.ambiguous_g_m)
        valid = inside & ~ambiguous
        # Actual sign can differ from intended offset side when another plane becomes active.
        labels = torch.where(g >= 0, 1., -1.)
        safe = torch.where(inside[:, None], inp, inputs.repeat_interleave(2, 0)).detach()
        return dict(inputs=safe, proposed=inp.detach(), sensors=ss, group=group,
                    radius=offset.abs(), labels=labels, valid=valid, in_limits=inside,
                    ambiguous=ambiguous, margin=config.margin_per_rad*offset.abs(),
                    g_m=g, plane=plane, expected_side=torch.sign(offset))


def valid_counts(batch):
    return torch.bincount(batch['group'][batch['valid']], minlength=32).float()


def slice_batch(batch, start, end):
    return {key: value[start:end] for key, value in batch.items()}


def sign_loss(model, batch, global_counts, *, world_size=1, config=NeighborhoodConfig()):
    """Equal intended-side/origin/sensor strata; denominator is GLOBAL valid count.

    max(.25*r - sign(actual_g)*f_s, 0)^2 is an inequality penalty, not f_s=+/-r.
    Margin is in model-field units and is a fixed candidate hyperparameter.
    Empty global strata contribute zero and are explicitly reported, not resampled.
    """
    if global_counts.shape != (32,) or not torch.isfinite(global_counts).all() or (global_counts < 0).any():
        raise ValueError('Invalid global neighborhood counts')
    x, ss, group = batch['inputs'], batch['sensors'], batch['group']
    if group.shape != ss.shape or torch.any(group // 4 != ss) or torch.any((group < 0) | (group >= 32)):
        raise ValueError('Sensor/stratum mismatch')
    with torch.autocast(x.device.type, enabled=False):
        pred = model(x.float()).float()
        if pred.shape != (len(x), 9):
            raise ValueError('Expected [union,S0,...,S7]')
        value = pred.gather(1, (ss+1)[:, None])[:, 0]
        valid, y = batch['valid'], batch['labels']
        hinge = torch.relu(batch['margin'] - y*value).square()
        weighted = torch.where(valid, hinge, torch.zeros_like(hinge))
        sums = torch.zeros(32, device=x.device).index_add(0, group, weighted)
        loss = config.loss_weight * (sums/global_counts.float().clamp_min(1)).sum()/32
        loss = loss + (pred * 0).sum()  # Last DDP sync connects ALL params even if a rank has no valid rows.
        with torch.no_grad():
            positive, actual = value >= 0, y > 0
            rows = torch.stack((torch.ones_like(value), batch['in_limits'], valid, batch['ambiguous'],
                valid & positive & actual, valid & ~positive & ~actual,
                valid & positive & ~actual, valid & ~positive & actual,
                valid & (y*value < batch['margin']), weighted,
                valid & (y != batch['expected_side']), torch.where(valid, value.abs(), 0.)), 1).double()
            stats = torch.zeros((32, len(COLUMNS)), device=x.device, dtype=torch.float64)
            stats.index_add_(0, group, rows)
    return loss*world_size, stats


def summary(stats, config=NeighborhoodConfig()):
    data = stats.detach().cpu().double().numpy()
    def one(a):
        v = dict(zip(COLUMNS, a.sum(0).tolist()))
        count = v['valid']
        v['sign_accuracy'] = (v['tp']+v['tn'])/count if count else None
        v['positive_recall'] = v['tp']/(v['tp']+v['fn']) if v['tp']+v['fn'] else None
        v['negative_recall'] = v['tn']/(v['tn']+v['fp']) if v['tn']+v['fp'] else None
        v['out_of_limits'] = v['attempted']-v['in_limits']
        return v
    result = one(data)
    result['loss'] = float(config.loss_weight*np.sum(data[:, 9]/np.maximum(data[:, 2], 1))/32)
    result['missing_strata'] = np.flatnonzero(data[:, 2] == 0).tolist()
    result['heads'] = {f's{s}': one(data[4*s:4*s+4]) for s in range(8)}
    result['strata'] = {str(i): one(data[i:i+1]) for i in range(32)}
    return result


def hash_queries(proposal_hash, labeled_hash, ids, batch):
    # Persist proposed out-of-limits points too; accepted-only hashing could hide exclusions.
    proposal_hash.update(np.ascontiguousarray(ids).tobytes())
    for key in ('proposed', 'sensors', 'radius', 'group'):
        proposal_hash.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
    for key in ('g_m', 'labels', 'valid', 'in_limits', 'plane', 'margin'):
        labeled_hash.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
