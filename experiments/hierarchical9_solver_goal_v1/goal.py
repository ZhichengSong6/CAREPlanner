"""Short differentiable projection/ascent SURROGATE, not the full runtime solver.

Raw network values and q-gradients determine the update; true FOV determines
its loss. There is NO distance/normal/zero/slope target. No old solver changes.
All newly computed intermediate q stay attached to theta through the graph.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def paired_planes(oracle, x, q, sensor):
    """Exact upstream FK and normalized six planes, with paired rather than shared x."""
    T = oracle.fk(oracle.specs[sensor], q)
    diff = x - T[:, :3, 3]
    pc = torch.einsum('nji,nj->ni', T[:, :3, :3], diff)
    a, b, z = pc.unbind(-1)
    tx, ty = math.tan(math.radians(oracle.hfov)/2), math.tan(math.radians(oracle.vfov)/2)
    nx, ny = math.sqrt(1+tx*tx), math.sqrt(1+ty*ty)
    return torch.stack(((a+z*tx)/nx, (-a+z*tx)/nx,
                        (b+z*ty)/ny, (-b+z*ty)/ny,
                        z-oracle.zmin, oracle.zmax-z), -1)-oracle.delta


def rollout(model, x, q0, sensor, mask, lo, hi, steps=3):
    """Three damped/capped projection updates, then ONE capped normalized ascent.

    Same individual update formulas as legacy runtime; intentionally NO learned
    early stopping, bisection or fallback branches. This is a training surrogate.
    Inactive joints are masked at every update. Do not detach any intermediate q.
    """
    q = q0.detach().clone().requires_grad_(True)
    for k in range(steps+1):
        y = model.forward_sensor(torch.cat((x, q), -1), sensor)
        g = torch.autograd.grad(y.sum(), q, create_graph=True, retain_graph=True)[0]*mask
        if k < steps:
            raw = y[:, None]*g/g.square().sum(-1, keepdim=True).clamp_min(1e-8)
            cap = (.25/raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
            dq = -.5*raw*cap
            dq = torch.where(g.square().sum(-1, keepdim=True) >= 1e-12, dq, torch.zeros_like(dq))
        else:
            gn = g.norm(dim=-1, keepdim=True)
            dq = .05*g/gn.clamp_min(1e-8)
            dq = torch.where(gn >= 1e-8, dq, torch.zeros_like(dq))
        q = torch.maximum(torch.minimum(q+dq, hi), lo)
    return q


def geometry_objective(model, batch, oracle, masks, lo, hi, cfg):
    losses, metrics = [], []
    with torch.enable_grad(), torch.autocast(device_type=lo.device.type, enabled=False):
        for s, (x, minus, plus) in enumerate(batch):
            with torch.no_grad():
                gp = paired_planes(oracle, x, plus, s).min(-1).values
                gm = paired_planes(oracle, x, minus, s).min(-1).values
                if (not torch.isfinite(gp).all() or not torch.isfinite(gm).all()
                        or not (gp > 0).all() or not (gm < 0).all()):
                    raise ValueError(f'Invalid ORIGINAL signed witness pair for S{s}; no relabelling')
                # Target tied to a REAL same-x/sensor feasible witness, not arbitrary 1cm.
                target = (cfg['witness_fraction']*gp).clamp(max=cfg['target_cap_m'])
            # Both originally outside and originally inside starts: retain easy cases.
            xx = torch.cat((x, x), 0)
            q0 = torch.cat((minus, plus), 0)
            qk = rollout(model, xx, q0, s, masks[s], lo, hi, cfg['projection_steps'])
            gk = paired_planes(oracle, xx, qk, s).min(-1).values
            deficit = F.relu((torch.cat((target, target))-gk)/cfg['geometry_scale_m'])
            loss = F.smooth_l1_loss(deficit, torch.zeros_like(deficit), reduction='mean')
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite differentiable FOV objective S{s}')
            losses.append(loss)
            n = len(minus)
            metrics.append(torch.stack((loss.detach(), gk.detach().mean(),
                (gk[:n]>=0).float().mean(), (gk[n:]>=0).float().mean(),
                gp.mean(), gm.mean(), target.mean(),
                (qk.detach()-q0).norm(dim=-1).mean())))
        return torch.stack(losses).mean(), torch.stack(metrics)


def weight_at(step, cfg):
    return cfg['weight']*min(1., max(0., (step-cfg['warmup_updates'])/cfg['ramp_updates']))


class WitnessPool:
    """Use ALREADY BUILT verified +/- V4 pair cache. Only original R1 train/val x.

    Source q and labels are never changed. A new local NumPy Generator is used for
    each draw, so the original R1 torch/NumPy/Python random streams are untouched.
    """
    def __init__(self, cache, pairs_path, train_x, val_x, cfg):
        import json
        self.cfg, self.x = cfg, cache.x
        p = Path(pairs_path)
        self.manifest = json.loads((p/'manifest.json').read_text())
        m = self.manifest
        if (m.get('format') != 'care_h9_v4_verified_pairs_v1'
                or m.get('status') != 'COMPLETE' or m.get('cache_identity') != cache.identity):
            raise ValueError('Wrong or incomplete previously built V4 pair cache')
        if np.intersect1d(train_x, val_x).size:
            raise ValueError('Original R1 train/validation x overlap')
        self.a, self.groups, self.counts = {}, {}, {}
        for split, allowed in (('train', train_x), ('val', val_x)):
            a = {k: np.load(p/split/(k+'.npy'), mmap_mode='r', allow_pickle=False)
                 for k in ('x_index', 'sensor', 'radius_id', 'q_plus', 'q_minus')}
            n = int(m['total_by_split'][split])
            if any(len(v)!=n for v in a.values()):
                raise ValueError('Pair-cache field length mismatch')
            eligible = np.isin(a['x_index'], allowed)
            cells = a['sensor'].astype(np.int16)*3+a['radius_id'].astype(np.int16)
            groups = [np.flatnonzero(eligible & (cells==c)) for c in range(24)]
            if any(len(g)==0 for g in groups):
                raise ValueError(f'Missing R1-{split} aligned sensor/radius pair stratum')
            self.a[split], self.groups[split] = a, groups
            self.counts[split] = [len(g) for g in groups]

    def draw(self, split, step, rank, device):
        a = self.a[split]
        n = self.cfg['pairs_per_sensor_per_rank']
        batch = []
        for s in range(8):
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.cfg['aux_seed'], step, rank, s, int(split=='val')]))
            ix = []
            for j in range(n):
                radius = (j+step+rank*n)%3
                pool = self.groups[split][s*3+radius]
                ix.append(pool[rng.integers(len(pool))])
            ix = np.asarray(ix, np.int64)
            def tensor(v):
                return torch.as_tensor(np.asarray(v).copy(), device=device, dtype=torch.float32)
            x = tensor(self.x[a['x_index'][ix]])
            batch.append((x, tensor(a['q_minus'][ix]), tensor(a['q_plus'][ix])))
        return batch


def inject_once(base, original_runner, ddp, batches, counts, weights, args,
                optimizer, scaler, aux_call):
    """One auxiliary graph per update/retry; reuse original runner/DDP/scaler.

    Original loss stats and scheduler metric unchanged. One final .backward() on
    original+aux sends PARAMETER gradients through DDP; autograd.grad is q-only.
    Aux is a local mean with equal per-rank/per-sensor counts -> DDP mean is correct.
    """
    saved = base.loss_for_microbatch
    calls = 0
    observed = None
    def combined(model, *data, **kw):
        nonlocal calls, observed
        loss, stats = saved(model, *data, **kw)
        # On AMP retries original runner replays exact same ordered microbatches.
        if calls % len(batches) == 0:
            aux, observed = aux_call(ddp.module)
            loss = loss+aux
        calls += 1
        return loss, stats
    base.loss_for_microbatch = combined
    try:
        result = original_runner(ddp, batches, counts, weights, args, optimizer, scaler)
    finally:
        base.loss_for_microbatch = saved
    if calls % len(batches) or observed is None:
        raise RuntimeError('Original R1 runner/auxiliary invocation contract changed')
    return result, observed
