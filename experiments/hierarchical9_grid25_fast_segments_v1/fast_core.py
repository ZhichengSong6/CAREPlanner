"""Target-only Adam refinement; same candidate objective, bounds and return schema.

Uses upstream FK/FOV arithmetic. It does NOT reduce q proposals, candidates,
Adam iterations, or change epsilon. No LOS and no neural model are involved.
"""
from __future__ import annotations
from contextlib import contextmanager
import time
import numpy as np
import torch

ENGINE = 'target_only_adam_reuse_forward_v1'


def target_value(ex, point, q, specs, a, sensor):
    # A singleton list skips the seven unused forward AND backward graphs.
    _, margins, _, _ = ex.visibility_g_batch(
        point, q, [specs[sensor]], a.horizontal_fov_deg,
        a.vertical_fov_deg, a.z_min, a.z_max, a.delta)
    return margins[:, 0] - a.delta


def refine(ex, point_np, q_init_np, all_chain_specs, q_min, q_max,
           args, device, target_sensor=-1):
    """Drop-in upstream refine_q_to_zero_level signature plus explicit ex module."""
    if args.zero_level_optimizer != 'adam' or not 0 <= target_sensor < len(all_chain_specs):
        raise ValueError('Fast engine supports the verified per-sensor Adam profile only')
    point = ex.torch_tensor(point_np, device)
    q = ex.torch_tensor(q_init_np, device).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([q], lr=args.lr)  # Same upstream defaults.
    best_q = q.detach().clone()
    best_abs = None
    initial_g = None
    for _ in range(args.max_iter):
        optimizer.zero_grad()
        g = target_value(ex, point, q, all_chain_specs, args, target_sensor)
        # The preceding post-update check equals this pre-update check. Reuse
        # this forward graph for backward, but do not detach the loss itself.
        with torch.no_grad():
            absolute = g.abs()
            if best_abs is None:
                initial_g = g.detach().clone()
                best_abs = absolute.clone()
            else:
                improve = absolute < best_abs
                best_abs[improve] = absolute[improve]
                best_q[improve] = q.detach()[improve]
        torch.mean(g * g).backward()
        optimizer.step()
        ex.clamp_q_(q, q_min, q_max)
    with torch.no_grad():
        # Includes the final iterate, just as the old post-update check does.
        last_g = target_value(ex, point, q, all_chain_specs, args, target_sensor)
        improve = last_g.abs() < best_abs
        best_q[improve] = q.detach()[improve]
        # Full eight-sensor diagnostics are still produced ONCE at the end.
        g, margins, active_sensor, active_planes = ex.visibility_objective_batch(
            point, best_q, all_chain_specs, args.horizontal_fov_deg,
            args.vertical_fov_deg, args.z_min, args.z_max, args.delta, target_sensor)
        keep = (g.abs() <= args.epsilon) & ((best_q > q_min) & (best_q < q_max)).all(dim=1)
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        return (best_q[keep].cpu().numpy(), g[keep].cpu().numpy(),
                margins[keep].cpu().numpy(), active_sensor[keep].cpu().numpy(),
                active_planes[keep].cpu().numpy(),
                q.detach().new_tensor(q_init_np)[keep].cpu().numpy(),
                initial_g[keep].cpu().numpy(), indices.cpu().numpy())


@contextmanager
def installed(gen, ex):
    """Only patch this process's imported generator; no source files are edited."""
    old = gen.refine_q_to_zero_level
    def replacement(*args, **kwargs):
        return refine(ex, *args, **kwargs)
    gen.refine_q_to_zero_level = replacement
    try:
        yield
    finally:
        gen.refine_q_to_zero_level = old


def compare_arrays(old, new, label, atol):
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape or old.dtype != new.dtype:
        raise ValueError(f'{label}: shape/dtype mismatch {old.shape}/{old.dtype} vs {new.shape}/{new.dtype}')
    if old.dtype.kind in 'biu':
        if not np.array_equal(old, new):
            raise ValueError(f'{label}: discrete selection/mask/face changed')
        return 0.
    if not np.array_equal(np.isfinite(old), np.isfinite(new)):
        raise ValueError(f'{label}: finite mask changed')
    if not np.array_equal(np.isposinf(old), np.isposinf(new)):
        raise ValueError(f'{label}: invalid +inf slots changed')
    if np.isnan(old).any() or np.isnan(new).any():
        raise ValueError(f'{label}: NaN')
    finite = np.isfinite(old)
    error = float(np.max(np.abs(old[finite]-new[finite]))) if finite.any() else 0.
    if error > atol:
        raise ValueError(f'{label}: error {error:.9g} exceeds {atol:.9g}; production STOP')
    return error


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def parity_gate(gen, ex, specs, lo, hi, limits, a, device, points, ids):
    """Same-device qualification using actual inherited proposal/iteration counts.

    Two grid points x eight sensors; retain exact acceptance/index agreement,
    q differences <=1e-5 rad, margins <=2e-6 m, derivatives <=5e-6.
    This is a sampled implementation check, NOT a proof for the entire grid.
    """
    rows = []
    for pid, point in zip(ids, points):
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, int(pid)]))
        qi = gen.sample_q_batch(rng, limits, a.joint_names, a.q_init_per_p)
        with torch.no_grad():
            _, margins, _, _ = ex.visibility_g_batch(
                ex.torch_tensor(point, device), ex.torch_tensor(qi, device), specs,
                a.horizontal_fov_deg, a.vertical_fov_deg, a.z_min, a.z_max, a.delta)
        margins = margins.cpu().numpy()
        for s in range(8):
            g0 = margins[:, s]-a.delta
            count = min(a.optimize_top_k_per_sensor, len(qi))
            idx = np.argpartition(np.abs(g0), count-1)[:count]
            idx = idx[np.argsort(np.abs(g0[idx]))]
            seeds = qi[idx]
            # Gradients on identical queries, before any Adam divergence.
            qs = ex.torch_tensor(seeds[:32], device).clone().requires_grad_(True)
            pt = ex.torch_tensor(point, device)
            go = ex.visibility_objective_batch(pt, qs, specs, a.horizontal_fov_deg,
                 a.vertical_fov_deg, a.z_min, a.z_max, a.delta, s)[0]
            jo = torch.autograd.grad(go.sum(), qs)[0]
            gn = target_value(ex, pt, qs, specs, a, s)
            jn = torch.autograd.grad(gn.sum(), qs)[0]
            ve = compare_arrays(go.detach().cpu().numpy(), gn.detach().cpu().numpy(), 'value', 2e-6)
            je = compare_arrays(jo.detach().cpu().numpy(), jn.detach().cpu().numpy(), 'gradient', 5e-6)
            # Alternate timed order; sync covers CUDA asynchronous execution.
            outputs = {}; elapsed = {}
            for name in (('old','fast') if (int(pid)+s)%2 == 0 else ('fast','old')):
                sync(device); start = time.perf_counter()
                outputs[name] = (ex.refine_q_to_zero_level(
                    point, seeds, specs, lo, hi, a, device, target_sensor=s)
                    if name == 'old' else refine(ex, point, seeds, specs, lo, hi, a, device, s))
                sync(device); elapsed[name] = time.perf_counter()-start
            errors = [compare_arrays(x,y,f'S{s}/output{k}',1e-5 if k in (0,5) else 2e-6)
                      for k,(x,y) in enumerate(zip(outputs['old'],outputs['fast']))]
            rows.append(dict(point_id=int(pid), sensor=s, kept=len(outputs['old'][0]),
                             value_max_abs=ve, gradient_max_abs=je, output_max_abs=errors,
                             old_seconds=elapsed['old'], fast_seconds=elapsed['fast']))
    old = sum(r['old_seconds'] for r in rows); new = sum(r['fast_seconds'] for r in rows)
    return dict(status='PASS', engine=ENGINE, device=str(device), gpu=torch.cuda.get_device_name(device)
                if torch.device(device).type=='cuda' else 'CPU', torch=str(torch.__version__),
                cuda=torch.version.cuda, rows=rows, refinement_speedup=old/new,
                q_atol_rad=1e-5, margin_atol_m=2e-6, gradient_atol=5e-6,
                scope='Same-device sampled parity; speedup is refinement ONLY, not whole dataset')
