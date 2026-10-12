"""Training-only, piecewise-differentiable copy of the frozen single-point solver.

NO change to runtime_probe/per_sensor_visibility_runtime. Discrete control-flow
predicates are stop-gradient; selected continuous steps retain their graph.
The reference-value parity test is mandatory; this is not a globally smooth
surrogate and does not differentiate changes of root-selection branch.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import Tensor
from torch.nn import functional as F

@dataclass(frozen=True)
class Settings:
    projection_iters: int = 10
    projection_damping: float = .5
    projection_epsilon_f: float = .03
    projection_max_step_norm: float = .25
    root_refine_iters: int = 12
    root_tolerance_f: float = .002
    branch_ascent_steps: int = 1
    branch_step_size: float = .05
    branch_max_step_norm: float = .25
    branch_fallback_ascent_steps: int = 8

SETTINGS = Settings()

def pick(mask: Tensor, a: Tensor, b: Tensor) -> Tensor:
    return torch.where(mask if a.ndim == 1 else mask[:, None], a, b)

def choose(a: Tensor) -> Tensor:
    return a.detach()

def clamp(q: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    return torch.maximum(torch.minimum(q, hi), lo)

def value_grad(score, q: Tensor, mask: Tensor, training: bool):
    # q is deliberately NOT detached: endpoints must stay connected to parameters.
    if not q.requires_grad:
        q = q.clone().requires_grad_(True)
    value = score(q)
    if value.ndim != 1 or len(value) != len(q):
        raise ValueError('Expected one score per independent sample')
    grad = torch.autograd.grad(value.sum(), q, create_graph=training,
                               retain_graph=True)[0] * mask
    if not torch.isfinite(value).all() or not torch.isfinite(grad).all():
        raise FloatingPointError('Nonfinite value/gradient in training rollout')
    return value, grad

def project(q, f, g, lo, hi, settings=SETTINGS):
    n2 = g.square().sum(1)
    raw = f[:, None] * g / n2.clamp_min(1e-8)[:, None]
    norm = torch.linalg.vector_norm(raw, dim=1)
    scale = (settings.projection_max_step_norm / norm.clamp_min(1e-8)).clamp_max(1.)
    return clamp(q - settings.projection_damping * raw * scale[:, None], lo, hi), choose(n2 >= 1e-12)

def ascend(q, g, lo, hi, settings=SETTINGS):
    norm = torch.linalg.vector_norm(g, dim=1)
    raw = settings.branch_step_size * g / norm.clamp_min(1e-8)[:, None]
    scale = (settings.branch_max_step_norm /
             torch.linalg.vector_norm(raw, dim=1).clamp_min(1e-8)).clamp_max(1.)
    return clamp(q + raw * scale[:, None], lo, hi), choose(norm >= 1e-8)

def bisect(score, qa, fa, qb, fb, selected, settings=SETTINGS):
    left, right, flo = qa, qb, fa
    good_left = choose(fa.abs() <= fb.abs())
    best_q, best_f = pick(good_left, qa, qb), pick(good_left, fa, fb)
    running = selected.clone()
    for _ in range(settings.root_refine_iters):
        if not running.any(): break
        mid = (left + right) * .5
        fm = score(mid)
        improve = running & choose(fm.abs() < best_f.abs())
        best_q, best_f = pick(improve, mid, best_q), pick(improve, fm, best_f)
        done = running & choose(fm.abs() <= settings.root_tolerance_f)
        best_q, best_f = pick(done, mid, best_q), pick(done, fm, best_f)
        running = running & ~done
        first_half = choose(flo * fm <= 0)
        right = pick(running & first_half, mid, right)
        left = pick(running & ~first_half, mid, left)
        flo = pick(running & ~first_half, fm, flo)
    return best_q, best_f

def solve(score, start: Tensor, mask: Tensor, lo: Tensor, hi: Tensor,
          *, training=True, settings=SETTINGS):
    """Batched single-obligation branch solver; values match the legacy solver.

    score(q) must operate independently per row (no batchnorm/cross-row mixing).
    Masks/limits use original seven-joint order. Existing runtime is untouched.
    """
    if start.ndim != 2 or start.shape[1] != 7 or mask.shape != start.shape:
        raise ValueError('Expected [N,7] start and chain masks')
    q = clamp(start.detach().clone().requires_grad_(True), lo, hi)
    f = score(q)
    best_q, best_f = q, f
    found = choose((f >= 0) | (f.abs() <= settings.projection_epsilon_f))
    qzero, fzero = q, f
    alive = ~found
    initial_accept = found.clone()
    bracket = torch.zeros_like(found)
    for _ in range(settings.projection_iters):
        if not alive.any(): break
        fv, grad = value_grad(score, q, mask, training)
        nq, valid = project(q, fv, grad, lo, hi, settings)
        alive = alive & valid
        if not alive.any(): break
        nf = score(nq)
        improved = alive & choose(nf > best_f)
        best_q, best_f = pick(improved, nq, best_q), pick(improved, nf, best_f)
        crossing = alive & choose((f * nf <= 0) & (f != nf))
        if crossing.any():
            rq, rf = bisect(score, q, f, nq, nf, crossing, settings)
            qzero, fzero = pick(crossing, rq, qzero), pick(crossing, rf, fzero)
            bracket |= crossing
        near = alive & ~crossing & choose(nf.abs() <= settings.projection_epsilon_f)
        qzero, fzero = pick(near, nq, qzero), pick(near, nf, fzero)
        found = found | crossing | near
        alive = alive & ~found
        q, f = pick(alive, nq, q), pick(alive, nf, f)

    # Root path: one normalized ascent. Fallback: eight steps, keep max score.
    qvis = pick(found, qzero, best_q)
    fallback_best, fallback_score = best_q, best_f
    root_live, fallback_live = found.clone(), ~found
    for i in range(max(settings.branch_ascent_steps, settings.branch_fallback_ascent_steps)):
        active_root = root_live & (i < settings.branch_ascent_steps)
        active_fallback = fallback_live & (i < settings.branch_fallback_ascent_steps)
        active = active_root | active_fallback
        if not active.any(): break
        _, grad = value_grad(score, qvis, mask, training)
        nxt, valid = ascend(qvis, grad, lo, hi, settings)
        root_live, fallback_live = root_live & valid, fallback_live & valid
        apply = active & valid
        qvis = pick(apply, nxt, qvis)
        nv = score(qvis)
        improved = active_fallback & valid & choose(nv > fallback_score)
        fallback_best = pick(improved, qvis, fallback_best)
        fallback_score = pick(improved, nv, fallback_score)
    candidate = pick(found, qvis, fallback_best)
    qzero = pick(found, qzero, best_q)
    if not torch.isfinite(candidate).all():
        raise FloatingPointError('Nonfinite candidate')
    return candidate, {'root_found':found, 'initial_accepted':initial_accept,
                       'bracket':bracket, 'q_zero':qzero}

def outcome_loss(score, geometry, x, qminus, qplus, sensors, masks, lo, hi, *, training=True):
    """Witness-conditioned terminal FOV penalty, no new distance/normal labels.

    Both negative and already-visible starts are retained. Positive witness only
    sets a feasible local target; NEVER requires a global infeasible query to pass.
    """
    with torch.no_grad():
        gm, gp = geometry(x, qminus, sensors), geometry(x, qplus, sensors)
        valid = (gm < -1e-5) & (gp > 1e-5)
        if not torch.isfinite(gm).all() or not torch.isfinite(gp).all():
            raise ValueError('Nonfinite analytic witness margin')
        target = torch.minimum(gp * .5, gp.new_full(gp.shape, .005))
    starts = torch.cat([qminus, qplus])
    xx, ss = torch.cat([x,x]), torch.cat([sensors,sensors])
    candidate, diag = solve(score, starts, masks[ss], lo, hi, training=training)
    g = geometry(xx, candidate, ss)
    target = torch.cat([target,target])
    error = ((target - g) / .01).clamp_min(0)
    eligible = torch.cat([valid,valid])
    count = eligible.sum().to(g.dtype)
    each = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
    penalty = each[eligible].sum() / count.clamp_min(1)
    # Connected zero keeps all parameter paths defined even when every sample stops.
    loss = penalty + score(starts).sum() * 0.
    n = len(x)
    d = valid.sum().clamp_min(1)
    metrics = torch.stack([loss.detach(), ((g[:n] >= 0) & valid).float().sum()/d,
        ((g[n:] >= 0) & valid).float().sum()/d, (g[:n].detach()*valid).sum()/d,
        (g[n:].detach()*valid).sum()/d, ((g >= target) & eligible).float().sum()/count.clamp_min(1),
        (diag['initial_accepted'] & eligible).float().sum()/count.clamp_min(1), count,
        (1.-valid.float()).mean()])
    return loss, metrics
