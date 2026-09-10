"""Pure numerical helpers for read-only boundary diagnostics (no ROS/runtime writes)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    root_tol_m: float = 1e-5
    root_iters: int = 40
    root_step_rad: float = 0.25
    line_search_steps: int = 10
    plane_gap_m: float = 1e-4
    min_gradient: float = 1e-6
    joint_margin_rad: float = 0.002
    fd_step_rad: float = 1e-3
    fd_relative_tol: float = 0.05
    tangent_radius_rad: float = 0.10
    offbank_min_rad: float = 1e-3
    offsets_rad: tuple = (0.005, 0.01, 0.02, 0.05)

    def validate(self):
        for key, value in asdict(self).items():
            values = value if isinstance(value, tuple) else (value,)
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"Expected positive finite {key}")
        if self.joint_margin_rad <= self.fd_step_rad:
            raise ValueError("joint_margin_rad must exceed fd_step_rad")


def json_safe(value):
    """NaN is missing/not measured, never a fabricated zero."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, obj):
    path.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def distribution(values):
    values = list(values)
    a = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if not len(a):
        return {"count": 0, "missing": len(values), "mean": None,
                "p05": None, "p50": None, "p95": None, "max": None}
    return {"count": len(a), "missing": len(values)-len(a), "mean": float(a.mean()),
            **{k: float(np.quantile(a, p)) for k, p in (("p05", .05), ("p50", .5), ("p95", .95))},
            "max": float(a.max())}


def rate(numerator, denominator):
    return {"count": int(denominator), "passed": int(numerator),
            "rate": numerator / denominator if denominator else None}


def cosine(a: torch.Tensor, b: torch.Tensor):
    na, nb = a.norm().item(), b.norm().item()
    if not math.isfinite(na + nb) or min(na, nb) < 1e-8:
        return None
    return float(torch.dot(a.reshape(-1), b.reshape(-1)).item() / (na * nb))


def within(q, lo, hi):
    return bool(torch.isfinite(q).all() and ((q >= lo) & (q <= hi)).all())


def bank_label(q: torch.Tensor, bank: torch.Tensor, mask: torch.Tensor, sign: float):
    """Same eps=1e-8 squared-distance floor as legacy labels, plus unclamped distance.

    bank contains ONLY valid finite rows for this x/sensor. No global sensor max.
    At an exact bank anchor the legacy distance floor is 1e-4 rad and gradient=0;
    this is NOT a valid boundary normal. Nearest ties are reported.
    """
    if bank.ndim != 2 or bank.shape[1] != 7 or not len(bank):
        raise ValueError("Expected nonempty finite bank [K,7]")
    if not torch.isfinite(bank).all() or not torch.isfinite(q).all():
        raise ValueError("Non-finite valid bank or query")
    d = (q[None] - bank) * mask[None]
    sq = d.square().sum(-1)
    vals, inds = sq.sort()
    k = int(sq.argmin())  # Match upstream torch.min first-index tie behavior.
    exact = vals[0].clamp_min(0).sqrt()
    floor = vals[0].clamp_min(1e-8).sqrt()
    grad = float(sign) * d[k] / floor
    gap = float((vals[1].sqrt() - exact).item()) if len(vals) > 1 else None
    return {"distance_rad": float(exact), "legacy_signed_value": float(sign * floor),
            "gradient": grad, "nearest_slot_in_valid_bank": k,
            "nearest_gap_rad": gap,
            "gradient_valid": bool(exact > 1e-4 and (gap is None or gap > 1e-6))}


def field_metrics(value, grad, mask, normal=None):
    projected = grad * mask
    return {"value": float(value), "abs_value": abs(float(value)),
            "raw_gradient_norm": float(grad.norm()),
            "masked_gradient_norm": float(projected.norm()),
            "inactive_gradient_norm": float((grad * (1-mask)).norm()),
            "raw_normal_cosine": cosine(grad, normal) if normal is not None else None,
            "masked_normal_cosine": cosine(projected, normal) if normal is not None else None}


def refine_boundary(value_grad: Callable, q, mask, lo, hi, cfg: Config):
    """Audit-only damped Newton/line search on analytic g_s, NOT a runtime solver change.

    It finds a zero (if successful), NOT a certified nearest C-space boundary.
    No joint clamping: infeasible proposals are backtracked, never re-labelled.
    """
    cfg.validate()
    q = q.detach().clone()
    if not within(q, lo, hi):
        return {"q": q, "g": None, "accepted": False, "reason": "seed_out_of_bounds", "iterations": 0}
    reason, history = "iteration_budget", []
    for it in range(cfg.root_iters + 1):
        g, grad = value_grad(q)
        history.append(float(g))
        grad = grad.detach() * mask
        if not math.isfinite(g) or not torch.isfinite(grad).all():
            reason = "nonfinite_geometry"
            break
        if abs(g) <= cfg.root_tol_m:
            return {"q": q, "g": g, "accepted": True, "reason": "analytic_boundary",
                    "iterations": it, "residual_history": history}
        if grad.norm() < cfg.min_gradient:
            reason = "degenerate_geometry_gradient"
            break
        if it == cfg.root_iters:
            break
        step = g * grad / grad.square().sum().clamp_min(1e-12)
        step *= min(1.0, cfg.root_step_rad / max(float(step.norm()), 1e-12))
        accepted = False
        for back in range(cfg.line_search_steps):
            candidate = q - step * (0.5 ** back)
            if not within(candidate, lo, hi):
                continue
            next_g, _ = value_grad(candidate)
            if math.isfinite(next_g) and abs(next_g) < abs(g):
                q, accepted = candidate.detach(), True
                break
        if not accepted:
            reason = "line_search_stalled"
            break
    return {"q": q, "g": history[-1] if history else None, "accepted": False,
            "reason": reason, "iterations": len(history)-1, "residual_history": history}


def qualify_boundary(oracle, x, q, s, mask, lo, hi, cfg):
    """Reject nonsmooth/degenerate/limit-near normals; log reasons, don't hide them."""
    g, grad = oracle.value_grad(x, q, s)
    planes = oracle.planes(x, q.reshape(1, 7), s).detach()[0]
    if not math.isfinite(g) or not torch.isfinite(grad).all() or not torch.isfinite(planes).all():
        raise RuntimeError("Non-finite analytic geometry")
    smallest = planes.sort().values
    active = int(planes.argmin())
    gap = float(smallest[1] - smallest[0])
    projected = grad * mask
    norm = float(projected.norm())
    margin = float(torch.minimum(q-lo, hi-q)[mask.bool()].min())
    reasons = []
    if not within(q, lo, hi): reasons.append("joint_limits")
    if abs(g) > cfg.root_tol_m: reasons.append("not_analytic_boundary")
    if gap < cfg.plane_gap_m: reasons.append("fov_plane_tie")
    if norm < cfg.min_gradient: reasons.append("degenerate_normal")
    if margin <= cfg.joint_margin_rad: reasons.append("near_joint_limit")
    fd_relative, fd_cos = None, None
    if not reasons:
        fd = torch.zeros_like(q)
        stable = True
        for j in torch.where(mask.bool())[0].tolist():
            dq = torch.zeros_like(q); dq[j] = cfg.fd_step_rad
            pair = oracle.planes(x, torch.stack([q+dq, q-dq]), s).detach()
            stable = stable and bool((pair.argmin(-1) == active).all())
            fd[j] = (pair[0].min() - pair[1].min()) / (2*cfg.fd_step_rad)
        fd_relative = float((fd-projected).norm() / projected.norm().clamp_min(1e-8))
        fd_cos = cosine(fd, projected)
        if not stable: reasons.append("fd_active_plane_changed")
        if fd_relative > cfg.fd_relative_tol: reasons.append("normal_fd_mismatch")
    return {"regular": not reasons, "reasons": reasons, "g_m": g,
            "active_plane": active, "plane_gap_m": gap, "normal_norm": norm,
            "joint_margin_rad": margin, "fd_relative_error": fd_relative, "fd_cosine": fd_cos,
            "normal": projected / max(norm, 1e-12) if norm >= cfg.min_gradient else None}


def tangent_seed(q, normal, mask, rng, radius):
    v = torch.as_tensor(rng.normal(size=7), device=q.device, dtype=q.dtype) * mask
    v = v - torch.dot(v, normal) * normal
    if v.norm() < 1e-8:
        return None
    return q + radius * v / v.norm()


class SensorField:
    """Model inference only; raw gradients are measured before runtime chain masking."""
    def __init__(self, model):
        self.model = model

    def value_grad(self, x, q, s):
        with torch.enable_grad(), torch.autocast(q.device.type, enabled=False):
            qv = q.detach().float().reshape(1, 7).clone().requires_grad_(True)
            y = self.model(torch.cat([x.detach().float().reshape(1, 3), qv], -1))
            if y.shape != (1, 8): raise ValueError(f"Expected sensor view [1,8], got {y.shape}")
            v = y[0, s]
            grad = torch.autograd.grad(v, qv)[0][0]
        if not torch.isfinite(v) or not torch.isfinite(grad).all():
            raise RuntimeError("Non-finite model field/gradient")
        return float(v.detach()), grad.detach()
