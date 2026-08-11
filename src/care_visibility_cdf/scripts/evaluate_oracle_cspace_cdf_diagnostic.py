#!/usr/bin/env python3
"""
CAREPlanner Visibility CDF: oracle configuration-space CDF diagnostic.

Purpose
-------
This is intentionally a *decision diagnostic*, not another open-ended sweep.
It answers one specific question raised by the inside-gradient experiment:

    In deep-visible configurations, does the learned gradient disagree with the
    true FOV-margin gradient because the network learned the wrong CDF geometry,
    or because a configuration-space signed distance field and the geometric
    FOV margin are genuinely different objectives away from the zero level set?

We numerically estimate the configuration-space signed Euclidean distance

    d_C(q; x) = +dist(q, {q': g(x,q')=0})   if g(x,q) > 0
                -dist(q, {q': g(x,q')=0})   if g(x,q) < 0

under joint limits.  If q_b is the nearest boundary point, then (when the
nearest point is locally unique)

    grad_q d_C ~= sign(g(q)) * (q - q_b) / ||q - q_b||.

The nearest boundary point is estimated numerically with:
  1) multi-start oracle Newton projection to g=0;
  2) boundary-tangent distance minimization (KKT refinement);
  3) selecting the valid boundary candidate with the smallest ||q_b-q||.

The script REUSES the already-computed balanced samples and gradients from
inside_gradient_diagnostic_samples.npz.  Therefore it does not resample data or
recompute learned gradients.

Recommended decisive run
------------------------
Use only three bins:
    outside_boundary, shallow_inside, deep_inside
with 1,000 samples/bin.  That is enough to distinguish the two hypotheses.

Interpretation
--------------
Deep-inside case:
  A) cos(grad f, grad d_C) high, but cos(grad g, grad d_C) low
       -> objective mismatch: learned field is CDF-like, while g is a different
          interior objective.  Do NOT force global grad-f == grad-g training.

  B) cos(grad f, grad d_C) also low
       -> learned deep-interior CDF geometry is inaccurate; training coverage /
          objective should be improved if deep-interior CDF behavior matters.

Important
---------
The oracle CDF is still a numerical estimate.  Search-quality diagnostics
(boundary residual, success rate, and closest-point KKT normal alignment) are
reported and must be checked before interpreting the gradient comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    load_npz_data,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_str_list(text: str) -> List[str]:
    values = [v.strip() for v in str(text).split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated name.")
    return values


def stats(values: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"), "std": float("nan"), "min": float("nan"),
            "p01": float("nan"), "p05": float("nan"), "p10": float("nan"),
            "p25": float("nan"), "p50": float("nan"), "p75": float("nan"),
            "p90": float("nan"), "p95": float("nan"), "p99": float("nan"),
            "max": float("nan"),
        }
    q = np.quantile(values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(q[0]), "p05": float(q[1]), "p10": float(q[2]),
        "p25": float(q[3]), "p50": float(q[4]), "p75": float(q[5]),
        "p90": float(q[6]), "p95": float(q[7]), "p99": float(q[8]),
        "max": float(np.max(values)),
    }


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    valid = (an > eps) & (bn > eps)
    out = np.full((a.shape[0],), np.nan, dtype=np.float64)
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
    out[valid] = np.clip(out[valid], -1.0, 1.0)
    return out, valid


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return float("nan")
    aa = a[m] - np.mean(a[m])
    bb = b[m] - np.mean(b[m])
    denom = math.sqrt(float(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom <= 1e-15:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def clamp_q(q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(q, q_max[None, :]), q_min[None, :])


@torch.no_grad()
def paired_oracle_visibility_g(
    x: torch.Tensor,
    q: torch.Tensor,
    oracle: PinocchioFOVOracle,
) -> torch.Tensor:
    """Exact paired oracle g(x[i],q[i]) without an NxN cross product."""
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected x [N,3], got {tuple(x.shape)}")
    if q.ndim != 2 or q.shape[1] != len(oracle.joint_names):
        raise ValueError(f"Expected q [N,{len(oracle.joint_names)}], got {tuple(q.shape)}")
    if x.shape[0] != q.shape[0]:
        raise ValueError("paired oracle requires x and q to have same batch size")

    x = x.to(device=q.device, dtype=q.dtype)
    ax = math.tan(math.radians(oracle.horizontal_fov_deg) * 0.5)
    ay = math.tan(math.radians(oracle.vertical_fov_deg) * 0.5)
    nx = math.sqrt(1.0 + ax * ax)
    ny = math.sqrt(1.0 + ay * ay)

    sensor_margins = []
    for chain in oracle._chains(q.device, q.dtype):
        transform = oracle._fk_sensor_batch(chain, q)
        rotation = transform[:, :3, :3]
        translation = transform[:, :3, 3]
        difference_world = x - translation
        point_sensor = torch.einsum("qji,qj->qi", rotation, difference_world)
        px, py, pz = point_sensor[:, 0], point_sensor[:, 1], point_sensor[:, 2]
        planes = torch.stack(
            [
                (px + pz * ax) / nx,
                (-px + pz * ax) / nx,
                (py + pz * ay) / ny,
                (-py + pz * ay) / ny,
                pz - oracle.z_min,
                oracle.z_max - pz,
            ],
            dim=-1,
        )
        sensor_margins.append(torch.min(planes, dim=-1).values - oracle.delta)
    return torch.max(torch.stack(sensor_margins, dim=-1), dim=-1).values


def finite_difference_oracle_gradient_paired(
    x: torch.Tensor,
    q: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    grads = []
    for j in range(q.shape[1]):
        qp = q.clone()
        qm = q.clone()
        qp[:, j] += eps
        qm[:, j] -= eps
        qp = clamp_q(qp, q_min, q_max)
        qm = clamp_q(qm, q_min, q_max)
        gp = paired_oracle_visibility_g(x, qp, oracle)
        gm = paired_oracle_visibility_g(x, qm, oracle)
        denom = qp[:, j] - qm[:, j]
        gj = torch.where(
            torch.abs(denom) > 1e-12,
            (gp - gm) / denom,
            torch.zeros_like(gp),
        )
        grads.append(gj)
    return torch.stack(grads, dim=-1)


def clip_vector_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_norm) / torch.clamp(n, min=1e-12), max=1.0)
    return v * scale


# -----------------------------------------------------------------------------
# Numerical closest-boundary estimator
# -----------------------------------------------------------------------------


def make_multistarts(
    q0: torch.Tensor,
    g0: torch.Tensor,
    grad_g0: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    num_starts: int,
    jitter: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Create starts [B,S,D]. Start 0=q0; start 1 follows local boundary direction."""
    b, d = q0.shape
    starts = q0[:, None, :].repeat(1, num_starts, 1)
    if num_starts <= 1:
        return starts

    # Strong deterministic seed: move from q0 toward the local g=0 boundary.
    gn = torch.linalg.norm(grad_g0, dim=-1, keepdim=True)
    gdir = grad_g0 / torch.clamp(gn, min=1e-10)
    sign = torch.where(g0 >= 0.0, torch.ones_like(g0), -torch.ones_like(g0))
    toward_boundary = -sign[:, None] * gdir
    starts[:, 1, :] = q0 + float(jitter) * toward_boundary

    if num_starts > 2:
        noise = rng.normal(size=(b, num_starts - 2, d)).astype(np.float32)
        noise_norm = np.linalg.norm(noise, axis=-1, keepdims=True)
        noise = noise / np.maximum(noise_norm, 1e-12)
        radii = rng.uniform(0.25, 1.0, size=(b, num_starts - 2, 1)).astype(np.float32)
        noise = noise * radii * float(jitter)
        starts[:, 2:, :] = q0[:, None, :] + torch.as_tensor(
            noise, device=q0.device, dtype=q0.dtype
        )

    flat = starts.reshape(-1, d)
    flat = clamp_q(flat, q_min, q_max)
    return flat.reshape(b, num_starts, d)


def newton_project_candidates(
    x_rep: torch.Tensor,
    q: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    fd_eps: float,
    boundary_tol: float,
    iterations: int,
    damping: float,
    max_step: float,
) -> torch.Tensor:
    """Oracle Newton projection q <- q - g grad(g)/||grad(g)||^2."""
    q = q.clone()
    for _ in range(iterations):
        g = paired_oracle_visibility_g(x_rep, q, oracle)
        done = torch.abs(g) <= float(boundary_tol)
        if bool(torch.all(done)):
            break
        grad = finite_difference_oracle_gradient_paired(
            x_rep, q, oracle, q_min, q_max, fd_eps
        )
        grad2 = torch.sum(grad * grad, dim=-1, keepdim=True)
        step = -g[:, None] * grad / torch.clamp(grad2, min=1e-12)
        step = float(damping) * clip_vector_norm(step, max_step)
        q_new = clamp_q(q + step, q_min, q_max)
        q = torch.where(done[:, None], q, q_new)
    return q


def tangent_closest_point_refine(
    x_rep: torch.Tensor,
    q0_rep: torch.Tensor,
    qb: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    fd_eps: float,
    boundary_tol: float,
    iterations: int,
    tangent_relax: float,
    tangent_max_step: float,
    correction_steps: int,
    correction_damping: float,
    correction_max_step: float,
) -> torch.Tensor:
    """Reduce ||qb-q0|| along the boundary, then Newton-correct back to g=0."""
    qb = qb.clone()
    accept_tol = max(2.0 * float(boundary_tol), 1e-5)

    for _ in range(iterations):
        g = paired_oracle_visibility_g(x_rep, qb, oracle)
        grad = finite_difference_oracle_gradient_paired(
            x_rep, qb, oracle, q_min, q_max, fd_eps
        )
        grad2 = torch.sum(grad * grad, dim=-1, keepdim=True)
        displacement = qb - q0_rep
        normal_part = (
            torch.sum(displacement * grad, dim=-1, keepdim=True)
            / torch.clamp(grad2, min=1e-12)
        ) * grad
        tangent = displacement - normal_part
        tangent_step = -float(tangent_relax) * tangent
        tangent_step = clip_vector_norm(tangent_step, tangent_max_step)
        proposal = clamp_q(qb + tangent_step, q_min, q_max)

        # Bring proposal back to the g=0 manifold.
        for _corr in range(correction_steps):
            gp = paired_oracle_visibility_g(x_rep, proposal, oracle)
            gradp = finite_difference_oracle_gradient_paired(
                x_rep, proposal, oracle, q_min, q_max, fd_eps
            )
            gradp2 = torch.sum(gradp * gradp, dim=-1, keepdim=True)
            corr = -gp[:, None] * gradp / torch.clamp(gradp2, min=1e-12)
            corr = float(correction_damping) * clip_vector_norm(corr, correction_max_step)
            proposal = clamp_q(proposal + corr, q_min, q_max)

        old_dist = torch.linalg.norm(qb - q0_rep, dim=-1)
        new_dist = torch.linalg.norm(proposal - q0_rep, dim=-1)
        new_res = torch.abs(paired_oracle_visibility_g(x_rep, proposal, oracle))
        old_res = torch.abs(g)

        accept = (
            (new_dist < old_dist - 1e-7)
            & (new_res <= torch.maximum(torch.full_like(new_res, accept_tol), old_res))
        )
        qb = torch.where(accept[:, None], proposal, qb)

    # Final residual cleanup.
    qb = newton_project_candidates(
        x_rep=x_rep,
        q=qb,
        oracle=oracle,
        q_min=q_min,
        q_max=q_max,
        fd_eps=fd_eps,
        boundary_tol=boundary_tol,
        iterations=max(3, correction_steps),
        damping=1.0,
        max_step=correction_max_step,
    )
    return qb


def solve_closest_boundary_batch(
    x: torch.Tensor,
    q0: torch.Tensor,
    g0: torch.Tensor,
    grad_g0: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    args,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    b, d = q0.shape
    s = int(args.multistarts)
    starts = make_multistarts(
        q0, g0, grad_g0, q_min, q_max, s, args.init_jitter, rng
    )
    q0_rep = q0[:, None, :].expand(-1, s, -1).reshape(-1, d)
    x_rep = x[:, None, :].expand(-1, s, -1).reshape(-1, 3)
    q = starts.reshape(-1, d)

    q = newton_project_candidates(
        x_rep, q, oracle, q_min, q_max,
        args.fd_eps, args.boundary_tol, args.newton_iters,
        args.newton_damping, args.max_newton_step,
    )

    if args.tangent_iters > 0:
        q = tangent_closest_point_refine(
            x_rep, q0_rep, q, oracle, q_min, q_max,
            args.fd_eps, args.boundary_tol, args.tangent_iters,
            args.tangent_relax, args.tangent_max_step,
            args.correction_steps, args.correction_damping,
            args.max_newton_step,
        )

    residual = torch.abs(paired_oracle_visibility_g(x_rep, q, oracle)).reshape(b, s)
    distance = torch.linalg.norm(q - q0_rep, dim=-1).reshape(b, s)
    success = residual <= float(args.boundary_tol)

    # Choose minimum-distance successful candidate. If none succeeds, keep the
    # minimum-residual candidate but mark the sample invalid for CDF analysis.
    scored_dist = torch.where(success, distance, torch.full_like(distance, float("inf")))
    has_success = torch.any(success, dim=1)
    best_success_idx = torch.argmin(scored_dist, dim=1)
    best_res_idx = torch.argmin(residual, dim=1)
    best_idx = torch.where(has_success, best_success_idx, best_res_idx)

    q_reshaped = q.reshape(b, s, d)
    arange = torch.arange(b, device=q.device)
    qb = q_reshaped[arange, best_idx]
    best_residual = residual[arange, best_idx]
    best_distance = distance[arange, best_idx]

    # Signed-distance gradient from the closest-point geometry.
    sign = torch.where(g0 >= 0.0, torch.ones_like(g0), -torch.ones_like(g0))
    cdf_dir = sign[:, None] * (q0 - qb) / torch.clamp(best_distance[:, None], min=1e-12)

    grad_g_boundary = finite_difference_oracle_gradient_paired(
        x, qb, oracle, q_min, q_max, args.fd_eps
    )
    gbn = torch.linalg.norm(grad_g_boundary, dim=-1)
    cdfn = torch.linalg.norm(cdf_dir, dim=-1)
    normal_cos = torch.sum(cdf_dir * grad_g_boundary, dim=-1) / torch.clamp(cdfn * gbn, min=1e-12)
    normal_cos = torch.clamp(normal_cos, -1.0, 1.0)

    # KKT tangential residual: nearest-boundary displacement should be normal to
    # the boundary. 0 is ideal.
    disp = qb - q0
    grad2 = torch.sum(grad_g_boundary * grad_g_boundary, dim=-1, keepdim=True)
    normal_part = (
        torch.sum(disp * grad_g_boundary, dim=-1, keepdim=True)
        / torch.clamp(grad2, min=1e-12)
    ) * grad_g_boundary
    tangent_part = disp - normal_part
    tangent_ratio = torch.linalg.norm(tangent_part, dim=-1) / torch.clamp(
        torch.linalg.norm(disp, dim=-1), min=1e-12
    )

    return {
        "qb": qb,
        "residual": best_residual,
        "distance": best_distance,
        "success": has_success,
        "cdf_dir": cdf_dir,
        "grad_g_boundary": grad_g_boundary,
        "boundary_normal_cos": normal_cos,
        "kkt_tangent_ratio": tangent_ratio,
    }


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    if not os.path.exists(args.inside_npz):
        raise FileNotFoundError(
            f"Missing inside-gradient NPZ: {args.inside_npz}\n"
            "Run the already-completed inside-gradient diagnostic first."
        )

    src = np.load(args.inside_npz, allow_pickle=False)
    required = ["x", "q", "bin_id", "bin_names", "f", "g", "grad_f", "grad_g"]
    missing = [k for k in required if k not in src.files]
    if missing:
        raise RuntimeError(f"inside-gradient NPZ missing required arrays: {missing}")

    bin_names = [str(v) for v in src["bin_names"].tolist()]
    unknown = [b for b in args.bins if b not in bin_names]
    if unknown:
        raise ValueError(f"Unknown bins {unknown}; available bins are {bin_names}")

    data = load_npz_data(args.data)
    q_min = torch.as_tensor(data["q_min"], device=device, dtype=torch.float32)
    q_max = torch.as_tensor(data["q_max"], device=device, dtype=torch.float32)

    oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        z_min=args.z_min,
        z_max=args.z_max,
        delta=args.delta,
    )

    selection_rng = np.random.default_rng(args.selection_seed)
    projection_rng = np.random.default_rng(args.projection_seed)

    selected = []
    bin_id_src = src["bin_id"].astype(np.int64)
    for name in args.bins:
        bid = bin_names.index(name)
        idx = np.flatnonzero(bin_id_src == bid)
        if len(idx) < args.samples_per_bin:
            raise RuntimeError(
                f"Bin {name} has only {len(idx)} samples; requested {args.samples_per_bin}."
            )
        chosen = selection_rng.choice(idx, size=args.samples_per_bin, replace=False)
        selected.append((name, bid, np.sort(chosen)))

    all_idx = np.concatenate([x[2] for x in selected])
    selected_bin_name = np.concatenate([
        np.full((args.samples_per_bin,), name, dtype=object) for name, _, _ in selected
    ])

    x_np = src["x"][all_idx].astype(np.float32)
    q_np = src["q"][all_idx].astype(np.float32)
    f_np = src["f"][all_idx].astype(np.float32)
    g_np = src["g"][all_idx].astype(np.float32)
    grad_f_np = src["grad_f"][all_idx].astype(np.float32)
    grad_g_np = src["grad_g"][all_idx].astype(np.float32)

    n = len(all_idx)
    qb_all = np.full_like(q_np, np.nan, dtype=np.float32)
    dist_all = np.full((n,), np.nan, dtype=np.float32)
    residual_all = np.full((n,), np.nan, dtype=np.float32)
    success_all = np.zeros((n,), dtype=bool)
    cdf_dir_all = np.full_like(q_np, np.nan, dtype=np.float32)
    grad_g_boundary_all = np.full_like(q_np, np.nan, dtype=np.float32)
    boundary_normal_cos_all = np.full((n,), np.nan, dtype=np.float32)
    kkt_tangent_ratio_all = np.full((n,), np.nan, dtype=np.float32)

    print("\n=== Oracle configuration-space CDF diagnostic ===")
    print(f"inside NPZ:             {args.inside_npz}")
    print(f"bins:                   {args.bins}")
    print(f"samples/bin:            {args.samples_per_bin}")
    print(f"total samples:          {n}")
    print(f"multistarts:            {args.multistarts}")
    print(f"init jitter:            {args.init_jitter}")
    print(f"Newton iters:           {args.newton_iters}")
    print(f"tangent refine iters:   {args.tangent_iters}")
    print(f"boundary tolerance:     {args.boundary_tol}")
    print(f"diagnostic step:        {args.diagnostic_step}")
    print("====================================================\n")

    start_time = time.time()
    batches = math.ceil(n / args.batch_size)
    for bi, start in enumerate(range(0, n, args.batch_size), start=1):
        end = min(start + args.batch_size, n)
        x_b = torch.as_tensor(x_np[start:end], device=device)
        q_b = torch.as_tensor(q_np[start:end], device=device)
        g_b = torch.as_tensor(g_np[start:end], device=device)
        gg_b = torch.as_tensor(grad_g_np[start:end], device=device)

        solved = solve_closest_boundary_batch(
            x=x_b,
            q0=q_b,
            g0=g_b,
            grad_g0=gg_b,
            oracle=oracle,
            q_min=q_min,
            q_max=q_max,
            args=args,
            rng=projection_rng,
        )

        qb_all[start:end] = solved["qb"].cpu().numpy()
        dist_all[start:end] = solved["distance"].cpu().numpy()
        residual_all[start:end] = solved["residual"].cpu().numpy()
        success_all[start:end] = solved["success"].cpu().numpy()
        cdf_dir_all[start:end] = solved["cdf_dir"].cpu().numpy()
        grad_g_boundary_all[start:end] = solved["grad_g_boundary"].cpu().numpy()
        boundary_normal_cos_all[start:end] = solved["boundary_normal_cos"].cpu().numpy()
        kkt_tangent_ratio_all[start:end] = solved["kkt_tangent_ratio"].cpu().numpy()

        if bi == 1 or bi % args.print_every == 0 or end == n:
            cur_success = float(np.mean(success_all[:end]))
            print(
                f"boundary batch {bi:04d}/{batches}: samples={end}/{n} "
                f"success={cur_success:.4f} elapsed={time.time()-start_time:.1f}s",
                flush=True,
            )

    valid = success_all & np.isfinite(dist_all) & (dist_all > 1e-8)
    signed_distance = np.where(g_np >= 0.0, dist_all, -dist_all).astype(np.float32)

    cos_f_cdf, v_fc = cosine_np(grad_f_np, cdf_dir_all)
    cos_g_cdf, v_gc = cosine_np(grad_g_np, cdf_dir_all)
    cos_f_g, v_fg = cosine_np(grad_f_np, grad_g_np)
    valid_fc = valid & v_fc
    valid_gc = valid & v_gc
    valid_fg = valid & v_fg

    # One small step along the estimated oracle CDF gradient.  This directly
    # tests whether maximizing true configuration-space distance also maximizes
    # the geometric FOV margin away from the boundary.
    cdf_delta_g = np.full((n,), np.nan, dtype=np.float32)
    ggrad_delta_g = np.full((n,), np.nan, dtype=np.float32)
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        x_b = torch.as_tensor(x_np[start:end], device=device)
        q_b = torch.as_tensor(q_np[start:end], device=device)
        g0_b = torch.as_tensor(g_np[start:end], device=device)
        cdf_b = torch.as_tensor(cdf_dir_all[start:end], device=device)
        gg_b = torch.as_tensor(grad_g_np[start:end], device=device)

        q_cdf = clamp_q(q_b + float(args.diagnostic_step) * cdf_b, q_min, q_max)
        gg_norm = torch.linalg.norm(gg_b, dim=-1, keepdim=True)
        gg_dir = gg_b / torch.clamp(gg_norm, min=1e-10)
        q_g = clamp_q(q_b + float(args.diagnostic_step) * gg_dir, q_min, q_max)
        gcdf = paired_oracle_visibility_g(x_b, q_cdf, oracle)
        ggnew = paired_oracle_visibility_g(x_b, q_g, oracle)
        cdf_delta_g[start:end] = (gcdf - g0_b).cpu().numpy()
        ggrad_delta_g[start:end] = (ggnew - g0_b).cpu().numpy()

    result: Dict = {
        "config": {
            "inside_npz": args.inside_npz,
            "data": args.data,
            "urdf": args.urdf,
            "device": str(device),
            "bins": list(args.bins),
            "samples_per_bin": args.samples_per_bin,
            "total_samples": n,
            "selection_seed": args.selection_seed,
            "projection_seed": args.projection_seed,
            "multistarts": args.multistarts,
            "init_jitter": args.init_jitter,
            "fd_eps": args.fd_eps,
            "boundary_tol": args.boundary_tol,
            "newton_iters": args.newton_iters,
            "newton_damping": args.newton_damping,
            "max_newton_step": args.max_newton_step,
            "tangent_iters": args.tangent_iters,
            "tangent_relax": args.tangent_relax,
            "tangent_max_step": args.tangent_max_step,
            "correction_steps": args.correction_steps,
            "diagnostic_step": args.diagnostic_step,
            "meaning": (
                "d_C_hat is a numerical signed Euclidean distance to the oracle "
                "g=0 visibility boundary under joint limits"
            ),
        },
        "search_quality": {},
        "bins": {},
    }

    result["search_quality"] = {
        "overall_boundary_success_rate": float(np.mean(success_all)),
        "boundary_residual_stats_success": stats(residual_all, success_all),
        "boundary_normal_cosine_stats_success": stats(boundary_normal_cos_all, valid),
        "kkt_tangent_ratio_stats_success": stats(kkt_tangent_ratio_all, valid),
    }

    print("\n=== Oracle-CDF gradient comparison ===")
    print(
        "bin                 N   bdry_ok  d_med   resid95  KKTcos_med  "
        "cos(f,CDF)  cos(g,CDF)  cos(f,g)  CDFstep dg>0"
    )

    for name in args.bins:
        m = selected_bin_name == name
        mv = m & valid
        mfc = m & valid_fc
        mgc = m & valid_gc
        mfg = m & valid_fg

        # KKT normal cosine should be positive; abs is useful at nonsmooth points,
        # but signed value is also stored.
        normal_abs = np.abs(boundary_normal_cos_all)
        entry = {
            "count": int(np.sum(m)),
            "g_stats": stats(g_np, m),
            "f_stats": stats(f_np, m),
            "oracle_cdf": {
                "boundary_success_rate": float(np.mean(success_all[m])),
                "signed_distance_stats": stats(signed_distance, mv),
                "distance_abs_stats": stats(dist_all, mv),
                "boundary_residual_stats": stats(residual_all, m & success_all),
                "boundary_normal_cosine_stats": stats(boundary_normal_cos_all, mv),
                "boundary_normal_abs_cosine_stats": stats(normal_abs, mv),
                "kkt_tangent_ratio_stats": stats(kkt_tangent_ratio_all, mv),
            },
            "gradient_comparison": {
                "cos_learned_vs_oracle_cdf": stats(cos_f_cdf, mfc),
                "cos_gmargin_vs_oracle_cdf": stats(cos_g_cdf, mgc),
                "cos_learned_vs_gmargin": stats(cos_f_g, mfg),
                "learned_vs_cdf_positive_rate": float(np.mean(cos_f_cdf[mfc] > 0.0)) if np.any(mfc) else float("nan"),
                "learned_vs_cdf_ge_0p8_rate": float(np.mean(cos_f_cdf[mfc] >= 0.8)) if np.any(mfc) else float("nan"),
                "gmargin_vs_cdf_positive_rate": float(np.mean(cos_g_cdf[mgc] > 0.0)) if np.any(mgc) else float("nan"),
            },
            "value_comparison": {
                "pearson_f_vs_signed_oracle_cdf": pearson(f_np[mv], signed_distance[mv]),
            },
            "one_step": {
                "step_size": float(args.diagnostic_step),
                "oracle_cdf_direction_delta_g_stats": stats(cdf_delta_g, mv),
                "oracle_cdf_direction_delta_g_positive_rate": float(np.mean(cdf_delta_g[mv] > 0.0)) if np.any(mv) else float("nan"),
                "oracle_g_gradient_delta_g_stats": stats(ggrad_delta_g, mv),
                "oracle_g_gradient_delta_g_positive_rate": float(np.mean(ggrad_delta_g[mv] > 0.0)) if np.any(mv) else float("nan"),
            },
        }
        result["bins"][name] = entry

        resid95 = entry["oracle_cdf"]["boundary_residual_stats"]["p95"]
        kkt_med = entry["oracle_cdf"]["boundary_normal_abs_cosine_stats"]["p50"]
        c_fc = entry["gradient_comparison"]["cos_learned_vs_oracle_cdf"]["p50"]
        c_gc = entry["gradient_comparison"]["cos_gmargin_vs_oracle_cdf"]["p50"]
        c_fg = entry["gradient_comparison"]["cos_learned_vs_gmargin"]["p50"]
        cdf_up = entry["one_step"]["oracle_cdf_direction_delta_g_positive_rate"]
        print(
            f"{name:19s} {entry['count']:4d}  "
            f"{entry['oracle_cdf']['boundary_success_rate']:.4f}  "
            f"{entry['oracle_cdf']['distance_abs_stats']['p50']:.4f}  "
            f"{resid95:.2e}    {kkt_med:.4f}      "
            f"{c_fc:+.4f}      {c_gc:+.4f}      {c_fg:+.4f}      {cdf_up:.4f}"
        )

    # Compact decision block.  It does not overclaim; it only classifies the
    # pattern if search quality is adequate.
    quality_ok = (
        result["search_quality"]["overall_boundary_success_rate"] >= args.min_success_rate
        and result["search_quality"]["boundary_residual_stats_success"]["p95"] <= 2.0 * args.boundary_tol
    )
    decision = {"search_quality_ok": bool(quality_ok), "classification": "inconclusive"}
    if "deep_inside" in result["bins"] and quality_ok:
        deep = result["bins"]["deep_inside"]["gradient_comparison"]
        fc = deep["cos_learned_vs_oracle_cdf"]["p50"]
        gc = deep["cos_gmargin_vs_oracle_cdf"]["p50"]
        if np.isfinite(fc) and np.isfinite(gc):
            if fc >= args.cdf_match_threshold and gc < args.objective_mismatch_threshold:
                decision["classification"] = "objective_mismatch_supported"
            elif fc < args.cdf_match_threshold:
                decision["classification"] = "learned_deep_cdf_geometry_mismatch"
            else:
                decision["classification"] = "mixed"
            decision["deep_median_cos_learned_vs_cdf"] = float(fc)
            decision["deep_median_cos_gmargin_vs_cdf"] = float(gc)
    result["decision"] = decision
    result["elapsed_sec"] = float(time.time() - start_time)

    print("\n=== Decision diagnostic ===")
    print(json.dumps(decision, indent=2))
    if not quality_ok:
        print(
            "[WARN] Numerical closest-boundary quality is insufficient. "
            "Do not interpret CDF-gradient comparisons yet."
        )

    outdir = os.path.dirname(args.output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(f"\nsaved JSON: {args.output}")

    if args.output_npz:
        npzdir = os.path.dirname(args.output_npz)
        if npzdir:
            os.makedirs(npzdir, exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            source_index=all_idx.astype(np.int64),
            bin_name=selected_bin_name.astype(str),
            x=x_np,
            q=q_np,
            f=f_np,
            g=g_np,
            grad_f=grad_f_np,
            grad_g=grad_g_np,
            q_boundary=qb_all,
            oracle_cdf_signed_distance=signed_distance,
            oracle_cdf_direction=cdf_dir_all,
            boundary_residual=residual_all,
            boundary_success=success_all,
            grad_g_boundary=grad_g_boundary_all,
            boundary_normal_cosine=boundary_normal_cos_all,
            kkt_tangent_ratio=kkt_tangent_ratio_all,
            cos_learned_vs_oracle_cdf=cos_f_cdf.astype(np.float32),
            cos_gmargin_vs_oracle_cdf=cos_g_cdf.astype(np.float32),
            cos_learned_vs_gmargin=cos_f_g.astype(np.float32),
            cdf_direction_delta_g=cdf_delta_g,
            ggradient_delta_g=ggrad_delta_g,
        )
        print(f"saved NPZ:  {args.output_npz}")

    print(f"elapsed_sec: {result['elapsed_sec']:.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="Oracle configuration-space CDF diagnostic")
    p.add_argument(
        "--inside-npz",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "inside_gradient_diagnostic/inside_gradient_diagnostic_samples.npz"
        ),
    )
    p.add_argument(
        "--data",
        default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz",
    )
    p.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    p.add_argument(
        "--bins",
        type=parse_str_list,
        default=parse_str_list("outside_boundary,shallow_inside,deep_inside"),
        help="Deliberately keep this diagnostic small; default uses only 3 decisive bins.",
    )
    p.add_argument("--samples-per-bin", type=int, default=1000)
    p.add_argument("--selection-seed", type=int, default=20260808)
    p.add_argument("--projection-seed", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=128)

    # Closest-boundary search.
    p.add_argument("--multistarts", type=int, default=6)
    p.add_argument("--init-jitter", type=float, default=0.10)
    p.add_argument("--fd-eps", type=float, default=1e-4)
    p.add_argument("--boundary-tol", type=float, default=1e-3)
    p.add_argument("--newton-iters", type=int, default=15)
    p.add_argument("--newton-damping", type=float, default=0.8)
    p.add_argument("--max-newton-step", type=float, default=0.25)
    p.add_argument("--tangent-iters", type=int, default=5)
    p.add_argument("--tangent-relax", type=float, default=0.5)
    p.add_argument("--tangent-max-step", type=float, default=0.10)
    p.add_argument("--correction-steps", type=int, default=2)
    p.add_argument("--correction-damping", type=float, default=1.0)
    p.add_argument("--diagnostic-step", type=float, default=0.01)

    # FOV parameters: identical to current standalone oracle experiment.
    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)

    # Diagnostic thresholds only; no parameter sweep is performed.
    p.add_argument("--min-success-rate", type=float, default=0.95)
    p.add_argument("--cdf-match-threshold", type=float, default=0.80)
    p.add_argument("--objective-mismatch-threshold", type=float, default=0.70)

    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument(
        "--output",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "oracle_cspace_cdf_diagnostic/oracle_cspace_cdf_diagnostic.json"
        ),
    )
    p.add_argument(
        "--output-npz",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "oracle_cspace_cdf_diagnostic/oracle_cspace_cdf_diagnostic_samples.npz"
        ),
    )
    args = p.parse_args()

    if args.samples_per_bin <= 0 or args.batch_size <= 0:
        p.error("--samples-per-bin and --batch-size must be positive")
    if args.multistarts <= 0:
        p.error("--multistarts must be positive")
    if args.fd_eps <= 0 or args.boundary_tol <= 0:
        p.error("--fd-eps and --boundary-tol must be positive")
    if args.newton_iters <= 0 or args.tangent_iters < 0:
        p.error("invalid projection iteration counts")
    if args.max_newton_step <= 0 or args.tangent_max_step <= 0:
        p.error("projection step limits must be positive")
    if args.diagnostic_step <= 0:
        p.error("--diagnostic-step must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
