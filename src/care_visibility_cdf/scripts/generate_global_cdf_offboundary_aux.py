#!/usr/bin/env python3
"""
Generate high-quality off-boundary global configuration-space CDF supervision
for CAREPlanner visibility-CDF training.

This script is intentionally narrow: it only generates *inside* samples with
true oracle FOV margin g in [g_min, g_max] (default [0.03, 0.12]).  For each
sample (x,q), it numerically solves the closest point q_b on the *global union*
visibility boundary g(x,q_b)=0 under joint limits, then stores

    target_d    = ||q - q_b||_2
    target_grad = (q - q_b) / ||q - q_b||_2

Only high-quality closest-boundary solutions are kept.

The x-pool reproduces the train/validation split used by
train_signed_visibility_cdf_pairwise_replace.py, so auxiliary labels are made
from training x points only and do not leak validation x points.

Dependencies
------------
This script reuses the already-validated standalone FOV oracle and numerical
closest-boundary solver from:
  - evaluate_direct_vs_projection_ascent.py
  - evaluate_oracle_cspace_cdf_diagnostic.py
Place this file in src/care_visibility_cdf/scripts/ beside those evaluators.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
)
from evaluate_oracle_cspace_cdf_diagnostic import (
    finite_difference_oracle_gradient_paired,
    paired_oracle_visibility_g,
    solve_closest_boundary_batch,
)


def clamp_q(q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(q, q_max[None, :]), q_min[None, :])


def np_stats(a: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0}
    q = np.quantile(a, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "max": float(np.max(a)),
    }


def reproduce_train_x_pool(
    dataset_path: str,
    val_count: int,
    split_seed: int,
):
    d = np.load(dataset_path, allow_pickle=True)
    for key in ["x", "valid_fov"]:
        if key not in d.files:
            raise RuntimeError(f"Dataset missing required key: {key}")

    x = d["x"].astype(np.float32)
    valid_fov = d["valid_fov"].astype(np.bool_)

    if "q_min" in d.files and "q_max" in d.files:
        q_min = d["q_min"].astype(np.float32)
        q_max = d["q_max"].astype(np.float32)
    else:
        q_min = np.full((7,), -math.pi, dtype=np.float32)
        q_max = np.full((7,), math.pi, dtype=np.float32)
        print("[WARN] q_min/q_max not found; using [-pi,pi].")

    valid_point = np.any(valid_fov, axis=(1, 2))
    valid_indices = np.where(valid_point)[0]
    if len(valid_indices) == 0:
        raise RuntimeError("No valid FOV x points in dataset.")

    rng = np.random.default_rng(split_seed)
    rng.shuffle(valid_indices)
    val_count_eff = min(val_count, max(1, len(valid_indices) // 10))
    val_indices = valid_indices[:val_count_eff].copy()
    train_indices = valid_indices[val_count_eff:].copy()
    if len(train_indices) == 0:
        raise RuntimeError("No training x points after validation split.")

    return x, train_indices, val_indices, q_min, q_max


def make_solver_args(args):
    """Namespace expected by solve_closest_boundary_batch()."""
    return SimpleNamespace(
        multistarts=args.multistarts,
        init_jitter=args.init_jitter,
        fd_eps=args.fd_eps,
        boundary_tol=args.boundary_tol,
        newton_iters=args.newton_iters,
        newton_damping=args.newton_damping,
        max_newton_step=args.max_newton_step,
        tangent_iters=args.tangent_iters,
        tangent_relax=args.tangent_relax,
        tangent_max_step=args.tangent_max_step,
        correction_steps=args.correction_steps,
        correction_damping=args.correction_damping,
    )


def generate(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    if not (0.0 < args.g_min < args.g_max):
        raise ValueError("Require 0 < g_min < g_max for inside auxiliary data.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.proposal_batch <= 0 or args.solve_batch <= 0:
        raise ValueError("batch sizes must be positive")

    x_all, train_x_idx, val_x_idx, q_min_np, q_max_np = reproduce_train_x_pool(
        args.data, args.val_count, args.split_seed
    )
    q_min = torch.as_tensor(q_min_np, device=device, dtype=torch.float32)
    q_max = torch.as_tensor(q_max_np, device=device, dtype=torch.float32)

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

    proposal_rng = np.random.default_rng(args.proposal_seed)
    projection_rng = np.random.default_rng(args.projection_seed)
    solver_args = make_solver_args(args)

    kept = {
        "x": [],
        "q": [],
        "target_d": [],
        "target_grad": [],
        "g": [],
        "q_boundary": [],
        "boundary_residual": [],
        "boundary_normal_cosine": [],
        "kkt_tangent_ratio": [],
        "x_index": [],
    }

    total_proposed = 0
    total_in_range = 0
    total_solved = 0
    total_boundary_success = 0
    total_quality_kept_before_trunc = 0
    start_time = time.time()
    iteration = 0

    print("\n=== Global-CDF off-boundary auxiliary generation ===")
    print(f"data:                    {args.data}")
    print(f"urdf:                    {args.urdf}")
    print(f"device:                  {device}")
    print(f"target samples:          {args.num_samples}")
    print(f"g range:                 [{args.g_min}, {args.g_max}]")
    print(f"train x pool:            {len(train_x_idx)}")
    print(f"held-out val x pool:     {len(val_x_idx)}")
    print(f"proposal batch:          {args.proposal_batch}")
    print(f"solve batch:             {args.solve_batch}")
    print(f"multistarts:             {args.multistarts}")
    print(f"boundary tol:            {args.boundary_tol}")
    print(f"quality kkt <=:          {args.max_kkt_tangent_ratio}")
    print(f"quality |normal cos| >=: {args.min_boundary_normal_abs_cos}")
    print("====================================================\n")

    while sum(len(v) for v in kept["target_d"]) < args.num_samples:
        iteration += 1

        # Candidate x are sampled only from the original training-x split.
        x_idx_np = proposal_rng.choice(
            train_x_idx, size=args.proposal_batch, replace=True
        ).astype(np.int64)
        x_np = x_all[x_idx_np]

        u = proposal_rng.random((args.proposal_batch, 7), dtype=np.float32)
        q_np = q_min_np[None, :] + u * (q_max_np - q_min_np)[None, :]

        x_t = torch.as_tensor(x_np, device=device, dtype=torch.float32)
        q_t = torch.as_tensor(q_np, device=device, dtype=torch.float32)
        with torch.no_grad():
            g_t = paired_oracle_visibility_g(x_t, q_t, oracle)

        g_np = g_t.cpu().numpy()
        in_range = (g_np >= args.g_min) & (g_np <= args.g_max)
        idx = np.flatnonzero(in_range)

        total_proposed += args.proposal_batch
        total_in_range += len(idx)
        if len(idx) == 0:
            if iteration % args.print_every == 0:
                print(
                    f"proposal {iteration:05d}: kept={sum(len(v) for v in kept['target_d'])}/{args.num_samples} "
                    f"in_range=0/{args.proposal_batch} elapsed={time.time()-start_time:.1f}s",
                    flush=True,
                )
            continue

        # Shuffle accepted proposals so repeated truncation does not bias by order.
        proposal_rng.shuffle(idx)
        x_sel_np = x_np[idx]
        q_sel_np = q_np[idx]
        g_sel_np = g_np[idx]
        x_idx_sel_np = x_idx_np[idx]

        for s0 in range(0, len(idx), args.solve_batch):
            s1 = min(s0 + args.solve_batch, len(idx))
            xb = torch.as_tensor(x_sel_np[s0:s1], device=device, dtype=torch.float32)
            qb = torch.as_tensor(q_sel_np[s0:s1], device=device, dtype=torch.float32)
            gb = torch.as_tensor(g_sel_np[s0:s1], device=device, dtype=torch.float32)

            grad_g0 = finite_difference_oracle_gradient_paired(
                xb, qb, oracle, q_min, q_max, args.fd_eps
            )
            solved = solve_closest_boundary_batch(
                x=xb,
                q0=qb,
                g0=gb,
                grad_g0=grad_g0,
                oracle=oracle,
                q_min=q_min,
                q_max=q_max,
                args=solver_args,
                rng=projection_rng,
            )

            success = solved["success"]
            residual = solved["residual"]
            distance = solved["distance"]
            normal_cos = solved["boundary_normal_cos"]
            kkt = solved["kkt_tangent_ratio"]
            cdf_dir = solved["cdf_dir"]

            finite = (
                torch.isfinite(distance)
                & torch.isfinite(residual)
                & torch.isfinite(normal_cos)
                & torch.isfinite(kkt)
                & torch.isfinite(cdf_dir).all(dim=-1)
            )
            quality = (
                success
                & finite
                & (distance > args.min_distance)
                & (residual <= args.boundary_tol)
                & (torch.abs(normal_cos) >= args.min_boundary_normal_abs_cos)
                & (kkt <= args.max_kkt_tangent_ratio)
            )

            total_solved += int(s1 - s0)
            total_boundary_success += int(success.sum().item())
            total_quality_kept_before_trunc += int(quality.sum().item())

            if not bool(torch.any(quality)):
                continue

            qm = quality.cpu().numpy()
            kept["x"].append(x_sel_np[s0:s1][qm].astype(np.float32))
            kept["q"].append(q_sel_np[s0:s1][qm].astype(np.float32))
            kept["target_d"].append(distance[quality].cpu().numpy().astype(np.float32))
            kept["target_grad"].append(cdf_dir[quality].cpu().numpy().astype(np.float32))
            kept["g"].append(g_sel_np[s0:s1][qm].astype(np.float32))
            kept["q_boundary"].append(solved["qb"][quality].cpu().numpy().astype(np.float32))
            kept["boundary_residual"].append(residual[quality].cpu().numpy().astype(np.float32))
            kept["boundary_normal_cosine"].append(normal_cos[quality].cpu().numpy().astype(np.float32))
            kept["kkt_tangent_ratio"].append(kkt[quality].cpu().numpy().astype(np.float32))
            kept["x_index"].append(x_idx_sel_np[s0:s1][qm].astype(np.int64))

            current = sum(len(v) for v in kept["target_d"])
            if current >= args.num_samples:
                break

        current = sum(len(v) for v in kept["target_d"])
        if iteration == 1 or iteration % args.print_every == 0 or current >= args.num_samples:
            in_rate = total_in_range / max(1, total_proposed)
            quality_rate = total_quality_kept_before_trunc / max(1, total_solved)
            print(
                f"proposal {iteration:05d}: kept={current}/{args.num_samples} "
                f"g-range-rate={in_rate:.4f} solver-quality-rate={quality_rate:.4f} "
                f"elapsed={time.time()-start_time:.1f}s",
                flush=True,
            )

    arrays = {}
    for key, chunks in kept.items():
        arrays[key] = np.concatenate(chunks, axis=0)[: args.num_samples]

    # Final deterministic shuffle; keeps all arrays aligned.
    order_rng = np.random.default_rng(args.output_shuffle_seed)
    order = order_rng.permutation(args.num_samples)
    for key in arrays:
        arrays[key] = arrays[key][order]

    grad_norm = np.linalg.norm(arrays["target_grad"], axis=1)
    summary = {
        "config": vars(args),
        "split": {
            "valid_train_x_count": int(len(train_x_idx)),
            "heldout_val_x_count": int(len(val_x_idx)),
            "split_seed": int(args.split_seed),
            "requested_val_count": int(args.val_count),
        },
        "generation": {
            "num_samples": int(args.num_samples),
            "total_proposed": int(total_proposed),
            "total_in_requested_g_range": int(total_in_range),
            "total_closest_boundary_solved": int(total_solved),
            "total_boundary_success": int(total_boundary_success),
            "total_quality_kept_before_truncation": int(total_quality_kept_before_trunc),
            "g_range_acceptance_rate": float(total_in_range / max(1, total_proposed)),
            "boundary_success_rate_among_solved": float(total_boundary_success / max(1, total_solved)),
            "quality_rate_among_solved": float(total_quality_kept_before_trunc / max(1, total_solved)),
            "elapsed_sec": float(time.time() - start_time),
        },
        "saved_stats": {
            "g": np_stats(arrays["g"]),
            "target_d": np_stats(arrays["target_d"]),
            "target_grad_norm": np_stats(grad_norm),
            "boundary_residual": np_stats(arrays["boundary_residual"]),
            "boundary_normal_abs_cosine": np_stats(np.abs(arrays["boundary_normal_cosine"])),
            "kkt_tangent_ratio": np_stats(arrays["kkt_tangent_ratio"]),
        },
        "meaning": (
            "High-quality inside global-union configuration-space CDF labels: "
            "target_d=distance to nearest oracle g=0 boundary; target_grad points "
            "toward increasing signed distance. All x are from the original training split."
        ),
    }

    outdir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        args.output,
        x=arrays["x"],
        q=arrays["q"],
        target_d=arrays["target_d"],
        target_grad=arrays["target_grad"],
        g=arrays["g"],
        q_boundary=arrays["q_boundary"],
        boundary_residual=arrays["boundary_residual"],
        boundary_normal_cosine=arrays["boundary_normal_cosine"],
        kkt_tangent_ratio=arrays["kkt_tangent_ratio"],
        x_index=arrays["x_index"],
        q_min=q_min_np,
        q_max=q_max_np,
        g_min=np.float32(args.g_min),
        g_max=np.float32(args.g_max),
        split_seed=np.int64(args.split_seed),
        val_count=np.int64(args.val_count),
    )

    summary_path = args.summary_json
    if not summary_path:
        stem, _ = os.path.splitext(args.output)
        summary_path = stem + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\n=== Saved auxiliary dataset ===")
    print(f"NPZ:     {args.output}")
    print(f"summary: {summary_path}")
    print(f"shape x/q/grad: {arrays['x'].shape} / {arrays['q'].shape} / {arrays['target_grad'].shape}")
    print(f"g mean/median:  {summary['saved_stats']['g']['mean']:+.5f} / {summary['saved_stats']['g']['p50']:+.5f}")
    print(f"d mean/median:  {summary['saved_stats']['target_d']['mean']:.5f} / {summary['saved_stats']['target_d']['p50']:.5f}")
    print(f"residual p95:   {summary['saved_stats']['boundary_residual']['p95']:.3e}")
    print(f"KKT ratio p95:  {summary['saved_stats']['kkt_tangent_ratio']['p95']:.5f}")
    print(f"elapsed_sec:    {summary['generation']['elapsed_sec']:.1f}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate high-quality deep/interior global-CDF auxiliary labels."
    )
    p.add_argument(
        "--data",
        default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz",
    )
    p.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    p.add_argument(
        "--output",
        default="src/care_visibility_cdf/data/visibility_offboundary_global_cdf_aux.npz",
    )
    p.add_argument("--summary-json", default="")

    # Match original train/val x split exactly.
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument("--split-seed", type=int, default=0)

    # Deliberately narrow interior range. Do not chase the medial-axis / maximum-margin region.
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--g-min", type=float, default=0.03)
    p.add_argument("--g-max", type=float, default=0.12)
    p.add_argument("--proposal-batch", type=int, default=4096)
    p.add_argument("--solve-batch", type=int, default=128)
    p.add_argument("--proposal-seed", type=int, default=20260808)
    p.add_argument("--projection-seed", type=int, default=17)
    p.add_argument("--output-shuffle-seed", type=int, default=314159)

    # Closest-boundary solver: same defaults as the decisive oracle-CDF diagnostic.
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

    # Keep only trustworthy nearest-boundary labels.
    p.add_argument("--min-distance", type=float, default=1e-4)
    p.add_argument("--max-kkt-tangent-ratio", type=float, default=0.05)
    p.add_argument("--min-boundary-normal-abs-cos", type=float, default=0.95)

    # Same FOV oracle as Exp1 evaluation.
    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)

    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--print-every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
