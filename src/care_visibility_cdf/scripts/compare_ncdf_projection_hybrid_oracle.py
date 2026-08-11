#!/usr/bin/env python3
"""Equal-budget comparison of three frozen-NCDF local guidance rules.

Methods
-------
A) direct
   Fixed normalized gradient ascent, matching the current online candidate.

B) projection
   Projection-like update toward the learned zero level set:

       dq = -damping * f / (||grad f||^2 + eps) * grad f

   The step is clipped by a per-iteration cap and the common total motion
   budget.  Backtracking requires learned f not to decrease.

C) hybrid
   If learned f < -tau_f, use the projection-like update above.  Once the
   learned field reaches the near-boundary band f >= -tau_f, switch to the
   same normalized direct ascent used by method A so that motion can cross the
   learned boundary rather than merely stopping on it.

Fairness
--------
All three methods start from the exact same (x, q0), use the same frozen
checkpoint, joint limits, trust-region box, and a common cumulative L2 path
budget.  The analytic oracle g is NEVER used to select an update or switch
modes.  It is used only after optimization to evaluate true FOV progress.

Evaluation cohorts are defined by the *initial true oracle margin* g0:

    near   : -0.03 < g0 < 0
    middle : -0.10 < g0 <= -0.03
    far    : g0 <= -0.10

This script is offline only and publishes no ROS commands.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    torch_load_checkpoint,
)
from export_ncdf_l4casadi import build_casadi_functions  # noqa: E402
from test_ncdf_local_optimizer import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF,
    read_joint_limits,
)
from test_ncdf_projected_ascent import eval_value_grad  # noqa: E402
from test_ncdf_projected_ascent_oracle import oracle_g  # noqa: E402


DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf_method_compare"
METHODS = ("direct", "projection", "hybrid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Equal-budget direct vs projection vs hybrid NCDF oracle comparison."
    )
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--urdf", default=str(DEFAULT_URDF))
    p.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--model-name", default="care_visibility_ncdf_method_compare")

    p.add_argument("--num-trials", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-sample-attempts", type=int, default=500000)

    # Common admissible motion.
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument(
        "--direct-step",
        type=float,
        default=0.01,
        help="Normalized direct-ascent L2 step length per iteration [rad].",
    )
    p.add_argument(
        "--path-budget",
        type=float,
        default=0.05,
        help="Common cumulative L2 path-length budget for every method [rad].",
    )
    p.add_argument(
        "--step-max",
        type=float,
        default=0.05,
        help="Per-joint trust-region radius |q-q0| [rad].",
    )

    # Projection-like rule.
    p.add_argument("--projection-damping", type=float, default=0.5)
    p.add_argument(
        "--projection-step-cap",
        type=float,
        default=0.05,
        help="Maximum L2 norm of one projection-like update before budget clipping [rad].",
    )
    p.add_argument("--projection-eps", type=float, default=1e-8)
    p.add_argument(
        "--tau-f",
        type=float,
        default=0.03,
        help="Hybrid switches from projection to direct ascent when learned f >= -tau_f.",
    )
    p.add_argument("--max-backtracks", type=int, default=4)
    p.add_argument("--grad-eps", type=float, default=1e-10)

    # Frozen conservative analytic FOV oracle. Evaluation only.
    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--base-frame", default="base_link")
    p.add_argument(
        "--oracle-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.01, 0.02],
    )
    return p.parse_args()


def clip_candidate(q_candidate, q0, q_min, q_max, step_max):
    lb = np.maximum(q_min, q0 - step_max)
    ub = np.minimum(q_max, q0 + step_max)
    return np.clip(q_candidate, lb, ub)


def monotone_step(
    value_grad_fn,
    x,
    q,
    dq,
    q0,
    q_min,
    q_max,
    step_max,
    max_backtracks,
):
    """Backtrack until learned f is nondecreasing; return accepted q or None."""
    f0, _ = eval_value_grad(value_grad_fn, x, q)
    for bt in range(max_backtracks + 1):
        scale = 0.5 ** bt
        q_candidate = clip_candidate(q + scale * dq, q0, q_min, q_max, step_max)
        actual = q_candidate - q
        actual_norm = float(np.linalg.norm(actual))
        if actual_norm <= 1e-12:
            continue
        f1, _ = eval_value_grad(value_grad_fn, x, q_candidate)
        if np.isfinite(f1) and f1 >= f0 - 1e-12:
            return q_candidate, actual_norm, bt
    return None, 0.0, max_backtracks + 1


def optimize_method(
    method: str,
    value_grad_fn,
    x: np.ndarray,
    q0: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    iterations: int,
    direct_step: float,
    path_budget: float,
    step_max: float,
    projection_damping: float,
    projection_step_cap: float,
    projection_eps: float,
    tau_f: float,
    max_backtracks: int,
    grad_eps: float,
):
    if method not in METHODS:
        raise ValueError(method)

    q = np.asarray(q0, dtype=np.float64).reshape(7).copy()
    x = np.asarray(x, dtype=np.float64).reshape(3)
    f_initial, _ = eval_value_grad(value_grad_fn, x, q)

    path_used = 0.0
    accepted_steps = 0
    backtracks = 0
    projection_steps = 0
    direct_steps = 0

    tic = time.perf_counter()
    for _ in range(iterations):
        remaining = path_budget - path_used
        if remaining <= 1e-12:
            break

        f, grad = eval_value_grad(value_grad_fn, x, q)
        grad_norm = float(np.linalg.norm(grad))
        if not np.isfinite(grad_norm) or grad_norm <= grad_eps:
            break

        use_projection = method == "projection" or (
            method == "hybrid" and f < -tau_f
        )

        if use_projection:
            denom = grad_norm * grad_norm + projection_eps
            dq = -projection_damping * f / denom * grad
            dq_norm = float(np.linalg.norm(dq))
            if not np.isfinite(dq_norm) or dq_norm <= 1e-12:
                break
            allowed = min(projection_step_cap, remaining)
            if dq_norm > allowed:
                dq *= allowed / dq_norm
            projection_steps += 1
        else:
            allowed = min(direct_step, remaining)
            dq = allowed * grad / grad_norm
            direct_steps += 1

        q_next, actual_norm, bt = monotone_step(
            value_grad_fn=value_grad_fn,
            x=x,
            q=q,
            dq=dq,
            q0=q0,
            q_min=q_min,
            q_max=q_max,
            step_max=step_max,
            max_backtracks=max_backtracks,
        )
        backtracks += bt
        if q_next is None:
            break

        q = q_next
        path_used += actual_norm
        accepted_steps += 1

    elapsed_ms = 1000.0 * (time.perf_counter() - tic)
    f_final, _ = eval_value_grad(value_grad_fn, x, q)

    return {
        "q_opt": q,
        "f_initial": f_initial,
        "f_final": f_final,
        "delta_f": f_final - f_initial,
        "path_used": path_used,
        "step_inf": float(np.max(np.abs(q - q0))),
        "step_l2_net": float(np.linalg.norm(q - q0)),
        "accepted_steps": accepted_steps,
        "backtracks": backtracks,
        "projection_steps": projection_steps,
        "direct_steps": direct_steps,
        "elapsed_ms": elapsed_ms,
    }


def cohort_name(g0: float) -> str:
    if g0 > -0.03:
        return "near"
    if g0 > -0.10:
        return "middle"
    return "far"


def summarize(method: str, rows: List[Dict], thresholds) -> Dict[str, float]:
    if not rows:
        return {"n": 0}
    df = np.asarray([r["delta_f"] for r in rows], dtype=np.float64)
    dg = np.asarray([r["delta_g"] for r in rows], dtype=np.float64)
    g1 = np.asarray([r["g_final"] for r in rows], dtype=np.float64)
    times = np.asarray([r["elapsed_ms"] for r in rows], dtype=np.float64)
    path = np.asarray([r["path_used"] for r in rows], dtype=np.float64)
    out = {
        "n": len(rows),
        "learned_improve": float(np.mean(df > 1e-8)),
        "oracle_improve": float(np.mean(dg > 1e-8)),
        "oracle_nondecrease": float(np.mean(dg >= -1e-8)),
        "mean_dg": float(np.mean(dg)),
        "median_dg": float(np.median(dg)),
        "crossing": float(np.mean(g1 >= 0.0)),
        "disagree": float(np.mean((df > 1e-8) & (dg < -1e-8))),
        "mean_time_ms": float(np.mean(times)),
        "max_time_ms": float(np.max(times)),
        "mean_path": float(np.mean(path)),
    }
    for t in thresholds:
        out[f"g_ge_{t:.3f}"] = float(np.mean(g1 >= t))
    return out


def print_summary_block(label: str, method_rows: Dict[str, List[Dict]], thresholds):
    print(f"\n=== {label} ===")
    print(
        "method       n    g_improve  mean_dg      cross(g>=0)  f_up/g_down  "
        "mean_path   mean_ms"
    )
    for method in METHODS:
        s = summarize(method, method_rows[method], thresholds)
        if s["n"] == 0:
            print(f"{method:10s} {0:4d}    n/a")
            continue
        print(
            f"{method:10s} {s['n']:4d}   "
            f"{100*s['oracle_improve']:8.1f}%  "
            f"{s['mean_dg']:+.6e}   "
            f"{100*s['crossing']:9.1f}%    "
            f"{100*s['disagree']:9.1f}%   "
            f"{s['mean_path']:.5f}     "
            f"{s['mean_time_ms']:.3f}"
        )

    print("  positive-margin rates:")
    for method in METHODS:
        s = summarize(method, method_rows[method], thresholds)
        if s["n"] == 0:
            continue
        pieces = []
        for t in thresholds:
            pieces.append(f"g>={t:+.3f}: {100*s[f'g_ge_{t:.3f}']:.1f}%")
        print(f"    {method:10s}  " + " | ".join(pieces))


def main() -> None:
    args = parse_args()
    for name in ("num_trials", "iterations"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    for name in (
        "direct_step", "path_budget", "step_max", "projection_step_cap",
        "projection_eps", "tau_f",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    if not (0.0 < args.projection_damping <= 1.0):
        raise ValueError("--projection-damping must lie in (0,1]")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf}")

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    print(f"[checkpoint] {checkpoint}")
    print(f"[urdf]       {urdf}")
    print(f"[device]     {device}")
    print(f"[samples]    {args.num_trials} initial-outside pairs shared by all methods")
    print(f"[budget]     cumulative L2 path <= {args.path_budget:.4f} rad")
    print(f"[trust]      per-joint |q-q0| <= {args.step_max:.4f} rad")
    print(f"[direct]     {args.iterations} x {args.direct_step:.4f} rad normalized steps")
    print(
        f"[projection] damping={args.projection_damping}, "
        f"step_cap={args.projection_step_cap:.4f} rad"
    )
    print(f"[hybrid]     projection while f < -{args.tau_f:.4f}; direct otherwise")
    print("[oracle]     evaluation only; never used for update/switch decisions")

    q_min, q_max = read_joint_limits(urdf, DEFAULT_JOINT_NAMES)
    ckpt = torch_load_checkpoint(str(checkpoint), device)
    model, _ = build_model_from_checkpoint(ckpt, device)
    _, _, _, value_grad_fn = build_casadi_functions(
        model=model,
        device=args.device,
        build_dir=build_dir,
        model_name=args.model_name,
    )
    value_grad_fn(np.asarray([[0.0, 0.0, 0.5]]), np.zeros((1, 7)))

    oracle = PinocchioFOVOracle(
        urdf_path=str(urdf),
        joint_names=list(DEFAULT_JOINT_NAMES),
        sensor_frames=list(DEFAULT_SENSOR_FRAMES),
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        z_min=args.z_min,
        z_max=args.z_max,
        delta=args.delta,
        base_frame=args.base_frame,
    )

    rng = np.random.default_rng(args.seed)
    span = q_max - q_min
    margin = np.minimum(0.1, 0.05 * span)
    q_lo = q_min + margin
    q_hi = q_max - margin

    rows: Dict[str, List[Dict]] = {m: [] for m in METHODS}
    attempts = 0
    collected = 0

    while collected < args.num_trials and attempts < args.max_sample_attempts:
        attempts += 1
        q0 = rng.uniform(q_lo, q_hi)
        x = np.asarray(
            [
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.0, 1.0),
            ],
            dtype=np.float64,
        )
        g0 = oracle_g(oracle, x, q0, device)
        if not np.isfinite(g0) or g0 >= 0.0:
            continue

        collected += 1
        cohort = cohort_name(g0)
        pieces = []
        for method in METHODS:
            r = optimize_method(
                method=method,
                value_grad_fn=value_grad_fn,
                x=x,
                q0=q0,
                q_min=q_min,
                q_max=q_max,
                iterations=args.iterations,
                direct_step=args.direct_step,
                path_budget=args.path_budget,
                step_max=args.step_max,
                projection_damping=args.projection_damping,
                projection_step_cap=args.projection_step_cap,
                projection_eps=args.projection_eps,
                tau_f=args.tau_f,
                max_backtracks=args.max_backtracks,
                grad_eps=args.grad_eps,
            )
            g1 = oracle_g(oracle, x, r["q_opt"], device)
            r.update(
                {
                    "g_initial": g0,
                    "g_final": g1,
                    "delta_g": g1 - g0,
                    "cohort": cohort,
                }
            )
            rows[method].append(r)
            pieces.append(
                f"{method[0].upper()}:dg={r['delta_g']:+.3e},g1={g1:+.3e},p={r['path_used']:.3f}"
            )

        print(
            f"[trial {collected:03d}/{args.num_trials:03d}] "
            f"{cohort:6s} g0={g0:+.3e} | " + " | ".join(pieces)
        )

    if collected < args.num_trials:
        raise RuntimeError(
            f"Collected only {collected}/{args.num_trials} initial-outside pairs "
            f"after {attempts} attempts."
        )

    print(f"\n[sampling] collected={collected}, attempts={attempts}")
    print_summary_block("ALL INITIAL-OUTSIDE", rows, args.oracle_thresholds)

    for cohort in ("near", "middle", "far"):
        subset = {
            method: [r for r in rows[method] if r["cohort"] == cohort]
            for method in METHODS
        }
        if any(subset[m] for m in METHODS):
            if cohort == "near":
                label = "NEAR: -0.03 < g0 < 0"
            elif cohort == "middle":
                label = "MIDDLE: -0.10 < g0 <= -0.03"
            else:
                label = "FAR: g0 <= -0.10"
            print_summary_block(label, subset, args.oracle_thresholds)

    print("\n[decision aid]")
    s_direct = summarize("direct", rows["direct"], args.oracle_thresholds)
    s_proj = summarize("projection", rows["projection"], args.oracle_thresholds)
    s_hybrid = summarize("hybrid", rows["hybrid"], args.oracle_thresholds)
    print(
        f"  hybrid - direct oracle-improve: "
        f"{100*(s_hybrid['oracle_improve']-s_direct['oracle_improve']):+.1f} percentage points"
    )
    print(
        f"  hybrid - direct mean delta-g  : "
        f"{s_hybrid['mean_dg']-s_direct['mean_dg']:+.6e}"
    )
    print(
        f"  hybrid - direct crossing rate : "
        f"{100*(s_hybrid['crossing']-s_direct['crossing']):+.1f} percentage points"
    )
    print(
        f"  hybrid - direct disagreement  : "
        f"{100*(s_hybrid['disagree']-s_direct['disagree']):+.1f} percentage points"
    )
    print(
        "  Interpretation: keep hybrid only if it gives a clear true-oracle gain, "
        "especially in FAR, without materially increasing f-up/g-down disagreement."
    )


if __name__ == "__main__":
    main()
