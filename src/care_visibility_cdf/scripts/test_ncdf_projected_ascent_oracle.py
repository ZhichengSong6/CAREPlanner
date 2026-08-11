#!/usr/bin/env python3
"""Validate fast L4CasADi projected NCDF ascent against the analytic FOV oracle.

This test intentionally samples only the active-sensing-relevant cohort:

    g(x, q_initial) < 0

where g is the conservative analytic multi-sensor FOV margin. It then applies
the fixed-iteration projected-gradient optimizer from
``test_ncdf_projected_ascent.py`` and measures whether the *true* oracle margin
also improves.

The learned NCDF remains guidance only. Oracle g is used here for evaluation,
not as an optimization objective or runtime stopping certificate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from test_ncdf_projected_ascent import optimize_once  # noqa: E402


DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf_projected_oracle"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate projected L4CasADi NCDF ascent using the analytic FOV oracle."
    )
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--urdf", default=str(DEFAULT_URDF))
    p.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--model-name", default="care_visibility_ncdf_projected_oracle")

    p.add_argument("--num-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-sample-attempts", type=int, default=200000)

    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--iter-step", type=float, default=0.01)
    p.add_argument("--step-max", type=float, default=0.05)
    p.add_argument("--vis-weight", type=float, default=1.0)
    p.add_argument("--task-weight", type=float, default=0.0)
    p.add_argument("--max-backtracks", type=int, default=4)
    p.add_argument("--grad-eps", type=float, default=1e-10)

    # Frozen conservative FOV oracle used by CARE visibility-CDF evaluation.
    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--base-frame", default="base_link")

    p.add_argument(
        "--oracle-thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02],
        help="True g thresholds reported after optimization.",
    )
    return p.parse_args()


def oracle_g(oracle: PinocchioFOVOracle, x: np.ndarray, q: np.ndarray, device: torch.device) -> float:
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(1, 3)
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device).reshape(1, 7)
    raw_margin, _ = oracle.signed_fov_margins(x_t, q_t)
    # raw_margin shape for one x / one q is [1, 1, num_sensors].
    # Conservative multi-sensor oracle: visible if any sensor has
    # raw_margin - delta >= 0.
    g = torch.max(raw_margin - oracle.delta, dim=-1).values
    return float(g.reshape(()).cpu())


def main() -> None:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.iter_step <= 0.0 or args.step_max <= 0.0:
        raise ValueError("--iter-step and --step-max must be positive")
    if args.task_weight < 0.0 or args.vis_weight <= 0.0:
        raise ValueError("Invalid objective weights")

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
    print(f"[cohort]     initial_outside: oracle g(x,q0) < 0")
    print(f"[iterations] {args.iterations}")
    print(f"[iter_step]  {args.iter_step:.6f} rad")
    print(f"[step_max]   {args.step_max:.6f} rad")
    print(f"[weights]    vis={args.vis_weight}, task={args.task_weight}")

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

    results = []
    attempts = 0
    while len(results) < args.num_trials and attempts < args.max_sample_attempts:
        attempts += 1
        q_current = rng.uniform(q_lo, q_hi)
        q_nominal = q_current.copy()
        x = np.asarray(
            [
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.0, 1.0),
            ],
            dtype=np.float64,
        )

        g0 = oracle_g(oracle, x, q_current, device)
        if not np.isfinite(g0) or g0 >= 0.0:
            continue

        r = optimize_once(
            value_grad_fn=value_grad_fn,
            x=x,
            q_current=q_current,
            q_nominal=q_nominal,
            q_min=q_min,
            q_max=q_max,
            iterations=args.iterations,
            iter_step=args.iter_step,
            step_max=args.step_max,
            vis_weight=args.vis_weight,
            task_weight=args.task_weight,
            max_backtracks=args.max_backtracks,
            grad_eps=args.grad_eps,
        )
        g1 = oracle_g(oracle, x, r["q_opt"], device)
        r["g_initial"] = g0
        r["g_final"] = g1
        r["delta_g"] = g1 - g0
        r["x"] = x
        r["q_current"] = q_current
        results.append(r)

        i = len(results)
        print(
            f"[trial {i:03d}/{args.num_trials:03d}] "
            f"df={r['delta_f']:+.4e} "
            f"g:{g0:+.4e}->{g1:+.4e} "
            f"dg={r['delta_g']:+.4e} "
            f"inside={int(g1 >= 0.0)} "
            f"time={r['elapsed_ms']:.2f}ms"
        )

    if len(results) < args.num_trials:
        raise RuntimeError(
            f"Collected only {len(results)}/{args.num_trials} initial-outside samples "
            f"after {attempts} attempts."
        )

    delta_f = np.asarray([r["delta_f"] for r in results], dtype=np.float64)
    g0 = np.asarray([r["g_initial"] for r in results], dtype=np.float64)
    g1 = np.asarray([r["g_final"] for r in results], dtype=np.float64)
    delta_g = g1 - g0
    solve_ms = np.asarray([r["elapsed_ms"] for r in results], dtype=np.float64)

    print("\n[summary: initial_outside]")
    print(f"  samples collected     : {len(results)} (attempts={attempts})")
    print(f"  learned-f improved    : {(delta_f > 1e-8).mean() * 100.0:.1f}%")
    print(f"  oracle-g improved     : {(delta_g > 1e-8).mean() * 100.0:.1f}%")
    print(f"  oracle-g nondecrease  : {(delta_g >= -1e-8).mean() * 100.0:.1f}%")
    print(f"  mean delta g          : {delta_g.mean():+.6e}")
    print(f"  median delta g        : {np.median(delta_g):+.6e}")
    print(f"  mean initial g        : {g0.mean():+.6e}")
    print(f"  mean final g          : {g1.mean():+.6e}")
    for threshold in args.oracle_thresholds:
        rate = (g1 >= threshold).mean() * 100.0
        print(f"  final g >= {threshold:+.3f}    : {rate:.1f}%")
    print(f"  mean optimizer time   : {solve_ms.mean():.3f} ms")
    print(f"  median optimizer time : {np.median(solve_ms):.3f} ms")
    print(f"  max optimizer time    : {solve_ms.max():.3f} ms")

    # A useful disagreement metric: learned NCDF rises while the true oracle
    # margin falls. This is the failure mode we care about before ROS rollout.
    disagree = (delta_f > 1e-8) & (delta_g < -1e-8)
    print(f"  f-up / g-down cases   : {disagree.mean() * 100.0:.1f}%")


if __name__ == "__main__":
    main()
