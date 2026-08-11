#!/usr/bin/env python3
"""Fast local projected-gradient visibility optimization using L4CasADi.

This is the online-oriented counterpart to test_ncdf_local_optimizer.py.
Instead of asking IPOPT to solve a nonsmooth ReLU-network NLP to strict
convergence, it takes a fixed number of bounded ascent steps using the already
verified CasADi q-gradient of the frozen NCDF.

The update maximizes

    J(q) = w_vis * f_theta(x, q)
           - 0.5 * w_task * ||(q - q_nominal) / step_max||^2

while projecting every candidate onto both the URDF joint limits and the local
trust-region box |q-q_current| <= step_max.

This script is offline only. It publishes no ROS commands and cannot move the
robot.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
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


DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf_projected"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline fixed-iteration projected-gradient NCDF optimization test."
    )
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--urdf", default=str(DEFAULT_URDF))
    p.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--model-name", default="care_visibility_ncdf_projected")

    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument(
        "--iter-step",
        type=float,
        default=0.01,
        help="Nominal L2 step length in radians for each normalized-gradient update.",
    )
    p.add_argument(
        "--step-max",
        type=float,
        default=0.05,
        help="Per-joint trust-region radius |q-q_current| in radians.",
    )
    p.add_argument("--vis-weight", type=float, default=1.0)
    p.add_argument("--task-weight", type=float, default=0.0)
    p.add_argument("--max-backtracks", type=int, default=4)
    p.add_argument("--grad-eps", type=float, default=1e-10)

    p.add_argument("--x", nargs=3, type=float, metavar=("X", "Y", "Z"))
    p.add_argument(
        "--q",
        nargs=7,
        type=float,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
    )
    p.add_argument(
        "--q-nominal",
        nargs=7,
        type=float,
        metavar=("QN1", "QN2", "QN3", "QN4", "QN5", "QN6", "QN7"),
    )
    return p.parse_args()


def eval_value_grad(value_grad_fn, x: np.ndarray, q: np.ndarray):
    out = value_grad_fn(
        np.asarray(x, dtype=np.float64).reshape(1, 3),
        np.asarray(q, dtype=np.float64).reshape(1, 7),
    )
    f = float(np.asarray(out[0]).reshape(()))
    grad = np.asarray(out[1], dtype=np.float64).reshape(7)
    return f, grad


def objective_and_grad(
    value_grad_fn,
    x: np.ndarray,
    q: np.ndarray,
    q_nominal: np.ndarray,
    step_max: float,
    vis_weight: float,
    task_weight: float,
):
    f, grad_f = eval_value_grad(value_grad_fn, x, q)
    scaled_dev = (q - q_nominal) / step_max
    objective = vis_weight * f - 0.5 * task_weight * float(np.dot(scaled_dev, scaled_dev))
    grad = vis_weight * grad_f - task_weight * (q - q_nominal) / (step_max * step_max)
    return objective, f, grad


def objective_only(
    value_grad_fn,
    x: np.ndarray,
    q: np.ndarray,
    q_nominal: np.ndarray,
    step_max: float,
    vis_weight: float,
    task_weight: float,
):
    f, _ = eval_value_grad(value_grad_fn, x, q)
    scaled_dev = (q - q_nominal) / step_max
    objective = vis_weight * f - 0.5 * task_weight * float(np.dot(scaled_dev, scaled_dev))
    return objective, f


def optimize_once(
    value_grad_fn,
    x: np.ndarray,
    q_current: np.ndarray,
    q_nominal: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    iterations: int,
    iter_step: float,
    step_max: float,
    vis_weight: float,
    task_weight: float,
    max_backtracks: int,
    grad_eps: float,
):
    x = np.asarray(x, dtype=np.float64).reshape(3)
    q_current = np.asarray(q_current, dtype=np.float64).reshape(7)
    q_nominal = np.asarray(q_nominal, dtype=np.float64).reshape(7)

    lb = np.maximum(q_min, q_current - step_max)
    ub = np.minimum(q_max, q_current + step_max)
    q = np.clip(q_nominal, lb, ub)

    obj_initial, f_initial, _ = objective_and_grad(
        value_grad_fn, x, q, q_nominal, step_max, vis_weight, task_weight
    )

    accepted_steps = 0
    backtracks_total = 0
    function_evals = 1

    tic = time.perf_counter()
    for _ in range(iterations):
        obj, _, grad = objective_and_grad(
            value_grad_fn, x, q, q_nominal, step_max, vis_weight, task_weight
        )
        function_evals += 1
        grad_norm = float(np.linalg.norm(grad))
        if not np.isfinite(grad_norm) or grad_norm <= grad_eps:
            break

        direction = grad / grad_norm
        trial_step = iter_step
        accepted = False

        for bt in range(max_backtracks + 1):
            q_candidate = np.clip(q + trial_step * direction, lb, ub)
            if np.allclose(q_candidate, q, atol=1e-12, rtol=0.0):
                break

            obj_candidate, _ = objective_only(
                value_grad_fn,
                x,
                q_candidate,
                q_nominal,
                step_max,
                vis_weight,
                task_weight,
            )
            function_evals += 1

            # Monotonic acceptance is deliberately simple and robust for this
            # first online-oriented test. No Wolfe assumptions are needed for
            # the ReLU network.
            if np.isfinite(obj_candidate) and obj_candidate >= obj - 1e-12:
                q = q_candidate
                accepted_steps += 1
                backtracks_total += bt
                accepted = True
                break
            trial_step *= 0.5

        if not accepted:
            break

    elapsed_ms = 1000.0 * (time.perf_counter() - tic)
    obj_final, f_final = objective_only(
        value_grad_fn, x, q, q_nominal, step_max, vis_weight, task_weight
    )
    function_evals += 1

    return {
        "q_opt": q,
        "f_initial": f_initial,
        "f_final": f_final,
        "delta_f": f_final - f_initial,
        "objective_initial": obj_initial,
        "objective_final": obj_final,
        "delta_objective": obj_final - obj_initial,
        "step_inf": float(np.max(np.abs(q - q_current))),
        "step_l2": float(np.linalg.norm(q - q_current)),
        "accepted_steps": accepted_steps,
        "backtracks": backtracks_total,
        "function_evals": function_evals,
        "elapsed_ms": elapsed_ms,
    }


def main() -> None:
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.iter_step <= 0.0:
        raise ValueError("--iter-step must be positive")
    if args.step_max <= 0.0:
        raise ValueError("--step-max must be positive")
    if args.vis_weight <= 0.0:
        raise ValueError("--vis-weight must be positive")
    if args.task_weight < 0.0:
        raise ValueError("--task-weight must be non-negative")
    if (args.x is None) != (args.q is None):
        raise ValueError("Provide --x and --q together, or neither")

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

    # Warm up generated external function before collecting timing numbers.
    value_grad_fn(np.asarray([[0.0, 0.0, 0.5]]), np.zeros((1, 7)))

    def run_one(x, q_current, q_nominal):
        return optimize_once(
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

    if args.x is not None:
        x = np.asarray(args.x, dtype=np.float64)
        q_current = np.asarray(args.q, dtype=np.float64)
        q_nominal = (
            np.asarray(args.q_nominal, dtype=np.float64)
            if args.q_nominal is not None
            else q_current.copy()
        )
        r = run_one(x, q_current, q_nominal)
        np.set_printoptions(precision=6, suppress=True)
        print("[input] x         =", x)
        print("[input] q_current =", q_current)
        print("[result] q_opt     =", r["q_opt"])
        for key in (
            "f_initial", "f_final", "delta_f", "objective_initial",
            "objective_final", "delta_objective", "step_inf", "step_l2",
            "accepted_steps", "backtracks", "function_evals", "elapsed_ms",
        ):
            print(f"[result] {key:18s}= {r[key]}")
        return

    rng = np.random.default_rng(args.seed)
    results = []
    span = q_max - q_min
    margin = np.minimum(0.1, 0.05 * span)
    lo = q_min + margin
    hi = q_max - margin

    for i in range(args.num_trials):
        q_current = rng.uniform(lo, hi)
        q_nominal = q_current.copy()
        x = np.asarray(
            [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.0, 1.0)],
            dtype=np.float64,
        )
        r = run_one(x, q_current, q_nominal)
        results.append(r)
        print(
            f"[trial {i + 1:03d}/{args.num_trials:03d}] "
            f"df={r['delta_f']:+.6e} "
            f"dJ={r['delta_objective']:+.6e} "
            f"step_inf={r['step_inf']:.4f} "
            f"steps={r['accepted_steps']}/{args.iterations} "
            f"time={r['elapsed_ms']:.2f}ms"
        )

    delta_f = np.asarray([r["delta_f"] for r in results], dtype=np.float64)
    delta_j = np.asarray([r["delta_objective"] for r in results], dtype=np.float64)
    solve_ms = np.asarray([r["elapsed_ms"] for r in results], dtype=np.float64)
    step_inf = np.asarray([r["step_inf"] for r in results], dtype=np.float64)
    accepted = np.asarray([r["accepted_steps"] for r in results], dtype=np.float64)

    print("\n[summary]")
    print(f"  learned-f improved    : {(delta_f > 1e-8).mean() * 100.0:.1f}%")
    print(f"  objective nondecrease : {(delta_j >= -1e-10).mean() * 100.0:.1f}%")
    print(f"  mean delta f          : {delta_f.mean():+.6e}")
    print(f"  median delta f        : {np.median(delta_f):+.6e}")
    print(f"  mean step inf         : {step_inf.mean():.6f} rad")
    print(f"  mean accepted steps   : {accepted.mean():.2f}/{args.iterations}")
    print(f"  mean solve time       : {solve_ms.mean():.3f} ms")
    print(f"  median solve time     : {np.median(solve_ms):.3f} ms")
    print(f"  max solve time        : {solve_ms.max():.3f} ms")


if __name__ == "__main__":
    main()
