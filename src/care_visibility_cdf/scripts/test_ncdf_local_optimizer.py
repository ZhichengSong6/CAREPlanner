#!/usr/bin/env python3
"""Offline local visibility optimizer for the frozen CARE NCDF.

Purpose
-------
Validate the optimization layer *before* connecting it to ROS execution.
Given a workspace sensing target x, a current configuration q_current, and a
nominal short-step configuration q_nominal, solve a small 7-DoF NLP:

    minimize_q  -w_vis * f_theta(x, q)
                + 0.5 * w_task * ||(q - q_nominal) / step_max||^2

subject to

    URDF joint limits
    |q - q_current| <= step_max

The default first test sets q_nominal == q_current and w_task == 0, i.e. it
asks a very simple question: within a small admissible joint step, can CasADi
use the L4CasADi NCDF to increase the learned visibility field?

This script does NOT publish commands and does NOT move the robot.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Sequence, Tuple

import casadi as cs
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
SRC_DIR = PACKAGE_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    build_model_from_checkpoint,
    torch_load_checkpoint,
)
from export_ncdf_l4casadi import build_casadi_functions  # noqa: E402


DEFAULT_CHECKPOINT = (
    PACKAGE_DIR
    / "checkpoints"
    / "exp1_yiming_k500_fov_signed"
    / "final.pt"
)
DEFAULT_URDF = SRC_DIR / "arm_description" / "urdf" / "Arm.urdf"
DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf_optimizer"
DEFAULT_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "wrist_joint1",
    "wrist_joint2",
    "wrist_joint3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline 7-DoF local CasADi optimization test using the frozen CARE visibility NCDF."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-name", default="care_visibility_ncdf_optimizer")

    parser.add_argument(
        "--step-max",
        type=float,
        default=0.05,
        help="Per-joint trust-region bound |q_opt-q_current| in radians.",
    )
    parser.add_argument(
        "--vis-weight",
        type=float,
        default=1.0,
        help="Weight multiplying -f_theta in the minimization objective.",
    )
    parser.add_argument(
        "--task-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on normalized deviation from q_nominal. Use 0 for the first "
            "visibility-only integration test."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--tol", type=float, default=1e-6)

    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--x",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional explicit sensing target in base_link. Requires --q.",
    )
    parser.add_argument(
        "--q",
        type=float,
        nargs=7,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="Optional explicit current 7-DoF joint configuration. Requires --x.",
    )
    parser.add_argument(
        "--q-nominal",
        type=float,
        nargs=7,
        metavar=("QN1", "QN2", "QN3", "QN4", "QN5", "QN6", "QN7"),
        help="Optional nominal short-step configuration; defaults to q_current.",
    )
    return parser.parse_args()


def read_joint_limits(urdf_path: Path, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    root = ET.parse(str(urdf_path)).getroot()
    by_name: Dict[str, ET.Element] = {
        joint.attrib.get("name", ""): joint for joint in root.findall("joint")
    }

    q_min = []
    q_max = []
    for name in joint_names:
        if name not in by_name:
            raise RuntimeError(f"Joint {name!r} not found in URDF: {urdf_path}")

        joint = by_name[name]
        joint_type = joint.attrib.get("type", "")
        if joint_type == "continuous":
            q_min.append(-math.pi)
            q_max.append(math.pi)
            continue

        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise RuntimeError(
                f"Joint {name!r} has no finite lower/upper limit in {urdf_path}."
            )
        q_min.append(float(limit.attrib["lower"]))
        q_max.append(float(limit.attrib["upper"]))

    return np.asarray(q_min, dtype=np.float64), np.asarray(q_max, dtype=np.float64)


def build_local_solver(l4c_model, step_max: float, vis_weight: float, task_weight: float,
                       max_iter: int, tol: float):
    q_opt = cs.MX.sym("q_opt", 7, 1)
    p = cs.MX.sym("p", 17, 1)
    x = p[0:3]
    q_current = p[3:10]
    q_nominal = p[10:17]

    z = cs.horzcat(x.T, q_opt.T)
    f = l4c_model(z)

    normalized_deviation = (q_opt - q_nominal) / step_max
    task_cost = 0.5 * task_weight * cs.sumsqr(normalized_deviation)
    objective = -vis_weight * f + task_cost

    nlp = {"x": q_opt, "p": p, "f": objective}
    opts = {
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": int(max_iter),
        "ipopt.tol": float(tol),
        "ipopt.hessian_approximation": "limited-memory",
    }
    solver = cs.nlpsol("care_ncdf_local_solver", "ipopt", nlp, opts)
    f_fn = cs.Function("care_ncdf_local_f", [x, q_opt], [f])
    return solver, f_fn


def solve_once(solver, f_fn, x: np.ndarray, q_current: np.ndarray,
               q_nominal: np.ndarray, q_min: np.ndarray, q_max: np.ndarray,
               step_max: float):
    x = np.asarray(x, dtype=np.float64).reshape(3)
    q_current = np.asarray(q_current, dtype=np.float64).reshape(7)
    q_nominal = np.asarray(q_nominal, dtype=np.float64).reshape(7)

    lbx = np.maximum(q_min, q_current - step_max)
    ubx = np.minimum(q_max, q_current + step_max)
    if np.any(lbx > ubx):
        raise RuntimeError("Empty local feasible set after intersecting joint/trust-region bounds.")

    q0 = np.clip(q_nominal, lbx, ubx)
    p_value = np.concatenate([x, q_current, q_nominal])

    f_current = float(np.asarray(f_fn(x, q_current)).reshape(()))
    f_nominal = float(np.asarray(f_fn(x, q_nominal)).reshape(()))

    tic = time.perf_counter()
    sol = solver(x0=q0, p=p_value, lbx=lbx, ubx=ubx)
    elapsed_ms = 1000.0 * (time.perf_counter() - tic)

    q_opt = np.asarray(sol["x"], dtype=np.float64).reshape(7)
    f_opt = float(np.asarray(f_fn(x, q_opt)).reshape(()))
    stats = solver.stats()

    return {
        "q_opt": q_opt,
        "f_current": f_current,
        "f_nominal": f_nominal,
        "f_opt": f_opt,
        "delta_f_current": f_opt - f_current,
        "delta_f_nominal": f_opt - f_nominal,
        "step_inf": float(np.max(np.abs(q_opt - q_current))),
        "step_l2": float(np.linalg.norm(q_opt - q_current)),
        "nominal_deviation_inf": float(np.max(np.abs(q_opt - q_nominal))),
        "elapsed_ms": elapsed_ms,
        "success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "")),
        "iter_count": int(stats.get("iter_count", -1)),
    }


def print_single_result(x: np.ndarray, q_current: np.ndarray, q_nominal: np.ndarray, result) -> None:
    np.set_printoptions(precision=6, suppress=True)
    print("[input] x              =", np.asarray(x))
    print("[input] q_current      =", np.asarray(q_current))
    print("[input] q_nominal      =", np.asarray(q_nominal))
    print("[result] success       =", result["success"], result["return_status"])
    print("[result] iterations    =", result["iter_count"])
    print("[result] q_opt          =", result["q_opt"])
    print(f"[result] f_current      = {result['f_current']:.8f}")
    print(f"[result] f_nominal      = {result['f_nominal']:.8f}")
    print(f"[result] f_opt          = {result['f_opt']:.8f}")
    print(f"[result] delta_f_current= {result['delta_f_current']:+.8f}")
    print(f"[result] delta_f_nominal= {result['delta_f_nominal']:+.8f}")
    print(f"[result] step_inf       = {result['step_inf']:.6f} rad")
    print(f"[result] step_l2        = {result['step_l2']:.6f} rad")
    print(f"[result] solve_time     = {result['elapsed_ms']:.3f} ms")


def main() -> None:
    args = parse_args()

    if args.step_max <= 0.0:
        raise ValueError("--step-max must be positive.")
    if args.vis_weight <= 0.0:
        raise ValueError("--vis-weight must be positive.")
    if args.task_weight < 0.0:
        raise ValueError("--task-weight must be non-negative.")
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")
    if (args.x is None) != (args.q is None):
        raise ValueError("Provide --x and --q together, or neither.")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf}")

    torch_device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False.")

    print(f"[checkpoint] {checkpoint}")
    print(f"[urdf]       {urdf}")
    print(f"[device]     {torch_device}")
    print(f"[step_max]   {args.step_max:.6f} rad")
    print(f"[weights]    vis={args.vis_weight}, task={args.task_weight}")

    q_min, q_max = read_joint_limits(urdf, DEFAULT_JOINT_NAMES)
    print("[limits] q_min =", q_min)
    print("[limits] q_max =", q_max)

    ckpt = torch_load_checkpoint(str(checkpoint), torch_device)
    model, _ = build_model_from_checkpoint(ckpt, torch_device)

    l4c_model, _, _, value_grad_fn = build_casadi_functions(
        model=model,
        device=args.device,
        build_dir=build_dir,
        model_name=args.model_name,
    )

    x_warm = np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32)
    q_warm = np.zeros((1, 7), dtype=np.float32)
    value_grad_fn(x_warm, q_warm)

    solver, f_fn = build_local_solver(
        l4c_model=l4c_model,
        step_max=args.step_max,
        vis_weight=args.vis_weight,
        task_weight=args.task_weight,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    if args.x is not None:
        x = np.asarray(args.x, dtype=np.float64)
        q_current = np.asarray(args.q, dtype=np.float64)
        q_nominal = (
            np.asarray(args.q_nominal, dtype=np.float64)
            if args.q_nominal is not None
            else q_current.copy()
        )
        result = solve_once(
            solver, f_fn, x, q_current, q_nominal, q_min, q_max, args.step_max
        )
        print_single_result(x, q_current, q_nominal, result)
        return

    rng = np.random.default_rng(args.seed)
    results = []
    for trial in range(args.num_trials):
        span = q_max - q_min
        margin = np.minimum(0.1, 0.05 * span)
        lo = q_min + margin
        hi = q_max - margin
        q_current = rng.uniform(lo, hi)
        q_nominal = q_current.copy()
        x = np.asarray(
            [
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.0, 1.0),
            ],
            dtype=np.float64,
        )

        result = solve_once(
            solver, f_fn, x, q_current, q_nominal, q_min, q_max, args.step_max
        )
        results.append(result)
        print(
            f"[trial {trial + 1:03d}/{args.num_trials:03d}] "
            f"success={int(result['success'])} "
            f"df={result['delta_f_current']:+.6e} "
            f"step_inf={result['step_inf']:.4f} "
            f"time={result['elapsed_ms']:.2f}ms"
        )

    success = np.asarray([r["success"] for r in results], dtype=bool)
    delta_f = np.asarray([r["delta_f_current"] for r in results], dtype=np.float64)
    solve_ms = np.asarray([r["elapsed_ms"] for r in results], dtype=np.float64)
    step_inf = np.asarray([r["step_inf"] for r in results], dtype=np.float64)

    improved = delta_f > 1e-8
    nondecreased = delta_f >= -1e-8

    print("\n[summary]")
    print(f"  solver success       : {success.mean() * 100.0:.1f}%")
    print(f"  learned-f improved   : {improved.mean() * 100.0:.1f}%")
    print(f"  learned-f nondecrease: {nondecreased.mean() * 100.0:.1f}%")
    print(f"  mean delta f         : {delta_f.mean():+.6e}")
    print(f"  median delta f       : {np.median(delta_f):+.6e}")
    print(f"  mean step inf        : {step_inf.mean():.6f} rad")
    print(f"  mean solve time      : {solve_ms.mean():.3f} ms")
    print(f"  median solve time    : {np.median(solve_ms):.3f} ms")
    print(f"  max solve time       : {solve_ms.max():.3f} ms")

    if not np.all(success):
        raise RuntimeError("At least one local NLP solve failed; inspect return_status above.")


if __name__ == "__main__":
    main()
