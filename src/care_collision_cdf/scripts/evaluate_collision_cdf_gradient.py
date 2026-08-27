#!/usr/bin/env python3
"""C5.1 semantic + latency diagnostic for the trained collision CDF.

This is offline and non-actuating. It validates the exact quantities that will
later be linearized inside the MPC:
  1) autograd dq gradient vs central finite difference,
  2) local +grad / -grad directionality,
  3) scene-level batch inference latency for K=20 horizon states.

The finite-difference test uses ONE workspace point per trial to avoid the
non-smooth argmin switching that is intrinsic to a scene-level min over points.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
SRC_DIR = PACKAGE_DIR.parent
REPO_ROOT = SRC_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collision_cdf_model import CollisionCDF


DEFAULT_CHECKPOINT = PACKAGE_DIR / "checkpoints" / "yiming_cdf" / "model_dict.pt"
DEFAULT_URDF = SRC_DIR / "arm_description" / "urdf" / "Arm.urdf"
JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "wrist_joint1",
    "wrist_joint2",
    "wrist_joint3",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--urdf", default=str(DEFAULT_URDF))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--checkpoint-key", default="latest")
    p.add_argument("--num-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fd-eps", type=float, default=1e-4)
    p.add_argument("--direction-step", type=float, default=1e-3)
    p.add_argument("--q-limit-margin", type=float, default=0.10)
    p.add_argument("--horizon-q", type=int, default=20)
    p.add_argument("--latency-points", type=int, nargs="+", default=[1, 10, 50, 100, 250])
    p.add_argument("--latency-repeats", type=int, default=8)
    p.add_argument("--output", default=str(REPO_ROOT / "outputs" / "c5_1_collision_cdf_diagnostic.json"))
    return p.parse_args()


def read_joint_limits(path: Path):
    root = ET.parse(str(path)).getroot()
    by_name = {j.attrib.get("name", ""): j for j in root.findall("joint")}
    lo, hi = [], []
    for name in JOINT_NAMES:
        j = by_name[name]
        if j.attrib.get("type") == "continuous":
            lo.append(-math.pi)
            hi.append(math.pi)
            continue
        lim = j.find("limit")
        lo.append(float(lim.attrib["lower"]))
        hi.append(float(lim.attrib["upper"]))
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


@torch.no_grad()
def distance_only(cdf: CollisionCDF, points_np, q_np):
    points = torch.as_tensor(points_np, dtype=torch.float32, device=cdf.device)
    q = torch.as_tensor(q_np, dtype=torch.float32, device=cdf.device)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if q.ndim == 1:
        q = q.reshape(1, 7)

    np_, nq = points.shape[0], q.shape[0]
    x_cat = points[:, None, :].expand(np_, nq, 3).reshape(-1, 3)
    q_cat = q[None, :, :].expand(np_, nq, 7).reshape(-1, 7)
    pred = cdf.model(torch.cat((x_cat, q_cat), dim=-1)).reshape(np_, nq)
    return torch.min(pred, dim=0).values.detach().cpu().numpy().astype(np.float64)


def evaluate_grad(cdf, point, q):
    with torch.enable_grad():
        d, g, idx = cdf.scene_distance_and_gradient(
            torch.as_tensor(point, dtype=torch.float32).reshape(1, 3),
            torch.as_tensor(q, dtype=torch.float32).reshape(1, 7),
        )
    return (
        float(d.cpu().numpy()[0]),
        g.cpu().numpy()[0].astype(np.float64),
        int(idx.cpu().numpy()[0]),
    )


def finite_difference(cdf, point, q, eps):
    fd = np.zeros(7, dtype=np.float64)
    for j in range(7):
        qp = q.copy()
        qm = q.copy()
        qp[j] += eps
        qm[j] -= eps
        dp = float(distance_only(cdf, point, qp)[0])
        dm = float(distance_only(cdf, point, qm)[0])
        fd[j] = (dp - dm) / (2.0 * eps)
    return fd


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    urdf = Path(args.urdf).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    if args.num_trials <= 0 or args.fd_eps <= 0 or args.direction_step <= 0:
        raise ValueError("invalid diagnostic parameters")

    q_min, q_max = read_joint_limits(urdf)
    span = q_max - q_min
    q_lo = q_min + args.q_limit_margin * span
    q_hi = q_max - args.q_limit_margin * span

    cdf = CollisionCDF(
        checkpoint_path=str(checkpoint),
        device=args.device,
        checkpoint_key=args.checkpoint_key,
        input_dims=10,
        output_dims=1,
    )

    print("[C5.1] checkpoint:", checkpoint)
    print("[C5.1] selected:", cdf.selected_checkpoint)
    print("[C5.1] architecture:", cdf.architecture)
    print("[C5.1] device:", cdf.device)
    print("[C5.1] q_min:", q_min)
    print("[C5.1] q_max:", q_max)

    rng = np.random.default_rng(args.seed)
    rows = []

    # CARE workspace, kept slightly away from the extreme outer shell.
    ws_lo = np.asarray([-0.80, -0.80, 0.05], dtype=np.float64)
    ws_hi = np.asarray([+0.80, +0.80, 1.05], dtype=np.float64)

    for trial in range(args.num_trials):
        q = rng.uniform(q_lo, q_hi)
        point = rng.uniform(ws_lo, ws_hi)

        d0, grad, _ = evaluate_grad(cdf, point, q)
        fd = finite_difference(cdf, point, q, args.fd_eps)
        err = grad - fd
        cos = cosine(grad, fd)
        grad_norm = float(np.linalg.norm(grad))

        if grad_norm > 1e-10:
            direction = grad / grad_norm
            q_plus = np.clip(q + args.direction_step * direction, q_min, q_max)
            q_minus = np.clip(q - args.direction_step * direction, q_min, q_max)
            d_plus = float(distance_only(cdf, point, q_plus)[0])
            d_minus = float(distance_only(cdf, point, q_minus)[0])
        else:
            d_plus = d0
            d_minus = d0

        rows.append(
            {
                "trial": trial,
                "distance": d0,
                "grad_norm": grad_norm,
                "fd_cosine": cos,
                "fd_l2_error": float(np.linalg.norm(err)),
                "fd_inf_error": float(np.max(np.abs(err))),
                "plus_delta": float(d_plus - d0),
                "minus_delta": float(d_minus - d0),
                "plus_increases": bool(d_plus > d0),
                "minus_decreases": bool(d_minus < d0),
            }
        )

    finite_cos = np.asarray(
        [r["fd_cosine"] for r in rows if np.isfinite(r["fd_cosine"])],
        dtype=np.float64,
    )
    fd_inf = np.asarray([r["fd_inf_error"] for r in rows], dtype=np.float64)
    plus = np.asarray([r["plus_delta"] for r in rows], dtype=np.float64)
    minus = np.asarray([r["minus_delta"] for r in rows], dtype=np.float64)

    semantic_summary = {
        "num_trials": len(rows),
        "fd_cosine_mean": float(np.mean(finite_cos)) if len(finite_cos) else None,
        "fd_cosine_median": float(np.median(finite_cos)) if len(finite_cos) else None,
        "fd_cosine_p05": float(np.quantile(finite_cos, 0.05)) if len(finite_cos) else None,
        "fd_inf_error_mean": float(np.mean(fd_inf)),
        "fd_inf_error_p95": float(np.quantile(fd_inf, 0.95)),
        "plus_gradient_increase_rate": float(np.mean(plus > 0.0)),
        "minus_gradient_decrease_rate": float(np.mean(minus < 0.0)),
        "plus_delta_mean": float(np.mean(plus)),
        "minus_delta_mean": float(np.mean(minus)),
    }

    print("[C5.1] semantic summary:")
    print(json.dumps(semantic_summary, indent=2))

    latency = {}
    q_batch = rng.uniform(q_lo, q_hi, size=(args.horizon_q, 7)).astype(np.float32)
    for n_points in args.latency_points:
        points = rng.uniform(ws_lo, ws_hi, size=(int(n_points), 3)).astype(np.float32)

        # warmup
        for _ in range(2):
            with torch.enable_grad():
                cdf.scene_distance_and_gradient(
                    torch.from_numpy(points), torch.from_numpy(q_batch)
                )
        sync(cdf.device)

        times = []
        for _ in range(args.latency_repeats):
            sync(cdf.device)
            t0 = time.perf_counter()
            with torch.enable_grad():
                cdf.scene_distance_and_gradient(
                    torch.from_numpy(points), torch.from_numpy(q_batch)
                )
            sync(cdf.device)
            times.append(1000.0 * (time.perf_counter() - t0))

        latency[str(int(n_points))] = {
            "horizon_q": int(args.horizon_q),
            "pairs": int(n_points) * int(args.horizon_q),
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }

    print("[C5.1] latency summary:")
    print(json.dumps(latency, indent=2))

    payload = {
        "checkpoint": str(checkpoint),
        "selected_checkpoint": cdf.selected_checkpoint,
        "architecture": cdf.architecture,
        "device": str(cdf.device),
        "fd_eps": args.fd_eps,
        "direction_step": args.direction_step,
        "semantic_summary": semantic_summary,
        "latency": latency,
        "trials": rows,
    }
    output.write_text(json.dumps(payload, indent=2))
    print("[C5.1] output:", output)
    print("[C5.1 DIAGNOSTIC COMPLETE]")


if __name__ == "__main__":
    main()
