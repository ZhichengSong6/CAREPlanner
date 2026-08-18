#!/usr/bin/env python3
"""Checkpoint-only geometry diagnostic for CAREPlanner Visibility CDF.

No training .npz is required.  The script uses only:
  1) the frozen VisCDF checkpoint,
  2) the CAREPlanner arm URDF,
  3) the analytic FOV oracle.

It evaluates:
  - empirical ||grad_q f|| distribution,
  - learned/oracle sign agreement,
  - whether normalized learned-gradient ascent increases true oracle g,
  - SDF-style projection q - f grad versus scale-invariant Newton projection
        q - f grad / ||grad||^2,
  - for oracle-outside states, the true oracle boundary distance ALONG the learned
    gradient ray (when a crossing exists), and whether |f| or |f|/||grad f|| is
    better calibrated to that local crossing distance.

Important: the ray-crossing distance is not the global shortest C-space distance to
visibility.  It is a local directional diagnostic that can be computed without the
q0-library training dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    model_value,
    model_value_and_grad_q,
    oracle_visibility_g,
    torch_load_checkpoint,
)

EPS = 1e-8

DEFAULT_Q_MIN = np.asarray([-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20], dtype=np.float32)
DEFAULT_Q_MAX = np.asarray([+3.14, +2.30, +3.14, +2.65, +3.14, +3.14, +1.20], dtype=np.float32)
DEFAULT_X_MIN = np.asarray([-0.95, -0.95, 0.00], dtype=np.float32)
DEFAULT_X_MAX = np.asarray([+0.95, +0.95, 1.15], dtype=np.float32)


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def parse_vec(text: str, n: int) -> np.ndarray:
    vals = [float(v) for v in text.replace(",", " ").split() if v]
    if len(vals) != n:
        raise ValueError(f"Expected {n} numbers, got {len(vals)} from {text!r}")
    return np.asarray(vals, dtype=np.float32)


def stats(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {k: float("nan") for k in ["mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "p99", "max"]}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "p05": float(np.quantile(v, 0.05)),
        "p25": float(np.quantile(v, 0.25)),
        "p50": float(np.quantile(v, 0.50)),
        "p75": float(np.quantile(v, 0.75)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
        "max": float(np.max(v)),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def clamp_q(q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    qc = torch.maximum(torch.minimum(q, q_max[None, :]), q_min[None, :])
    changed = torch.any(torch.abs(qc - q) > 1e-10, dim=-1)
    return qc, changed


def sample_x(args, device: torch.device, target_index: int) -> torch.Tensor:
    if args.fixed_target:
        x = parse_vec(args.fixed_target, 3)
        return torch.as_tensor(x[None, :], device=device)

    if args.x_source == "unit_box":
        x = torch.rand((1, 3), device=device)
        x[:, 0:2] -= 0.5
        return x

    x_min = torch.as_tensor(parse_vec(args.x_min, 3), device=device)
    x_max = torch.as_tensor(parse_vec(args.x_max, 3), device=device)
    return x_min[None, :] + torch.rand((1, 3), device=device) * (x_max - x_min)[None, :]


def oracle_crossing_along_gradient(
    x: torch.Tensor,
    q0: torch.Tensor,
    direction: torch.Tensor,
    g0: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    max_distance: float,
    coarse_steps: int,
    bisect_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For g0<0 samples, find first g>=0 along q0 + alpha*direction.

    Returns crossing_distance [N] (NaN if no crossing) and reached [N].
    """
    n = q0.shape[0]
    crossing = torch.full((n,), float("nan"), device=q0.device)
    reached = torch.zeros((n,), dtype=torch.bool, device=q0.device)
    if n == 0:
        return crossing, reached

    lo = torch.zeros((n,), device=q0.device)
    hi = torch.zeros((n,), device=q0.device)
    prev_alpha = 0.0
    active = torch.ones((n,), dtype=torch.bool, device=q0.device)

    for step in range(1, coarse_steps + 1):
        alpha = max_distance * float(step) / float(coarse_steps)
        q = q0 + alpha * direction
        q, _ = clamp_q(q, q_min, q_max)
        g = oracle_visibility_g(x, q, oracle)
        newly = active & (g >= 0.0)
        if torch.any(newly):
            lo[newly] = prev_alpha
            hi[newly] = alpha
            reached[newly] = True
            active[newly] = False
        prev_alpha = alpha
        if not torch.any(active):
            break

    if torch.any(reached):
        idx = torch.nonzero(reached, as_tuple=False).squeeze(-1)
        q_sel = q0[idx]
        d_sel = direction[idx]
        lo_sel = lo[idx].clone()
        hi_sel = hi[idx].clone()
        for _ in range(bisect_steps):
            mid = 0.5 * (lo_sel + hi_sel)
            q_mid = q_sel + mid[:, None] * d_sel
            q_mid, _ = clamp_q(q_mid, q_min, q_max)
            g_mid = oracle_visibility_g(x, q_mid, oracle)
            outside = g_mid < 0.0
            lo_sel = torch.where(outside, mid, lo_sel)
            hi_sel = torch.where(outside, hi_sel, mid)
        crossing[idx] = hi_sel

    return crossing, reached


def summarize(arr: Dict[str, np.ndarray], mask: np.ndarray) -> Dict:
    mask = np.asarray(mask, dtype=bool)
    out = {"count": int(mask.sum())}
    if not np.any(mask):
        return out

    def v(k):
        return arr[k][mask]

    out.update({
        "f": stats(v("f")),
        "g": stats(v("g")),
        "grad_norm": stats(v("grad_norm")),
        "eikonal_abs_error": stats(v("eikonal_abs_error")),
        "local_levelset_distance_abs_f_over_gradnorm": stats(v("local_dist")),
        "sdf_projection_abs_f_residual": stats(v("f_sdf_abs")),
        "newton_projection_abs_f_residual": stats(v("f_newton_abs")),
        "sdf_projection_abs_oracle_g": stats(v("g_sdf_abs")),
        "newton_projection_abs_oracle_g": stats(v("g_newton_abs")),
        "oracle_boundary_rate_sdf_eps003": float(np.mean(v("g_sdf_abs") < 0.03)),
        "oracle_boundary_rate_newton_eps003": float(np.mean(v("g_newton_abs") < 0.03)),
        "learned_boundary_rate_sdf_eps003": float(np.mean(v("f_sdf_abs") < 0.03)),
        "learned_boundary_rate_newton_eps003": float(np.mean(v("f_newton_abs") < 0.03)),
        "sdf_projection_clamp_rate": float(np.mean(v("clamped_sdf") > 0.5)),
        "newton_projection_clamp_rate": float(np.mean(v("clamped_newton") > 0.5)),
        "delta_f_after_normalized_ascent": stats(v("delta_f_ascent")),
        "delta_g_after_normalized_ascent": stats(v("delta_g_ascent")),
        "oracle_g_improves_after_ascent_rate": float(np.mean(v("delta_g_ascent") > 0.0)),
    })

    cross = v("oracle_crossing_distance")
    valid = np.isfinite(cross)
    out["oracle_crossing_found_rate"] = float(np.mean(valid))
    if np.any(valid):
        raw = np.abs(v("f"))[valid]
        local = v("local_dist")[valid]
        crossv = cross[valid]
        out["oracle_crossing_distance_along_learned_gradient"] = stats(crossv)
        out["corr_raw_abs_f_vs_oracle_crossing_distance"] = corr(raw, crossv)
        out["corr_local_distance_vs_oracle_crossing_distance"] = corr(local, crossv)
        out["raw_abs_f_over_crossing_distance"] = stats(raw / np.maximum(crossv, 1e-8))
        out["local_distance_over_crossing_distance"] = stats(local / np.maximum(crossv, 1e-8))
    return out


def make_plots(output_dir: str, arr: Dict[str, np.ndarray]):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(arr["grad_norm"], bins=100, density=True)
    plt.axvline(1.0, linestyle="--", linewidth=1.2)
    plt.xlabel(r"$\|\nabla_q f_V\|_2$")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_norm_hist.png"), dpi=180)
    plt.close()

    m = (arr["g"] < 0.0) & np.isfinite(arr["oracle_crossing_distance"])
    idx = np.where(m)[0]
    if len(idx) > 0:
        if len(idx) > 5000:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx, 5000, replace=False)
        cross = arr["oracle_crossing_distance"][idx]
        raw = np.abs(arr["f"][idx])
        local = arr["local_dist"][idx]

        lim = float(max(np.max(cross), np.max(raw), 1e-3))
        plt.figure(figsize=(6, 6))
        plt.scatter(cross, raw, s=5, alpha=0.3)
        plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0)
        plt.xlabel("oracle crossing distance along learned gradient [rad]")
        plt.ylabel("raw |f_V|")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "raw_f_vs_oracle_crossing_distance.png"), dpi=180)
        plt.close()

        lim = float(max(np.max(cross), np.max(local), 1e-3))
        plt.figure(figsize=(6, 6))
        plt.scatter(cross, local, s=5, alpha=0.3)
        plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0)
        plt.xlabel("oracle crossing distance along learned gradient [rad]")
        plt.ylabel(r"$|f_V|/\|\nabla f_V\|$ [rad approx.]")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "local_distance_vs_oracle_crossing_distance.png"), dpi=180)
        plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(arr["delta_f_ascent"], arr["delta_g_ascent"], s=5, alpha=0.25)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.axvline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel(r"$\Delta f_V$ after normalized gradient ascent")
    plt.ylabel(r"$\Delta g$ true oracle")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learned_vs_oracle_ascent_delta.png"), dpi=180)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt")
    p.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--num-targets", type=int, default=16)
    p.add_argument("--q-per-target", type=int, default=128)
    p.add_argument("--x-source", choices=["workspace", "unit_box"], default="workspace")
    p.add_argument("--fixed-target", default="", help="Optional 'x,y,z'; if supplied, num-targets repeats the same point.")
    p.add_argument("--x-min", default="-0.95,-0.95,0.0")
    p.add_argument("--x-max", default="0.95,0.95,1.15")
    p.add_argument("--q-min", default="-3.14,-2.30,-3.14,-2.65,-3.14,-3.14,-1.20")
    p.add_argument("--q-max", default="3.14,2.30,3.14,2.65,3.14,3.14,1.20")

    p.add_argument("--ascent-step", type=float, default=0.05)
    p.add_argument("--crossing-max-distance", type=float, default=0.50)
    p.add_argument("--crossing-coarse-steps", type=int, default=20)
    p.add_argument("--crossing-bisect-steps", type=int, default=10)
    p.add_argument("--max-crossing-samples-per-target", type=int, default=128)

    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--output-dir", default="outputs/viscdf_checkpoint_only_geometry")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    q_min = torch.as_tensor(parse_vec(args.q_min, 7), device=device)
    q_max = torch.as_tensor(parse_vec(args.q_max, 7), device=device)

    ckpt = torch_load_checkpoint(resolve(args.checkpoint), device)
    model, ckpt_args = build_model_from_checkpoint(ckpt, device)
    oracle = PinocchioFOVOracle(
        resolve(args.urdf),
        DEFAULT_JOINT_NAMES,
        DEFAULT_SENSOR_FRAMES,
        args.horizontal_fov_deg,
        args.vertical_fov_deg,
        args.z_min,
        args.z_max,
        args.delta,
        base_frame="base_link",
    )

    storage: Dict[str, List[np.ndarray]] = {}
    def add(name: str, t: torch.Tensor):
        storage.setdefault(name, []).append(t.detach().cpu().numpy().reshape(-1))

    for ti in range(args.num_targets):
        x = sample_x(args, device, ti)
        q = q_min[None, :] + torch.rand((args.q_per_target, 7), device=device) * (q_max - q_min)[None, :]

        f, grad, _ = model_value_and_grad_q(x, q, model)
        g = oracle_visibility_g(x, q, oracle)
        grad_norm = torch.linalg.norm(grad, dim=-1)
        direction = grad / torch.clamp(grad_norm[:, None], min=EPS)
        local_dist = torch.abs(f) / torch.clamp(grad_norm, min=EPS)

        q_sdf_raw = q - f[:, None] * grad
        q_sdf, clamped_sdf = clamp_q(q_sdf_raw, q_min, q_max)
        q_newton_raw = q - f[:, None] * grad / torch.clamp((grad_norm * grad_norm)[:, None], min=EPS)
        q_newton, clamped_newton = clamp_q(q_newton_raw, q_min, q_max)

        f_sdf = model_value(x, q_sdf, model)
        f_newton = model_value(x, q_newton, model)
        g_sdf = oracle_visibility_g(x, q_sdf, oracle)
        g_newton = oracle_visibility_g(x, q_newton, oracle)

        q_ascent_raw = q + args.ascent_step * direction
        q_ascent, _ = clamp_q(q_ascent_raw, q_min, q_max)
        f_ascent = model_value(x, q_ascent, model)
        g_ascent = oracle_visibility_g(x, q_ascent, oracle)

        crossing = torch.full_like(f, float("nan"))
        outside_idx = torch.nonzero(g < 0.0, as_tuple=False).squeeze(-1)
        if outside_idx.numel() > args.max_crossing_samples_per_target:
            outside_idx = outside_idx[: args.max_crossing_samples_per_target]
        if outside_idx.numel() > 0:
            cross_sub, _ = oracle_crossing_along_gradient(
                x=x,
                q0=q[outside_idx],
                direction=direction[outside_idx],
                g0=g[outside_idx],
                oracle=oracle,
                q_min=q_min,
                q_max=q_max,
                max_distance=args.crossing_max_distance,
                coarse_steps=args.crossing_coarse_steps,
                bisect_steps=args.crossing_bisect_steps,
            )
            crossing[outside_idx] = cross_sub

        add("f", f)
        add("g", g)
        add("grad_norm", grad_norm)
        add("eikonal_abs_error", torch.abs(grad_norm - 1.0))
        add("local_dist", local_dist)
        add("f_sdf_abs", torch.abs(f_sdf))
        add("f_newton_abs", torch.abs(f_newton))
        add("g_sdf_abs", torch.abs(g_sdf))
        add("g_newton_abs", torch.abs(g_newton))
        add("clamped_sdf", clamped_sdf.float())
        add("clamped_newton", clamped_newton.float())
        add("delta_f_ascent", f_ascent - f)
        add("delta_g_ascent", g_ascent - g)
        add("oracle_crossing_distance", crossing)

        sign_agree = ((f >= 0.0) == (g >= 0.0)).float().mean().item()
        outside = g < 0.0
        improve = ((g_ascent - g)[outside] > 0.0).float().mean().item() if torch.any(outside) else float("nan")
        print(
            f"[target {ti+1:02d}/{args.num_targets}] x={x.detach().cpu().numpy().reshape(-1).round(4).tolist()} "
            f"sign_agree={sign_agree:.3f} grad_norm_mean={grad_norm.mean().item():.3f} "
            f"p95={torch.quantile(grad_norm, 0.95).item():.3f} outside_g_improve_rate={improve:.3f}"
        )

    arr = {k: np.concatenate(v) for k, v in storage.items()}
    cohorts = {
        "all": np.ones_like(arr["g"], dtype=bool),
        "oracle_outside": arr["g"] < 0.0,
        "oracle_far_outside_g_le_-0.05": arr["g"] <= -0.05,
        "oracle_near_boundary_abs_g_le_0.03": np.abs(arr["g"]) <= 0.03,
        "oracle_inside": arr["g"] >= 0.0,
        "oracle_deep_inside_g_ge_0.05": arr["g"] >= 0.05,
    }

    summary = {
        "experiment": "viscdf_checkpoint_only_geometry",
        "checkpoint": resolve(args.checkpoint),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_training_weights": {
            k: ckpt_args.get(k)
            for k in ["weight_sdf", "weight_eikonal", "weight_tension", "weight_grad", "near_zero_ratio", "near_zero_std"]
            if k in ckpt_args
        },
        "num_samples": int(arr["g"].size),
        "sign_agreement_all": float(np.mean((arr["f"] >= 0.0) == (arr["g"] >= 0.0))),
        "cohorts": {name: summarize(arr, mask) for name, mask in cohorts.items()},
        "note": (
            "oracle_crossing_distance is distance along the learned normalized gradient ray, not global shortest C-space distance. "
            "No training NPZ/q0 library is used."
        ),
    }

    out_dir = resolve(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w") as fobj:
        json.dump(summary, fobj, indent=2)
    np.savez_compressed(os.path.join(out_dir, "samples.npz"), **arr)
    make_plots(out_dir, arr)

    print("\n================ FINAL SUMMARY ================")
    print("output_dir:", out_dir)
    print("sign agreement:", summary["sign_agreement_all"])
    print("gradient norm / all:", json.dumps(summary["cohorts"]["all"]["grad_norm"], indent=2))
    outside = summary["cohorts"]["oracle_outside"]
    print("outside oracle-g improvement rate after normalized ascent:", outside.get("oracle_g_improves_after_ascent_rate"))
    print("outside crossing-found rate:", outside.get("oracle_crossing_found_rate"))
    print("corr raw |f| vs oracle crossing distance:", outside.get("corr_raw_abs_f_vs_oracle_crossing_distance"))
    print("corr |f|/||grad|| vs oracle crossing distance:", outside.get("corr_local_distance_vs_oracle_crossing_distance"))
    print("oracle boundary rate after q-f*grad:", outside.get("oracle_boundary_rate_sdf_eps003"))
    print("oracle boundary rate after Newton projection:", outside.get("oracle_boundary_rate_newton_eps003"))


if __name__ == "__main__":
    main()
