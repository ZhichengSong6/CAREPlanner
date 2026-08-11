#!/usr/bin/env python3
"""
CAREPlanner Visibility CDF: inside-gradient diagnostic.

Goal
----
Diagnose where the learned configuration-space gradient is useful relative to
an exact FOV-only oracle gradient, especially after the target is already near
or inside the FOV.

Samples are BALANCED across true-oracle margin strata:

    far_outside       : g < -0.03
    outside_boundary  : -0.03 <= g < 0
    shallow_inside    : 0 <= g < 0.01
    middle_inside     : 0.01 <= g < 0.03
    deep_inside       : g >= 0.03

For each selected matched (x,q) pair this script measures:
  * learned f(x,q)
  * true oracle g(x,q)
  * grad_q f from autograd
  * grad_q g from joint-limit-aware finite differences
  * cosine / angle between grad_q f and grad_q g
  * ||grad_q f|| and ||grad_q g||
  * one-step normalized learned ascent for configurable step sizes:
        P(Delta g > 0), E[Delta g], median Delta g,
        P(f goes up while g goes down), clamp rate
  * an oracle-gradient one-step baseline with the same step size.

The evaluator reuses the already validated standalone model / URDF / FOV
backend from evaluate_direct_vs_projection_ascent.py.  It does NOT import old
training, data-generation, self-occlusion, or validation helpers.

Recommended workflow
--------------------
1) Smoke test with --samples-per-bin 100 and --batch-q 1000.
2) Formal run with 5,000-10,000 samples/bin.
3) Upload the JSON + NPZ for analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PinocchioFOVOracle,
    build_model_from_checkpoint,
    load_npz_data,
    normalize_checkpoint_args,
    sample_q,
    sample_x,
    torch_load_checkpoint,
)


BIN_ORDER = [
    "far_outside",
    "outside_boundary",
    "shallow_inside",
    "middle_inside",
    "deep_inside",
]

BIN_DEFS = {
    "far_outside": {"low": None, "high": -0.03, "low_inclusive": False, "high_inclusive": False},
    "outside_boundary": {"low": -0.03, "high": 0.0, "low_inclusive": True, "high_inclusive": False},
    "shallow_inside": {"low": 0.0, "high": 0.01, "low_inclusive": True, "high_inclusive": False},
    "middle_inside": {"low": 0.01, "high": 0.03, "low_inclusive": True, "high_inclusive": False},
    "deep_inside": {"low": 0.03, "high": None, "low_inclusive": True, "high_inclusive": False},
}


def parse_float_list(text: str) -> List[float]:
    values = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def threshold_key(v: float) -> str:
    return f"{float(v):g}"


def clamp_q(q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(q, q_max[None, :]), q_min[None, :])


def bin_mask_torch(g: torch.Tensor, name: str) -> torch.Tensor:
    d = BIN_DEFS[name]
    mask = torch.ones_like(g, dtype=torch.bool)
    if d["low"] is not None:
        mask &= g >= float(d["low"]) if d["low_inclusive"] else g > float(d["low"])
    if d["high"] is not None:
        mask &= g <= float(d["high"]) if d["high_inclusive"] else g < float(d["high"])
    return mask


def bin_mask_np(g: np.ndarray, name: str) -> np.ndarray:
    d = BIN_DEFS[name]
    mask = np.ones(g.shape, dtype=bool)
    if d["low"] is not None:
        mask &= g >= float(d["low"]) if d["low_inclusive"] else g > float(d["low"])
    if d["high"] is not None:
        mask &= g <= float(d["high"]) if d["high_inclusive"] else g < float(d["high"])
    return mask


def stats(values: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    q = np.quantile(values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p10": float(q[2]),
        "p25": float(q[3]),
        "p50": float(q[4]),
        "p75": float(q[5]),
        "p90": float(q[6]),
        "p95": float(q[7]),
        "p99": float(q[8]),
        "max": float(np.max(values)),
    }


@torch.no_grad()
def paired_oracle_visibility_g(
    x: torch.Tensor,
    q: torch.Tensor,
    oracle: PinocchioFOVOracle,
) -> torch.Tensor:
    """Exact paired oracle g for x[i], q[i], avoiding an NxN cross-product."""
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected x [N,3], got {tuple(x.shape)}")
    if q.ndim != 2 or q.shape[1] != len(oracle.joint_names):
        raise ValueError(f"Expected q [N,{len(oracle.joint_names)}], got {tuple(q.shape)}")
    if x.shape[0] != q.shape[0]:
        raise ValueError("paired oracle requires x and q to have the same batch size")

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
        px = point_sensor[:, 0]
        py = point_sensor[:, 1]
        pz = point_sensor[:, 2]
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


def paired_model_value_and_grad_q(
    x: torch.Tensor,
    q: torch.Tensor,
    model: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Paired f(x[i],q[i]) and grad_q f for a batch of matched samples."""
    q_grad = q.detach().clone().requires_grad_(True)
    inputs = torch.cat([x.detach(), q_grad], dim=-1)
    f = model(inputs).reshape(-1)
    grad = torch.autograd.grad(
        outputs=f,
        inputs=q_grad,
        grad_outputs=torch.ones_like(f),
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
    return f.detach(), grad.detach()


@torch.no_grad()
def paired_model_value(x: torch.Tensor, q: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    return model(torch.cat([x, q], dim=-1)).reshape(-1)


def finite_difference_oracle_gradient_paired(
    x: torch.Tensor,
    q: torch.Tensor,
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Joint-limit-aware finite-difference gradient of paired exact oracle g."""
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


def normalized_step(
    q: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    grad_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalized gradient step, then joint-limit clamp.

    Returns q_new, actual_step_norm, clamped_event.
    Samples with ||grad|| <= grad_eps do not move.
    """
    norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
    valid = norm > grad_eps
    direction = torch.where(valid, grad / torch.clamp(norm, min=grad_eps), torch.zeros_like(grad))
    q_raw = q + float(step_size) * direction
    q_new = clamp_q(q_raw, q_min, q_max)
    actual = torch.linalg.norm(q_new - q, dim=-1)
    clamped = torch.any(torch.abs(q_new - q_raw) > 1e-8, dim=-1)
    return q_new, actual, clamped


def collect_balanced_samples(
    data: Mapping[str, np.ndarray],
    oracle: PinocchioFOVOracle,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    device: torch.device,
    x_source: str,
    samples_per_bin: int,
    batch_q: int,
    max_sampling_trials: int,
    print_every: int,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, int]]:
    """Rejection-sample balanced (x,q) pairs from the requested true-g strata."""
    xs: Dict[str, List[torch.Tensor]] = {name: [] for name in BIN_ORDER}
    qs: Dict[str, List[torch.Tensor]] = {name: [] for name in BIN_ORDER}
    gs: Dict[str, List[torch.Tensor]] = {name: [] for name in BIN_ORDER}
    counts = {name: 0 for name in BIN_ORDER}
    raw_seen = {name: 0 for name in BIN_ORDER}

    start = time.time()
    trial = 0
    while min(counts.values()) < samples_per_bin:
        trial += 1
        if trial > max_sampling_trials:
            missing = {k: samples_per_bin - v for k, v in counts.items() if v < samples_per_bin}
            raise RuntimeError(
                f"Could not fill all bins after {max_sampling_trials} x-trials. "
                f"Missing: {missing}. Increase --max-sampling-trials or --batch-q."
            )

        x = sample_x(data, device, x_source)
        q = sample_q(q_min, q_max, batch_q, device)

        # One x against many q: use the backend's exact cross-product oracle path.
        raw_margin, _ = oracle.signed_fov_margins(x, q)
        g = torch.max(raw_margin - oracle.delta, dim=-1).values.squeeze(0)

        for name in BIN_ORDER:
            m = bin_mask_torch(g, name)
            idx_all = torch.nonzero(m, as_tuple=False).squeeze(-1)
            raw_seen[name] += int(idx_all.numel())
            need = samples_per_bin - counts[name]
            if need <= 0 or idx_all.numel() == 0:
                continue
            take = idx_all[:need]
            n = int(take.numel())
            xs[name].append(x.expand(n, -1).detach().cpu())
            qs[name].append(q[take].detach().cpu())
            gs[name].append(g[take].detach().cpu())
            counts[name] += n

        if trial == 1 or trial % print_every == 0 or min(counts.values()) >= samples_per_bin:
            ctext = " ".join(f"{k}={counts[k]}/{samples_per_bin}" for k in BIN_ORDER)
            print(f"sampling trial {trial:04d}: {ctext} elapsed={time.time()-start:.1f}s", flush=True)

    result: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in BIN_ORDER:
        result[name] = {
            "x": torch.cat(xs[name], dim=0)[:samples_per_bin],
            "q": torch.cat(qs[name], dim=0)[:samples_per_bin],
            "g": torch.cat(gs[name], dim=0)[:samples_per_bin],
        }
    return result, raw_seen


def evaluate(args) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_npz_data(args.data)
    ckpt = torch_load_checkpoint(args.checkpoint, device)
    model, ckpt_args = build_model_from_checkpoint(ckpt, device)
    ckpt_args = normalize_checkpoint_args(ckpt_args)
    model.eval()

    q_min = torch.as_tensor(data["q_min"], device=device, dtype=torch.float32)
    q_max = torch.as_tensor(data["q_max"], device=device, dtype=torch.float32)

    horizontal_fov_deg = float(ckpt_args.get("horizontal_fov_deg", args.horizontal_fov_deg))
    vertical_fov_deg = float(ckpt_args.get("vertical_fov_deg", args.vertical_fov_deg))
    z_min = float(ckpt_args.get("z_min", args.z_min))
    z_max = float(ckpt_args.get("z_max", args.z_max))
    delta = float(ckpt_args.get("delta", args.delta))

    oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
        z_min=z_min,
        z_max=z_max,
        delta=delta,
    )

    print("\n=== Inside-gradient diagnostic ===")
    print(f"checkpoint:             {args.checkpoint}")
    print(f"data:                   {args.data}")
    print(f"urdf:                   {args.urdf}")
    print(f"device:                 {device}")
    print(f"seed:                   {args.seed}")
    print(f"x_source:               {args.x_source}")
    print(f"samples/bin:            {args.samples_per_bin}")
    print(f"sampling q batch:       {args.batch_q}")
    print(f"gradient batch:         {args.gradient_batch_size}")
    print(f"fd_eps:                 {args.fd_eps}")
    print(f"one-step sizes:         {args.step_sizes}")
    print(f"gradient norm epsilon:  {args.grad_norm_eps}")
    print("==================================\n")

    overall_start = time.time()
    collected, raw_seen = collect_balanced_samples(
        data=data,
        oracle=oracle,
        q_min=q_min,
        q_max=q_max,
        device=device,
        x_source=args.x_source,
        samples_per_bin=args.samples_per_bin,
        batch_q=args.batch_q,
        max_sampling_trials=args.max_sampling_trials,
        print_every=args.sampling_print_every,
    )

    # Flatten balanced bins into one paired dataset.
    x_all_cpu = torch.cat([collected[name]["x"] for name in BIN_ORDER], dim=0)
    q_all_cpu = torch.cat([collected[name]["q"] for name in BIN_ORDER], dim=0)
    g_sample_cpu = torch.cat([collected[name]["g"] for name in BIN_ORDER], dim=0)
    bin_id = np.concatenate([
        np.full((args.samples_per_bin,), i, dtype=np.int16) for i, _ in enumerate(BIN_ORDER)
    ])

    n_total = x_all_cpu.shape[0]
    f_chunks: List[np.ndarray] = []
    g_chunks: List[np.ndarray] = []
    grad_f_chunks: List[np.ndarray] = []
    grad_g_chunks: List[np.ndarray] = []

    for start in range(0, n_total, args.gradient_batch_size):
        end = min(start + args.gradient_batch_size, n_total)
        x_b = x_all_cpu[start:end].to(device=device, dtype=torch.float32)
        q_b = q_all_cpu[start:end].to(device=device, dtype=torch.float32)

        f_b, grad_f_b = paired_model_value_and_grad_q(x_b, q_b, model)
        g_b = paired_oracle_visibility_g(x_b, q_b, oracle)
        grad_g_b = finite_difference_oracle_gradient_paired(
            x_b, q_b, oracle, q_min, q_max, args.fd_eps
        )

        f_chunks.append(f_b.cpu().numpy().astype(np.float32))
        g_chunks.append(g_b.cpu().numpy().astype(np.float32))
        grad_f_chunks.append(grad_f_b.cpu().numpy().astype(np.float32))
        grad_g_chunks.append(grad_g_b.cpu().numpy().astype(np.float32))

        batch_index = start // args.gradient_batch_size + 1
        total_batches = math.ceil(n_total / args.gradient_batch_size)
        if batch_index == 1 or batch_index % args.print_every == 0 or end == n_total:
            print(
                f"gradient batch {batch_index:04d}/{total_batches}: "
                f"samples={end}/{n_total} elapsed={time.time()-overall_start:.1f}s",
                flush=True,
            )

    f_np = np.concatenate(f_chunks)
    g_np = np.concatenate(g_chunks)
    grad_f_np = np.concatenate(grad_f_chunks)
    grad_g_np = np.concatenate(grad_g_chunks)

    # Sampling-vs-paired oracle consistency check.
    g_sample_np = g_sample_cpu.numpy().astype(np.float32)
    g_recheck_err = float(np.max(np.abs(g_np - g_sample_np)))
    if g_recheck_err > 5e-5:
        raise RuntimeError(
            f"Paired oracle consistency check failed: max |g_paired-g_sample|={g_recheck_err:.3e}"
        )

    grad_f_norm = np.linalg.norm(grad_f_np, axis=1)
    grad_g_norm = np.linalg.norm(grad_g_np, axis=1)
    valid_cos = (grad_f_norm > args.grad_norm_eps) & (grad_g_norm > args.grad_norm_eps)
    cosine = np.full((n_total,), np.nan, dtype=np.float64)
    dot = np.sum(grad_f_np * grad_g_np, axis=1)
    cosine[valid_cos] = dot[valid_cos] / (
        grad_f_norm[valid_cos] * grad_g_norm[valid_cos]
    )
    cosine[valid_cos] = np.clip(cosine[valid_cos], -1.0, 1.0)
    angle_deg = np.full((n_total,), np.nan, dtype=np.float64)
    angle_deg[valid_cos] = np.degrees(np.arccos(cosine[valid_cos]))

    result: Dict = {
        "config": {
            "checkpoint": args.checkpoint,
            "data": args.data,
            "urdf": args.urdf,
            "device": str(device),
            "seed": args.seed,
            "x_source": args.x_source,
            "samples_per_bin": args.samples_per_bin,
            "total_selected_samples": int(n_total),
            "batch_q": args.batch_q,
            "gradient_batch_size": args.gradient_batch_size,
            "fd_eps": args.fd_eps,
            "step_sizes": [float(v) for v in args.step_sizes],
            "grad_norm_eps": args.grad_norm_eps,
            "fov": {
                "horizontal_fov_deg": horizontal_fov_deg,
                "vertical_fov_deg": vertical_fov_deg,
                "z_min": z_min,
                "z_max": z_max,
                "delta": delta,
            },
            "bin_definitions": BIN_DEFS,
        },
        "sampling": {
            "raw_candidates_seen_per_bin": {k: int(v) for k, v in raw_seen.items()},
            "paired_oracle_recheck_max_abs_error": g_recheck_err,
        },
        "bins": {},
    }

    print("\n=== Gradient alignment by true-oracle margin bin ===")
    print("bin                N      mean_g   cos_mean  cos_med   angle_med  cos>0   |df|med   |dg|med  valid")

    # Per-bin gradient summaries.
    for i, name in enumerate(BIN_ORDER):
        m = bin_id == i
        vm = m & valid_cos
        entry = {
            "count": int(np.sum(m)),
            "g_stats": stats(g_np, m),
            "f_stats": stats(f_np, m),
            "gradient": {
                "valid_cosine_rate": float(np.mean(valid_cos[m])),
                "learned_grad_norm_stats": stats(grad_f_norm, m),
                "oracle_grad_norm_stats": stats(grad_g_norm, m),
                "cosine_stats": stats(cosine, vm),
                "angle_deg_stats": stats(angle_deg, vm),
                "cosine_positive_rate_valid": float(np.mean(cosine[vm] > 0.0)) if np.any(vm) else float("nan"),
                "cosine_ge_0p5_rate_valid": float(np.mean(cosine[vm] >= 0.5)) if np.any(vm) else float("nan"),
                "cosine_ge_0p8_rate_valid": float(np.mean(cosine[vm] >= 0.8)) if np.any(vm) else float("nan"),
                "learned_zero_grad_rate": float(np.mean(grad_f_norm[m] <= args.grad_norm_eps)),
                "oracle_zero_grad_rate": float(np.mean(grad_g_norm[m] <= args.grad_norm_eps)),
            },
            "one_step": {},
        }
        result["bins"][name] = entry

        cs = entry["gradient"]["cosine_stats"]
        ang = entry["gradient"]["angle_deg_stats"]
        print(
            f"{name:18s} {entry['count']:6d}  {entry['g_stats']['mean']:+.5f}   "
            f"{cs['mean']:+.4f}    {cs['p50']:+.4f}     {ang['p50']:7.2f}   "
            f"{entry['gradient']['cosine_positive_rate_valid']:.4f}   "
            f"{entry['gradient']['learned_grad_norm_stats']['p50']:.4f}    "
            f"{entry['gradient']['oracle_grad_norm_stats']['p50']:.4f}   "
            f"{entry['gradient']['valid_cosine_rate']:.4f}"
        )

    # One-step diagnostics for every requested step size.  Process in batches so
    # large formal runs do not duplicate all sample tensors on GPU.
    one_step_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for step in args.step_sizes:
        key = threshold_key(step)
        tmp = {
            "learned_delta_g": [],
            "learned_delta_f": [],
            "learned_actual_step": [],
            "learned_clamped": [],
            "oracle_delta_g": [],
            "oracle_actual_step": [],
            "oracle_clamped": [],
        }
        for start in range(0, n_total, args.gradient_batch_size):
            end = min(start + args.gradient_batch_size, n_total)
            x_b = x_all_cpu[start:end].to(device=device, dtype=torch.float32)
            q_b = q_all_cpu[start:end].to(device=device, dtype=torch.float32)
            gf_b = torch.as_tensor(grad_f_np[start:end], device=device)
            gg_b = torch.as_tensor(grad_g_np[start:end], device=device)
            f0_b = torch.as_tensor(f_np[start:end], device=device)
            g0_b = torch.as_tensor(g_np[start:end], device=device)

            q_l, l_actual, l_clamp = normalized_step(
                q_b, gf_b, step, q_min, q_max, args.grad_norm_eps
            )
            f_l = paired_model_value(x_b, q_l, model)
            g_l = paired_oracle_visibility_g(x_b, q_l, oracle)

            q_o, o_actual, o_clamp = normalized_step(
                q_b, gg_b, step, q_min, q_max, args.grad_norm_eps
            )
            g_o = paired_oracle_visibility_g(x_b, q_o, oracle)

            tmp["learned_delta_g"].append((g_l - g0_b).cpu().numpy())
            tmp["learned_delta_f"].append((f_l - f0_b).cpu().numpy())
            tmp["learned_actual_step"].append(l_actual.cpu().numpy())
            tmp["learned_clamped"].append(l_clamp.cpu().numpy())
            tmp["oracle_delta_g"].append((g_o - g0_b).cpu().numpy())
            tmp["oracle_actual_step"].append(o_actual.cpu().numpy())
            tmp["oracle_clamped"].append(o_clamp.cpu().numpy())

        one_step_arrays[key] = {
            k: np.concatenate(v).astype(np.float32 if "clamped" not in k else np.bool_)
            for k, v in tmp.items()
        }

        for i, name in enumerate(BIN_ORDER):
            m = bin_id == i
            a = one_step_arrays[key]
            ld_g = a["learned_delta_g"]
            ld_f = a["learned_delta_f"]
            od_g = a["oracle_delta_g"]
            learned_f_up_g_down = (ld_f > 0.0) & (ld_g < 0.0)
            result["bins"][name]["one_step"][key] = {
                "step_size": float(step),
                "learned": {
                    "delta_g_stats": stats(ld_g, m),
                    "delta_f_stats": stats(ld_f, m),
                    "delta_g_positive_rate": float(np.mean(ld_g[m] > 0.0)),
                    "delta_g_negative_rate": float(np.mean(ld_g[m] < 0.0)),
                    "f_up_g_down_rate": float(np.mean(learned_f_up_g_down[m])),
                    "actual_step_norm_stats": stats(a["learned_actual_step"], m),
                    "clamp_rate": float(np.mean(a["learned_clamped"][m])),
                },
                "oracle_gradient_baseline": {
                    "delta_g_stats": stats(od_g, m),
                    "delta_g_positive_rate": float(np.mean(od_g[m] > 0.0)),
                    "delta_g_negative_rate": float(np.mean(od_g[m] < 0.0)),
                    "actual_step_norm_stats": stats(a["oracle_actual_step"], m),
                    "clamp_rate": float(np.mean(a["oracle_clamped"][m])),
                },
            }

    print("\n=== One-step true-margin improvement ===")
    print("bin                step   learned dg>0  mean_dg    f_up_g_down   oracle dg>0  oracle mean_dg")
    for name in BIN_ORDER:
        for step in args.step_sizes:
            e = result["bins"][name]["one_step"][threshold_key(step)]
            print(
                f"{name:18s} {step:5.3f}      "
                f"{e['learned']['delta_g_positive_rate']:.4f}    "
                f"{e['learned']['delta_g_stats']['mean']:+.5f}      "
                f"{e['learned']['f_up_g_down_rate']:.4f}         "
                f"{e['oracle_gradient_baseline']['delta_g_positive_rate']:.4f}       "
                f"{e['oracle_gradient_baseline']['delta_g_stats']['mean']:+.5f}"
            )

    # A compact diagnosis-oriented summary at the top level.
    result["compact_summary"] = {}
    for name in BIN_ORDER:
        ge = result["bins"][name]["gradient"]
        result["compact_summary"][name] = {
            "mean_g": result["bins"][name]["g_stats"]["mean"],
            "cosine_mean": ge["cosine_stats"]["mean"],
            "cosine_median": ge["cosine_stats"]["p50"],
            "angle_median_deg": ge["angle_deg_stats"]["p50"],
            "cosine_positive_rate_valid": ge["cosine_positive_rate_valid"],
            "learned_grad_norm_median": ge["learned_grad_norm_stats"]["p50"],
            "oracle_grad_norm_median": ge["oracle_grad_norm_stats"]["p50"],
            "one_step": {
                threshold_key(step): {
                    "learned_delta_g_positive_rate": result["bins"][name]["one_step"][threshold_key(step)]["learned"]["delta_g_positive_rate"],
                    "learned_mean_delta_g": result["bins"][name]["one_step"][threshold_key(step)]["learned"]["delta_g_stats"]["mean"],
                    "f_up_g_down_rate": result["bins"][name]["one_step"][threshold_key(step)]["learned"]["f_up_g_down_rate"],
                    "oracle_delta_g_positive_rate": result["bins"][name]["one_step"][threshold_key(step)]["oracle_gradient_baseline"]["delta_g_positive_rate"],
                    "oracle_mean_delta_g": result["bins"][name]["one_step"][threshold_key(step)]["oracle_gradient_baseline"]["delta_g_stats"]["mean"],
                }
                for step in args.step_sizes
            },
        }

    result["elapsed_sec"] = float(time.time() - overall_start)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(f"\nsaved JSON: {args.output}")

    if args.output_npz:
        npz_dir = os.path.dirname(args.output_npz)
        if npz_dir:
            os.makedirs(npz_dir, exist_ok=True)
        payload = {
            "x": x_all_cpu.numpy().astype(np.float32),
            "q": q_all_cpu.numpy().astype(np.float32),
            "bin_id": bin_id.astype(np.int16),
            "bin_names": np.asarray(BIN_ORDER),
            "f": f_np.astype(np.float32),
            "g": g_np.astype(np.float32),
            "grad_f": grad_f_np.astype(np.float32),
            "grad_g": grad_g_np.astype(np.float32),
            "grad_f_norm": grad_f_norm.astype(np.float32),
            "grad_g_norm": grad_g_norm.astype(np.float32),
            "cosine": cosine.astype(np.float32),
            "angle_deg": angle_deg.astype(np.float32),
            "valid_cosine": valid_cos.astype(np.bool_),
        }
        for step in args.step_sizes:
            key = threshold_key(step).replace(".", "p")
            a = one_step_arrays[threshold_key(step)]
            for field, arr in a.items():
                payload[f"step_{key}_{field}"] = arr
        np.savez_compressed(args.output_npz, **payload)
        print(f"saved NPZ:  {args.output_npz}")

    print(f"elapsed_sec: {result['elapsed_sec']:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="CAREPlanner inside-gradient diagnostic")
    parser.add_argument(
        "--checkpoint",
        default="src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt",
    )
    parser.add_argument(
        "--data",
        default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz",
    )
    parser.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    parser.add_argument(
        "--x-source",
        choices=["unit_box", "random_box", "dataset"],
        default="unit_box",
    )
    parser.add_argument("--samples-per-bin", type=int, default=10000)
    parser.add_argument(
        "--batch-q",
        type=int,
        default=4000,
        help="Random q candidates evaluated per sampled x during balanced collection.",
    )
    parser.add_argument("--max-sampling-trials", type=int, default=20000)
    parser.add_argument("--sampling-print-every", type=int, default=20)
    parser.add_argument("--gradient-batch-size", type=int, default=512)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument(
        "--step-sizes",
        type=parse_float_list,
        default=parse_float_list("0.01,0.025,0.05,0.1"),
    )
    parser.add_argument("--grad-norm-eps", type=float, default=1e-8)
    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=66.0)
    parser.add_argument("--z-min", type=float, default=0.20)
    parser.add_argument("--z-max", type=float, default=0.70)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument(
        "--output",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "inside_gradient_diagnostic/inside_gradient_diagnostic.json"
        ),
    )
    parser.add_argument(
        "--output-npz",
        default=(
            "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/"
            "inside_gradient_diagnostic/inside_gradient_diagnostic_samples.npz"
        ),
    )
    args = parser.parse_args()

    if args.samples_per_bin <= 0:
        parser.error("--samples-per-bin must be positive")
    if args.batch_q <= 0 or args.gradient_batch_size <= 0:
        parser.error("--batch-q and --gradient-batch-size must be positive")
    if args.max_sampling_trials <= 0:
        parser.error("--max-sampling-trials must be positive")
    if args.fd_eps <= 0:
        parser.error("--fd-eps must be positive")
    if any(v <= 0 for v in args.step_sizes):
        parser.error("all --step-sizes must be positive")
    if args.grad_norm_eps <= 0:
        parser.error("--grad-norm-eps must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
