#!/usr/bin/env python3
"""
Metric diagnostic for CAREPlanner visibility CDF and Yiming's released neural CDF.

This script answers two different questions without conflating them:

A) CAREPlanner VisCDF on OUR 7-DoF arm
   - Is |f_V| calibrated to the configuration-space distance target used in training?
   - Is ||grad_q f_V|| close to one?
   - Does grad_q f_V align with the training-target direction?
   - Is the scale-invariant local level-set distance |f|/||grad f|| better calibrated
     than the raw value |f|?
   - If we make one SDF-style projection q-f*grad, does it approach the true FOV
     oracle boundary?  How does this compare with normalized Newton projection
     q-f*grad/||grad||^2 and with the q0-library target projection?

B) OPTIONAL: Yiming Li et al.'s released Franka neural CDF
   - What is the empirical distribution of ||grad_q f_C|| in the released model?
   - How well does the official one-step projection q-d*grad self-project to d=0?
   - How does normalized Newton projection compare?

Important:
  The optional Yiming section uses the released FRANKA model, so its values must NOT
  be directly compared point-by-point with CAREPlanner's custom arm.  It is included
  only to answer whether the released neural CDF actually has unit-ish gradient in
  practice.  A true dual-field same-point comparison requires a collision CDF trained
  for the CAREPlanner arm.

Expected repository location:
  src/care_visibility_cdf/scripts/evaluate_dual_cspace_field_geometry.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from typing import Dict, List, Mapping, Optional, Tuple

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
    torch_load_checkpoint,
)
from train_signed_visibility_cdf_pairwise_replace import (  # noqa: E402
    VisibilityQ0Dataset,
    decode_per_sensor_distance_and_grad,
    union_signed_targets,
)


EPS = 1e-8


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {k: float("nan") for k in ["mean", "std", "min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = torch.linalg.norm(a, dim=-1)
    bn = torch.linalg.norm(b, dim=-1)
    return torch.sum(a * b, dim=-1) / torch.clamp(an * bn, min=EPS)


def _pairwise_model_value_and_grad(
    x: torch.Tensor,
    q_shared: torch.Tensor,
    model: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cartesian x/q pairs. Returns [Bx,Bq] values and [Bx,Bq,7] gradients."""
    bx = x.shape[0]
    bq = q_shared.shape[0]
    x_flat = x[:, None, :].expand(bx, bq, 3).reshape(bx * bq, 3)
    q_flat = (
        q_shared[None, :, :]
        .expand(bx, bq, 7)
        .reshape(bx * bq, 7)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    pred = model(torch.cat([x_flat, q_flat], dim=-1)).reshape(-1)
    grad = torch.autograd.grad(
        pred,
        q_flat,
        grad_outputs=torch.ones_like(pred),
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
    return pred.reshape(bx, bq), grad.reshape(bx, bq, 7)


@torch.no_grad()
def _pairwise_model_value(
    x: torch.Tensor,
    q_pair: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    """Paired x_i with q_{i,j}. q_pair=[Bx,Bq,7], returns [Bx,Bq]."""
    bx, bq, _ = q_pair.shape
    x_flat = x[:, None, :].expand(bx, bq, 3).reshape(bx * bq, 3)
    q_flat = q_pair.reshape(bx * bq, 7)
    return model(torch.cat([x_flat, q_flat], dim=-1)).reshape(bx, bq)


@torch.no_grad()
def _paired_oracle_g(
    x: torch.Tensor,
    q_pair: torch.Tensor,
    oracle: PinocchioFOVOracle,
) -> torch.Tensor:
    """Paired x_i with its Bq configurations. Loops only over Bx."""
    out = []
    for i in range(x.shape[0]):
        raw, _ = oracle.signed_fov_margins(x[i : i + 1], q_pair[i])
        g_sensor = raw - oracle.delta
        g = torch.max(g_sensor, dim=-1).values.squeeze(0)
        out.append(g)
    return torch.stack(out, dim=0)


def _clamp_q_pair(
    q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_clamped = torch.maximum(
        torch.minimum(q, q_max[None, None, :]), q_min[None, None, :]
    )
    changed = torch.any(torch.abs(q_clamped - q) > 1e-10, dim=-1)
    return q_clamped, changed


def _summarize_vis_cohort(arr: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(mask.sum())
    result = {"count": n}
    if n == 0:
        return result

    def v(name):
        return np.asarray(arr[name]).reshape(-1)[mask]

    gt_abs = v("gt_abs")
    pred_abs = v("pred_abs")
    local_dist = v("local_dist")

    valid_dist = gt_abs > 1e-4
    result.update(
        {
            "pred_grad_norm": _quantiles(v("pred_grad_norm")),
            "gt_grad_norm": _quantiles(v("gt_grad_norm")),
            "eikonal_abs_error": _quantiles(v("eikonal_abs_error")),
            "grad_cosine_to_gt": _quantiles(v("grad_cosine")),
            "value_abs_error": _quantiles(v("value_abs_error")),
            "raw_abs_f": _quantiles(pred_abs),
            "gt_abs_distance": _quantiles(gt_abs),
            "local_abs_f_over_gradnorm": _quantiles(local_dist),
            "corr_raw_abs_f_vs_gt_distance": _corrcoef(pred_abs, gt_abs),
            "corr_local_distance_vs_gt_distance": _corrcoef(local_dist, gt_abs),
            "oracle_abs_g_after_sdf_projection": _quantiles(v("oracle_abs_g_sdf")),
            "oracle_abs_g_after_newton_projection": _quantiles(v("oracle_abs_g_newton")),
            "oracle_abs_g_after_gt_projection": _quantiles(v("oracle_abs_g_gt")),
            "learned_abs_f_after_sdf_projection": _quantiles(v("learned_abs_f_sdf")),
            "learned_abs_f_after_newton_projection": _quantiles(v("learned_abs_f_newton")),
            "oracle_boundary_rate_sdf_eps003": float(np.mean(v("oracle_abs_g_sdf") < 0.03)),
            "oracle_boundary_rate_newton_eps003": float(np.mean(v("oracle_abs_g_newton") < 0.03)),
            "oracle_boundary_rate_gt_eps003": float(np.mean(v("oracle_abs_g_gt") < 0.03)),
            "sdf_projection_clamp_rate": float(np.mean(v("clamped_sdf") > 0.5)),
            "newton_projection_clamp_rate": float(np.mean(v("clamped_newton") > 0.5)),
            "gt_projection_clamp_rate": float(np.mean(v("clamped_gt") > 0.5)),
        }
    )
    if np.any(valid_dist):
        ratio_raw = pred_abs[valid_dist] / np.maximum(gt_abs[valid_dist], 1e-8)
        ratio_local = local_dist[valid_dist] / np.maximum(gt_abs[valid_dist], 1e-8)
        result["raw_value_over_gt_distance"] = _quantiles(ratio_raw)
        result["local_distance_over_gt_distance"] = _quantiles(ratio_local)
    return result


def run_viscdf(args, device: torch.device) -> Tuple[Dict, Dict[str, np.ndarray]]:
    print("\n========== CAREPlanner VisCDF metric diagnostic ==========")
    data_path = _resolve(args.data)
    checkpoint_path = _resolve(args.checkpoint)
    urdf_path = _resolve(args.urdf)

    dataset = VisibilityQ0Dataset(data_path, val_count=args.val_count, seed=args.seed)
    q_min, q_max = dataset.q_limits(device)
    sensor_masks = dataset.sensor_masks(device)

    ckpt = torch_load_checkpoint(checkpoint_path, device)
    model, ckpt_args = build_model_from_checkpoint(ckpt, device)

    oracle = PinocchioFOVOracle(
        urdf_path=urdf_path,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        z_min=args.z_min,
        z_max=args.z_max,
        delta=args.delta,
        base_frame="base_link",
    )

    collected: Dict[str, List[np.ndarray]] = {}

    def add(name: str, tensor: torch.Tensor):
        collected.setdefault(name, []).append(tensor.detach().cpu().numpy().reshape(-1))

    for batch_idx in range(args.num_batches):
        x, qlib, valid, _ = dataset.sample_x_batch(
            args.batch_x, split=args.split, device=device
        )
        q = q_min[None, :] + torch.rand(
            (args.batch_q, 7), device=device
        ) * (q_max - q_min)[None, :]

        # Approximate configuration-space distance target used to train VisCDF.
        d_s, grad_d_s, has_sensor = decode_per_sensor_distance_and_grad(
            qlib=qlib,
            valid=valid,
            q_query=q,
            sensor_masks=sensor_masks,
            x_chunk=args.decode_x_chunk,
        )
        raw_margin, sign_s = oracle.signed_fov_margins(x, q)
        target, target_grad = union_signed_targets(
            d_s=d_s,
            grad_d_s=grad_d_s,
            sign_s=sign_s,
            has_sensor=has_sensor,
        )

        pred, pred_grad = _pairwise_model_value_and_grad(x, q, model)
        pred_grad_norm = torch.linalg.norm(pred_grad, dim=-1)
        gt_grad_norm = torch.linalg.norm(target_grad, dim=-1)
        grad_cosine = _cosine(pred_grad, target_grad)

        oracle_g = torch.max(raw_margin - oracle.delta, dim=-1).values
        q_grid = q[None, :, :].expand(args.batch_x, args.batch_q, 7)

        # 1) Exact-SDF-assumption projection: q <- q - f grad.
        sdf_step = pred[..., None] * pred_grad
        q_sdf_raw = q_grid - sdf_step
        q_sdf, clamped_sdf = _clamp_q_pair(q_sdf_raw, q_min, q_max)

        # 2) Scale-invariant normalized Newton step to the local f=0 tangent plane.
        grad_norm_sq = torch.sum(pred_grad * pred_grad, dim=-1)
        newton_step = pred[..., None] * pred_grad / torch.clamp(
            grad_norm_sq[..., None], min=EPS
        )
        q_newton_raw = q_grid - newton_step
        q_newton, clamped_newton = _clamp_q_pair(q_newton_raw, q_min, q_max)

        # 3) Projection using the q0-library signed distance target and direction.
        gt_step = target[..., None] * target_grad
        q_gt_raw = q_grid - gt_step
        q_gt, clamped_gt = _clamp_q_pair(q_gt_raw, q_min, q_max)

        oracle_g_sdf = _paired_oracle_g(x, q_sdf, oracle)
        oracle_g_newton = _paired_oracle_g(x, q_newton, oracle)
        oracle_g_gt = _paired_oracle_g(x, q_gt, oracle)
        learned_f_sdf = _pairwise_model_value(x, q_sdf, model)
        learned_f_newton = _pairwise_model_value(x, q_newton, model)

        local_dist = torch.abs(pred) / torch.clamp(pred_grad_norm, min=EPS)

        add("pred", pred)
        add("pred_abs", torch.abs(pred))
        add("target", target)
        add("gt_abs", torch.abs(target))
        add("oracle_g", oracle_g)
        add("pred_grad_norm", pred_grad_norm)
        add("gt_grad_norm", gt_grad_norm)
        add("eikonal_abs_error", torch.abs(pred_grad_norm - 1.0))
        add("grad_cosine", grad_cosine)
        add("value_abs_error", torch.abs(pred - target))
        add("local_dist", local_dist)
        add("sdf_step_norm", torch.linalg.norm(sdf_step, dim=-1))
        add("newton_step_norm", torch.linalg.norm(newton_step, dim=-1))
        add("gt_step_norm", torch.linalg.norm(gt_step, dim=-1))
        add("oracle_abs_g_sdf", torch.abs(oracle_g_sdf))
        add("oracle_abs_g_newton", torch.abs(oracle_g_newton))
        add("oracle_abs_g_gt", torch.abs(oracle_g_gt))
        add("learned_abs_f_sdf", torch.abs(learned_f_sdf))
        add("learned_abs_f_newton", torch.abs(learned_f_newton))
        add("clamped_sdf", clamped_sdf.float())
        add("clamped_newton", clamped_newton.float())
        add("clamped_gt", clamped_gt.float())

        print(
            f"[viscdf] batch {batch_idx + 1}/{args.num_batches}: "
            f"grad_norm mean={pred_grad_norm.mean().item():.3f} "
            f"p95={torch.quantile(pred_grad_norm.reshape(-1), 0.95).item():.3f} "
            f"cos(gt)={grad_cosine.mean().item():.3f} "
            f"|f-target|={torch.abs(pred-target).mean().item():.4f}"
        )

    arrays = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    g = arrays["oracle_g"]
    cohorts = {
        "all": np.ones_like(g, dtype=bool),
        "outside": g < 0.0,
        "far_outside_g_le_-0.05": g <= -0.05,
        "near_boundary_abs_g_le_0.03": np.abs(g) <= 0.03,
        "inside": g >= 0.0,
        "deep_inside_g_ge_0.05": g >= 0.05,
    }

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_step": int(ckpt.get("step", -1)),
        "checkpoint_args": {
            k: ckpt_args.get(k)
            for k in [
                "weight_sdf",
                "weight_eikonal",
                "weight_tension",
                "weight_grad",
                "near_zero_ratio",
                "near_zero_std",
            ]
            if k in ckpt_args
        },
        "num_pairs": int(arrays["pred"].size),
        "cohorts": {
            name: _summarize_vis_cohort(arrays, mask)
            for name, mask in cohorts.items()
        },
    }
    return summary, arrays


def _load_yiming_mlp(cdf_dir: str, device: torch.device, step: int):
    franka_dir = os.path.join(os.path.abspath(os.path.expanduser(cdf_dir)), "frankaemika")
    mlp_path = os.path.join(franka_dir, "mlp.py")
    model_path = os.path.join(franka_dir, "model_dict.pt")
    if not os.path.isfile(mlp_path) or not os.path.isfile(model_path):
        raise FileNotFoundError(
            "Expected <cdf_dir>/frankaemika/{mlp.py,model_dict.pt}. "
            f"Got cdf_dir={cdf_dir!r}."
        )

    spec = importlib.util.spec_from_file_location("yiming_cdf_mlp", mlp_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    model = module.MLPRegression(
        input_dims=10,
        output_dims=1,
        mlp_layers=[1024, 512, 256, 128, 128],
        skips=[],
        act_fn=torch.nn.ReLU,
        nerf=True,
    ).to(device)
    try:
        state_dicts = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state_dicts = torch.load(model_path, map_location=device)

    available_steps = sorted(int(k) for k in state_dicts.keys())
    selected = step if step >= 0 else available_steps[-1]
    if selected not in state_dicts:
        raise KeyError(
            f"Yiming model step {selected} not found. Available tail={available_steps[-10:]}"
        )
    model.load_state_dict(state_dicts[selected], strict=True)
    model.eval()
    return model, selected


def run_yiming_released(args, device: torch.device) -> Tuple[Dict, Dict[str, np.ndarray]]:
    print("\n========== Yiming released Franka neural CDF diagnostic ==========")
    model, selected_step = _load_yiming_mlp(args.yiming_cdf_dir, device, args.yiming_step)

    # Joint limits copied from Yiming's RDF PandaLayer used by the released CDF.
    q_min = torch.tensor(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        device=device,
        dtype=torch.float32,
    )
    q_max = torch.tensor(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        device=device,
        dtype=torch.float32,
    )

    collected: Dict[str, List[np.ndarray]] = {}

    def add(name: str, tensor: torch.Tensor):
        collected.setdefault(name, []).append(tensor.detach().cpu().numpy().reshape(-1))

    for batch_idx in range(args.yiming_num_batches):
        # Matches the released eval domain: x,y in [-0.5,0.5], z in [0,1].
        x = torch.rand((args.yiming_batch_q, 3), device=device)
        x[:, 0:2] -= 0.5
        q = q_min[None, :] + torch.rand(
            (args.yiming_batch_q, 7), device=device
        ) * (q_max - q_min)[None, :]
        q_leaf = q.detach().clone().requires_grad_(True)
        pred = model(torch.cat([x, q_leaf], dim=-1)).reshape(-1)
        grad = torch.autograd.grad(
            pred,
            q_leaf,
            grad_outputs=torch.ones_like(pred),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        grad_norm = torch.linalg.norm(grad, dim=-1)

        # Projection used by the released code assumes an SDF-like unit gradient.
        q_sdf_raw = q - pred[:, None] * grad
        q_sdf = torch.maximum(torch.minimum(q_sdf_raw, q_max[None, :]), q_min[None, :])
        clamped_sdf = torch.any(torch.abs(q_sdf_raw - q_sdf) > 1e-10, dim=-1)

        # Scale-invariant local Newton projection.
        grad_norm_sq = torch.sum(grad * grad, dim=-1)
        q_newton_raw = q - pred[:, None] * grad / torch.clamp(
            grad_norm_sq[:, None], min=EPS
        )
        q_newton = torch.maximum(
            torch.minimum(q_newton_raw, q_max[None, :]), q_min[None, :]
        )
        clamped_newton = torch.any(
            torch.abs(q_newton_raw - q_newton) > 1e-10, dim=-1
        )

        with torch.no_grad():
            residual_sdf = torch.abs(model(torch.cat([x, q_sdf], dim=-1)).reshape(-1))
            residual_newton = torch.abs(
                model(torch.cat([x, q_newton], dim=-1)).reshape(-1)
            )

        add("pred", pred)
        add("pred_abs", torch.abs(pred))
        add("grad_norm", grad_norm)
        add("eikonal_abs_error", torch.abs(grad_norm - 1.0))
        add("local_dist", torch.abs(pred) / torch.clamp(grad_norm, min=EPS))
        add("residual_sdf", residual_sdf)
        add("residual_newton", residual_newton)
        add("clamped_sdf", clamped_sdf.float())
        add("clamped_newton", clamped_newton.float())

        print(
            f"[yiming-cdf] batch {batch_idx + 1}/{args.yiming_num_batches}: "
            f"grad_norm mean={grad_norm.mean().item():.3f} "
            f"p95={torch.quantile(grad_norm, 0.95).item():.3f} "
            f"|d_after_official_proj|={residual_sdf.mean().item():.4f} "
            f"|d_after_newton|={residual_newton.mean().item():.4f}"
        )

    arrays = {k: np.concatenate(v, axis=0) for k, v in collected.items()}
    summary = {
        "cdf_dir": os.path.abspath(os.path.expanduser(args.yiming_cdf_dir)),
        "model_step": int(selected_step),
        "num_samples": int(arrays["pred"].size),
        "grad_norm": _quantiles(arrays["grad_norm"]),
        "eikonal_abs_error": _quantiles(arrays["eikonal_abs_error"]),
        "abs_cdf_value": _quantiles(arrays["pred_abs"]),
        "local_abs_d_over_gradnorm": _quantiles(arrays["local_dist"]),
        "abs_value_after_official_q_minus_d_grad": _quantiles(arrays["residual_sdf"]),
        "abs_value_after_normalized_newton": _quantiles(arrays["residual_newton"]),
        "official_projection_clamp_rate": float(np.mean(arrays["clamped_sdf"] > 0.5)),
        "newton_projection_clamp_rate": float(np.mean(arrays["clamped_newton"] > 0.5)),
    }
    return summary, arrays


def _make_plots(output_dir: str, vis: Dict[str, np.ndarray], yiming: Optional[Dict[str, np.ndarray]]):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.hist(vis["pred_grad_norm"], bins=100, density=True, alpha=0.75, label="CAREPlanner VisCDF")
    if yiming is not None:
        plt.hist(yiming["grad_norm"], bins=100, density=True, alpha=0.55, label="Yiming released neural CDF")
    plt.axvline(1.0, linestyle="--", linewidth=1.5, label="unit norm")
    plt.xlabel(r"$\|\nabla_q f\|_2$")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_norm_hist.png"), dpi=180)
    plt.close()

    # Raw field magnitude versus the q0-library C-space distance target.
    idx = np.arange(vis["gt_abs"].size)
    if idx.size > 10000:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, 10000, replace=False)
    max_val = float(max(np.max(vis["gt_abs"][idx]), np.max(vis["pred_abs"][idx]), 1e-3))
    plt.figure(figsize=(6, 6))
    plt.scatter(vis["gt_abs"][idx], vis["pred_abs"][idx], s=4, alpha=0.25)
    plt.plot([0, max_val], [0, max_val], linestyle="--", linewidth=1.2)
    plt.xlabel("q0-library target |d_V^GT| [rad]")
    plt.ylabel("network |f_V| [network units]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viscdf_raw_value_vs_gt_distance.png"), dpi=180)
    plt.close()

    max_local = float(max(np.max(vis["gt_abs"][idx]), np.max(vis["local_dist"][idx]), 1e-3))
    plt.figure(figsize=(6, 6))
    plt.scatter(vis["gt_abs"][idx], vis["local_dist"][idx], s=4, alpha=0.25)
    plt.plot([0, max_local], [0, max_local], linestyle="--", linewidth=1.2)
    plt.xlabel("q0-library target |d_V^GT| [rad]")
    plt.ylabel(r"local level-set distance $|f_V|/\|\nabla f_V\|$ [rad approx.]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viscdf_local_distance_vs_gt_distance.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(vis["gt_abs"][idx], vis["pred_grad_norm"][idx], s=4, alpha=0.25)
    plt.axhline(1.0, linestyle="--", linewidth=1.2)
    plt.xlabel("q0-library target |d_V^GT| [rad]")
    plt.ylabel(r"$\|\nabla_q f_V\|_2$")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viscdf_gradnorm_vs_gt_distance.png"), dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt",
    )
    parser.add_argument(
        "--data",
        default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz",
    )
    parser.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--batch-x", type=int, default=16)
    parser.add_argument("--batch-q", type=int, default=256)
    parser.add_argument("--decode-x-chunk", type=int, default=16)

    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=66.0)
    parser.add_argument("--z-min", type=float, default=0.20)
    parser.add_argument("--z-max", type=float, default=0.70)
    parser.add_argument("--delta", type=float, default=0.01)

    parser.add_argument(
        "--yiming-cdf-dir",
        default="",
        help="Optional clone of https://github.com/idiap/cdf. If set, also evaluate the released Franka neural CDF.",
    )
    parser.add_argument("--yiming-step", type=int, default=49900)
    parser.add_argument("--yiming-num-batches", type=int, default=10)
    parser.add_argument("--yiming-batch-q", type=int, default=2048)

    parser.add_argument(
        "--output-dir",
        default="outputs/dual_cspace_field_geometry",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = _resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    vis_summary, vis_arrays = run_viscdf(args, device)
    yiming_summary = None
    yiming_arrays = None
    if args.yiming_cdf_dir:
        yiming_summary, yiming_arrays = run_yiming_released(args, device)

    result = {
        "experiment": "dual_cspace_field_geometry",
        "interpretation_note": (
            "VisCDF GT distances are q0-library approximations used by training; "
            "Yiming released CDF is for Franka and is evaluated only for empirical gradient norm/projection behavior."
        ),
        "viscdf": vis_summary,
        "yiming_released_cdf": yiming_summary,
    }

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    npz_payload = {f"vis_{k}": v for k, v in vis_arrays.items()}
    if yiming_arrays is not None:
        npz_payload.update({f"yiming_{k}": v for k, v in yiming_arrays.items()})
    np.savez_compressed(os.path.join(output_dir, "samples.npz"), **npz_payload)

    _make_plots(output_dir, vis_arrays, yiming_arrays)

    print("\n================ FINAL SUMMARY ================")
    print(f"output_dir: {output_dir}")
    print(f"summary:    {json_path}")
    print("\n[VisCDF / all]")
    all_vis = vis_summary["cohorts"]["all"]
    print("gradient norm:", json.dumps(all_vis["pred_grad_norm"], indent=2))
    print("eikonal error:", json.dumps(all_vis["eikonal_abs_error"], indent=2))
    print("grad cosine to q0-GT:", json.dumps(all_vis["grad_cosine_to_gt"], indent=2))
    print(
        "corr raw |f| vs GT distance =",
        all_vis["corr_raw_abs_f_vs_gt_distance"],
    )
    print(
        "corr |f|/||grad|| vs GT distance =",
        all_vis["corr_local_distance_vs_gt_distance"],
    )
    print(
        "oracle boundary rate after q-f*grad =",
        all_vis["oracle_boundary_rate_sdf_eps003"],
    )
    print(
        "oracle boundary rate after q-f*grad/||grad||^2 =",
        all_vis["oracle_boundary_rate_newton_eps003"],
    )
    print(
        "oracle boundary rate after GT q0 projection =",
        all_vis["oracle_boundary_rate_gt_eps003"],
    )
    if yiming_summary is not None:
        print("\n[Yiming released Franka neural CDF]")
        print("gradient norm:", json.dumps(yiming_summary["grad_norm"], indent=2))
        print("eikonal error:", json.dumps(yiming_summary["eikonal_abs_error"], indent=2))
        print(
            "residual after official q-d*grad:",
            json.dumps(yiming_summary["abs_value_after_official_q_minus_d_grad"], indent=2),
        )
        print(
            "residual after normalized Newton:",
            json.dumps(yiming_summary["abs_value_after_normalized_newton"], indent=2),
        )


if __name__ == "__main__":
    main()
