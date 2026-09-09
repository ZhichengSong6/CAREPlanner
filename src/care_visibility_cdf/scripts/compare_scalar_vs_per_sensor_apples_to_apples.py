#!/usr/bin/env python3
"""Apples-to-apples scalar-vs-8head VisCDF comparison.

Both checkpoints are evaluated on EXACTLY the same held-out x and random q.

Field metrics use the scalar Exp1 target definition:
    f_union^GT(x,q) = max_s sign_s(x,q) * d_s(x,q)
with the corresponding winner-branch GT gradient.

Planning metrics start from the same (x,q_init) pairs for both models and use
identical projection/ascent hyperparameters.  For the 8-head model, the runtime
union field is max_s f_s and autograd differentiates through the current winner.

This script is intentionally diagnostic-only: it does not change training or
runtime CAREPlanner code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from train_signed_visibility_cdf_pairwise_replace import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    MLP,
    VisibilityQ0Dataset,
    PinocchioFOVOracle,
    _parse_mlp_layers,
    decode_per_sensor_distance_and_grad,
    make_input_pairs,
    pairwise_signed_fov_margins,
    sample_random_q,
    union_signed_targets,
)


def _load_model(path: str, out_dim: int, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cargs = ckpt.get("args", {})

    raw_layers = cargs.get("mlp_layers", "1024,512,256,128,128")
    layers = _parse_mlp_layers(raw_layers)
    raw_skips = cargs.get("skips", "")
    skips = tuple(
        int(v.strip()) for v in str(raw_skips).split(",") if v.strip()
    )

    model = MLP(
        in_dim=10,
        out_dim=out_dim,
        activation=cargs.get("activation", "relu"),
        model_arch=cargs.get("model_arch", "yiming"),
        mlp_layers=layers,
        skips=skips,
        nerf=bool(cargs.get("nerf", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def _field_value_grad(
    model: torch.nn.Module,
    inp: torch.Tensor,
    mode: str,
    valid_mask: torch.Tensor | None = None,
):
    """Return union value [N] and d(union)/dq [N,7]."""
    q = inp[:, 3:10].detach().clone().requires_grad_(True)
    x = inp[:, :3].detach()
    pred = model(torch.cat([x, q], dim=-1))

    if mode == "scalar":
        value = pred.reshape(-1)
    elif mode == "eight":
        if pred.ndim != 2 or pred.shape[1] != 8:
            raise RuntimeError(f"8-head output must be [N,8], got {tuple(pred.shape)}")
        if valid_mask is None:
            raise RuntimeError("8-head union requires the per-x sensor availability mask")
        if valid_mask.shape != pred.shape:
            raise RuntimeError(
                f"valid_mask {tuple(valid_mask.shape)} != pred {tuple(pred.shape)}"
            )
        pred = torch.where(valid_mask, pred, torch.full_like(pred, -float("inf")))
        value = pred.max(dim=1).values
    else:
        raise ValueError(mode)

    grad = torch.autograd.grad(
        value,
        q,
        grad_outputs=torch.ones_like(value),
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]
    return value.detach(), grad.detach()


def _pair_value_grad(
    model: torch.nn.Module,
    x: torch.Tensor,
    q_pair: torch.Tensor,
    mode: str,
    sensor_available: torch.Tensor | None = None,
):
    """x [Bx,3], q_pair [Bx,Bq,7] -> value [Bx,Bq], grad [Bx,Bq,7]."""
    bx, bq, _ = q_pair.shape
    qf = q_pair.detach().clone().reshape(bx * bq, 7).requires_grad_(True)
    xf = x[:, None, :].expand(bx, bq, 3).reshape(bx * bq, 3)
    pred = model(torch.cat([xf, qf], dim=-1))

    if mode == "scalar":
        value = pred.reshape(-1)
    else:
        if sensor_available is None:
            raise RuntimeError("8-head union requires sensor_available")
        mask = sensor_available[:, None, :].expand(bx, bq, 8).reshape(bx * bq, 8)
        pred = torch.where(mask, pred, torch.full_like(pred, -float("inf")))
        value = pred.max(dim=1).values

    grad = torch.autograd.grad(
        value,
        qf,
        grad_outputs=torch.ones_like(value),
        retain_graph=False,
        create_graph=False,
        only_inputs=True,
    )[0]

    return (
        value.detach().reshape(bx, bq),
        grad.detach().reshape(bx, bq, 7),
    )


@torch.no_grad()
def _pair_value(
    model: torch.nn.Module,
    x: torch.Tensor,
    q_pair: torch.Tensor,
    mode: str,
    sensor_available: torch.Tensor | None = None,
):
    bx, bq, _ = q_pair.shape
    qf = q_pair.reshape(bx * bq, 7)
    xf = x[:, None, :].expand(bx, bq, 3).reshape(bx * bq, 3)
    pred = model(torch.cat([xf, qf], dim=-1))
    if mode == "scalar":
        value = pred.reshape(-1)
    else:
        if sensor_available is None:
            raise RuntimeError("8-head union requires sensor_available")
        mask = sensor_available[:, None, :].expand(bx, bq, 8).reshape(bx * bq, 8)
        pred = torch.where(mask, pred, torch.full_like(pred, -float("inf")))
        value = pred.max(dim=1).values
    return value.reshape(bx, bq)


@torch.no_grad()
def _oracle_pair_g(
    oracle: PinocchioFOVOracle,
    x: torch.Tensor,
    q_pair: torch.Tensor,
):
    raw, _ = pairwise_signed_fov_margins(
        fov_oracle=oracle,
        x=x,
        q_pair=q_pair,
        pair_chunk=512,
    )
    return (raw - oracle.delta).max(dim=-1).values


def _clamp(q, q_min, q_max):
    return torch.maximum(
        torch.minimum(q, q_max[None, None, :]),
        q_min[None, None, :],
    )


def _projection(
    model,
    mode,
    x,
    q_init,
    q_min,
    q_max,
    iters,
    damping,
    max_step,
    sensor_available=None,
):
    q = q_init.detach().clone()
    eps = 1e-8
    for _ in range(iters):
        f, grad = _pair_value_grad(
            model, x, q, mode, sensor_available=sensor_available
        )
        g2 = (grad * grad).sum(dim=-1, keepdim=True)
        step = f[..., None] * grad / torch.clamp(g2, min=eps)
        sn = torch.linalg.norm(step, dim=-1, keepdim=True)
        if max_step > 0:
            step = step * torch.clamp(
                max_step / torch.clamp(sn, min=eps), max=1.0
            )
        q = _clamp(q - damping * step, q_min, q_max).detach()
    return q


def _ascent_snapshots(
    model,
    mode,
    x,
    q_start,
    q_min,
    q_max,
    step_size,
    max_step,
    snapshots,
    sensor_available=None,
):
    q = q_start.detach().clone()
    out = {}
    eps = 1e-8
    max_k = max(snapshots)
    for k in range(1, max_k + 1):
        _, grad = _pair_value_grad(
            model, x, q, mode, sensor_available=sensor_available
        )
        gn = torch.linalg.norm(grad, dim=-1, keepdim=True)
        direction = grad / torch.clamp(gn, min=eps)
        step = step_size * direction
        sn = torch.linalg.norm(step, dim=-1, keepdim=True)
        if max_step > 0:
            step = step * torch.clamp(
                max_step / torch.clamp(sn, min=eps), max=1.0
            )
        q = _clamp(q + step, q_min, q_max).detach()
        if k in snapshots:
            out[k] = q.clone()
    return out


def _accumulate_field(agg, pred, grad, target, target_grad):
    err = pred - target
    cos = F.cosine_similarity(grad, target_grad, dim=-1, eps=1e-6)
    n = pred.numel()
    agg["n"] += n
    agg["abs"] += float(err.abs().sum().cpu())
    agg["sq"] += float(err.square().sum().cpu())
    agg["sign"] += int(((pred >= 0) == (target >= 0)).sum().item())
    agg["cos"] += float(cos.sum().cpu())
    agg["norm_err"] += float(
        (grad.norm(dim=-1) - target_grad.norm(dim=-1)).abs().sum().cpu()
    )


def _field_summary(a):
    n = int(a["n"])
    return {
        "count": n,
        "mae": _safe_div(a["abs"], n),
        "rmse": math.sqrt(_safe_div(a["sq"], n)) if n else float("nan"),
        "sign_accuracy": _safe_div(a["sign"], n),
        "gradient_cosine_mean": _safe_div(a["cos"], n),
        "gradient_norm_abs_error_mean": _safe_div(a["norm_err"], n),
    }


def _planning_summary(records):
    out = {}
    n = int(records["n"])
    outside_n = int(records["outside_n"])
    out["count"] = n
    out["initial_outside_count"] = outside_n

    for key in (
        "proj_learned_boundary_003",
        "proj_oracle_boundary_005",
        "proj_oracle_boundary_010",
        "proj_oracle_boundary_030",
        "proj_inside",
    ):
        out[key] = _safe_div(records[key], n)

    for key in (
        "proj_oracle_boundary_005_outside",
        "proj_oracle_boundary_010_outside",
        "proj_oracle_boundary_030_outside",
        "proj_inside_outside",
    ):
        out[key] = _safe_div(records[key], outside_n)

    for k in (1, 3, 5, 10):
        for thr in (0.0, 0.005, 0.01, 0.03):
            tag = str(thr).replace(".", "p")
            key = f"asc{k}_g_ge_{tag}"
            key_o = f"{key}_outside"
            out[key] = _safe_div(records[key], n)
            out[key_o] = _safe_div(records[key_o], outside_n)

    return out


def _accumulate_planning(
    rec,
    f_proj,
    g_init,
    g_proj,
    ascent_g,
):
    n = g_init.numel()
    outside = g_init < 0
    no = int(outside.sum().item())
    rec["n"] += n
    rec["outside_n"] += no

    rec["proj_learned_boundary_003"] += int((f_proj.abs() < 0.03).sum().item())
    for tol in (0.005, 0.01, 0.03):
        tag = f"{int(tol*1000):03d}"
        m = g_proj.abs() < tol
        rec[f"proj_oracle_boundary_{tag}"] += int(m.sum().item())
        rec[f"proj_oracle_boundary_{tag}_outside"] += int(
            (m & outside).sum().item()
        )
    inside = g_proj >= 0
    rec["proj_inside"] += int(inside.sum().item())
    rec["proj_inside_outside"] += int((inside & outside).sum().item())

    for k, g in ascent_g.items():
        for thr in (0.0, 0.005, 0.01, 0.03):
            tag = str(thr).replace(".", "p")
            m = g >= thr
            key = f"asc{k}_g_ge_{tag}"
            rec[key] += int(m.sum().item())
            rec[f"{key}_outside"] += int((m & outside).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scalar-checkpoint",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "exp1_yiming_k500_fov_signed/final.pt"
        ),
    )
    ap.add_argument(
        "--eight-checkpoint",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "per_sensor_e2e_fullbatch_seed0/final.pt"
        ),
    )
    ap.add_argument(
        "--data",
        default=(
            "src/care_visibility_cdf/data/"
            "visibility_yiming_style_grid30_q20000_k500_fovonly.npz"
        ),
    )
    ap.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--seed", type=int, default=123)

    # Field test: same held-out x and q for both models.
    ap.add_argument("--num-batches", type=int, default=20)
    ap.add_argument("--batch-x", type=int, default=128)
    ap.add_argument("--batch-q", type=int, default=64)
    ap.add_argument("--val-count", type=int, default=1000)
    ap.add_argument("--decode-x-chunk", type=int, default=64)

    # Planning test: smaller set, but exact same starts for both models.
    ap.add_argument("--planning-batches", type=int, default=10)
    ap.add_argument("--planning-batch-x", type=int, default=8)
    ap.add_argument("--planning-batch-q", type=int, default=64)
    ap.add_argument("--projection-iters", type=int, default=10)
    ap.add_argument("--projection-damping", type=float, default=0.5)
    ap.add_argument("--projection-max-step", type=float, default=0.25)
    ap.add_argument("--ascent-step", type=float, default=0.05)
    ap.add_argument("--ascent-max-step", type=float, default=0.25)

    ap.add_argument(
        "--output",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "per_sensor_e2e_fullbatch_seed0/"
            "scalar_vs_8head_apples_to_apples_masked_final.json"
        ),
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.scalar_checkpoint):
        raise FileNotFoundError(args.scalar_checkpoint)
    if not os.path.exists(args.eight_checkpoint):
        raise FileNotFoundError(args.eight_checkpoint)

    scalar, scalar_ckpt = _load_model(args.scalar_checkpoint, 1, device)
    eight, eight_ckpt = _load_model(args.eight_checkpoint, 8, device)

    dataset = VisibilityQ0Dataset(
        args.data, val_count=args.val_count, seed=0
    )
    oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=50.0,
        vertical_fov_deg=66.0,
        z_min=0.20,
        z_max=0.70,
        delta=0.01,
    )
    masks = dataset.sensor_masks(device)
    q_min, q_max = dataset.q_limits(device=device)

    field = {
        "scalar": defaultdict(float),
        "eight_union": defaultdict(float),
    }

    print("\n=== FIELD: SAME HELD-OUT (x,q) ===")
    for bi in range(args.num_batches):
        x, qlib, valid, _ = dataset.sample_x_batch(
            args.batch_x, split="val", device=device
        )
        q = sample_random_q(dataset, args.batch_q, device)

        with torch.no_grad():
            d_s, grad_d_s, has_s = decode_per_sensor_distance_and_grad(
                qlib=qlib,
                valid=valid,
                q_query=q,
                sensor_masks=masks,
                x_chunk=args.decode_x_chunk,
            )
            _, sign_s = oracle.signed_fov_margins(x, q)
            target, target_grad = union_signed_targets(
                d_s=d_s,
                grad_d_s=grad_d_s,
                sign_s=sign_s,
                has_sensor=has_s,
            )
            inp = make_input_pairs(x, q)

        tgt = target.reshape(-1)
        tgt_grad = target_grad.reshape(-1, 7)

        ps, gs = _field_value_grad(scalar, inp, "scalar")
        pair_valid_mask = has_s[:, None, :].expand(
            args.batch_x, args.batch_q, 8
        ).reshape(-1, 8)
        pe, ge = _field_value_grad(
            eight, inp, "eight", valid_mask=pair_valid_mask
        )

        _accumulate_field(field["scalar"], ps, gs, tgt, tgt_grad)
        _accumulate_field(field["eight_union"], pe, ge, tgt, tgt_grad)

        ss = _field_summary(field["scalar"])
        es = _field_summary(field["eight_union"])
        print(
            f"[field] {bi+1:02d}/{args.num_batches} "
            f"scalar mae={ss['mae']:.4f} cos={ss['gradient_cosine_mean']:.4f} "
            f"| 8head mae={es['mae']:.4f} cos={es['gradient_cosine_mean']:.4f}",
            flush=True,
        )

    field_summary = {
        "scalar": _field_summary(field["scalar"]),
        "eight_union": _field_summary(field["eight_union"]),
    }

    planning = {
        "scalar": defaultdict(float),
        "eight_union": defaultdict(float),
    }

    print("\n=== PLANNING: SAME STARTS / SAME HYPERPARAMETERS ===")
    snapshots = (1, 3, 5, 10)
    for bi in range(args.planning_batches):
        x, _, valid_plan, _ = dataset.sample_x_batch(
            args.planning_batch_x, split="val", device=device
        )
        sensor_available_plan = valid_plan.any(dim=1)
        q_shared = sample_random_q(
            dataset, args.planning_batch_q, device
        )
        q_init = q_shared[None, :, :].expand(
            args.planning_batch_x, -1, -1
        ).contiguous()

        g_init = _oracle_pair_g(oracle, x, q_init)

        for name, model, mode in (
            ("scalar", scalar, "scalar"),
            ("eight_union", eight, "eight"),
        ):
            sensor_available = (
                sensor_available_plan if mode == "eight" else None
            )
            q_proj = _projection(
                model=model,
                mode=mode,
                x=x,
                q_init=q_init,
                q_min=q_min,
                q_max=q_max,
                iters=args.projection_iters,
                damping=args.projection_damping,
                max_step=args.projection_max_step,
                sensor_available=sensor_available,
            )
            f_proj = _pair_value(
                model, x, q_proj, mode, sensor_available=sensor_available
            )
            g_proj = _oracle_pair_g(oracle, x, q_proj)

            q_snaps = _ascent_snapshots(
                model=model,
                mode=mode,
                x=x,
                q_start=q_proj,
                q_min=q_min,
                q_max=q_max,
                step_size=args.ascent_step,
                max_step=args.ascent_max_step,
                snapshots=snapshots,
                sensor_available=sensor_available,
            )
            ascent_g = {
                k: _oracle_pair_g(oracle, x, qk)
                for k, qk in q_snaps.items()
            }
            _accumulate_planning(
                planning[name], f_proj, g_init, g_proj, ascent_g
            )

        sm = _planning_summary(planning["scalar"])
        em = _planning_summary(planning["eight_union"])
        print(
            f"[plan] {bi+1:02d}/{args.planning_batches} "
            f"proj|g|<.03 scalar={sm['proj_oracle_boundary_030']:.4f} "
            f"8head={em['proj_oracle_boundary_030']:.4f} "
            f"| asc10 g>=.01 scalar={sm['asc10_g_ge_0p01']:.4f} "
            f"8head={em['asc10_g_ge_0p01']:.4f}",
            flush=True,
        )

    planning_summary = {
        "scalar": _planning_summary(planning["scalar"]),
        "eight_union": _planning_summary(planning["eight_union"]),
    }

    result = {
        "config": vars(args),
        "scalar_checkpoint_step": int(scalar_ckpt.get("step", -1)),
        "eight_checkpoint_step": int(eight_ckpt.get("step", -1)),
        "eight_union_sensor_masking": (
            "per-x q0 availability mask; unavailable heads excluded from max"
        ),
        "field": field_summary,
        "planning": planning_summary,
    }

    # Compact deltas: positive means 8-head is better for accuracy/cosine,
    # negative means 8-head is better for error.
    s = field_summary["scalar"]
    e = field_summary["eight_union"]
    result["field_delta_8head_minus_scalar"] = {
        "mae": e["mae"] - s["mae"],
        "rmse": e["rmse"] - s["rmse"],
        "sign_accuracy": e["sign_accuracy"] - s["sign_accuracy"],
        "gradient_cosine_mean": (
            e["gradient_cosine_mean"] - s["gradient_cosine_mean"]
        ),
        "gradient_norm_abs_error_mean": (
            e["gradient_norm_abs_error_mean"]
            - s["gradient_norm_abs_error_mean"]
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print("\n================ FINAL COMPARISON ================")
    print("[FIELD scalar]")
    print(json.dumps(field_summary["scalar"], indent=2))
    print("[FIELD 8-head union]")
    print(json.dumps(field_summary["eight_union"], indent=2))
    print("[FIELD delta 8-head - scalar]")
    print(json.dumps(result["field_delta_8head_minus_scalar"], indent=2))

    print("\n[PLANNING scalar]")
    print(json.dumps(planning_summary["scalar"], indent=2))
    print("[PLANNING 8-head union]")
    print(json.dumps(planning_summary["eight_union"], indent=2))
    print(f"\n[OUTPUT] {args.output}")


if __name__ == "__main__":
    main()
