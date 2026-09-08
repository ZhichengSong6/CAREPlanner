#!/usr/bin/env python3
"""End-to-end per-sensor visibility CDF training.

This is the mode-preserving counterpart of
train_signed_visibility_cdf_pairwise_replace.py.

The existing q0 dataset already stores one zero-level library per sensor:
    q[p,k,:,s], valid_fov[p,k,s], sensor_chain_masks[s,:].

The scalar baseline constructs all eight signed fields and then collapses them
with max_s before supervision.  This trainer deliberately does NOT perform that
max.  It learns

    F_theta(x,q) = [f_0(x,q), ..., f_7(x,q)]

end-to-end from random initialization.

Self-occlusion remains outside the learned field.  Each head represents the
corresponding analytic FOV configuration-space signed distance field.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_signed_visibility_cdf_pairwise_replace import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    MLP,
    VisibilityQ0Dataset,
    PinocchioFOVOracle,
    _parse_mlp_layers,
    sync_if_cuda,
    sample_random_q,
    sample_pairwise_near_q_per_x,
    decode_per_sensor_distance_and_grad,
    decode_pairwise_per_sensor_distance_and_grad,
    pairwise_signed_fov_margins,
    make_input_pairs,
    make_pairwise_input_pairs,
)


NUM_SENSORS = 8


def per_sensor_signed_targets(
    d_s: torch.Tensor,
    grad_d_s: torch.Tensor,
    sign_s: torch.Tensor,
    has_sensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the eight signed fields without the baseline's sensor-wise max.

    Args:
        d_s:        [B_x, B_n, S]
        grad_d_s:   [B_x, B_n, S, 7]
        sign_s:     [B_x, B_n, S]
        has_sensor: [B_x, S]

    Returns:
        target:      [B_x, B_n, S]
        target_grad: [B_x, B_n, S, 7]
        valid_mask:  [B_x, B_n, S]

    A point/sensor pair is supervised only if that spatial point has at least
    one valid q0 for that sensor in the stored zero-level library.
    """
    if d_s.ndim != 3 or grad_d_s.ndim != 4 or sign_s.shape != d_s.shape:
        raise ValueError(
            f"bad per-sensor target shapes: d={tuple(d_s.shape)} "
            f"grad={tuple(grad_d_s.shape)} sign={tuple(sign_s.shape)}"
        )

    Bx, Bn, S = d_s.shape
    if S != NUM_SENSORS:
        raise ValueError(f"expected {NUM_SENSORS} sensors, got {S}")
    if has_sensor.shape != (Bx, S):
        raise ValueError(
            f"has_sensor must be {(Bx,S)}, got {tuple(has_sensor.shape)}"
        )

    valid_mask = has_sensor[:, None, :].expand(Bx, Bn, S)
    target = sign_s * d_s
    target_grad = sign_s[..., None] * grad_d_s

    finite = torch.isfinite(target) & torch.isfinite(target_grad).all(dim=-1)
    valid_mask = valid_mask & finite

    target = torch.where(valid_mask, target, torch.zeros_like(target))
    target_grad = torch.where(
        valid_mask[..., None], target_grad, torch.zeros_like(target_grad)
    )
    return target, target_grad, valid_mask


def _mean_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def _balanced_head_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, list]:
    """Mean within each sensor first, then mean across available sensors.

    This prevents sensors with more q0-supporting workspace points from
    dominating the training objective, which is important when alternative
    sensing modes are the quantity we want to preserve.
    """
    if values.shape != mask.shape:
        raise ValueError(
            f"balanced mean shape mismatch {tuple(values.shape)} vs {tuple(mask.shape)}"
        )
    per_head = []
    head_values = []
    for s in range(values.shape[1]):
        if torch.any(mask[:, s]):
            v = values[mask[:, s], s].mean()
            per_head.append(v)
            head_values.append(float(v.detach().cpu()))
        else:
            head_values.append(float("nan"))
    if not per_head:
        return values.sum() * 0.0, head_values
    return torch.stack(per_head).mean(), head_values


def compute_per_sensor_losses(
    model: nn.Module,
    inp: torch.Tensor,
    target: torch.Tensor,
    target_grad: torch.Tensor,
    valid_mask: torch.Tensor,
    weights: Dict[str, float],
):
    """Per-head value + gradient geometry, plus union-envelope consistency.

    All eight heads receive explicit supervision.  Gradient/eikonal/tension are
    also computed head-by-head, so non-winning sensor geometry is learned rather
    than only the current union winner.
    """
    if target.ndim != 2 or target.shape[1] != NUM_SENSORS:
        raise ValueError(f"target must be [N,8], got {tuple(target.shape)}")
    if target_grad.shape != (target.shape[0], NUM_SENSORS, 7):
        raise ValueError(
            f"target_grad must be [N,8,7], got {tuple(target_grad.shape)}"
        )
    if valid_mask.shape != target.shape:
        raise ValueError(
            f"valid_mask must be {tuple(target.shape)}, got {tuple(valid_mask.shape)}"
        )

    x_inputs = inp[:, :3].detach()
    q_inputs = inp[:, 3:10].detach().clone().requires_grad_(True)
    model_inputs = torch.cat([x_inputs, q_inputs], dim=-1)

    pred = model(model_inputs)
    if pred.shape != target.shape:
        raise RuntimeError(
            f"model output must be {tuple(target.shape)}, got {tuple(pred.shape)}"
        )

    # Sensor-balanced value supervision.
    sqerr = (pred - target).square()
    sensor_sdf_loss, sdf_per_head = _balanced_head_mean(sqerr, valid_mask)

    # Union-envelope consistency is an auxiliary objective only.  It preserves
    # the original scalar field semantics while leaving all branch identities
    # available at runtime.
    neg_inf = torch.full_like(pred, -float("inf"))
    pred_masked = torch.where(valid_mask, pred, neg_inf)
    target_masked = torch.where(valid_mask, target, neg_inf)
    row_valid = valid_mask.any(dim=1)
    pred_union = pred_masked.max(dim=1).values
    target_union = target_masked.max(dim=1).values
    union_sdf_loss = _mean_valid(
        (pred_union - target_union).square(), row_valid
    )

    grad_losses = []
    eikonal_losses = []
    tension_losses = []
    grad_cos_per_head = []
    grad_norm_per_head = []

    for s in range(NUM_SENSORS):
        mask_s = valid_mask[:, s]

        # The network is pointwise, therefore grad of the sum produces the
        # per-row gradient df_s(x_i,q_i)/dq_i.
        grad_pred_s = torch.autograd.grad(
            outputs=pred[:, s],
            inputs=q_inputs,
            grad_outputs=torch.ones_like(pred[:, s]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if torch.any(mask_s):
            gt_s = target_grad[:, s, :]
            cos_s = F.cosine_similarity(
                grad_pred_s, gt_s, dim=-1, eps=1e-6
            )
            grad_loss_s = (1.0 - cos_s[mask_s]).mean()
            eikonal_s = torch.abs(
                grad_pred_s.norm(2, dim=-1) - 1.0
            )[mask_s].mean()

            # Match the scalar baseline's tension definition exactly:
            # gradient of the summed first-derivative components.
            dd_grad_s = torch.autograd.grad(
                outputs=grad_pred_s,
                inputs=q_inputs,
                grad_outputs=torch.ones_like(grad_pred_s),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            tension_s = dd_grad_s.square().sum(dim=-1)[mask_s].mean()

            grad_losses.append(grad_loss_s)
            eikonal_losses.append(eikonal_s)
            tension_losses.append(tension_s)
            grad_cos_per_head.append(float(cos_s[mask_s].detach().mean().cpu()))
            grad_norm_per_head.append(
                float(grad_pred_s[mask_s].detach().norm(2, dim=-1).mean().cpu())
            )
        else:
            grad_cos_per_head.append(float("nan"))
            grad_norm_per_head.append(float("nan"))

    zero = pred.sum() * 0.0
    grad_loss = torch.stack(grad_losses).mean() if grad_losses else zero
    eikonal_loss = (
        torch.stack(eikonal_losses).mean() if eikonal_losses else zero
    )
    tension_loss = (
        torch.stack(tension_losses).mean() if tension_losses else zero
    )

    loss = (
        weights["sdf"] * sensor_sdf_loss
        + weights["union"] * union_sdf_loss
        + weights["eikonal"] * eikonal_loss
        + weights["tension"] * tension_loss
        + weights["grad"] * grad_loss
    )

    with torch.no_grad():
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        valid_counts = valid_mask.sum(dim=0)
        # Ranking metrics are useful during training because mode preservation
        # is the reason for this model.
        masked_pred_rank = torch.where(valid_mask, pred, neg_inf)
        masked_gt_rank = torch.where(valid_mask, target, neg_inf)
        gt_winner = masked_gt_rank.argmax(dim=1)
        pred_winner = masked_pred_rank.argmax(dim=1)
        winner_acc = (
            (gt_winner[row_valid] == pred_winner[row_valid]).float().mean()
            if torch.any(row_valid)
            else torch.tensor(float("nan"), device=pred.device)
        )

    stats = {
        "loss": float(loss.detach().cpu()),
        "sensor_sdf_loss": float(sensor_sdf_loss.detach().cpu()),
        "union_sdf_loss": float(union_sdf_loss.detach().cpu()),
        "eikonal_loss": float(eikonal_loss.detach().cpu()),
        "tension_loss": float(tension_loss.detach().cpu()),
        "grad_loss": float(grad_loss.detach().cpu()),
        "pred_mean": float(pred_valid.detach().mean().cpu())
        if pred_valid.numel() else float("nan"),
        "pred_abs_mean": float(pred_valid.detach().abs().mean().cpu())
        if pred_valid.numel() else float("nan"),
        "target_mean": float(target_valid.detach().mean().cpu())
        if target_valid.numel() else float("nan"),
        "target_abs_mean": float(target_valid.detach().abs().mean().cpu())
        if target_valid.numel() else float("nan"),
        "positive_ratio": float((target_valid > 0).float().mean().cpu())
        if target_valid.numel() else float("nan"),
        "winner_accuracy": float(winner_acc.detach().cpu()),
        "valid_supervision_count": int(valid_mask.sum().detach().cpu()),
        "encoded_dim": float(getattr(model, "encoded_dim", 10)),
    }
    for s in range(NUM_SENSORS):
        stats[f"s{s}_sdf_loss"] = sdf_per_head[s]
        stats[f"s{s}_grad_cosine"] = grad_cos_per_head[s]
        stats[f"s{s}_grad_norm_mean"] = grad_norm_per_head[s]
        stats[f"s{s}_valid_count"] = int(valid_counts[s].detach().cpu())

    return loss, stats


def run_batch(
    model: nn.Module,
    dataset: VisibilityQ0Dataset,
    fov_oracle: PinocchioFOVOracle,
    device: torch.device,
    batch_x: int,
    batch_q: int,
    split: str,
    weights: Dict[str, float],
    decode_x_chunk: int,
    near_zero_ratio: float,
    near_zero_std: float,
    profile: bool = False,
):
    timings = {}
    t0 = time.perf_counter()

    x, qlib, valid, _ = dataset.sample_x_batch(
        batch_x, split=split, device=device
    )
    sensor_masks = dataset.sensor_masks(device=device)
    sync_if_cuda(device)
    timings["sample_x"] = time.perf_counter() - t0

    Bx = x.shape[0]
    use_near = split == "train" and near_zero_ratio > 0.0
    near_per_x = (
        max(0, min(batch_q, int(round(batch_q * near_zero_ratio))))
        if use_near else 0
    )
    rand_q_count = batch_q - near_per_x
    if rand_q_count <= 0:
        raise RuntimeError("Need at least one random-q sample per x.")

    t1 = time.perf_counter()
    q_rand = sample_random_q(dataset, rand_q_count, device=device)
    q_near = None
    if near_per_x > 0:
        q_near = sample_pairwise_near_q_per_x(
            dataset=dataset,
            qlib=qlib,
            valid=valid,
            sensor_masks=sensor_masks,
            near_per_x=near_per_x,
            device=device,
            near_zero_std=near_zero_std,
        )
    sync_if_cuda(device)
    timings["sample_q"] = time.perf_counter() - t1

    # Cartesian random-q branch.
    td = time.perf_counter()
    d_cart, grad_cart, has_cart = decode_per_sensor_distance_and_grad(
        qlib=qlib,
        valid=valid,
        q_query=q_rand,
        sensor_masks=sensor_masks,
        x_chunk=decode_x_chunk,
    )
    sync_if_cuda(device)
    timings["decode_cart"] = time.perf_counter() - td

    tf = time.perf_counter()
    _, sign_cart = fov_oracle.signed_fov_margins(x, q_rand)
    sync_if_cuda(device)
    timings["fov_cart"] = time.perf_counter() - tf

    target_cart, grad_target_cart, mask_cart = per_sensor_signed_targets(
        d_cart, grad_cart, sign_cart, has_cart
    )

    inp_parts = [make_input_pairs(x, q_rand)]
    target_parts = [target_cart.reshape(Bx * rand_q_count, NUM_SENSORS)]
    grad_parts = [
        grad_target_cart.reshape(Bx * rand_q_count, NUM_SENSORS, 7)
    ]
    mask_parts = [mask_cart.reshape(Bx * rand_q_count, NUM_SENSORS)]

    # Optional pairwise near-zero branch.  Default remains 0.0 to match Exp1.
    if q_near is not None:
        tdn = time.perf_counter()
        d_near, grad_near, has_near = (
            decode_pairwise_per_sensor_distance_and_grad(
                qlib=qlib,
                valid=valid,
                q_pair=q_near,
                sensor_masks=sensor_masks,
                x_chunk=max(1, min(decode_x_chunk, 128)),
            )
        )
        sync_if_cuda(device)
        timings["decode_near"] = time.perf_counter() - tdn

        tfn = time.perf_counter()
        _, sign_near = pairwise_signed_fov_margins(
            fov_oracle=fov_oracle,
            x=x,
            q_pair=q_near,
            pair_chunk=512,
        )
        sync_if_cuda(device)
        timings["fov_near"] = time.perf_counter() - tfn

        target_near, grad_target_near, mask_near = (
            per_sensor_signed_targets(
                d_near, grad_near, sign_near, has_near
            )
        )

        inp_parts.append(make_pairwise_input_pairs(x, q_near))
        target_parts.append(
            target_near.reshape(Bx * near_per_x, NUM_SENSORS)
        )
        grad_parts.append(
            grad_target_near.reshape(Bx * near_per_x, NUM_SENSORS, 7)
        )
        mask_parts.append(
            mask_near.reshape(Bx * near_per_x, NUM_SENSORS)
        )

    inp = torch.cat(inp_parts, dim=0)
    target = torch.cat(target_parts, dim=0)
    target_grad = torch.cat(grad_parts, dim=0)
    valid_mask = torch.cat(mask_parts, dim=0)

    expected = Bx * batch_q
    if inp.shape[0] != expected:
        raise RuntimeError(
            f"pair budget mismatch: expected {expected}, got {inp.shape[0]}"
        )

    tl = time.perf_counter()
    loss, stats = compute_per_sensor_losses(
        model=model,
        inp=inp,
        target=target,
        target_grad=target_grad,
        valid_mask=valid_mask,
        weights=weights,
    )
    sync_if_cuda(device)
    timings["loss_forward"] = time.perf_counter() - tl

    if profile:
        for k, v in timings.items():
            stats[f"time_{k}"] = float(v)
        stats["time_decode"] = float(
            timings.get("decode_cart", 0.0)
            + timings.get("decode_near", 0.0)
        )
        stats["time_fov"] = float(
            timings.get("fov_cart", 0.0)
            + timings.get("fov_near", 0.0)
        )

    return loss, stats


def save_checkpoint(
    path, model, optimizer, scheduler, args, step, best_val, stats
):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "args": vars(args),
            "step": int(step),
            "best_val": float(best_val),
            "stats": stats,
            "output_semantics": "per_sensor_signed_visibility_cdf",
            "sensor_frames": list(DEFAULT_SENSOR_FRAMES),
            "joint_names": list(DEFAULT_JOINT_NAMES),
            "out_dim": NUM_SENSORS,
        },
        path,
    )


def load_checkpoint(
    path, model, optimizer=None, scheduler=None, device="cpu"
):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return int(ckpt.get("step", 0)), float(
        ckpt.get("best_val", float("inf"))
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Train an end-to-end 8-head per-sensor signed VisCDF."
    )
    p.add_argument(
        "--data",
        default=(
            "src/care_visibility_cdf/data/"
            "visibility_yiming_style_grid30_q20000_k500_fovonly.npz"
        ),
    )
    p.add_argument(
        "--urdf", default="src/arm_description/urdf/Arm.urdf"
    )
    p.add_argument(
        "--out-dir",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "exp_per_sensor_yiming_k500_fov_signed"
        ),
    )

    p.add_argument("--steps", type=int, default=50000)
    # Eight head-wise first/second derivative graphs are materially heavier
    # than the scalar baseline.  These defaults are conservative; the formal
    # runner exposes them as environment variables so we can scale them to the
    # training GPU after a short profile without changing the method.
    p.add_argument("--batch-x", type=int, default=512)
    p.add_argument("--batch-q", type=int, default=64)
    p.add_argument("--val-batch-x", type=int, default=256)
    p.add_argument("--val-batch-q", type=int, default=64)
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument("--decode-x-chunk", type=int, default=64)

    p.add_argument("--model-arch", choices=["yiming"], default="yiming")
    p.add_argument("--mlp-layers", default="1024,512,256,128,128")
    p.add_argument("--skips", default="")
    p.add_argument(
        "--nerf", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--activation", choices=["relu"], default="relu")
    p.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-patience", type=int, default=5000)
    p.add_argument("--scheduler-threshold", type=float, default=0.01)
    p.add_argument("--scheduler-eps", type=float, default=1e-4)

    p.add_argument("--weight-sdf", type=float, default=5.0)
    p.add_argument("--weight-union", type=float, default=1.0)
    p.add_argument("--weight-eikonal", type=float, default=0.01)
    p.add_argument("--weight-tension", type=float, default=0.01)
    p.add_argument("--weight-grad", type=float, default=0.1)

    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)

    p.add_argument("--near-zero-ratio", type=float, default=0.0)
    p.add_argument("--near-zero-std", type=float, default=0.03)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--resume", default="")
    p.add_argument("--profile", action="store_true")

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="care_visibility_cdf")
    p.add_argument(
        "--wandb-name", default="exp_per_sensor_yiming_k500_fov_signed"
    )
    p.add_argument(
        "--wandb-group", default="per_sensor_yiming_k500_fov_signed"
    )
    p.add_argument(
        "--wandb-tags",
        default="per_sensor,8head,k500,fov,signed,yiming,end_to_end",
    )
    p.add_argument("--wandb-log-every", type=int, default=10)

    args = p.parse_args()
    if args.batch_x <= 0 or args.batch_q <= 0 or args.steps <= 0:
        p.error("steps, batch-x and batch-q must be positive")
    if not 0.0 <= args.near_zero_ratio < 1.0:
        p.error("near-zero-ratio must be in [0,1)")
    return args


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb_run = None
    if args.wandb:
        import wandb
        tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group or None,
            tags=tags,
            config=vars(args),
            dir=args.out_dir,
        )

    dataset = VisibilityQ0Dataset(
        path=args.data, val_count=args.val_count, seed=args.seed
    )
    if dataset.S != NUM_SENSORS:
        raise RuntimeError(
            f"dataset has S={dataset.S}; expected {NUM_SENSORS}"
        )

    # Dataset-level branch coverage is a hard sanity check for mode-preserving
    # training.
    valid_per_sensor = (
        dataset.valid_cpu.any(dim=1).sum(dim=0).cpu().numpy().astype(int)
    )
    print("\n=== Per-sensor dataset coverage ===")
    for s, frame in enumerate(DEFAULT_SENSOR_FRAMES):
        print(
            f"S{s} {frame:28s}: "
            f"{valid_per_sensor[s]}/{dataset.P} points"
        )
    if np.any(valid_per_sensor == 0):
        raise RuntimeError(
            f"at least one sensor has zero q0 coverage: {valid_per_sensor.tolist()}"
        )
    print("===================================\n")

    fov_oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        z_min=args.z_min,
        z_max=args.z_max,
        delta=args.delta,
    )

    mlp_layers = _parse_mlp_layers(args.mlp_layers)
    skips = tuple(
        int(v.strip()) for v in args.skips.split(",") if v.strip()
    )
    model = MLP(
        in_dim=10,
        out_dim=NUM_SENSORS,
        activation=args.activation,
        model_arch=args.model_arch,
        mlp_layers=mlp_layers,
        skips=skips,
        nerf=args.nerf,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=args.scheduler_eps,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.amp and device.type == "cuda")
    )

    weights = {
        "sdf": args.weight_sdf,
        "union": args.weight_union,
        "eikonal": args.weight_eikonal,
        "tension": args.weight_tension,
        "grad": args.weight_grad,
    }

    start_step = 0
    best_val = float("inf")
    if args.resume:
        start_step, best_val = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        print(
            f"[resume] {args.resume} step={start_step} best={best_val:.6f}"
        )

    print("\n=== Per-Sensor Training Config ===")
    print(f"device:          {device}")
    print(f"data:            {args.data}")
    print(f"out_dir:         {args.out_dir}")
    print(f"output:          8 sensor-specific signed CDF heads")
    print(f"initialization:  random / end-to-end")
    print(f"steps:           {args.steps}")
    print(f"batch_x/q:       {args.batch_x} / {args.batch_q}")
    print(f"val_batch_x/q:   {args.val_batch_x} / {args.val_batch_q}")
    print(f"weights:         {weights}")
    print(f"mlp_layers:      {args.mlp_layers}")
    print(f"near_zero_ratio: {args.near_zero_ratio}")
    print(f"amp:             {args.amp}")
    print("==================================\n")

    t_start = time.time()
    last_train_stats = {}

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(
            enabled=(args.amp and device.type == "cuda")
        ):
            loss, train_stats = run_batch(
                model=model,
                dataset=dataset,
                fov_oracle=fov_oracle,
                device=device,
                batch_x=args.batch_x,
                batch_q=args.batch_q,
                split="train",
                weights=weights,
                decode_x_chunk=args.decode_x_chunk,
                near_zero_ratio=args.near_zero_ratio,
                near_zero_std=args.near_zero_std,
                profile=args.profile,
            )

        tb = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step(loss.detach())
        sync_if_cuda(device)
        if args.profile:
            train_stats["time_backward_step"] = float(
                time.perf_counter() - tb
            )

        last_train_stats = train_stats

        if wandb_run is not None and (
            step == 1 or step % args.wandb_log_every == 0
        ):
            payload = {f"train/{k}": v for k, v in train_stats.items()}
            payload["train/lr"] = optimizer.param_groups[0]["lr"]
            wandb_run.log(payload, step=step)

        if step == 1 or step % args.log_every == 0:
            print(
                f"[train] step={step:06d} "
                f"loss={train_stats['loss']:.6f} "
                f"sensor_sdf={train_stats['sensor_sdf_loss']:.6f} "
                f"union={train_stats['union_sdf_loss']:.6f} "
                f"grad={train_stats['grad_loss']:.6f} "
                f"eik={train_stats['eikonal_loss']:.6f} "
                f"tension={train_stats['tension_loss']:.6f} "
                f"winner={train_stats['winner_accuracy']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"elapsed={time.time()-t_start:.1f}s",
                flush=True,
            )
            if args.profile:
                print(
                    f"[profile] decode={train_stats.get('time_decode',-1):.3f}s "
                    f"fov={train_stats.get('time_fov',-1):.3f}s "
                    f"loss_forward={train_stats.get('time_loss_forward',-1):.3f}s "
                    f"backward={train_stats.get('time_backward_step',-1):.3f}s",
                    flush=True,
                )

        if step == 1 or step % args.val_every == 0:
            model.eval()
            # Gradients wrt q are still required in eval; do not use no_grad.
            with torch.cuda.amp.autocast(
                enabled=(args.amp and device.type == "cuda")
            ):
                val_loss, val_stats = run_batch(
                    model=model,
                    dataset=dataset,
                    fov_oracle=fov_oracle,
                    device=device,
                    batch_x=args.val_batch_x,
                    batch_q=args.val_batch_q,
                    split="val",
                    weights=weights,
                    decode_x_chunk=args.decode_x_chunk,
                    near_zero_ratio=0.0,
                    near_zero_std=args.near_zero_std,
                    profile=args.profile,
                )

            val_scalar = float(val_stats["loss"])
            print(
                f"[val]   step={step:06d} "
                f"loss={val_stats['loss']:.6f} "
                f"sensor_sdf={val_stats['sensor_sdf_loss']:.6f} "
                f"union={val_stats['union_sdf_loss']:.6f} "
                f"grad={val_stats['grad_loss']:.6f} "
                f"winner={val_stats['winner_accuracy']:.4f}",
                flush=True,
            )

            if wandb_run is not None:
                payload = {f"val/{k}": v for k, v in val_stats.items()}
                payload["val/best_loss"] = min(best_val, val_scalar)
                wandb_run.log(payload, step=step)

            if val_scalar < best_val:
                best_val = val_scalar
                path = os.path.join(args.out_dir, "best.pt")
                save_checkpoint(
                    path, model, optimizer, scheduler, args, step,
                    best_val, {"train": train_stats, "val": val_stats}
                )
                print(
                    f"[save] best={path} val_loss={best_val:.6f}",
                    flush=True,
                )

        if step % args.save_every == 0:
            path = os.path.join(args.out_dir, f"step_{step:06d}.pt")
            save_checkpoint(
                path, model, optimizer, scheduler, args, step,
                best_val, {"train": train_stats}
            )
            print(f"[save] {path}", flush=True)

    final_path = os.path.join(args.out_dir, "final.pt")
    save_checkpoint(
        final_path, model, optimizer, scheduler, args, args.steps,
        best_val, {"train": last_train_stats}
    )
    print(f"[done] final={final_path} best_val={best_val:.6f}")

    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["final_step"] = args.steps
        wandb_run.finish()


if __name__ == "__main__":
    main()
