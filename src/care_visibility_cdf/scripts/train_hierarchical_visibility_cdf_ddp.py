#!/usr/bin/env python3
"""4-GPU exact-global-batch trainer for hierarchical 9-output VisCDF.

Scientific batch semantics are kept identical to the scalar Exp1 and the
previous exact 8-head run:

    global x batch = 4000
    shared q batch = 100
    Cartesian pairs = 400,000 / optimizer step
    4 GPUs, each rank receives 1000 x values and the SAME 100 q values

Model:
    shared trunk 30 -> 1024 -> 512 -> 256
    dedicated union branch 256 -> 128 -> 128 -> 1
    8 dedicated sensor branches 256 -> 128 -> 128 -> 1

Loss:
    sensor objective:
        5*SDF + 0.1*gradient + 0.01*eikonal + 0.01*tension
    union objective (direct old scalar target):
        5*SDF + 0.1*gradient + 0.01*eikonal + 0.01*tension
    weak hierarchy consistency:
        0.1 * MSE(union_head, max(valid sensor heads))

The dedicated union head therefore receives the same type of value/gradient
supervision that made the original scalar field useful for projection, while
sensor-specific nonlinear heads retain mode identity for fallback.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from train_signed_visibility_cdf_pairwise_replace import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    VisibilityQ0Dataset,
    PinocchioFOVOracle,
    decode_per_sensor_distance_and_grad,
    make_input_pairs,
)
from train_per_sensor_visibility_cdf import (
    NUM_SENSORS,
    per_sensor_signed_targets,
)
from train_per_sensor_visibility_cdf_ddp import (
    allreduce_sum,
    cleanup_distributed,
    global_normalizers,
    is_main,
    materialize_local_x,
    rank_print,
    sample_global_indices,
    sample_shared_q,
    setup_distributed,
)
from hierarchical_visibility_cdf_model import (
    HierarchicalVisibilityCDF,
    OUTPUT_DIM,
)


def empty_raw_stats(device):
    return {
        "sensor_sdf_sum": torch.zeros(
            NUM_SENSORS, dtype=torch.float64, device=device
        ),
        "sensor_grad_sum": torch.zeros(
            NUM_SENSORS, dtype=torch.float64, device=device
        ),
        "sensor_eik_sum": torch.zeros(
            NUM_SENSORS, dtype=torch.float64, device=device
        ),
        "sensor_tension_sum": torch.zeros(
            NUM_SENSORS, dtype=torch.float64, device=device
        ),
        "sensor_count": torch.zeros(
            NUM_SENSORS, dtype=torch.float64, device=device
        ),
        "union_sdf_sum": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "union_grad_sum": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "union_eik_sum": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "union_tension_sum": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "consistency_sum": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "union_count": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "winner_correct": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
        "winner_count": torch.zeros(
            (), dtype=torch.float64, device=device
        ),
    }


def add_raw_stats(dst, src):
    for key in dst:
        dst[key] += src[key]


def _second_derivative_tension(
    grad_pred: torch.Tensor,
    q_inputs: torch.Tensor,
) -> torch.Tensor:
    dd = torch.autograd.grad(
        outputs=grad_pred,
        inputs=q_inputs,
        grad_outputs=torch.ones_like(grad_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return dd.square().sum(dim=-1)


def loss_for_microbatch(
    ddp_model: DDP,
    inp: torch.Tensor,
    target: torch.Tensor,
    target_grad: torch.Tensor,
    valid_mask: torch.Tensor,
    global_sensor_rows: torch.Tensor,
    global_union_rows: torch.Tensor,
    active_heads: torch.Tensor,
    weights: Dict[str, float],
    world_size: int,
):
    """Exact globally normalized differentiable objective for one local microbatch."""
    x_inputs = inp[:, :3].detach()
    q_inputs = inp[:, 3:10].detach().clone().requires_grad_(True)
    model_inputs = torch.cat([x_inputs, q_inputs], dim=-1)

    pred_all = ddp_model(model_inputs)
    if pred_all.ndim != 2 or pred_all.shape[1] != OUTPUT_DIM:
        raise RuntimeError(
            f"hierarchical model must output [N,{OUTPUT_DIM}], "
            f"got {tuple(pred_all.shape)}"
        )
    if target.ndim != 2 or target.shape[1] != NUM_SENSORS:
        raise RuntimeError(f"target shape invalid: {tuple(target.shape)}")

    pred_union = pred_all[:, 0]
    pred_sensor = pred_all[:, 1:]

    raw = empty_raw_stats(pred_all.device)
    active_n = int(active_heads.sum().item())
    if active_n <= 0:
        raise RuntimeError("No active sensor heads in global batch.")

    zero = pred_all.sum() * 0.0
    sensor_sdf = zero
    sensor_grad = zero
    sensor_eik = zero
    sensor_tension = zero

    # ------------------------------------------------------------------
    # 8 independent sensor objectives.
    # ------------------------------------------------------------------
    for s in range(NUM_SENSORS):
        if not bool(active_heads[s].item()):
            continue

        mask_s = valid_mask[:, s]
        count_s = int(mask_s.sum().item())
        grad_pred_s = torch.autograd.grad(
            outputs=pred_sensor[:, s],
            inputs=q_inputs,
            grad_outputs=torch.ones_like(pred_sensor[:, s]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if count_s <= 0:
            continue

        denom = global_sensor_rows[s].to(dtype=pred_all.dtype)

        sdf_vec = (pred_sensor[:, s] - target[:, s]).square()
        sdf_sum = sdf_vec[mask_s].sum()
        sensor_sdf = sensor_sdf + sdf_sum / denom / active_n

        gt_grad_s = target_grad[:, s, :]
        cos_s = F.cosine_similarity(
            grad_pred_s, gt_grad_s, dim=-1, eps=1e-6
        )
        grad_vec = 1.0 - cos_s
        eik_vec = torch.abs(grad_pred_s.norm(2, dim=-1) - 1.0)
        tension_vec = _second_derivative_tension(
            grad_pred_s, q_inputs
        )

        grad_sum = grad_vec[mask_s].sum()
        eik_sum = eik_vec[mask_s].sum()
        tension_sum = tension_vec[mask_s].sum()

        sensor_grad = sensor_grad + grad_sum / denom / active_n
        sensor_eik = sensor_eik + eik_sum / denom / active_n
        sensor_tension = sensor_tension + tension_sum / denom / active_n

        raw["sensor_sdf_sum"][s] += sdf_sum.detach().to(torch.float64)
        raw["sensor_grad_sum"][s] += grad_sum.detach().to(torch.float64)
        raw["sensor_eik_sum"][s] += eik_sum.detach().to(torch.float64)
        raw["sensor_tension_sum"][s] += (
            tension_sum.detach().to(torch.float64)
        )
        raw["sensor_count"][s] += float(count_s)

    # ------------------------------------------------------------------
    # Dedicated union head target and gradient.
    # GT is EXACTLY the old scalar definition: max over available sensors.
    # ------------------------------------------------------------------
    neg_inf = torch.full_like(target, -float("inf"))
    gt_rank = torch.where(valid_mask, target, neg_inf)
    pred_sensor_rank = torch.where(valid_mask, pred_sensor, neg_inf)
    row_valid = valid_mask.any(dim=1)

    gt_union, gt_winner = gt_rank.max(dim=1)
    # Avoid inf values outside the valid cohort. They are never included in
    # the loss, but finite placeholders keep diagnostics/autograd clean.
    gt_union = torch.where(
        row_valid, gt_union, torch.zeros_like(gt_union)
    )

    gather_index = gt_winner[:, None, None].expand(-1, 1, 7)
    gt_union_grad = torch.gather(
        target_grad, dim=1, index=gather_index
    ).squeeze(1)

    union_denom = global_union_rows.to(dtype=pred_all.dtype)
    local_union_count = int(row_valid.sum().item())

    grad_pred_union = torch.autograd.grad(
        outputs=pred_union,
        inputs=q_inputs,
        grad_outputs=torch.ones_like(pred_union),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    if local_union_count > 0:
        union_sdf_vec = (pred_union - gt_union).square()
        union_sdf_sum = union_sdf_vec[row_valid].sum()
        union_sdf = union_sdf_sum / union_denom

        union_cos = F.cosine_similarity(
            grad_pred_union, gt_union_grad, dim=-1, eps=1e-6
        )
        union_grad_vec = 1.0 - union_cos
        union_eik_vec = torch.abs(
            grad_pred_union.norm(2, dim=-1) - 1.0
        )
        union_tension_vec = _second_derivative_tension(
            grad_pred_union, q_inputs
        )

        union_grad_sum = union_grad_vec[row_valid].sum()
        union_eik_sum = union_eik_vec[row_valid].sum()
        union_tension_sum = union_tension_vec[row_valid].sum()

        union_grad = union_grad_sum / union_denom
        union_eik = union_eik_sum / union_denom
        union_tension = union_tension_sum / union_denom

        # Weak structural consistency. Sensor max is masked exactly like GT.
        pred_sensor_max = pred_sensor_rank.max(dim=1).values
        consistency_vec = (pred_union - pred_sensor_max).square()
        consistency_sum = consistency_vec[row_valid].sum()
        consistency = consistency_sum / union_denom

        raw["union_sdf_sum"] += union_sdf_sum.detach().to(torch.float64)
        raw["union_grad_sum"] += union_grad_sum.detach().to(torch.float64)
        raw["union_eik_sum"] += union_eik_sum.detach().to(torch.float64)
        raw["union_tension_sum"] += (
            union_tension_sum.detach().to(torch.float64)
        )
        raw["consistency_sum"] += (
            consistency_sum.detach().to(torch.float64)
        )
        raw["union_count"] += float(local_union_count)

        with torch.no_grad():
            pred_winner = pred_sensor_rank[row_valid].argmax(dim=1)
            raw["winner_correct"] += (
                pred_winner == gt_winner[row_valid]
            ).sum().to(torch.float64)
            raw["winner_count"] += float(local_union_count)
    else:
        union_sdf = zero
        union_grad = zero
        union_eik = zero
        union_tension = zero
        consistency = zero

    sensor_objective = (
        weights["sdf"] * sensor_sdf
        + weights["grad"] * sensor_grad
        + weights["eikonal"] * sensor_eik
        + weights["tension"] * sensor_tension
    )
    union_objective = (
        weights["sdf"] * union_sdf
        + weights["grad"] * union_grad
        + weights["eikonal"] * union_eik
        + weights["tension"] * union_tension
    )

    global_objective_local = (
        weights["sensor_objective"] * sensor_objective
        + weights["union_objective"] * union_objective
        + weights["consistency"] * consistency
    )

    # DDP averages gradients across ranks; compensate so the average equals
    # the desired globally normalized sum.
    backward_loss = float(world_size) * global_objective_local
    return backward_loss, raw


def reduce_and_summarize(
    raw,
    global_sensor_rows,
    global_union_rows,
    active_heads,
    weights,
):
    total = {key: allreduce_sum(value) for key, value in raw.items()}
    active = [
        s for s in range(NUM_SENSORS)
        if bool(active_heads[s].item())
    ]

    def sensor_balanced(key):
        vals = []
        for s in active:
            denom = float(global_sensor_rows[s].item())
            vals.append(
                float(total[key][s].item()) / max(denom, 1.0)
            )
        return float(sum(vals) / len(vals)) if vals else float("nan")

    sensor_sdf = sensor_balanced("sensor_sdf_sum")
    sensor_grad = sensor_balanced("sensor_grad_sum")
    sensor_eik = sensor_balanced("sensor_eik_sum")
    sensor_tension = sensor_balanced("sensor_tension_sum")

    union_denom = max(float(global_union_rows.item()), 1.0)
    union_sdf = float(total["union_sdf_sum"].item()) / union_denom
    union_grad = float(total["union_grad_sum"].item()) / union_denom
    union_eik = float(total["union_eik_sum"].item()) / union_denom
    union_tension = (
        float(total["union_tension_sum"].item()) / union_denom
    )
    consistency = (
        float(total["consistency_sum"].item()) / union_denom
    )

    sensor_objective = (
        weights["sdf"] * sensor_sdf
        + weights["grad"] * sensor_grad
        + weights["eikonal"] * sensor_eik
        + weights["tension"] * sensor_tension
    )
    union_objective = (
        weights["sdf"] * union_sdf
        + weights["grad"] * union_grad
        + weights["eikonal"] * union_eik
        + weights["tension"] * union_tension
    )
    loss = (
        weights["sensor_objective"] * sensor_objective
        + weights["union_objective"] * union_objective
        + weights["consistency"] * consistency
    )

    winner_accuracy = (
        float(total["winner_correct"].item())
        / max(float(total["winner_count"].item()), 1.0)
    )

    stats = {
        "loss": float(loss),
        "sensor_objective": float(sensor_objective),
        "union_objective": float(union_objective),
        "sensor_sdf_loss": float(sensor_sdf),
        "sensor_grad_loss": float(sensor_grad),
        "sensor_eikonal_loss": float(sensor_eik),
        "sensor_tension_loss": float(sensor_tension),
        "union_sdf_loss": float(union_sdf),
        "union_grad_loss": float(union_grad),
        "union_eikonal_loss": float(union_eik),
        "union_tension_loss": float(union_tension),
        "consistency_loss": float(consistency),
        "winner_accuracy": float(winner_accuracy),
    }

    for s in range(NUM_SENSORS):
        denom = float(global_sensor_rows[s].item())
        stats[f"s{s}_count"] = int(denom)
        if denom > 0:
            stats[f"s{s}_sdf_loss"] = (
                float(total["sensor_sdf_sum"][s].item()) / denom
            )
            stats[f"s{s}_grad_loss"] = (
                float(total["sensor_grad_sum"][s].item()) / denom
            )
        else:
            stats[f"s{s}_sdf_loss"] = float("nan")
            stats[f"s{s}_grad_loss"] = float("nan")
    return stats


def run_global_batch(
    ddp_model,
    dataset,
    oracle,
    device,
    rank,
    world_size,
    global_batch_x,
    batch_q,
    microbatch_x,
    split,
    weights,
    decode_x_chunk,
    scaler=None,
    optimizer=None,
    do_backward=True,
    profile=False,
):
    if global_batch_x % world_size != 0:
        raise RuntimeError(
            f"global_batch_x={global_batch_x} must divide world_size={world_size}"
        )
    local_x_n = global_batch_x // world_size
    if microbatch_x <= 0 or microbatch_x > local_x_n:
        microbatch_x = local_x_n

    t0 = time.perf_counter()

    global_idx = sample_global_indices(
        dataset, split, global_batch_x, device
    )
    q_shared = sample_shared_q(dataset, batch_q, device)
    x_local, qlib_local, valid_local = materialize_local_x(
        dataset, global_idx, rank, world_size, device
    )
    sensor_masks = dataset.sensor_masks(device=device)

    global_sensor_rows, global_union_rows, active_heads = (
        global_normalizers(valid_local, batch_q)
    )

    raw_total = empty_raw_stats(device)
    n_micro = math.ceil(local_x_n / microbatch_x)

    if do_backward:
        if optimizer is None or scaler is None:
            raise RuntimeError("Training requires optimizer and scaler.")
        optimizer.zero_grad(set_to_none=True)

    for mi, start in enumerate(range(0, local_x_n, microbatch_x)):
        end = min(start + microbatch_x, local_x_n)
        x_m = x_local[start:end]
        qlib_m = qlib_local[start:end]
        valid_m = valid_local[start:end]

        with torch.no_grad():
            d_s, grad_d_s, has_s = decode_per_sensor_distance_and_grad(
                qlib=qlib_m,
                valid=valid_m,
                q_query=q_shared,
                sensor_masks=sensor_masks,
                x_chunk=decode_x_chunk,
            )
            _, sign_s = oracle.signed_fov_margins(x_m, q_shared)
            target, target_grad, mask = per_sensor_signed_targets(
                d_s, grad_d_s, sign_s, has_s
            )

        bx_m = end - start
        inp = make_input_pairs(x_m, q_shared)
        target = target.reshape(
            bx_m * batch_q, NUM_SENSORS
        )
        target_grad = target_grad.reshape(
            bx_m * batch_q, NUM_SENSORS, 7
        )
        mask = mask.reshape(
            bx_m * batch_q, NUM_SENSORS
        )

        sync_ctx = (
            contextlib.nullcontext()
            if (not do_backward or mi == n_micro - 1)
            else ddp_model.no_sync()
        )
        with sync_ctx:
            with torch.cuda.amp.autocast(enabled=True):
                backward_loss, raw = loss_for_microbatch(
                    ddp_model=ddp_model,
                    inp=inp,
                    target=target,
                    target_grad=target_grad,
                    valid_mask=mask,
                    global_sensor_rows=global_sensor_rows,
                    global_union_rows=global_union_rows,
                    active_heads=active_heads,
                    weights=weights,
                    world_size=world_size,
                )
            if do_backward:
                scaler.scale(backward_loss).backward()

        add_raw_stats(raw_total, raw)

        del (
            d_s,
            grad_d_s,
            sign_s,
            target,
            target_grad,
            mask,
            inp,
            backward_loss,
        )

    if do_backward:
        scaler.step(optimizer)
        scaler.update()

    stats = reduce_and_summarize(
        raw_total,
        global_sensor_rows,
        global_union_rows,
        active_heads,
        weights,
    )
    stats["global_batch_x"] = int(global_batch_x)
    stats["batch_q"] = int(batch_q)
    stats["pairs_per_update"] = int(global_batch_x * batch_q)
    stats["local_batch_x"] = int(local_x_n)
    stats["microbatch_x"] = int(microbatch_x)
    stats["microbatches_per_rank"] = int(n_micro)
    stats["elapsed_sec"] = float(time.perf_counter() - t0)

    if profile:
        local_peak = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        peak_t = torch.tensor(
            local_peak, dtype=torch.float64, device=device
        )
        dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
        stats["max_peak_allocated_gib"] = float(peak_t.item())
        torch.cuda.reset_peak_memory_stats(device)

    return stats


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    args,
    step,
    best_val,
    stats,
):
    if not is_main():
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bare = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model_state": bare.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
            "step": int(step),
            "best_val": float(best_val),
            "stats": stats,
            "output_semantics": (
                "hierarchical_union_plus_per_sensor_signed_visibility_cdf"
            ),
            "output_layout": {
                "union_index": 0,
                "sensor_slice": [1, 9],
                "sensor_frames": list(DEFAULT_SENSOR_FRAMES),
            },
            "joint_names": list(DEFAULT_JOINT_NAMES),
            "out_dim": OUTPUT_DIM,
            "architecture": {
                "shared_layers": args.shared_layers,
                "branch_layers": args.branch_layers,
                "nerf": bool(args.nerf),
                "dedicated_union_head": True,
                "sensor_specific_nonlinear_heads": True,
            },
            "distributed_training": {
                "world_size": dist.get_world_size(),
                "exact_global_batch": True,
                "shared_q_across_ranks": True,
                "pairs_per_update": (
                    int(args.global_batch_x) * int(args.batch_q)
                ),
            },
        },
        path,
    )


def load_resume(path, ddp_model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    bare = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    bare.load_state_dict(ckpt["model_state"], strict=True)
    if ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return int(ckpt.get("step", 0)), float(
        ckpt.get("best_val", float("inf"))
    )


def parse_args():
    p = argparse.ArgumentParser()
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
            "hierarchical9_e2e_fullbatch_seed0"
        ),
    )

    p.add_argument("--steps", type=int, default=50000)

    # Exact Exp1-matched batch.
    p.add_argument("--global-batch-x", type=int, default=4000)
    p.add_argument("--batch-q", type=int, default=100)
    p.add_argument("--microbatch-x", type=int, default=250)

    p.add_argument("--val-global-batch-x", type=int, default=512)
    p.add_argument("--val-batch-q", type=int, default=100)
    p.add_argument("--val-microbatch-x", type=int, default=128)
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument("--decode-x-chunk", type=int, default=64)

    p.add_argument(
        "--shared-layers", default="1024,512,256"
    )
    p.add_argument(
        "--branch-layers", default="128,128"
    )
    p.add_argument(
        "--nerf", action=argparse.BooleanOptionalAction, default=True
    )

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--scheduler-patience", type=int, default=5000)
    p.add_argument("--scheduler-threshold", type=float, default=0.01)
    p.add_argument("--scheduler-eps", type=float, default=1e-4)

    # Same core field weights as scalar/8-head training.
    p.add_argument("--weight-sdf", type=float, default=5.0)
    p.add_argument("--weight-grad", type=float, default=0.1)
    p.add_argument("--weight-eikonal", type=float, default=0.01)
    p.add_argument("--weight-tension", type=float, default=0.01)
    p.add_argument("--weight-sensor-objective", type=float, default=1.0)
    p.add_argument("--weight-union-objective", type=float, default=1.0)
    p.add_argument("--weight-consistency", type=float, default=0.1)

    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--resume", default="")

    args = p.parse_args()
    for name in (
        "global_batch_x",
        "batch_q",
        "val_global_batch_x",
        "val_batch_q",
        "steps",
    ):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_','-')} must be positive")
    return args


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()

    try:
        if world_size != 4:
            raise RuntimeError(
                f"Formal hierarchical run requires 4 GPUs, got {world_size}"
            )
        if args.global_batch_x % world_size != 0:
            raise RuntimeError("global-batch-x must be divisible by 4")
        if args.val_global_batch_x % world_size != 0:
            raise RuntimeError("val-global-batch-x must be divisible by 4")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if is_main():
            os.makedirs(args.out_dir, exist_ok=True)
            with open(
                os.path.join(args.out_dir, "train_args.json"), "w"
            ) as f:
                json.dump(vars(args), f, indent=2)
        dist.barrier()

        dataset = VisibilityQ0Dataset(
            path=args.data,
            val_count=args.val_count,
            seed=args.seed,
        )
        if dataset.S != NUM_SENSORS:
            raise RuntimeError(
                f"Expected {NUM_SENSORS} sensors, got {dataset.S}"
            )

        valid_per_sensor = (
            dataset.valid_cpu.any(dim=1)
            .sum(dim=0)
            .cpu()
            .numpy()
            .astype(int)
        )
        if is_main():
            print("\n=== Per-sensor dataset coverage ===")
            for s, frame in enumerate(DEFAULT_SENSOR_FRAMES):
                print(
                    f"S{s} {frame:28s}: "
                    f"{valid_per_sensor[s]}/{dataset.P}"
                )
            print("===================================\n")
        if np.any(valid_per_sensor == 0):
            raise RuntimeError(
                "At least one sensor has zero supervision coverage."
            )

        oracle = PinocchioFOVOracle(
            urdf_path=args.urdf,
            joint_names=DEFAULT_JOINT_NAMES,
            sensor_frames=DEFAULT_SENSOR_FRAMES,
            horizontal_fov_deg=args.horizontal_fov_deg,
            vertical_fov_deg=args.vertical_fov_deg,
            z_min=args.z_min,
            z_max=args.z_max,
            delta=args.delta,
        )

        model = HierarchicalVisibilityCDF(
            in_dim=10,
            shared_layers=args.shared_layers,
            branch_layers=args.branch_layers,
            nerf=args.nerf,
            num_sensors=NUM_SENSORS,
        ).to(device)

        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        optimizer = torch.optim.Adam(
            ddp_model.parameters(), lr=args.lr
        )
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
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        weights = {
            "sdf": args.weight_sdf,
            "grad": args.weight_grad,
            "eikonal": args.weight_eikonal,
            "tension": args.weight_tension,
            "sensor_objective": args.weight_sensor_objective,
            "union_objective": args.weight_union_objective,
            "consistency": args.weight_consistency,
        }

        start_step = 0
        best_val = float("inf")
        if args.resume:
            start_step, best_val = load_resume(
                args.resume,
                ddp_model,
                optimizer,
                scheduler,
                device,
            )
            rank_print(
                f"[resume] {args.resume} step={start_step} "
                f"best={best_val:.6f}"
            )

        rank_print("\n=== HIERARCHICAL 9-OUTPUT VISCDF / EXACT DDP ===")
        rank_print(f"world_size:             {world_size}")
        rank_print("node intent:            4 x RTX 3090")
        rank_print(
            f"shared trunk:           30 -> {args.shared_layers}"
        )
        rank_print(
            f"9 branches:             256 -> {args.branch_layers} -> 1"
        )
        rank_print(
            f"parameters:             {model.parameter_count():,}"
        )
        rank_print(
            f"global x / shared q:    {args.global_batch_x} / {args.batch_q}"
        )
        rank_print(
            f"pairs / optimizer:      {args.global_batch_x * args.batch_q}"
        )
        rank_print(
            f"local x / GPU:          {args.global_batch_x // world_size}"
        )
        rank_print(
            f"microbatch x / GPU:     {args.microbatch_x}"
        )
        rank_print(f"steps:                  {args.steps}")
        rank_print(f"weights:                {weights}")
        rank_print("union supervision:      direct scalar target + gradient")
        rank_print("sensor supervision:     8 independent signed CDFs")
        rank_print("initialization:         random / end-to-end")
        rank_print("================================================\n")

        t_start = time.time()
        last_train = {}

        for step in range(start_step + 1, args.steps + 1):
            ddp_model.train()
            train_stats = run_global_batch(
                ddp_model=ddp_model,
                dataset=dataset,
                oracle=oracle,
                device=device,
                rank=rank,
                world_size=world_size,
                global_batch_x=args.global_batch_x,
                batch_q=args.batch_q,
                microbatch_x=args.microbatch_x,
                split="train",
                weights=weights,
                decode_x_chunk=args.decode_x_chunk,
                scaler=scaler,
                optimizer=optimizer,
                do_backward=True,
                profile=args.profile,
            )
            scheduler.step(train_stats["loss"])
            last_train = train_stats

            if is_main() and (
                step == 1 or step % args.log_every == 0
            ):
                peak = (
                    f" peak={train_stats.get('max_peak_allocated_gib', float('nan')):.2f}GiB"
                    if args.profile else ""
                )
                print(
                    f"[train] step={step:06d} "
                    f"loss={train_stats['loss']:.6f} "
                    f"S_sdf={train_stats['sensor_sdf_loss']:.6f} "
                    f"S_grad={train_stats['sensor_grad_loss']:.6f} "
                    f"U_sdf={train_stats['union_sdf_loss']:.6f} "
                    f"U_grad={train_stats['union_grad_loss']:.6f} "
                    f"cons={train_stats['consistency_loss']:.6f} "
                    f"winner={train_stats['winner_accuracy']:.4f} "
                    f"step_time={train_stats['elapsed_sec']:.2f}s "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"elapsed={(time.time()-t_start)/3600:.2f}h"
                    f"{peak}",
                    flush=True,
                )

            if step == 1 or step % args.val_every == 0:
                ddp_model.eval()
                val_stats = run_global_batch(
                    ddp_model=ddp_model,
                    dataset=dataset,
                    oracle=oracle,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    global_batch_x=args.val_global_batch_x,
                    batch_q=args.val_batch_q,
                    microbatch_x=args.val_microbatch_x,
                    split="val",
                    weights=weights,
                    decode_x_chunk=args.decode_x_chunk,
                    scaler=None,
                    optimizer=None,
                    do_backward=False,
                    profile=args.profile,
                )
                val_scalar = float(val_stats["loss"])

                if is_main():
                    print(
                        f"[val]   step={step:06d} "
                        f"loss={val_stats['loss']:.6f} "
                        f"S_sdf={val_stats['sensor_sdf_loss']:.6f} "
                        f"S_grad={val_stats['sensor_grad_loss']:.6f} "
                        f"U_sdf={val_stats['union_sdf_loss']:.6f} "
                        f"U_grad={val_stats['union_grad_loss']:.6f} "
                        f"cons={val_stats['consistency_loss']:.6f} "
                        f"winner={val_stats['winner_accuracy']:.4f}",
                        flush=True,
                    )

                if val_scalar < best_val:
                    best_val = val_scalar
                    save_checkpoint(
                        os.path.join(args.out_dir, "best.pt"),
                        ddp_model,
                        optimizer,
                        scheduler,
                        args,
                        step,
                        best_val,
                        {"train": train_stats, "val": val_stats},
                    )

            if step % args.save_every == 0:
                save_checkpoint(
                    os.path.join(
                        args.out_dir, f"step_{step:06d}.pt"
                    ),
                    ddp_model,
                    optimizer,
                    scheduler,
                    args,
                    step,
                    best_val,
                    {"train": train_stats},
                )

        save_checkpoint(
            os.path.join(args.out_dir, "final.pt"),
            ddp_model,
            optimizer,
            scheduler,
            args,
            args.steps,
            best_val,
            {"train": last_train},
        )
        rank_print(
            f"[done] final={os.path.join(args.out_dir, 'final.pt')} "
            f"best_val={best_val:.6f}"
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
