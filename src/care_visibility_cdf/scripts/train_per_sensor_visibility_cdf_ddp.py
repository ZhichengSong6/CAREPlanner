#!/usr/bin/env python3
"""4-GPU exact-global-batch trainer for the 8-head per-sensor VisCDF.

The scientific batch semantics intentionally match the scalar Exp1 baseline:

    global x batch = 4000
    shared q batch = 100
    Cartesian pair batch = 4000 x 100 = 400,000 pairs / optimizer step

The 4000 x samples are split across DDP ranks, while the SAME 100 q samples are
broadcast from rank 0 to all ranks.  Optional per-rank x microbatching changes
only memory usage, never the effective/global optimizer batch.

Loss normalization is global (not a mean of rank-local means).  This matters
for the sensor-balanced objective because sensor q0 coverage differs by sensor.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from train_signed_visibility_cdf_pairwise_replace import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    MLP,
    VisibilityQ0Dataset,
    PinocchioFOVOracle,
    _parse_mlp_layers,
    decode_per_sensor_distance_and_grad,
    make_input_pairs,
)
from train_per_sensor_visibility_cdf import (
    NUM_SENSORS,
    per_sensor_signed_targets,
)


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_print(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def setup_distributed():
    if "RANK" not in os.environ:
        raise RuntimeError(
            "This script must be launched with torchrun, e.g. "
            "torchrun --standalone --nproc_per_node=4 ..."
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def broadcast_int64(values, src, device):
    if dist.get_rank() == src:
        t = torch.as_tensor(values, dtype=torch.long, device=device)
    else:
        t = torch.empty(len(values), dtype=torch.long, device=device)
    dist.broadcast(t, src=src)
    return t


def sample_global_indices(
    dataset: VisibilityQ0Dataset,
    split: str,
    global_batch_x: int,
    device: torch.device,
):
    if split == "train":
        pool = dataset.train_indices_cpu
    elif split == "val":
        pool = dataset.val_indices_cpu
    else:
        raise ValueError(split)

    rank = dist.get_rank()
    if rank == 0:
        ridx = torch.randint(
            0, len(pool), (global_batch_x,), dtype=torch.long
        )
        idx_cpu = pool[ridx]
        idx = idx_cpu.to(device=device, non_blocking=True)
    else:
        idx = torch.empty(global_batch_x, dtype=torch.long, device=device)
    dist.broadcast(idx, src=0)
    return idx


def sample_shared_q(
    dataset: VisibilityQ0Dataset,
    batch_q: int,
    device: torch.device,
):
    q_min, q_max = dataset.q_limits(device=device)
    rank = dist.get_rank()
    if rank == 0:
        u = torch.rand((batch_q, dataset.J), device=device)
        q = q_min[None, :] + u * (q_max - q_min)[None, :]
    else:
        q = torch.empty(
            (batch_q, dataset.J), dtype=torch.float32, device=device
        )
    dist.broadcast(q, src=0)
    return q


def materialize_local_x(
    dataset: VisibilityQ0Dataset,
    global_indices: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
):
    n_global = int(global_indices.numel())
    if n_global % world_size != 0:
        raise RuntimeError(
            f"global_batch_x={n_global} must be divisible by world_size={world_size}"
        )
    local_n = n_global // world_size
    start = rank * local_n
    end = start + local_n
    idx_cpu = global_indices[start:end].detach().cpu()

    x = dataset.x_cpu[idx_cpu].to(device=device, non_blocking=True)
    qlib = dataset.qlib_cpu[idx_cpu].to(device=device, non_blocking=True)
    valid = dataset.valid_cpu[idx_cpu].to(device=device, non_blocking=True)
    return x, qlib, valid


def allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def global_normalizers(
    valid_local: torch.Tensor,
    batch_q: int,
):
    # valid_local: [local_x,K,S].  For pure Cartesian random-q Exp1 sampling,
    # availability depends only on x/sensor, then repeats for all shared q.
    has_sensor_x = valid_local.any(dim=1)  # [local_x,S]
    local_sensor_rows = has_sensor_x.sum(dim=0).to(torch.float64) * batch_q
    global_sensor_rows = allreduce_sum(local_sensor_rows)

    local_union_rows = (
        has_sensor_x.any(dim=1).sum().to(torch.float64) * batch_q
    )
    global_union_rows = allreduce_sum(local_union_rows)

    active_heads = global_sensor_rows > 0
    num_active_heads = int(active_heads.sum().item())
    if num_active_heads <= 0:
        raise RuntimeError("No globally supervised sensor head in this batch.")

    return global_sensor_rows, global_union_rows, active_heads


def empty_raw_stats(device):
    return {
        "sdf_sum": torch.zeros(NUM_SENSORS, dtype=torch.float64, device=device),
        "grad_sum": torch.zeros(NUM_SENSORS, dtype=torch.float64, device=device),
        "eik_sum": torch.zeros(NUM_SENSORS, dtype=torch.float64, device=device),
        "tension_sum": torch.zeros(NUM_SENSORS, dtype=torch.float64, device=device),
        "count": torch.zeros(NUM_SENSORS, dtype=torch.float64, device=device),
        "union_sq_sum": torch.zeros((), dtype=torch.float64, device=device),
        "union_count": torch.zeros((), dtype=torch.float64, device=device),
        "winner_correct": torch.zeros((), dtype=torch.float64, device=device),
        "winner_count": torch.zeros((), dtype=torch.float64, device=device),
    }


def add_raw_stats(dst, src):
    for k in dst:
        dst[k] += src[k]


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
    """Differentiable local contribution to the exact global objective.

    DDP averages parameter gradients across ranks.  Therefore each rank's
    differentiable contribution is multiplied by world_size so that the DDP
    average equals the desired global sum/global-count objective.
    """
    x_inputs = inp[:, :3].detach()
    q_inputs = inp[:, 3:10].detach().clone().requires_grad_(True)
    model_inputs = torch.cat([x_inputs, q_inputs], dim=-1)
    pred = ddp_model(model_inputs)

    if pred.shape != target.shape:
        raise RuntimeError(
            f"model output {tuple(pred.shape)} != target {tuple(target.shape)}"
        )

    active_n = int(active_heads.sum().item())
    sensor_sdf = pred.sum() * 0.0
    grad_loss = pred.sum() * 0.0
    eik_loss = pred.sum() * 0.0
    tension_loss = pred.sum() * 0.0

    raw = empty_raw_stats(pred.device)

    for s in range(NUM_SENSORS):
        if not bool(active_heads[s].item()):
            continue
        mask_s = valid_mask[:, s]
        count_s = int(mask_s.sum().item())

        # Value term.
        sq = (pred[:, s] - target[:, s]).square()
        if count_s > 0:
            sq_sum = sq[mask_s].sum()
            denom = global_sensor_rows[s].to(dtype=sq_sum.dtype)
            sensor_sdf = sensor_sdf + sq_sum / denom / active_n
            raw["sdf_sum"][s] += sq_sum.detach().to(torch.float64)
            raw["count"][s] += count_s

        # Per-sensor first derivative.
        grad_pred_s = torch.autograd.grad(
            outputs=pred[:, s],
            inputs=q_inputs,
            grad_outputs=torch.ones_like(pred[:, s]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        if count_s > 0:
            gt_s = target_grad[:, s, :]
            cos_s = F.cosine_similarity(
                grad_pred_s, gt_s, dim=-1, eps=1e-6
            )
            grad_vec = 1.0 - cos_s
            eik_vec = torch.abs(grad_pred_s.norm(2, dim=-1) - 1.0)

            dd_grad_s = torch.autograd.grad(
                outputs=grad_pred_s,
                inputs=q_inputs,
                grad_outputs=torch.ones_like(grad_pred_s),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            tension_vec = dd_grad_s.square().sum(dim=-1)

            denom = global_sensor_rows[s].to(dtype=pred.dtype)
            gsum = grad_vec[mask_s].sum()
            esum = eik_vec[mask_s].sum()
            tsum = tension_vec[mask_s].sum()

            grad_loss = grad_loss + gsum / denom / active_n
            eik_loss = eik_loss + esum / denom / active_n
            tension_loss = tension_loss + tsum / denom / active_n

            raw["grad_sum"][s] += gsum.detach().to(torch.float64)
            raw["eik_sum"][s] += esum.detach().to(torch.float64)
            raw["tension_sum"][s] += tsum.detach().to(torch.float64)

    # Union consistency.  Masking is identical to the single-GPU trainer.
    neg_inf = torch.full_like(pred, -float("inf"))
    pred_rank = torch.where(valid_mask, pred, neg_inf)
    gt_rank = torch.where(valid_mask, target, neg_inf)
    row_valid = valid_mask.any(dim=1)

    pred_union = pred_rank.max(dim=1).values
    gt_union = gt_rank.max(dim=1).values
    union_sq = (pred_union - gt_union).square()
    union_sum = union_sq[row_valid].sum()
    union_denom = global_union_rows.to(dtype=pred.dtype)
    union_loss = union_sum / union_denom

    raw["union_sq_sum"] += union_sum.detach().to(torch.float64)
    raw["union_count"] += row_valid.sum().detach().to(torch.float64)

    with torch.no_grad():
        if torch.any(row_valid):
            gt_winner = gt_rank[row_valid].argmax(dim=1)
            pred_winner = pred_rank[row_valid].argmax(dim=1)
            raw["winner_correct"] += (
                gt_winner == pred_winner
            ).sum().to(torch.float64)
            raw["winner_count"] += row_valid.sum().to(torch.float64)

    global_objective_local = (
        weights["sdf"] * sensor_sdf
        + weights["union"] * union_loss
        + weights["eikonal"] * eik_loss
        + weights["tension"] * tension_loss
        + weights["grad"] * grad_loss
    )

    # DDP averages gradients over ranks.
    backward_loss = float(world_size) * global_objective_local
    return backward_loss, raw


def reduce_and_summarize(
    raw,
    global_sensor_rows,
    global_union_rows,
    active_heads,
    weights,
):
    total = {k: allreduce_sum(v) for k, v in raw.items()}

    active = [s for s in range(NUM_SENSORS) if bool(active_heads[s].item())]
    active_n = max(1, len(active))

    def sensor_balanced(key):
        vals = []
        for s in active:
            denom = float(global_sensor_rows[s].item())
            vals.append(float(total[key][s].item()) / max(denom, 1.0))
        return float(sum(vals) / len(vals)) if vals else float("nan")

    sensor_sdf = sensor_balanced("sdf_sum")
    grad = sensor_balanced("grad_sum")
    eik = sensor_balanced("eik_sum")
    tension = sensor_balanced("tension_sum")
    union = float(total["union_sq_sum"].item()) / max(
        float(global_union_rows.item()), 1.0
    )

    loss = (
        weights["sdf"] * sensor_sdf
        + weights["union"] * union
        + weights["eikonal"] * eik
        + weights["tension"] * tension
        + weights["grad"] * grad
    )
    winner = float(total["winner_correct"].item()) / max(
        float(total["winner_count"].item()), 1.0
    )

    stats = {
        "loss": loss,
        "sensor_sdf_loss": sensor_sdf,
        "union_sdf_loss": union,
        "grad_loss": grad,
        "eikonal_loss": eik,
        "tension_loss": tension,
        "winner_accuracy": winner,
    }
    for s in range(NUM_SENSORS):
        denom = float(global_sensor_rows[s].item())
        stats[f"s{s}_count"] = int(denom)
        if denom > 0:
            stats[f"s{s}_sdf_loss"] = (
                float(total["sdf_sum"][s].item()) / denom
            )
            stats[f"s{s}_grad_loss"] = (
                float(total["grad_sum"][s].item()) / denom
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

    global_sensor_rows, global_union_rows, active_heads = global_normalizers(
        valid_local, batch_q
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

        Bm = end - start
        inp = make_input_pairs(x_m, q_shared)
        target = target.reshape(Bm * batch_q, NUM_SENSORS)
        target_grad = target_grad.reshape(Bm * batch_q, NUM_SENSORS, 7)
        mask = mask.reshape(Bm * batch_q, NUM_SENSORS)

        # Only the last microbatch synchronizes gradients.  Earlier gradients
        # remain local and are included in the final DDP all-reduce.
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

        del d_s, grad_d_s, sign_s, target, target_grad, mask, inp, backward_loss

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
        local_peak = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_t = torch.tensor(local_peak, dtype=torch.float64, device=device)
        dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
        stats["max_peak_allocated_gib"] = float(peak_t.item())
        torch.cuda.reset_peak_memory_stats(device)

    return stats


def save_checkpoint(path, model, optimizer, scheduler, args, step, best_val, stats):
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
            "output_semantics": "per_sensor_signed_visibility_cdf",
            "sensor_frames": list(DEFAULT_SENSOR_FRAMES),
            "joint_names": list(DEFAULT_JOINT_NAMES),
            "out_dim": NUM_SENSORS,
            "distributed_training": {
                "world_size": dist.get_world_size(),
                "exact_global_batch": True,
                "shared_q_across_ranks": True,
            },
        },
        path,
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
    p.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    p.add_argument(
        "--out-dir",
        default=(
            "src/care_visibility_cdf/checkpoints/"
            "per_sensor_e2e_fullbatch_seed0"
        ),
    )

    p.add_argument("--steps", type=int, default=50000)

    # Formal Exp1-matched global batch.
    p.add_argument("--global-batch-x", type=int, default=4000)
    p.add_argument("--batch-q", type=int, default=100)
    p.add_argument(
        "--microbatch-x",
        type=int,
        default=250,
        help=(
            "Per-rank x microbatch. Memory implementation only; "
            "effective batch remains global_batch_x x batch_q."
        ),
    )

    # Match old Exp1 validation batch too.
    p.add_argument("--val-global-batch-x", type=int, default=512)
    p.add_argument("--val-batch-q", type=int, default=100)
    p.add_argument("--val-microbatch-x", type=int, default=128)
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument("--decode-x-chunk", type=int, default=64)

    p.add_argument("--mlp-layers", default="1024,512,256,128,128")
    p.add_argument("--skips", default="")
    p.add_argument(
        "--nerf", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--activation", choices=["relu"], default="relu")
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

    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", action="store_true")

    args = p.parse_args()
    if args.global_batch_x <= 0 or args.batch_q <= 0:
        p.error("global batch dimensions must be positive")
    if args.val_global_batch_x <= 0 or args.val_batch_q <= 0:
        p.error("validation batch dimensions must be positive")
    return args


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()

    try:
        if world_size != 4:
            raise RuntimeError(
                f"Formal configuration requires exactly 4 GPUs, got {world_size}."
            )
        if args.global_batch_x % world_size != 0:
            raise RuntimeError("global-batch-x must be divisible by 4")
        if args.val_global_batch_x % world_size != 0:
            raise RuntimeError("val-global-batch-x must be divisible by 4")

        # Same split on every rank; per-rank RNG is offset only for operations
        # that are not explicitly broadcast.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if is_main():
            os.makedirs(args.out_dir, exist_ok=True)
            with open(os.path.join(args.out_dir, "train_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
        dist.barrier()

        # Each rank loads the same read-only q0 library.  Only q/x/valid/masks
        # are materialized by VisibilityQ0Dataset.
        dataset = VisibilityQ0Dataset(
            path=args.data, val_count=args.val_count, seed=args.seed
        )
        if dataset.S != NUM_SENSORS:
            raise RuntimeError(f"Expected 8 sensors, got {dataset.S}")

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

        layers = _parse_mlp_layers(args.mlp_layers)
        skips = tuple(
            int(v.strip()) for v in args.skips.split(",") if v.strip()
        )
        model = MLP(
            in_dim=10,
            out_dim=NUM_SENSORS,
            activation=args.activation,
            model_arch="yiming",
            mlp_layers=layers,
            skips=skips,
            nerf=args.nerf,
        ).to(device)

        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
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
            "union": args.weight_union,
            "eikonal": args.weight_eikonal,
            "tension": args.weight_tension,
            "grad": args.weight_grad,
        }

        rank_print("\n=== EXACT GLOBAL-BATCH 8-HEAD VISCDF ===")
        rank_print(f"world_size:            {world_size}")
        rank_print(f"global x batch:        {args.global_batch_x}")
        rank_print(f"shared q batch:        {args.batch_q}")
        rank_print(
            f"pairs / optimizer:     {args.global_batch_x * args.batch_q}"
        )
        rank_print(
            f"local x / GPU:         {args.global_batch_x // world_size}"
        )
        rank_print(f"microbatch x / GPU:    {args.microbatch_x}")
        rank_print(
            "microbatches / GPU:    "
            f"{math.ceil((args.global_batch_x//world_size)/args.microbatch_x)}"
        )
        rank_print(f"steps:                 {args.steps}")
        rank_print(f"weights:               {weights}")
        rank_print("shared q across ranks:  YES")
        rank_print("global loss norm:       YES")
        rank_print("=========================================\n")

        best_val = float("inf")
        t_start = time.time()
        last_train = {}

        for step in range(1, args.steps + 1):
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

            # All ranks see the same globally reduced training loss.
            scheduler.step(train_stats["loss"])
            last_train = train_stats

            if is_main() and (step == 1 or step % args.log_every == 0):
                extra = (
                    f" peak={train_stats.get('max_peak_allocated_gib', float('nan')):.2f}GiB"
                    if args.profile else ""
                )
                print(
                    f"[train] step={step:06d} "
                    f"loss={train_stats['loss']:.6f} "
                    f"sensor_sdf={train_stats['sensor_sdf_loss']:.6f} "
                    f"union={train_stats['union_sdf_loss']:.6f} "
                    f"grad={train_stats['grad_loss']:.6f} "
                    f"eik={train_stats['eikonal_loss']:.6f} "
                    f"tension={train_stats['tension_loss']:.6f} "
                    f"winner={train_stats['winner_accuracy']:.4f} "
                    f"batch={train_stats['pairs_per_update']} "
                    f"step_time={train_stats['elapsed_sec']:.2f}s "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"elapsed={(time.time()-t_start)/3600:.2f}h"
                    f"{extra}",
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
                val_scalar = val_stats["loss"]

                if is_main():
                    print(
                        f"[val]   step={step:06d} "
                        f"loss={val_stats['loss']:.6f} "
                        f"sensor_sdf={val_stats['sensor_sdf_loss']:.6f} "
                        f"union={val_stats['union_sdf_loss']:.6f} "
                        f"grad={val_stats['grad_loss']:.6f} "
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
                    os.path.join(args.out_dir, f"step_{step:06d}.pt"),
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
        rank_print(f"[done] best_val={best_val:.6f}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
