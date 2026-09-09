#!/usr/bin/env python3
"""Four-GPU, from-scratch hierarchical9 baseline for CAREPlanner.

Repository q0 loading, FOV oracle, target construction and Cartesian sampling
are reused, NOT reimplemented. Runtime / URDF / safety components are untouched.
This entrypoint never loads a scalar/8-head checkpoint as initialization.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import HierarchicalVisibilityCDF
from objective import LossWeights, STAT_NAMES, counts_from_mask, loss_for_microbatch, summarize

HERE = Path(__file__).resolve().parent
FORMAT = "careplanner_hierarchical9_scratch_v1"
BASE_COMMIT = "841a4a992386388aa0cdb00c885f3c607c8e997c"


def log(*items: Any) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*items, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state()}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu())


def atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def synchronized_check(ok: bool | torch.Tensor, message: str, device: torch.device) -> None:
    flag = torch.as_tensor(ok, device=device, dtype=torch.int32).reshape(())
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if flag.item() != 1:
        raise RuntimeError(message)


def load_repo_api(repo: Path) -> dict:
    scripts = repo / "src/care_visibility_cdf/scripts"
    files = [scripts / name for name in (
        "train_signed_visibility_cdf_pairwise_replace.py",
        "train_per_sensor_visibility_cdf.py",
        "train_per_sensor_visibility_cdf_ddp.py",
    )]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing repository dependency: {path}")
    sys.path.insert(0, str(scripts))
    from train_signed_visibility_cdf_pairwise_replace import (
        VisibilityQ0Dataset, PinocchioFOVOracle, DEFAULT_JOINT_NAMES,
        DEFAULT_SENSOR_FRAMES, decode_per_sensor_distance_and_grad, make_input_pairs,
    )
    from train_per_sensor_visibility_cdf import per_sensor_signed_targets
    from train_per_sensor_visibility_cdf_ddp import (
        sample_global_indices, sample_shared_q, materialize_local_x,
    )
    return locals()


def fingerprint(repo: Path, args: argparse.Namespace) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    paths = list(HERE.glob("*.py")) + [
        repo / "src/care_visibility_cdf/scripts" / name for name in (
            "train_signed_visibility_cdf_pairwise_replace.py",
            "train_per_sensor_visibility_cdf.py", "train_per_sensor_visibility_cdf_ddp.py",
        )
    ]
    hashes = {str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p):
              hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    # Avoid claiming data integrity from file size alone, and avoid hashing 7GB
    # four times. Record identity metadata; use an external dataset hash if needed.
    return {"repo_commit": commit, "reviewed_base_commit": BASE_COMMIT,
            "source_sha256": hashes, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "data": str(Path(args.data).resolve()), "data_bytes": Path(args.data).stat().st_size,
            "data_mtime_ns": Path(args.data).stat().st_mtime_ns,
            "urdf_sha256": hashlib.sha256(Path(args.urdf).read_bytes()).hexdigest()}


def prepare_batch(api: dict, dataset: Any, oracle: Any, args: argparse.Namespace,
                  device: torch.device, split: str) -> tuple[list, torch.Tensor]:
    """Decode once; counts include the actual finite-label mask.

    Target tensors are small relative to activation graphs. Keeping targets for
    all local microbatches permits exact global normalization and retrying AMP
    overflow without silently changing the optimizer's batch.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    bx = args.global_batch_x if split == "train" else args.val_global_batch_x
    bq = args.batch_q if split == "train" else args.val_batch_q
    micro = args.microbatch_x if split == "train" else args.val_microbatch_x
    # Fixed validation cohort with saved/restored train RNG. Does not consume
    # training random draws. This is a diagnostic, not the final qualification.
    saved_rng = rng_state() if split == "val" else None
    try:
        if split == "val":
            seed_everything(args.seed + 100003)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            indices = api["sample_global_indices"](dataset, split, bx, device)
            q = api["sample_shared_q"](dataset, bq, device)
            x, qlib, valid = api["materialize_local_x"](dataset, indices, rank, world, device)
            sensor_masks = dataset.sensor_masks(device=device)
            batches = []
            counts = torch.zeros(9, dtype=torch.float32, device=device)
            for start in range(0, len(x), micro):
                end = min(start + micro, len(x))
                ds, dgrad, has = api["decode_per_sensor_distance_and_grad"](
                    qlib=qlib[start:end], valid=valid[start:end], q_query=q,
                    sensor_masks=sensor_masks, x_chunk=args.decode_x_chunk,
                )
                _, sign = oracle.signed_fov_margins(x[start:end], q)
                target, target_grad, mask = api["per_sensor_signed_targets"](ds, dgrad, sign, has)
                mask = mask.reshape(-1, 8).contiguous()
                batches.append((
                    api["make_input_pairs"](x[start:end], q).float(),
                    target.reshape(-1, 8).float().contiguous(),
                    target_grad.reshape(-1, 8, 7).float().contiguous(), mask,
                ))
                counts += counts_from_mask(mask)
                del ds, dgrad, sign, target, target_grad
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            if counts[0] <= 0:
                raise RuntimeError("No supervised rows in global batch")
            return batches, counts
    finally:
        if saved_rng is not None:
            restore_rng(saved_rng)


def amp_context(amp: str):
    return torch.autocast("cuda", enabled=amp != "off",
                          dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


def run_prepared_batch(model: DDP, batches: list, counts: torch.Tensor,
                       weights: LossWeights, args: argparse.Namespace,
                       optimizer: Any = None, scaler: Any = None) -> dict:
    device = counts.device
    training = optimizer is not None
    world = dist.get_world_size()
    # GradScaler overflow retries reuse these exact samples, and do NOT count
    # as optimizer updates. Never report 50k attempted/skipped steps as 50k updates.
    for attempt in range(args.max_amp_retries + 1):
        if training:
            optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros((9, len(STAT_NAMES)), dtype=torch.float64, device=device)
        model.train(training)
        for i, batch in enumerate(batches):
            context = model.no_sync() if training and i < len(batches) - 1 else contextlib.nullcontext()
            with context, torch.enable_grad():
                with amp_context(args.amp):
                    loss, stats = loss_for_microbatch(
                        model if training else model.module, *batch, counts, weights,
                        world_size=world, training=training,
                    )
                synchronized_check(
                    torch.isfinite(loss.detach()) & torch.isfinite(stats).all(),
                    "Non-finite field/input-gradient loss. Stopping, not saving a false final.pt. "
                    "Check the data; for a precision diagnostic use AMP=bf16 or AMP=off.", device,
                )
                if training:
                    scaler.scale(loss).backward()
            totals += stats
            del loss, stats
        if not training:
            break
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        local_finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all().int()
        dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
        if local_finite.item() == 1:
            # No clipping: preserve the baseline optimizer objective/behavior.
            scaler.step(optimizer)
            scaler.update()
            break
        if args.amp != "fp16" or attempt == args.max_amp_retries:
            raise RuntimeError("Non-finite parameter gradients after AMP retries; stopping")
        new_scale = scaler.get_scale() / 2.0
        scaler.update(new_scale=new_scale)  # Resets per-optimizer state after unscale_.
        log(f"[amp-retry] same batch, attempt={attempt + 1}, scale={new_scale:g}")
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if not torch.equal(totals[:, 0].float(), counts):
        raise RuntimeError("Accumulated supervision counts do not match global denominators")
    stats = summarize(totals, weights)
    if not math.isfinite(stats["loss"]):
        raise RuntimeError("Non-finite globally reduced loss")
    stats["amp_retries"] = attempt if training else 0
    stats["grad_scale"] = scaler.get_scale() if training else None
    return stats


def print_stats(split: str, step: int, stats: dict, lr: float, seconds: float,
                peak: float) -> None:
    h = stats["heads"]
    def number(head: str, key: str) -> float:
        return h.get(head, {}).get(key, float("nan"))
    log(f"[{split}] step={step:06d} loss={stats['loss']:.6f} "
        f"U_mae={number('union','mae'):.5f} U_cos={number('union','grad_cosine'):.4f} "
        f"S6_cos={number('s6','grad_cosine'):.4f} S7_cos={number('s7','grad_cosine'):.4f} "
        f"winner={stats['winner_accuracy']:.4f} lr={lr:.3e} "
        f"seconds={seconds:.2f} peak={peak:.2f}GiB scale={stats.get('grad_scale')}")


def save_checkpoint(path: Path, model: DDP, optimizer: Any, scheduler: Any,
                    scaler: Any, args: argparse.Namespace, step: int,
                    best_val: float, stats: dict, metadata: dict, api: dict) -> None:
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng_state())
    error = [None]
    if dist.get_rank() == 0:
        try:
            saved_args = vars(args).copy()
            saved_args.update(shared_layers="1024,512,256", branch_layers="128,128", nerf=True)
            atomic_save({
                "format": FORMAT, "model_state": model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(), "rng_by_rank": states,
                "args": saved_args, "step": step, "best_val": best_val, "stats": stats,
                "initialization": "random_from_scratch", "frozen_parameters": 0,
                "output_semantics": "hierarchical_union_plus_per_sensor_signed_visibility_cdf",
                "output_layout": {"union_index": 0, "sensor_slice": [1, 9],
                                  "sensor_frames": list(api["DEFAULT_SENSOR_FRAMES"])},
                "joint_names": list(api["DEFAULT_JOINT_NAMES"]), "out_dim": 9,
                "architecture": {"shared_layers": "1024,512,256", "branch_layers": "128,128",
                                 "nerf": True, "activation": "relu", "dedicated_union_head": True,
                                 "sensor_specific_nonlinear_heads": True},
                "distributed_training": {"world_size": dist.get_world_size(),
                    "exact_global_batch": True, "shared_q_across_ranks": True,
                    "pairs_per_update": args.global_batch_x * args.batch_q},
                "metadata": metadata,
            }, path)
        except Exception as exc:
            error[0] = repr(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"Checkpoint write failed: {error[0]}")


def resume_training(path: str, model: DDP, optimizer: Any, scheduler: Any,
                    scaler: Any, args: argparse.Namespace) -> tuple[int, float]:
    # Full checkpoint contains RNG/optimizer Python objects. Load ONLY a trusted
    # checkpoint created by this script, not a downloaded third-party pickle.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != FORMAT or ckpt.get("initialization") != "random_from_scratch":
        raise RuntimeError("Only this script's interrupted scratch run may be resumed; "
                           "scalar/8-head/other initialization checkpoints are rejected")
    previous = ckpt["args"]
    changing = {"steps", "out_dir", "resume_training", "log_every", "save_every", "repo",
                "max_amp_retries"}
    for key, value in vars(args).items():
        if key not in changing and previous.get(key) != value:
            raise RuntimeError(f"Resume config mismatch: {key}: {previous.get(key)!r} != {value!r}")
    if ckpt["distributed_training"]["world_size"] != dist.get_world_size():
        raise RuntimeError("Resume requires the same world size")
    model.module.load_state_dict(ckpt["model_state"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    restore_rng(ckpt["rng_by_rank"][dist.get_rank()])
    return int(ckpt["step"]), float(ckpt["best_val"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=str(HERE.parents[1]))
    p.add_argument("--data", default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz")
    p.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--global-batch-x", type=int, default=4000)
    p.add_argument("--batch-q", type=int, default=100)
    p.add_argument("--microbatch-x", type=int, default=250)
    p.add_argument("--val-global-batch-x", type=int, default=512)
    p.add_argument("--val-batch-q", type=int, default=100)
    p.add_argument("--val-microbatch-x", type=int, default=128)
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument("--decode-x-chunk", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    p.add_argument("--max-amp-retries", type=int, default=16)
    p.add_argument("--weight-sdf", type=float, default=5.0)
    p.add_argument("--weight-grad", type=float, default=0.1)
    p.add_argument("--weight-eikonal", type=float, default=0.01)
    p.add_argument("--weight-tension", type=float, default=0.01)
    p.add_argument("--weight-union-objective", type=float, default=1.0)
    p.add_argument("--weight-sensor-objective", type=float, default=1.0)
    p.add_argument("--weight-consistency", type=float, default=0.1)
    p.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=0.20)
    p.add_argument("--z-max", type=float, default=0.70)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume-training", default="", help="Explicit recovery of this script's own run only")
    args = p.parse_args()
    for name in ("steps", "global_batch_x", "batch_q", "microbatch_x", "val_global_batch_x",
                 "val_batch_q", "val_microbatch_x", "val_count", "decode_x_chunk", "log_every",
                 "val_every", "save_every"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.max_amp_retries < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error("Invalid max-amp-retries or learning rate")
    if not 0 < args.z_min < args.z_max or args.delta < 0:
        p.error("Invalid FOV depth limits or conservative delta")
    args.repo = str(Path(args.repo).expanduser().resolve())
    for name in ("data", "urdf", "out_dir"):
        value = Path(getattr(args, name)).expanduser()
        setattr(args, name, str((Path(args.repo) / value).resolve() if not value.is_absolute() else value.resolve()))
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with python -m torch.distributed.run --standalone --nproc_per_node=4")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        world = dist.get_world_size()
        if world != 4:
            raise RuntimeError(f"Formal and smoke training require 4 GPUs, got {world}")
        if args.global_batch_x % world or args.val_global_batch_x % world:
            raise ValueError("Global x batches must be divisible by 4")
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by this CUDA environment")
        weights = LossWeights(**{key: getattr(args, "weight_" + key) for key in asdict(LossWeights())})
        weights.validate()
        for path in (args.data, args.urdf):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        out = Path(args.out_dir)
        # Fresh training refuses a nonempty directory. No implicit warmstart.
        ok = bool(args.resume_training) or not out.exists() or not any(out.iterdir())
        synchronized_check(ok, f"Output directory is nonempty: {out}; choose a new OUT", device)
        if dist.get_rank() == 0:
            out.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        seed_everything(args.seed)
        api = load_repo_api(Path(args.repo))
        dataset = api["VisibilityQ0Dataset"](path=args.data, val_count=args.val_count, seed=args.seed)
        if dataset.S != 8 or dataset.J != 7:
            raise RuntimeError("Expected 8 sensors and 7 joints")
        if len(dataset.train_indices_cpu) == 0 or len(dataset.val_indices_cpu) == 0:
            raise RuntimeError("Train and validation spatial splits must both be nonempty")
        coverage = dataset.valid_cpu.any(dim=1).sum(dim=0)
        if torch.any(coverage == 0):
            raise RuntimeError(f"Sensor without q0 coverage: {coverage.tolist()}")
        oracle = api["PinocchioFOVOracle"](
            urdf_path=args.urdf, joint_names=api["DEFAULT_JOINT_NAMES"],
            sensor_frames=api["DEFAULT_SENSOR_FRAMES"], horizontal_fov_deg=args.horizontal_fov_deg,
            vertical_fov_deg=args.vertical_fov_deg, z_min=args.z_min, z_max=args.z_max, delta=args.delta,
        )
        # ALL parameters are newly initialized; no scalar checkpoint is read.
        model = HierarchicalVisibilityCDF().to(device)
        if not all(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Unexpected frozen parameter")
        ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
                  find_unused_parameters=False)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5000, threshold=0.01,
            threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-4,
        )
        # Match the existing scaler's default starting scale. Retries are logged.
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16", init_scale=65536.0)
        metadata = fingerprint(Path(args.repo), args)
        step, best_val = 0, float("inf")
        if args.resume_training:
            step, best_val = resume_training(args.resume_training, ddp, optimizer, scheduler, scaler, args)
        if step >= args.steps:
            raise ValueError("Requested steps must exceed the resumed step")
        if dist.get_rank() == 0:
            with (out / ("train_args.json" if step == 0 else f"resume_args_{step}.json")).open("w") as f:
                json.dump({**vars(args), "weights": asdict(weights), "metadata": metadata}, f, indent=2)
        log("\n=== HIERARCHICAL9 SCRATCH / EXACT GLOBAL BATCH ===")
        log("initialization:", "random_from_scratch" if step == 0 else f"explicit recovery at step {step}")
        log("shared: 30 -> 1024 -> 512 -> 256; nine decoders: 256 -> 128 -> 128 -> 1")
        log(f"parameters={model.parameter_count():,}; trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        log(f"global_x={args.global_batch_x}, shared_q={args.batch_q}, pairs/update={args.global_batch_x * args.batch_q}")
        log(f"local_x={args.global_batch_x // world}, microbatch_x={args.microbatch_x}, AMP={args.amp}")
        log("loss weights:", asdict(weights))
        log("coverage:", coverage.tolist(), "out:", out)
        log("Final comparison uses final.pt; runtime is NOT changed.\n")
        last_val: dict = {}
        train_stats: dict = {}
        for step in range(step + 1, args.steps + 1):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            batches, counts = prepare_batch(api, dataset, oracle, args, device, "train")
            train_stats = run_prepared_batch(ddp, batches, counts, weights, args, optimizer, scaler)
            del batches
            scheduler.step(train_stats["loss"])
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 1024**3, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            train_stats.update(pairs_per_update=args.global_batch_x * args.batch_q,
                               elapsed_sec=elapsed, max_peak_allocated_gib=float(peak.item()))
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print_stats("train", step, train_stats, optimizer.param_groups[0]["lr"], elapsed, float(peak))
                if dist.get_rank() == 0:
                    with (out / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps({"split": "train", "step": step, **train_stats}, allow_nan=False) + "\n")
            if step == 1 or step % args.val_every == 0 or step == args.steps:
                started = time.perf_counter()
                batches, counts = prepare_batch(api, dataset, oracle, args, device, "val")
                last_val = run_prepared_batch(ddp, batches, counts, weights, args)
                del batches
                torch.cuda.synchronize(device)
                print_stats("val", step, last_val, optimizer.param_groups[0]["lr"], time.perf_counter()-started, float(peak))
                if dist.get_rank() == 0:
                    with (out / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps({"split": "val", "step": step, **last_val}, allow_nan=False) + "\n")
                best_val = min(best_val, last_val["loss"])
            # latest.pt permits recovery, never substitutes for the final comparison.
            if step == 1 or step % args.save_every == 0 or step == args.steps:
                save_checkpoint(out / "latest.pt", ddp, optimizer, scheduler, scaler, args,
                                step, best_val, {"train": train_stats, "val": last_val}, metadata, api)
        save_checkpoint(out / "final.pt", ddp, optimizer, scheduler, scaler, args,
                        args.steps, best_val, {"train": train_stats, "val": last_val}, metadata, api)
        log(f"[done] successful_optimizer_updates={args.steps} final={out / 'final.pt'}")
    finally:
        # No barrier here: a failed rank must not strand the surviving ranks.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
