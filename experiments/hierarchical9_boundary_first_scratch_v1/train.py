#!/usr/bin/env python3
"""One formal 4-GPU scratch run, using the already verified Global/V3/V4 cache."""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from data import Cache, EXPECTED_CACHE, FIELDS, sha, source_identity, write_json
from objective import loss_and_metrics, average_parameter_gradients

HERE = Path(__file__).resolve().parent
FORMAT = "care_h9_boundary_first_scratch_v1"


def config() -> dict:
    c = json.loads((HERE / "protocol.json").read_text())
    if c["initialization"] != "random_from_scratch" or c["world_size"] != 4:
        raise ValueError("Wrong formal protocol")
    for key in ("steps", "warmup_updates", "log_every", "val_every", "checkpoint_every"):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(key)
    if c["warmup_updates"] >= c["steps"]:
        raise ValueError("Warmup must precede cosine decay")
    for key in ("lr_peak", "lr_final", "gradient_clip_norm", "local_scale_rad", "v3_scale_rad", "sign_temperature"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(key)
    if any(not math.isfinite(w) or w <= 0 for w in c["weights"].values()):
        raise ValueError("Invalid weights")
    for kind in FIELDS:
        if c["batch"][kind] <= 0 or c["batch"][kind] % 4:
            raise ValueError("Training batch must divide four ranks")
        if kind != "v3" and (c["val_batch"][kind] <= 0 or c["val_batch"][kind] % 4):
            raise ValueError("Validation batch must divide four ranks")
    return c


def learning_rate(c: dict, step: int) -> float:
    if step <= c["warmup_updates"]:
        return c["lr_peak"] * step/c["warmup_updates"]
    t = (step-c["warmup_updates"])/(c["steps"]-c["warmup_updates"])
    return c["lr_final"] + (c["lr_peak"]-c["lr_final"])*(1+math.cos(math.pi*t))/2


def tensor_metrics(values: dict, *, detailed: bool) -> dict:
    out = {}
    for key, value in values.items():
        if value.numel() != 1 and not detailed:
            continue
        x = value.detach().cpu()
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite metric: {key}")
        out[key] = float(x.item()) if x.numel() == 1 else x.tolist()
    return out


def draw(cache: Cache, c: dict, split: str, step: int, rank: int, world: int, device):
    sizes = c["batch"] if split == "train" else c["val_batch"]
    return {k: cache.batch(split, k, cache.ids(split, k, sizes[k], c["stream_seed"],
                        step, rank, world), device) for k in FIELDS}


def state_digest(model) -> str:
    h = hashlib.sha256()
    for key, t in model.state_dict().items():
        h.update(key.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_checkpoint(path: Path, payload: dict):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    write_json(path.with_suffix(path.suffix+".sha256.json"), {"sha256": sha(path)})


def checked_load(path: Path):
    guard = json.loads(path.with_suffix(path.suffix+".sha256.json").read_text())
    if sha(path) != guard["sha256"]:
        raise ValueError(f"Checkpoint checksum failed: {path}")
    # Only load our own locally produced, trusted checkpoint.
    return torch.load(path, map_location="cpu", weights_only=False)


def preflight(cache_root: Path):
    c = config()
    cache = Cache(cache_root, verify=True)
    print(f"[preflight] cache={cache.identity}", flush=True)
    print(f"[preflight] package={source_identity(HERE)}; scratch updates={c['steps']}", flush=True)
    print(json.dumps(cache.manifest["counts"], indent=2), flush=True)


def train(args):
    from model import build_model
    c = config()
    if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Run through run.sh; four allocated GPUs are required")
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != c["world_size"]:
        raise ValueError("This protocol requires exactly four ranks")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    out = args.out.resolve()
    identity = source_identity(HERE)
    try:
        error = [None]
        if rank == 0:
            try:
                out.mkdir(parents=True, exist_ok=True)
                if (out / "final.pt").exists():
                    raise FileExistsError("Completed final checkpoint exists; not resuming")
                if args.resume is None and ((out / "latest.pt").exists() or (out / "run.json").exists()):
                    raise FileExistsError("Run has started; use explicit resume")
                cache = Cache(args.cache, verify=True)
                cache.build_indices(out / "indices")
            except Exception as exc:
                error[0] = repr(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(error[0])
        cache = Cache(args.cache)
        cache.load_indices(out / "indices", verify=True)
        random.seed(c["seed"])
        np.random.seed(c["seed"])
        torch.manual_seed(c["seed"])
        torch.cuda.manual_seed_all(c["seed"])
        model = build_model("R1").to(device=device, dtype=torch.float32)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=c["lr_peak"])
        initial_hash = state_digest(model)
        start, best_score, best_step = 0, float("inf"), None
        if args.resume is not None:
            cp = checked_load(args.resume.resolve())
            expected = {"format": FORMAT, "cache_identity": cache.identity,
                        "index_identity": cache.index_identity, "package_identity": identity,
                        "world_size": world, "protocol": c}
            for key, value in expected.items():
                if cp.get(key) != value:
                    raise ValueError(f"Resume mismatch: {key}")
            model.load_state_dict(cp["model_state"], strict=True)
            optimizer.load_state_dict(cp["optimizer_state"])
            start = int(cp["step"])
            best_score, best_step = cp["best_score"], cp["best_step"]
            initial_hash = cp["initial_state_sha256"]
        if not 0 <= start < c["steps"]:
            raise ValueError("No remaining updates")
        run = {"status": "RUNNING", "format": FORMAT, "initialization": c["initialization"],
               "initial_state_sha256": initial_hash, "package_identity": identity,
               "cache_identity": cache.identity, "index_identity": cache.index_identity,
               "world_size": world, "protocol": c, "architecture": model.architecture(),
               "supervision": "global sign + valid V3 + stratified local V4 + zero/normal on boundary",
               "new_labels_generated": False, "solver_evaluation": "NOT_RUN",
               "resumed_from_step": start}
        if rank == 0:
            write_json(out / "run.json", run)
            print("[start] random-init boundary-first; no R1/V2 weights loaded", flush=True)
            print(json.dumps(run, indent=2), flush=True)
        last_val = {}
        for step in range(start+1, c["steps"]+1):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            batch = draw(cache, c, "train", step, rank, world, device)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(c, step)
            optimizer.zero_grad(set_to_none=True)
            loss, values = loss_and_metrics(model, batch, c, training=True)
            finite = torch.isfinite(loss.detach()).int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            average_parameter_gradients(model)
            raw_gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            del batch, loss
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter()-t0, device=device)
            peak = torch.tensor(torch.cuda.max_memory_allocated(device)/2**30, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if step == 1 or step % c["log_every"] == 0 or step == c["steps"]:
                tm = tensor_metrics(values, detailed=False)
                tm.update(split="train", step=step, seconds=float(elapsed), peak_gib=float(peak),
                          lr=learning_rate(c, step), parameter_grad_norm_before_clip=float(raw_gnorm),
                          cumulative_rows={k: step*c["batch"][k] for k in FIELDS})
                if rank == 0:
                    print("[train] " + json.dumps(tm), flush=True)
                    with (out / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(tm, allow_nan=False)+"\n")
            del values
            is_best = False
            if step == 1 or step % c["val_every"] == 0 or step == c["steps"]:
                model.eval()
                vb = draw(cache, c, "val", 0, rank, world, device)
                # enable_grad is required to evaluate the actual input gradient.
                with torch.enable_grad():
                    vl, vv = loss_and_metrics(model, vb, c, training=False)
                last_val = tensor_metrics(vv, detailed=True)
                last_val.update(split="val", step=step, v3_population="all validation rows",
                                other_populations="fixed stratified/local and random/global samples")
                del vl, vv, vb
                score = last_val["selection_score"]
                if score < best_score:
                    best_score, best_step, is_best = score, step, True
                if rank == 0:
                    print("[val] " + json.dumps(last_val), flush=True)
                    with (out / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(last_val, allow_nan=False)+"\n")
                    write_json(out / "validation_latest.json", last_val)
                    if is_best:
                        write_json(out / "validation_best.json", last_val)
            save = is_best or step == 1 or step % c["checkpoint_every"] == 0 or step == c["steps"]
            if save and rank == 0:
                payload = {"format": FORMAT, "completed": step == c["steps"], "step": step,
                    "initialization": c["initialization"], "initial_state_sha256": initial_hash,
                    "architecture": model.architecture(), "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "protocol": c,
                    "cache_identity": cache.identity, "index_identity": cache.index_identity,
                    "package_identity": identity, "world_size": world,
                    "best_score": best_score, "best_step": best_step, "val": last_val,
                    "sampling": "stateless seed/step/rank/world; no dropout or RNG-dependent layers"}
                atomic_checkpoint(out / "latest.pt", payload)
                if is_best:
                    atomic_checkpoint(out / "best_val.pt", payload)
                if step % c["checkpoint_every"] == 0:
                    atomic_checkpoint(out / f"step_{step:06d}.pt", payload)
                if step == c["steps"]:
                    atomic_checkpoint(out / "final.pt", payload)
                run.update(successful_updates=step, best_step=best_step, best_score=best_score,
                           cumulative_rows={k: step*c["batch"][k] for k in FIELDS})
                if step == c["steps"]:
                    run.update(status="COMPLETE", final_sha256=sha(out / "final.pt"),
                               best_val_sha256=sha(out / "best_val.pt"))
                write_json(out / "run.json", run)
            if save:
                dist.barrier()
        if rank == 0:
            print(f"[done] {out}; solver success NOT_RUN", flush=True)
    finally:
        dist.destroy_process_group()


def verify(out: Path):
    run = json.loads((out / "run.json").read_text())
    if run.get("status") != "COMPLETE" or run.get("successful_updates") != run["protocol"]["steps"]:
        raise ValueError("Formal run not complete")
    for name, key in (("final.pt", "final_sha256"), ("best_val.pt", "best_val_sha256")):
        if sha(out / name) != run[key]:
            raise ValueError(f"Corrupt {name}")
    print("[verified] formal COMPLETE; final=" + run["final_sha256"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["preflight", "train", "verify", "identity"])
    p.add_argument("--cache", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if args.command == "identity":
        print(source_identity(HERE)); return
    if args.command in ("preflight", "train") and args.cache is None:
        p.error("--cache required")
    if args.command in ("train", "verify") and args.out is None:
        p.error("--out required")
    if args.command == "preflight": preflight(args.cache)
    elif args.command == "verify": verify(args.out)
    else: train(args)


if __name__ == "__main__":
    main()
