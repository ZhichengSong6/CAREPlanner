#!/usr/bin/env python3
"""Four-GPU scratch-50k trainer for the R0/R1/R2 architecture/routing study.

Scientific controls:
- all arms start from random initialization (no V1/P0/E checkpoint loading),
- 50,000 successful optimizer updates,
- exact global Cartesian batch 4000 x 100 = 400k pairs/update,
- same original H9 objective and optimizer/scheduler,
- same model seed and an architecture-independent training-stream seed,
- rolling SHA256 over every global x-index batch + shared q batch.

R0/R1 use ordinary joint backprop. R2 keeps all private sensor paths on the full
original objective every step, but blocks sensor->shared-early parameter gradients
in that main pass and adds exactly one sensor's full original head objective to the
shared early trunk per step, round-robin S0..S7. Input-q derivatives are preserved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import importlib.util
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = REPO / "experiments/hierarchical9_scratch_v1"
FORMAT = "care_h9_scratch50k_r012_v1"
ARMS = ("R0", "R1", "R2")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the frozen upstream scratch trainer helpers without letting its absolute
# imports accidentally resolve this experiment's local model.py.
_scratch_model = _load("r012_upstream_model", SCRATCH / "model.py")
_scratch_objective = _load("r012_upstream_objective", SCRATCH / "objective.py")
_old_model = sys.modules.get("model")
_old_objective = sys.modules.get("objective")
sys.modules["model"] = _scratch_model
sys.modules["objective"] = _scratch_objective
try:
    base = _load("r012_upstream_train", SCRATCH / "train.py")
finally:
    if _old_model is None: sys.modules.pop("model", None)
    else: sys.modules["model"] = _old_model
    if _old_objective is None: sys.modules.pop("objective", None)
    else: sys.modules["objective"] = _old_objective

models = _load("r012_models", HERE / "model.py")
obj = _scratch_objective


def log(*items: Any) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*items, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class StreamChain:
    """Architecture-independent rolling digest of global train sample identities."""
    def __init__(self, api: dict, initial: str = "0" * 64, count: int = 0) -> None:
        self.api = api
        self.chain = initial
        self.count = int(count)
        self.capture = False
        self.pending = b""
        self._idx = api["sample_global_indices"]
        self._q = api["sample_shared_q"]

    def install(self) -> None:
        tracker = self
        def indices(dataset, split, global_batch_x, device):
            out = tracker._idx(dataset, split, global_batch_x, device)
            tracker.capture = split == "train"
            if tracker.capture and dist.get_rank() == 0:
                tracker.pending = out.detach().cpu().numpy().astype(np.int64, copy=False).tobytes()
            return out
        def qsample(dataset, batch_q, device):
            out = tracker._q(dataset, batch_q, device)
            if tracker.capture:
                if dist.get_rank() == 0:
                    qbytes = out.detach().float().cpu().numpy().astype(np.float32, copy=False).tobytes()
                    chunk = hashlib.sha256(tracker.pending + qbytes).digest()
                    tracker.chain = hashlib.sha256(bytes.fromhex(tracker.chain) + chunk).hexdigest()
                    tracker.count += 1
                tracker.capture = False
                tracker.pending = b""
            return out
        self.api["sample_global_indices"] = indices
        self.api["sample_shared_q"] = qsample


def selected_sensor_local_objective(model, batch, counts9, weights, sensor_id: int) -> torch.Tensor:
    """Rank-local piece of one sensor's full original head objective.

    Denominator is global. Summing gradients across ranks therefore gives L_s.
    Selecting one s per step makes the shared update an unbiased/round-robin
    replacement for mean_s L_s without simultaneous sensor-sensor interference.
    """
    inputs, target, target_grad, sensor_mask = batch
    valid = sensor_mask[:, sensor_id].bool()
    denom = counts9[sensor_id + 1].float()
    if denom <= 0 or not valid.any():
        return inputs.sum() * 0.0
    q = inputs[:, 3:10].detach().float().clone().requires_grad_(True)
    x = inputs[:, :3].detach().float()
    y = model.forward_sensor(torch.cat((x, q), dim=1), sensor_id, freeze_sensor_early=False).float()
    value_error = y[valid] - target[valid, sensor_id].float()
    sdf = value_error.square().sum(dtype=torch.float32)
    grad_q = torch.autograd.grad(y.sum(), q, create_graph=True, retain_graph=True)[0].float()
    cosine = F.cosine_similarity(grad_q[valid], target_grad[valid, sensor_id].float(), dim=-1, eps=1e-6)
    grad = (1.0 - cosine).sum(dtype=torch.float32)
    norm = torch.linalg.vector_norm(grad_q[valid], dim=-1)
    eik = (norm - 1.0).abs().sum(dtype=torch.float32)
    if weights.tension > 0:
        hvec = torch.autograd.grad(grad_q.sum(), q, create_graph=True, retain_graph=True)[0].float()
        tension = hvec[valid].square().sum(dtype=torch.float32)
    else:
        tension = y.sum() * 0.0
    return weights.sensor_objective * (
        weights.sdf * sdf + weights.grad * grad + weights.eikonal * eik + weights.tension * tension
    ) / denom


def run_r2_batch(ddp: DDP, batches: list, counts: torch.Tensor, weights, args, optimizer, scaler, step: int) -> dict:
    device = counts.device
    world = dist.get_world_size()
    selected = (step - 1) % 8
    early = ddp.module.early_parameters()
    for attempt in range(args.max_amp_retries + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros((9, len(obj.STAT_NAMES)), dtype=torch.float64, device=device)
        early_aux = [torch.zeros_like(p, dtype=torch.float32) for p in early]
        ddp.train(True)
        for i, batch in enumerate(batches):
            # Main objective: all private branches train normally. Sensor paths use
            # detached early *parameters* (not detached activations), preserving q derivatives.
            context = ddp.no_sync() if i < len(batches) - 1 else __import__("contextlib").nullcontext()
            with ddp.module.sensor_early_parameter_frozen():
                with context, torch.enable_grad(), base.amp_context(args.amp):
                    loss, stats = obj.loss_for_microbatch(
                        ddp, *batch, counts, weights, world_size=world, training=True
                    )
                base.synchronized_check(
                    torch.isfinite(loss.detach()) & torch.isfinite(stats).all(),
                    "Non-finite R2 main objective", device,
                )
                scaler.scale(loss).backward()
            # Auxiliary selected sensor -> shared-early only, in FP32.
            with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
                aux = selected_sensor_local_objective(ddp.module, batch, counts, weights, selected)
                gs = torch.autograd.grad(aux, early, allow_unused=False)
            for dst, g in zip(early_aux, gs):
                dst.add_(g.detach().float())
            totals += stats
            del loss, stats, aux, gs

        # local pieces / global denominator -> SUM across ranks gives global selected L_s gradient.
        for g in early_aux:
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
        scaler.unscale_(optimizer)
        for p, g in zip(early, early_aux):
            if p.grad is None: p.grad = g.to(dtype=p.dtype)
            else: p.grad.add_(g.to(dtype=p.grad.dtype))
        gradients = [p.grad for p in ddp.parameters() if p.grad is not None]
        local_finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all().int()
        dist.all_reduce(local_finite, op=dist.ReduceOp.MIN)
        if local_finite.item() == 1:
            scaler.step(optimizer)
            scaler.update()
            break
        if args.amp != "fp16" or attempt == args.max_amp_retries:
            raise RuntimeError("Non-finite R2 parameter gradients after AMP retries")
        new_scale = scaler.get_scale() / 2.0
        scaler.update(new_scale=new_scale)
        log(f"[amp-retry] R2 same batch attempt={attempt + 1} scale={new_scale:g}")

    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if not torch.equal(totals[:, 0].float(), counts):
        raise RuntimeError("R2 accumulated counts != global denominators")
    result = obj.summarize(totals, weights)
    result.update(amp_retries=attempt, grad_scale=scaler.get_scale(), selected_shared_sensor=selected)
    return result


def source_fingerprints() -> dict[str, str]:
    paths = [HERE / n for n in ("model.py", "train.py")] + [SCRATCH / "objective.py", SCRATCH / "train.py"]
    return {str(p.relative_to(REPO)): sha256_file(p) for p in paths}


def save_checkpoint(path: Path, ddp: DDP, optimizer, scheduler, scaler, args, step: int,
                    best_val: float, last_stats: dict, stream: StreamChain, metadata: dict) -> None:
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, base.rng_state())
    payload = None
    if dist.get_rank() == 0:
        payload = {
            "format": FORMAT, "arm": args.arm, "completed": path.name == "final.pt",
            "model_state": ddp.module.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "rng_by_rank": states, "args": vars(args).copy(), "step": int(step),
            "best_val": float(best_val), "stats": last_stats,
            "initialization": "random_from_scratch", "frozen_parameters": 0,
            "architecture": ddp.module.architecture(),
            "routing": "all_head_joint" if args.arm in ("R0", "R1") else "union_plus_one_sensor_round_robin_to_shared",
            "training_stream_sha256": stream.chain, "training_stream_updates": stream.count,
            "source_sha256": source_fingerprints(), "metadata": metadata,
            "distributed_training": {"world_size": 4, "exact_global_batch": True,
                "shared_q_across_ranks": True, "pairs_per_update": args.global_batch_x * args.batch_q},
        }
        base.atomic_save(payload, path)
    dist.barrier()


def load_resume(path: Path, ddp: DDP, optimizer, scheduler, scaler, args):
    cp = torch.load(path, map_location="cpu", weights_only=False)
    if cp.get("format") != FORMAT or cp.get("arm") != args.arm or cp.get("initialization") != "random_from_scratch":
        raise RuntimeError("Resume checkpoint lineage mismatch")
    previous = cp["args"]
    mutable = {"steps", "resume_training", "out_dir", "log_every", "val_every", "save_every"}
    for k, v in vars(args).items():
        if k not in mutable and previous.get(k) != v:
            raise RuntimeError(f"Resume config mismatch {k}: {previous.get(k)!r} != {v!r}")
    if cp["distributed_training"]["world_size"] != 4:
        raise RuntimeError("Resume requires 4 GPUs")
    ddp.module.load_state_dict(cp["model_state"], strict=True)
    optimizer.load_state_dict(cp["optimizer_state"]); scheduler.load_state_dict(cp["scheduler_state"]); scaler.load_state_dict(cp["scaler_state"])
    base.restore_rng(cp["rng_by_rank"][dist.get_rank()])
    return int(cp["step"]), float(cp["best_val"]), cp.get("training_stream_sha256", "0"*64), int(cp.get("training_stream_updates", 0))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--data", required=True); p.add_argument("--urdf", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=50000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream-seed", type=int, default=190915)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--amp", choices=("fp16","bf16","off"), default="fp16")
    p.add_argument("--max-amp-retries", type=int, default=16)
    p.add_argument("--global-batch-x", type=int, default=4000); p.add_argument("--batch-q", type=int, default=100); p.add_argument("--microbatch-x", type=int, default=250)
    p.add_argument("--val-global-batch-x", type=int, default=512); p.add_argument("--val-batch-q", type=int, default=100); p.add_argument("--val-microbatch-x", type=int, default=128)
    p.add_argument("--val-count", type=int, default=1000); p.add_argument("--decode-x-chunk", type=int, default=64)
    p.add_argument("--weight-sdf", type=float, default=5.0); p.add_argument("--weight-grad", type=float, default=0.1)
    p.add_argument("--weight-eikonal", type=float, default=0.01); p.add_argument("--weight-tension", type=float, default=0.01)
    p.add_argument("--weight-union-objective", type=float, default=1.0); p.add_argument("--weight-sensor-objective", type=float, default=1.0)
    p.add_argument("--weight-consistency", type=float, default=0.1)
    p.add_argument("--horizontal-fov-deg", type=float, default=50.0); p.add_argument("--vertical-fov-deg", type=float, default=66.0)
    p.add_argument("--z-min", type=float, default=.2); p.add_argument("--z-max", type=float, default=.7); p.add_argument("--delta", type=float, default=.01)
    p.add_argument("--log-every", type=int, default=100); p.add_argument("--val-every", type=int, default=1000); p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--resume-training", default="")
    args = p.parse_args()
    for n in ("steps","global_batch_x","batch_q","microbatch_x","val_global_batch_x","val_batch_q","val_microbatch_x","val_count","decode_x_chunk","log_every","val_every","save_every"):
        if getattr(args,n) <= 0: p.error(f"{n} must be positive")
    if args.global_batch_x % 4 or args.val_global_batch_x % 4: p.error("global x batches must be divisible by 4")
    for n in ("repo","data","urdf","out_dir"):
        setattr(args,n,str(Path(getattr(args,n)).expanduser().resolve()))
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc_per_node=4")
    local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank); device = torch.device("cuda",local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    try:
        if dist.get_world_size() != 4: raise RuntimeError("R012 formal study requires exactly 4 GPUs")
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16 unsupported")
        out = Path(args.out_dir)
        fresh_ok = bool(args.resume_training) or not out.exists() or not any(out.iterdir())
        base.synchronized_check(fresh_ok, f"Output directory nonempty: {out}", device)
        if dist.get_rank()==0: out.mkdir(parents=True,exist_ok=True)
        dist.barrier()

        # Dataset split seed is common. Model initialization is also common seed=0.
        base.seed_everything(args.seed)
        api = base.load_repo_api(Path(args.repo))
        dataset = api["VisibilityQ0Dataset"](path=args.data,val_count=args.val_count,seed=args.seed)
        oracle = api["PinocchioFOVOracle"](urdf_path=args.urdf,joint_names=api["DEFAULT_JOINT_NAMES"],sensor_frames=api["DEFAULT_SENSOR_FRAMES"],
            horizontal_fov_deg=args.horizontal_fov_deg,vertical_fov_deg=args.vertical_fov_deg,z_min=args.z_min,z_max=args.z_max,delta=args.delta)
        model = models.build_model(args.arm).to(device)
        if not all(p.requires_grad for p in model.parameters()): raise RuntimeError("Unexpected frozen parameter")
        ddp = DDP(model,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False)
        weights = obj.LossWeights(**{k:getattr(args,"weight_"+k) for k in asdict(obj.LossWeights())}); weights.validate()
        optimizer = torch.optim.Adam(ddp.parameters(),lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=5000,threshold=.01,threshold_mode="rel",cooldown=0,min_lr=0,eps=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp=="fp16",init_scale=65536.)
        metadata = {"repo_commit": subprocess.check_output(["git","-C",args.repo,"rev-parse","HEAD"],text=True).strip(),
            "torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"data":args.data,"data_bytes":Path(args.data).stat().st_size,
            "urdf_sha256":sha256_file(Path(args.urdf)),"model_seed":args.seed,"stream_seed":args.stream_seed}

        start=0; best=float("inf"); chain="0"*64; chain_count=0
        if args.resume_training:
            start,best,chain,chain_count=load_resume(Path(args.resume_training),ddp,optimizer,scheduler,scaler,args)
        else:
            # Critical fairness control: reset sampling RNG after architecture-specific
            # initialization so R0/R1/R2 see exactly the same subsequent global stream.
            base.seed_everything(args.stream_seed)
        stream=StreamChain(api,chain,chain_count); stream.install()
        if dist.get_rank()==0:
            (out/"train_args.json").write_text(json.dumps({**vars(args),"weights":asdict(weights),"metadata":metadata,"architecture":model.architecture()},indent=2)+"\n")
            run={"status":"RUNNING","format":FORMAT,"arm":args.arm,"steps":args.steps,"successful_updates":start,"model_seed":args.seed,"stream_seed":args.stream_seed,
                 "architecture":model.architecture(),"routing":"all_head_joint" if args.arm in ("R0","R1") else "union_plus_one_sensor_round_robin_to_shared"}
            (out/"run.json").write_text(json.dumps(run,indent=2)+"\n")
        log(f"[r012] arm={args.arm} scratch=yes params={model.parameter_count():,} world=4 stream_seed={args.stream_seed}")
        log("architecture:",model.architecture()); log("loss:",asdict(weights))

        train_stats={}; last_val={}
        for step in range(start+1,args.steps+1):
            torch.cuda.reset_peak_memory_stats(device); t0=time.perf_counter()
            batches,counts=base.prepare_batch(api,dataset,oracle,args,device,"train")
            if args.arm=="R2": train_stats=run_r2_batch(ddp,batches,counts,weights,args,optimizer,scaler,step)
            else: train_stats=base.run_prepared_batch(ddp,batches,counts,weights,args,optimizer,scaler)
            del batches; scheduler.step(train_stats["loss"]); torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-t0; peak=torch.tensor(torch.cuda.max_memory_allocated(device)/1024**3,device=device);dist.all_reduce(peak,op=dist.ReduceOp.MAX)
            train_stats.update(elapsed_sec=elapsed,max_peak_allocated_gib=float(peak),training_stream_sha256=stream.chain)
            if step==1 or step%args.log_every==0 or step==args.steps:
                h=train_stats["heads"]; sel=train_stats.get("selected_shared_sensor")
                log(f"[train] {args.arm} {step}/{args.steps} loss={train_stats['loss']:.6f} Ucos={h['union']['grad_cosine']:.4f} S0cos={h['s0']['grad_cosine']:.4f} S6cos={h['s6']['grad_cosine']:.4f} sel={sel} sec={elapsed:.2f} stream={stream.chain[:12]}")
                if dist.get_rank()==0:
                    with (out/"metrics.jsonl").open("a") as f: f.write(json.dumps({"split":"train","step":step,**train_stats},allow_nan=False)+"\n")
            if step==1 or step%args.val_every==0 or step==args.steps:
                vt=time.perf_counter(); vb,vc=base.prepare_batch(api,dataset,oracle,args,device,"val")
                last_val=base.run_prepared_batch(ddp,vb,vc,weights,args); del vb
                val_elapsed=time.perf_counter()-vt; best=min(best,last_val["loss"])
                if dist.get_rank()==0:
                    with (out/"metrics.jsonl").open("a") as f: f.write(json.dumps({"split":"val","step":step,"elapsed_sec":val_elapsed,**last_val},allow_nan=False)+"\n")
                log(f"[val] {args.arm} step={step} loss={last_val['loss']:.6f} Ucos={last_val['heads']['union']['grad_cosine']:.4f} S0cos={last_val['heads']['s0']['grad_cosine']:.4f} S6cos={last_val['heads']['s6']['grad_cosine']:.4f}")
            if step%args.save_every==0 and step<args.steps:
                save_checkpoint(out/"latest.pt",ddp,optimizer,scheduler,scaler,args,step,best,{"train":train_stats,"val":last_val},stream,metadata)
                if dist.get_rank()==0:
                    run=json.loads((out/"run.json").read_text());run.update(successful_updates=step,training_stream_sha256=stream.chain);(out/"run.json").write_text(json.dumps(run,indent=2)+"\n")

        save_checkpoint(out/"final.pt",ddp,optimizer,scheduler,scaler,args,args.steps,best,{"train":train_stats,"val":last_val},stream,metadata)
        dist.barrier()
        if dist.get_rank()==0:
            final_sha=sha256_file(out/"final.pt"); run=json.loads((out/"run.json").read_text());run.update(status="COMPLETE",successful_updates=args.steps,
                training_stream_sha256=stream.chain,training_stream_updates=stream.count,final_sha256=final_sha,best_val=best,
                source_sha256=source_fingerprints());(out/"run.json").write_text(json.dumps(run,indent=2)+"\n")
            print(f"[done] r012_arm_complete arm={args.arm} updates={args.steps} stream={stream.chain} sha={final_sha}",flush=True)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
