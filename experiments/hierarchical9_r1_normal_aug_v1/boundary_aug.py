"""Add ONLY a small analytic boundary-normal direction objective to ORIGINAL R1 updates.

The original R1 objective, optimizer, LR scheduler, random train stream and
AMP retry semantics remain owned by R012/base, not reimplemented as new labels.
Boundary gradient is added as a manually all-reduced parameter gradient BEFORE
the original GradScaler optimizer.step(); no clipping, and no second optimizer.
"""
from __future__ import annotations

import contextlib
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

SENSORS = 8


def load_original_cache_reader(repo: Path):
    path = repo / "experiments/hierarchical9_paired_slope_scratch_v1/data.py"
    spec = importlib.util.spec_from_file_location("r1_aug_frozen_cache_reader", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class BoundaryDirection:
    """Stateless sample RNG; NO torch RNG calls, NO writes to cached labels."""
    def __init__(self, repo: Path, cache_path: Path, dataset, cfg: dict):
        mod = load_original_cache_reader(repo)
        self.cache = mod.Cache(cache_path, verify=False)
        self.cfg = cfg
        self.step = 0
        base_x = dataset.x_cpu.numpy()
        if self.cache.x.shape != base_x.shape:
            raise ValueError("R1 x and V4 cached x shapes differ; cannot align indices")
        delta = float(np.max(np.abs(np.asarray(self.cache.x, np.float64) -
                                    np.asarray(base_x, np.float64))))
        if delta > 1e-6:
            raise ValueError(f"R1 x and V4 x-index mapping mismatch: {delta}")
        # CRITICAL: V4 has its OWN train/val split. Never allow R1's validation
        # x into the augmented TRAIN samples, even if V4 called them train.
        train_x = np.asarray(dataset.train_indices_np, np.int64)
        val_x = np.asarray(dataset.val_indices_np, np.int64)
        if np.intersect1d(train_x, val_x).size:
            raise ValueError("R1 train/val x groups overlap")
        self.groups = {}
        for split, allow in (("train", train_x), ("val", val_x)):
            # Validation diagnostics use the V4 VAL source, without training on it.
            source = self.cache.a[split]["boundary"]
            x = np.asarray(source["x_index"])
            sensor = np.asarray(source["sensor"])
            admissible = np.isin(x, allow)
            self.groups[split] = []
            for sid in range(SENSORS):
                ids = np.flatnonzero(admissible & (sensor == sid)).astype(np.int64)
                if len(ids) == 0:
                    raise ValueError(f"No aligned R1-{split} V4 boundary anchors for sensor {sid}")
                self.groups[split].append(ids)
        self.cache_identity = self.cache.identity
        self.counts = {split: [len(v) for v in groups] for split, groups in self.groups.items()}
        self.orig_prepare_calls = 0

    def batch(self, split: str, step: int, rank: int, device: torch.device):
        k = int(self.cfg["anchors_per_sensor_per_rank"])
        chosen = []
        for sid, pool in enumerate(self.groups[split]):
            rng = np.random.default_rng(np.random.SeedSequence(
                [int(self.cfg["aux_stream_seed"]), int(step), int(rank), sid, split == "val"]))
            chosen.append(pool[rng.integers(len(pool), size=k)])
        ids = np.concatenate(chosen)
        z = self.cache.a[split]["boundary"]
        x_index = np.asarray(z["x_index"][ids], np.int64)
        x = np.asarray(self.cache.x[x_index], np.float32).copy()
        q = np.asarray(z["q"][ids], np.float32).copy()
        norm = np.asarray(z["grad"][ids], np.float32).copy()
        sensor = np.asarray(z["sensor"][ids], np.int64)
        expected = np.repeat(np.arange(SENSORS), k)
        if not np.array_equal(sensor, expected):
            raise RuntimeError("Boundary sensor strata mismatch")
        if (not np.isfinite(q).all() or not np.isfinite(norm).all() or
                np.max(np.abs(np.linalg.norm(norm, axis=1)-1)) > 5e-3):
            raise ValueError("Nonunit/invalid analytic boundary normals")
        return (torch.as_tensor(x, device=device),
                torch.as_tensor(q, device=device),
                torch.as_tensor(norm, device=device),
                torch.as_tensor(sensor, device=device, dtype=torch.long))

    @staticmethod
    def direction_loss(model, batch, cfg, *, training: bool, monitor: bool):
        x, q0, normal, sid = batch
        q = q0.detach().clone().requires_grad_(True)
        activations = {}
        hooks = []
        if monitor:
            # First sensor-head ReLU only; observe, do not alter activations.
            for s in range(SENSORS):
                def record(_module, _inputs, out, sensor=s):
                    activations[sensor] = out.detach()
                hooks.append(model.sensor_heads[s][1].register_forward_hook(record))
        try:
            with torch.enable_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                prediction = model(torch.cat([x.float(), q.float()], dim=1)).float()
                y = prediction.gather(1, (sid+1).unsqueeze(1)).squeeze(1)
                grad = torch.autograd.grad(y.sum(), q, create_graph=training,
                                           retain_graph=training)[0].float()
                cos = F.cosine_similarity(grad, normal, dim=1, eps=1e-6)
                if not torch.isfinite(grad).all() or not torch.isfinite(cos).all():
                    raise FloatingPointError("Nonfinite analytic boundary direction")
                per = (1. - cos).reshape(SENSORS, -1).mean(dim=1)
                loss = float(cfg["normal_direction_weight"]) * per.mean()
                metrics = torch.zeros((SENSORS, 5), device=q.device, dtype=torch.float64)
                for s in range(SENSORS):
                    keep = sid == s
                    metrics[s, 0] = cos[keep].detach().mean().double()
                    metrics[s, 1] = grad[keep].detach().norm(dim=1).mean().double()
                    metrics[s, 2] = y[keep].detach().abs().mean().double()
                    if monitor:
                        layer = activations[s][keep]
                        metrics[s, 3] = (layer > 0).double().mean()
                        metrics[s, 4] = (layer > 0).any(dim=0).double().sum()
                return loss, metrics
        finally:
            for h in hooks:
                h.remove()

    def metrics(self, tensor: torch.Tensor, training: bool):
        # Every rank draws EXACTLY the same number of anchors per sensor.
        result = tensor.detach().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
        return {
            "kind": "train" if training else "val",
            "normal_cosine": [float(v) for v in result[:, 0]],
            "normal_grad_norm": [float(v) for v in result[:, 1]],
            "boundary_abs_field": [float(v) for v in result[:, 2]],
            "first_head_relu_active": [float(v) for v in result[:, 3]],
            "first_head_relu_ever_active_count": [float(v) for v in result[:, 4]],
        }


def attach_to_original_r1(trainer, augmentation: BoundaryDirection | None, cfg: dict):
    """Install a narrow change at R1's parameter-update boundary, NOT in its sampler."""
    original = trainer.base.run_prepared_batch
    base = trainer.base

    def run_batch(ddp, batches, counts, weights, args, optimizer=None, scaler=None):
        if optimizer is None:
            result = original(ddp, batches, counts, weights, args)
            # Original validation loss and scheduler metric are unchanged.
            if augmentation is not None:
                val_batch = augmentation.batch("val", 0, dist.get_rank(), counts.device)
                _, mm = augmentation.direction_loss(ddp.module, val_batch, cfg,
                                                     training=False, monitor=True)
                result["boundary_direction_diagnostic"] = augmentation.metrics(mm, False)
            return result
        if augmentation is None:
            return original(ddp, batches, counts, weights, args, optimizer, scaler)
        augmentation.step += 1
        step = augmentation.step
        device = counts.device
        world = dist.get_world_size()
        aux_batch = augmentation.batch("train", step, dist.get_rank(), device)
        monitor = step == 1 or step % int(args.log_every) == 0 or step == args.steps
        aux_report = {}
        # This loop mirrors R1 base.run_prepared_batch, and reuses the EXACT
        # upstream objective. Only additional lines are marked as AUGMENT.
        for attempt in range(args.max_amp_retries + 1):
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros((9, len(base.STAT_NAMES)), dtype=torch.float64, device=device)
            ddp.train(True)
            for i, batch in enumerate(batches):
                context = ddp.no_sync() if i < len(batches) - 1 else contextlib.nullcontext()
                with context, torch.enable_grad():
                    with base.amp_context(args.amp):
                        loss, stats = base.loss_for_microbatch(
                            ddp, *batch, counts, weights, world_size=world, training=True)
                    base.synchronized_check(
                        torch.isfinite(loss.detach()) & torch.isfinite(stats).all(),
                        "Non-finite ORIGINAL R1 objective", device)
                    scaler.scale(loss).backward()
                totals += stats
                del loss, stats
            # AUGMENT: use model.module instead of another DDP forward to avoid
            # the DDP reducer hooks with second-order input derivatives.
            extra, metrics = augmentation.direction_loss(
                ddp.module, aux_batch, cfg, training=True, monitor=monitor)
            if not math.isfinite(float(extra.detach())):
                raise FloatingPointError("Nonfinite auxiliary normal direction loss")
            params = list(ddp.module.parameters())
            scaled = extra * (float(scaler.get_scale()) if scaler.is_enabled() else 1.)
            g = torch.autograd.grad(scaled, params, allow_unused=True)
            flat = torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1)
                              for p, v in zip(params, g)])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= world
            cursor = 0
            for p in params:
                n = p.numel()
                piece = flat[cursor:cursor+n].reshape_as(p)
                if p.grad is None:
                    p.grad = piece.clone()
                else:
                    p.grad.add_(piece)
                cursor += n
            if monitor:
                aux_report = augmentation.metrics(metrics, True)
                aux_report.update(weight=float(cfg["normal_direction_weight"]),
                                  loss=float(extra.detach()),
                                  scaled_parameter_grad_norm=float(flat.norm().detach()) /
                                      (float(scaler.get_scale()) if scaler.is_enabled() else 1.),
                                  sample_count=8*int(cfg["anchors_per_sensor_per_rank"])*world,
                                  step=step)
            del extra, metrics, g, scaled, flat
            # The original R1 optimizer/scaler logic: NO clipping, same retry.
            scaler.unscale_(optimizer)
            gradients = [p.grad for p in ddp.parameters() if p.grad is not None]
            finite = torch.stack([torch.isfinite(v).all() for v in gradients]).all().int()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() == 1:
                scaler.step(optimizer)
                scaler.update()
                break
            if args.amp != "fp16" or attempt == args.max_amp_retries:
                raise RuntimeError("Nonfinite baseline+aux gradients after AMP retries")
            new_scale = scaler.get_scale() / 2.
            scaler.update(new_scale=new_scale)
            trainer.log(f"[amp-retry] same R1+boundary batch attempt={attempt+1} scale={new_scale:g}")
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if not torch.equal(totals[:, 0].float(), counts):
            raise RuntimeError("Original R1 supervision counts changed")
        stats = base.summarize(totals, weights)
        if not math.isfinite(stats["loss"]):
            raise RuntimeError("Nonfinite ORIGINAL R1 loss")
        stats["amp_retries"] = attempt
        stats["grad_scale"] = scaler.get_scale()
        if monitor:
            stats["boundary_direction_diagnostic"] = aux_report
        stats["scheduler_metric"] = "original_R1_loss_unmodified"
        return stats

    trainer.base.run_prepared_batch = run_batch
    return run_batch
