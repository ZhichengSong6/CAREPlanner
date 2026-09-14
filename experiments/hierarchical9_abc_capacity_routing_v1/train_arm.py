#!/usr/bin/env python3
"""Train one routed A/B/C arm on one GPU from the exact same P0 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import abc_protocol as proto
from abc_model import build_arm, function_equivalence
import sensor_objective as routed

old = proto.old


def sample_global_indices_single(dataset, split, global_batch_x, device):
    """Exact upstream rank-0 index sampling without torch.distributed."""
    if split == "train":
        pool = dataset.train_indices_cpu
    elif split == "val":
        pool = dataset.val_indices_cpu
    else:
        raise ValueError(split)
    ridx = torch.randint(0, len(pool), (global_batch_x,), dtype=torch.long)
    return pool[ridx].to(device=device, non_blocking=True)


def sample_shared_q_single(dataset, batch_q, device):
    """Exact upstream rank-0 uniform-q draw without distributed broadcast."""
    q_min, q_max = dataset.q_limits(device=device)
    u = torch.rand((batch_q, dataset.J), device=device)
    return q_min[None, :] + u * (q_max - q_min)[None, :]


def prepare_batch_single(api, dataset, oracle, args, device, split, baseline, obj):
    bx = args.global_batch_x if split == "train" else args.val_global_batch_x
    bq = args.batch_q if split == "train" else args.val_batch_q
    micro = args.microbatch_x if split == "train" else args.val_microbatch_x
    saved = baseline.rng_state() if split == "val" else None
    try:
        if split == "val":
            baseline.seed_everything(args.seed + 100003)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            indices = sample_global_indices_single(dataset, split, bx, device)
            q = sample_shared_q_single(dataset, bq, device)
            # materialize_local_x does not require a process group. rank=0/world=1
            # materializes the full Cartesian batch for this independent GPU arm.
            x, qlib, valid = api["materialize_local_x"](dataset, indices, 0, 1, device)
            sensor_masks = dataset.sensor_masks(device=device)
            batches = []
            counts8 = torch.zeros(8, dtype=torch.float32, device=device)
            counts9 = torch.zeros(9, dtype=torch.float32, device=device)
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
                    target_grad.reshape(-1, 8, 7).float().contiguous(),
                    mask,
                ))
                counts8 += routed.counts_from_mask(mask)
                counts9 += obj.counts_from_mask(mask)
                del ds, dgrad, sign, target, target_grad
            if counts8.sum() <= 0 or counts9[0] <= 0:
                raise RuntimeError("No supervised rows")
            identity = hashlib.sha256()
            identity.update(np.ascontiguousarray(indices.detach().cpu().numpy()).tobytes())
            identity.update(np.ascontiguousarray(q.detach().cpu().float().numpy()).tobytes())
            return batches, counts8, counts9, identity.digest()
    finally:
        if saved is not None:
            baseline.restore_rng(saved)


def amp_context(device, amp):
    return torch.autocast(device_type=device.type, enabled=amp != "off",
                          dtype=torch.float16 if amp == "fp16" else torch.bfloat16)


def run_sensor_batches(model, batches, counts8, weights, args, scaler=None, optimizer=None):
    training = optimizer is not None
    device = counts8.device
    for attempt in range(args.max_amp_retries + 1):
        if training:
            optimizer.zero_grad(set_to_none=True)
        stats = torch.zeros((8, len(routed.STAT_NAMES)), dtype=torch.float64, device=device)
        model.train(training)
        for batch in batches:
            precision = args.amp if training else "off"
            with torch.enable_grad(), amp_context(device, precision):
                loss, st = routed.loss_for_microbatch(
                    model, *batch, counts8, weights, training=training
                )
            if not (torch.isfinite(loss.detach()) and torch.isfinite(st).all()):
                raise RuntimeError("Nonfinite routed sensor loss")
            if training:
                scaler.scale(loss).backward()
            stats += st
        if not training:
            break
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients:
            raise RuntimeError("No trainable gradients")
        finite = all(bool(torch.isfinite(g).all().item()) for g in gradients)
        if finite:
            scaler.step(optimizer)
            scaler.update()
            break
        if args.amp != "fp16" or attempt == args.max_amp_retries:
            raise RuntimeError("Nonfinite routed parameter gradients")
        scaler.update(new_scale=scaler.get_scale()/2.0)
    result = routed.summarize(stats, weights)
    result["amp_retries"] = attempt if training else 0
    result["grad_scale"] = scaler.get_scale() if training else None
    return result


def run_full_validation(model, batches, counts9, weights, obj):
    device = counts9.device
    total = torch.zeros((9, len(obj.STAT_NAMES)), dtype=torch.float64, device=device)
    model.eval()
    for batch in batches:
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            loss, st = obj.loss_for_microbatch(
                model, *batch, counts9, weights, world_size=1, training=False
            )
        if not (torch.isfinite(loss.detach()) and torch.isfinite(st).all()):
            raise RuntimeError("Nonfinite unchanged global validation objective")
        total += st
    return obj.summarize(total, weights)


def save_checkpoint(path, model, optimizer, scaler, args, step, validation,
                    p0_sha, cache_identity, stream_sha, initial_equivalence,
                    baseline, *, final=False):
    state = dict(
        format=proto.FORMAT,
        arm=args.arm,
        mode=args.mode,
        completed=final,
        parent="P0",
        parent_sha256=p0_sha,
        parent_updates=52000,
        extra_updates=step,
        total_updates=52000+step,
        cache_manifest_sha256=cache_identity,
        optimizer_policy="fresh_Adam_sensor_specific_only_constant_lr_no_scheduler_no_clipping",
        training_objective="original_per_sensor_global_objective_only_no_union_no_consistency",
        union_path_frozen=True,
        rng_tag=proto.RNG_TAG,
        args=vars(args),
        architecture=model.architecture(),
        initial_equivalence=initial_equivalence,
        training_stream_sha256=stream_sha,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict(),
        validation=validation,
        abc_source_sha256=proto.fingerprints(),
        source_sha256=old.source_fingerprints(),
    )
    if final:
        proto.assert_checkpoint(state, _GLOBAL_P0, p0_sha, cache_identity, require_pilot=args.mode=="pilot")
    baseline.atomic_save(state, Path(path))


_GLOBAL_P0 = None


def main():
    global _GLOBAL_P0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--arm", choices=proto.ARMS, required=True)
    ap.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    args0 = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("One allocated CUDA GPU is required per arm")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    root = args0.reference_root.resolve()
    cache, p0, p0_sha = proto.load_reference_root(root)
    _GLOBAL_P0 = p0
    out = proto.output_dir(root, args0.arm, args0.mode)
    args = proto.make_args(p0, out, args0.arm, args0.mode)
    if out.exists():
        raise FileExistsError(f"No overwrite/resume: {out}")
    out.mkdir(parents=True)

    baseline = old.module("train", old.SCRATCH)
    obj = old.module("objective", old.SCRATCH)
    weights = obj.LossWeights()
    artifact = Path(args.artifact_root)
    api = baseline.load_repo_api(old.REPO)
    dataset = api["VisibilityQ0Dataset"](str(artifact/old.DATA_REL), val_count=1000, seed=0)
    cache.verify_dataset(dataset, artifact/old.DATA_REL)
    oracle = api["PinocchioFOVOracle"](
        urdf_path=str(old.URDF), joint_names=api["DEFAULT_JOINT_NAMES"],
        sensor_frames=api["DEFAULT_SENSOR_FRAMES"], horizontal_fov_deg=50.,
        vertical_fov_deg=66., z_min=.2, z_max=.7, delta=.01,
    )

    baseline.seed_everything(args.seed)
    base = proto.p0_model(p0, device).eval().requires_grad_(False)
    model = build_arm(base, args.arm).to(device=device, dtype=torch.float32)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16", init_scale=1024.)

    val_batches, val_counts8, val_counts9, _ = prepare_batch_single(
        api, dataset, oracle, args, device, "val", baseline, obj
    )
    probe = val_batches[0][0][:32]
    equivalence = function_equivalence(base, model, probe)
    if equivalence["max_abs_value_error"] > 2e-6 or equivalence["max_abs_q_gradient_error"] > 2e-6:
        raise RuntimeError(f"{args.arm} is not function-equivalent to P0 at initialization: {equivalence}")

    initial = dict(
        routed_sensor=run_sensor_batches(model, val_batches, val_counts8, weights, args),
        unchanged_global=run_full_validation(model, val_batches, val_counts9, weights, obj),
    )
    proto.write_json(out/"initial_validation.json", initial)
    proto.write_json(out/"run.json", dict(
        status="RUNNING", arm=args.arm, mode=args.mode, args=vars(args),
        parent="P0", parent_sha256=p0_sha, cache_manifest_sha256=cache.identity,
        architecture=model.architecture(), initial_equivalence=equivalence,
        abc_source_sha256=proto.fingerprints(),
    ))
    print(f"[abc] arm={args.arm} mode={args.mode} P0={p0_sha} total={model.total_parameter_count()} "
          f"trainable={model.trainable_parameter_count()} equivalence={equivalence}", flush=True)

    stream = hashlib.sha256()
    validation = initial
    for step in range(1, args.steps+1):
        started = time.perf_counter()
        # Arm-specific construction may consume RNG (B's adapter down-projection).
        # Reseeding HERE guarantees identical train sample streams for A/B/C.
        baseline.seed_everything(proto.abc_seed_for_update(args.seed, step))
        batches, counts8, counts9, identity = prepare_batch_single(
            api, dataset, oracle, args, device, "train", baseline, obj
        )
        stream.update(identity)
        torch.cuda.reset_peak_memory_stats(device)
        train_stats = run_sensor_batches(model, batches, counts8, weights, args, scaler, optimizer)
        torch.cuda.synchronize(device)
        record = dict(
            update=step, arm=args.arm, sensor=train_stats,
            seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        )
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            h = train_stats["heads"]
            print(f"[train] {args.arm} {step}/{args.steps} loss={train_stats['loss']:.6f} "
                  f"S0cos={h['s0']['grad_cosine']:.4f} S6cos={h['s6']['grad_cosine']:.4f} "
                  f"S7cos={h['s7']['grad_cosine']:.4f} sec={record['seconds']:.2f}", flush=True)
            with (out/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False)+"\n")
        if step == 1 or step % args.val_every == 0 or step == args.steps:
            validation = dict(
                routed_sensor=run_sensor_batches(model, val_batches, val_counts8, weights, args),
                unchanged_global=run_full_validation(model, val_batches, val_counts9, weights, obj),
            )
            with (out/"validation.jsonl").open("a") as f:
                f.write(json.dumps({"update":step, **validation}, allow_nan=False)+"\n")
            vh = validation["routed_sensor"]["heads"]
            print(f"[val] {args.arm} update={step} loss={validation['routed_sensor']['loss']:.6f} "
                  f"S0cos={vh['s0']['grad_cosine']:.4f} S6cos={vh['s6']['grad_cosine']:.4f} "
                  f"S7cos={vh['s7']['grad_cosine']:.4f}", flush=True)
            save_checkpoint(out/"latest.pt", model, optimizer, scaler, args, step, validation,
                            p0_sha, cache.identity, stream.hexdigest(), equivalence, baseline)
        del batches

    save_checkpoint(out/"final.pt", model, optimizer, scaler, args, args.steps, validation,
                    p0_sha, cache.identity, stream.hexdigest(), equivalence, baseline, final=True)
    run = json.loads((out/"run.json").read_text())
    run.update(status="COMPLETE", successful_updates=args.steps,
               training_stream_sha256=stream.hexdigest(), final_sha256=proto.sha256(out/"final.pt"))
    proto.write_json(out/"run.json", run)
    print(f"[done] abc_arm_complete arm={args.arm} mode={args.mode} updates={args.steps} output={out}", flush=True)


if __name__ == "__main__":
    main()
