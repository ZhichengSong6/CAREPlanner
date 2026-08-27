#!/usr/bin/env python3
"""GPU micro-benchmark for the signed CAREPlanner collision CDF.

This intentionally does not import rospy.  It can therefore run in the CUDA
training environment even when that environment does not contain ROS Python
packages.

The measured operation is exactly the learned-field primitive needed online:

    (p_b, q_b), b=1..B  ->  d_b, grad_q d_b

Two GPU paths are timed:
  host_input:
      points/q begin as CPU tensors on every repetition.  CollisionCDF moves
      them to CUDA internally.  This includes host->device transfer and most
      closely matches the current ROS service call path.
  resident_input:
      points/q are already CUDA tensors.  This isolates network + autograd cost.

An optional CPU benchmark uses the same checkpoint/model/runtime definitions
for an apples-to-apples reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from collision_cdf_model import CollisionCDF  # noqa: E402


DEFAULT_BATCH_SIZES = [50, 100, 200, 500, 1000, 1650, 2000, 3000, 5000]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-key", default="latest")
    p.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu-reference", action="store_true")
    p.add_argument("--cpu-repeats", type=int, default=20)
    p.add_argument("--output", required=True)
    return p.parse_args()


def percentile(values, q):
    a = np.asarray(values, dtype=np.float64)
    return float(np.quantile(a, q))


def summarize_ms(values):
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "p05_ms": percentile(values, 0.05),
        "p95_ms": percentile(values, 0.95),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def make_inputs(batch_size, seed):
    rng = np.random.default_rng(seed)
    # CARE confidence-map workspace bounds.
    p = np.empty((batch_size, 3), dtype=np.float32)
    p[:, 0] = rng.uniform(-0.95, 0.95, size=batch_size)
    p[:, 1] = rng.uniform(-0.95, 0.95, size=batch_size)
    p[:, 2] = rng.uniform(0.0, 1.15, size=batch_size)

    # Current robot joint limits.
    q_min = np.asarray(
        [-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20],
        dtype=np.float32,
    )
    q_max = np.asarray(
        [3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20],
        dtype=np.float32,
    )
    q = rng.uniform(q_min, q_max, size=(batch_size, 7)).astype(np.float32)
    return torch.from_numpy(p), torch.from_numpy(q)


def call_pair(cdf, points, q):
    with torch.enable_grad():
        d, g = cdf.pair_distance_and_gradient(points, q)
    return d, g


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_path(cdf, points, q, warmup, repeats, force_host_input):
    device = cdf.device

    for _ in range(warmup):
        if force_host_input:
            p_call = points.cpu()
            q_call = q.cpu()
        else:
            p_call = points.to(device)
            q_call = q.to(device)
        call_pair(cdf, p_call, q_call)
    sync(device)

    timings = []
    output_copy_timings = []
    for _ in range(repeats):
        if force_host_input:
            p_call = points.cpu()
            q_call = q.cpu()
        else:
            p_call = points.to(device)
            q_call = q.to(device)

        sync(device)
        t0 = time.perf_counter()
        d, g = call_pair(cdf, p_call, q_call)
        sync(device)
        t1 = time.perf_counter()

        # Current ROS server converts returned distance/gradient to CPU after
        # inference.  Record that copy separately so we can estimate service
        # end-to-end cost rather than hiding device->host transfer.
        t2 = time.perf_counter()
        _ = d.cpu().numpy()
        _ = g.cpu().numpy()
        sync(device)
        t3 = time.perf_counter()

        timings.append((t1 - t0) * 1000.0)
        output_copy_timings.append((t3 - t2) * 1000.0)

    out = summarize_ms(timings)
    out["output_d2h"] = summarize_ms(output_copy_timings)
    out["estimated_inference_plus_d2h_median_ms"] = (
        out["median_ms"] + out["output_d2h"]["median_ms"]
    )
    return out


def build_cdf(checkpoint, device, activation, checkpoint_key):
    return CollisionCDF(
        checkpoint=checkpoint,
        device=device,
        checkpoint_key=checkpoint_key,
        input_dims=10,
        output_dims=1,
        nerf=True,
        activation=activation,
    )


def main():
    args = parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.cpu_repeats <= 0:
        raise ValueError("invalid warmup/repeat count")
    if any(b <= 0 for b in args.batch_sizes):
        raise ValueError("all batch sizes must be positive")

    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    output = os.path.abspath(os.path.expanduser(args.output))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requested but torch.cuda.is_available() is False. "
            f"torch={torch.__version__}, python={sys.executable}"
        )

    cuda = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(cuda)
    props = torch.cuda.get_device_properties(cuda)

    print("[GPU BENCH] python:", sys.executable)
    print("[GPU BENCH] torch:", torch.__version__)
    print("[GPU BENCH] cuda runtime:", torch.version.cuda)
    print("[GPU BENCH] gpu:", gpu_name)
    print("[GPU BENCH] total VRAM GiB:", props.total_memory / (1024**3))
    print("[GPU BENCH] checkpoint:", checkpoint)
    print("[GPU BENCH] activation:", args.activation)
    print("[GPU BENCH] batches:", args.batch_sizes)
    print("[GPU BENCH] warmup/repeats:", args.warmup, args.repeats)

    cdf_gpu = build_cdf(
        checkpoint,
        "cuda",
        args.activation,
        args.checkpoint_key,
    )

    result = {
        "checkpoint": checkpoint,
        "activation": args.activation,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "gpu_total_memory_gib": props.total_memory / (1024**3),
        "architecture": cdf_gpu.architecture,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": [],
    }

    cdf_cpu = None
    if args.cpu_reference:
        cdf_cpu = build_cdf(
            checkpoint,
            "cpu",
            args.activation,
            args.checkpoint_key,
        )

    for i, batch_size in enumerate(args.batch_sizes):
        points_cpu, q_cpu = make_inputs(batch_size, args.seed + i)

        # Resident inputs are allocated once before timing.
        points_gpu = points_cpu.to(cuda)
        q_gpu = q_cpu.to(cuda)
        sync(cuda)

        host_stats = time_path(
            cdf_gpu,
            points_cpu,
            q_cpu,
            args.warmup,
            args.repeats,
            force_host_input=True,
        )
        resident_stats = time_path(
            cdf_gpu,
            points_gpu,
            q_gpu,
            args.warmup,
            args.repeats,
            force_host_input=False,
        )

        row = {
            "batch_size": batch_size,
            "gpu_host_input": host_stats,
            "gpu_resident_input": resident_stats,
        }

        if cdf_cpu is not None:
            cpu_stats = time_path(
                cdf_cpu,
                points_cpu,
                q_cpu,
                min(args.warmup, 5),
                args.cpu_repeats,
                force_host_input=True,
            )
            row["cpu"] = cpu_stats
            row["speedup_cpu_over_gpu_host_median"] = (
                cpu_stats["median_ms"] / host_stats["median_ms"]
            )
            row["speedup_cpu_over_gpu_resident_median"] = (
                cpu_stats["median_ms"] / resident_stats["median_ms"]
            )

        result["results"].append(row)

        message = (
            f"[B={batch_size:5d}] "
            f"GPU host={host_stats['median_ms']:.3f} ms "
            f"(p95={host_stats['p95_ms']:.3f}) | "
            f"resident={resident_stats['median_ms']:.3f} ms "
            f"(p95={resident_stats['p95_ms']:.3f}) | "
            f"D2H={host_stats['output_d2h']['median_ms']:.3f} ms"
        )
        if "cpu" in row:
            message += (
                f" | CPU={row['cpu']['median_ms']:.3f} ms "
                f"| speedup={row['speedup_cpu_over_gpu_host_median']:.1f}x"
            )
        print(message)

    # Highlight the batch size closest to the observed C5.2 full-horizon median.
    target = min(
        result["results"],
        key=lambda row: abs(row["batch_size"] - 1650),
    )
    result["c5_2_reference_batch_size"] = target["batch_size"]
    result["c5_2_reference_gpu_host_median_ms"] = target[
        "gpu_host_input"
    ]["median_ms"]
    result["c5_2_reference_gpu_resident_median_ms"] = target[
        "gpu_resident_input"
    ]["median_ms"]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("")
    print("[GPU BENCH RESULT]", output)
    print(
        "[GPU BENCH C5.2 ~1650 pairs] host={:.3f} ms resident={:.3f} ms".format(
            result["c5_2_reference_gpu_host_median_ms"],
            result["c5_2_reference_gpu_resident_median_ms"],
        )
    )


if __name__ == "__main__":
    main()
