#!/usr/bin/env python3

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from urdf_parser_py.urdf import URDF

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    get_joint_limits,
)

from extract_visibility_zero_level_sets import (  # noqa: E402
    prepare_chain_specs,
    visibility_g_batch,
    refine_q_to_zero_level,
    greedy_farthest_downsample,
    sample_q_batch,
    torch_tensor,
    prepare_occlusion_context,
    compute_q0_occlusion_torch,
    compute_q0_occlusion,
)


SENSOR_CHAIN_MASKS = np.asarray(
    [
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],

        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],

        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],

        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.float32,
)


def build_workspace_grid(args) -> np.ndarray:
    xs = np.linspace(args.x_min, args.x_max, args.grid[0], dtype=np.float32)
    ys = np.linspace(args.y_min, args.y_max, args.grid[1], dtype=np.float32)
    zs = np.linspace(args.z_min_bound, args.z_max_bound, args.grid[2], dtype=np.float32)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points.astype(np.float32)


def select_point_indices(num_points: int, args) -> np.ndarray:
    indices = np.arange(num_points, dtype=np.int64)

    rng = np.random.default_rng(args.p_seed)
    if args.shuffle_p:
        rng.shuffle(indices)

    if args.p_count > 0:
        indices = indices[: min(args.p_count, len(indices))]

    if args.p_num_shards > 1:
        if args.p_shard_id < 0 or args.p_shard_id >= args.p_num_shards:
            raise ValueError("--p-shard-id must be in [0, p_num_shards)")
        indices = indices[args.p_shard_id :: args.p_num_shards]

    return indices.astype(np.int64)


def allocate_yiming_style_arrays(num_p: int, k: int, num_joints: int, num_sensors: int):
    data = {}

    # Yiming-style layout.
    # q[p, k, :, s] = 7D q0 for point p, q0 index k, sensor group s.
    # Invalid q0 slots are filled by +inf, matching Yiming's style.
    data["x"] = np.zeros((num_p, 3), dtype=np.float32)
    data["q"] = np.full((num_p, k, num_joints, num_sensors), np.inf, dtype=np.float32)
    data["k"] = np.zeros((num_p,), dtype=np.int64)

    data["valid_fov"] = np.zeros((num_p, k, num_sensors), dtype=np.bool_)
    data["occluded"] = np.zeros((num_p, k, num_sensors), dtype=np.bool_)
    data["valid_with_occlusion"] = np.zeros((num_p, k, num_sensors), dtype=np.bool_)

    data["g"] = np.full((num_p, k, num_sensors), np.inf, dtype=np.float32)
    data["sensor_margins"] = np.full((num_p, k, num_sensors, num_sensors), np.inf, dtype=np.float32)
    data["active_planes"] = np.full((num_p, k, num_sensors, num_sensors), -1, dtype=np.int16)

    # Project-native layout for easier debugging.
    # sensor_q0_templates[p, s, k, :] is the transpose-equivalent of q[p, k, :, s].
    data["grid_points"] = data["x"]
    data["p_indices"] = data["k"]

    data["sensor_q0_templates"] = np.full((num_p, num_sensors, k, num_joints), np.inf, dtype=np.float32)
    data["sensor_q0_g"] = np.full((num_p, num_sensors, k), np.inf, dtype=np.float32)
    data["sensor_q0_sensor_margins"] = np.full((num_p, num_sensors, k, num_sensors), np.inf, dtype=np.float32)
    data["sensor_q0_active_planes"] = np.full((num_p, num_sensors, k, num_sensors), -1, dtype=np.int16)

    data["sensor_q0_occluded"] = np.zeros((num_p, num_sensors, k), dtype=np.bool_)
    data["sensor_q0_valid_fov"] = np.zeros((num_p, num_sensors, k), dtype=np.bool_)
    data["sensor_q0_valid_with_occlusion"] = np.zeros((num_p, num_sensors, k), dtype=np.bool_)

    data["num_sensor_q0"] = np.zeros((num_p, num_sensors), dtype=np.int32)
    data["num_sensor_q0_occluded"] = np.zeros((num_p, num_sensors), dtype=np.int32)
    data["num_sensor_q0_visible"] = np.zeros((num_p, num_sensors), dtype=np.int32)
    data["num_sensor_q0_before_downsample"] = np.zeros((num_p, num_sensors), dtype=np.int32)

    return data


def store_sensor_block(
    data: Dict[str, np.ndarray],
    out_idx: int,
    p_idx: int,
    point: np.ndarray,
    sensor_idx: int,
    q0: np.ndarray,
    g0: np.ndarray,
    sensor_margins0: np.ndarray,
    active_planes0: np.ndarray,
    occluded0: np.ndarray,
    args,
):
    if len(q0) == 0:
        return

    data["num_sensor_q0_before_downsample"][out_idx, sensor_idx] = len(q0)

    keep = greedy_farthest_downsample(
        q0,
        args.q0_per_sensor,
        args.seed + 1000003 * (out_idx + 1) + 1009 * sensor_idx,
    )

    q0 = q0[keep]
    g0 = g0[keep]
    sensor_margins0 = sensor_margins0[keep]
    active_planes0 = active_planes0[keep]
    occluded0 = occluded0[keep]

    count = min(len(q0), args.q0_per_sensor)
    if count <= 0:
        return

    q0 = q0[:count]
    g0 = g0[:count]
    sensor_margins0 = sensor_margins0[:count]
    active_planes0 = active_planes0[:count]
    occluded0 = occluded0[:count]

    valid_fov0 = np.ones((count,), dtype=np.bool_)
    valid_with_occ0 = valid_fov0 & (~occluded0)

    # Yiming-style layout.
    data["q"][out_idx, :count, :, sensor_idx] = q0
    data["valid_fov"][out_idx, :count, sensor_idx] = valid_fov0
    data["occluded"][out_idx, :count, sensor_idx] = occluded0
    data["valid_with_occlusion"][out_idx, :count, sensor_idx] = valid_with_occ0

    data["g"][out_idx, :count, sensor_idx] = g0
    data["sensor_margins"][out_idx, :count, sensor_idx, :] = sensor_margins0
    data["active_planes"][out_idx, :count, sensor_idx, :] = active_planes0

    # Project-native layout.
    data["sensor_q0_templates"][out_idx, sensor_idx, :count, :] = q0
    data["sensor_q0_g"][out_idx, sensor_idx, :count] = g0
    data["sensor_q0_sensor_margins"][out_idx, sensor_idx, :count, :] = sensor_margins0
    data["sensor_q0_active_planes"][out_idx, sensor_idx, :count, :] = active_planes0

    data["sensor_q0_occluded"][out_idx, sensor_idx, :count] = occluded0
    data["sensor_q0_valid_fov"][out_idx, sensor_idx, :count] = valid_fov0
    data["sensor_q0_valid_with_occlusion"][out_idx, sensor_idx, :count] = valid_with_occ0

    data["num_sensor_q0"][out_idx, sensor_idx] = count
    data["num_sensor_q0_occluded"][out_idx, sensor_idx] = int(np.count_nonzero(occluded0))
    data["num_sensor_q0_visible"][out_idx, sensor_idx] = int(np.count_nonzero(valid_with_occ0))


def generate_for_one_point(
    point: np.ndarray,
    q_init: np.ndarray,
    all_chain_specs,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    occlusion_context,
    args,
    device,
):
    num_sensors = len(args.sensor_frames)
    results = []

    with torch.no_grad():
        _, sensor_margins_init_t, _, _ = visibility_g_batch(
            torch_tensor(point, device),
            torch_tensor(q_init, device),
            all_chain_specs,
            args.horizontal_fov_deg,
            args.vertical_fov_deg,
            args.z_min,
            args.z_max,
            args.delta,
        )
        sensor_margins_init = sensor_margins_init_t.detach().cpu().numpy()

    for sensor_idx in range(num_sensors):
        # Per-sensor FOV zero-level objective:
        # g_s(x, q) = margin_s(x, q) - delta.
        g_sensor_init = sensor_margins_init[:, sensor_idx] - args.delta

        top_k = min(args.optimize_top_k_per_sensor, len(q_init))
        if top_k <= 0:
            results.append(None)
            continue

        candidate_unsorted = np.argpartition(np.abs(g_sensor_init), top_k - 1)[:top_k]
        candidate_indices = candidate_unsorted[np.argsort(np.abs(g_sensor_init[candidate_unsorted]))]
        q_candidates = q_init[candidate_indices]

        (
            q0,
            g0,
            sensor_margins0,
            _active_sensor0,
            active_planes0,
            _q0_init,
            _q0_init_g0,
            _kept_candidate_indices,
        ) = refine_q_to_zero_level(
            point_np=point,
            q_init_np=q_candidates,
            all_chain_specs=all_chain_specs,
            q_min=q_min,
            q_max=q_max,
            args=args,
            device=device,
            target_sensor=sensor_idx,
        )

        if len(q0) == 0:
            results.append(None)
            continue

        sensor_indices = np.full((len(q0),), sensor_idx, dtype=np.int16)

        if occlusion_context is None:
            occluded = np.zeros((len(q0),), dtype=np.bool_)
        elif occlusion_context["backend"] == "torch":
            occluded = compute_q0_occlusion_torch(
                point,
                q0,
                sensor_indices,
                occlusion_context,
                args,
                device,
            )
        else:
            occluded = compute_q0_occlusion(
                point,
                q0,
                sensor_indices,
                args.joint_names,
                occlusion_context,
                args,
            )

        results.append(
            {
                "q0": q0,
                "g0": g0,
                "sensor_margins0": sensor_margins0,
                "active_planes0": active_planes0,
                "occluded": occluded,
            }
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Yiming-style visibility CDF data generator with per-sensor Q0 and self-occlusion labels."
    )

    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--joint-names", nargs="*", default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--sensor-frames", nargs="*", default=DEFAULT_SENSOR_FRAMES)

    parser.add_argument("--x-min", type=float, default=-0.95)
    parser.add_argument("--x-max", type=float, default=0.95)
    parser.add_argument("--y-min", type=float, default=-0.95)
    parser.add_argument("--y-max", type=float, default=0.95)
    parser.add_argument("--z-min-bound", type=float, default=0.0)
    parser.add_argument("--z-max-bound", type=float, default=1.15)
    parser.add_argument("--grid", nargs=3, type=int, default=[30, 30, 24])

    parser.add_argument("--p-count", type=int, default=0)
    parser.add_argument("--shuffle-p", action="store_true")
    parser.add_argument("--p-seed", type=int, default=0)
    parser.add_argument("--p-num-shards", type=int, default=1)
    parser.add_argument("--p-shard-id", type=int, default=0)

    parser.add_argument("--q-init-per-p", type=int, default=20000)
    parser.add_argument("--optimize-top-k-per-sensor", type=int, default=100)
    parser.add_argument("--q0-per-sensor", type=int, default=100)

    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=66.0)
    parser.add_argument("--z-min", type=float, default=0.20)
    parser.add_argument("--z-max", type=float, default=0.70)
    parser.add_argument("--delta", type=float, default=0.01)

    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--zero-level-optimizer", choices=["adam", "lbfgs"], default="lbfgs")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)

    parser.add_argument("--occlusion-urdf", default="")
    parser.add_argument("--occlusion-backend", choices=["auto", "cpu", "torch"], default="auto")
    parser.add_argument("--ray-start-offset", type=float, default=0.03)
    parser.add_argument("--point-end-offset", type=float, default=0.005)
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--ignore-start-inside", action="store_true", default=True)
    parser.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    parser.add_argument("--ignore-links", nargs="*", default=[])

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.joint_names) != 7:
        raise RuntimeError(f"Expected 7 joints, got {len(args.joint_names)}")
    if len(args.sensor_frames) != 8:
        raise RuntimeError(f"Expected 8 sensors, got {len(args.sensor_frames)}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    rng = np.random.default_rng(args.seed)

    robot = URDF.from_xml_file(args.urdf)
    joint_limits = get_joint_limits(robot, args.joint_names)

    q_min_np = np.asarray([joint_limits[name][0] for name in args.joint_names], dtype=np.float32)
    q_max_np = np.asarray([joint_limits[name][1] for name in args.joint_names], dtype=np.float32)

    q_min = torch_tensor(q_min_np, device)
    q_max = torch_tensor(q_max_np, device)

    all_chain_specs = prepare_chain_specs(
        robot,
        args.base_frame,
        args.sensor_frames,
        args.joint_names,
        device,
    )

    occlusion_context = prepare_occlusion_context(
        args,
        args.sensor_frames,
        args.joint_names,
        device,
    )

    grid_all = build_workspace_grid(args)
    p_indices = select_point_indices(len(grid_all), args)
    points = grid_all[p_indices]

    num_p = len(points)
    num_joints = len(args.joint_names)
    num_sensors = len(args.sensor_frames)
    k_q0 = args.q0_per_sensor

    data = allocate_yiming_style_arrays(
        num_p=num_p,
        k=k_q0,
        num_joints=num_joints,
        num_sensors=num_sensors,
    )

    data["x"][:] = points
    data["k"][:] = p_indices

    print("")
    print("=== Visibility CDF Data Generator ===")
    print(f"urdf:                         {args.urdf}")
    print(f"output:                       {args.output}")
    print(f"device:                       {device}")
    print(f"grid:                         {args.grid}")
    print(f"total grid points:            {len(grid_all)}")
    print(f"selected points:              {num_p}")
    print(f"p shard:                      {args.p_shard_id}/{args.p_num_shards}")
    print(f"q_init_per_p:                 {args.q_init_per_p}")
    print(f"optimize_top_k_per_sensor:    {args.optimize_top_k_per_sensor}")
    print(f"q0_per_sensor K:              {args.q0_per_sensor}")
    print(f"zero_level_optimizer:         {args.zero_level_optimizer}")
    print(f"max_iter:                     {args.max_iter}")
    print(f"epsilon:                      {args.epsilon}")
    print(f"FOV:                          h={args.horizontal_fov_deg}, v={args.vertical_fov_deg}, z=[{args.z_min}, {args.z_max}], delta={args.delta}")
    print(f"occlusion_urdf:               {args.occlusion_urdf if args.occlusion_urdf else '<none>'}")
    print(f"occlusion_backend:            {occlusion_context['backend'] if occlusion_context is not None else '<none>'}")
    print("")
    print("Yiming-style output:")
    print(f"  x:                    [{num_p}, 3]")
    print(f"  q:                    [{num_p}, {k_q0}, 7, 8]")
    print(f"  k:                    [{num_p}]")
    print(f"  valid_fov:            [{num_p}, {k_q0}, 8]")
    print(f"  occluded:             [{num_p}, {k_q0}, 8]")
    print(f"  valid_with_occlusion: [{num_p}, {k_q0}, 8]")
    print(f"  sensor_chain_masks:   [8, 7]")
    print("")
    print("sensor_chain_masks:")
    for s, frame in enumerate(args.sensor_frames):
        print(f"  {s}: {frame:28s} {SENSOR_CHAIN_MASKS[s].astype(int).tolist()}")

    t0 = time.time()

    for out_idx, point in enumerate(points):
        q_init = sample_q_batch(
            rng=rng,
            joint_limits=joint_limits,
            joint_names=args.joint_names,
            batch_size=args.q_init_per_p,
        )

        sensor_results = generate_for_one_point(
            point=point,
            q_init=q_init,
            all_chain_specs=all_chain_specs,
            q_min=q_min,
            q_max=q_max,
            occlusion_context=occlusion_context,
            args=args,
            device=device,
        )

        for sensor_idx, result in enumerate(sensor_results):
            if result is None:
                continue

            store_sensor_block(
                data=data,
                out_idx=out_idx,
                p_idx=int(p_indices[out_idx]),
                point=point,
                sensor_idx=sensor_idx,
                q0=result["q0"],
                g0=result["g0"],
                sensor_margins0=result["sensor_margins0"],
                active_planes0=result["active_planes0"],
                occluded0=result["occluded"],
                args=args,
            )

        if args.progress_every > 0 and ((out_idx + 1) % args.progress_every == 0 or out_idx + 1 == num_p):
            elapsed = time.time() - t0
            valid_fov_count = int(np.count_nonzero(data["valid_fov"][: out_idx + 1]))
            visible_count = int(np.count_nonzero(data["valid_with_occlusion"][: out_idx + 1]))
            occ_count = int(np.count_nonzero(data["valid_fov"][: out_idx + 1] & data["occluded"][: out_idx + 1]))

            print(
                f"[progress] p {out_idx + 1}/{num_p} "
                f"valid_fov_q0={valid_fov_count} "
                f"visible_q0={visible_count} "
                f"occluded_q0={occ_count} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    valid_point_fov = np.any(data["valid_fov"], axis=(1, 2))
    valid_point_with_occlusion = np.any(data["valid_with_occlusion"], axis=(1, 2))

    save_dict = {}

    for key, value in data.items():
        save_dict[key] = value

    save_dict["valid_point_fov"] = valid_point_fov.astype(np.bool_)
    save_dict["valid_point_with_occlusion"] = valid_point_with_occlusion.astype(np.bool_)

    save_dict["sensor_chain_masks"] = SENSOR_CHAIN_MASKS.astype(np.float32)

    save_dict["joint_names"] = np.asarray(args.joint_names)
    save_dict["sensor_frames"] = np.asarray(args.sensor_frames)

    save_dict["q_min"] = q_min_np.astype(np.float32)
    save_dict["q_max"] = q_max_np.astype(np.float32)

    save_dict["horizontal_fov_deg"] = np.asarray(args.horizontal_fov_deg, dtype=np.float32)
    save_dict["vertical_fov_deg"] = np.asarray(args.vertical_fov_deg, dtype=np.float32)
    save_dict["z_min"] = np.asarray(args.z_min, dtype=np.float32)
    save_dict["z_max"] = np.asarray(args.z_max, dtype=np.float32)
    save_dict["delta"] = np.asarray(args.delta, dtype=np.float32)
    save_dict["epsilon"] = np.asarray(args.epsilon, dtype=np.float32)

    save_dict["grid_shape"] = np.asarray(args.grid, dtype=np.int32)
    save_dict["x_min"] = np.asarray(args.x_min, dtype=np.float32)
    save_dict["x_max"] = np.asarray(args.x_max, dtype=np.float32)
    save_dict["y_min"] = np.asarray(args.y_min, dtype=np.float32)
    save_dict["y_max"] = np.asarray(args.y_max, dtype=np.float32)
    save_dict["z_min_bound"] = np.asarray(args.z_min_bound, dtype=np.float32)
    save_dict["z_max_bound"] = np.asarray(args.z_max_bound, dtype=np.float32)

    save_dict["q_init_per_p"] = np.asarray(args.q_init_per_p, dtype=np.int32)
    save_dict["optimize_top_k_per_sensor"] = np.asarray(args.optimize_top_k_per_sensor, dtype=np.int32)
    save_dict["q0_per_sensor"] = np.asarray(args.q0_per_sensor, dtype=np.int32)
    save_dict["zero_level_optimizer"] = np.asarray(args.zero_level_optimizer)
    save_dict["max_iter"] = np.asarray(args.max_iter, dtype=np.int32)
    save_dict["lr"] = np.asarray(args.lr, dtype=np.float32)
    save_dict["lbfgs_lr"] = np.asarray(args.lbfgs_lr, dtype=np.float32)

    save_dict["occlusion_urdf"] = np.asarray(args.occlusion_urdf)
    save_dict["occlusion_backend"] = np.asarray(occlusion_context["backend"] if occlusion_context is not None else "none")
    save_dict["has_occlusion_labels"] = np.asarray(occlusion_context is not None, dtype=np.bool_)

    save_dict["seed"] = np.asarray(args.seed, dtype=np.int32)
    save_dict["p_seed"] = np.asarray(args.p_seed, dtype=np.int32)
    save_dict["p_num_shards"] = np.asarray(args.p_num_shards, dtype=np.int32)
    save_dict["p_shard_id"] = np.asarray(args.p_shard_id, dtype=np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save_dict)

    elapsed = time.time() - t0

    valid_fov_count = int(np.count_nonzero(data["valid_fov"]))
    visible_count = int(np.count_nonzero(data["valid_with_occlusion"]))
    occ_count = int(np.count_nonzero(data["valid_fov"] & data["occluded"]))

    print("")
    print("=== Data Generation Summary ===")
    print(f"processed points:             {num_p}")
    print(f"valid_point_fov:              {int(valid_point_fov.sum())}")
    print(f"valid_point_with_occlusion:   {int(valid_point_with_occlusion.sum())}")
    print(f"valid_fov q0:                 {valid_fov_count}")
    print(f"visible q0:                   {visible_count}")
    print(f"occluded q0:                  {occ_count}")
    print(f"occluded ratio among fov q0:  {occ_count / max(valid_fov_count, 1):.6f}")
    print("")
    print("Per-sensor q0 statistics:")
    for s, frame in enumerate(args.sensor_frames):
        n_all = data["num_sensor_q0"][:, s]
        n_occ = data["num_sensor_q0_occluded"][:, s]
        n_vis = data["num_sensor_q0_visible"][:, s]
        print(
            f"  {s}: {frame:28s} "
            f"all_mean={n_all.mean():.3f} "
            f"visible_mean={n_vis.mean():.3f} "
            f"occluded_mean={n_occ.mean():.3f}"
        )
    print("")
    print(f"elapsed_sec:                  {elapsed:.2f}")
    print(f"saved:                        {args.output}")


if __name__ == "__main__":
    main()
