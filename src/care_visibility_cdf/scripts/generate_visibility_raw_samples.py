#!/usr/bin/env python3

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from urdf_parser_py.urdf import URDF

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    compute_visibility,
    find_chain_joints,
    fk_transform,
    get_joint_limits,
)
from check_visibility_self_occlusion import (  # noqa: E402
    load_collision_primitives,
    raycast_self_occlusion,
)


def build_grid(bounds: np.ndarray, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    xs = np.linspace(bounds[0, 0], bounds[0, 1], grid_shape[0])
    ys = np.linspace(bounds[1, 0], bounds[1, 1], grid_shape[1])
    zs = np.linspace(bounds[2, 0], bounds[2, 1], grid_shape[2])
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def sample_q_batch(
    rng: np.random.Generator,
    joint_limits: Dict[str, Tuple[float, float]],
    joint_names: List[str],
    batch_size: int,
) -> np.ndarray:
    q = np.zeros((batch_size, len(joint_names)), dtype=np.float64)
    for joint_idx, name in enumerate(joint_names):
        lower, upper = joint_limits[name]
        q[:, joint_idx] = rng.uniform(lower, upper, size=batch_size)
    return q


def q_row_to_map(joint_names: List[str], q_row: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(joint_names, q_row)}


def compute_sensor_transforms(sensor_chains, joint_names: List[str], q_row: np.ndarray):
    q_map = q_row_to_map(joint_names, q_row)
    return [fk_transform(chain, q_map) for chain in sensor_chains]


def prepare_occlusion_context(args, sensor_frames):
    if not args.occlusion_urdf:
        return None

    robot = URDF.from_xml_file(args.occlusion_urdf)
    primitives = load_collision_primitives(robot)
    if not primitives:
        raise RuntimeError(
            f"No supported collision primitives found in {args.occlusion_urdf}. "
            "Expected box/cylinder/sphere collision geometry."
        )

    return {
        "robot": robot,
        "primitives": primitives,
        "sensor_chains": [
            find_chain_joints(robot, args.base_frame, frame)
            for frame in sensor_frames
        ],
        "collision_chains": [
            find_chain_joints(robot, args.base_frame, primitive["link"])
            for primitive in primitives
        ],
    }


def compute_sensor_occlusion_for_sample(point, q_row, joint_names, occlusion_context, args):
    if occlusion_context is None:
        return np.zeros((0,), dtype=np.bool_)

    q_map = q_row_to_map(joint_names, q_row)
    occluded = np.zeros((len(occlusion_context["sensor_chains"]),), dtype=np.bool_)
    for sensor_idx, chain in enumerate(occlusion_context["sensor_chains"]):
        sensor_transform = fk_transform(chain, q_map)
        is_occluded, _ = raycast_self_occlusion(
            occlusion_context["robot"],
            occlusion_context["collision_chains"],
            occlusion_context["primitives"],
            sensor_transform,
            point.astype(np.float64),
            q_map,
            args,
        )
        occluded[sensor_idx] = is_occluded
    return occluded


def compute_visibility_with_occlusion(sensor_margins, sensor_occluded, delta):
    sensor_fov_visible = sensor_margins >= delta
    sensor_visible = sensor_fov_visible & (~sensor_occluded)
    visible = np.any(sensor_visible, axis=1)

    masked_margins = np.where(sensor_visible, sensor_margins, -np.inf)
    best_sensor = np.argmax(masked_margins, axis=1).astype(np.int16)
    best_margin = masked_margins[np.arange(len(masked_margins)), best_sensor].astype(np.float32)

    no_visible = ~visible
    best_sensor[no_visible] = -1
    best_margin[no_visible] = -np.inf

    return (
        sensor_fov_visible.astype(np.bool_),
        sensor_visible.astype(np.bool_),
        best_sensor,
        best_margin,
        visible.astype(np.bool_),
    )


def append_chunk(chunk_store, p_chunk, q_chunk, sensor_margins, active_planes,
                 best_sensor, best_margin, g, visible, occlusion_chunk=None):
    chunk_store["p"].append(p_chunk.astype(np.float32))
    chunk_store["q"].append(q_chunk.astype(np.float32))
    chunk_store["sensor_margins"].append(sensor_margins.astype(np.float32))
    chunk_store["active_planes"].append(active_planes.astype(np.int16))
    chunk_store["best_sensor"].append(best_sensor.astype(np.int16))
    chunk_store["best_margin"].append(best_margin.astype(np.float32))
    chunk_store["g"].append(g.astype(np.float32))
    chunk_store["visible"].append(visible.astype(np.bool_))
    chunk_store["best_sensor_fov"].append(best_sensor.astype(np.int16))
    chunk_store["best_margin_fov"].append(best_margin.astype(np.float32))
    chunk_store["g_fov"].append(g.astype(np.float32))
    chunk_store["visible_fov"].append(visible.astype(np.bool_))
    if occlusion_chunk is not None:
        for key, value in occlusion_chunk.items():
            chunk_store[key].append(value)


def concat_store(chunk_store):
    return {key: np.concatenate(values, axis=0) for key, values in chunk_store.items()}


def print_running_stats(num_samples, visible_count, best_sensor_counts, g_min, g_max, g_sum,
                        sensor_frames, elapsed):
    print("")
    print("=== Raw Visibility Dataset Summary ===")
    print(f"num_samples:     {num_samples}")
    print(f"visible_samples: {visible_count}")
    print(f"visible_ratio:   {visible_count / max(num_samples, 1):.6f}")
    print(f"g min/mean/max:  {g_min: .6f} / {g_sum / max(num_samples, 1): .6f} / {g_max: .6f}")
    print(f"elapsed_sec:     {elapsed:.2f}")
    print("")
    print("best_sensor histogram:")
    for idx, frame in enumerate(sensor_frames):
        print(f"  {idx}: {frame:28s} {int(best_sensor_counts[idx]):8d}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate M3 raw visibility samples for visibility CDF data collection."
    )
    parser.add_argument("--urdf", required=True, help="Path to robot URDF.")
    parser.add_argument("--output", required=True, help="Output .npz file.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--joint-names", nargs="*", default=DEFAULT_JOINT_NAMES)
    parser.add_argument("--sensor-frames", nargs="*", default=DEFAULT_SENSOR_FRAMES)

    parser.add_argument("--x-min", type=float, default=-0.95)
    parser.add_argument("--x-max", type=float, default=0.95)
    parser.add_argument("--y-min", type=float, default=-0.95)
    parser.add_argument("--y-max", type=float, default=0.95)
    parser.add_argument("--z-min-bound", type=float, default=0.0)
    parser.add_argument("--z-max-bound", type=float, default=1.15)

    parser.add_argument("--grid", nargs=3, type=int, default=[20, 20, 16],
                        help="Workspace p grid shape. CDF paper uses a volumetric grid; this is our M3 equivalent.")
    parser.add_argument("--q-per-point", type=int, default=16,
                        help="Number of random q samples for each workspace grid point.")
    parser.add_argument("--max-points", type=int, default=0,
                        help="Optional cap on number of grid points, useful for quick tests.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=66.0)
    parser.add_argument("--z-min", type=float, default=0.20,
                        help="Conservative ToF near range in sensor frame.")
    parser.add_argument("--z-max", type=float, default=0.70,
                        help="Conservative ToF far range in sensor frame.")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Visibility margin subtracted from best_margin to define g.")
    parser.add_argument("--progress-every", type=int, default=200,
                        help="Print progress every this many workspace grid points.")
    parser.add_argument("--occlusion-urdf", default="",
                        help="Optional simplified collision URDF for self-occlusion labels.")
    parser.add_argument("--ray-start-offset", type=float, default=0.03)
    parser.add_argument("--point-end-offset", type=float, default=0.005)
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--ignore-start-inside", action="store_true", default=True)
    parser.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    parser.add_argument("--ignore-links", nargs="*", default=[])

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    robot = URDF.from_xml_file(args.urdf)
    joint_limits = get_joint_limits(robot, args.joint_names)
    sensor_chains = [
        find_chain_joints(robot, args.base_frame, frame)
        for frame in args.sensor_frames
    ]
    occlusion_context = prepare_occlusion_context(args, args.sensor_frames)

    bounds = np.array([
        [args.x_min, args.x_max],
        [args.y_min, args.y_max],
        [args.z_min_bound, args.z_max_bound],
    ], dtype=np.float64)
    grid_points = build_grid(bounds, tuple(args.grid))
    if args.max_points > 0:
        grid_points = grid_points[:args.max_points]

    total_samples = len(grid_points) * args.q_per_point
    print("")
    print("=== M3 Raw Visibility Sampling Config ===")
    print(f"urdf:          {args.urdf}")
    print(f"output:        {args.output}")
    print(f"bounds:        x[{args.x_min}, {args.x_max}], y[{args.y_min}, {args.y_max}], z[{args.z_min_bound}, {args.z_max_bound}]")
    print(f"grid:          {args.grid} -> {len(grid_points)} p samples")
    print(f"q_per_point:   {args.q_per_point}")
    print(f"total_samples: {total_samples}")
    print(f"fov:           h={args.horizontal_fov_deg} deg, v={args.vertical_fov_deg} deg")
    print(f"range:         z_min={args.z_min}, z_max={args.z_max}")
    print(f"delta:         {args.delta}")
    print(f"occlusion_urdf:{args.occlusion_urdf if args.occlusion_urdf else '<none>'}")
    print(f"seed:          {args.seed}")

    chunk_store = {
        "p": [],
        "q": [],
        "sensor_margins": [],
        "active_planes": [],
        "best_sensor": [],
        "best_margin": [],
        "g": [],
        "visible": [],
        "best_sensor_fov": [],
        "best_margin_fov": [],
        "g_fov": [],
        "visible_fov": [],
    }
    if occlusion_context is not None:
        chunk_store.update({
            "sensor_occluded": [],
            "sensor_fov_visible": [],
            "sensor_visible_with_occlusion": [],
            "best_visible_sensor": [],
            "best_visible_margin": [],
            "visible_with_occlusion": [],
        })

    visible_count = 0
    best_sensor_counts = np.zeros(len(args.sensor_frames), dtype=np.int64)
    g_min = np.inf
    g_max = -np.inf
    g_sum = 0.0
    num_done = 0
    t0 = time.time()

    for point_idx, point in enumerate(grid_points):
        q_batch = sample_q_batch(rng, joint_limits, args.joint_names, args.q_per_point)
        p_batch = np.repeat(point.reshape(1, 3), args.q_per_point, axis=0)

        sensor_margins_list = []
        active_planes_list = []
        best_sensor_list = []
        best_margin_list = []
        g_list = []
        visible_list = []
        sensor_occluded_list = []

        for q_row in q_batch:
            sensor_transforms = compute_sensor_transforms(sensor_chains, args.joint_names, q_row)
            sensor_margins, active_planes, best_sensor, best_margin, g, visible = compute_visibility(
                point.reshape(1, 3), sensor_transforms, args
            )
            sensor_margins_list.append(sensor_margins[0])
            active_planes_list.append(active_planes[0])
            best_sensor_list.append(best_sensor[0])
            best_margin_list.append(best_margin[0])
            g_list.append(g[0])
            visible_list.append(visible[0])
            if occlusion_context is not None:
                sensor_occluded_list.append(
                    compute_sensor_occlusion_for_sample(
                        point,
                        q_row,
                        args.joint_names,
                        occlusion_context,
                        args,
                    )
                )

        sensor_margins_arr = np.asarray(sensor_margins_list)
        active_planes_arr = np.asarray(active_planes_list)
        best_sensor_arr = np.asarray(best_sensor_list)
        best_margin_arr = np.asarray(best_margin_list)
        g_arr = np.asarray(g_list)
        visible_arr = np.asarray(visible_list)
        occlusion_chunk = None
        if occlusion_context is not None:
            sensor_occluded_arr = np.asarray(sensor_occluded_list, dtype=np.bool_)
            (
                sensor_fov_visible_arr,
                sensor_visible_arr,
                best_visible_sensor_arr,
                best_visible_margin_arr,
                visible_with_occlusion_arr,
            ) = compute_visibility_with_occlusion(
                sensor_margins_arr,
                sensor_occluded_arr,
                args.delta,
            )
            occlusion_chunk = {
                "sensor_occluded": sensor_occluded_arr.astype(np.bool_),
                "sensor_fov_visible": sensor_fov_visible_arr.astype(np.bool_),
                "sensor_visible_with_occlusion": sensor_visible_arr.astype(np.bool_),
                "best_visible_sensor": best_visible_sensor_arr.astype(np.int16),
                "best_visible_margin": best_visible_margin_arr.astype(np.float32),
                "visible_with_occlusion": visible_with_occlusion_arr.astype(np.bool_),
            }

        append_chunk(
            chunk_store,
            p_batch,
            q_batch,
            sensor_margins_arr,
            active_planes_arr,
            best_sensor_arr,
            best_margin_arr,
            g_arr,
            visible_arr,
            occlusion_chunk,
        )

        num_done += args.q_per_point
        visible_count += int(visible_arr.sum())
        best_sensor_counts += np.bincount(best_sensor_arr, minlength=len(args.sensor_frames))
        g_min = min(g_min, float(g_arr.min()))
        g_max = max(g_max, float(g_arr.max()))
        g_sum += float(g_arr.sum())

        if args.progress_every > 0 and (point_idx + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"[progress] p {point_idx + 1}/{len(grid_points)} "
                f"samples {num_done}/{total_samples} "
                f"visible_ratio {visible_count / max(num_done, 1):.4f} "
                f"elapsed {elapsed:.1f}s"
            )

    data = concat_store(chunk_store)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        **data,
        grid_points=grid_points.astype(np.float32),
        joint_names=np.asarray(args.joint_names),
        sensor_frames=np.asarray(args.sensor_frames),
        bounds=bounds.astype(np.float32),
        grid_shape=np.asarray(args.grid, dtype=np.int32),
        q_per_point=np.asarray(args.q_per_point, dtype=np.int32),
        horizontal_fov_deg=np.asarray(args.horizontal_fov_deg, dtype=np.float32),
        vertical_fov_deg=np.asarray(args.vertical_fov_deg, dtype=np.float32),
        z_min=np.asarray(args.z_min, dtype=np.float32),
        z_max=np.asarray(args.z_max, dtype=np.float32),
        delta=np.asarray(args.delta, dtype=np.float32),
        occlusion_urdf=np.asarray(args.occlusion_urdf),
        has_occlusion_labels=np.asarray(occlusion_context is not None, dtype=np.bool_),
        seed=np.asarray(args.seed, dtype=np.int32),
    )

    elapsed = time.time() - t0
    print_running_stats(
        total_samples,
        visible_count,
        best_sensor_counts,
        g_min,
        g_max,
        g_sum,
        args.sensor_frames,
        elapsed,
    )
    print("")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
