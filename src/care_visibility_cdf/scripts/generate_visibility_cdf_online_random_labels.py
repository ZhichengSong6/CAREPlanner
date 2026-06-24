#!/usr/bin/env python3

import argparse
import os
import sys
import time

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
from generate_visibility_raw_samples import (  # noqa: E402
    compute_sensor_occlusion_for_sample,
    compute_visibility_with_occlusion,
    prepare_occlusion_context,
    q_row_to_map,
    sample_q_batch,
)


def names_from_npz(data, key, default):
    if key in data:
        return [str(x) for x in data[key]]
    return list(default)


def scalar_from_npz(data, key, default):
    if key in data:
        return float(np.asarray(data[key]).item())
    return default


def select_valid_q0_set(q0_data, row):
    q0_templates = q0_data["q0_templates"].astype(np.float32)
    q0_g = q0_data["q0_g"]
    num_q0 = q0_data["num_q0"].astype(np.int64)

    k = int(num_q0[row])
    if k <= 0:
        return None, None

    finite = np.isfinite(q0_g[row, :k]) & np.isfinite(q0_templates[row, :k]).all(axis=1)
    if "q0_valid_with_occlusion" in q0_data:
        finite = finite & q0_data["q0_valid_with_occlusion"][row, :k].astype(np.bool_)

    original_indices = np.where(finite)[0].astype(np.int16)
    if len(original_indices) == 0:
        return None, None

    return q0_templates[row, original_indices], original_indices


def q_distance_l2(q_batch, q0_set, weights):
    diff = q_batch[:, None, :] - q0_set[None, :, :]
    if weights is not None:
        diff = diff * weights.reshape(1, 1, -1)
    return np.linalg.norm(diff, axis=2)


def q_distance_l1(q_batch, q0_set, weights):
    diff = np.abs(q_batch[:, None, :] - q0_set[None, :, :])
    if weights is not None:
        diff = diff * weights.reshape(1, 1, -1)
    return np.sum(diff, axis=2)


def compute_sensor_transforms(sensor_chains, joint_names, q_row):
    q_map = q_row_to_map(joint_names, q_row)
    return [fk_transform(chain, q_map) for chain in sensor_chains]


def compute_one_visibility(point, q_row, sensor_chains, joint_names, args, occlusion_context):
    sensor_transforms = compute_sensor_transforms(sensor_chains, joint_names, q_row)
    sensor_margins, active_planes, best_sensor, best_margin, g, visible_fov = compute_visibility(
        point.reshape(1, 3),
        sensor_transforms,
        args,
    )

    sensor_margins = sensor_margins[0].astype(np.float32)
    active_planes = active_planes[0].astype(np.int16)
    best_sensor_fov = int(best_sensor[0])
    best_margin_fov = float(best_margin[0])
    g_fov = float(g[0])
    visible_fov = bool(visible_fov[0])

    if occlusion_context is None:
        return {
            "sensor_margins": sensor_margins,
            "active_planes": active_planes,
            "best_sensor_fov": best_sensor_fov,
            "best_margin_fov": best_margin_fov,
            "g_fov": g_fov,
            "visible_fov": visible_fov,
            "visible": visible_fov,
            "best_visible_sensor": best_sensor_fov if visible_fov else -1,
            "best_visible_margin": best_margin_fov if visible_fov else -np.inf,
            "sensor_occluded": np.zeros((len(sensor_margins),), dtype=np.bool_),
        }

    sensor_occluded = compute_sensor_occlusion_for_sample(
        point,
        q_row,
        joint_names,
        occlusion_context,
        args,
    )
    (
        _sensor_fov_visible,
        _sensor_visible,
        best_visible_sensor,
        best_visible_margin,
        visible_with_occlusion,
    ) = compute_visibility_with_occlusion(
        sensor_margins.reshape(1, -1),
        sensor_occluded.reshape(1, -1),
        args.delta,
    )

    return {
        "sensor_margins": sensor_margins,
        "active_planes": active_planes,
        "best_sensor_fov": best_sensor_fov,
        "best_margin_fov": best_margin_fov,
        "g_fov": g_fov,
        "visible_fov": visible_fov,
        "visible": bool(visible_with_occlusion[0]),
        "best_visible_sensor": int(best_visible_sensor[0]),
        "best_visible_margin": float(best_visible_margin[0]),
        "sensor_occluded": sensor_occluded.astype(np.bool_),
    }


def append(store, key, value):
    store[key].append(value)


def concat(store, key, dtype=None):
    out = np.concatenate(store[key], axis=0)
    if dtype is not None:
        out = out.astype(dtype)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate yiming-style VCDF training labels: use M4 Q0(p) as zero-level database, "
            "online-sample random q, and label each (p,q) by signed min distance to Q0(p)."
        )
    )
    parser.add_argument("--urdf", required=True, help="Robot URDF for FK and joint limits.")
    parser.add_argument("--q0", required=True, help="M4 zero-level .npz file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--joint-names", nargs="*", default=None)
    parser.add_argument("--sensor-frames", nargs="*", default=None)
    parser.add_argument("--q-per-point", type=int, default=100)
    parser.add_argument("--max-points", type=int, default=0,
                        help="Optional cap on valid M4 p rows for quick tests.")
    parser.add_argument("--metric", choices=["l2", "l1"], default="l2")
    parser.add_argument("--also-save-other-metric", action="store_true")
    parser.add_argument("--joint-weights", nargs="*", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-seed", type=int, default=0,
                        help="Seed used only when --shuffle-p is set.")
    parser.add_argument("--shuffle-p", action="store_true",
                        help="Shuffle valid M4 p rows before optional --max-points.")
    parser.add_argument("--progress-every", type=int, default=100)

    parser.add_argument("--horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--vertical-fov-deg", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)

    parser.add_argument("--occlusion-urdf", default="",
                        help="Simplified collision URDF. If omitted, sign uses FOV only.")
    parser.add_argument("--ray-start-offset", type=float, default=0.03)
    parser.add_argument("--point-end-offset", type=float, default=0.005)
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--ignore-start-inside", action="store_true", default=True)
    parser.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    parser.add_argument("--ignore-links", nargs="*", default=[])
    args = parser.parse_args()

    q0_data = np.load(args.q0, allow_pickle=True)
    required = ["grid_points", "p_indices", "q0_templates", "q0_g", "num_q0"]
    missing = [key for key in required if key not in q0_data]
    if missing:
        raise RuntimeError(f"q0 file missing required keys: {missing}")

    joint_names = args.joint_names or names_from_npz(q0_data, "joint_names", DEFAULT_JOINT_NAMES)
    sensor_frames = args.sensor_frames or names_from_npz(q0_data, "sensor_frames", DEFAULT_SENSOR_FRAMES)

    args.horizontal_fov_deg = (
        scalar_from_npz(q0_data, "horizontal_fov_deg", 50.0)
        if args.horizontal_fov_deg is None else args.horizontal_fov_deg
    )
    args.vertical_fov_deg = (
        scalar_from_npz(q0_data, "vertical_fov_deg", 66.0)
        if args.vertical_fov_deg is None else args.vertical_fov_deg
    )
    args.z_min = scalar_from_npz(q0_data, "z_min", 0.20) if args.z_min is None else args.z_min
    args.z_max = scalar_from_npz(q0_data, "z_max", 0.70) if args.z_max is None else args.z_max
    args.delta = scalar_from_npz(q0_data, "delta", 0.01) if args.delta is None else args.delta

    if args.joint_weights is not None:
        weights = np.asarray(args.joint_weights, dtype=np.float32)
        if weights.shape != (len(joint_names),):
            raise RuntimeError(f"Expected {len(joint_names)} joint weights, got {weights.shape}")
    else:
        weights = None

    rng = np.random.default_rng(args.seed)
    robot = URDF.from_xml_file(args.urdf)
    joint_limits = get_joint_limits(robot, joint_names)
    sensor_chains = [find_chain_joints(robot, args.base_frame, frame) for frame in sensor_frames]
    occlusion_context = prepare_occlusion_context(args, sensor_frames)

    grid_points = q0_data["grid_points"].astype(np.float32)
    p_indices = q0_data["p_indices"].astype(np.int64)
    valid_mask = q0_data["valid_mask_with_occlusion"].astype(np.bool_) if "valid_mask_with_occlusion" in q0_data else q0_data["valid_mask"].astype(np.bool_)

    valid_rows = np.where(valid_mask & (q0_data["num_q0"].astype(np.int64) > 0))[0].astype(np.int64)
    if args.shuffle_p:
        p_rng = np.random.default_rng(args.p_seed)
        valid_rows = p_rng.permutation(valid_rows)
    if args.max_points > 0:
        valid_rows = valid_rows[:args.max_points]
    if len(valid_rows) == 0:
        raise RuntimeError("No valid M4 p rows found.")

    total_samples = len(valid_rows) * args.q_per_point
    print("")
    print("=== M5 Online Random VCDF Label Config ===")
    print(f"urdf:           {args.urdf}")
    print(f"q0:             {args.q0}")
    print(f"output:         {args.output}")
    print(f"valid p rows:   {len(valid_rows)}")
    print(f"q_per_point:    {args.q_per_point}")
    print(f"total samples:  {total_samples}")
    print(f"metric:         {args.metric}")
    print(f"fov/range:      h={args.horizontal_fov_deg}, v={args.vertical_fov_deg}, z=[{args.z_min}, {args.z_max}]")
    print(f"delta:          {args.delta}")
    print(f"occlusion_urdf: {args.occlusion_urdf if args.occlusion_urdf else '<none>'}")
    print(f"seed:           {args.seed}")

    store = {
        "p": [], "q": [], "cdf_l2": [], "cdf_l1": [],
        "distance_l2": [], "distance_l1": [], "sign": [], "visible": [],
        "nearest_q0_index_l2": [], "nearest_q0_index_l1": [],
        "p_index": [], "q0_file_row": [], "raw_index": [],
        "g_fov": [], "best_sensor_fov": [], "best_margin_fov": [], "visible_fov": [],
        "best_visible_sensor": [], "best_visible_margin": [],
    }

    visible_count = 0
    visible_fov_count = 0
    skipped_no_q0 = 0
    raw_counter = 0
    t0 = time.time()

    for out_row_idx, q0_row in enumerate(valid_rows):
        q0_set, q0_original_indices = select_valid_q0_set(q0_data, q0_row)
        if q0_set is None:
            skipped_no_q0 += 1
            continue

        p_idx = int(p_indices[q0_row])
        point = grid_points[q0_row]
        q_batch = sample_q_batch(rng, joint_limits, joint_names, args.q_per_point).astype(np.float32)
        p_batch = np.repeat(point.reshape(1, 3), args.q_per_point, axis=0).astype(np.float32)

        dist_l2_all = q_distance_l2(q_batch, q0_set, weights)
        nearest_l2_local = np.argmin(dist_l2_all, axis=1)
        dist_l2 = dist_l2_all[np.arange(len(q_batch)), nearest_l2_local].astype(np.float32)
        nearest_l2 = q0_original_indices[nearest_l2_local].astype(np.int16)

        dist_l1_all = q_distance_l1(q_batch, q0_set, weights)
        nearest_l1_local = np.argmin(dist_l1_all, axis=1)
        dist_l1 = dist_l1_all[np.arange(len(q_batch)), nearest_l1_local].astype(np.float32)
        nearest_l1 = q0_original_indices[nearest_l1_local].astype(np.int16)

        visible = np.zeros((args.q_per_point,), dtype=np.bool_)
        visible_fov = np.zeros((args.q_per_point,), dtype=np.bool_)
        g_fov = np.zeros((args.q_per_point,), dtype=np.float32)
        best_sensor_fov = np.zeros((args.q_per_point,), dtype=np.int16)
        best_margin_fov = np.zeros((args.q_per_point,), dtype=np.float32)
        best_visible_sensor = np.zeros((args.q_per_point,), dtype=np.int16)
        best_visible_margin = np.zeros((args.q_per_point,), dtype=np.float32)

        for q_idx, q_row in enumerate(q_batch):
            info = compute_one_visibility(
                point,
                q_row,
                sensor_chains,
                joint_names,
                args,
                occlusion_context,
            )
            visible[q_idx] = info["visible"]
            visible_fov[q_idx] = info["visible_fov"]
            g_fov[q_idx] = info["g_fov"]
            best_sensor_fov[q_idx] = info["best_sensor_fov"]
            best_margin_fov[q_idx] = info["best_margin_fov"]
            best_visible_sensor[q_idx] = info["best_visible_sensor"]
            best_visible_margin[q_idx] = info["best_visible_margin"]

        sign = np.where(visible, 1.0, -1.0).astype(np.float32)
        cdf_l2 = sign * dist_l2
        cdf_l1 = sign * dist_l1

        append(store, "p", p_batch)
        append(store, "q", q_batch)
        append(store, "cdf_l2", cdf_l2.astype(np.float32))
        append(store, "cdf_l1", cdf_l1.astype(np.float32))
        append(store, "distance_l2", dist_l2.astype(np.float32))
        append(store, "distance_l1", dist_l1.astype(np.float32))
        append(store, "sign", sign.astype(np.float32))
        append(store, "visible", visible.astype(np.bool_))
        append(store, "nearest_q0_index_l2", nearest_l2)
        append(store, "nearest_q0_index_l1", nearest_l1)
        append(store, "p_index", np.full((args.q_per_point,), p_idx, dtype=np.int64))
        append(store, "q0_file_row", np.full((args.q_per_point,), q0_row, dtype=np.int64))
        append(store, "raw_index", np.arange(raw_counter, raw_counter + args.q_per_point, dtype=np.int64))
        append(store, "g_fov", g_fov)
        append(store, "best_sensor_fov", best_sensor_fov)
        append(store, "best_margin_fov", best_margin_fov)
        append(store, "visible_fov", visible_fov)
        append(store, "best_visible_sensor", best_visible_sensor)
        append(store, "best_visible_margin", best_visible_margin)

        raw_counter += args.q_per_point
        visible_count += int(np.count_nonzero(visible))
        visible_fov_count += int(np.count_nonzero(visible_fov))

        if args.progress_every > 0 and (out_row_idx + 1) % args.progress_every == 0:
            elapsed = time.time() - t0
            done = (out_row_idx + 1) * args.q_per_point
            print(
                f"[progress] p {out_row_idx + 1}/{len(valid_rows)} "
                f"samples {done}/{total_samples} "
                f"visible_ratio {visible_count / max(done, 1):.4f} "
                f"elapsed {elapsed:.1f}s"
            )

    if not store["q"]:
        raise RuntimeError("No labels generated.")

    distance_l2 = concat(store, "distance_l2", np.float32)
    distance_l1 = concat(store, "distance_l1", np.float32)
    cdf_l2 = concat(store, "cdf_l2", np.float32)
    cdf_l1 = concat(store, "cdf_l1", np.float32)
    if args.metric == "l2":
        primary_cdf = cdf_l2
        primary_distance = distance_l2
        primary_nearest = concat(store, "nearest_q0_index_l2", np.int16)
    else:
        primary_cdf = cdf_l1
        primary_distance = distance_l1
        primary_nearest = concat(store, "nearest_q0_index_l1", np.int16)

    save_data = {
        "p": concat(store, "p", np.float32),
        "q": concat(store, "q", np.float32),
        "cdf": primary_cdf.astype(np.float32),
        "distance_unsigned": primary_distance.astype(np.float32),
        "sign": concat(store, "sign", np.float32),
        "visible": concat(store, "visible", np.bool_),
        "nearest_q0_index": primary_nearest.astype(np.int16),
        "p_index": concat(store, "p_index", np.int64),
        "raw_index": concat(store, "raw_index", np.int64),
        "q0_file_row": concat(store, "q0_file_row", np.int64),
        "g_fov": concat(store, "g_fov", np.float32),
        "best_sensor_fov": concat(store, "best_sensor_fov", np.int16),
        "best_margin_fov": concat(store, "best_margin_fov", np.float32),
        "visible_fov": concat(store, "visible_fov", np.bool_),
        "best_visible_sensor": concat(store, "best_visible_sensor", np.int16),
        "best_visible_margin": concat(store, "best_visible_margin", np.float32),
        "joint_names": np.asarray(joint_names),
        "sensor_frames": np.asarray(sensor_frames),
        "metric": np.asarray(args.metric),
        "raw_path": np.asarray("online_random_q"),
        "q0_path": np.asarray(args.q0),
        "visible_key": np.asarray("visible_with_occlusion" if occlusion_context is not None else "visible_fov"),
        "q_per_point": np.asarray(args.q_per_point, dtype=np.int32),
        "joint_weights": np.asarray([] if weights is None else weights, dtype=np.float32),
        "horizontal_fov_deg": np.asarray(args.horizontal_fov_deg, dtype=np.float32),
        "vertical_fov_deg": np.asarray(args.vertical_fov_deg, dtype=np.float32),
        "z_min": np.asarray(args.z_min, dtype=np.float32),
        "z_max": np.asarray(args.z_max, dtype=np.float32),
        "delta": np.asarray(args.delta, dtype=np.float32),
        "occlusion_urdf": np.asarray(args.occlusion_urdf),
        "seed": np.asarray(args.seed, dtype=np.int32),
        "p_seed": np.asarray(args.p_seed, dtype=np.int32),
        "source": np.asarray("m4_q0_database_plus_online_random_q"),
    }
    if args.also_save_other_metric or args.metric == "l2":
        save_data["distance_l2"] = distance_l2
        save_data["cdf_l2"] = cdf_l2
        save_data["nearest_q0_index_l2"] = concat(store, "nearest_q0_index_l2", np.int16)
    if args.also_save_other_metric or args.metric == "l1":
        save_data["distance_l1"] = distance_l1
        save_data["cdf_l1"] = cdf_l1
        save_data["nearest_q0_index_l1"] = concat(store, "nearest_q0_index_l1", np.int16)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save_data)

    elapsed = time.time() - t0
    visible_out = save_data["visible"]
    visible_fov_out = save_data["visible_fov"]
    print("")
    print("=== M5 Online Random VCDF Label Summary ===")
    print(f"q0:                   {args.q0}")
    print(f"output:               {args.output}")
    print(f"metric:               {args.metric}")
    print(f"processed p rows:     {len(valid_rows)}")
    print(f"skipped no q0:        {skipped_no_q0}")
    print(f"samples generated:    {len(primary_cdf)}")
    print(f"visible_fov samples:  {int(np.count_nonzero(visible_fov_out))}")
    print(f"visible samples:      {int(np.count_nonzero(visible_out))}")
    print(f"visible ratio:        {visible_out.mean():.6f}")
    print(f"cdf min/mean/max:     {primary_cdf.min(): .6f} / {primary_cdf.mean(): .6f} / {primary_cdf.max(): .6f}")
    print(f"unsigned min/mean/max:{primary_distance.min(): .6f} / {primary_distance.mean(): .6f} / {primary_distance.max(): .6f}")
    print(f"elapsed_sec:          {elapsed:.2f}")
    print(f"saved:                {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
