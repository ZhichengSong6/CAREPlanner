#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np


def require_keys(data, keys, name):
    missing = [key for key in keys if key not in data]
    if missing:
        raise RuntimeError(f"{name} missing required keys: {missing}")


def q_distance(q_batch, q0_set, metric, weights):
    diff = q_batch[:, None, :] - q0_set[None, :, :]
    if weights is not None:
        diff = diff * weights.reshape(1, 1, -1)

    if metric == "l2":
        return np.linalg.norm(diff, axis=2)
    if metric == "l1":
        return np.sum(np.abs(diff), axis=2)
    raise RuntimeError(f"Unsupported metric: {metric}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate signed visibility CDF training labels from M3 raw samples and M4 Q0 templates."
    )
    parser.add_argument("--raw", required=True, help="M3 raw visibility .npz with self-occlusion labels.")
    parser.add_argument("--q0", required=True, help="M4 global q0 .npz with self-occlusion filtering.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", choices=["l2", "l1"], default="l2",
                        help="Primary CDF distance metric.")
    parser.add_argument("--also-save-other-metric", action="store_true",
                        help="Save both L1 and L2 labels regardless of --metric.")
    parser.add_argument("--joint-weights", nargs="*", type=float, default=None,
                        help="Optional 7 joint weights applied before distance computation.")
    parser.add_argument("--require-occlusion-labels", action="store_true", default=True)
    parser.add_argument("--allow-fov-only-sign", dest="require_occlusion_labels", action="store_false")
    args = parser.parse_args()

    raw = np.load(args.raw, allow_pickle=True)
    q0_data = np.load(args.q0, allow_pickle=True)

    require_keys(
        raw,
        ["p", "q", "grid_points", "q_per_point", "joint_names", "sensor_frames", "g", "visible"],
        "raw",
    )
    require_keys(
        q0_data,
        ["p_indices", "grid_points", "q0_templates", "q0_g", "num_q0", "valid_mask"],
        "q0",
    )

    if args.require_occlusion_labels and "visible_with_occlusion" not in raw:
        raise RuntimeError(
            "raw dataset has no visible_with_occlusion. Run annotate_visibility_raw_self_occlusion.py first, "
            "or pass --allow-fov-only-sign explicitly."
        )

    q_per_point = int(raw["q_per_point"])
    raw_p = raw["p"].astype(np.float32)
    raw_q = raw["q"].astype(np.float32)
    raw_grid_points = raw["grid_points"].astype(np.float32)

    q0_p_indices = q0_data["p_indices"].astype(np.int64)
    q0_templates = q0_data["q0_templates"].astype(np.float32)
    q0_g = q0_data["q0_g"]
    num_q0 = q0_data["num_q0"].astype(np.int64)
    valid_mask = q0_data["valid_mask"].astype(np.bool_)
    valid_mask_with_occ = (
        q0_data["valid_mask_with_occlusion"].astype(np.bool_)
        if "valid_mask_with_occlusion" in q0_data
        else valid_mask
    )

    if args.joint_weights is not None:
        weights = np.asarray(args.joint_weights, dtype=np.float32)
        if weights.shape != (raw_q.shape[1],):
            raise RuntimeError(f"Expected {raw_q.shape[1]} joint weights, got {weights.shape}")
    else:
        weights = None

    visible_key = "visible_with_occlusion" if "visible_with_occlusion" in raw else "visible"
    visible = raw[visible_key].astype(np.bool_)
    sign_all = np.where(visible, 1.0, -1.0).astype(np.float32)

    out_p = []
    out_q = []
    out_p_index = []
    out_raw_index = []
    out_q0_file_row = []
    out_visible = []
    out_sign = []
    out_nearest_q0 = []
    out_distance_l2 = []
    out_distance_l1 = []
    out_cdf_l2 = []
    out_cdf_l1 = []
    out_g_fov = []
    out_best_sensor_fov = []
    out_best_margin_fov = []
    out_visible_fov = []

    skipped_no_q0 = 0
    skipped_bad_raw_range = 0

    for q0_row, p_idx in enumerate(q0_p_indices):
        if not valid_mask_with_occ[q0_row] or num_q0[q0_row] <= 0:
            skipped_no_q0 += 1
            continue

        start = int(p_idx) * q_per_point
        end = start + q_per_point
        if start < 0 or end > len(raw_q):
            skipped_bad_raw_range += 1
            continue

        q0_set = q0_templates[q0_row, :num_q0[q0_row]]
        q0_valid = np.isfinite(q0_g[q0_row, :num_q0[q0_row]])
        q0_set = q0_set[q0_valid]
        if len(q0_set) == 0:
            skipped_no_q0 += 1
            continue

        q_batch = raw_q[start:end]
        p_batch = raw_p[start:end]
        expected_p = raw_grid_points[p_idx]
        if not np.allclose(p_batch, expected_p.reshape(1, 3), atol=1e-5):
            raise RuntimeError(
                f"Raw sample layout mismatch at p_idx={p_idx}. "
                "Expected contiguous q_per_point samples for each grid point."
            )

        dist_l2_all = q_distance(q_batch, q0_set, "l2", weights)
        nearest_l2 = np.argmin(dist_l2_all, axis=1).astype(np.int16)
        dist_l2 = dist_l2_all[np.arange(len(q_batch)), nearest_l2].astype(np.float32)

        dist_l1_all = q_distance(q_batch, q0_set, "l1", weights)
        nearest_l1 = np.argmin(dist_l1_all, axis=1).astype(np.int16)
        dist_l1 = dist_l1_all[np.arange(len(q_batch)), nearest_l1].astype(np.float32)

        sign = sign_all[start:end]
        cdf_l2 = sign * dist_l2
        cdf_l1 = sign * dist_l1

        raw_indices = np.arange(start, end, dtype=np.int64)
        out_p.append(p_batch)
        out_q.append(q_batch)
        out_p_index.append(np.full((len(q_batch),), p_idx, dtype=np.int64))
        out_raw_index.append(raw_indices)
        out_q0_file_row.append(np.full((len(q_batch),), q0_row, dtype=np.int64))
        out_visible.append(visible[start:end])
        out_sign.append(sign)
        out_nearest_q0.append(nearest_l2 if args.metric == "l2" else nearest_l1)
        out_distance_l2.append(dist_l2)
        out_distance_l1.append(dist_l1)
        out_cdf_l2.append(cdf_l2.astype(np.float32))
        out_cdf_l1.append(cdf_l1.astype(np.float32))
        out_g_fov.append(raw["g"][start:end].astype(np.float32))
        out_best_sensor_fov.append(raw["best_sensor"][start:end].astype(np.int16))
        out_best_margin_fov.append(raw["best_margin"][start:end].astype(np.float32))
        out_visible_fov.append(raw["visible"][start:end].astype(np.bool_))

    if not out_q:
        raise RuntimeError("No training labels generated.")

    p_out = np.concatenate(out_p, axis=0).astype(np.float32)
    q_out = np.concatenate(out_q, axis=0).astype(np.float32)
    p_index_out = np.concatenate(out_p_index, axis=0)
    raw_index_out = np.concatenate(out_raw_index, axis=0)
    q0_file_row_out = np.concatenate(out_q0_file_row, axis=0)
    visible_out = np.concatenate(out_visible, axis=0).astype(np.bool_)
    sign_out = np.concatenate(out_sign, axis=0).astype(np.float32)
    nearest_q0_out = np.concatenate(out_nearest_q0, axis=0).astype(np.int16)
    distance_l2_out = np.concatenate(out_distance_l2, axis=0).astype(np.float32)
    distance_l1_out = np.concatenate(out_distance_l1, axis=0).astype(np.float32)
    cdf_l2_out = np.concatenate(out_cdf_l2, axis=0).astype(np.float32)
    cdf_l1_out = np.concatenate(out_cdf_l1, axis=0).astype(np.float32)

    primary_unsigned = distance_l2_out if args.metric == "l2" else distance_l1_out
    primary_cdf = cdf_l2_out if args.metric == "l2" else cdf_l1_out

    save_data = {
        "p": p_out,
        "q": q_out,
        "cdf": primary_cdf.astype(np.float32),
        "distance_unsigned": primary_unsigned.astype(np.float32),
        "sign": sign_out,
        "visible": visible_out,
        "nearest_q0_index": nearest_q0_out,
        "p_index": p_index_out,
        "raw_index": raw_index_out,
        "q0_file_row": q0_file_row_out,
        "g_fov": np.concatenate(out_g_fov, axis=0).astype(np.float32),
        "best_sensor_fov": np.concatenate(out_best_sensor_fov, axis=0).astype(np.int16),
        "best_margin_fov": np.concatenate(out_best_margin_fov, axis=0).astype(np.float32),
        "visible_fov": np.concatenate(out_visible_fov, axis=0).astype(np.bool_),
        "joint_names": raw["joint_names"],
        "sensor_frames": raw["sensor_frames"],
        "metric": np.asarray(args.metric),
        "raw_path": np.asarray(args.raw),
        "q0_path": np.asarray(args.q0),
        "visible_key": np.asarray(visible_key),
        "q_per_point": np.asarray(q_per_point, dtype=np.int32),
        "joint_weights": np.asarray([] if weights is None else weights, dtype=np.float32),
    }
    if args.also_save_other_metric or args.metric == "l2":
        save_data["distance_l2"] = distance_l2_out
        save_data["cdf_l2"] = cdf_l2_out
    if args.also_save_other_metric or args.metric == "l1":
        save_data["distance_l1"] = distance_l1_out
        save_data["cdf_l1"] = cdf_l1_out

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save_data)

    print("")
    print("=== Visibility CDF Label Summary ===")
    print(f"raw:                  {args.raw}")
    print(f"q0:                   {args.q0}")
    print(f"output:               {args.output}")
    print(f"metric:               {args.metric}")
    print(f"visible_key:          {visible_key}")
    print(f"q0 p entries:         {len(q0_p_indices)}")
    print(f"skipped no q0:        {skipped_no_q0}")
    print(f"skipped bad raw range:{skipped_bad_raw_range}")
    print(f"samples generated:    {len(q_out)}")
    print(f"visible samples:      {int(np.count_nonzero(visible_out))}")
    print(f"visible ratio:        {visible_out.mean():.6f}")
    print(f"cdf min/mean/max:     {primary_cdf.min(): .6f} / {primary_cdf.mean(): .6f} / {primary_cdf.max(): .6f}")
    print(f"unsigned min/mean/max:{primary_unsigned.min(): .6f} / {primary_unsigned.mean(): .6f} / {primary_unsigned.max(): .6f}")
    print(f"saved:                {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
