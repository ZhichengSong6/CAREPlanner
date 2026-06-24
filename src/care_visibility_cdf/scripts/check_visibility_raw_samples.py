#!/usr/bin/env python3

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
from urdf_parser_py.urdf import URDF

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    compute_visibility,
    find_chain_joints,
    fk_transform,
    get_joint_limits,
)


def q_row_to_map(joint_names, q_row):
    return {name: float(value) for name, value in zip(joint_names, q_row)}


def recompute_one(robot, base_frame, sensor_frames, joint_names, point, q_row, args):
    q_map = q_row_to_map(joint_names, q_row)
    sensor_transforms = [
        fk_transform(find_chain_joints(robot, base_frame, frame), q_map)
        for frame in sensor_frames
    ]
    sensor_margins, active_planes, best_sensor, best_margin, g, visible = compute_visibility(
        point.reshape(1, 3), sensor_transforms, args
    )
    return (
        sensor_margins[0],
        active_planes[0],
        int(best_sensor[0]),
        float(best_margin[0]),
        float(g[0]),
        bool(visible[0]),
    )


def check_required_keys(data):
    required = [
        "p",
        "q",
        "sensor_margins",
        "active_planes",
        "best_sensor",
        "best_margin",
        "g",
        "visible",
        "joint_names",
        "sensor_frames",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "z_min",
        "z_max",
        "delta",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"Missing required keys in dataset: {missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Check M3 raw visibility sample dataset consistency."
    )
    parser.add_argument("--input", required=True, help="Input .npz raw visibility dataset.")
    parser.add_argument("--urdf", default="", help="Optional URDF path for recomputation checks.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--num-recompute", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    check_required_keys(data)

    p = data["p"]
    q = data["q"]
    sensor_margins = data["sensor_margins"]
    active_planes = data["active_planes"]
    best_sensor = data["best_sensor"]
    best_margin = data["best_margin"]
    g = data["g"]
    visible = data["visible"]
    joint_names = [str(name) for name in data["joint_names"]]
    sensor_frames = [str(name) for name in data["sensor_frames"]]
    delta = float(data["delta"])

    print("")
    print("=== Dataset Shape ===")
    print(f"input:          {args.input}")
    print(f"p:              {p.shape}")
    print(f"q:              {q.shape}")
    print(f"sensor_margins: {sensor_margins.shape}")
    print(f"active_planes:  {active_planes.shape}")
    print(f"best_sensor:    {best_sensor.shape}")
    print(f"best_margin:    {best_margin.shape}")
    print(f"g:              {g.shape}")
    print(f"visible:        {visible.shape}")

    n = len(p)
    if q.shape != (n, len(joint_names)):
        raise RuntimeError(f"Unexpected q shape: {q.shape}")
    if sensor_margins.shape != (n, len(sensor_frames)):
        raise RuntimeError(f"Unexpected sensor_margins shape: {sensor_margins.shape}")
    if active_planes.shape != (n, len(sensor_frames)):
        raise RuntimeError(f"Unexpected active_planes shape: {active_planes.shape}")

    best_margin_expected = np.max(sensor_margins, axis=1)
    best_sensor_expected = np.argmax(sensor_margins, axis=1)
    g_expected = best_margin_expected - delta
    visible_expected = g_expected >= 0.0

    best_margin_err = np.max(np.abs(best_margin - best_margin_expected))
    best_sensor_mismatch = np.count_nonzero(best_sensor != best_sensor_expected)
    g_err = np.max(np.abs(g - g_expected))
    visible_mismatch = np.count_nonzero(visible != visible_expected)

    print("")
    print("=== Field Consistency ===")
    print(f"max |best_margin - max(sensor_margins)|: {best_margin_err:.9e}")
    print(f"best_sensor mismatches:                  {best_sensor_mismatch}")
    print(f"max |g - (best_margin - delta)|:         {g_err:.9e}")
    print(f"visible mismatches:                      {visible_mismatch}")

    print("")
    print("=== Distribution ===")
    print(f"num_samples:     {n}")
    print(f"visible_samples: {int(np.sum(visible))}")
    print(f"visible_ratio:   {np.mean(visible):.6f}")
    print(f"g min/mean/max:  {np.min(g): .6f} / {np.mean(g): .6f} / {np.max(g): .6f}")
    print(
        "best_margin min/mean/max: "
        f"{np.min(best_margin): .6f} / {np.mean(best_margin): .6f} / {np.max(best_margin): .6f}"
    )

    print("")
    print("best_sensor histogram:")
    counts = np.bincount(best_sensor.astype(np.int64), minlength=len(sensor_frames))
    for idx, frame in enumerate(sensor_frames):
        print(f"  {idx}: {frame:28s} {int(counts[idx]):8d}")

    if args.urdf:
        rng = np.random.default_rng(args.seed)
        sample_count = min(args.num_recompute, n)
        indices = rng.choice(n, size=sample_count, replace=False)
        robot = URDF.from_xml_file(args.urdf)
        get_joint_limits(robot, joint_names)

        recompute_args = SimpleNamespace(
            horizontal_fov_deg=float(data["horizontal_fov_deg"]),
            vertical_fov_deg=float(data["vertical_fov_deg"]),
            z_min=float(data["z_min"]),
            z_max=float(data["z_max"]),
            delta=delta,
        )

        max_margin_err = 0.0
        max_g_err = 0.0
        best_sensor_recompute_mismatch = 0
        visible_recompute_mismatch = 0

        for idx in indices:
            margins_r, _, best_sensor_r, _, g_r, visible_r = recompute_one(
                robot,
                args.base_frame,
                sensor_frames,
                joint_names,
                p[idx],
                q[idx],
                recompute_args,
            )
            max_margin_err = max(
                max_margin_err,
                float(np.max(np.abs(margins_r - sensor_margins[idx]))),
            )
            max_g_err = max(max_g_err, abs(g_r - float(g[idx])))

            if best_sensor_r != int(best_sensor[idx]):
                best_sensor_recompute_mismatch += 1
            if visible_r != bool(visible[idx]):
                visible_recompute_mismatch += 1

        print("")
        print("=== URDF Recompute Check ===")
        print(f"num_recomputed:                 {sample_count}")
        print(f"max sensor_margins error:       {max_margin_err:.9e}")
        print(f"max g error:                    {max_g_err:.9e}")
        print(f"best_sensor recompute mismatch: {best_sensor_recompute_mismatch}")
        print(f"visible recompute mismatch:     {visible_recompute_mismatch}")

    failed = (
        best_margin_err > args.tol
        or best_sensor_mismatch != 0
        or g_err > args.tol
        or visible_mismatch != 0
    )
    if failed:
        print("")
        print("[FAIL] Dataset consistency check failed.")
        sys.exit(1)

    print("")
    print("[OK] Dataset consistency check passed.")


if __name__ == "__main__":
    main()