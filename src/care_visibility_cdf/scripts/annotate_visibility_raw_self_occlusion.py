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

from generate_visibility_raw_samples import (  # noqa: E402
    compute_sensor_occlusion_for_sample,
    compute_visibility_with_occlusion,
    prepare_occlusion_context,
)
from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
)


def names_from_npz(data, key, default):
    if key in data:
        return [str(name) for name in data[key]]
    return list(default)


def copy_npz_fields(data):
    return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser(
        description="Annotate an existing M3 raw visibility dataset with self-occlusion labels."
    )
    parser.add_argument("--input", required=True, help="Input raw visibility .npz.")
    parser.add_argument("--output", required=True, help="Output raw visibility .npz with occlusion fields.")
    parser.add_argument("--occlusion-urdf", required=True, help="Simplified collision URDF.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--joint-names", nargs="*", default=None)
    parser.add_argument("--sensor-frames", nargs="*", default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--ray-start-offset", type=float, default=0.03)
    parser.add_argument("--point-end-offset", type=float, default=0.005)
    parser.add_argument("--min-hit-distance", type=float, default=0.0)
    parser.add_argument("--min-ray-length", type=float, default=1e-4)
    parser.add_argument("--ignore-start-inside", action="store_true", default=True)
    parser.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    parser.add_argument("--ignore-links", nargs="*", default=[])
    args = parser.parse_args()

    raw = np.load(args.input, allow_pickle=True)
    p = raw["p"].astype(np.float64)
    q = raw["q"].astype(np.float64)
    sensor_margins = raw["sensor_margins"].astype(np.float64)
    delta = float(raw["delta"])
    joint_names = args.joint_names or names_from_npz(raw, "joint_names", DEFAULT_JOINT_NAMES)
    sensor_frames = args.sensor_frames or names_from_npz(raw, "sensor_frames", DEFAULT_SENSOR_FRAMES)

    args.joint_names = joint_names
    args.sensor_frames = sensor_frames
    occlusion_context = prepare_occlusion_context(args, sensor_frames)

    n = len(p)
    num_sensors = len(sensor_frames)
    sensor_occluded = np.zeros((n, num_sensors), dtype=np.bool_)

    print("")
    print("=== Raw Self-Occlusion Annotation Config ===")
    print(f"input:           {args.input}")
    print(f"output:          {args.output}")
    print(f"occlusion_urdf:  {args.occlusion_urdf}")
    print(f"samples:         {n}")
    print(f"sensors:         {num_sensors}")
    print(f"chunk_size:      {args.chunk_size}")
    print(f"delta:           {delta}")

    t0 = time.time()
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        for idx in range(start, end):
            sensor_occluded[idx] = compute_sensor_occlusion_for_sample(
                p[idx],
                q[idx],
                joint_names,
                occlusion_context,
                args,
            )
        if args.progress_every > 0 and end % args.progress_every == 0:
            elapsed = time.time() - t0
            print(f"[progress] {end}/{n} elapsed {elapsed:.1f}s")

    (
        sensor_fov_visible,
        sensor_visible_with_occlusion,
        best_visible_sensor,
        best_visible_margin,
        visible_with_occlusion,
    ) = compute_visibility_with_occlusion(sensor_margins, sensor_occluded, delta)

    data = copy_npz_fields(raw)
    # Explicit FOV aliases. Keep legacy fields unchanged for backward compatibility.
    data["best_sensor_fov"] = raw["best_sensor"].astype(np.int16)
    data["best_margin_fov"] = raw["best_margin"].astype(np.float32)
    data["g_fov"] = raw["g"].astype(np.float32)
    data["visible_fov"] = raw["visible"].astype(np.bool_)

    data["sensor_occluded"] = sensor_occluded.astype(np.bool_)
    data["sensor_fov_visible"] = sensor_fov_visible.astype(np.bool_)
    data["sensor_visible_with_occlusion"] = sensor_visible_with_occlusion.astype(np.bool_)
    data["best_visible_sensor"] = best_visible_sensor.astype(np.int16)
    data["best_visible_margin"] = best_visible_margin.astype(np.float32)
    data["visible_with_occlusion"] = visible_with_occlusion.astype(np.bool_)
    data["occlusion_urdf"] = np.asarray(args.occlusion_urdf)
    data["has_occlusion_labels"] = np.asarray(True, dtype=np.bool_)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **data)

    elapsed = time.time() - t0
    fov_visible = raw["visible"].astype(np.bool_)
    changed = np.count_nonzero(fov_visible != visible_with_occlusion)
    print("")
    print("=== Raw Self-Occlusion Annotation Summary ===")
    print(f"samples:                  {n}")
    print(f"visible_fov:              {int(np.count_nonzero(fov_visible))}")
    print(f"visible_with_occlusion:   {int(np.count_nonzero(visible_with_occlusion))}")
    print(f"visibility changed:       {changed}")
    print(f"changed ratio:            {changed / max(n, 1):.6f}")
    print(f"sensor occluded entries:  {int(np.count_nonzero(sensor_occluded))}")
    print("sensor occluded ratio per sensor:")
    for idx, frame in enumerate(sensor_frames):
        print(f"  {idx}: {frame:28s} {sensor_occluded[:, idx].mean():.6f}")
    print(f"elapsed_sec:              {elapsed:.2f}")
    print(f"saved:                    {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
