#!/usr/bin/env python3

import argparse
import glob
import os
import re
import numpy as np


FIRST_DIM_KEYS = {
    "x",
    "q",
    "k",
    "valid_fov",
    "occluded",
    "valid_with_occlusion",
    "g",
    "sensor_margins",
    "active_planes",

    "grid_points",
    "p_indices",
    "sensor_q0_templates",
    "sensor_q0_g",
    "sensor_q0_sensor_margins",
    "sensor_q0_active_planes",
    "sensor_q0_occluded",
    "sensor_q0_valid_fov",
    "sensor_q0_valid_with_occlusion",
    "num_sensor_q0",
    "num_sensor_q0_occluded",
    "num_sensor_q0_visible",
    "num_sensor_q0_before_downsample",

    "valid_point_fov",
    "valid_point_with_occlusion",
}


def natural_key(path):
    name = os.path.basename(path)
    nums = re.findall(r"\d+", name)
    return [int(x) for x in nums] if nums else [0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sort-by-k", action="store_true", default=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob), key=natural_key)
    if not paths:
        raise RuntimeError(f"No shard files found: {args.input_glob}")

    print(f"[merge] found {len(paths)} shards")

    shards = []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        if "x" not in d or "q" not in d or "k" not in d:
            raise RuntimeError(f"Shard missing x/q/k: {path}")
        shards.append((path, d))
        print(f"  {os.path.basename(path):70s} P={d['x'].shape[0]}")

    ref = shards[0][1]
    all_keys = list(ref.files)

    merged = {}

    for key in all_keys:
        if key in FIRST_DIM_KEYS:
            vals = [d[key] for _, d in shards if key in d.files]
            merged[key] = np.concatenate(vals, axis=0)
        else:
            merged[key] = ref[key]

    if args.sort_by_k:
        order = np.argsort(merged["k"])
        for key in list(merged.keys()):
            arr = merged[key]
            if key in FIRST_DIM_KEYS and hasattr(arr, "shape") and len(arr.shape) > 0 and arr.shape[0] == len(order):
                merged[key] = arr[order]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **merged)

    print("")
    print("=== Merge Summary ===")
    print(f"num_shards:                 {len(paths)}")
    print(f"output:                     {args.output}")
    print(f"x:                          {merged['x'].shape}")
    print(f"q:                          {merged['q'].shape}")
    print(f"k:                          {merged['k'].shape}")
    print(f"valid_fov:                  {merged['valid_fov'].shape}")
    print(f"occluded:                   {merged['occluded'].shape}")
    print(f"valid_with_occlusion:       {merged['valid_with_occlusion'].shape}")
    print(f"valid_fov q0:               {int(np.count_nonzero(merged['valid_fov']))}")
    print(f"visible q0:                 {int(np.count_nonzero(merged['valid_with_occlusion']))}")
    print(f"occluded valid q0:          {int(np.count_nonzero(merged['valid_fov'] & merged['occluded']))}")
    print(f"finite q slots:             {int(np.isfinite(merged['q']).all(axis=2).sum())}")


if __name__ == "__main__":
    main()
