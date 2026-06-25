#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Merge M4 visibility Q0 shard .npz files.")
    parser.add_argument("--pattern", required=True, help="Glob pattern for shard .npz files.")
    parser.add_argument("--output", required=True, help="Merged output .npz path.")
    parser.add_argument("--expected-p-count", type=int, default=0,
                        help="Optional expected total number of merged p rows, e.g. 21600.")
    parser.add_argument("--require-contiguous", action="store_true",
                        help="Require merged p_indices to be exactly 0..N-1 after sorting.")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise RuntimeError(f"No shard files matched pattern: {args.pattern}")

    print("=== Merge Visibility Q0 Shards ===")
    for p in paths:
        print("shard:", p)

    shards = [np.load(p, allow_pickle=True) for p in paths]
    p_lists = []
    n_lists = []
    for path, d in zip(paths, shards):
        if "p_indices" not in d:
            raise RuntimeError(f"Shard missing p_indices: {path}")
        pidx = d["p_indices"].astype(np.int64)
        p_lists.append(pidx)
        n_lists.append(len(pidx))
        print(f"  rows={len(pidx)} p_min={pidx.min() if len(pidx) else 'NA'} p_max={pidx.max() if len(pidx) else 'NA'}")

    all_p = np.concatenate(p_lists, axis=0)
    unique_p, counts = np.unique(all_p, return_counts=True)
    if len(unique_p) != len(all_p):
        dup = unique_p[counts > 1][:20]
        raise RuntimeError(f"Duplicate p_indices found. Examples: {dup}")

    if args.expected_p_count > 0 and len(all_p) != args.expected_p_count:
        raise RuntimeError(f"Merged rows {len(all_p)} != expected {args.expected_p_count}")

    order = np.argsort(all_p)
    sorted_p = all_p[order]
    if args.require_contiguous:
        expected = np.arange(len(sorted_p), dtype=np.int64)
        if len(sorted_p) == 0 or not np.array_equal(sorted_p, expected):
            missing = np.setdiff1d(expected, sorted_p)[:20]
            extra = np.setdiff1d(sorted_p, expected)[:20]
            raise RuntimeError(f"p_indices are not contiguous 0..N-1. missing={missing}, extra={extra}")

    out = {}
    first = shards[0]
    keys = list(first.files)
    for key in keys:
        vals = [d[key] for d in shards]
        # Merge row-wise arrays whose first dimension is exactly this shard's row count.
        rowwise = True
        for v, n in zip(vals, n_lists):
            if not isinstance(v, np.ndarray) or v.ndim < 1 or v.shape[0] != n:
                rowwise = False
                break
        if rowwise:
            cat = np.concatenate(vals, axis=0)
            out[key] = cat[order]
        else:
            # For scalar metadata, keep the first shard value.
            out[key] = vals[0]

    out["p_indices"] = sorted_p
    out["merged_from_shards"] = np.asarray(paths, dtype=object)
    out["num_merged_shards"] = np.asarray(len(paths), dtype=np.int32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **out)

    print("\n=== Merge Summary ===")
    print("saved:", args.output)
    print("merged_p:", len(sorted_p))
    print("unique_p:", len(np.unique(sorted_p)))
    print("p_min/max:", sorted_p.min(), sorted_p.max())
    for k in ["grid_points", "q0_templates", "sensor_q0_templates", "q0_g", "sensor_q0_g", "num_q0", "num_sensor_q0"]:
        if k in out:
            print(k, out[k].shape)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
