#!/usr/bin/env python3
"""Read-only coordinate-aware Grid25 holdout preflight (NOT a training split).

The old fresh holdout JSONL includes both x_index (old grid) and x coordinates.
Only use coordinates; old x_index cannot be applied to new 25mm grids.
Outputs proposed exclusion counts by radius. Never touches dataset or launches GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_fresh(path):
    coords, old_indices, groups = [], set(), {}
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require("x" in row and "x_index" in row,
                    f"Fresh holdout line {line_no} missing explicit x/x_index")
            xyz = np.asarray(row["x"], dtype=np.float64)
            require(xyz.shape == (3,) and np.isfinite(xyz).all(),
                    f"Fresh holdout line {line_no}: invalid x coordinate")
            coords.append(xyz)
            old_indices.add(int(row["x_index"]))
            group = str(row.get("group", "UNSPECIFIED"))
            groups[group] = groups.get(group, 0) + 1
    require(coords, "Empty old fresh holdout: no coordinate checks possible")
    # Coordinates are source identity; an old integer index is NEVER a new Grid25 index.
    uniq = np.unique(np.round(np.asarray(coords), decimals=7), axis=0)
    return uniq, len(coords), len(old_indices), groups


def workspace_coverage(fresh_xyz, origin, extent, tol=1e-6):
    """Describe old holdout positions relative to Grid25; do not discard any."""
    fresh = np.asarray(fresh_xyz, dtype=np.float64)
    low = np.asarray(origin, dtype=np.float64)
    high = np.asarray(extent, dtype=np.float64)
    require(fresh.ndim == 2 and fresh.shape[1] == 3 and
            len(fresh) > 0 and np.isfinite(fresh).all(),
            "Invalid old fresh holdout coordinate array")
    require(low.shape == (3,) and high.shape == (3,) and
            np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all(),
            "Invalid Grid25 workspace bounds")
    below = fresh < (low - tol)
    above = fresh > (high + tol)
    outside = (below | above).any(axis=1)
    # Exact distance to the closed workspace box, zero for inside points.
    gap = np.maximum(np.maximum(low - fresh, fresh - high), 0.0)
    box_dist_mm = np.linalg.norm(gap, axis=1) * 1000
    positions = np.flatnonzero(outside)
    return {
        "grid25_workspace_min_xyz_m": low.tolist(),
        "grid25_workspace_max_xyz_m": high.tolist(),
        "fresh_xyz_min_m": fresh.min(axis=0).tolist(),
        "fresh_xyz_max_m": fresh.max(axis=0).tolist(),
        "fresh_unique_positions_inside_workspace": int((~outside).sum()),
        "fresh_unique_positions_outside_workspace": int(outside.sum()),
        "outside_low_xyz_counts": below.sum(axis=0).astype(int).tolist(),
        "outside_high_xyz_counts": above.sum(axis=0).astype(int).tolist(),
        "max_distance_outside_workspace_mm": (
            float(box_dist_mm[outside].max()) if outside.any() else 0.0
        ),
        "outside_position_examples": [
            {
                "xyz_m": fresh[i].tolist(),
                "outside_distance_to_box_mm": float(box_dist_mm[i]),
            }
            for i in positions[:8]
        ],
    }


def spatial_coverage(x, grid_shape, origin, step, fresh_xyz, radii_mm):
    """Locate small neighborhoods on the canonical ij mesh; no NxM dense matrix."""
    x = np.asarray(x, dtype=np.float32)
    shape = tuple(int(v) for v in grid_shape)
    require(len(shape) == 3 and math.prod(shape) == len(x), "Bad Grid25 grid shape")
    require(x.shape == (math.prod(shape), 3), "Bad Grid25 x array")
    origin = np.asarray(origin, dtype=np.float64)
    require(origin.shape == (3,) and step > 0, "Invalid grid origin/step")
    # Verify flat-index semantics before relying on coordinate-to-index mapping.
    check = sorted({0, len(x) // 3, len(x) // 2, len(x) - 1})
    for idx in check:
        cell = np.array(np.unravel_index(idx, shape), dtype=np.float64)
        require(np.allclose(x[idx], origin + cell * step, atol=1e-6, rtol=0),
                "Grid25 storage order is not the expected ij coordinate lattice")
    report = {}
    nearest_dist = []
    masks = {int(r): np.zeros(len(x), dtype=bool) for r in radii_mm}
    require(all(r > 0 for r in masks), "Exclusion radii must be positive")
    for xyz in fresh_xyz:
        frac = (xyz - origin) / step
        at = np.rint(np.clip(frac, 0, np.asarray(shape) - 1)).astype(int)
        nearest_id = int(np.ravel_multi_index(tuple(at), shape))
        nearest_dist.append(float(np.linalg.norm(x[nearest_id].astype(float) - xyz)))
        for radius_mm, mask in masks.items():
            radius = radius_mm / 1000.0
            # Candidate cube intentionally padded beyond the radius.
            h = math.ceil(radius / step) + 1
            lo = np.maximum(at - h, 0)
            hi = np.minimum(at + h + 1, shape)
            ax = [np.arange(lo[d], hi[d], dtype=np.int32) for d in range(3)]
            xyz_grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
            ids = np.ravel_multi_index(tuple(xyz_grid.T), shape)
            delta = x[ids].astype(float) - xyz
            mask[ids[np.einsum("ij,ij->i", delta, delta) <= radius * radius + 1e-12]] = True
    for radius_mm, mask in masks.items():
        excluded = int(mask.sum())
        report[f"{radius_mm}mm"] = {
            "grid25_points_excluded": excluded,
            "fraction_of_grid": excluded / len(x),
            "remaining_candidates": int(len(x) - excluded),
            "fresh_unique_positions_within_radius_of_grid": int(
                np.count_nonzero(np.asarray(nearest_dist) <= radius_mm / 1000.0 + 1e-12)
            ),
        }
    return {
        "old_fresh_nearest_grid25_median_distance_mm": float(np.median(nearest_dist) * 1000),
        "old_fresh_nearest_grid25_max_distance_mm": float(np.max(nearest_dist) * 1000),
        "candidate_spatial_exclusions": report,
    }


def archive_metadata(path):
    required = {"x", "q", "k", "valid_fov", "sensor_chain_masks", "q_min", "q_max"}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        missing = sorted(k for k in required if (k + ".npy") not in names)
        require(not missing, f"Merged file missing required members: {missing}")
        sizes = {k.removesuffix(".npy"): int(z.getinfo(k).file_size)
                 for k in names if k.endswith(".npy")}
    return {
        "uncompressed_npz_members_bytes": int(sum(sizes.values())),
        "raw_q_member_bytes": int(sizes["q"]),
        "raw_valid_fov_bytes": int(sizes["valid_fov"]),
        "raw_x_member_bytes": int(sizes["x"]),
        "compressed_npz_file_bytes": int(path.stat().st_size),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(os.environ.get(
        "GRID25_ROOT", "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")))
    p.add_argument("--fresh-starts", type=Path, required=True)
    p.add_argument("--radii-mm", type=int, nargs="+", default=[25, 50, 75])
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    complete_file = root / "formal/dataset/complete.json"
    job_file = root / "formal/job.json"
    merged = root / "formal/dataset/visibility_grid25_fovonly.npz"
    for file in (complete_file, job_file, merged, a.fresh_starts):
        require(file.is_file(), f"Missing required file: {file}")
    complete = json.loads(complete_file.read_text())
    job = json.loads(job_file.read_text())
    require(complete["status"] == "COMPLETE" and
            complete["points"] == job["cfg"]["grid_points"] == 173225,
            "Not a completed Grid25 dataset")
    require(complete.get("training_ready") is False, "Unexpected training contract")
    require(complete.get("job_sha256") == sha(job_file), "Job provenance mismatch")
    require(job["cfg"]["spacing_mm"] == 25, "Expected 25mm spacing")
    require(not job["reference"]["has_occlusion_labels"], "Expected FOV-only")
    fresh, nrows, old_nidx, groups = load_fresh(a.fresh_starts)
    x = np.load(root / "formal/x.npy", mmap_mode="r", allow_pickle=False)
    origin = np.asarray([b[0] for b in job["cfg"]["workspace_mm"]], dtype=float) / 1000.0
    extent = np.asarray([b[1] for b in job["cfg"]["workspace_mm"]], dtype=float) / 1000.0
    workspace = workspace_coverage(fresh, origin, extent)
    # Old fresh targets may lie outside the new Grid25 box. Keep every old
    # coordinate for distance-based exclusions: border-adjacent outsiders can
    # still overlap training neighborhoods, while distant ones exclude nothing.
    spatial = spatial_coverage(x, job["cfg"]["grid_shape"], origin, 0.025,
                               fresh, a.radii_mm)
    archive = archive_metadata(merged)
    usage = shutil.disk_usage(root)
    result = {
        "status": "SPLIT_PREFLIGHT_ONLY_NOT_TRAINING_READY",
        "dataset_complete_sha256": sha(complete_file),
        "old_fresh_source": str(a.fresh_starts.resolve()),
        "old_fresh_sha256": sha(a.fresh_starts),
        "fresh_rows": nrows,
        "fresh_unique_positions": int(len(fresh)),
        "fresh_old_grid_unique_indices": old_nidx,
        "fresh_groups": groups,
        "old_x_indices_reused": False,
        **workspace,
        **spatial,
        "outside_workspace_policy": (
            "Old fresh xyz are retained for Euclidean-radius exclusion near "
            "Grid25 boundaries; out-of-domain points do not become Grid25 test "
            "samples or new grid indices."
        ),
        "coordinate_frame_assumption": (
            "Old fresh x and Grid25 positions refer to the same base_link "
            "frame; confirm provenance before defining official splits."
        ),
        **archive,
        "free_disk_bytes_at_grid25_root": int(usage.free),
        "memory_warning": "Do not np.load and materialize full compressed q; use mmap / streaming caches.",
        "next_step": "Review exclusion radius; build new Grid25 coordinate-based splits and compatible bank cache, then V4 label pilot.",
        "no_generated_split_or_training_labels": True,
    }
    out = a.out or root / "analysis/grid25_quality_v1/split_preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    require(out.resolve() not in {complete_file.resolve(), job_file.resolve(), merged.resolve()},
            "Refusing to overwrite production metadata")
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(result, indent=2))
    if workspace["fresh_unique_positions_outside_workspace"]:
        print("[NOTICE] Old fresh holdout contains "
              f'{workspace["fresh_unique_positions_outside_workspace"]} '
              "out-of-workspace unique positions. Kept all coordinates for "
              "distance-based boundary exclusions; no old index reused.")
    print("[DONE] Read-only Grid25 fresh-holdout preflight; NO training/split generated")
    print("[report]", out)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise SystemExit("[STOP] " + str(exc))
