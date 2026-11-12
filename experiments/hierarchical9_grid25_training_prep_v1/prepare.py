#!/usr/bin/env python3
"""Grid25 training preparation v1: coordinate-safe split + optional memmap Q0 cache.

This DOES NOT create V4 labels or start training. Original formal/* is read-only.
The old fresh cohort is EXTERNAL test data, never indexed with Grid25 IDs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GRID_QUALITY = REPO / "experiments/hierarchical9_grid25_quality_v1"
LABEL_CACHE = REPO / "experiments/hierarchical9_offline_relabel_v1"
sys.path.insert(0, str(GRID_QUALITY))

from split_preflight import load_fresh, workspace_coverage, archive_metadata

FORMAT = "careplanner_grid25_training_preparation_v1"
DEFAULT_ROOT = Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")
DEFAULT_FRESH = Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1/evaluation_fresh_holdout_v1/starts.jsonl")
REQUIRED_BANK = ("x", "q", "k", "valid_fov", "sensor_chain_masks", "q_min", "q_max")


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda: f.read(8 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_npz(path, **arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())


def root_metadata(root):
    root = Path(root).resolve()
    complete_path = root / "formal/dataset/complete.json"
    job_path = root / "formal/job.json"
    archive = root / "formal/dataset/visibility_grid25_fovonly.npz"
    grid_path = root / "formal/x.npy"
    for p in (complete_path, job_path, archive, grid_path):
        check(p.is_file(), "Missing verified Grid25 input: " + str(p))
    c = load_json(complete_path)
    j = load_json(job_path)
    cfg = j["cfg"]
    check(c.get("status") == "COMPLETE" and c.get("points") == 173225 and
          cfg.get("grid_shape") == [65, 65, 41] and cfg.get("grid_points") == 173225,
          "Expected complete 65x65x41 Grid25")
    check(c.get("training_ready") is False and
          j["reference"]["has_occlusion_labels"] is False,
          "Unexpected Grid25 label scope; cannot treat as V4 supervision")
    check(digest(job_path) == c["job_sha256"], "Grid25 job identity changed")
    check(digest(grid_path) == j["x_sha256"], "Grid25 coordinates changed")
    check(c["files"].get(archive.name), "Missing merged archive checksum")
    return root, c, j, archive, grid_path


def grid_xyz_and_coords(grid_path, cfg):
    x = np.load(grid_path, mmap_mode="r", allow_pickle=False)
    shape = tuple(cfg["grid_shape"])
    check(x.shape == (int(np.prod(shape)), 3) and x.dtype == np.float32,
          "Unexpected x.npy layout")
    origin = np.asarray([w[0] for w in cfg["workspace_mm"]], dtype=float) / 1000
    end = np.asarray([w[1] for w in cfg["workspace_mm"]], dtype=float) / 1000
    step = cfg["spacing_mm"] / 1000.0
    check(step == 0.025 and cfg.get("frame") == "base_link", "Expected base_link Grid25")
    for i in [0, 1, len(x) // 2, len(x) - 1]:
        ijk = np.unravel_index(i, shape)
        expected = origin + step * np.array(ijk)
        check(np.allclose(x[i], expected, atol=1e-6, rtol=0),
              "Grid coordinate order mismatch")
    return x, shape, origin, end, step


def fresh_guard(x, shape, origin, step, fresh, radius_mm):
    """Euclidean distance, preserving old fresh XYZ even outside the new box."""
    check(radius_mm > 0, "Fresh guard radius must be positive")
    guard = np.zeros(len(x), dtype=bool)
    dims = np.asarray(shape, dtype=int)
    radius_m = radius_mm / 1000.0
    h = int(np.ceil(radius_m / step)) + 1
    for target in fresh:
        center = np.rint(np.clip((target - origin) / step, 0, dims - 1)).astype(int)
        lo, hi = np.maximum(center - h, 0), np.minimum(center + h + 1, dims)
        axes = [np.arange(lo[i], hi[i], dtype=int) for i in range(3)]
        ijk = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        ids = np.ravel_multi_index(tuple(ijk.T), shape)
        delta = np.asarray(x[ids], dtype=float) - target
        guard[ids[np.einsum("ij,ij->i", delta, delta) <= radius_m**2 + 1e-12]] = True
    return guard


def grid_buffer(seed_mask, shape, radius_mm, step):
    """Euclidean morphological dilation on the actual ij lattice, no wraparound."""
    check(radius_mm >= 0, "Validation buffer radius must be nonnegative")
    seed = np.asarray(seed_mask, dtype=bool).reshape(shape)
    result = np.zeros(shape, dtype=bool)
    h = int(np.ceil(radius_mm / 1000.0 / step))
    for dx in range(-h, h + 1):
        for dy in range(-h, h + 1):
            for dz in range(-h, h + 1):
                if (dx*dx+dy*dy+dz*dz)*step**2 > (radius_mm/1000.0)**2 + 1e-12:
                    continue
                shifts = (dx, dy, dz)
                source = []
                target = []
                for dim, n in zip(shifts, shape):
                    source.append(slice(max(0, -dim), min(n, n - dim)))
                    target.append(slice(max(0, dim), min(n, n + dim)))
                result[tuple(target)] |= seed[tuple(source)]
    return result.reshape(-1)


def split_masks(x, shape, origin, step, fresh, *, fresh_mm=75,
                val_buffer_mm=75, block_cells=8, val_fraction=0.10, seed=20261112):
    """Deterministic spatial block validation + fresh/validation guard bands."""
    check(0 < val_fraction < 0.5 and block_cells > 0 and fresh_mm > 0 and
          val_buffer_mm >= 0, "Invalid split protocol")
    nf = len(x)
    fresh_excluded = fresh_guard(x, shape, origin, step, fresh, fresh_mm)
    eligible = ~fresh_excluded
    idx = np.arange(nf, dtype=np.int64)
    ijk = np.stack(np.unravel_index(idx, shape), axis=1)
    block = ijk // int(block_cells)
    block_shape = np.ceil(np.asarray(shape) / block_cells).astype(int)
    block_id = np.ravel_multi_index(tuple(block.T), tuple(block_shape))
    options = np.unique(block_id[eligible])
    rng = np.random.default_rng(seed)
    target = max(1, int(round(eligible.sum() * val_fraction)))
    val = np.zeros(nf, dtype=bool)
    for group in rng.permutation(options):
        val |= eligible & (block_id == group)
        if int(val.sum()) >= target:
            break
    check(val.any(), "Empty validation set")
    val_buffer = eligible & ~val & grid_buffer(val, shape, val_buffer_mm, step)
    train = eligible & ~val & ~val_buffer
    check(train.any(), "Validation blocks/buffer consumed all training positions")
    check(not np.any(train & val) and not np.any(train & fresh_excluded),
          "Train/val/fresh overlap")
    check(not np.any(train & grid_buffer(val, shape, val_buffer_mm, step)),
          "Validation buffer leakage")
    check(not np.any(train & fresh_guard(x, shape, origin, step, fresh, fresh_mm)),
          "External fresh holdout leakage")
    check(int(train.sum()+val.sum()+val_buffer.sum()+fresh_excluded.sum()) == nf,
          "Split partitions do not cover Grid25 exactly")
    return {
        "train": np.flatnonzero(train),
        "val": np.flatnonzero(val),
        "fresh_guard": np.flatnonzero(fresh_excluded),
        "val_guard": np.flatnonzero(val_buffer),
    }


def planned_source(root, fresh_path):
    root, complete, job, archive, grid = root_metadata(root)
    check(Path(fresh_path).is_file(), "Missing fresh starts")
    return root, complete, job, archive, grid


def create_plan(args):
    root, c, j, merged, grid = planned_source(args.root, args.fresh_starts)
    check(args.confirm_same_base_link,
          "Confirm old fresh XYZ and Grid25 both use base_link: pass --confirm-same-base-link")
    x, shape, origin, extent, step = grid_xyz_and_coords(grid, j["cfg"])
    fresh, rows, old_idx, groups = load_fresh(args.fresh_starts)
    workspace = workspace_coverage(fresh, origin, extent)
    masks = split_masks(x, shape, origin, step, fresh,
                        fresh_mm=args.fresh_mm, val_buffer_mm=args.val_buffer_mm,
                        block_cells=args.block_cells, val_fraction=args.val_fraction,
                        seed=args.seed)
    # If the earlier 75mm preflight exists, assert split retains its exact spatial
    # exclusion count rather than silently changing the established protocol.
    prior = root / "analysis/grid25_quality_v1/split_preflight.json"
    if prior.is_file():
        prev = load_json(prior)
        radius = f"{args.fresh_mm}mm"
        if (prev.get("old_fresh_sha256") == digest(args.fresh_starts) and
            prev.get("dataset_complete_sha256") == digest(root / "formal/dataset/complete.json") and
            radius in prev.get("candidate_spatial_exclusions", {})):
            expected = prev["candidate_spatial_exclusions"][radius]["grid25_points_excluded"]
            check(len(masks["fresh_guard"]) == expected,
                  "Fresh exclusion disagrees with previous spatial preflight")
    target = args.out or root / "derived/grid25_trainprep_v1"
    check(not target.exists() and not target.is_symlink(), "Preparation destination already exists; no overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".trainprep.build.", dir=target.parent))
    try:
        split_file = temp / "splits.npz"
        write_npz(split_file, train_x_index=masks["train"], val_x_index=masks["val"],
                  fresh_guard_x_index=masks["fresh_guard"],
                  val_guard_x_index=masks["val_guard"], old_fresh_xyz_m=fresh)
        info = {
            "format": FORMAT, "status": "SPLITS_READY_NOT_TRAINING_READY",
            "grid25_root": str(root),
            "grid_complete_sha256": digest(root / "formal/dataset/complete.json"),
            "grid_job_sha256": c["job_sha256"],
            "grid_xyz_sha256": j["x_sha256"],
            "source_npz": str(merged),
            "source_npz_expected_sha256": c["files"][merged.name],
            "source_npz_bytes": merged.stat().st_size,
            "source_npz_mtime_ns": merged.stat().st_mtime_ns,
            "fresh_starts": str(Path(args.fresh_starts).resolve()),
            "fresh_starts_sha256": digest(args.fresh_starts),
            "fresh_rows": rows, "fresh_unique_positions": len(fresh),
            "fresh_old_grid_unique_indices": old_idx,
            "fresh_groups": groups,
            "frame_assumption": "CONFIRMED_USER: both source coordinates refer to base_link",
            "old_x_indices_reused": False,
            "split_policy": dict(fresh_guard_mm=args.fresh_mm,
                                 validation_buffer_mm=args.val_buffer_mm,
                                 val_block_cells=args.block_cells,
                                 val_fraction_target=args.val_fraction, seed=args.seed),
            "split_counts": {name: int(len(ids)) for name, ids in masks.items()},
            "splits_sha256": digest(split_file),
            "workspace_relation": workspace,
            "external_test": "Old fresh holdout XYZ outside/inside Grid25; not Grid25 row indices",
            "has_occlusion_labels": False, "training_ready": False,
            "labels_ready": False,
            "notes": "Spatial splits and Q0 bank only. V4 pairs and supervised label caches must be generated separately."
        }
        write_json(temp / "manifest.json", info)
        os.rename(temp, target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print("[SPLITS PREPARED]", target)
    print(json.dumps({"split_counts": info["split_counts"],
                      "fresh_outside_positions": workspace["fresh_unique_positions_outside_workspace"],
                      "training_ready": False}, indent=2))
    return target


def verify_plan(out):
    out = Path(out).resolve()
    info = load_json(out / "manifest.json")
    check(info.get("format") == FORMAT and info.get("training_ready") is False and
          info.get("old_x_indices_reused") is False, "Not a valid Grid25 trainprep manifest")
    root, c, j, merged, grid = root_metadata(info["grid25_root"])
    check(info["grid_complete_sha256"] == digest(root / "formal/dataset/complete.json")
          and info["grid_job_sha256"] == c["job_sha256"] and
          info["grid_xyz_sha256"] == j["x_sha256"], "Grid25 source changed")
    check(info["source_npz_expected_sha256"] == c["files"][merged.name]
          and info["source_npz_bytes"] == merged.stat().st_size
          and info["source_npz_mtime_ns"] == merged.stat().st_mtime_ns,
          "Merged Grid25 archive changed")
    check(info["fresh_starts_sha256"] == digest(info["fresh_starts"]),
          "Old fresh holdout file changed")
    check(info["splits_sha256"] == digest(out / "splits.npz"), "Split snapshot changed")
    with np.load(out / "splits.npz", allow_pickle=False) as z:
        sets = [np.asarray(z[k]) for k in ("train_x_index", "val_x_index",
               "fresh_guard_x_index", "val_guard_x_index")]
        check(sum(len(ids) for ids in sets) == 173225, "Split coverage changed")
        occupied = np.concatenate(sets)
        check(occupied.dtype.kind in "iu" and np.array_equal(np.sort(occupied), np.arange(173225)),
              "Split partitions overlap or have missing Grid25 rows")
        sizes = [len(ids) for ids in sets]
        names = ("train", "val", "fresh_guard", "val_guard")
        check(all(n == len(ids) for n, ids in zip(
            (info["split_counts"][v] for v in names), sets)), "Split count mismatch")
    print("[SPLITS VERIFIED] counts=", dict(zip(names, sizes)))
    return info, root, merged


def build_cache(out):
    out = Path(out).resolve()
    info, root, source = verify_plan(out)
    check(not (out / "bank_cache").exists(), "Bank cache already exists; no overwrite")
    sys.path.insert(0, str(LABEL_CACHE))
    import cache_io
    # 20GB+ streaming operation, explicitly separate from lightweight split plan.
    with zipfile.ZipFile(source) as archive:
        keys = set(archive.namelist())
        for k in REQUIRED_BANK:
            check(k + ".npy" in keys, "Source missing bank key: " + k)
        members = cache_io.REQUIRED + [x for x in cache_io.OPTIONAL if x + ".npy" in keys]
        uncompressed = sum(archive.getinfo(k+".npy").file_size for k in members)
    free = shutil.disk_usage(out).free
    check(free > uncompressed + (5 << 30),
          "Insufficient disk for mmap bank cache and 5GiB headroom")
    # Source's full checksum has previously been verified after merging; check
    # again before generating a derived cache so stale/corrupt input is rejected.
    check(digest(source) == info["source_npz_expected_sha256"],
          "Merged source hash differs from final published dataset")
    result = cache_io.import_bank(source, out / "bank_cache")
    check(result["source_sha256"] == info["source_npz_expected_sha256"],
          "Cache source hash mismatch")
    bank = cache_io.BankCache(out / "bank_cache")
    check(bank.x.shape == (173225, 3) and bank.q.shape[0] == 173225 and
          bank.q.shape[2:] == (7, 8) and bank.valid.shape == (173225, bank.q.shape[1], 8),
          "Wrong bank schema after extraction")
    with np.load(out / "splits.npz", allow_pickle=False) as z:
        sample = np.concatenate((z["train_x_index"][:3], z["val_x_index"][:3]))
    for idx in sample:
        check(np.array_equal(np.asarray(bank.x[int(idx)]),
                             np.load(root / "formal/x.npy", mmap_mode="r")[int(idx)]),
              "Bank x sample differs from frozen canonical grid")
    print("[CACHE READY] Memory-mapped Q0 bank, not V4 labels:", out / "bank_cache")
    print("[CACHE SOURCE SHA256]", result["source_sha256"])
    print("[TRAINING READY] False; V4 labels/caches and final adapter still required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "cache", "verify"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fresh-starts", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--confirm-same-base-link", action="store_true")
    parser.add_argument("--fresh-mm", type=int, default=75)
    parser.add_argument("--val-buffer-mm", type=int, default=75)
    parser.add_argument("--block-cells", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.mode == "plan":
        create_plan(args)
    elif args.mode == "cache":
        target = args.out or args.root / "derived/grid25_trainprep_v1"
        build_cache(target)
    else:
        target = args.out or args.root / "derived/grid25_trainprep_v1"
        verify_plan(target)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, FileExistsError,
            zipfile.BadZipFile) as exc:
        raise SystemExit("[STOP] " + str(exc))
