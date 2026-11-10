#!/usr/bin/env python3
"""Read-only Grid25 Q0-bank audit after final merge.

Reads small counts from all canonical shard NPZ files and detailed arrays from
a few deterministic shards. Does not load the 36GB merged q array, alter the
dataset, prepare V4 pairs, or launch training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

EXPECTED_ROW_KEYS = (
    "x", "q", "k", "valid_fov", "g", "sensor_margins", "active_planes",
    "num_sensor_q0", "num_sensor_q0_before_downsample", "valid_point_fov",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".audit.", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def sampled_ids(total, count):
    if count <= 0:
        return set()
    return set(int(i) for i in np.linspace(0, total - 1, num=min(total, count), dtype=int))


def inspect_shard_values(archive, sid, ids, count, job):
    """Bounded-memory spot check, never a replacement for full merge verify."""
    with np.load(archive, allow_pickle=False) as z:
        q = z["q"]
        valid = z["valid_fov"]
        g = z["g"]
        x = z["x"]
        k = z["k"]
        require(q.shape == (len(ids), count, 7, 8), f"shard {sid}: q shape")
        require(valid.shape == (len(ids), count, 8), f"shard {sid}: valid shape")
        require(g.shape == (len(ids), count, 8), f"shard {sid}: g shape")
        require(np.array_equal(k, ids), f"shard {sid}: k identity")
        require(np.isfinite(x).all(), f"shard {sid}: nonfinite positions")
        finite = np.isfinite(q).all(axis=2)
        require(np.array_equal(finite, valid), f"shard {sid}: q finite mask != valid")
        trans = q.transpose(0, 1, 3, 2)
        require(np.isposinf(trans[~valid]).all(), f"shard {sid}: invalid q slots not +inf")
        good_q = trans[valid]
        lo = np.asarray(job["cfg"]["q_min"], dtype=float)
        hi = np.asarray(job["cfg"]["q_max"], dtype=float)
        if len(good_q):
            require(((good_q > lo) & (good_q < hi)).all(), f"shard {sid}: q joint limits")
            epsilon = float(job["reference"]["epsilon"])
            require(np.isfinite(g[valid]).all(), f"shard {sid}: nonfinite valid g")
            require((np.abs(g[valid]) <= epsilon + 1e-6).all(),
                    f"shard {sid}: q0 exceeds inherited epsilon")
        require(np.array_equal(valid.sum(axis=1), z["num_sensor_q0"]),
                f"shard {sid}: valid / sensor counts disagree")
        require(np.array_equal(valid.any(axis=(1, 2)), z["valid_point_fov"]),
                f"shard {sid}: point mask mismatch")
    return len(ids)


def audit(root, sample_shards=8):
    root = Path(root).resolve()
    formal = root / "formal"
    job_path = formal / "job.json"
    complete_path = formal / "dataset" / "complete.json"
    merged = formal / "dataset" / "visibility_grid25_fovonly.npz"
    require(job_path.is_file() and complete_path.is_file() and merged.is_file(),
            "Missing merged dataset or metadata; do not run training")
    job, complete = read_json(job_path), read_json(complete_path)
    cfg = job["cfg"]
    points = int(cfg["grid_points"])
    shard_size = int(cfg["points_per_shard"])
    total = (points + shard_size - 1) // shard_size
    require(points == 173225 and total == 2707, "Unexpected Grid25 size")
    require(complete["status"] == "COMPLETE" and complete["points"] == points,
            "Incomplete dataset metadata")
    require(complete.get("training_ready") is False,
            "Unexpected training-ready contract; inspect before using existing training")
    require(complete["format"] == cfg["format"], "Dataset format mismatch")
    require(complete["job_sha256"] == digest(job_path), "Dataset/job provenance mismatch")
    require(merged.name in complete.get("files", {}), "Merged archive digest not recorded")
    require(job["reference"]["has_occlusion_labels"] is False,
            "This audit is for FOV-only Q0 banks")
    with zipfile.ZipFile(merged) as z:
        names = set(z.namelist())
        missing = {k + ".npy" for k in EXPECTED_ROW_KEYS} - names
        require(not missing, f"Missing merged NPZ members: {sorted(missing)}")

    k = int(job["reference"]["q0_per_sensor"])
    all_q0 = np.zeros(8, dtype=np.int64)
    unresolved = np.zeros(8, dtype=np.int64)
    supported_x_sensor = np.zeros(8, dtype=np.int64)
    support_hist = np.zeros(9, dtype=np.int64)
    point_valid = 0
    sampled = sampled_ids(total, sample_shards)
    sampled_points = 0
    for sid in range(total):
        dest = formal / "shards" / f"{sid:05d}"
        marker = dest / "complete.json"
        archive = dest / "data.npz"
        require(marker.is_file() and archive.is_file(), f"Missing shard {sid:05d}")
        data = read_json(marker)
        indices = np.arange(sid * shard_size, min((sid + 1) * shard_size, points), dtype=np.int64)
        require(data["status"] == "COMPLETE" and data["shard_id"] == sid,
                f"Invalid shard metadata: {sid}")
        require(data["point_ids"] == indices.tolist(), f"Wrong shard point IDs: {sid}")
        require(data["job_sha256"] == complete["job_sha256"], f"Shard {sid} provenance mismatch")
        require(data["data_sha256"] == complete["shard_hashes"][str(sid)],
                f"Shard {sid} final provenance mismatch")
        with np.load(archive, allow_pickle=False) as z:
            counts = np.asarray(z["num_sensor_q0"])
        require(counts.shape == (len(indices), 8) and np.issubdtype(counts.dtype, np.integer),
                f"Shard {sid}: count schema")
        require(((counts >= 0) & (counts <= k)).all(), f"Shard {sid}: invalid q0 count")
        support = counts > 0
        per_sensor = counts.sum(axis=0, dtype=np.int64)
        require(per_sensor.tolist() == data["by_sensor"], f"Shard {sid}: count metadata mismatch")
        require(int(counts.sum()) == data["valid_q0"], f"Shard {sid}: total count mismatch")
        require(int(support.any(axis=1).sum()) == data["points_with_q0"],
                f"Shard {sid}: point coverage mismatch")
        require((~support).sum(axis=0).astype(int).tolist() == data["unresolved_sensor_pairs"],
                f"Shard {sid}: unresolved metadata mismatch")
        all_q0 += per_sensor
        supported_x_sensor += support.sum(axis=0)
        unresolved += (~support).sum(axis=0)
        point_valid += int(support.any(axis=1).sum())
        support_hist += np.bincount(support.sum(axis=1), minlength=9)
        if sid in sampled:
            sampled_points += inspect_shard_values(archive, sid, indices, k, job)
    require(int(support_hist.sum()) == points, "Incomplete point coverage")
    require(all_q0.tolist() == complete["valid_q0_by_sensor"], "Final sensor q0 totals mismatch")
    require(unresolved.tolist() == complete["unresolved_x_sensor_by_sensor"],
            "Final unresolved sensor-pair totals mismatch")
    require(point_valid == complete["points_with_q0"], "Final point support totals mismatch")

    result = {
        "status": "PASS_Q0_AUDIT_NOT_TRAINING_READY",
        "scope": "FOV-only q0 coverage; no LOS or exact signed-distance labels",
        "grid_points": points,
        "shards_checked": total,
        "merged_archive": str(merged),
        "complete_json_sha256": digest(complete_path),
        "sampled_shards_with_q_g_checks": len(sampled),
        "sampled_points_with_q_g_checks": sampled_points,
        "q0_per_sensor_cap": k,
        "points_with_any_q0": point_valid,
        "points_without_q0": points - point_valid,
        "point_coverage_fraction": point_valid / points,
        "supported_sensor_count_histogram_0_to_8": support_hist.tolist(),
        "valid_q0_by_sensor": all_q0.tolist(),
        "supported_x_by_sensor": supported_x_sensor.tolist(),
        "unresolved_x_sensor_by_sensor": unresolved.tolist(),
        "sensor_support_fraction": (supported_x_sensor / points).tolist(),
        "has_occlusion_labels": False,
        "training_ready": False,
        "next_step": "Rebuild Grid25-specific train/val/holdout splits and V4 pairs/caches; do not reuse old x_index.",
        "limitations": "No full NPZ hash rerun here; run existing fast_segments run.sh verify for that.",
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(os.environ.get(
        "GRID25_ROOT", "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")))
    p.add_argument("--sample-shards", type=int, default=8)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    require(0 <= args.sample_shards <= 2707, "Invalid --sample-shards")
    report = audit(args.root, args.sample_shards)
    out = args.out or args.root / "analysis" / "grid25_quality_v1" / "report.json"
    atomic_json(out, report)
    print(json.dumps(report, indent=2))
    print("[PASS] Grid25 Q0 audit; not yet training-ready")
    print("[report]", out)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise SystemExit("[STOP] " + str(exc))
