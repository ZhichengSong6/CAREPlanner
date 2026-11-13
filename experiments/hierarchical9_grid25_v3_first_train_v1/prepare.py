#!/usr/bin/env python3
"""Freeze one completed Grid25 production segment's verified V3 labels for GPU training.

Produces only (x,q,sensor signed distance,valid mask) per (x_index,q_slot).
Never invents gradient targets, nearest-q0 labels, or globally exact union labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNNER = REPO / "experiments/hierarchical9_grid25_v4_production_runner_v1"
sys.path.insert(0, str(RUNNER))
import runner as production

DEFAULT_ROOT = Path(
    "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
)
FORMAT = "care_grid25_segment0_v3_sensor_distance_train_v1"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as fd:
        for chunk in iter(lambda: fd.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create(root, out):
    root, out = root.resolve(), out.resolve()
    require(not out.exists(), "Frozen train dataset already exists; never overwrite")
    seg = production.segment_out(root, 0)
    # Verifies manifest, original label source hashes, per-shard sha,
    # exact split attribution and stage complete markers.
    production.verify_segment(seg, complete_required=True)
    _, manifest, bank, plan = production.open_segment(seg)
    xids = np.asarray(plan["x_index"], dtype=np.int64)
    split = np.asarray(plan["split"], dtype=np.uint8)
    require(len(xids) == 1024 and np.all(np.diff(xids) > 0),
            "Expected frozen first production segment of 1024 new-grid x positions")
    config = production.prod.Config(**manifest["config"])
    nq = int(config.v3_q_per_x)
    require(nq == 4, "Frozen V3 queries-per-x changed")
    n = len(xids) * nq
    xx = np.repeat(np.asarray(bank.x[xids], dtype=np.float32), nq, axis=0)
    qq = np.full((n, 7), np.nan, np.float32)
    values = np.zeros((n, 8), np.float32)
    valid = np.zeros((n, 8), dtype=bool)
    filled = np.zeros(n, dtype=bool)
    x_index = np.repeat(xids, nq)
    split_rows = np.repeat(split, nq)
    q_slot = np.tile(np.arange(nq, dtype=np.uint8), len(xids))
    by_shard = {}
    nshards = production.prod.shard_count(len(xids), config.x_shard_size)
    for shard in range(nshards):
        path, stamp = production.prod.stage_paths(seg, "v3_labels", shard)
        require(path.is_file() and stamp.is_file(), "Missing verified V3 shard")
        by_shard[path.name] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            ix = np.asarray(z["x_index"], dtype=np.int64)
            slot = np.asarray(z["q_slot"], dtype=np.int64)
            sensor = np.asarray(z["sensor"], dtype=np.int64)
            zs = np.asarray(z["split"], dtype=np.uint8)
            q = np.asarray(z["q"], dtype=np.float32)
            vals = np.asarray(z["new_value"], dtype=np.float32)
            is_valid = np.asarray(z["value_valid"], dtype=bool)
            reason = np.asarray(z["reason"])
            ref = np.asarray(z["reference_g_m"], dtype=np.float32)
        require(len(ix) > 0 and q.shape == (len(ix), 7)
                and all(len(arr) == len(ix)
                        for arr in (slot, sensor, zs, vals, is_valid, reason, ref)),
                "Malformed V3 shard rows")
        pos = np.searchsorted(xids, ix)
        require(np.all(pos < len(xids)) and np.array_equal(xids[pos], ix),
                "V3 data includes unexpected grid position")
        require(np.all((slot >= 0) & (slot < nq)) and
                np.all((sensor >= 0) & (sensor < 8)),
                "Invalid q slot or sensor ID")
        row = pos * nq + slot
        require(np.array_equal(split_rows[row], zs),
                "Train/val leakage in V3 labels")
        require(np.isfinite(q).all(), "Nonfinite query q")
        for i, rid in enumerate(row):
            rid = int(rid)
            if filled[rid]:
                require(np.array_equal(qq[rid], q[i]),
                        "Selected V3 q differs across sensors at same x/slot")
            else:
                qq[rid] = q[i]
                filled[rid] = True
            s = int(sensor[i])
            require(not valid[rid, s], "Duplicate already-valid (x, q, sensor) label")
            if is_valid[i]:
                require(reason[i] == "VALID" and np.isfinite(vals[i])
                        and np.isfinite(ref[i]), "Unexpected supposedly valid V3 label")
                # Expected FOV sign: inside is positive, outside negative.
                require(vals[i] * ref[i] >= -1e-5,
                        "V3 sign contradiction between FOV margin and distance")
                valid[rid, s] = True
                values[rid, s] = vals[i]
    require(filled.all(), "Some frozen x/q slots had no produced V3 queries")
    require(int(valid.sum()) == int(production.read(seg/"v3_summary.json")["valid_values"]),
            "Selected labels disagree with stage summary")
    require((valid.sum(axis=1) > 0).all(), "Queries with no finite V3 value")
    require(np.all(np.isfinite(qq)) and np.all(np.isfinite(xx)), "Nonfinite input")
    require(len(np.unique(xids[split == 0])) == 845 and
            len(np.unique(xids[split == 1])) == 179,
            "First segment train/val spatial composition changed")
    counts = {
        "x": len(xids),
        "queries": n,
        "train_x": int((split == 0).sum()),
        "val_x": int((split == 1).sum()),
        "train_queries": int((split_rows == 0).sum()),
        "val_queries": int((split_rows == 1).sum()),
        "valid_sensor_labels": int(valid.sum()),
        "train_valid_sensor_labels": int(valid[split_rows == 0].sum()),
        "val_valid_sensor_labels": int(valid[split_rows == 1].sum()),
        "complete_8_sensor_queries": int(valid.all(axis=1).sum()),
    }
    require(counts["train_valid_sensor_labels"] > 0 and
            counts["val_valid_sensor_labels"] > 0,
            "Empty valid training or validation labels")
    manifest_out = {
        "format": FORMAT,
        "status": "FROZEN_DISTANCE_ONLY",
        "segment_root": str(seg),
        "segment_manifest_sha256": sha(seg/"manifest.json"),
        "segment_complete_sha256": sha(seg/"segment_complete.json"),
        "production_source_v3_shards": by_shard,
        "preflight_x_plan_sha256": manifest["preflight_x_plan_sha256"],
        "counts": counts,
        "training_ready_full_grid25": False,
        "supervision": {
            "sensor_v3_value": "VALID only; radian signed distance to locally solved FOV boundary",
            "sensor_gradient": "DISABLED: no full FD-PASS mask",
            "v4_tube_value_and_normal": "NOT USED",
            "union_head": "ONLY optional max-of-8 V3 proxy when all 8 sensor labels valid; not exact union SDF",
            "gradient_eikonal_tension": "DISABLED in first-train v1",
            "train_val_split": "Frozen 75mm buffered Grid25 train/val; no random resplit",
        }
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".grid25_first_train.", dir=out.parent))
    try:
        with (tmp/"labels.npz").open("xb") as fd:
            np.savez_compressed(fd, x=xx, q=qq, x_index=x_index,
                                q_slot=q_slot, split=split_rows,
                                target=values, valid=valid)
            fd.flush()
            os.fsync(fd.fileno())
        manifest_out["labels_sha256"] = sha(tmp/"labels.npz")
        with (tmp/"manifest.json").open("x") as fd:
            json.dump(manifest_out, fd, indent=2, sort_keys=True)
            fd.write("\n")
            fd.flush()
            os.fsync(fd.fileno())
        os.rename(tmp, out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print("[V3 TRAINING DATA FROZEN]", out)
    print(json.dumps(counts, indent=2), flush=True)


def validate(out):
    out = out.resolve()
    m = json.loads((out/"manifest.json").read_text())
    require(m["format"] == FORMAT and m["status"] == "FROZEN_DISTANCE_ONLY",
            "Wrong training adapter format")
    require(sha(out/"labels.npz") == m["labels_sha256"],
            "Training data checksum changed")
    with np.load(out/"labels.npz", allow_pickle=False) as z:
        x, q, xi, split, target, valid = [z[k] for k in
                                         ("x","q","x_index","split","target","valid")]
        require(len(x) == m["counts"]["queries"] and
                x.shape == (len(x), 3) and q.shape == (len(x), 7) and
                target.shape == valid.shape == (len(x), 8),
                "Training snapshot shape changed")
        require(len(set(xi[split == 0]).intersection(xi[split == 1])) == 0,
                "Train/val x overlap")
        require(np.isfinite(target[valid]).all() and
                int(valid.sum()) == m["counts"]["valid_sensor_labels"],
                "Training label mask mismatch")
    print("[TRAIN SNAPSHOT VERIFIED]", out)
    print(json.dumps(m["counts"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "verify"))
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.root/"derived/grid25_v3_first_train_v1/dataset_segment_000"
    if args.mode == "prepare":
        create(args.root, out)
    else:
        validate(out)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, FileNotFoundError) as e:
        raise SystemExit("[STOP] "+str(e))
