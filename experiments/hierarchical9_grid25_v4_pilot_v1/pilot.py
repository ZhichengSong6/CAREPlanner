#!/usr/bin/env python3
"""Bounded Grid25 V4 pilot reusing the FROZEN full V4 global/V3/V4 workers.

Never imports old x_index as Grid25 IDs. Uses only verified new Grid25 spatial
splits and read-only mmap q0 bank. Does not train or publish full training labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREP = ROOT / "experiments/hierarchical9_grid25_training_prep_v1"
FULL = ROOT / "experiments/hierarchical9_full_dataset_v4_v1"
sys.path.insert(0, str(PREP))
from prepare import verify_plan  # Grid25 native provenance/split verification
sys.path.insert(0, str(FULL))
import prod_common as prod

FORMAT = "careplanner_grid25_v4_pilot_v1"
STAGES = ("global_pool", "v3_labels", "v4_tubes")
REQUIRED_SUMMARIES = ("global_pool_summary.json", "v3_summary.json", "v4_summary.json")
DEFAULT_ROOT = Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")
DEFAULT_URDF = ROOT / "src/arm_description/urdf/Arm.urdf"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_safe(obj):
    return prod.cache_io.clean_json(obj)


def save_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(json_safe(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def code_fingerprint():
    # Hash the exact upstream algorithms and their imported dependencies,
    # not merely the name of the full_dataset_v4_v1 directory.
    local = [HERE / "pilot.py", HERE / "run.sh", HERE / "worker.sbatch"]
    upstream = prod.code_identity()
    return {"pilot": {str(p.relative_to(ROOT)): sha(p) for p in local},
            "upstream_full_v4": upstream}


def select_pilot_positions(bank, train_pool, val_pool, *, ntrain, nval, seed):
    """Pick representatives with rich and reduced sensor support, deterministically."""
    require(ntrain >= 1 and nval >= 1 and ntrain <= len(train_pool) and
            nval <= len(val_pool), "Pilot train/val selection sizes invalid")
    output = []
    for split, pool, count in ((0, train_pool, ntrain), (1, val_pool, nval)):
        pool = np.asarray(pool, dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([seed, split]))
        candidates = rng.choice(pool, size=min(len(pool), max(64, 12 * count)), replace=False)
        counts = np.stack([np.asarray(bank.valid[int(i)], bool).sum(axis=0)
                           for i in candidates], axis=0).astype(np.uint16)
        support = counts > 0
        eligible = np.flatnonzero(support.any(axis=1))
        require(len(eligible) >= count, f"Split {split}: too few supported x in representative pool")
        nheads = support.sum(axis=1)
        chosen = []
        # First try a full 8-head x so V3/V4 exercises all sensor heads.
        full = [i for i in eligible if nheads[i] == 8]
        require(full, f"Split {split}: no all-8-supported x in representative pool")
        chosen.append(int(full[0]))
        # Then a partial-support x to exercise UNRESOLVED handling.
        if count >= 2:
            partial = [i for i in eligible if 0 < nheads[i] < 8 and i not in chosen]
            if partial:
                chosen.append(int(partial[0]))
        # All remaining representative x selected without duplicates.
        for i in eligible:
            if len(chosen) >= count:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
        require(len(chosen) == count, "Pilot selection did not reach requested size")
        for i in chosen:
            output.append((int(candidates[i]), split, counts[i]))
    return output


def prepare(args):
    trainprep = args.trainprep.resolve()
    info, root, merged = verify_plan(trainprep)
    require(info.get("status") == "SPLITS_READY_NOT_TRAINING_READY",
            "Grid25 split plan is not the expected immutable source")
    bank_path = trainprep / "bank_cache"
    require((bank_path / "bank_manifest.json").is_file(),
            "Missing complete mmap q0 bank: run Grid25 training_prep cache stage first")
    bank = prod.cache_io.BankCache(bank_path)
    require(bank.manifest["source_sha256"] == info["source_npz_expected_sha256"] and
            bank.manifest["source_stat"]["path"] == str(merged),
            "Q0 cache is not the verified Grid25 dataset")
    require(bank.x.shape == (173225, 3) and bank.q.shape == (173225, 500, 7, 8)
            and bank.valid.shape == (173225, 500, 8), "Unexpected Q0 bank shape")
    with np.load(trainprep / "splits.npz", allow_pickle=False) as z:
        tr = np.asarray(z["train_x_index"], dtype=np.int64)
        va = np.asarray(z["val_x_index"], dtype=np.int64)
        guard = np.concatenate((z["fresh_guard_x_index"], z["val_guard_x_index"]))
    require(len(np.intersect1d(tr, va)) == 0 and
            len(np.intersect1d(np.concatenate((tr, va)), guard)) == 0,
            "Grid25 spatial split leakage")

    selections = select_pilot_positions(
        bank, tr, va, ntrain=args.train_x, nval=args.val_x, seed=args.seed)
    xi = np.asarray([x for x, _, _ in selections], np.int64)
    split = np.asarray([s for _, s, _ in selections], np.uint8)
    counts = np.asarray([vc for _, _, vc in selections], np.uint16)
    require(len(np.unique(xi)) == len(xi), "Pilot x overlap")
    require(set(xi[split == 0]).issubset(set(tr)) and
            set(xi[split == 1]).issubset(set(va)), "Pilot split attribution mismatch")
    require((counts > 0).any(axis=1).all(), "Pilot selected unsupported points")
    # Frozen original production policy: 128 global q/x, 4 selected V3 q/x,
    # 32 anchors/(x,s), 25mm new-grid IDs as independent deterministic seeds.
    cfg = prod.Config(candidate_q_per_x=128, v3_q_per_x=4,
                      anchors_per_sensor=32, x_shard_size=1, seed=260927)
    cfg.validate()
    out = args.out.resolve()
    require(not out.exists() and not out.is_symlink(),
            "Pilot output already exists; no implicit overwrite")
    for protected in (root / "formal", bank_path):
        require(not (out == protected or out.is_relative_to(protected) or
                     protected.is_relative_to(out)), "Output overlaps source")
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".v4pilot.", dir=out.parent))
    try:
        prod.write_npz(temp / "x_plan.npz", dict(
            x_index=xi, split=split, valid_count=counts, support=counts > 0))
        # Do not select old fresh x_indices: each row is indexed in the NEW Grid25 bank.
        m = {
            "format": FORMAT,
            "status": "PILOT_PREPARED_NOT_TRAINING_READY",
            "trainprep": str(trainprep),
            "grid25_root": str(root),
            "bank_cache": str(bank_path),
            "bank_manifest_sha256": sha(bank_path / "bank_manifest.json"),
            "grid25_complete_sha256": info["grid_complete_sha256"],
            "grid25_npz_sha256": info["source_npz_expected_sha256"],
            "grid25_splits_sha256": info["splits_sha256"],
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": sha(args.urdf),
            "config": asdict(cfg),
            "solver": asdict(prod.core.SolverConfig()),
            "verify": asdict(prod.core.VerifyConfig()),
            "x_plan_sha256": sha(temp / "x_plan.npz"),
            "code_fingerprint": code_fingerprint(),
            "counts": {
                "selected_x": len(xi),
                "train_x": int((split == 0).sum()),
                "val_x": int((split == 1).sum()),
                "supported_x_sensor_pairs": int((counts > 0).sum()),
                "v3_sensor_query_tasks": int((counts > 0).sum() * cfg.v3_q_per_x),
                "v4_anchor_budget": int(np.minimum(counts, cfg.anchors_per_sensor).sum()),
            },
            "scope": "Grid25 2+2 representative V4 PILOT; FOV-only geometry; never V4 full-scale training labels",
            "old_grid_x_indices_reused": False,
            "has_los_labels": False,
            "training_ready": False,
        }
        save_json_exclusive(temp / "manifest.json", m)
        os.rename(temp, out)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print("[PILOT PREPARED]", out)
    print(json.dumps({"counts": m["counts"], "selected_new_grid_x_index": xi.tolist(),
                      "selected_support_heads": (counts > 0).sum(axis=1).tolist(),
                      "training_ready": False}, indent=2), flush=True)


def open_pilot(out):
    out = Path(out).resolve()
    m = read_json(out / "manifest.json")
    require(m.get("format") == FORMAT and m.get("status") ==
            "PILOT_PREPARED_NOT_TRAINING_READY" and
            m.get("training_ready") is False and
            m.get("old_grid_x_indices_reused") is False,
            "Invalid Grid25 pilot identity")
    require(code_fingerprint() == m["code_fingerprint"],
            "Label source/solver code identity changed; do not mix results")
    info, root, merged = verify_plan(Path(m["trainprep"]))
    require(info["grid_complete_sha256"] == m["grid25_complete_sha256"] and
            info["source_npz_expected_sha256"] == m["grid25_npz_sha256"] and
            info["splits_sha256"] == m["grid25_splits_sha256"], "Grid25 training split/source changed")
    require(sha(Path(m["bank_cache"]) / "bank_manifest.json") ==
            m["bank_manifest_sha256"], "Q0 bank manifest changed")
    bank = prod.cache_io.BankCache(m["bank_cache"])
    require(bank.manifest["source_sha256"] == m["grid25_npz_sha256"],
            "Wrong Grid25 source in Q0 bank")
    require(sha(m["urdf"]) == m["urdf_sha256"], "URDF changed since pilot preparation")
    require(sha(out / "x_plan.npz") == m["x_plan_sha256"], "Pilot spatial plan changed")
    with np.load(out / "x_plan.npz", allow_pickle=False) as z:
        plan = {k: z[k] for k in z.files}
    require(len(plan["x_index"]) == m["counts"]["selected_x"] and
            len(np.unique(plan["x_index"])) == len(plan["x_index"]),
            "Pilot positions duplicated")
    with np.load(Path(m["trainprep"]) / "splits.npz", allow_pickle=False) as z:
        tr, va = z["train_x_index"], z["val_x_index"]
    require(np.isin(plan["x_index"][plan["split"] == 0], tr).all() and
            np.isin(plan["x_index"][plan["split"] == 1], va).all() and
            np.isin(plan["split"], [0, 1]).all(), "Pilot train/val leakage")
    return out, m, bank, plan


def upstream_module(stage):
    # Import existing V4 production workers WITHOUT changing their algorithms.
    # Only substitute the read-only data open_run (new split and mmap bank).
    import global_pool
    import v3_labels
    import v4_tubes
    for module in (global_pool, v3_labels, v4_tubes):
        module.open_run = open_pilot
    # Upstream global_pool.merge uses old_io from its __main__ block.
    global_pool.old_io = prod.cache_io
    return {"global_pool": global_pool, "v3_labels": v3_labels,
            "v4_tubes": v4_tubes}[stage]


def stage_summary(out, stage):
    module = upstream_module(stage)
    a = SimpleNamespace(out=Path(out))
    module.merge(a)  # upstream immutable full V4 summary logic
    path = Path(out) / {"global_pool": "global_pool_summary.json",
                        "v3_labels": "v3_summary.json",
                        "v4_tubes": "v4_summary.json"}[stage]
    return read_json(path)


def geometry_gate(out):
    out, m, bank, plan = open_pilot(out)
    gate_path = out / "geometry_preflight.json"
    if gate_path.exists():
        record = read_json(gate_path)
        require(record.get("status") == "PASS" and
                record.get("pilot_manifest_sha256") == sha(out / "manifest.json") and
                record.get("urdf_sha256") == m["urdf_sha256"],
                "Cached geometry gate does not match pilot source")
        return
    oracle = prod.RepoOracle(ROOT, m["urdf"], "cpu",
                             joint_names=bank.joints, sensor_frames=bank.sensors)
    report = oracle.verify(bank, plan["x_index"][:2], np.zeros((1, 7), np.float32))
    require(report["status"] == "PASS", "Original geometry/Jacobian/old target parity failed")
    save_json_exclusive(gate_path, {
        "status": "PASS",
        "pilot_manifest_sha256": sha(out / "manifest.json"),
        "urdf_sha256": m["urdf_sha256"],
        "upstream_parity": report,
    })
    print("[GEOMETRY GATE PASS]", json.dumps(json_safe(report), indent=2), flush=True)


def one_worker(stage, out, urdf, rank, world):
    module = upstream_module(stage)
    a = SimpleNamespace(out=Path(out), urdf=Path(urdf), device="cpu",
                        rank=rank, world=world)
    module.worker(a)
    return rank


def run(args):
    out, m, _, plan = open_pilot(args.out)
    require(1 <= args.workers <= 8, "Pilot CPU workers must be 1..8")
    geometry_gate(out)
    for stage in STAGES:
        done = stage_summary_path(out, stage)
        if done.is_file():
            print(f"[RESUME] {stage} summary already exists; verifying shards", flush=True)
            verify_stage_data(out, stage)
            continue
        print(f"[STAGE] {stage}; selected x={len(plan['x_index'])}; workers={args.workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(one_worker, stage, str(out), m["urdf"],
                                 rank, args.workers) for rank in range(args.workers)]
            for task in as_completed(tasks):
                print(f"[STAGE WORKER COMPLETE] {stage} rank={task.result()}", flush=True)
        stage_summary(out, stage)
        verify_stage_data(out, stage)
    final_report = verify_pilot(out, require_complete=True)
    print("[PILOT COMPLETE] FOV-only Grid25 V4 labels passed stage-hash/data-provenance gates", flush=True)
    print(json.dumps(final_report["metrics"], indent=2), flush=True)
    print("[TRAINING READY] False; pilot only, not full Grid25 labeled data", flush=True)


def stage_summary_path(out, stage):
    return Path(out) / {"global_pool": "global_pool_summary.json",
                        "v3_labels": "v3_summary.json",
                        "v4_tubes": "v4_summary.json"}[stage]


def verify_stage_data(out, stage):
    out, m, bank, plan = open_pilot(out)
    n = len(plan["x_index"])
    summaries = {"global_pool": "global_pool_summary.json",
                 "v3_labels": "v3_summary.json",
                 "v4_tubes": "v4_summary.json"}
    path = out / summaries[stage]
    require(path.is_file(), "Missing stage summary " + str(path))
    cfg = prod.Config(**m["config"])
    if stage == "v4_tubes":
        anchors = prod.verify_stage(out, "v4_anchors", n)
        tubes = prod.verify_stage(out, "v4_tubes", n)
        require(len(anchors) == len(tubes) == n, "Incomplete V4 anchor/tube shards")
    else:
        checks = prod.verify_stage(out, stage, n)
        require(len(checks) == n, "Incomplete stage shards: " + stage)
    # Each row uses the new-grid pilot x_index and a preserved train/val split.
    for sh in range(n):
        split = int(plan["split"][sh])
        xi = int(plan["x_index"][sh])
        stages = ("v4_anchors", "v4_tubes") if stage == "v4_tubes" else (stage,)
        for sub in stages:
            path, _ = prod.stage_paths(out, sub, sh)
            with np.load(path, allow_pickle=False) as z:
                require(np.all(z["x_index"] == xi), f"Wrong pilot x in {sub}/{sh}")
                require(np.all(z["split"] == split), f"Wrong pilot split in {sub}/{sh}")
                if sub == "v3_labels":
                    require(np.isfinite(z["q"]).all(), "Invalid V3 query q")
                    require(np.isfinite(z["new_value"][z["value_valid"]]).all(),
                            "Non-finite qualified V3 value")
                    require(np.isfinite(z["gradient_candidate"][z["gradient_candidate_valid"]]).all(),
                            "Non-finite V3 candidate gradient")
                if sub == "v4_tubes":
                    require(np.isfinite(z["q"]).all() and np.isfinite(z["grad"]).all()
                            and np.isfinite(z["value"]).all(), "Invalid V4 tube data")
                    require(np.all(np.isin(np.round(z["value"], 6),
                                               [0., -.005, .005, -.01, .01, -.02, .02])),
                            "Unexpected V4 normal offset")


def verify_pilot(out, require_complete=False):
    out, m, bank, plan = open_pilot(out)
    summaries = {}
    for stage in STAGES:
        path = stage_summary_path(out, stage)
        if path.is_file():
            verify_stage_data(out, stage)
            summaries[stage] = read_json(path)
        elif require_complete:
            raise RuntimeError("Pilot incomplete; missing " + str(path))
    if summaries:
        require((out / "geometry_preflight.json").is_file(), "Missing parity gate file")
        gate = read_json(out / "geometry_preflight.json")
        require(gate["status"] == "PASS" and
                gate["pilot_manifest_sha256"] == sha(out / "manifest.json"),
                "Geometry gate mismatch")
    completed = len(summaries) == len(STAGES)
    report = {"status": "PILOT_COMPLETE" if completed else "PILOT_PARTIAL",
              "training_ready": False, "selected_x": len(plan["x_index"]),
              "train_x": int((plan["split"] == 0).sum()),
              "val_x": int((plan["split"] == 1).sum()),
              "metrics": summaries,
              "limitations": [
                  "No LOS or self-occlusion labels",
                  "V3 gradient_candidate_valid does NOT imply finite-difference PASS; not a verified training grad mask",
                  "V4 tubes are accepted strict regular, same-face local normal offsets; not full signed distance everywhere",
                  "Small representative pilot does not establish full-grid training readiness",
              ]}
    if completed:
        dest = out / "pilot_report.json"
        if dest.exists():
            require(read_json(dest) == report, "Pilot report mutated")
        else:
            save_json_exclusive(dest, report)
    print("[PILOT VERIFIED]" if completed else "[PILOT PARTIAL]",
          "train=", report["train_x"], "val=", report["val_x"],
          "stages=", list(summaries), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "work", "verify", "status"])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--trainprep", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--train-x", type=int, default=2)
    p.add_argument("--val-x", type=int, default=2)
    p.add_argument("--seed", type=int, default=260927)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.trainprep is None:
        a.trainprep = a.root / "derived/grid25_trainprep_v1"
    if a.out is None:
        a.out = a.root / "derived/grid25_v4_pilot_v1/pilot_train2_val2"
    if a.mode == "prepare":
        prepare(a)
    elif a.mode == "work":
        run(a)
    elif a.mode == "status":
        if Path(a.out, "manifest.json").is_file():
            out, m, bank, plan = open_pilot(a.out)
            print(json.dumps({
                "pilot": str(out),
                "counts": m["counts"],
                "stages": {stage: stage_summary_path(out, stage).exists()
                           for stage in STAGES},
                "geometry_gate": (out / "geometry_preflight.json").exists(),
            }, indent=2), flush=True)
        else:
            print("[NOT PREPARED]", a.out)
    else:
        verify_pilot(a.out, require_complete=False)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, FileNotFoundError, OSError) as exc:
        raise SystemExit("[STOP] " + str(exc))
