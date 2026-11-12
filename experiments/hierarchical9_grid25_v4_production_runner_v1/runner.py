#!/usr/bin/env python3
"""One manually submitted Grid25 V4 production segment, immutable and resumable.

Reuse original full V4 global_pool / v3_labels / v4_tubes workers unchanged.
Only replace their open_run data adapter to the frozen Grid25 production plan.

This DOES NOT produce FD-verified gradients, train models, start additional
segments, or approve final training readiness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFLIGHT = REPO / "experiments/hierarchical9_grid25_v4_production_preflight_v1"
PILOT = REPO / "experiments/hierarchical9_grid25_v4_pilot_v1"
FULL = REPO / "experiments/hierarchical9_full_dataset_v4_v1"
sys.path.insert(0, str(PREFLIGHT))
import preflight as pre
sys.path.insert(0, str(PILOT))
import pilot as p

prod = p.prod
FORMAT = "careplanner_grid25_v4_production_segment_v1"
DEFAULT_ROOT = Path(
    "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
)
STAGES = ("global_pool", "v3_labels", "v4_tubes")
STAGE_SUMMARIES = {
    "global_pool": "global_pool_summary.json",
    "v3_labels": "v3_summary.json",
    "v4_tubes": "v4_summary.json",
}
FROZEN_FILES = ("runner.py", "run.sh", "submit.sh", "worker.sbatch")


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for buf in iter(lambda: handle.read(8 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_exclusive(path, data):
    path = Path(path)
    with path.open("x") as f:
        json.dump(prod.cache_io.clean_json(data), f, indent=2, sort_keys=True,
                  allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def runner_code_hashes():
    return {f: sha(HERE / f) for f in FROZEN_FILES}


def production_root(grid_root):
    return Path(grid_root).resolve() / "derived/grid25_v4_production_v1"


def preflight_root(grid_root):
    return Path(grid_root).resolve() / "derived/grid25_v4_production_preflight_v1"


def segment_out(grid_root, segment_id):
    require(isinstance(segment_id, int) and segment_id >= 0,
            "Segment ID must be nonnegative")
    return production_root(grid_root) / f"segment_{segment_id:03d}"


def segment_slice(plan, segment):
    start, end = int(segment["start"]), int(segment["stop"])
    require(0 <= start < end <= len(plan["x_index"]),
            "Invalid frozen segment interval")
    return {k: np.asarray(v[start:end]).copy()
            for k, v in plan.items()
            if k in ("x_index", "split", "valid_count", "support")}


def frozen_source(grid_root):
    root = preflight_root(grid_root)
    pre.verify(root)
    report = read(root / "report.json")
    with np.load(root / "x_plan.npz", allow_pickle=False) as z:
        plan = {k: z[k] for k in z.files}
    segments = read(root / "segments.json")
    require(len(segments) == report["total_work"]["manual_segments"],
            "Preflight segment list count changed")
    require(len(plan["x_index"]) == report["total_work"]["x"],
            "Preflight position count changed")
    require(prod.code_identity() ==
            report["provenance"]["original_algorithm_sha256"],
            "Original V3/V4 code identity differs from accepted preflight")
    return root, report, plan, segments


def prepare_one(grid_root, segment_id):
    pre_root, report, plan, segments = frozen_source(grid_root)
    require(0 <= segment_id < len(segments), "Segment out of planned range")
    segment = segments[segment_id]
    require(segment["segment_id"] == segment_id, "Segment ID/order mismatch")
    target = segment_out(grid_root, segment_id)
    require(not target.exists() and not target.is_symlink(),
            "Segment already prepared. Use status/submit; do not overwrite")
    local = segment_slice(plan, segment)
    size = len(local["x_index"])
    require(size == segment["x_count"] and np.all(np.diff(local["x_index"]) > 0),
            "Production segment x ordering/length changed")
    require(np.array_equal(local["support"], local["valid_count"] > 0),
            "Segment support/count disagreement")
    require(int(np.count_nonzero(local["support"])) ==
            int(segment["supported_x_sensor_pairs"]),
            "Segment sensor support totals disagree")
    require(int(np.minimum(local["valid_count"], 32).sum()) ==
            int(segment["v4_anchor_budget"]),
            "Segment frozen anchor budget disagrees")
    # The original full V4 worker assumes 128-x atomic shards.
    cfg = prod.Config(**report["frozen_label_config"])
    cfg.validate()
    require(cfg.x_shard_size == 128 and
            len(local["x_index"]) <= 8 * cfg.x_shard_size and
            report["budget_assumptions"]["workers"] == 4,
            "Unapproved segment/worker resource protocol")
    complete_segments = len(segments)
    bank_path = Path(report["trainprep"]) / "bank_cache"
    urdf = REPO / "src/arm_description/urdf/Arm.urdf"
    require(urdf.is_file(), "Frozen robot URDF missing")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".segment_{segment_id:03d}.",
                                 dir=target.parent))
    try:
        with (temp / "x_plan.npz").open("xb") as f:
            np.savez_compressed(f, **local)
            f.flush()
            os.fsync(f.fileno())
        manifest = dict(
            format=FORMAT, status="PREPARED", training_ready=False,
            segment_id=segment_id, number_of_planned_segments=complete_segments,
            segment_plan=segment,
            preflight_root=str(pre_root),
            preflight_report_sha256=sha(pre_root / "report.json"),
            preflight_x_plan_sha256=report["x_plan_sha256"],
            preflight_segments_sha256=report["segments_sha256"],
            x_plan_sha256=sha(temp / "x_plan.npz"),
            bank_cache=str(bank_path),
            bank_manifest_sha256=report["provenance"]["bank_manifest_sha256"],
            source_npz_sha256=report["provenance"]["original_npz_sha256"],
            trainprep=str(report["trainprep"]),
            urdf=str(urdf.resolve()), urdf_sha256=sha(urdf),
            config=asdict(cfg),
            solver=asdict(prod.core.SolverConfig()),
            verify=asdict(prod.core.VerifyConfig()),
            original_algorithms=prod.code_identity(),
            runner_source_hashes=runner_code_hashes(),
            counts=dict(
                x=size,
                train=int(np.count_nonzero(local["split"] == 0)),
                val=int(np.count_nonzero(local["split"] == 1)),
                supported_x_sensor_pairs=int(np.count_nonzero(local["support"])),
                v3_tasks=int(np.count_nonzero(local["support"]) * cfg.v3_q_per_x),
                v4_anchor_budget=int(np.minimum(local["valid_count"], 32).sum())),
            limitations=[
                "One manual segment; never automatically runs subsequent segments",
                "Original FOV-only Q0 bank; no LOS or self-occlusion",
                "V3 value_valid and gradient_candidate_valid are NOT FD-PASS",
                "V4 tube local normal offsets are NOT globally nearest signed distance",
                "No final full-dataset training-ready marker is created",
            ],
        )
        write_exclusive(temp / "manifest.json", manifest)
        os.rename(temp, target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print("[SEGMENT PREPARED]", target)
    print(json.dumps({"segment": segment_id, "counts": manifest["counts"],
                      "number_of_planned_segments": complete_segments,
                      "training_ready": False}, indent=2), flush=True)
    return target


def open_segment(out):
    out = Path(out).resolve()
    m = read(out / "manifest.json")
    require(m.get("format") == FORMAT and m.get("status") == "PREPARED"
            and m.get("training_ready") is False,
            "Invalid production segment identity")
    require(runner_code_hashes() == m["runner_source_hashes"],
            "Runner code changed; refuse mixing shards across code versions")
    require(prod.code_identity() == m["original_algorithms"],
            "Original production V3/V4 algorithm changed")
    pre_root = Path(m["preflight_root"])
    require(sha(pre_root / "report.json") == m["preflight_report_sha256"],
            "Preflight report changed")
    pre_root, r, full_plan, segments = frozen_source(pre_root.parents[1])
    require(r["x_plan_sha256"] == m["preflight_x_plan_sha256"] and
            r["segments_sha256"] == m["preflight_segments_sha256"],
            "Frozen full-grid plans changed")
    seg = segments[int(m["segment_id"])]
    require(seg == m["segment_plan"], "Frozen segment plan changed")
    require(sha(out / "x_plan.npz") == m["x_plan_sha256"],
            "Local x_plan changed")
    with np.load(out / "x_plan.npz", allow_pickle=False) as z:
        plan = {k: z[k] for k in z.files}
    expected = segment_slice(full_plan, seg)
    require(plan.keys() == expected.keys() and
            all(np.array_equal(plan[k], expected[k]) for k in expected),
            "Local segment x indices or split drifted from Grid25 plan")
    require(int(np.count_nonzero(plan["support"])) ==
            m["counts"]["supported_x_sensor_pairs"], "Local support totals changed")
    require(sha(m["urdf"]) == m["urdf_sha256"],
            "Robot URDF changed")
    require(sha(Path(m["bank_cache"]) / "bank_manifest.json") ==
            m["bank_manifest_sha256"], "Q0 bank identity changed")
    bank = prod.cache_io.BankCache(m["bank_cache"])
    require(bank.manifest["source_sha256"] == m["source_npz_sha256"],
            "Q0 cache comes from another dataset")
    require(bank.q.shape == (173225, 500, 7, 8) and
            bank.valid.shape == (173225, 500, 8),
            "Unexpected Grid25 bank schema")
    return out, m, bank, plan


def original_module(stage):
    import global_pool
    import v3_labels
    import v4_tubes
    for module in (global_pool, v3_labels, v4_tubes):
        module.open_run = open_segment
    global_pool.old_io = prod.cache_io
    return dict(global_pool=global_pool, v3_labels=v3_labels,
                v4_tubes=v4_tubes)[stage]


def stage_paths(stage):
    return ("v4_anchors", "v4_tubes") if stage == "v4_tubes" else (stage,)


def verify_one_stage(out, stage):
    out, m, _, plan = open_segment(out)
    require(stage in STAGES, "Unknown production stage")
    cfg = prod.Config(**m["config"])
    n = prod.shard_count(len(plan["x_index"]), cfg.x_shard_size)
    for sub in stage_paths(stage):
        prod.verify_stage(out, sub, n)
        for sh in range(n):
            a = sh * cfg.x_shard_size
            b = min(len(plan["x_index"]), a + cfg.x_shard_size)
            ix = plan["x_index"][a:b]
            sp = plan["split"][a:b]
            support = plan["support"][a:b]
            counts = plan["valid_count"][a:b]
            path, _ = prod.stage_paths(out, sub, sh)
            with np.load(path, allow_pickle=False) as z:
                indices = z["x_index"]
                actual_split = z["split"]
                if sub == "global_pool":
                    require(np.array_equal(indices, ix) and
                            np.array_equal(actual_split, sp) and
                            np.array_equal(z["support"], support),
                            f"Wrong global pool coverage in shard {sh}")
                else:
                    require(np.isin(indices, ix).all(),
                            f"Unexpected new-grid x in {sub} shard {sh}")
                    expected_split = dict(zip(ix.tolist(), sp.tolist()))
                    require(np.array_equal(actual_split,
                            np.asarray([expected_split[int(i)] for i in indices])),
                            f"Split mismatch in {sub} shard {sh}")
                if sub == "v3_labels":
                    require(len(indices) == int(np.count_nonzero(support)
                                               * cfg.v3_q_per_x),
                            "Unexpected V3 task count")
                    require(np.isfinite(z["q"]).all(), "V3 q contains nonfinite")
                    valid = z["value_valid"]
                    require(np.isfinite(z["new_value"][valid]).all(),
                            "V3 valid value contains nonfinite")
                    # Critical: original gradient_candidate_valid is not a
                    # reoptimized FD audit; do not mark it FD-PASS.
                    require(np.isfinite(z["gradient_candidate"]
                                           [z["gradient_candidate_valid"]]).all(),
                            "V3 finite gradient candidate check failed")
                if sub == "v4_anchors":
                    require(len(indices) == int(np.minimum(counts, 32).sum()),
                            "V4 anchor budget differs from frozen plan")
                if sub == "v4_tubes":
                    require(np.isfinite(z["q"]).all() and
                            np.isfinite(z["grad"]).all() and
                            np.isfinite(z["value"]).all(),
                            "V4 tube q/gradient/offset nonfinite")
                    require(len(indices) <= 7 * int(np.minimum(counts, 32).sum()),
                            "V4 tube count exceeds frozen budget")
                    offsets = np.asarray((0., -.005, .005, -.01, .01, -.02, .02))
                    actual = np.asarray(z["value"], dtype=float)
                    if len(actual):
                        require(np.all(np.min(
                            np.abs(actual[:, None] - offsets[None, :]), axis=1
                        ) <= 5e-7), "Unexpected V4 tube offsets")
    name = STAGE_SUMMARIES[stage]
    require((out / name).is_file(), f"Stage summary missing: {name}")
    report = read(out / name)
    require(report.get("status") == "COMPLETE", "Upstream stage summary not complete")
    print("[STAGE VERIFIED]", stage, "shards=", n, flush=True)
    return report


def geometry_gate(out, m, bank, plan):
    path = out / "geometry_preflight.json"
    if path.exists():
        record = read(path)
        require(record.get("status") == "PASS" and
                record.get("manifest_sha256") == sha(out / "manifest.json") and
                record.get("urdf_sha256") == m["urdf_sha256"],
                "Frozen geometry preflight identity mismatch")
        return
    oracle = prod.RepoOracle(REPO, m["urdf"], "cpu",
                             joint_names=bank.joints, sensor_frames=bank.sensors)
    report = oracle.verify(bank, plan["x_index"][:2],
                           np.zeros((1, 7), dtype=np.float32))
    require(report["status"] == "PASS", "Geometry/Jacobian parity gate failed")
    write_exclusive(path, dict(
        status="PASS", manifest_sha256=sha(out / "manifest.json"),
        urdf_sha256=m["urdf_sha256"], reference_check=report))
    print("[GEOMETRY GATE PASS]", flush=True)


def stage_worker(stage, out, rank, world, urdf):
    module = original_module(stage)
    a = SimpleNamespace(out=Path(out), rank=rank, world=world,
                        urdf=Path(urdf), device="cpu")
    module.worker(a)
    return rank


def stage_merge(out, stage):
    module = original_module(stage)
    module.merge(SimpleNamespace(out=Path(out)))


def status(out):
    out, m, _, plan = open_segment(out)
    cfg = prod.Config(**m["config"])
    nsh = prod.shard_count(len(plan["x_index"]), cfg.x_shard_size)
    progress = {}
    for stage in ("global_pool", "v3_labels", "v4_anchors", "v4_tubes"):
        folder = out / stage
        progress[stage] = dict(published_shards=(
            sum(1 for i in range(nsh)
                if (folder / f"shard_{i:04d}.npz").is_file() and
                   (folder / f"shard_{i:04d}.json").is_file())
        ), expected_shards=nsh)
    report = dict(segment=m["segment_id"], counts=m["counts"],
                  progress=progress, completed=(out/"segment_complete.json").exists(),
                  training_ready=False)
    print(json.dumps(report, indent=2), flush=True)
    return report


def verify_segment(out, complete_required=False):
    out, m, bank, plan = open_segment(out)
    found = [s for s in STAGES if (out / STAGE_SUMMARIES[s]).is_file()]
    for stage in found:
        verify_one_stage(out, stage)
    if complete_required:
        require(found == list(STAGES), "Production segment is not complete")
    if len(found) == len(STAGES):
        complete = dict(
            format=FORMAT, status="SEGMENT_COMPLETE",
            segment_id=m["segment_id"], manifest_sha256=sha(out/"manifest.json"),
            per_stage_summaries={name: sha(out/STAGE_SUMMARIES[stage])
                                 for stage, name in STAGE_SUMMARIES.items()},
            x_plan_sha256=m["x_plan_sha256"],
            training_ready=False,
            limitations="No complete Grid25 training labels or independently FD-verified gradient mask",
        )
        marker = out / "segment_complete.json"
        if marker.exists():
            require(read(marker) == complete, "Completed production segment marker changed")
        else:
            write_exclusive(marker, complete)
        print("[SEGMENT VERIFIED COMPLETE]", m["segment_id"], flush=True)
    else:
        print("[SEGMENT PARTIAL]", m["segment_id"], "complete stages:", found, flush=True)


def run(out, workers=4):
    require(workers == 4, "Frozen 4 CPU workers required")
    out, m, bank, plan = open_segment(out)
    lock = out / ".run.lock"
    with lock.open("a+") as fd:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker owns this segment; refuse overlap") from exc
        geometry_gate(out, m, bank, plan)
        for stage in STAGES:
            if (out / STAGE_SUMMARIES[stage]).exists():
                print("[RESUME] Stage summary exists:", stage, flush=True)
                verify_one_stage(out, stage)
                continue
            print("[STAGE BEGIN]", stage, "segment=", m["segment_id"],
                  "workers=", workers, flush=True)
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(stage_worker, stage, str(out), rank,
                                       workers, m["urdf"]) for rank in range(workers)]
                for f in as_completed(futures):
                    print("[WORKER FINISHED]", stage, f.result(), flush=True)
            stage_merge(out, stage)
            verify_one_stage(out, stage)
        verify_segment(out, complete_required=True)
        print("[PRODUCTION SEGMENT COMPLETE]", m["segment_id"], flush=True)
        print("[TRAINING READY] False; this is only one segment", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare","status","work","verify"))
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare_one(args.root, args.segment)
    else:
        out = args.out or segment_out(args.root, args.segment)
        if args.mode == "status":
            status(out)
        elif args.mode == "verify":
            verify_segment(out, complete_required=True)
        else:
            run(out, args.workers)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, FileNotFoundError, OSError) as exc:
        raise SystemExit("[STOP] " + str(exc))
