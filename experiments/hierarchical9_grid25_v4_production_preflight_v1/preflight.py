#!/usr/bin/env python3
"""Read-only full-scale Grid25 V4 production capacity preflight.

Creates a frozen, independently indexed production plan and cost report.
Does NOT generate V3/V4 labels, run FD probes, submit Slurm, or train.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAINPREP = REPO / "experiments/hierarchical9_grid25_training_prep_v1"
EXTENDED = REPO / "experiments/hierarchical9_grid25_v4_extended_v1"
PILOT = REPO / "experiments/hierarchical9_grid25_v4_pilot_v1"
sys.path.insert(0, str(TRAINPREP))
from prepare import verify_plan
sys.path.insert(0, str(PILOT))
import pilot as p

FORMAT = "careplanner_grid25_v4_production_preflight_v1"
DEFAULT_ROOT = Path(
    "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1"
)


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def count_support(bank_valid, ids, block=256):
    """Exact per-x counts by streaming only mmap valid_fov, never q or NPZ."""
    ids = np.asarray(ids, dtype=np.int64)
    require(ids.ndim == 1 and block > 0, "Invalid bank scan parameters")
    require(np.all(ids >= 0) and np.all(ids < bank_valid.shape[0]),
            "Grid25 x indices out of bank bounds")
    counts = np.empty((len(ids), 8), dtype=np.uint16)
    for start in range(0, len(ids), block):
        stop = min(start + block, len(ids))
        selection = np.asarray(bank_valid[ids[start:stop]], dtype=bool)
        require(selection.ndim == 3 and selection.shape[2] == 8,
                "Unexpected valid_fov count layout")
        nvalid = selection.sum(axis=1, dtype=np.int64)
        require(np.all(nvalid <= np.iinfo(np.uint16).max),
                "Valid slot count exceeds stored integer capacity")
        counts[start:stop] = nvalid
    return counts


def build_shards(n, counts, *, shard_x=128):
    """Deterministic contiguous x-plan slices for future manual staged jobs."""
    require(shard_x > 0 and n == len(counts), "Invalid shard sizing")
    shards = []
    for start in range(0, n, shard_x):
        stop = min(start + shard_x, n)
        c = counts[start:stop]
        shards.append(dict(
            shard_id=len(shards), start=start, stop=stop, x_count=stop-start,
            supported_x_sensor_pairs=int(np.count_nonzero(c)),
            v3_tasks=int(np.count_nonzero(c) * 4),
            v4_anchor_budget=int(np.minimum(c, 32).sum()),
        ))
    return shards


def build_segments(shards, *, max_shards=8):
    """Group whole atomic shards; segments never share x or shard IDs."""
    require(max_shards >= 1 and len(shards) > 0, "Segment size/available shard count invalid")
    segments = []
    for begin in range(0, len(shards), max_shards):
        group = shards[begin:begin + max_shards]
        segments.append(dict(
            segment_id=len(segments),
            first_shard=group[0]["shard_id"],
            last_shard_exclusive=group[-1]["shard_id"]+1,
            start=group[0]["start"], stop=group[-1]["stop"],
            x_count=sum(s["x_count"] for s in group),
            v3_tasks=sum(s["v3_tasks"] for s in group),
            v4_anchor_budget=sum(s["v4_anchor_budget"] for s in group),
            supported_x_sensor_pairs=sum(s["supported_x_sensor_pairs"] for s in group),
        ))
    require(segments[0]["start"] == 0 and
            all(a["stop"] == b["start"] for a, b in zip(segments, segments[1:])),
            "Segment order/coverage mismatch")
    return segments


def pilot_rates(global_summary, v3_summary, v4_summary, pilot_manifest):
    """Measured empirical worker CPU-seconds per work unit, not a guarantee."""
    n = int(pilot_manifest["counts"]["selected_x"])
    tasks = int(v3_summary["tasks"])
    anchors = int(v4_summary["anchors"])
    require(n >= 1 and tasks >= 1 and anchors >= 1,
            "Extended pilot lacks timing denominators")
    require(int(v3_summary["valid_values"]) <= tasks and
            int(v4_summary["refined"]) <= anchors,
            "Extended pilot summary malformed")
    return dict(
        global_cpu_seconds_per_x=float(global_summary["worker_seconds"]) / n,
        v3_cpu_seconds_per_task=float(v3_summary["shard_worker_seconds"]) / tasks,
        v4_cpu_seconds_per_anchor=float(v4_summary["shard_worker_seconds"]) / anchors,
        pilot_positions=n,
        pilot_v3_tasks=tasks,
        pilot_v4_anchors=anchors,
        pilot_v3_valid_fraction=float(v3_summary["valid_fraction"]),
        pilot_v4_refined_fraction=float(v4_summary["refined_fraction"]),
        pilot_v4_regular_fraction=float(v4_summary["regular_fraction"]),
    )


def project_cost(unit, rates, workers=4, multiplier=2.0, overhead_minutes=10):
    require(workers >= 1 and math.isfinite(multiplier) and multiplier >= 1.0
            and math.isfinite(overhead_minutes) and overhead_minutes >= 0,
            "Invalid resource-planning assumptions")
    cpu = dict(
        global_pool=float(unit["x_count"]) * rates["global_cpu_seconds_per_x"],
        v3_labels=float(unit["v3_tasks"]) * rates["v3_cpu_seconds_per_task"],
        v4_tubes=float(unit["v4_anchor_budget"]) * rates["v4_cpu_seconds_per_anchor"],
    )
    total = sum(cpu.values())
    return dict(
        estimated_worker_cpu_hours_by_stage={k: round(v/3600, 3) for k, v in cpu.items()},
        estimated_worker_cpu_hours=round(total/3600, 3),
        baseline_ideal_wall_hours_at_workers=round(total / workers / 3600, 3),
        conservative_wall_hours_at_workers=round(
            multiplier*total/workers/3600 + overhead_minutes/60, 3),
    )


def prepare(args):
    root = args.root.resolve()
    trainprep = args.trainprep or root / "derived/grid25_trainprep_v1"
    extended = args.extended or root / "derived/grid25_v4_extended_v1/pilot_train32_val32"
    target = args.out or root / "derived/grid25_v4_production_preflight_v1"

    require(args.scan_block > 0 and args.shard_x > 0 and args.max_shards_per_job > 0,
            "Bad shard/scan configuration")
    require(args.shard_x == 128,
            "Frozen full-V4 original stage shard size is 128 x; change requires a separate protocol")
    require(args.max_shards_per_job <= 8,
            "Pilot preflight caps manual production segments at 8 shards/1024 x")
    require(args.workers == 4 and args.multiplier >= 1.0,
            "This preflight uses the tested 4-CPU allocation and nonoptimistic multiplier")
    require(not target.exists() and not target.is_symlink(),
            "Output exists; no overwrite. Use a new --out for changed assumptions")
    info, verified_root, merged = verify_plan(trainprep)
    require(root == verified_root, "Grid25 source root mismatches training split")
    out, pilot_manifest, bank, pilot_plan = p.open_pilot(extended)
    p.verify_pilot(out, require_complete=True)
    require(pilot_manifest["counts"]["train_x"] == 32 and
            pilot_manifest["counts"]["val_x"] == 32,
            "Production budget requires the completed 32+32 extended pilot")
    require(Path(pilot_manifest["trainprep"]).resolve() == Path(trainprep).resolve()
            and pilot_manifest["grid25_splits_sha256"] == info["splits_sha256"]
            and pilot_manifest["grid25_npz_sha256"] == info["source_npz_expected_sha256"],
            "Pilot does not match the frozen Grid25 training data")
    require(bank.manifest["status"] == "COMPLETE" and
            bank.manifest["source_sha256"] == info["source_npz_expected_sha256"],
            "Q0 bank provenance mismatch")
    require(bank.valid.shape == (173225, 500, 8), "Unexpected Q0 bank shape")
    with np.load(trainprep / "splits.npz", allow_pickle=False) as z:
        train = np.asarray(z["train_x_index"], dtype=np.int64)
        val = np.asarray(z["val_x_index"], dtype=np.int64)
        fresh = np.asarray(z["fresh_guard_x_index"], dtype=np.int64)
        vguard = np.asarray(z["val_guard_x_index"], dtype=np.int64)
    require(len(train) == 109490 and len(val) == 16711,
            "Unexpected Grid25 training/validation population")
    require(len(np.unique(np.concatenate((train, val, fresh, vguard)))) == 173225,
            "Split leakage / excluded rows changed")
    # Frozen label protocol: the V4 full scale original uses all eligible x,
    # 128 arbitrary q/x, 4 selected q/x, and <=32 anchor/(supported x,s).
    config = p.prod.Config(candidate_q_per_x=128, v3_q_per_x=4,
                           anchors_per_sensor=32, x_shard_size=128, seed=260927)
    config.validate()
    x_idx = np.concatenate((train, val))
    split = np.concatenate((np.zeros(len(train), np.uint8),
                            np.ones(len(val), np.uint8)))
    order = np.argsort(x_idx, kind="stable")
    x_idx, split = x_idx[order], split[order]
    require(len(x_idx) == 126201 and np.all(np.diff(x_idx) > 0),
            "Grid25 production x indices duplicated or unordered")
    counts = count_support(bank.valid, x_idx, args.scan_block)
    support = counts > 0
    require(support.any(axis=1).all(), "Unresolved entire x; review full data audit")
    by = {}
    for sp, name in ((0, "train"), (1, "val")):
        c = counts[split == sp]
        by[name] = dict(
            x=int(len(c)),
            supported_x_sensor_pairs=int(np.count_nonzero(c)),
            unresolved_x_sensor_pairs=int(len(c)*8-np.count_nonzero(c)),
            supported_x_by_sensor=(c>0).sum(axis=0).astype(int).tolist(),
            valid_q0_by_sensor=c.sum(axis=0, dtype=np.int64).astype(int).tolist(),
            v3_tasks=int(np.count_nonzero(c)*4),
            v4_anchor_budget=int(np.minimum(c, 32).sum()),
        )
    shards = build_shards(len(x_idx), counts, shard_x=args.shard_x)
    segments = build_segments(shards, max_shards=args.max_shards_per_job)
    rates = pilot_rates(
        p.read_json(out/"global_pool_summary.json"),
        p.read_json(out/"v3_summary.json"),
        p.read_json(out/"v4_summary.json"), pilot_manifest)
    total_unit = dict(x_count=len(x_idx),
                      v3_tasks=int(np.count_nonzero(counts)*4),
                      v4_anchor_budget=int(np.minimum(counts, 32).sum()))
    total_cost = project_cost(total_unit, rates, args.workers,
                              args.multiplier, args.overhead_minutes)
    for segment in segments:
        segment["estimated_cost"] = project_cost(
            segment, rates, args.workers, args.multiplier, args.overhead_minutes)
    worst = max(s["estimated_cost"]["conservative_wall_hours_at_workers"] for s in segments)
    # A 24h Slurm allocation is a hard limit. This is a cost forecast, not a
    # promise that no task can exceed the pilot-measured per-unit runtimes.
    require(worst < 20,
            "Largest planned segment exceeds 20h conservative budget; "
            "change only max-shards-per-job or collect more timing evidence")
    estimated_fd_all_probes = None
    fd_report = extended / "extended_fd_audit_v1/audit_summary.json"
    if fd_report.is_file():
        fd = p.read_json(fd_report)
        require(fd["status"] == "FD_COMPLETE" and fd["fixed_strata"] == 32
                and fd["selected_strata"] == 32, "Unexpected FD summary")
        estimated_fd_all_probes = dict(
            pilot_fd_pass=int(fd["fd_pass_count"]),
            pilot_selected_strata=int(fd["selected_strata"]),
            full_fd_validation="NOT PLANNED; per-row reoptimization may cost far more than basic V3"
        )
    report = dict(
        format=FORMAT,
        status="COST_PREFLIGHT_ONLY_NOT_SUBMITTED",
        training_ready=False,
        grid25_root=str(root),
        trainprep=str(Path(trainprep).resolve()),
        extended_pilot=str(extended.resolve()),
        source_npz=str(merged),
        provenance=dict(
            complete_sha256=info["grid_complete_sha256"],
            splits_sha256=info["splits_sha256"],
            original_npz_sha256=info["source_npz_expected_sha256"],
            bank_manifest_sha256=p.sha(Path(trainprep)/"bank_cache/bank_manifest.json"),
            pilot_manifest_sha256=p.sha(extended/"manifest.json"),
            pilot_summary_sha256={
                filename: p.sha(extended/filename) for filename in (
                    "global_pool_summary.json","v3_summary.json","v4_summary.json")
            },
            urdf_sha256=pilot_manifest["urdf_sha256"],
            original_algorithm_sha256=pilot_manifest["code_fingerprint"]["upstream_full_v4"],
        ),
        frozen_label_config=asdict(config),
        split_counts=by,
        total_work=dict(
            x=total_unit["x_count"],
            supported_sensor_pairs=int(np.count_nonzero(support)),
            unresolved_sensor_pairs=int(len(x_idx)*8-np.count_nonzero(support)),
            global_query_q=int(len(x_idx)*128),
            global_sensor_signs=int(len(x_idx)*128*8),
            v3_tasks=total_unit["v3_tasks"],
            v4_anchor_budget=total_unit["v4_anchor_budget"],
            v4_max_tube_candidates=int(total_unit["v4_anchor_budget"]*7),
            shards=len(shards), manual_segments=len(segments),
        ),
        pilot_observed_rates=rates,
        budget_assumptions=dict(
            workers=args.workers, cpu_only=True, gpus=0,
            conservative_multiplier=args.multiplier,
            per_job_fixed_overhead_minutes=args.overhead_minutes,
            assumed_job_walltime_limit_hours=24,
            manual_max_shards_per_job=args.max_shards_per_job,
            old_grid_x_indices_reused=False,
        ),
        estimated_total_cost=total_cost,
        max_segment_conservative_wall_hours=worst,
        fd_evidence=estimated_fd_all_probes,
        production_policy=dict(
            no_auto_submit=True, no_production_worker_in_this_entry=True,
            do_not_reuse_old_x_index=True,
            unresolved_sensor_is_not_infeasible=True,
            no_los_or_self_occlusion=True,
            gradient_candidate_not_fd_pass=True,
            gradient_fd_hold_is_masked=True,
            unchanged_v4_original_uniform_q_policy=True,
            no_training_labels_published=True,
        ),
        limitations=[
            "Only 64 x pilot timings; cost projections are indicative, not guarantees.",
            "V3 task difficulty varies by sensor/position; queue wait, shared FS load and worker imbalance omitted.",
            "No global SDF or verified whole-dataset gradient mask is created.",
            "This is only a frozen x-index plan and production cost report, not an executable production submission."
        ])
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".production_preflight.", dir=target.parent))
    try:
        with (tmp/"x_plan.npz").open("xb") as f:
            np.savez_compressed(f, x_index=x_idx, split=split,
                                valid_count=counts, support=support)
            f.flush()
            os.fsync(f.fileno())
        report["x_plan_sha256"] = p.sha(tmp/"x_plan.npz")
        with (tmp/"shards.json").open("x") as f:
            json.dump(shards, f, indent=2)
            f.write("\n")
        report["shards_sha256"] = p.sha(tmp/"shards.json")
        with (tmp/"segments.json").open("x") as f:
            json.dump(segments, f, indent=2)
            f.write("\n")
        report["segments_sha256"] = p.sha(tmp/"segments.json")
        with (tmp/"report.json").open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.rename(tmp, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    print("[PRODUCTION PREFLIGHT READY]", target)
    print(json.dumps({
        "split_counts": {k: v["x"] for k, v in by.items()},
        "total_work": report["total_work"],
        "observed_v3_cpu_s_per_task": rates["v3_cpu_seconds_per_task"],
        "estimated_total_cost": report["estimated_total_cost"],
        "max_segment_conservative_wall_hours": worst,
        "training_ready": False,
        "automatic_submit": False,
    }, indent=2))
    return target


def verify(out):
    out = Path(out).resolve()
    r = json.loads((out/"report.json").read_text())
    require(r["format"] == FORMAT and r["status"] == "COST_PREFLIGHT_ONLY_NOT_SUBMITTED"
            and r["training_ready"] is False and
            r["production_policy"]["no_auto_submit"] is True,
            "Wrong production preflight manifest")
    for file, digest_name in (
        ("x_plan.npz","x_plan_sha256"),
        ("shards.json","shards_sha256"),
        ("segments.json","segments_sha256")
    ):
        require(p.sha(out/file) == r[digest_name], "Frozen production plan changed: " + file)
    with np.load(out/"x_plan.npz", allow_pickle=False) as z:
        ids, split, counts, support = (z[k] for k in
                                      ("x_index","split","valid_count","support"))
        require(len(ids) == r["total_work"]["x"] and
                np.array_equal(counts>0, support) and
                np.array_equal(np.unique(ids), ids) and
                np.isin(split, [0,1]).all(),
                "Plan arrays mismatch")
        require(int((split==0).sum()) == r["split_counts"]["train"]["x"]
                and int((split==1).sum()) == r["split_counts"]["val"]["x"],
                "Plan split counts mismatch")
        require(int(support.sum()) == r["total_work"]["supported_sensor_pairs"]
                and int(np.minimum(counts,32).sum()) == r["total_work"]["v4_anchor_budget"],
                "Plan counts changed")
    shards = json.loads((out/"shards.json").read_text())
    segments = json.loads((out/"segments.json").read_text())
    require(len(shards) == r["total_work"]["shards"] and
            len(segments) == r["total_work"]["manual_segments"] and
            segments[0]["start"] == 0 and segments[-1]["stop"] == len(ids)
            and all(x["stop"]==y["start"] for x,y in zip(segments,segments[1:])),
            "Plan segments do not fully partition Grid25")
    print("[PRODUCTION PREFLIGHT VERIFIED] all plans immutable and complete")
    print("[NO TRAINING] No full labels generated and no Slurm jobs submitted")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare","verify"))
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--trainprep", type=Path)
    ap.add_argument("--extended", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--scan-block", type=int, default=256)
    ap.add_argument("--shard-x", type=int, default=128)
    ap.add_argument("--max-shards-per-job", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--multiplier", type=float, default=2.0)
    ap.add_argument("--overhead-minutes", type=float, default=10.)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        verify(args.out or args.root/"derived/grid25_v4_production_preflight_v1")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, FileNotFoundError) as e:
        raise SystemExit("[STOP] "+str(e))
