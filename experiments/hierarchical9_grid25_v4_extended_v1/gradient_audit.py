#!/usr/bin/env python3
"""Independent, frozen-stratum FD audit of Grid25 extended V3 distance gradients.

No new V3/V4 labels and no training. Uses original v2 two-scale/two-sided
verify_gradient, original V3 SLSQP/screen/reference gates. Invalid labels
remain in the audit denominator; no post-hoc replacement of failed samples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments/hierarchical9_grid25_v4_pilot_v1"
sys.path.insert(0, str(BASE))
import pilot as p

prod = p.prod
DEFAULT_ROOT = p.DEFAULT_ROOT
FORMAT = "careplanner_grid25_v4_extended_fd_audit_v1"
OUTPUT_SUBDIR = "extended_fd_audit_v1"
STRATA = [(split, s, sign) for split in (0, 1)
          for s in range(8) for sign in (-1, 1)]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return p.sha(path)


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"No overwrite allowed: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("x") as f:
            json.dump(prod.cache_io.clean_json(data), f, sort_keys=True,
                      indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        # Same task is never deliberately submitted twice. os.link refuses
        # to replace an existing task record even under a race.
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def deterministic_strata(rows, *, sign_guard_m=0.0):
    """Select one query per split × sensor × sign using only FOV pool identities.

    Each row: (shard, x_index, split, sensor, q_slot, pool_index, margin_m).
    Nothing about V3 success, gradients or V4 validity is an input.
    """
    groups = defaultdict(list)
    for shard, xi, split, sensor, slot, pool_idx, margin in rows:
        require(split in (0, 1) and 0 <= sensor < 8 and
                0 <= slot < 4 and np.isfinite(margin), "Malformed pool row")
        if abs(float(margin)) <= sign_guard_m:
            continue
        sign = 1 if margin >= 0 else -1
        candidate = dict(shard=int(shard), x_index=int(xi), split=int(split),
                         sensor=int(sensor), q_slot=int(slot),
                         pool_index=int(pool_idx), query_fov_sign=sign,
                         query_fov_margin_m=float(margin))
        groups[split, sensor, sign].append(candidate)
    plan = []
    missing = []
    for split, sensor, sign in STRATA:
        group = groups.get((split, sensor, sign), [])
        if not group:
            missing.append(dict(split=split, sensor=sensor, sign=sign))
            continue
        # Hash query identity only; zero dependence on label validity.
        def rank(entry):
            identity = (entry["x_index"], entry["split"],
                        entry["sensor"], entry["pool_index"], entry["q_slot"])
            return hashlib.sha256(repr(identity).encode()).hexdigest()
        selection = min(group, key=rank)
        plan.append(dict(task_id=len(plan), **selection))
    return plan, missing


def pool_rows(out, plan):
    """Only global_pool: exact upstream prior-to-optimization query pool."""
    rows = []
    for shard, (xi, split) in enumerate(zip(plan["x_index"], plan["split"])):
        path, _ = prod.stage_paths(out, "global_pool", shard)
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z["x_index"], [xi]) and
                    np.array_equal(z["split"], [split]),
                    f"Wrong Grid25 new-grid point at global_pool shard {shard}")
            support = z["support"][0]
            selected = z["v3_selected_index"][0]
            margins = z["g_pool"][0]
            for s in np.flatnonzero(support):
                for slot, pool_idx in enumerate(selected):
                    rows.append((shard, int(xi), int(split), int(s),
                                 int(slot), int(pool_idx),
                                 float(margins[int(pool_idx), int(s)])))
    return rows


def audit_root(out):
    return Path(out).resolve() / OUTPUT_SUBDIR


def make_plan(out):
    out, pilot_m, _, plan = p.open_pilot(out)
    p.verify_pilot(out, require_complete=True)
    require(int(pilot_m["counts"]["train_x"]) >= 16 and
            int(pilot_m["counts"]["val_x"]) >= 16,
            "FD audit expects the extended pilot, not the old 2+2 baseline")
    root = audit_root(out)
    require(not root.exists() and not root.is_symlink(),
            "FD audit already prepared; no overwrite")
    # Validate global pool files before freezing selectors.
    prod.verify_stage(out, "global_pool", len(plan["x_index"]))
    rows = pool_rows(out, plan)
    solver = prod.core.SolverConfig(**pilot_m["solver"])
    selection, missing = deterministic_strata(rows, sign_guard_m=solver.sign_guard_m)
    root.mkdir(parents=True)
    atomic_json(root / "fd_plan.json", {
        "format": FORMAT, "selection_policy": "one fixed hash-ranked row per split/sensor/sign",
        "sign_guard_m": float(solver.sign_guard_m),
        "selected": selection, "missing_strata": missing,
        "all_strata": len(STRATA),
        "no_selection_using_v3_success": True,
    })
    atomic_json(root / "audit_manifest.json", {
        "format": FORMAT, "status": "PREPARED", "training_ready": False,
        "pilot_root": str(out),
        "pilot_manifest_sha256": sha(out / "manifest.json"),
        "pilot_x_plan_sha256": sha(out / "x_plan.npz"),
        "pilot_code_fingerprint": pilot_m["code_fingerprint"],
        "fd_plan_sha256": sha(root / "fd_plan.json"),
        "audit_source_sha256": sha(HERE / "gradient_audit.py"),
        "selected_tasks": len(selection), "missing_strata": len(missing),
        "scientific_method": "v2.verify_gradient, two-scale two-sided and one-sided reoptimized signed distance",
        "scope": "diagnostic only; gradients NOT automatically enabled as training supervision",
    })
    print("[FD PLAN PREPARED]", root)
    print(json.dumps({"fixed_tasks": len(selection), "missing_strata": len(missing),
                      "missing": missing, "training_ready": False}, indent=2))


def open_audit(out):
    out, m, bank, plan = p.open_pilot(out)
    root = audit_root(out)
    a = read(root / "audit_manifest.json")
    frozen = read(root / "fd_plan.json")
    require(a["format"] == frozen["format"] == FORMAT and
            a["status"] == "PREPARED" and a["training_ready"] is False,
            "Invalid FD audit identity")
    require(a["pilot_manifest_sha256"] == sha(out / "manifest.json") and
            a["pilot_x_plan_sha256"] == sha(out / "x_plan.npz") and
            a["pilot_code_fingerprint"] == m["code_fingerprint"],
            "Extended Pilot source or algorithm changed")
    require(a["fd_plan_sha256"] == sha(root / "fd_plan.json") and
            a["audit_source_sha256"] == sha(HERE / "gradient_audit.py"),
            "FD audit plan/code changed")
    require(len(frozen["selected"]) == a["selected_tasks"],
            "Fixed audit task count changed")
    require(frozen["no_selection_using_v3_success"] is True,
            "FD audit selections must not depend on success")
    return out, m, bank, plan, root, frozen


def match_v3_record(out, selected):
    shard = selected["shard"]
    path, _ = prod.stage_paths(out, "v3_labels", shard)
    with np.load(path, allow_pickle=False) as z:
        match = ((z["x_index"] == selected["x_index"]) &
                 (z["split"] == selected["split"]) &
                 (z["sensor"] == selected["sensor"]) &
                 (z["q_slot"] == selected["q_slot"]) &
                 (z["pool_index"] == selected["pool_index"]))
        require(int(match.sum()) == 1, "V3 row absent or duplicated in frozen stage")
        idx = int(np.flatnonzero(match)[0])
        keys = ("q", "reference_g_m", "new_value", "value_valid",
                "gradient_candidate", "gradient_candidate_valid", "reason",
                "selected_plane", "stationarity", "attempts", "successes")
        return {key: np.asarray(z[key][idx]).copy() for key in keys}


def audit_one(selected, out, m, bank, oracle):
    s = int(selected["sensor"])
    xi = int(selected["x_index"])
    frozen = match_v3_record(out, selected)
    q = np.asarray(frozen["q"], dtype=np.float64)
    ref = float(frozen["reference_g_m"])
    require((1 if ref >= 0 else -1) == selected["query_fov_sign"] and
            abs(ref - selected["query_fov_margin_m"]) <= 1e-5,
            "Frozen source q sign/margin mismatch")
    result = dict(task_id=selected["task_id"], source=dict(selected),
                  original_v3=dict(value_valid=bool(frozen["value_valid"]),
                                   reason=str(frozen["reason"]),
                                   selected_plane=int(frozen["selected_plane"]),
                                   attempts=int(frozen["attempts"]),
                                   successes=int(frozen["successes"])),
                  fd_status="NOT_RUN", fd_reasons=[], fd_pass=False,
                  gradient_is_full_verified_training_label=False)
    if not bool(frozen["value_valid"]):
        result["fd_status"] = "SKIPPED_ORIGINAL_V3_INVALID"
        return result
    mask = np.asarray(bank.masks[s], float)
    points = bank.sensor_bank(xi, s)
    x = np.asarray(bank.x[xi], float)
    geo = prod.AnalyticGeometry(oracle, x, s)
    cfg = prod.core.SolverConfig(**m["solver"])
    vc = prod.core.VerifyConfig(**m["verify"])
    import v3_labels
    # Repeat central V3 solve under exactly the same original solver,
    # verify it matches the frozen V3 outputs; never silently relabel.
    central = prod.core.solve_attempts(q, points, mask, bank.lo, bank.hi, geo, cfg)
    base = prod.core.screen_candidates(q, mask, bank.lo, bank.hi, geo, central, cfg, vc)
    base = v3_labels.v3.reference_gate(oracle, x, s, q, base, cfg)
    require(bool(base["value_valid"]) == bool(frozen["value_valid"]) and
            base["value"] is not None and
            np.isclose(base["value"], float(frozen["new_value"]), rtol=0, atol=5e-6),
            "Recomputed center differs from saved original V3 label")
    require(np.allclose(np.asarray(base["gradient_candidate"], float),
                        np.asarray(frozen["gradient_candidate"], float),
                        rtol=0, atol=5e-5, equal_nan=True),
            "Recomputed gradient candidate differs from saved original V3")
    # Reuse V2 original probe budget; qualified original roots as warm seeds
    # (same verified_core.label_verified policy).
    warm = [c["q_star"] for c in base["clusters"][:2] if c["qualified"]]
    probe_cfg = replace(cfg, starts=vc.probe_starts, maxiter=vc.probe_maxiter)
    def probe(trial):
        attempts = prod.core.solve_attempts(trial, points, mask, bank.lo,
                                            bank.hi, geo, probe_cfg, warm)
        out = prod.core.screen_candidates(trial, mask, bank.lo, bank.hi,
                                          geo, attempts, cfg, vc)
        out = v3_labels.v3.reference_gate(oracle, x, s, trial, out, cfg)
        out["attempts"] = attempts
        return out
    verified = prod.core.verify_gradient(
        base, q, mask, bank.lo, bank.hi, probe, cfg, vc)
    status = str(verified["verification"]["status"])
    result.update(
        fd_status=status,
        fd_reasons=list(verified["gradient_reasons"]),
        fd_pass=bool(status == "PASS" and verified["grad_valid"]
                     and verified["value_valid"]),
        fd_step_rad=verified["verification"].get("step_rad"),
        fd_directions=verified["verification"].get("directions"),
        fd_expected_probes=verified["verification"].get("expected_probes"),
        fd_completed_probes=verified["verification"].get("completed_probes"),
        fd_max_error=verified["verification"].get("max_error"),
        fd_value_still_valid=bool(verified["value_valid"]),
        gradient_is_full_verified_training_label=False,
    )
    return result


def worker_chunk(out, rank, world):
    out, m, bank, plan, root, frozen = open_audit(out)
    tasks = frozen["selected"][rank::world]
    if not tasks:
        return dict(rank=rank, complete=0)
    oracle = prod.RepoOracle(
        p.ROOT, m["urdf"], "cpu", joint_names=bank.joints,
        sensor_frames=bank.sensors)
    completed = 0
    for task in tasks:
        path = root / "fd_records" / f"{task['task_id']:02d}.json"
        if path.is_file():
            old = read(path)
            require(old["source"] == task and
                    old["task_id"] == task["task_id"],
                    "Previously completed FD task identity mismatch")
            print(f"[FD RESUME] task={task['task_id']}", flush=True)
            completed += 1
            continue
        row = audit_one(task, out, m, bank, oracle)
        atomic_json(path, row)
        print("[FD RESULT]", task["task_id"], "split", task["split"],
              "sensor", task["sensor"], "sign", task["query_fov_sign"],
              "status", row["fd_status"], flush=True)
        completed += 1
    return dict(rank=rank, complete=completed)


def audit_run(out, workers):
    require(1 <= workers <= 4, "Expected 1..4 CPU workers")
    out, _, _, _, _, frozen = open_audit(out)
    # Immutable original labels are fully verified before FD selection.
    p.verify_stage_data(out, "v3_labels")
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(worker_chunk, out, r, workers)
                   for r in range(workers)]
        for f in as_completed(futures):
            print("[FD WORKER FINISHED]", f.result(), flush=True)
    audit_summary(out, require_complete=True)


def diagnostics(out, bank, plan):
    """Full extended pilot descriptive reports, not a training approval gate."""
    sign = {
        "uniform_signs_all_q": np.zeros((8, 2), np.int64),
        "selected_v3_signs": np.zeros((8, 2), np.int64),
        "supported_state_available": np.zeros((8, 2), np.int64),
        "supported_state_covered": np.zeros((8, 2), np.int64),
        "split_selected_v3": np.zeros((2, 8, 2), np.int64),
    }
    s2 = dict(v3_reasons=Counter(), v4_anchor_reasons=Counter(),
              v3_invalid_examples=[], v4_failed_examples=[],
              v3_by_split=[dict(total=0, valid=0) for _ in (0, 1)])
    n = len(plan["x_index"])
    for sh in range(n):
        pg, _ = prod.stage_paths(out, "global_pool", sh)
        with np.load(pg, allow_pickle=False) as z:
            g = z["g_pool"][0]
            selected = z["v3_selected_index"][0]
            for s in range(8):
                sign["uniform_signs_all_q"][s, 0] += int(np.count_nonzero(g[:, s] < 0))
                sign["uniform_signs_all_q"][s, 1] += int(np.count_nonzero(g[:, s] >= 0))
                sign["selected_v3_signs"][s, 0] += int(np.count_nonzero(g[selected, s] < 0))
                sign["selected_v3_signs"][s, 1] += int(np.count_nonzero(g[selected, s] >= 0))
                sp = int(plan["split"][sh])
                sign["split_selected_v3"][sp, s, 0] += int(np.count_nonzero(g[selected, s] < 0))
                sign["split_selected_v3"][sp, s, 1] += int(np.count_nonzero(g[selected, s] >= 0))
            sign["supported_state_available"] += np.asarray(z["sign_state_available"][0], np.int64)
            sign["supported_state_covered"] += np.asarray(z["sign_state_covered"][0], np.int64)
        v3, _ = prod.stage_paths(out, "v3_labels", sh)
        with np.load(v3, allow_pickle=False) as z:
            for i in np.flatnonzero(z["sensor"] == 2):
                sp = int(z["split"][i])
                s2["v3_reasons"][str(z["reason"][i])] += 1
                s2["v3_by_split"][sp]["total"] += 1
                if z["value_valid"][i]:
                    s2["v3_by_split"][sp]["valid"] += 1
                elif len(s2["v3_invalid_examples"]) < 100:
                    s2["v3_invalid_examples"].append(dict(
                        x_index=int(z["x_index"][i]), split=sp,
                        q_slot=int(z["q_slot"][i]),
                        pool_index=int(z["pool_index"][i]),
                        reason=str(z["reason"][i]),
                        reference_g_m=float(z["reference_g_m"][i]),
                        attempts=int(z["attempts"][i]),
                        successes=int(z["successes"][i]),
                        stationarity=float(z["stationarity"][i])
                    ))
        v4, _ = prod.stage_paths(out, "v4_anchors", sh)
        with np.load(v4, allow_pickle=False) as z:
            for i in np.flatnonzero(z["sensor"] == 2):
                reason = str(z["reason"][i])
                s2["v4_anchor_reasons"][reason] += 1
                if not z["refined"][i] and len(s2["v4_failed_examples"]) < 100:
                    s2["v4_failed_examples"].append(dict(
                        x_index=int(z["x_index"][i]), split=int(z["split"][i]),
                        slot=int(z["slot"][i]), reason=reason,
                        boundary_g_m=float(z["boundary_g_m"][i]),
                        iterations=int(z["iterations"][i]),
                        geometry_calls=int(z["geometry_calls"][i]),
                    ))
    summary = prod.cache_io.clean_json({
        "sign_balance": {k: v.tolist() for k, v in sign.items()},
        "notes": {
            "sign_counts": "FOV sign only; negative means outside",
            "selected_v3_signs": "Counts all sensors including unsupported sensors; not all selected labels are produced",
            "potential_ablation": "Do not change source V4 query selection inside this experiment; sign balancing is separate",
        },
        "S2_diagnostics": s2,
    })
    return summary


def audit_summary(out, require_complete=False):
    out, m, bank, plan, root, frozen = open_audit(out)
    count = Counter()
    reasons = Counter()
    by = defaultdict(lambda: Counter())
    records = []
    missing = []
    for task in frozen["selected"]:
        path = root / "fd_records" / f"{task['task_id']:02d}.json"
        if not path.is_file():
            missing.append(task["task_id"])
            continue
        r = read(path)
        require(r["source"] == task, "Changed FD task identity")
        count[r["fd_status"]] += 1
        by[f"split{task['split']}_S{task['sensor']}_sign{task['query_fov_sign']}"][r["fd_status"]] += 1
        for reason in r["fd_reasons"]:
            reasons[reason] += 1
        records.append(r)
    if require_complete:
        require(not missing, "FD audit still has missing tasks: " + str(missing))
    extras = []
    pth = root / "fd_records"
    if pth.is_dir():
        expected_names = {f"{t['task_id']:02d}.json" for t in frozen["selected"]}
        extras = sorted(f.name for f in pth.iterdir() if f.name.endswith(".json")
                        and f.name not in expected_names)
    require(not extras, "Unexpected FD result files: " + str(extras))
    diagnostics_result = diagnostics(out, bank, plan) if not missing else None
    report = dict(
        status="FD_COMPLETE" if not missing else "FD_PARTIAL",
        training_ready=False,
        fixed_strata=len(STRATA),
        selected_strata=len(frozen["selected"]),
        missing_source_strata=frozen["missing_strata"],
        completed_audit_tasks=len(records),
        incomplete_tasks=missing,
        fd_status_counts=dict(count),
        gradient_holds_by_reason=dict(reasons),
        fd_pass_count=sum(bool(r["fd_pass"]) for r in records),
        fd_pass_fraction_of_fixed_selected=(
            sum(bool(r["fd_pass"]) for r in records) / max(len(frozen["selected"]), 1)),
        by_fixed_stratum={k: dict(v) for k, v in sorted(by.items())},
        diagnostics=diagnostics_result,
        limitations=[
            "Only sampled existing selected V3 queries; never certify unsampled gradients.",
            "Missing positive/negative source strata remain missing; no retrospective replacement.",
            "FD PASS is a local numerical label check, not global nearest-boundary proof.",
            "Original full Grid25 training_ready remains false.",
        ])
    if not missing:
        target = root / "audit_summary.json"
        if target.is_file():
            require(read(target) == report, "Summary identity changed")
        else:
            atomic_json(target, report)
    print("[FD AUDIT]", report["status"], "fixed strata=", len(STRATA),
          "selected=", len(frozen["selected"]), "fd_pass=", report["fd_pass_count"],
          "still_missing=", len(missing), flush=True)
    if not missing:
        print("[REPORT]", root / "audit_summary.json", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["plan", "work", "verify", "status"])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.out is None:
        a.out = a.root / "derived/grid25_v4_extended_v1/pilot_train32_val32"
    if a.mode == "plan":
        make_plan(a.out)
    elif a.mode == "work":
        audit_run(a.out, a.workers)
    elif a.mode == "verify":
        audit_summary(a.out, require_complete=True)
    else:
        out, _, _, _, root, frozen = open_audit(a.out)
        done = (len(list((root / "fd_records").glob("*.json")))
                if (root / "fd_records").is_dir() else 0)
        print(json.dumps({"status": "PREPARED", "selected": len(frozen["selected"]),
                          "completed": done, "audit_root": str(root)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, KeyError, FileNotFoundError) as exc:
        raise SystemExit("[STOP] " + str(exc))
