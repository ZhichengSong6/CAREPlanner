#!/usr/bin/env python3
"""Mainline B: fixed-V1 boundary-label audit and paired per-sensor solve probes.

Adds diagnostics only. Never writes checkpoint/data/runtime files. Use a separate
worktree from mainline A; --artifact-root may point at the original server repo.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from core import (Config, SensorField, bank_label, cosine, distribution, field_metrics,
                  json_safe, qualify_boundary, rate, refine_boundary, tangent_seed,
                  within, write_json)
from oracle import SensorOracle
from runtime_probe import analytic_ascent, make_probe, run_probe

REPO = Path(__file__).resolve().parents[2]
V1 = "979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199"
OLD8 = "43f962729adcd17aa114edb9fc410facbbb97ebe7343f0ad3309fe50d273acdb"
BASE = "dcde98b2dd75b1f590d34b4c2e7e1bee97b66bc6"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""): h.update(b)
    return h.hexdigest()


def emit(file, obj):
    file.write(json.dumps(json_safe(obj), ensure_ascii=False, allow_nan=False) + "\n")
    file.flush()


class Results:
    def __init__(self):
        self.boundaries = defaultdict(list)
        self.profiles = defaultdict(list)
        self.solves = defaultdict(list)
        self.excluded_starts = Counter()
        self.generation_counts = Counter()

    def summary(self):
        b, p, sol = {}, {}, {}
        for key, rows in self.boundaries.items():
            b[key] = {"attempts": len(rows), "regular": sum(r["regular"] for r in rows),
                "reasons": dict(Counter(reason for r in rows for reason in r["reasons"])),
                "abs_g_m": distribution(abs(r["g_m"]) for r in rows),
                "bank_distance_rad": distribution(r["bank_distance_rad"] for r in rows),
                "models": {}}
            for name in ("h9", "old8"):
                b[key]["models"][name] = {metric: distribution(r["models"][name][metric]
                    for r in rows if r["regular"]) for metric in
                    ("abs_value", "raw_gradient_norm", "masked_gradient_norm", "inactive_gradient_norm",
                     "raw_normal_cosine", "masked_normal_cosine")}
        for key, rows in self.profiles.items():
            valid = [r for r in rows if r.get("in_limits")]
            p[key] = {"attempts": len(rows), "in_limits": len(valid),
                "two_sided_expected_sign": rate(sum(r["expected_sign_ok"] for r in valid), len(valid)),
                "same_active_plane": rate(sum(r["same_active_plane"] for r in valid), len(valid)),
                "bank_normal_cosine": distribution(r["bank_normal_cosine"] for r in valid),
                "bank_distance_over_abs_offset": distribution(r["bank_distance_over_abs_offset"] for r in valid),
                "g_linearization_abs_error_m": distribution(r["linearization_abs_error_m"] for r in valid),
                "models": {name: {"sign_accuracy_vs_analytic_g": rate(sum(r["models"][name]["sign_correct"]
                    for r in valid), len(valid)), "raw_gradient_norm": distribution(r["models"][name]["raw_gradient_norm"]
                    for r in valid)} for name in ("h9", "old8")}}
        for key, rows in self.solves.items():
            sol[key] = {"paired_starts": len(rows), "models": {}}
            for name in ("h9", "old8"):
                rr = [r["models"][name] for r in rows]
                sol[key]["models"][name] = {"fov_pass": rate(sum(r["fov_pass"] for r in rr), len(rr)),
                    "fov_margin_001_pass": rate(sum(r["fov_margin_001_pass"] for r in rr), len(rr)),
                    "predicted_bracket": rate(sum(r["predicted_bracket_found"] for r in rr), len(rr)),
                    "predicted_root_002": rate(sum(r["predicted_root_within_002"] for r in rr), len(rr)),
                    "zero_abs_g_lt_001": rate(sum(abs(r["zero_g_m"]) < .001 for r in rr), len(rr)),
                    "candidate_g_m": distribution(r["candidate_g_m"] for r in rr),
                    "solver_ms": distribution(r["solver_ms"] for r in rr),
                    "value_calls": distribution(r["value_calls"] for r in rr),
                    "gradient_calls": distribution(r["gradient_calls"] for r in rr),
                    "joint_clamp_events": distribution(r["joint_clamp_events"] for r in rr),
                    "root_sources": dict(Counter(r["root_source"] for r in rr)),
                    "failure_stages": dict(Counter(r["failure_stage"] for r in rr))}
            sol[key]["paired_fov_outcomes"] = dict(Counter(
                ("both_pass" if r["models"]["h9"]["fov_pass"] and r["models"]["old8"]["fov_pass"] else
                 "h9_only" if r["models"]["h9"]["fov_pass"] else
                 "old8_only" if r["models"]["old8"]["fov_pass"] else "both_fail") for r in rows))
            ar = [r["analytic_control"] for r in rows if "analytic_control" in r]
            sol[key]["analytic_control"] = {"fov_pass": rate(sum(r["fov_pass"] for r in ar), len(ar)),
                "solver_ms": distribution(r["solver_ms"] for r in ar),
                "warning": "different objective/budget; no infeasibility inference"}
        return {"boundary": b, "profiles": p, "solves": sol,
                "excluded_starts": dict(self.excluded_starts), "generation_counts": dict(self.generation_counts)}


def boundary_record(oracle, fields, x, q, s, bank, mask, lo, hi, cfg, identity):
    qual = qualify_boundary(oracle, x, q, s, mask, lo, hi, cfg)
    label = bank_label(q, bank, mask, 1. if qual["g_m"] >= 0 else -1.)
    if identity["origin"] == "offbank_refined" and label["distance_rad"] < cfg.offbank_min_rad:
        qual["regular"] = False
        qual["reasons"].append("offbank_too_close_to_original_bank")
    n = qual["normal"] if qual["regular"] else None
    models = {name: field_metrics(*field.value_grad(x, q, s), mask, n)
              for name, field in fields.items()}
    return {**identity, **qual, "q": q.tolist(), "x": x.tolist(),
            "bank_distance_rad": label["distance_rad"],
            "legacy_signed_target": label["legacy_signed_value"],
            "bank_nearest_gap_rad": label["nearest_gap_rad"], "models": models}


def normal_profile(oracle, fields, x, q0, normal, s, bank, mask, lo, hi, cfg, identity):
    rows = []
    g0, dg0 = oracle.value_grad(x, q0, s)
    plane0 = int(oracle.planes(x, q0.reshape(1, 7), s)[0].argmin())
    for radius in cfg.offsets_rad:
        for side in (-1, 1):
            t = side * radius
            q = q0 + t * normal
            row = {**identity, "offset_rad": t, "q": q.tolist(), "in_limits": within(q, lo, hi)}
            if row["in_limits"]:
                g = oracle.value(x, q, s)
                label = bank_label(q, bank, mask, 1. if g >= 0 else -1.)
                row.update(g_m=g, expected_sign_ok=(g*side > cfg.root_tol_m),
                    same_active_plane=int(oracle.planes(x, q.reshape(1, 7), s)[0].argmin()) == plane0,
                    bank_distance_rad=label["distance_rad"],
                    legacy_signed_target=label["legacy_signed_value"],
                    bank_distance_over_abs_offset=label["distance_rad"]/abs(t),
                    bank_normal_cosine=cosine(label["gradient"], normal) if label["gradient_valid"] else None,
                    bank_gradient_valid=label["gradient_valid"],
                    linearization_abs_error_m=abs(g-(g0+t*float(torch.dot(dg0, normal)))), models={})
                for name, field in fields.items():
                    v, grad = field.value_grad(x, q, s)
                    row["models"][name] = {**field_metrics(v, grad, mask, normal),
                                            "sign_correct": (v >= 0) == (g >= 0)}
            else:
                row["skip_reason"] = "offset_out_of_bounds_not_clamped"
            rows.append(row)
    return rows


def verify_runtime_parity(probes, x, q):
    """Compare instrumented adapter to the imported unmodified optimizer methods."""
    from per_sensor_visibility_runtime import PerSensorVisibilityRuntime as Base
    checked = 0
    for probe in probes.values():
        plain = object.__new__(Base)
        plain.__dict__.update(probe.__dict__)
        for s in (0, 7):
            a = probe._optimize_branch(x.reshape(1, 3), q.reshape(1, 7).clone(), s)
            b = plain._optimize_branch(x.reshape(1, 3), q.reshape(1, 7).clone(), s)
            for key in ("q_zero", "q_candidate", "final_score", "f_zero"):
                if not np.allclose(a[key], b[key], rtol=0, atol=1e-6, equal_nan=True):
                    raise RuntimeError(f"Runtime adapter parity failed: {s} {key}")
            if a["root_source"] != b["root_source"]: raise RuntimeError("Root source mismatch")
            checked += 1
    return {"status": "PASS", "comparisons": checked}


def preflight(dataset, oracle, probes, device):
    from train_signed_visibility_cdf_pairwise_replace import decode_per_sensor_distance_and_grad
    from per_sensor_visibility_runtime import PerSensorVisibilityRuntime as Base
    lo, hi = dataset.q_limits(device)
    rng = np.random.default_rng(314159)
    ids = dataset.val_indices_cpu[:2]
    qs = torch.tensor(rng.uniform(lo.cpu().numpy(), hi.cpu().numpy(), (8, 7)), device=device, dtype=torch.float32)
    parity = oracle.verify_against_upstream(dataset.x_cpu[ids[0]].to(device), qs)
    masks = dataset.sensor_masks(device)
    derived = torch.zeros_like(masks)
    for s, chain in enumerate(oracle.specs):
        for spec in chain:
            if spec["q_index"] >= 0 and spec["type"] != "fixed":
                derived[s, spec["q_index"]] = 1
    if not torch.equal(masks, derived):
        raise ValueError("Dataset sensor chain masks do not match actual reference FK chains")
    lib, valid = dataset.qlib_cpu[ids].to(device), dataset.valid_cpu[ids].to(device)
    ds, gs, has = decode_per_sensor_distance_and_grad(lib, valid, qs, masks, x_chunk=2)
    err = 0.
    for i in range(len(ids)):
        for s in range(8):
            if not has[i, s]: continue
            bank = lib[i, :, :, s][valid[i, :, s]]
            for j, q in enumerate(qs):
                ours = bank_label(q, bank, masks[s], 1.)
                err = max(err, abs(ours["legacy_signed_value"]-float(ds[i, j, s])),
                          float((ours["gradient"]-gs[i, j, s]).abs().max()))
    if err > 2e-5: raise RuntimeError(f"Legacy label parity failed: {err}")
    methods = ("_clamp", "_projection_step", "_ascent_step", "_refine_branch_root", "_optimize_branch")
    result = {"oracle": parity, "sensor_chain_masks": {"status": "PASS"}, "bank_label": {"status": "PASS", "max_abs_error": err},
        "solver_method_sha256": {k: hashlib.sha256(inspect.getsource(getattr(Base, k)).encode()).hexdigest()
                                  for k in methods}}
    if probes:
        result["runtime_probe"] = verify_runtime_parity(probes, dataset.x_cpu[ids[0]].to(device), qs[0])
    return result


def markdown(report):
    lines = ["# Mainline B: boundary and per-sensor solve audit", "",
        "Read-only diagnostics. FOV-only; LOS/GCDF/VBC/trajectory/tracking/seen = NOT_RUN.",
        "Off-bank tangent-refined samples are NOT independent uniform samples or nearest-boundary solutions.",
        "Root tolerance/normal exclusions are diagnostic settings, not modified runtime safety thresholds.", "",
        "## Boundary coverage and geometry (conditional on regular accepted boundaries)",
        "| Group | attempts | regular | bank gap median(rad) | H9 abs(f) mean | old8 abs(f) mean | H9 raw normal cos | old8 raw normal cos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(v): return "N/A" if v is None else f"{v:.5f}"
    for k, r in sorted(report["boundary"].items()):
        h, o = r["models"]["h9"], r["models"]["old8"]
        lines.append(f"| {k} | {r['attempts']} | {r['regular']} | " + " | ".join(fmt(v) for v in
            (r["bank_distance_rad"]["p50"], h["abs_value"]["mean"], o["abs_value"]["mean"],
             h["raw_normal_cosine"]["mean"], o["raw_normal_cosine"]["mean"])) + " |")
    lines += ["", "## Matched outside-start per-sensor solver (actual imported runtime optimizer)",
        "| Group | starts | H9 FOV pass | old8 FOV pass | H9-only | old8-only | H9 median ms | old8 median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in sorted(report["solves"].items()):
        h, o = r["models"]["h9"], r["models"]["old8"]
        pair = r["paired_fov_outcomes"]
        lines.append(f"| {k} | {r['paired_starts']} | {fmt(h['fov_pass']['rate'])} | {fmt(o['fov_pass']['rate'])} | "
            f"{pair.get('h9_only',0)} | {pair.get('old8_only',0)} | {fmt(h['solver_ms']['p50'])} | {fmt(o['solver_ms']['p50'])} |")
    lines += ["", "See report.json for conditional denominators, missing values and all profile/radius statistics.",
        "generation.jsonl records every generation attempt/rejection; no resampling to hide failures.",
        "boundary.jsonl / profiles.jsonl / solves.jsonl preserve exact x,s,q and solver histories.",
        "Learned initial-positive and best-effort q_zero are NOT evidence of a found root.",
        "Analytic ascent control has a different objective/budget; failure is NOT proof of infeasibility.",
        "This tool does not decide to train V2, switch runtime, or certify a trajectory."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", default=str(REPO))
    p.add_argument("--data", default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz")
    p.add_argument("--h9", default="src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt")
    p.add_argument("--old8", default="src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--stage", choices=("all", "boundary"), default="all")
    p.add_argument("--points-per-sensor", type=int, default=64)
    p.add_argument("--anchors-per-point", type=int, default=2)
    p.add_argument("--random-starts-per-point", type=int, default=2)
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--analytic-control", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    for key in ("points_per_sensor", "anchors_per_point", "random_starts_per_point"):
        if getattr(args, key) < 1: p.error(f"{key} must be positive")
    if args.seed < 0: p.error("seed must be nonnegative")
    root = Path(args.artifact_root).expanduser().resolve()
    for key in ("data", "h9", "old8"):
        v = Path(getattr(args, key)).expanduser()
        setattr(args, key, str((root/v).resolve() if not v.is_absolute() else v.resolve()))
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing nonempty output: {out}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("No allocated GPU; use Slurm")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    for path, expected in ((args.h9, V1), (args.old8, OLD8)):
        if Path(path).name != "final.pt" or sha(path) != expected:
            raise ValueError(f"Not the fixed evaluated V1/baseline final.pt: {path}")
    sys.path.insert(0, str(REPO/"src/care_visibility_cdf/scripts"))
    sys.path.insert(0, str(REPO/"experiments/hierarchical9_scratch_v1"))
    import evaluate as ev
    from train_signed_visibility_cdf_pairwise_replace import VisibilityQ0Dataset, DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES
    new, nm = ev.load_hierarchical(args.h9, device)
    old, om = ev.load_legacy(args.old8, 8, device)
    models = {"h9": ev.HeadView(new, "sensors"), "old8": old}
    fields = {k: SensorField(v) for k, v in models.items()}
    urdf = REPO/"src/arm_description/urdf/Arm.urdf"
    if nm["training_metadata"].get("urdf_sha256") != sha(urdf): raise ValueError("URDF differs from V1 training")
    if nm["training_metadata"].get("data_bytes") != Path(args.data).stat().st_size:
        raise ValueError("Dataset size differs from training (size is not a content hash)")
    dataset = VisibilityQ0Dataset(args.data, val_count=1000, seed=0)
    if len(dataset.val_indices_cpu) != 1000 or dataset.S != 8 or dataset.J != 7:
        raise ValueError("Unexpected dataset/split")
    lo, hi = dataset.q_limits(device)
    masks = dataset.sensor_masks(device)
    oracle = SensorOracle(urdf, device, DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES)
    cfg = Config(); cfg.validate()
    probes = {k: make_probe(v, masks, lo, hi) for k, v in models.items()} if args.stage == "all" else {}
    checks = preflight(dataset, oracle, probes, device)
    print("[preflight] " + json.dumps(checks), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    source_paths = list(Path(__file__).parent.glob("*.py")) + [
        REPO/"src/care_visibility_cdf/scripts"/f for f in (
        "per_sensor_visibility_runtime.py", "extract_visibility_zero_level_sets.py",
        "train_signed_visibility_cdf_pairwise_replace.py", "train_per_sensor_visibility_cdf.py")]
    source_paths += [REPO/"experiments/hierarchical9_scratch_v1"/f for f in ("evaluate.py", "model.py", "objective.py")]
    manifest = {"status": "PREFLIGHT_ONLY" if args.preflight_only else "RUNNING", "args": vars(args),
        "boundary_config": asdict(cfg), "checkpoints": {"h9": nm, "old8": om}, "preflight": checks,
        "source_commit": git.stdout.strip(), "reviewed_base": BASE,
        "source_sha256": {str(v.relative_to(REPO)): sha(v) for v in source_paths},
        "urdf_sha256": sha(urdf), "data_identity": {"path": args.data, "bytes": Path(args.data).stat().st_size,
             "mtime_ns": Path(args.data).stat().st_mtime_ns, "content_hash": "NOT_COMPUTED"},
        "split": {"seed": 0, "validation_points": dataset.val_indices_cpu.tolist(),
                  "is_independent_final_test": False},
        "protocol": {"field_precision": "FP32, no AMP/TF32", "label_distance_floor_rad": .0001,
            "offbank": "one tangent perturbation of refined bank root, analytic re-refinement, min bank distance gate",
            "normal_profile_reference": "anchor normal; offsets are local probes, not exact global signed distance",
            "local_solver_offsets_rad": [.02, .05], "solver_anchors_per_point": 1,
            "projection": [10, .5, .03, .25], "root": [12, .002], "ascent": [1, .05, .25],
            "root_not_found_best_effort_ascent_steps": 8, "gradient_chain_mask": True,
            "analytic_control": "18 normalized steps, .05 rad; not equal-objective/equal-time comparison",
            "self_occlusion": "NOT_RUN", "collision_path_safety": "NOT_RUN", "runtime_switch": False},
        "environment": {"torch": str(torch.__version__), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    write_json(out/"manifest.json", manifest)
    if args.preflight_only:
        print("[done] boundary_audit_preflight_only", flush=True); return
    res, started, solve_index = Results(), time.perf_counter(), 0
    handles = {n: (out/(n+".jsonl")).open("w", encoding="utf-8") for n in ("generation", "boundary", "profiles", "solves")}
    try:
        def generation(row):
            key = f"S{row.get('sensor', '?')}/{row.get('kind')}/{row.get('reason', 'unspecified')}"
            res.generation_counts[key] += 1
            emit(handles["generation"], row)

        def solve_start(x, q, s, mask, identity, witness=None):
            nonlocal solve_index
            g = oracle.value(x, q, s)
            if not within(q, lo, hi) or g >= -cfg.root_tol_m:
                reason = "out_of_bounds" if not within(q, lo, hi) else "not_strictly_outside"
                res.excluded_starts[f"S{s}/{identity['cohort']}/{reason}"] += 1
                generation({**identity, "kind": "solver_start_excluded", "reason": reason,
                      "q": q.tolist(), "x": x.tolist(), "g_m": g})
                return
            row = {**identity, "q_init": q.tolist(), "x": x.tolist(), "initial_g_m": g,
                   "positive_side_witness_q": witness, "models": {}}
            # Alternate model timing order, never alter the paired starting q.
            names = ("h9", "old8") if solve_index % 2 == 0 else ("old8", "h9")
            solve_index += 1
            for name in names: row["models"][name] = run_probe(probes[name], oracle, x, q, s)
            if args.analytic_control: row["analytic_control"] = analytic_ascent(oracle, x, q, s, mask, lo, hi)
            emit(handles["solves"], row)
            # Keep histories on disk, not duplicated in RAM summaries.
            compact = {**row, "models": {k: {kk: vv for kk, vv in r.items() if not kk.endswith("history")}
                                          for k, r in row["models"].items()}}
            if "analytic_control" in compact:
                compact["analytic_control"] = {k: v for k, v in compact["analytic_control"].items() if k != "history"}
            res.solves[f"S{s}/{identity['cohort']}"].append(compact)

        def inspect_anchor(x, q, s, bank, mask, identity, solve_this):
            row = boundary_record(oracle, fields, x, q, s, bank, mask, lo, hi, cfg, identity)
            emit(handles["boundary"], row)
            res.boundaries[f"S{s}/{identity['origin']}"].append(row)
            if not row["regular"] or identity["origin"] == "bank_raw": return row
            profile = normal_profile(oracle, fields, x, q, row["normal"], s, bank, mask, lo, hi, cfg, identity)
            for r in profile:
                emit(handles["profiles"], r)
                res.profiles[f"S{s}/{identity['origin']}/offset={r['offset_rad']:+g}"].append(r)
            if probes and solve_this:
                for radius in (.02, .05):
                    neg = next(r for r in profile if r["offset_rad"] == -radius)
                    pos = next(r for r in profile if r["offset_rad"] == radius)
                    ident = {**identity, "cohort": f"local_{identity['origin']}_r{radius:g}"}
                    if not all(r.get("in_limits") and r.get("expected_sign_ok") for r in (neg, pos)):
                        res.excluded_starts[f"S{s}/{ident['cohort']}/no_two_sided_witness"] += 1
                        generation({**ident, "kind": "solver_start_excluded", "reason": "no_two_sided_witness"})
                        continue
                    solve_start(x, torch.tensor(neg["q"], device=device, dtype=torch.float32), s, mask, ident, pos["q"])
            return row

        # These draws do not depend on model construction, timings or success.
        val = dataset.val_indices_cpu.numpy()
        for s in range(8):
            # Materialize empty groups, so missing boundary coverage is explicit.
            for origin in ("bank_raw", "bank_refined", "offbank_refined"):
                res.boundaries[f"S{s}/{origin}"]
                if origin != "bank_raw":
                    for radius in cfg.offsets_rad:
                        for side in (-1, 1):
                            res.profiles[f"S{s}/{origin}/offset={side*radius:+g}"]
            if probes:
                for cohort in ("uniform_outside", "local_bank_refined_r0.02", "local_bank_refined_r0.05",
                               "local_offbank_refined_r0.02", "local_offbank_refined_r0.05"):
                    res.solves[f"S{s}/{cohort}"]
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, s]))
            available = dataset.valid_cpu[dataset.val_indices_cpu, :, s].any(dim=1).numpy()
            pool = val[available]
            if not len(pool): raise RuntimeError(f"S{s}: no held-out q0 support")
            ids = rng.choice(pool, size=min(args.points_per_sensor, len(pool)), replace=False)
            for pi, idx in enumerate(ids):
                point_rng = np.random.default_rng(np.random.SeedSequence([args.seed, s, int(idx)]))
                x = dataset.x_cpu[int(idx)].to(device)
                valid = dataset.valid_cpu[int(idx), :, s]
                bank = dataset.qlib_cpu[int(idx), :, :, s][valid].to(device)
                if not torch.isfinite(bank).all(): raise ValueError(f"Non-finite valid bank: x={idx}, S{s}")
                mask = masks[s]
                slots = point_rng.choice(len(bank), size=min(args.anchors_per_point, len(bank)), replace=False)
                original_slots = torch.where(valid)[0].tolist()
                for ai, slot in enumerate(slots):
                    identity = {"sensor": s, "x_index": int(idx), "anchor_index": ai,
                                "library_slot": original_slots[int(slot)], "origin": "bank_raw"}
                    raw = bank[int(slot)].clone()
                    inspect_anchor(x, raw, s, bank, mask, identity, False)
                    refined = refine_boundary(lambda q: oracle.value_grad(x, q, s), raw, mask, lo, hi, cfg)
                    generation({**identity, "kind": "bank_refinement", **refined})
                    identity["origin"] = "bank_refined"
                    row = inspect_anchor(x, refined["q"], s, bank, mask, identity, ai == 0)
                    if not row["regular"]:
                        generation({**identity, "kind": "offbank_generation_skipped", "reason": "parent_not_regular"})
                        continue
                    seed = tangent_seed(refined["q"], row["normal"], mask, point_rng, cfg.tangent_radius_rad)
                    identity = {**identity, "origin": "offbank_refined"}
                    if seed is None:
                        generation({**identity, "kind": "offbank_generation_skipped", "reason": "degenerate_tangent"})
                        continue
                    off = refine_boundary(lambda q: oracle.value_grad(x, q, s), seed, mask, lo, hi, cfg)
                    generation({**identity, "kind": "offbank_refinement", "seed_q": seed.tolist(), **off})
                    inspect_anchor(x, off["q"], s, bank, mask, identity, ai == 0)
                # Separate RNG prevents different boundary failures changing random starts.
                random_rng = np.random.default_rng(np.random.SeedSequence([args.seed, s, int(idx), 999]))
                qs = random_rng.uniform(lo.cpu().numpy(), hi.cpu().numpy(), (args.random_starts_per_point, 7))
                if probes:
                    for ri, qr in enumerate(qs):
                        solve_start(x, torch.tensor(qr, device=device, dtype=torch.float32), s, mask,
                                    {"sensor": s, "x_index": int(idx), "random_index": ri, "cohort": "uniform_outside"})
                print(f"[audit] S{s} point={pi+1}/{len(ids)} x_index={idx} paired_solves={solve_index}", flush=True)
            write_json(out/"partial_report.json", res.summary())
        report = {"status": "COMPLETE", "elapsed_seconds": time.perf_counter()-started, **res.summary()}
        write_json(out/"report.json", report)
        (out/"summary.md").write_text(markdown(report), encoding="utf-8")
        manifest["status"] = "COMPLETE"
        manifest["elapsed_seconds"] = report["elapsed_seconds"]
        write_json(out/"manifest.json", manifest)
        print(f"[done] mainline_b_boundary_audit_complete output={out}", flush=True)
    except BaseException as exc:
        manifest.update(status="FAILED", failure_type=type(exc).__name__, failure=str(exc))
        write_json(out/"manifest.json", manifest)
        write_json(out/"partial_report.json", res.summary())
        raise
    finally:
        for file in handles.values(): file.close()


if __name__ == "__main__":
    main()
