"""Merge complete, checksum-guarded shards. Report missing/nonfinite diagnostics."""
from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
from common import checked_starts, cohort, read_json, sha, stats, write_json


def load_rows(out: Path, stem: str, world: int, n: int):
    rows = []
    for rank in range(world):
        path = out / f"{stem}.rank{rank}.jsonl"
        rows += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows.sort(key=lambda x: x["case_id"])
    if [r["case_id"] for r in rows] != list(range(n)):
        raise ValueError(f"Incomplete or duplicate {stem} cases")
    return rows


def exact_p(a: int, b: int) -> float:
    # Python integer division handles the huge binomial integer without float overflow.
    n = a+b
    if n == 0:
        return 1.0
    return min(1., (2*sum(math.comb(n, i) for i in range(min(a, b)+1)))/(2**n))


def paired(rows, candidate):
    counts = Counter()
    for r in rows:
        c = bool(r["models"][candidate]["fov_pass"])
        b = bool(r["models"]["R1"]["fov_pass"])
        counts["both_pass" if c and b else "candidate_only" if c else "R1_only" if b else "both_fail"] += 1
    a, b = counts["candidate_only"], counts["R1_only"]
    return dict(counts, net=a-b, paired_sign_p_descriptive=exact_p(a, b))


def nested_values(rows, name, keys):
    result = []
    for row in rows:
        value = row["models"][name]
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        result.append(float(value) if value is not None else float("nan"))
    return result


def boundary_summary(rows, names):
    report = {}
    for sp, split in ((0, "train"), (1, "val")):
        report[split] = {}
        for s in range(8):
            group = [r for r in rows if r["split"] == sp and r["sensor"] == s]
            entry = {"count": len(group), "geometry_flags": dict(Counter(f for r in group for f in r["geometry"]["flags"])), "models": {}}
            for name in names:
                entry["models"][name] = {
                    key: stats(nested_values(group, name, [key]))
                    for key in ("pair_slope", "pair_slope_abs_error", "pair_both_sign_correct",
                                "boundary_cosine", "boundary_grad_norm", "boundary_inactive_grad_norm",
                                "midpoint_bias", "side_mean_squared_error_rad2")}
                entry["models"][name]["boundary_zero_abs"] = stats(np.abs(nested_values(group, name, ["f_zero"])))
                entry["models"][name]["direction_delta_g_m"] = stats(nested_values(group, name, ["direction_probe", "delta_g_m"]))
                entry["models"][name]["direction_status"] = dict(Counter(r["models"][name]["direction_probe"]["status"] for r in group))
                entry["models"][name]["nonfinite_rows"] = sum(r["models"][name]["nonfinite"] for r in group)
            report[split][f"S{s}"] = entry
    return report


def merge(out: Path):
    job = read_json(out / "job.json")
    cfg = job["protocol"]
    selection = read_json(out / "selection.json")
    if sha(out/"selected.npz") != selection["selected_sha256"]:
        raise ValueError("Frozen boundary sample plan changed")
    # Completion markers also verify that ranks consumed the same inputs and did not mutate weights.
    for rank in range(cfg["world"]):
        marker = read_json(out/f"rank{rank}.complete.json")
        if marker["status"] != "COMPLETE" or marker["selected_sha256"] != selection["selected_sha256"]:
            raise ValueError("Incomplete/inconsistent rank")
        for name, digest in marker["files"].items():
            if sha(out/name) != digest:
                raise ValueError(f"Rank result changed: {name}")
    solver = load_rows(out, "solves", cfg["world"], cfg["solver_cases"])
    boundary = load_rows(out, "boundary", cfg["world"], selection["selected_pairs"])
    names = list(solver[0]["models"])
    expected_models = set(job["aliases"].values())
    starts = checked_starts(Path(job["paths"]["starts"]), cfg)
    for row, spec in zip(solver, starts):
        if set(row["models"]) != expected_models or any(row.get(k) != v for k, v in spec.items() if k not in ("case_id", "models")):
            raise ValueError("Solver shard no longer matches its frozen original start/models")
    cohorts, by_sensor = {}, {}
    for c in ("local", "uniform", "all"):
        rows = solver if c == "all" else [r for r in solver if cohort(r) == c]
        cohorts[c] = {"count": len(rows), "models": {}, "paired": {}}
        for name in names:
            passed = sum(bool(r["models"][name]["fov_pass"]) for r in rows)
            cohorts[c]["models"][name] = {"passed": passed, "rate": passed/len(rows),
                "failure_stages": dict(Counter(r["models"][name]["failure_stage"] for r in rows)),
                "solver_ms": stats(nested_values(rows, name, ["solver_ms"])),
                "gradient_calls": stats(nested_values(rows, name, ["gradient_calls"]))}
            if name != "R1":
                cohorts[c]["paired"][name] = paired(rows, name)
    for s in range(8):
        rows = [r for r in solver if int(r["sensor"]) == s]
        by_sensor[f"S{s}"] = {"count": len(rows), "passed": {
            name: sum(bool(r["models"][name]["fov_pass"]) for r in rows) for name in names}}
    expected = {"local": 618, "uniform": 673, "all": 1291}
    baseline_ok = all(cohorts[c]["models"]["R1"]["passed"] == n for c, n in expected.items())
    geometry_flags = dict(Counter(f for r in boundary for f in r["geometry"]["flags"]))
    mapping = [read_json(out/f"mapping.rank{r}.json") for r in range(cfg["world"])]
    if any(not m["workspace_index_arrays_equal"] for m in mapping):
        geometry_flags["WORKSPACE_INDEX_MAPPING_REVIEW"] = sum(not m["workspace_index_arrays_equal"] for m in mapping)
    grad_reports = [read_json(out/f"gradients.rank{r}.json") for r in range(cfg["world"])]
    activation = [read_json(out/f"activations.rank{r}.json") for r in range(cfg["world"])]
    report = {
        "status": "COMPLETE" if baseline_ok else "BASELINE_MISMATCH",
        "scientific_status": "DATA_REVIEW_REQUIRED" if geometry_flags else "REVIEW_REQUIRED",
        "baseline_reproduced": baseline_ok, "models": job["models"], "model_aliases": job["aliases"],
        "starts_sha256": cfg["starts_sha256"], "solver_settings": cfg["solver_settings"],
        "cohorts": cohorts, "per_sensor_solver": by_sensor,
        "boundary_sample_plan": selection, "geometry_flags": geometry_flags,
        "boundary": boundary_summary(boundary, names),
        "gradient_diagnostics": grad_reports, "activation_diagnostics": activation, "mapping_checks": mapping,
        "limitations": ["Frozen starts are a reused regression set, not an untouched generalization test.",
            "Paired sign-test p-values do not correct for multiple starts sharing one spatial x.",
            "Boundary checks use a prespecified subset of original eligible V4 pairs; no global distance certificate.",
            "Both checkpoints are evaluated with CURRENT PS losses for gradient diagnosis, not R1 historical training loss.",
            "Raw gradient conflict and sampled ReLU activity are diagnostics, not proof of causality or Adam update effects.",
            "No training, LOS, collision, trajectory, GCDF/VBC or actual-seen certification. No production promotion."],
    }
    write_json(out/"report.json", report)
    lines = ["# R1 vs Paired-Slope: combined read-only audit", "",
             f"Status: {report['status']}; scientific review: {report['scientific_status']}",
             f"Starts: {cfg['solver_cases']}; SHA256: {cfg['starts_sha256']}",
             f"Checkpoint aliases after exact state comparison: {json.dumps(job['aliases'])}", "",
             "## Fixed solver (identical epsilon=0.03; no threshold sweep)", "",
             "| Cohort | N | " + " | ".join(names) + " |",
             "|---|---:|" + "---:|"*len(names)]
    for c, data in cohorts.items():
        lines.append(f"| {c} | {data['count']} | " + " | ".join(f"{data['models'][n]['rate']:.5f}" for n in names) + " |")
    lines += ["", "## Paired wins/losses (all frozen starts)"]
    for n, x in cohorts["all"]["paired"].items():
        lines.append(f"- {n}: new-only={x.get('candidate_only',0)}, R1-only={x.get('R1_only',0)}, net={x['net']:+d}, descriptive p={x['paired_sign_p_descriptive']:.6g}")
    lines += ["", "## Independent geometry flags", json.dumps(geometry_flags, sort_keys=True), "",
              "## Same-point boundary diagnostics: validation subset", "",
              "| Sensor | Model | slope MAE | both signs | normal cosine | zero MAE | delta analytic g |",
              "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return "N/A" if v is None else f"{v:.6g}"
    for s, item in report["boundary"]["val"].items():
        for name, v in item["models"].items():
            vals = [v[k]["mean"] for k in ("pair_slope_abs_error", "pair_both_sign_correct", "boundary_cosine", "boundary_zero_abs", "direction_delta_g_m")]
            lines.append(f"| {s} | {name} | " + " | ".join(fmt(x) for x in vals) + " |")
    lines += ["", "## Gradient diagnostics (4 exact global training-batch reconstructions)",
              "See report.json / gradient_diagnostics: each component x shared/union/S0..S7 norms and cosines; no optimizer step.",
              "Component gradients must sum to the unmodified objective gradient (checked per model/replica).",
              "", "## Limits"] + ["- " + x for x in report["limitations"]]
    (out/"summary.md").write_text("\n".join(lines)+"\n")
    print((out/"summary.md").read_text(), flush=True)
    if not baseline_ok:
        raise ValueError("R1 historical solver counts not reproduced; results retained, no false COMPLETE")
    return report
