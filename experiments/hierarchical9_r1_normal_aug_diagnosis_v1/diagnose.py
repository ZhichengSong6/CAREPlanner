#!/usr/bin/env python3
"""Read-only retrospective audit: original R1 vs controlled R1+normal augment.

No solver calls, no optimizer, no modification of checkpoints/data/results.
Every case must come from the same 1963 frozen starts and reproduce the
already completed equal-solver comparison. Labels are DESCRIPTIVE, not causal.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

EXPECTED_STARTS = "0e4711fc21628fdbb341191ec634415781696800a2386a4ad48e592c34f3198c"
EXPECTED_R1 = "4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002"
VARIANT = "care_h9_r1_boundary_direction_aug_v1"
ARMS = ("R1", "R1_AUG")
EXPECTED_COUNTS = {"local": 683, "uniform": 1280, "all": 1963}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        for number, text in enumerate(f, 1):
            if text.strip():
                try:
                    yield json.loads(text)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def clean(value):
    """Nonfinite statistics must be null, never silently coerced to zero."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(path, item):
    path = Path(path)
    with path.open("x", encoding="utf-8") as f:
        json.dump(clean(item), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_stats(values):
    v = sorted(float(x) for x in values if finite(x))
    if not v:
        return {"n": len(values), "valid": 0, "mean": None, "median": None, "p10": None, "p90": None}
    def quantile(p):
        z = (len(v) - 1) * p
        i = int(z)
        return v[i] * (1 - (z - i)) + v[min(i + 1, len(v) - 1)] * (z - i)
    return {"n": len(values), "valid": len(v), "mean": statistics.fmean(v),
            "median": quantile(.5), "p10": quantile(.1), "p90": quantile(.9)}


def l2(x, y):
    if (not isinstance(x, list) or not isinstance(y, list) or
            len(x) != 7 or len(y) != 7 or not all(finite(a) for a in x+y)):
        return None
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(x, y)))


def cohort(group):
    if "/local_" in group:
        return "local"
    if group.endswith("uniform_outside"):
        return "uniform"
    raise ValueError(f"Unknown frozen start cohort: {group!r}")


def verify_inputs(a):
    r1 = a.r012 / "formal/R1"
    aug = a.aug / "formal"
    solver = a.solver
    starts = a.starts
    if sha(starts) != EXPECTED_STARTS:
        raise ValueError("Frozen starts SHA mismatch")
    sr = read_json(starts.parent/"manifest.json")
    if sr.get("starts_sha256") != EXPECTED_STARTS:
        raise ValueError("Frozen starts source manifest mismatch")
    frozen = list(read_jsonl(starts))
    if len(frozen) != 1963:
        raise ValueError("Expected exactly 1963 starts")
    old_run = read_json(r1/"run.json")
    new_run = read_json(aug/"run.json")
    if sha(r1/"final.pt") != EXPECTED_R1 or old_run.get("final_sha256") != EXPECTED_R1:
        raise ValueError("Original R1 checkpoint identity mismatch")
    if (old_run.get("status") != "COMPLETE" or old_run.get("arm") != "R1" or
        old_run.get("successful_updates") != 50000 or
        old_run.get("training_stream_updates") != 50000):
        raise ValueError("Original R1 run incomplete")
    if (new_run.get("status") != "COMPLETE" or
        new_run.get("variant") != VARIANT or new_run.get("successful_updates") != 50000 or
        new_run.get("baseline_stream_exact_match") is not True or
        new_run.get("training_stream_sha256") != old_run.get("training_stream_sha256") or
        new_run.get("original_r1_final_sha256") != EXPECTED_R1 or
        sha(aug/"final.pt") != new_run.get("final_sha256")):
        raise ValueError("Augmented training/checkpoint/stream mismatch")
    finished = read_json(solver/"complete.json")
    recorded = read_json(solver/"report.json")
    mf = read_json(solver/"manifest.json")
    if (finished.get("status") != "COMPLETE" or
        int(finished.get("rows", -1)) != 1963 or
        recorded.get("status") != "COMPLETE" or
        recorded.get("starts_sha256") != EXPECTED_STARTS or
        int(recorded.get("count", -1)) != 1963 or
        recorded.get("eps_f") != .03 or
        mf.get("starts_sha256") != EXPECTED_STARTS or
        mf.get("r1_sha256") != EXPECTED_R1 or
        mf.get("aug_sha256") != new_run.get("final_sha256")):
        raise ValueError("Fixed solver report/provenance mismatch")

    records = []
    for rank in range(4):
        file = solver/f"solves.rank{rank}.jsonl"
        for entry in read_jsonl(file):
            records.append(entry)
    records.sort(key=lambda row: row["case_id"])
    if [row["case_id"] for row in records] != list(range(1963)):
        raise ValueError("Duplicate or missing solver cases")

    for idx, (got, want) in enumerate(zip(records, frozen)):
        for key in ("x", "q_init", "sensor", "group"):
            if got.get(key) != want.get(key):
                raise ValueError(f"Frozen case {idx} mismatch: {key}")
        if got["case_id"] != idx or not isinstance(got["sensor"], int) or not 0 <= got["sensor"] < 8:
            raise ValueError(f"Invalid case/sensor: {idx}")
        for arm in ARMS:
            info = got.get("models", {}).get(arm)
            if info is None:
                raise ValueError(f"Missing {arm} solver result at case {idx}")
            if not isinstance(info.get("fov_pass"), bool):
                raise ValueError(f"{idx}: invalid fov_pass type {arm}")
            margin = info.get("candidate_g_m")
            limits = info.get("candidate_in_limits")
            if not finite(margin) or not isinstance(limits, bool):
                raise ValueError(f"{idx}: nonfinite/invalid candidate geometry {arm}")
            if info["fov_pass"] != (limits and margin >= 0):
                raise ValueError(f"{idx}: analytic FOV pass inconsistent {arm}")
            if not isinstance(info.get("root_source"), str):
                raise ValueError(f"{idx}: missing root source {arm}")
            if not isinstance(info.get("failure_stage"), str):
                raise ValueError(f"{idx}: missing failure stage {arm}")
        one, two = (got["models"][name].get("initial_g_m") for name in ARMS)
        if not finite(one) or not finite(two) or abs(float(one) - float(two)) > 1e-5:
            raise ValueError(f"{idx}: paired start analytic g disagreement")

    for group in ("local", "uniform"):
        total = sum(cohort(row["group"]) == group for row in records)
        if total != EXPECTED_COUNTS[group]:
            raise ValueError(f"Unexpected {group} cases: {total}")
    counts = {group:{arm:sum(row["models"][arm]["fov_pass"] for row in records
                    if group == "all" or cohort(row["group"]) == group)
                     for arm in ARMS}
              for group in ("local", "uniform", "all")}
    if [counts["local"]["R1"], counts["uniform"]["R1"], counts["all"]["R1"]] != [618, 673, 1291]:
        raise ValueError("Historical R1 FOV solver baseline not reproduced")
    for group in counts:
        for arm in ARMS:
            official = recorded["aggregate"][group][arm]
            if counts[group][arm] != official["pass"]:
                raise ValueError(f"Official report vs raw cases differ: {group}/{arm}")
    gain = sum(not row["models"]["R1"]["fov_pass"] and row["models"]["R1_AUG"]["fov_pass"] for row in records)
    loss = sum(row["models"]["R1"]["fov_pass"] and not row["models"]["R1_AUG"]["fov_pass"] for row in records)
    if (recorded.get("paired") != {"aug_only": gain, "R1_only": loss, "net_gain": gain-loss}):
        raise ValueError("Official paired comparison differs from raw solver cases")
    return {"frozen": frozen, "rows": records, "old_run": old_run, "new_run": new_run,
            "source_report": recorded, "counts": counts,
            "paired": {"aug_only": gain, "R1_only": loss, "net": gain-loss},
            "files": {str(p): sha(p) for p in [starts,r1/"run.json",r1/"final.pt",
                r1/"metrics.jsonl",aug/"run.json",aug/"final.pt",aug/"metrics.jsonl",
                solver/"report.json",solver/"complete.json",solver/"manifest.json"] +
                [solver/f"solves.rank{i}.jsonl" for i in range(4)]}}


def category(r):
    old = r["models"]["R1"]["fov_pass"]
    new = r["models"]["R1_AUG"]["fov_pass"]
    if old and not new: return "lost"
    if new and not old: return "gained"
    if old and new: return "both_pass"
    return "both_fail"


def extract_case(row):
    old, aug = (row["models"][n] for n in ARMS)
    initial_g = old.get("initial_g_m")
    o_g, a_g = old.get("candidate_g_m"), aug.get("candidate_g_m")
    d = {
        "case_id": row["case_id"], "cohort": cohort(row["group"]), "group": row["group"],
        "sensor": row["sensor"], "outcome": category(row),
        "r1_pass": old["fov_pass"], "aug_pass": aug["fov_pass"],
        "initial_analytic_g_m": initial_g,
        "r1_candidate_g_m": o_g, "aug_candidate_g_m": a_g,
        "delta_candidate_g_m": a_g - o_g,
        "r1_zero_g_m": old.get("zero_g_m"), "aug_zero_g_m": aug.get("zero_g_m"),
        "r1_initial_score": old.get("initial_score"), "aug_initial_score": aug.get("initial_score"),
        "delta_initial_score": (aug["initial_score"]-old["initial_score"]
                                if finite(old.get("initial_score")) and finite(aug.get("initial_score")) else None),
        "r1_final_score": old.get("final_score"), "aug_final_score": aug.get("final_score"),
        "r1_f_zero": old.get("f_zero"), "aug_f_zero": aug.get("f_zero"),
        "r1_root_source": old.get("root_source"), "aug_root_source": aug.get("root_source"),
        "r1_solution_mode": old.get("solution_mode"), "aug_solution_mode": aug.get("solution_mode"),
        "r1_failure_stage": old.get("failure_stage"), "aug_failure_stage": aug.get("failure_stage"),
        "r1_projection_steps": max(0, len(old.get("projection_history", []))-1),
        "aug_projection_steps": max(0, len(aug.get("projection_history", []))-1),
        "r1_ascent_steps": len(old.get("ascent_history", [])),
        "aug_ascent_steps": len(aug.get("ascent_history", [])),
        "r1_joint_clamps": old.get("joint_clamp_events"),
        "aug_joint_clamps": aug.get("joint_clamp_events"),
        "r1_value_calls": old.get("value_calls"), "aug_value_calls": aug.get("value_calls"),
        "r1_gradient_calls": old.get("gradient_calls"), "aug_gradient_calls": aug.get("gradient_calls"),
        "r1_q_displacement_rad": l2(row["q_init"],old.get("q_candidate")),
        "aug_q_displacement_rad": l2(row["q_init"],aug.get("q_candidate")),
        "candidate_q_separation_rad": l2(old.get("q_candidate"),aug.get("q_candidate")),
        "r1_q_zero_distance_rad": l2(row["q_init"],old.get("q_zero")),
        "aug_q_zero_distance_rad": l2(row["q_init"],aug.get("q_zero")),
        "r1_initial_near_zero": finite(old.get("initial_score")) and abs(old["initial_score"]) <= .03,
        "aug_initial_near_zero": finite(aug.get("initial_score")) and abs(aug["initial_score"]) <= .03,
        "r1_initial_false_positive": finite(initial_g) and initial_g < 0 and
                                     finite(old.get("initial_score")) and old["initial_score"] >= 0,
        "aug_initial_false_positive": finite(initial_g) and initial_g < 0 and
                                      finite(aug.get("initial_score")) and aug["initial_score"] >= 0,
        "r1_initial_sign_agrees": (old["initial_score"] >= 0) == (initial_g >= 0),
        "aug_initial_sign_agrees": (aug["initial_score"] >= 0) == (initial_g >= 0),
        "root_source_changed": old["root_source"] != aug["root_source"],
        "failure_stage_changed": old["failure_stage"] != aug["failure_stage"],
        "aug_analytic_margin_worse": a_g < o_g,
    }
    # Observe if solver root/tolerance is in a *different* region from true FOV.
    # This is descriptive: predicted root is NOT a geometry certificate.
    d["r1_predicted_root_analytic_fail"] = (old.get("root_source") not in (
        "none", "branch_root_not_found", "degenerate_projection_gradient") and
        finite(old.get("zero_g_m")) and old["zero_g_m"] < 0)
    d["aug_predicted_root_analytic_fail"] = (aug.get("root_source") not in (
        "none", "branch_root_not_found", "degenerate_projection_gradient") and
        finite(aug.get("zero_g_m")) and aug["zero_g_m"] < 0)
    return d


def cross(rows, a, b, limit=24):
    c = Counter((str(r[a]), str(r[b])) for r in rows)
    return [{"from":x, "to":y, "count":n} for (x, y), n in
            sorted(c.items(), key=lambda x: (-x[1], x[0]))[:limit]]


def summarize_cases(cases, source):
    by_category = {cat:[r for r in cases if r["outcome"] == cat]
                   for cat in ("lost","gained","both_pass","both_fail")}
    pair = [r for r in cases if r["outcome"] in ("lost","gained")]
    c = {}
    for label, group in {"all":cases,**by_category}.items():
        c[label] = {
          "count": len(group),
          "by_cohort": dict(Counter(x["cohort"] for x in group)),
          "by_sensor": {f"S{s}":sum(x["sensor"] == s for x in group) for s in range(8)},
          "root_source_transitions": cross(group,"r1_root_source","aug_root_source"),
          "failure_stage_transitions": cross(group,"r1_failure_stage","aug_failure_stage"),
          "initial_near_zero": {
            n:sum(x[f"{n}_initial_near_zero"] for x in group)
            for n in ("r1","aug")},
          "initial_false_positive": {
            n:sum(x[f"{n}_initial_false_positive"] for x in group)
            for n in ("r1","aug")},
          "initial_sign_agrees": {
            n:sum(x[f"{n}_initial_sign_agrees"] for x in group)
            for n in ("r1","aug")},
          "predicted_root_analytic_fail": {
            n:sum(x[f"{n}_predicted_root_analytic_fail"] for x in group)
            for n in ("r1","aug")},
          "root_source_changed": sum(x["root_source_changed"] for x in group),
          "margin_worse_in_aug": sum(x["aug_analytic_margin_worse"] for x in group),
          "margin_delta_m": numeric_stats([x["delta_candidate_g_m"] for x in group]),
          "candidate_separation_rad": numeric_stats([x["candidate_q_separation_rad"] for x in group]),
          "aug_initial_score": numeric_stats([x["aug_initial_score"] for x in group]),
          "r1_initial_score": numeric_stats([x["r1_initial_score"] for x in group]),
          "r1_candidate_margin_m": numeric_stats([x["r1_candidate_g_m"] for x in group]),
          "aug_candidate_margin_m": numeric_stats([x["aug_candidate_g_m"] for x in group]),
        }
    sens={}
    for sensor in range(8):
        subset=[x for x in cases if x["sensor"]==sensor]
        sens[f"S{sensor}"]={
          "n":len(subset),
          "r1_pass":sum(x["r1_pass"] for x in subset),
          "aug_pass":sum(x["aug_pass"] for x in subset),
          "lost":sum(x["outcome"]=="lost" for x in subset),
          "gained":sum(x["outcome"]=="gained" for x in subset),
          "margin_delta_m":numeric_stats([x["delta_candidate_g_m"] for x in subset]),
        }
    coh={}
    for name in ("local","uniform"):
        subset=[x for x in cases if x["cohort"]==name]
        coh[name]={"n":len(subset),"lost":sum(x["outcome"]=="lost" for x in subset),
                   "gained":sum(x["outcome"]=="gained" for x in subset),
                   "r1_pass":sum(x["r1_pass"] for x in subset),
                   "aug_pass":sum(x["aug_pass"] for x in subset)}
    examples={}
    for label in ("lost","gained"):
        subset=by_category[label]
        subset.sort(key=lambda x: (x["delta_candidate_g_m"] if label=="lost"
                                   else -x["delta_candidate_g_m"], x["case_id"]))
        examples[label]=[x["case_id"] for x in subset[:20]]
    return {"status":"COMPLETE","scientific_status":"DESCRIPTIVE_NOT_CAUSAL",
            "official_paired":source["paired"],"categories":c,"cohorts":coh,
            "per_sensor":sens,"paired_case_ids":{label:[x["case_id"] for x in by_category[label]]
                                                for label in ("lost","gained")},
            "largest_margin_shift_examples":examples,
            "limits":["No new solver evaluation; recorded margins and learned values only.",
                      "Selected case comparisons cannot attribute causal effects to the normal loss.",
                      "The 1963 starts are repeatedly reused regression data, not unseen holdout."]}


def metrics_path_data(path):
    rows = list(read_jsonl(path))
    result = {}
    for x in rows:
        if x.get("split") != "val": continue
        step = x.get("step")
        if not isinstance(step, int) or step < 1 or step > 50000 or step in result:
            raise ValueError(f"Duplicate/invalid validation step {step} in {path}")
        if not isinstance(x.get("heads"), dict) or not finite(x.get("loss")):
            raise ValueError("Malformed R1 validation metrics")
        result[step] = x
    if 50000 not in result or len(result) < 20:
        raise ValueError("Missing complete original-objective validation sequence")
    return result


def metric_value(entry, head, field):
    if head == "all":
        return entry.get(field)
    return entry.get("heads", {}).get(head, {}).get(field)


def metrics_comparison(r1_metric_path, aug_metric_path):
    old, aug = metrics_path_data(r1_metric_path), metrics_path_data(aug_metric_path)
    common = sorted(set(old) & set(aug))
    if 50000 not in common or len(common) < 20:
        raise ValueError("R1/aug validation steps cannot be matched")
    metric_fields = {
        "all": ("loss","union_objective","sensor_objective","consistency_loss","winner_accuracy"),
        "union": ("rmse","mae","grad_cosine","grad_norm","sign_accuracy","eikonal_loss"),
        **{f"s{s}":("rmse","mae","grad_cosine","grad_norm","sign_accuracy","eikonal_loss") for s in range(8)}
    }
    series = []
    last = {}
    for step in common:
        summary={"step":step}
        for head, fields in metric_fields.items():
            for field in fields:
                name=f"{head}.{field}"
                v0,v1=metric_value(old[step],head,field),metric_value(aug[step],head,field)
                if not (finite(v0) and finite(v1)):
                    raise ValueError(f"Missing valid original-objective metric {name} step={step}")
                summary[name]={"r1":v0,"aug":v1,"delta":v1-v0}
        if step == 50000: last=summary
        series.append(summary)
    boundary_diag={}
    first_slices=[step for step in common if step in (1,1000,5000,10000,25000,50000)]
    for step in first_slices:
        diag=aug[step].get("boundary_direction_diagnostic")
        if not isinstance(diag, dict):
            raise ValueError(f"Missing auxiliary boundary diagnostic at aug validation step {step}")
        for field in ("normal_cosine","normal_grad_norm","boundary_abs_field",
                      "first_head_relu_active","first_head_relu_ever_active_count"):
            vector=diag.get(field)
            if not isinstance(vector,list) or len(vector)!=8 or not all(finite(z) for z in vector):
                raise ValueError(f"Invalid boundary diagnostic {field} at step {step}")
        boundary_diag[str(step)]={field:diag[field] for field in
                                  ("normal_cosine","normal_grad_norm","boundary_abs_field",
                                   "first_head_relu_active","first_head_relu_ever_active_count")}
    # Baseline original R1 has no augmentation-specific ReLU diagnostic.
    return {"compared_validation_steps":len(common), "steps":common,
            "last":last,"all_steps":series,"aug_boundary_diagnostic":boundary_diag,
            "boundary_diagnostic_baseline":"NOT_RECORDED_IN_ORIGINAL_R1_METRICS",
            "interpretation":"Metrics use original R1 label/objective on each matched validation step; auxiliary boundary metrics are observational."}


def rows_to_csv(path, rows, fields):
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for row in rows:writer.writerow({k:clean(row.get(k)) for k in fields})


def format_num(x,places=4):
    return "NA" if not finite(x) else f"{x:.{places}f}"


def markdown(case, metrics, boundary_note):
    c=case["categories"]
    last=metrics["last"]
    table=[
        "# R1 vs R1+Normal: Paired Loss/Gain and Training Drift Audit",
        "",
        "**READ-ONLY retrospective diagnosis; no training, no solver rerun, no threshold changes.**",
        "Existing frozen 1963 FOV-only starts, epsilon_f=0.03. Not an execution/trajectory/LOS certificate.",
        "",
        "## Fixed baseline and paired transitions",
        "",
        "| Cohort | N | R1 successes | Aug successes | Gained | Lost | Net |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for name,entry in [("local",case["cohorts"]["local"]),("uniform",case["cohorts"]["uniform"])]:
        table.append(f"| {name} | {entry['n']} | {entry['r1_pass']} | {entry['aug_pass']} | {entry['gained']} | {entry['lost']} | {entry['gained']-entry['lost']:+d} |")
    table.append(f"| all | 1963 | 1291 | {1237 if c['lost']['count']==136 and c['gained']['count']==82 else 1291+c['gained']['count']-c['lost']['count']} | {c['gained']['count']} | {c['lost']['count']} | {c['gained']['count']-c['lost']['count']:+d} |")
    table += ["", "## Per sensor paired outcomes","",
              "| Sensor | N | R1 | Aug | Aug-only | R1-only | Net |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for s,entry in case["per_sensor"].items():
        table.append(f"| {s} | {entry['n']} | {entry['r1_pass']} | {entry['aug_pass']} | {entry['gained']} | {entry['lost']} | {entry['gained']-entry['lost']:+d} |")
    table += ["", "## Candidate failure: descriptive co-occurrences","",
              "| Outcome | N | Root source changed | Aug candidate g lower | Aug predicted-root analytic FOV fails | Aug initial |score|<=.03 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in ("lost","gained","both_pass","both_fail"):
        e=c[name]
        table.append(f"| {name} | {e['count']} | {e['root_source_changed']} | {e['margin_worse_in_aug']} | {e['predicted_root_analytic_fail']['aug']} | {e['initial_near_zero']['aug']} |")
    table += ["", "### Root-source transitions on lost cases", ""]
    for r in c["lost"]["root_source_transitions"][:10]:
        table.append(f"- {r['from']} → {r['to']}: {r['count']}")
    table += ["", "### Candidate analytic margin shift (Aug − R1)", "",
              f"- Lost cases: median {format_num(c['lost']['margin_delta_m']['median'],6)} m; 10–90% {format_num(c['lost']['margin_delta_m']['p10'],6)} to {format_num(c['lost']['margin_delta_m']['p90'],6)} m.",
              f"- Gained cases: median {format_num(c['gained']['margin_delta_m']['median'],6)} m.",
              "",
              "## Fixed 50k original R1-objective validation comparison", "",
              "| Metric | R1 | R1+Normal | Difference |",
              "|---|---:|---:|---:|"]
    for key in ("all.loss","union.grad_cosine","union.rmse","s0.grad_cosine","s2.grad_cosine","s3.grad_cosine","s6.grad_cosine","s7.grad_cosine",
                "s0.rmse","s2.rmse","s3.rmse","s6.rmse","s7.rmse"):
        e=last[key];table.append(f"| {key} | {format_num(e['r1'],6)} | {format_num(e['aug'],6)} | {format_num(e['delta'],6)} |")
    table += ["",f"Matched full validation steps: {metrics['compared_validation_steps']}. Time series in metrics_comparison.json.",
              "", "## Same-point, frozen model boundary response",
              "",
              boundary_note,
              "",
              "## Artifacts", "",
              "- cases_all.csv: all 1963 aligned case features and observed transitions.",
              "- cases_paired.csv: only R1-only losses and augmented-only gains.",
              "- case_audit.json: descriptive aggregates, failure/root transitions, case IDs.",
              "- metrics_comparison.json: matched original-objective validation curves + augmented boundary diagnostics.",
              "- boundary_same_points.json/csv: optional held-out R1-split V4 validation anchors, both models evaluated on exactly the same points.",
              "- manifest.json: source and output hashes, fixed protocol.",
              "",
              "## Scientific limits", "",
              "- Negative/positive analytic margin change is ASSOCIATED with a changed candidate, not proof that the normal-direction loss caused it.",
              "- Differences in candidate root source and predicted score scale are observations from the unchanged runtime, not a new solver policy.",
              "- Original R1 did not log ReLU activity; it is compared only in the separate same-point frozen-model probe, when enabled.",
              "- This reused regression set cannot support an untouched generalization claim.",
              "- Do not select a new threshold, promote checkpoints, or infer training-step causality from this report."]
    return "\n".join(table)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r012",type=Path,required=True)
    p.add_argument("--aug",type=Path,required=True)
    p.add_argument("--solver",type=Path,required=True)
    p.add_argument("--starts",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--repo",type=Path,default=None)
    p.add_argument("--cache",type=Path,default=None)
    p.add_argument("--with-models",action="store_true",help="CPU same-point 8-sensor boundary diagnostics; no GPU")
    a=p.parse_args()
    a.r012=a.r012.resolve(); a.aug=a.aug.resolve()
    a.solver=a.solver.resolve(); a.starts=a.starts.resolve();a.out=a.out.resolve()
    if a.out.exists():
        raise FileExistsError(f"Refuse to overwrite diagnostic output: {a.out}")
    protected=[a.r012,a.aug,a.solver,a.starts.parent]
    for src in protected:
        if a.out == src or a.out.is_relative_to(src) or src.is_relative_to(a.out):
            raise ValueError(f"Diagnostic output overlaps protected source {src}")
    values=verify_inputs(a)
    all_rows=[extract_case(row) for row in values["rows"]]
    case=summarize_cases(all_rows,values)
    met=metrics_comparison(a.r012/"formal/R1/metrics.jsonl",
                           a.aug/"formal/metrics.jsonl")
    print("[verified] 1963 exact frozen starts; R1=1291, R1+normal="+str(values["counts"]["all"]["R1_AUG"]),
          "paired: gained="+str(values["paired"]["aug_only"])+", lost="+str(values["paired"]["R1_only"]),flush=True)
    boundary=None
    if a.with_models:
        if a.repo is None or a.cache is None:
            raise ValueError("--with-models requires --repo and --cache")
        from boundary_cpu import inspect_boundary
        boundary=inspect_boundary(a.repo.resolve(),a.cache.resolve(),a.r012,a.aug)
        print("[boundary] aligned original-R1 validation anchors, 8 sensors; both checkpoints unchanged",flush=True)
    a.out.mkdir(parents=True)
    fieldnames=list(all_rows[0])
    rows_to_csv(a.out/"cases_all.csv",all_rows,fieldnames)
    rows_to_csv(a.out/"cases_paired.csv",
                [r for r in all_rows if r["outcome"] in ("lost","gained")],fieldnames)
    write_json(a.out/"case_audit.json",case)
    write_json(a.out/"metrics_comparison.json",met)
    if boundary is not None:
        write_json(a.out/"boundary_same_points.json",boundary["summary"])
        rows_to_csv(a.out/"boundary_same_points.csv",boundary["rows"],list(boundary["rows"][0]))
    note=("Both frozen models evaluated on matched V4 validation anchors; see boundary_same_points.json."
          if boundary is not None else
          "NOT_RUN (--with-models omitted). Historical R1 metrics do not contain ReLU observations.")
    (a.out/"summary.md").write_text(markdown(case,met,note),encoding="utf-8")
    outputfiles={str(x.relative_to(a.out)):sha(x) for x in a.out.iterdir() if x.is_file()}
    write_json(a.out/"manifest.json",{
        "format":"care_h9_r1_normal_aug_case_diagnosis_v1","status":"COMPLETE",
        "source_hashes_sha256":values["files"],"output_hashes_sha256":outputfiles,
        "r1_training_stream":values["old_run"]["training_stream_sha256"],
        "aug_training_stream":values["new_run"]["training_stream_sha256"],
        "solver":"RECORDED; NOT RE-RUN","new_solver_calls":0,"optimizer_updates":0,
        "boundary_model_probe":a.with_models,"requested_boundary_cache":str(a.cache) if a.cache else None,
        "reused_frozen_starts_sha256":EXPECTED_STARTS,
        "scope":"Descriptive association and fixed-validation diagnostics; not causal attribution",
    })
    print((a.out/"summary.md").read_text(),flush=True)
    print(f"[COMPLETE] {a.out}",flush=True)


if __name__ == "__main__":
    main()
