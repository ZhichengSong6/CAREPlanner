#!/usr/bin/env python3
"""Read-only failure-mechanism diagnosis for completed R1 vs NEW_VALUE solver comparison."""
from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

SETS=("R1_only","NEW_VALUE_only","both_pass","both_fail")
MODELS=("R1","NEW_VALUE")

def safe_num(v):
    return float(v) if isinstance(v,(int,float)) and math.isfinite(float(v)) else None
def dist(vals):
    a=np.asarray([float(v) for v in vals if v is not None and math.isfinite(float(v))],dtype=np.float64)
    if not len(a):return {"n":0,"mean":None,"p10":None,"p50":None,"p90":None}
    return {"n":int(len(a)),"mean":float(a.mean()),"p10":float(np.quantile(a,.1)),"p50":float(np.quantile(a,.5)),"p90":float(np.quantile(a,.9))}
def kind(r):
    a=bool(r["models"]["NEW_VALUE"]["fov_pass"]);b=bool(r["models"]["R1"]["fov_pass"])
    return "both_pass" if a and b else "NEW_VALUE_only" if a else "R1_only" if b else "both_fail"
def history_stats(m):
    ph=m.get("projection_history") or []; ah=m.get("ascent_history") or []
    return {
      "projection_steps":len(ph)-1 if ph else 0,
      "ascent_steps":len(ah),
      "projection_clamps":sum(bool(x.get("joint_limit_clamped")) for x in ph),
      "ascent_clamps":sum(bool(x.get("joint_limit_clamped")) for x in ah),
      "projection_clipped":sum(bool(x.get("algorithm_step_clipped")) for x in ph),
      "projection_degenerate":sum(bool(x.get("degenerate")) for x in ph),
      "ascent_degenerate":sum(bool(x.get("degenerate")) for x in ah),
      "projection_grad_norms":[safe_num(x.get("grad_norm")) for x in ph],
      "ascent_grad_norms":[safe_num(x.get("grad_norm")) for x in ah],
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    src=a.input.resolve()
    if src.is_dir():
        parts=sorted(src.glob("solves.rank*.jsonl"))
        if not parts: raise FileNotFoundError(f"No solves.rank*.jsonl in {src}")
        rows=[]
        for p in parts:
            rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
        rows.sort(key=lambda r:int(r["case_id"]))
        ids=[int(r["case_id"]) for r in rows]
        if ids!=list(range(len(rows))):
            raise ValueError("Rank shards are incomplete, duplicated, or non-contiguous")
        input_desc=[str(p) for p in parts]
    else:
        rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        rows.sort(key=lambda r:int(r["case_id"]))
        ids=[int(r["case_id"]) for r in rows]
        if ids!=list(range(len(rows))):
            raise ValueError("Input rows are incomplete, duplicated, or non-contiguous")
        input_desc=[str(src)]
    out={"status":"COMPLETE","input":input_desc,"count":len(rows),"sets":{}}
    for k in SETS:
        rr=[r for r in rows if kind(r)==k]
        block={"count":len(rr),"by_sensor":dict(Counter(f"S{r['sensor']}" for r in rr)),"by_cohort":dict(Counter("local" if "/local_" in r["group"] else "uniform" for r in rr)),"models":{}}
        for n in MODELS:
            ms=[r["models"][n] for r in rr]
            hs=[history_stats(m) for m in ms]
            block["models"][n]={
              "failure_stage":dict(Counter(m.get("failure_stage") for m in ms)),
              "root_source":dict(Counter(m.get("root_source") for m in ms)),
              "solution_mode":dict(Counter(m.get("solution_mode") for m in ms)),
              "initial_score":dist([safe_num(m.get("initial_score")) for m in ms]),
              "best_score":dist([safe_num(m.get("best_score")) for m in ms]),
              "final_score":dist([safe_num(m.get("final_score")) for m in ms]),
              "f_zero":dist([abs(safe_num(m.get("f_zero"))) if safe_num(m.get("f_zero")) is not None else None for m in ms]),
              "zero_g_m":dist([safe_num(m.get("zero_g_m")) for m in ms]),
              "candidate_g_m":dist([safe_num(m.get("candidate_g_m")) for m in ms]),
              "solver_ms":dist([safe_num(m.get("solver_ms")) for m in ms]),
              "projection_steps":dist([h["projection_steps"] for h in hs]),
              "ascent_steps":dist([h["ascent_steps"] for h in hs]),
              "projection_clamps_total":int(sum(h["projection_clamps"] for h in hs)),
              "ascent_clamps_total":int(sum(h["ascent_clamps"] for h in hs)),
              "projection_clipped_total":int(sum(h["projection_clipped"] for h in hs)),
              "projection_grad_norm":dist([v for h in hs for v in h["projection_grad_norms"]]),
              "ascent_grad_norm":dist([v for h in hs for v in h["ascent_grad_norms"]]),
            }
        out["sets"][k]=block
    reg=out["sets"]["R1_only"]
    root_pairs=Counter()
    stage_pairs=Counter()
    margin_bins=Counter()
    learned_vs_analytic=[]
    for r in rows:
        if kind(r)!="R1_only":continue
        a0=r["models"]["R1"];b0=r["models"]["NEW_VALUE"]
        root_pairs[(a0.get("root_source"),b0.get("root_source"))]+=1
        stage_pairs[(a0.get("failure_stage"),b0.get("failure_stage"))]+=1
        g=safe_num(b0.get("candidate_g_m"))
        if g is None:margin_bins["missing"]+=1
        elif g>=0:margin_bins[">=0"]+=1
        elif g>=-.01:margin_bins["[-.01,0)"]+=1
        elif g>=-.03:margin_bins["[-.03,-.01)"]+=1
        elif g>=-.10:margin_bins["[-.10,-.03)"]+=1
        else:margin_bins["<-.10"]+=1
        fs=safe_num(b0.get("final_score"))
        if fs is not None and g is not None:learned_vs_analytic.append((fs,g))
    out["r1_only_mechanism"]={
      "root_source_pairs":{"R1="+str(x)+" | NEW="+str(y):int(v) for (x,y),v in root_pairs.most_common()},
      "failure_stage_pairs":{"R1="+str(x)+" | NEW="+str(y):int(v) for (x,y),v in stage_pairs.most_common()},
      "new_candidate_g_bins":dict(margin_bins),
      "new_final_score_minus_analytic_g":dist([f-g for f,g in learned_vs_analytic]),
    }
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"diagnosis.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    lines=["# R1 vs NEW_VALUE solver failure diagnosis","",f"Rows: {len(rows)}","",
      "## Outcome sets","", "| set | count |","|---|---:|"]+[f"| {k} | {out['sets'][k]['count']} |" for k in SETS]
    lines+=["","## R1-only regressions by sensor",""]
    for s,c in sorted(reg["by_sensor"].items()):lines.append(f"- {s}: {c}")
    lines+=["","## R1-only: NEW_VALUE failure stage",""]
    for x,c in reg["models"]["NEW_VALUE"]["failure_stage"].items():lines.append(f"- {x}: {c}")
    lines+=["","## R1-only: NEW_VALUE root source",""]
    for x,c in reg["models"]["NEW_VALUE"]["root_source"].items():lines.append(f"- {x}: {c}")
    lines+=["","## R1-only: NEW_VALUE analytic candidate margin bins",""]
    for x,c in out["r1_only_mechanism"]["new_candidate_g_bins"].items():lines.append(f"- {x}: {c}")
    lines+=["","## R1-only: most common root-source transitions",""]
    for x,c in list(out["r1_only_mechanism"]["root_source_pairs"].items())[:12]:lines.append(f"- {x}: {c}")
    lines+=["","## Key medians",""]
    for n in MODELS:
        z=reg["models"][n];lines.append(f"- {n}: |f_zero| p50={z['f_zero']['p50']}, zero_g p50={z['zero_g_m']['p50']}, candidate_g p50={z['candidate_g_m']['p50']}, final learned score p50={z['final_score']['p50']}, solver_ms p50={z['solver_ms']['p50']}")
    lines+=["","Interpretation note: learned scores and analytic g are different units/scales; their difference is diagnostic only, not a calibrated physical error."]
    (a.output/"diagnosis.md").write_text("\n".join(lines)+"\n")
    print((a.output/"diagnosis.md").read_text())

if __name__=="__main__":main()
