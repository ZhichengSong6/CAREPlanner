#!/usr/bin/env python3
"""Analyze matched V1-vs-R1 Case026 runtime qualification runs."""
from __future__ import annotations
import argparse, csv, json, math, os, re, statistics
from pathlib import Path

TOK=re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
BRANCH=re.compile(r"PER-SENSOR BRANCH ACCEPTED S(\d+) rank=(\d+).*branch_ms=([0-9.+-eE]+)")

def tokens(path:Path):
    if not path.is_file(): return []
    out=[]
    with path.open(newline="",errors="replace") as f:
        rd=csv.reader(f); h=next(rd,[])
        if not h: return out
        di=h.index("field.data") if "field.data" in h else 1
        for row in rd:
            if len(row)>di:
                d=dict(TOK.findall(",".join(row[di:])))
                if d: out.append(d)
    return out

def as_int(v,d=0):
    try: return int(float(v))
    except Exception: return d

def as_float(v,d=math.nan):
    try:
        x=float(v); return x if math.isfinite(x) else d
    except Exception: return d

def dist(xs):
    xs=sorted(x for x in xs if math.isfinite(x))
    if not xs: return None
    def q(p):
        z=p*(len(xs)-1); i=int(z); j=min(i+1,len(xs)-1); w=z-i
        return xs[i]*(1-w)+xs[j]*w
    return {"count":len(xs),"median":statistics.median(xs),"mean":statistics.fmean(xs),"p95":q(.95),"max":max(xs)}

def hard_hold_count(run:Path):
    p=run/"execution_gcdf_hard_hold.csv"
    if not p.is_file(): return 0
    n=0
    for line in p.read_text(errors="replace").splitlines():
        low=line.lower()
        if "hard_hold=1" in low or "hard_hold=true" in low: n+=1
    return n

def generator_stats(path:Path):
    text=path.read_text(errors="replace") if path.is_file() else ""
    accepted=[]; ms=[]
    for m in BRANCH.finditer(text):
        accepted.append({"sensor_id":int(m.group(1)),"rank":int(m.group(2)),"branch_ms":float(m.group(3))})
        ms.append(float(m.group(3)))
    return {
        "hybrid_enabled_count":text.count("PER-SENSOR HYBRID ENABLED"),
        "accepted_branch_count":len(accepted),
        "all_rejected_count":text.count("per-sensor branches all rejected"),
        "accepted_branches":accepted,
        "branch_latency_ms":dist(ms),
    }

def acquisition(run:Path):
    rows=tokens(run/"visibility_acquisition_summary.csv")
    rem=[as_int(r.get("remaining_obligation_count"),-1) for r in rows]
    rem=[x for x in rem if x>=0]
    seen=[as_int(r.get("seen_obligation_count"),-1) for r in rows]
    seen=[x for x in seen if x>=0]
    return {
        "records":len(rows),
        "complete_ever":any(r.get("complete")=="1" for r in rows),
        "remaining_final":rem[-1] if rem else None,
        "remaining_min":min(rem) if rem else None,
        "seen_final":seen[-1] if seen else None,
        "seen_max":max(seen) if seen else None,
    }

def one(label,run_dir,eval_path,gen_log):
    run=Path(run_dir).resolve(); ev=json.load(open(eval_path))
    hh=hard_hold_count(run); gs=generator_stats(Path(gen_log)); acq=acquisition(run)
    execution_clean=(int(ev.get("execution_vbc_unsafe_records",0))==0 and hh==0)
    if int(ev.get("commit_count",0))>0:
        certified_execution=bool(ev.get("overall_safe")) and bool(ev.get("gcdf_commit_certified")) and bool(ev.get("vbc_commit_certified")) and bool(ev.get("execution_vbc_safe"))
        evidence="CERTIFIED_EXECUTION" if certified_execution else "EXECUTION_SAFETY_FAILED"
    else:
        certified_execution=False
        evidence="NO_COMMIT_EXECUTION_EVIDENCE"
    return {
        "label":label,"run_dir":str(run),"evaluation":ev,"hard_hold_true":hh,
        "generator":gs,"acquisition":acq,"execution_clean":execution_clean,
        "certified_execution":certified_execution,"execution_evidence":evidence,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--v1-run-dir",required=True); ap.add_argument("--r1-run-dir",required=True)
    ap.add_argument("--v1-eval",required=True); ap.add_argument("--r1-eval",required=True)
    ap.add_argument("--v1-generator-log",required=True); ap.add_argument("--r1-generator-log",required=True)
    ap.add_argument("--targeted-compare",default="")
    a=ap.parse_args()
    root=Path(a.root).resolve(); root.mkdir(parents=True,exist_ok=True)
    v=one("V1",a.v1_run_dir,a.v1_eval,a.v1_generator_log)
    r=one("R1",a.r1_run_dir,a.r1_eval,a.r1_generator_log)
    targeted=None
    if a.targeted_compare and Path(a.targeted_compare).is_file():
        targeted=json.load(open(a.targeted_compare))

    task_no_regression=not (v["evaluation"].get("task_success") is True and r["evaluation"].get("task_success") is not True)
    acq_no_regression=not (v["acquisition"]["complete_ever"] and not r["acquisition"]["complete_ever"])
    targeted_valid=True if targeted is None else bool(targeted.get("valid_diagnostic"))
    r1_has_execution=int(r["evaluation"].get("commit_count",0))>0
    safety_gate=bool(r["execution_clean"] and (r["certified_execution"] if r1_has_execution else True))

    if not targeted_valid:
        verdict="HOLD_TARGETED_INVALID"
    elif not r["execution_clean"] or (r1_has_execution and not r["certified_execution"]):
        verdict="HOLD_R1_SAFETY_FAILURE"
    elif not task_no_regression:
        verdict="HOLD_TASK_REGRESSION"
    elif not acq_no_regression:
        verdict="HOLD_ACQUISITION_REGRESSION"
    elif not r1_has_execution:
        verdict="INCONCLUSIVE_NO_R1_EXECUTION_EVIDENCE"
    else:
        verdict="R1_RUNTIME_QUALIFIED"

    report={
        "qualification":"phase_e_v1_r1_case026_runtime",
        "V1":v,"R1":r,"targeted":targeted,
        "gates":{
            "targeted_valid":targeted_valid,
            "r1_execution_clean":r["execution_clean"],
            "r1_has_execution_evidence":r1_has_execution,
            "r1_certified_execution":r["certified_execution"],
            "task_no_regression":task_no_regression,
            "acquisition_no_regression":acq_no_regression,
            "safety_gate":safety_gate,
        },
        "verdict":verdict,
        "promotion_semantics":"Only q_vis sensor branch model changed. Scalar projector, FOV/LOS, Sparse-SCP, VBC, GCDF, tracker unchanged.",
    }
    (root/"runtime_compare.json").write_text(json.dumps(report,indent=2,allow_nan=True))

    def yes(x): return "YES" if x else "NO"
    lines=[
        "# V1 vs R1 Case026 runtime qualification","",
        f"Verdict: **{verdict}**","",
        "| Metric | V1 | R1 |","|---|---:|---:|",
        f"| task_success | {v['evaluation'].get('task_success')} | {r['evaluation'].get('task_success')} |",
        f"| commit_count | {v['evaluation'].get('commit_count')} | {r['evaluation'].get('commit_count')} |",
        f"| overall_safe | {v['evaluation'].get('overall_safe')} | {r['evaluation'].get('overall_safe')} |",
        f"| execution_vbc_unsafe_records | {v['evaluation'].get('execution_vbc_unsafe_records')} | {r['evaluation'].get('execution_vbc_unsafe_records')} |",
        f"| GCDF hard_hold true | {v['hard_hold_true']} | {r['hard_hold_true']} |",
        f"| acquisition complete ever | {yes(v['acquisition']['complete_ever'])} | {yes(r['acquisition']['complete_ever'])} |",
        f"| remaining obligations final | {v['acquisition']['remaining_final']} | {r['acquisition']['remaining_final']} |",
        f"| accepted sensor branches | {v['generator']['accepted_branch_count']} | {r['generator']['accepted_branch_count']} |",
        f"| all sensor branches rejected | {v['generator']['all_rejected_count']} | {r['generator']['all_rejected_count']} |",
        "",
        "## Gates","",
    ]
    for k,val in report["gates"].items(): lines.append(f"- {k}: **{val}**")
    lines += ["","R1 may only be promoted when runtime safety evidence is clean. A Case026 NO_CLEAR branch outcome is allowed; false FOV/LOS acceptance is not."]
    (root/"runtime_summary.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report,indent=2,allow_nan=True))
    print("[done]",verdict,root/"runtime_compare.json")

if __name__=="__main__": main()
