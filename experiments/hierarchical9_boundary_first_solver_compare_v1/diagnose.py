#!/usr/bin/env python3
"""Full-case FOV failure diagnostics on exactly the same frozen paired rows."""
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np

CANDIDATES=("BF_BEST","BF_FINAL","BF_BEST_002","BF_FINAL_002")
def summary(vals):
    a=np.asarray([float(x) for x in vals if x is not None and math.isfinite(float(x))],dtype=float)
    return {"count":int(len(a)),"mean":float(a.mean()) if len(a) else None,
            "median":float(np.median(a)) if len(a) else None,
            "p90":float(np.quantile(a,.9)) if len(a) else None}
def outcome(row,name):
    a=bool(row["models"][name]["fov_pass"])
    b=bool(row["models"]["R1"]["fov_pass"])
    return ("both_pass" if a and b else "candidate_only" if a else "R1_only" if b else "both_fail")
def bins(vals):
    c=Counter()
    for x in vals:
        if x is None or not math.isfinite(float(x)):
            c["nonfinite"]+=1
        elif x>=0:c[">=0"]+=1
        elif x>=-.01:c["[-.01,0)"]+=1
        elif x>=-.03:c["[-.03,-.01)"]+=1
        elif x>=-.10:c["[-.10,-.03)"]+=1
        else:c["<-0.10"]+=1
    return dict(c)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    rows=[]
    for f in sorted(a.input.resolve().glob("solves.rank*.jsonl")):
        rows.extend(json.loads(x) for x in f.read_text().splitlines() if x.strip())
    rows.sort(key=lambda r:r["case_id"])
    if [r["case_id"] for r in rows]!=list(range(1963)):
        raise ValueError("Expected exactly 1963 independent case IDs")
    report={"status":"COMPLETE","count":len(rows),"models":{}}
    lines=["# Boundary-first scratch solver diagnosis","","Frozen paired cases: 1963.",""]
    for name in CANDIDATES:
        cases=Counter(outcome(r,name) for r in rows)
        regress=[r for r in rows if outcome(r,name)=="R1_only"]
        bf=[r["models"][name] for r in regress]
        stages=Counter(v.get("failure_stage") for v in bf)
        roots=Counter(v.get("root_source") for v in bf)
        bysensor=Counter(f"S{r['sensor']}" for r in regress)
        candidate_bins=bins([v.get("candidate_g_m") for v in bf])
        times=summary([r["models"][name].get("solver_ms") for r in rows])
        block={"outcomes":dict(cases),"r1_only_failure_stages":dict(stages),
               "r1_only_root_sources":dict(roots),"r1_only_by_sensor":dict(bysensor),
               "r1_only_candidate_g_bins":candidate_bins,"solver_time_ms":times}
        report["models"][name]=block
        lines += [f"## {name}","","| Outcome | Count |","|---|---:|"]
        lines += [f"| {key} | {cases.get(key,0)} |" for key in
                  ("candidate_only","R1_only","both_pass","both_fail")]
        lines += ["","R1-only regressions: failure stage: "+json.dumps(dict(stages),sort_keys=True),
                  "R1-only regressions: root source: "+json.dumps(dict(roots),sort_keys=True),
                  "R1-only regressions: analytic candidate g (m): "+json.dumps(candidate_bins,sort_keys=True),
                  "R1-only regressions by sensor: "+json.dumps(dict(sorted(bysensor.items()))),
                  "Solver time (ms): "+json.dumps(times,sort_keys=True),""]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"diagnosis.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (a.output/"diagnosis.md").write_text("\n".join(lines)+"\n")
    print((a.output/"diagnosis.md").read_text(),flush=True)
if __name__=="__main__":
    main()
