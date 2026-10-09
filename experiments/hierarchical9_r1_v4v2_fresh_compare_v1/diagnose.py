#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np
SETS=("R1_only","V2_50K_only","both_pass","both_fail");MODELS=("R1","V2_50K")
def safe(v):return float(v) if isinstance(v,(int,float)) and math.isfinite(float(v)) else None
def dist(vals):
    a=np.asarray([float(v) for v in vals if v is not None and math.isfinite(float(v))],float)
    if not len(a):return {"n":0,"mean":None,"p10":None,"p50":None,"p90":None}
    return {"n":int(len(a)),"mean":float(a.mean()),"p10":float(np.quantile(a,.1)),"p50":float(np.quantile(a,.5)),"p90":float(np.quantile(a,.9))}
def kind(r):
    a=bool(r["models"]["V2_50K"]["fov_pass"]);b=bool(r["models"]["R1"]["fov_pass"])
    return "both_pass" if a and b else "V2_50K_only" if a else "R1_only" if b else "both_fail"
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    parts=sorted(a.input.resolve().glob("solves.rank*.jsonl"));rows=[]
    for p in parts:rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    rows.sort(key=lambda r:int(r["case_id"]))
    if [int(r["case_id"]) for r in rows]!=list(range(len(rows))):raise ValueError("incomplete shards")
    out={"status":"COMPLETE","count":len(rows),"sets":{}}
    for k in SETS:
        rr=[r for r in rows if kind(r)==k];block={"count":len(rr),"by_sensor":dict(Counter(f"S{r['sensor']}" for r in rr)),"by_cohort":dict(Counter("local" if "/local_" in r["group"] else "uniform" for r in rr)),"models":{}}
        for n in MODELS:
            ms=[r["models"][n] for r in rr];block["models"][n]={"failure_stage":dict(Counter(m.get("failure_stage") for m in ms)),"root_source":dict(Counter(m.get("root_source") for m in ms)),"candidate_g_m":dist([safe(m.get("candidate_g_m")) for m in ms]),"zero_g_m":dist([safe(m.get("zero_g_m")) for m in ms]),"final_score":dist([safe(m.get("final_score")) for m in ms]),"solver_ms":dist([safe(m.get("solver_ms")) for m in ms])}
        out["sets"][k]=block
    bins=Counter()
    for r in rows:
        if kind(r)!="R1_only":continue
        g=safe(r["models"]["V2_50K"].get("candidate_g_m"))
        if g is None:bins["missing"]+=1
        elif g>=0:bins[">=0"]+=1
        elif g>=-.01:bins["[-.01,0)"]+=1
        elif g>=-.03:bins["[-.03,-.01)"]+=1
        elif g>=-.10:bins["[-.10,-.03)"]+=1
        else:bins["<-.10"]+=1
    out["r1_only_v2_candidate_g_bins"]=dict(bins);a.output.mkdir(parents=True,exist_ok=True);(a.output/"diagnosis.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    lines=["# R1 vs V2-50K fresh solver diagnosis","",f"Rows: {len(rows)}","","| outcome | count |","|---|---:|"]+[f"| {k} | {out['sets'][k]['count']} |" for k in SETS]
    lines+=["","## R1-only regressions: V2 failure stage"]+[f"- {k}: {v}" for k,v in out["sets"]["R1_only"]["models"]["V2_50K"]["failure_stage"].items()]
    lines+=["","## R1-only regressions: V2 analytic candidate margin bins"]+[f"- {k}: {v}" for k,v in bins.items()]
    lines+=["","## Outcome counts by sensor"]
    for name in ("R1_only","V2_50K_only"):
        lines.append(f"### {name}");lines += [f"- {k}: {v}" for k,v in sorted(out["sets"][name]["by_sensor"].items())]
    (a.output/"diagnosis.md").write_text("\n".join(lines)+"\n");print((a.output/"diagnosis.md").read_text())
if __name__=="__main__":main()
