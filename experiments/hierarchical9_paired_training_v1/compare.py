#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
from common import ARMS,write_json
def close(a,b,tol=2e-7):
 return a is None and b is None or a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);a=ap.parse_args();root=a.root.resolve();rows={}
 for arm in ARMS:
  d=root/arm;run=json.loads((d/"run.json").read_text());m=json.loads((d/"final_metrics.json").read_text());initial=json.loads((d/"initial.json").read_text())
  if run.get("status")!="COMPLETE":raise ValueError(f"{arm} incomplete")
  rows[arm]=(run,m,initial)
 if len({r[0]["stream_sha256"] for r in rows.values()})!=1 or len({r[0]["parent_sha256"] for r in rows.values()})!=1 or len({r[0]["cache_index_sha256"] for r in rows.values()})!=1:raise ValueError("fairness identity mismatch")
 baseline=rows["old_value"][2]["val"]
 keys=lambda v:(v["targets"]["new"]["sensor_mae_mean"],v["targets"]["old"]["sensor_mae_mean"],v["analytic_sensor_sign_accuracy"])
 bk=keys(baseline)
 for arm in ARMS:
  ak=keys(rows[arm][2]["val"])
  if not all(close(x,y) for x,y in zip(ak,bk)):raise ValueError(f"initial frozen-R1 metrics differ across arms: {arm} {ak} vs {bk}")
 report={"status":"COMPLETE_REVIEW_REQUIRED","same_stream":True,"same_parent":True,"same_cache":True,"initial_metric_tolerance":2e-7,
  "frozen_r1":{"val_new_sensor_mae":baseline["targets"]["new"]["sensor_mae_mean"],"val_old_sensor_mae":baseline["targets"]["old"]["sensor_mae_mean"],
   "analytic_sensor_sign_accuracy":baseline["analytic_sensor_sign_accuracy"]},"arms":{}}
 for arm,(run,m,_) in rows.items():
  v=m["val"];report["arms"][arm]={"val_new_sensor_mae":v["targets"]["new"]["sensor_mae_mean"],"val_old_sensor_mae":v["targets"]["old"]["sensor_mae_mean"],
   "val_new_union_mae":v["targets"]["new"]["union"].get("mae"),"val_old_union_mae":v["targets"]["old"]["union"].get("mae"),
   "analytic_sensor_sign_accuracy":v["analytic_sensor_sign_accuracy"],"parent_value_drift_mae":v["parent_value_drift_mae"],"final_sha256":run["final_sha256"]}
 report["paired_deltas"]={"value_new_minus_old_on_new_mae":report["arms"]["new_value"]["val_new_sensor_mae"]-report["arms"]["old_value"]["val_new_sensor_mae"],
  "valuegrad_new_minus_old_on_new_mae":report["arms"]["new_value_grad"]["val_new_sensor_mae"]-report["arms"]["old_value_grad"]["val_new_sensor_mae"],
  "new_value_minus_frozen_r1_on_new_mae":report["arms"]["new_value"]["val_new_sensor_mae"]-report["frozen_r1"]["val_new_sensor_mae"],
  "new_value_grad_minus_frozen_r1_on_new_mae":report["arms"]["new_value_grad"]["val_new_sensor_mae"]-report["frozen_r1"]["val_new_sensor_mae"],
  "value_new_minus_old_sign":report["arms"]["new_value"]["analytic_sensor_sign_accuracy"]-report["arms"]["old_value"]["analytic_sensor_sign_accuracy"],
  "valuegrad_new_minus_old_sign":report["arms"]["new_value_grad"]["analytic_sensor_sign_accuracy"]-report["arms"]["old_value_grad"]["analytic_sensor_sign_accuracy"]}
 write_json(root/"comparison.json",report)
 b=report["frozen_r1"];lines=["# R1 paired-label micro-training comparison","",f"Frozen R1 baseline: val_new_MAE={b['val_new_sensor_mae']:.6f}, val_old_MAE={b['val_old_sensor_mae']:.6f}, analytic_sign={b['analytic_sensor_sign_accuracy']:.6f}.","",
  "| arm | val MAE vs NEW | val MAE vs OLD | analytic sign | parent drift |","|---|---:|---:|---:|---:|"]
 for arm in ARMS:
  r=report["arms"][arm];lines.append(f"| {arm} | {r['val_new_sensor_mae']:.6f} | {r['val_old_sensor_mae']:.6f} | {r['analytic_sensor_sign_accuracy']:.6f} | {r['parent_value_drift_mae']:.6f} |")
 lines+=["","Negative MAE delta means the left/new arm is lower on held-out NEW labels. Frozen-R1 deltas answer whether fine-tuning improved this paired validation metric. This micro-study alone does not prove solver/runtime improvement.","",json.dumps(report["paired_deltas"],indent=2)]
 (root/"comparison.md").write_text("\n".join(lines)+"\n");print((root/"comparison.md").read_text(),flush=True)
if __name__=="__main__":main()
