#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from common import ARMS,write_json
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);a=ap.parse_args();root=a.root.resolve();rows={}
 for arm in ARMS:
  d=root/arm;run=json.loads((d/"run.json").read_text());m=json.loads((d/"final_metrics.json").read_text())
  if run.get("status")!="COMPLETE":raise ValueError(f"{arm} incomplete")
  rows[arm]=(run,m)
 if len({r[0]["stream_sha256"] for r in rows.values()})!=1 or len({r[0]["parent_sha256"] for r in rows.values()})!=1 or len({r[0]["cache_index_sha256"] for r in rows.values()})!=1:raise ValueError("fairness identity mismatch")
 report={"status":"COMPLETE_REVIEW_REQUIRED","same_stream":True,"same_parent":True,"same_cache":True,"arms":{}}
 for arm,(run,m) in rows.items():
  v=m["val"];report["arms"][arm]={"val_new_sensor_mae":v["targets"]["new"]["sensor_mae_mean"],"val_old_sensor_mae":v["targets"]["old"]["sensor_mae_mean"],
   "val_new_union_mae":v["targets"]["new"]["union"].get("mae"),"val_old_union_mae":v["targets"]["old"]["union"].get("mae"),
   "analytic_sensor_sign_accuracy":v["analytic_sensor_sign_accuracy"],"parent_value_drift_mae":v["parent_value_drift_mae"],"final_sha256":run["final_sha256"]}
 report["paired_deltas"]={"value_new_minus_old_on_new_mae":report["arms"]["new_value"]["val_new_sensor_mae"]-report["arms"]["old_value"]["val_new_sensor_mae"],
  "valuegrad_new_minus_old_on_new_mae":report["arms"]["new_value_grad"]["val_new_sensor_mae"]-report["arms"]["old_value_grad"]["val_new_sensor_mae"],
  "value_new_minus_old_sign":report["arms"]["new_value"]["analytic_sensor_sign_accuracy"]-report["arms"]["old_value"]["analytic_sensor_sign_accuracy"],
  "valuegrad_new_minus_old_sign":report["arms"]["new_value_grad"]["analytic_sensor_sign_accuracy"]-report["arms"]["old_value_grad"]["analytic_sensor_sign_accuracy"]}
 write_json(root/"comparison.json",report)
 lines=["# R1 paired-label micro-training comparison","","| arm | val MAE vs NEW | val MAE vs OLD | analytic sign | parent drift |","|---|---:|---:|---:|---:|"]
 for arm in ARMS:
  r=report["arms"][arm];lines.append(f"| {arm} | {r['val_new_sensor_mae']:.6f} | {r['val_old_sensor_mae']:.6f} | {r['analytic_sensor_sign_accuracy']:.6f} | {r['parent_value_drift_mae']:.6f} |")
 lines+=["","Negative NEW-minus-OLD MAE delta means the new-label arm fits held-out new labels better. This micro-study alone does not prove solver/runtime improvement.","",json.dumps(report["paired_deltas"],indent=2)]
 (root/"comparison.md").write_text("\n".join(lines)+"\n");print((root/"comparison.md").read_text(),flush=True)
if __name__=="__main__":main()
