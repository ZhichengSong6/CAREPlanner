#!/usr/bin/env python3
"""One FOV-only validation search comparison of all THREE frozen H9 runs.

Reuse exactly the existing fixed 64-case study / q / analytic conservative FOV /
eight-sensor projection-root-ascent budget. No LOS, no self-occlusion, no
changes to safety thresholds, no training and no new test cases.
The original Val split is not an independent final test set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/"experiments/hierarchical9_grid25_v3_runtime_integration_v1"))
import fov_val_pair as prior  # noqa: E402

ROOT=Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")


def require(ok,msg):
    if not ok:raise RuntimeError("[THIRD H9 FOV STOP] "+msg)


def run(root):
    root=root.resolve()
    output=root/"derived/grid25_v3_boundary_pairs_v1/fov_val_three_models_v1.json"
    require(not output.exists(),"Third round paired FOV report already exists")
    previous=root/"derived/grid25_v3_runtime_integration_v1/fov_val_baseline_vs_boundary_v1.json"
    require(previous.is_file(),"Original 64-case baseline comparison unavailable")
    prior_report=prior.read(previous)
    require(prior_report.get("status")=="COMPLETE"
            and prior_report.get("seed")==260930
            and prior_report.get("cases_requested")==64
            and prior_report.get("fov_only") is True,
            "Original frozen 64-case FOV comparison identity differs")
    torch.set_num_threads(4)
    oracle=prior.FOVOnlyOracle()
    cases,manifest=prior.fixed_cases(root,oracle,64,260930)
    require(len(prior_report["records"])==len(cases)
            and all(row["case"]==case for row,case
                    in zip(prior_report["records"],cases)),
            "Original 64 held-out queries changed")
    models={
        "baseline_v3":root/"derived/grid25_v3_full_training_v1/run_r1_v3_full/best.pt",
        "boundary_weighted_v3":root/"derived/grid25_v3_boundary_focus_v1/run_r1_v3_boundary_focus/best.pt",
        "boundary_pairs_v3":root/"derived/grid25_v3_boundary_pairs_v1/run_r1_v3_pairs_aug/best.pt",
    }
    runtime={name:prior.make_runtime(path) for name,path in models.items()}
    records=[]
    for i,case in enumerate(cases,1):
        rows={}
        for name,model in runtime.items():
            rows[name]=prior.score_case(model,case)
        records.append(dict(case=case,models=rows))
        if i%16==0:
            print("[THIRD H9 FOV PROGRESS]",i,"/64",flush=True)
    summaries={name:prior.summarize([
        r["models"][name] for r in records
    ]) for name in runtime}
    old_new={
        "third_only":sum(r["models"]["boundary_pairs_v3"]["success"]
                         and not r["models"]["baseline_v3"]["success"]
                         for r in records),
        "baseline_only":sum(r["models"]["baseline_v3"]["success"]
                            and not r["models"]["boundary_pairs_v3"]["success"]
                            for r in records),
    }
    report=dict(
        format="careplanner_h9_v3_third_round_fov_val_three_v1",
        status="COMPLETE",fov_only=True,no_los_or_self_occlusion=True,
        independent_test=False,validation_split_used_for_model_selection=True,
        same_64_frozen_val_cases=True,
        source_dataset_labels_sha256=manifest["labels_sha256"],
        checkpoint_sha256={name:prior.sha(path) for name,path in models.items()},
        summary=summaries,paired_old_vs_third=old_new,
        records=records,
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix="."+output.name+".",dir=output.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(report,f,indent=2,allow_nan=False,sort_keys=True)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(temp,output)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    for name,result in summaries.items():
        print("[THIRD H9 FOV RESULT]",name,
              f"{result['success_count']}/64",
              f"rate={result['success_rate']:.4f}",flush=True)
    print("[THIRD H9 FOV PAIRED]",json.dumps(old_new,sort_keys=True),flush=True)
    print("[THIRD H9 FOV COMPLETE]",output,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    args=p.parse_args()
    run(args.root)


if __name__=="__main__":
    try:main()
    except (RuntimeError,ValueError,KeyError,OSError,FileNotFoundError) as e:
        raise SystemExit(str(e))
