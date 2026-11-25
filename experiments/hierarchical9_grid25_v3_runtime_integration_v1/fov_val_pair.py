#!/usr/bin/env python3
"""Single controlled FOV-only search comparison on frozen held-out Grid25 Val x.

Compare formal 20k baseline H9 R1 vs 20k boundary-weighted H9 R1.
- Same 64 distinct frozen Val x, one fixed q slot each, identical joint limits.
- Select cases *before* loading either model, independent of model predictions.
- Start from configurations initially outside ALL 8 analytic FOVs, so the
  success metric counts genuine visibility acquisition rather than trivial
  initial success.
- Same per-sensor projection/root/ascent algorithm and eight branch attempts.
- Ranking evaluated at the common initial q. NO scalar coarse projection:
  this isolates H9-driven search, not the whole runtime pipeline.
- Only original conservative FOV oracle is used; NO LOS, self-occlusion,
  GCDF, VBC, trajectory/robot control, or extra training.
The held-out Val split was used for checkpoint selection; this is a controlled
validation comparison, NOT an independent untouched final-test estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from urdf_parser_py.urdf import URDF

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
VIS=REPO/"src/care_visibility_cdf/scripts"
if str(VIS) not in sys.path:
    sys.path.insert(0,str(VIS))
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime  # noqa: E402
from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_JOINT_NAMES,DEFAULT_SENSOR_FRAMES,
)
from validate_visibility_oracle import (  # noqa: E402
    find_chain_joints,fk_transform,sensor_margin,
)
from check_visibility_self_occlusion import q_row_to_map  # noqa: E402

ROOT=Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")
Q_MIN=np.array([-3.14,-2.30,-3.14,-2.65,-3.14,-3.14,-1.20],dtype=np.float64)
Q_MAX=np.array([ 3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20],dtype=np.float64)
GEO_ARGS=dict(
    conservative_hfov_deg=50.0,conservative_vfov_deg=66.0,
    conservative_z_min=0.20,conservative_z_max=0.70,
    conservative_delta=0.01,
)


def require(ok,message):
    if not ok:
        raise RuntimeError("[FOV VAL PAIR STOP] "+message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


class FOVOnlyOracle:
    def __init__(self):
        path=REPO/"src/arm_description/urdf/Arm.urdf"
        require(path.is_file(),"Missing reference Arm.urdf")
        robot=URDF.from_xml_file(str(path))
        self.chains=[
            find_chain_joints(robot,"base_link",frame)
            for frame in DEFAULT_SENSOR_FRAMES
        ]

    def margins(self,x,q):
        q_map=q_row_to_map(DEFAULT_JOINT_NAMES,q)
        margins=[]
        for chain in self.chains:
            transform=fk_transform(chain,q_map)
            margin,_,_=sensor_margin(
                x,transform,50.,66.,.20,.70
            )
            margins.append(float(margin-.01))
        return np.asarray(margins,dtype=np.float64)


def fixed_cases(root,oracle,num_cases,seed):
    dataset=root/"derived/grid25_v3_full_training_v1/dataset_segments_000_123"
    manifest=read(dataset/"manifest.json")
    require(manifest.get("format")=="care_grid25_full_v3only_distance_snapshot_v1"
            and manifest.get("status")=="FROZEN"
            and manifest["counts"]["train_x"]==109490
            and manifest["counts"]["val_x"]==16711,
            "Frozen 124-segment spatial train/val identity changed")
    require(sha(dataset/"labels.npz")==manifest["labels_sha256"],
            "Frozen Grid25 labels SHA mismatch")
    with np.load(dataset/"labels.npz",allow_pickle=False) as z:
        x=np.asarray(z["x"],dtype=np.float64)
        q=np.asarray(z["q"],dtype=np.float64)
        split=np.asarray(z["split"],dtype=np.uint8)
        xid=np.asarray(z["x_index"],dtype=np.int64)
        slots=np.asarray(z["q_slot"],dtype=np.uint8)
    val=np.flatnonzero(split==1)
    require(len(val)==16711*4 and len(val)%4==0,
            "Unexpected frozen validation-query count")
    groups=val.reshape(-1,4)
    require(np.all(xid[groups]==xid[groups[:,0],None])
            and np.array_equal(slots[groups],np.tile(np.arange(4),(len(groups),1)))
            and len(np.unique(xid[groups[:,0]]))==16711,
            "Frozen Val x or q-slot grouping changed")
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(groups))
    cases=[]
    for group_index in perm:
        group=groups[int(group_index)]
        row=int(group[int(rng.integers(0,4))])
        q0=q[row].copy()
        point=x[row].copy()
        if not np.isfinite(point).all() or not np.isfinite(q0).all():
            continue
        if not np.all(q0>=Q_MIN) or not np.all(q0<=Q_MAX):
            continue
        margins=oracle.margins(point,q0)
        if float(np.max(margins))>=0:
            continue
        cases.append(dict(
            x_index=int(xid[row]),q_slot=int(slots[row]),
            point_xyz=[float(v) for v in point],
            q_initial=[float(v) for v in q0],
            max_initial_fov_margin=float(np.max(margins)),
        ))
        if len(cases)==num_cases:
            break
    require(len(cases)==num_cases,
            f"Only {len(cases)} globally FOV-invisible Val queries found")
    return cases,manifest


def make_runtime(path):
    require(path.is_file(),"Missing H9 checkpoint: "+str(path))
    return PerSensorVisibilityRuntime(
        checkpoint_path=str(path),
        reference_urdf_path=str(REPO/"src/arm_description/urdf/Arm.urdf"),
        self_filter_urdf_path=str(
            REPO/"src/arm_description/urdf/Arm_with_self_filter_collision.urdf"),
        device=torch.device("cpu"),q_min=Q_MIN,q_max=Q_MAX,
        projection_iters=10,projection_damping=0.5,
        projection_epsilon_f=0.03,projection_max_step_norm=0.25,
        root_refine_iters=12,root_tolerance_f=0.002,
        branch_ascent_steps=1,branch_step_size=0.05,
        branch_max_step_norm=0.25,branch_fallback_ascent_steps=8,
        max_branch_attempts=8,min_conservative_g=0.0,
        require_primitive_los=False,**GEO_ARGS,
    )


def score_case(model,case):
    x=np.asarray(case["point_xyz"],dtype=np.float64).reshape(1,3)
    q=np.asarray(case["q_initial"],dtype=np.float64).reshape(1,7)
    pts=torch.as_tensor(x,dtype=torch.float32)
    seed=torch.as_tensor(q,dtype=torch.float32)
    scores,ranking=model.rank_sensors(pts,seed)
    attempts=[]
    selected=None
    start=time.perf_counter()
    for sensor in ranking.tolist():
        sensor=int(sensor)
        branch=model._optimize_branch(pts,seed,sensor)
        geometry=model._candidate_geometry(x,branch["q_candidate"],sensor)
        require(not geometry["any_primitive_self_occluded"],
                "FOV-only experiment unexpectedly evaluated LOS")
        passed=bool(geometry["accepted"])
        require(passed==(float(geometry["min_conservative_g"])>=-1e-12),
                "FOV-only acceptance differs from original geometric predicate")
        attempts.append(dict(
            sensor_id=sensor,
            predicted_initial_value=float(branch["initial_score"]),
            predicted_final_value=float(branch["final_score"]),
            fov_margin=float(geometry["min_conservative_g"]),
            fov_pass=passed,
            projection_steps=max(0,len(branch["projection_history"])-1),
            ascent_steps=len(branch["ascent_history"]),
            root_source=branch["root_source"],
            q_candidate=[float(v) for v in branch["q_candidate"]],
        ))
        if passed:
            selected=dict(
                sensor_id=sensor,
                rank=len(attempts),
                q=[float(v) for v in branch["q_candidate"]],
                conservative_fov_margin=float(geometry["min_conservative_g"]),
                projection_steps=attempts[-1]["projection_steps"],
                ascent_steps=attempts[-1]["ascent_steps"],
            )
            break
    return dict(
        success=selected is not None,
        selected=selected,
        attempts=attempts,
        ranking=[int(v) for v in ranking],
        scores_at_common_initial_q=[float(v) for v in scores],
        elapsed_s=float(time.perf_counter()-start),
    )


def summarize(rows):
    good=[r for r in rows if r["success"]]
    return dict(
        num_cases=len(rows),
        success_count=len(good),
        success_rate=len(good)/len(rows),
        mean_attempts_to_first_success=(
            float(np.mean([r["selected"]["rank"] for r in good])) if good else None),
        mean_projection_steps_on_success=(
            float(np.mean([r["selected"]["projection_steps"] for r in good])) if good else None),
        mean_accepted_fov_margin=(
            float(np.mean([r["selected"]["conservative_fov_margin"] for r in good])) if good else None),
        mean_cpu_case_seconds=float(np.mean([r["elapsed_s"] for r in rows])),
    )


def run(args):
    require(8<=args.cases<=256 and 0<=args.seed<2**32,
            "Use 8..256 fixed validation cases and a valid seed")
    out=(args.output or (
        args.root/"derived/grid25_v3_runtime_integration_v1/"
        "fov_val_baseline_vs_boundary_v1.json"
    )).expanduser().resolve()
    require(not out.exists(),"Report already exists; refuse overwrite: "+str(out))
    torch.set_num_threads(4)
    oracle=FOVOnlyOracle()
    print("[FOV VAL PAIR] Selecting model-independent held-out Val cases",flush=True)
    cases,manifest=fixed_cases(args.root,oracle,args.cases,args.seed)
    print("[FOV VAL PAIR] Fixed globally invisible seeds:",
          len(cases),flush=True)
    paths={
        "baseline_v3":args.root/"derived/grid25_v3_full_training_v1/run_r1_v3_full/best.pt",
        "boundary_weighted_v3":args.root/"derived/grid25_v3_boundary_focus_v1/run_r1_v3_boundary_focus/best.pt",
    }
    models={name:make_runtime(p) for name,p in paths.items()}
    records=[]
    for index,case in enumerate(cases):
        comparison={
            name:score_case(model,case)
            for name,model in models.items()
        }
        records.append(dict(case=case,models=comparison))
        if (index+1)%16==0 or index+1==len(cases):
            counts={name:sum(int(r["models"][name]["success"]) for r in records)
                    for name in models}
            print("[FOV VAL PAIR PROGRESS]",index+1,"/",len(cases),
                  "successes=",json.dumps(counts,sort_keys=True),flush=True)

    summaries={
        name:summarize([r["models"][name] for r in records])
        for name in models
    }
    a=summaries["baseline_v3"]["success_count"]
    b=summaries["boundary_weighted_v3"]["success_count"]
    both=sum(r["models"]["baseline_v3"]["success"] and
             r["models"]["boundary_weighted_v3"]["success"] for r in records)
    new_only=sum(not r["models"]["baseline_v3"]["success"] and
                 r["models"]["boundary_weighted_v3"]["success"] for r in records)
    old_only=sum(r["models"]["baseline_v3"]["success"] and
                 not r["models"]["boundary_weighted_v3"]["success"] for r in records)
    result=dict(
        format="careplanner_grid25_h9_fov_val_pair_v1",
        status="COMPLETE",
        sample_definition="One q slot per distinct Val x; initial q outside all eight analytic conservative FOVs",
        selection_independent_of_both_models=True,
        validation_split_used_for_model_selection=True,
        independent_test=False,
        fov_only=True,no_los_or_self_occlusion=True,no_robot_control=True,
        seed=args.seed,cases_requested=args.cases,
        frozen_dataset_sha256=manifest["labels_sha256"],
        checkpoint_sha256={name:sha(path) for name,path in paths.items()},
        same_budget=dict(
            ranking_at="same initial q",scalar_coarse_projection=False,
            per_sensor_projection_iters=10,root_refine_iters=12,
            ascent_steps=1,fallback_ascent_steps=8,
            sensor_attempt_budget=8,
            conservative_fov=GEO_ARGS,
        ),
        summaries=summaries,
        paired=dict(both_success=both,boundary_only_success=new_only,
                    baseline_only_success=old_only,
                    neither_success=len(records)-both-new_only-old_only,
                    success_count_difference=b-a),
        records=records,
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=out.parent,prefix="."+out.name+".")
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(result,f,indent=2,allow_nan=False)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print("[FOV VAL PAIR COMPLETE]",out,flush=True)
    for name,data in summaries.items():
        print("[FOV VAL RESULT]",name,
              f"success={data['success_count']}/{data['num_cases']}",
              "rate=",round(data["success_rate"],4),
              "mean_attempts_on_success=",data["mean_attempts_to_first_success"],
              flush=True)
    print("[FOV VAL PAIRED]",json.dumps(result["paired"],sort_keys=True),flush=True)
    print("[FOV VAL LIMIT] Validation comparison, not an independent test; no LOS or collision evaluation",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path)
    p.add_argument("--cases",type=int,default=64)
    p.add_argument("--seed",type=int,default=260930)
    args=p.parse_args()
    args.root=args.root.expanduser().resolve()
    run(args)


if __name__=="__main__":
    try:main()
    except (RuntimeError,KeyError,OSError,ValueError,FileNotFoundError) as exc:
        raise SystemExit(str(exc))
