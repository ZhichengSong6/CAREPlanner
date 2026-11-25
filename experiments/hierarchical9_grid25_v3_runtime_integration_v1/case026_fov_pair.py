#!/usr/bin/env python3
"""One controlled, FOV-only Case026 comparison of the two formal H9 R1 runs.

Same original scalar q_zero, measured branch seed, S4-first ranking budget,
R1 runtime projection/root/ascent, URDF, conservative FOV and joint limits.
Only swaps full-Grid25 V3 distance best.pt vs boundary-weighted V3 best.pt.

IMPORTANT: No LOS/self-occlusion predicate is evaluated or required here.
This checks *FOV candidate generation*, not full CAREPlanner safety.
No retraining, Slurm, GPU, robot motion, or model-dependent seed selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO/"scripts"))
sys.path.insert(0, str(REPO/"src/care_visibility_cdf/scripts"))

from test_phase_e_case026_targeted_per_sensor_fallback import (  # noqa: E402
    TARGET, MEASURED_SEED, Q_MIN, Q_MAX,
)
from per_sensor_visibility_runtime import (  # noqa: E402
    PerSensorVisibilityRuntime,
)

ROOT_DEFAULT=Path("/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1")


def require(ok,message):
    if not ok:
        raise RuntimeError("[CASE026 FOV PAIR STOP] "+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_case(root):
    file=root/"derived/grid25_v3_runtime_integration_v1/case026/grid25_v3_h9.json"
    require(file.is_file(), "Original Case026 q_zero trace missing: "+str(file))
    old=json.loads(file.read_text())
    require(np.allclose(old["target_xyz"],TARGET,atol=1e-8,rtol=0)
            and np.allclose(old["measured_seed_q"],MEASURED_SEED,atol=1e-8,rtol=0),
            "Original Case026 target/seed changed")
    q_zero=np.asarray(old["scalar"]["q_zero"],dtype=np.float64).reshape(7)
    require(np.isfinite(q_zero).all() and
            np.all(q_zero>=Q_MIN-1e-8) and np.all(q_zero<=Q_MAX+1e-8),
            "Original scalar q_zero is invalid")
    return q_zero


def one_model(label,ckpt,points,q_seed,q_zero):
    # Use exactly the existing H9 branch optimizer; only disable LOS.
    runtime=PerSensorVisibilityRuntime(
        checkpoint_path=str(ckpt),
        reference_urdf_path=str(REPO/"src/arm_description/urdf/Arm.urdf"),
        self_filter_urdf_path=str(REPO/"src/arm_description/urdf/Arm_with_self_filter_collision.urdf"),
        device=torch.device("cpu"),
        q_min=Q_MIN,q_max=Q_MAX,
        projection_iters=10,
        projection_damping=0.5,
        projection_epsilon_f=0.03,
        projection_max_step_norm=0.25,
        root_refine_iters=12,
        root_tolerance_f=0.002,
        branch_ascent_steps=1,
        branch_step_size=0.05,
        branch_max_step_norm=0.25,
        branch_fallback_ascent_steps=8,
        max_branch_attempts=8,
        min_conservative_g=0.0,
        conservative_hfov_deg=50.0,
        conservative_vfov_deg=66.0,
        conservative_z_min=0.20,
        conservative_z_max=0.70,
        conservative_delta=0.01,
        require_primitive_los=False,
    )
    x=torch.as_tensor(points,dtype=torch.float32)
    q0=torch.as_tensor(q_zero.reshape(1,7),dtype=torch.float32)
    start=torch.as_tensor(q_seed.reshape(1,7),dtype=torch.float32)
    scores,rank=runtime.rank_sensors(x,q0)
    order=[4]+[int(s) for s in rank if int(s)!=4]
    start_t=time.perf_counter()
    attempts=[]
    first=None
    for sid in order:
        # Every branch gets the identical original measured seed.
        branch=runtime._optimize_branch(x,start,sid)
        actual=runtime._candidate_geometry(
            points,branch["q_candidate"],sid
        )
        require(not actual["any_primitive_self_occluded"],
                "FOV-only mode unexpectedly performed LOS checking")
        passed=bool(actual["min_conservative_g"]>=0.0)
        require(passed==actual["accepted"],
                "FOV-only acceptance differs from conservative analytic margin")
        record=dict(
            sensor_id=sid,
            initial_h9_score=branch["initial_score"],
            final_h9_score=branch["final_score"],
            conservative_fov_margin=actual["min_conservative_g"],
            fov_pass=passed,
            root_source=branch["root_source"],
            q_candidate=branch["q_candidate"],
            selected_source="identical_measured_q",
        )
        attempts.append(record)
        if passed and first is None:
            first=dict(sensor_id=sid,q=branch["q_candidate"],
                       conservative_fov_margin=actual["min_conservative_g"])
    elapsed=time.perf_counter()-start_t
    result=dict(
        checkpoint=str(ckpt),
        checkpoint_sha256=sha(ckpt),
        checkpoint_step=int(runtime.checkpoint["step"]),
        ranking=[int(s) for s in rank],
        attempted_order=order,
        scores_at_scalar_q_zero=[float(v) for v in scores],
        attempts=attempts,
        fov_pass_count=sum(int(a["fov_pass"]) for a in attempts),
        first_fov_pass=first,
        elapsed_seconds=elapsed,
    )
    print("[CASE026 FOV-ONLY]",label,
          "passing_branches=",result["fov_pass_count"],
          "first_success_S=",(-1 if first is None else first["sensor_id"]),
          "S4_margin=",next(a["conservative_fov_margin"] for a in attempts if a["sensor_id"]==4),
          "branch_seconds=",round(elapsed,3),flush=True)
    return result


def run(args):
    root=args.root.expanduser().resolve()
    out=(args.output or
         root/"derived/grid25_v3_runtime_integration_v1/case026_fov_only_pair_v1.json")
    out=Path(out).expanduser().resolve()
    require(not out.exists(), "Result exists; do not overwrite: "+str(out))
    q_zero=frozen_case(root)
    first=root/"derived/grid25_v3_full_training_v1/run_r1_v3_full/best.pt"
    second=root/"derived/grid25_v3_boundary_focus_v1/run_r1_v3_boundary_focus/best.pt"
    require(first.is_file() and second.is_file(),"Both completed H9 best.pt files required")
    torch.set_num_threads(4)
    points=TARGET.reshape(1,3)
    runs={}
    for label,ckpt in (("baseline_full_v3",first),("boundary_weighted_v3",second)):
        runs[label]=one_model(label,ckpt,points,MEASURED_SEED,q_zero)
    result=dict(
        format="careplanner_case026_fov_only_h9_pair_v1",
        status="COMPLETE",
        no_self_occlusion_or_los=True,
        no_gcdf_vbc_trajectory_check=True,
        no_neural_training=True,
        same_scalar_q_zero=True,
        same_measured_branch_seed=True,
        target_xyz=TARGET.tolist(),
        q_zero=q_zero.tolist(),
        measured_q=MEASURED_SEED.tolist(),
        experiment="Original 20k H9 R1 vs boundary-weighted 20k H9 R1; identical Case026 FOV-only 8-branch search",
        runs=runs,
    )
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+out.name+".",dir=out.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print("[CASE026 FOV PAIR COMPLETE]",out,flush=True)
    print("[CASE026 FOV-ONLY LIMITATION] One case, identical branch optimizer; no LOS or robot safety certification",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT_DEFAULT)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    try:main()
    except (RuntimeError,OSError,ValueError,KeyError,FileNotFoundError) as e:
        raise SystemExit(str(e))
