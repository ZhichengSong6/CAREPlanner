#!/usr/bin/env python3
"""One focused Case026 diagnostic + S4 geometry-guided multistart rescue.

Uses the already-frozen H9 Case026 failure and a separately discovered
geometry witness. The witness is evaluated AFTER search, never used as a seed.

Only move joints in the S4 sensor kinematic ancestor chain, since those are
the joints the V3 FOV gradient is allowed to move. Actual conservative FOV
and primitive self-occlusion are the acceptance gates. If a candidate is
found it is *not* trajectory/GCDF/VBC certified.

No change to production defaults, checkpoint, loss or training data.
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
from scipy.stats import qmc
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(HERE))
from geometry_case026 import (  # noqa: E402
    GeometryOnlyCase026, TARGET, MEASURED_SEED, KNOWN_BLOCKED_QVIS,
    Q_MIN, Q_MAX,
)
from per_sensor_visibility_runtime import build_per_sensor_model  # noqa: E402
from evaluate_direct_vs_projection_ascent import DEFAULT_JOINT_NAMES  # noqa: E402

SENSOR=4


def require(condition,message):
    if not condition:
        raise RuntimeError("[CASE026 S4 REPAIR STOP] "+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def best_path(root):
    return root/"derived/grid25_v3_full_training_v1/run_r1_v3_full/best.pt"


def model_diagnosis(model,q):
    row=np.concatenate((TARGET,np.asarray(q,dtype=np.float64)))
    inp=torch.as_tensor(row,dtype=torch.float32).reshape(1,10)
    inp.requires_grad_(True)
    pred=model(inp)[0,SENSOR]
    grad=torch.autograd.grad(pred,inp)[0][0,3:]
    return dict(
        h9_s4_predicted_value=float(pred.detach()),
        h9_s4_q_gradient=[float(v) for v in grad.detach().numpy()],
        h9_s4_q_gradient_norm=float(torch.linalg.vector_norm(grad).item()),
    )


def run(args):
    require(0<=args.seed<2**32
            and 0<=args.local_samples<=100000
            and 0<=args.global_samples<=100000,
            "Invalid bounded sampling controls")
    root=args.root.expanduser().resolve()
    previous=root/"derived/grid25_v3_runtime_integration_v1/case026/grid25_v3_h9.json"
    geometry_report=root/"derived/grid25_v3_runtime_integration_v1/case026_geometry_only.json"
    proof=root/"derived/grid25_v3_runtime_integration_v1/runtime_check_cpu.json"
    output=args.output or (
        root/"derived/grid25_v3_runtime_integration_v1/case026_s4_repair.json"
    )
    output=Path(output).expanduser().resolve()
    require(not output.exists(), "Output exists; refuse overwrite: "+str(output))
    for p in (previous,geometry_report,proof,best_path(root)):
        require(p.is_file(), "Previous H9/geometry/checkpoint artifact missing: "+str(p))
    h9=read(previous)
    reference=read(geometry_report)
    verification=read(proof)
    checkpoint=best_path(root)
    require(np.allclose(np.asarray(h9["target_xyz"],float),TARGET,rtol=0,atol=1e-8)
            and np.allclose(np.asarray(h9["measured_seed_q"],float),MEASURED_SEED,
                            rtol=0,atol=1e-8)
            and reference.get("verdict")=="FOUND_FOV_AND_PRIMITIVE_LOS_CANDIDATE"
            and reference["selected"]["sensor_id"]==SENSOR,
            "Mismatched Case026 or missing independent S4 geometry witness")
    require(verification.get("status")=="PASS"
            and verification.get("checkpoint_sha256")==sha(checkpoint),
            "H9 checkpoint differs from previously verified model")
    s4=next((a for a in h9["per_sensor"]["attempts"]
             if int(a["sensor_id"])==SENSOR),None)
    require(s4 is not None, "Original H9 Case026 did not attempt S4")
    oracle=GeometryOnlyCase026(
        REPO/"src/arm_description/urdf/Arm.urdf",
        REPO/"src/arm_description/urdf/Arm_with_self_filter_collision.urdf",
    )
    chain_names={j.name for j in oracle.sensor_chains[SENSOR]}
    active=np.asarray(
        [i for i,name in enumerate(DEFAULT_JOINT_NAMES) if name in chain_names],
        dtype=np.int64,
    )
    inactive=np.asarray(
        [i for i in range(7) if i not in set(active.tolist())],dtype=np.int64
    )
    require(1<=len(active)<=7,"S4 sensor chain has no movable joints")
    known=oracle.check(KNOWN_BLOCKED_QVIS,SENSOR)
    require(known["any_primitive_self_occluded"]
            and abs(known["min_conservative_g"]-0.018168)<0.003,
            "Frozen historical S4 geometry sanity failed")
    witness_q=np.asarray(reference["selected"]["q"],dtype=float)
    witness_geometry=oracle.check(witness_q,SENSOR)
    require(witness_geometry["accepted"],"Prior independent geometry witness no longer passes")
    h9_q=np.asarray(s4["q_candidate"],dtype=float)
    h9_q_zero=np.asarray(s4["q_zero"],dtype=float)
    require(h9_q.shape==(7,) and h9_q_zero.shape==(7,)
            and np.isfinite(h9_q).all(),
            "Original H9 S4 q malformed")

    # Trusted model loader verifies checkpoint SHA and training provenance.
    model,ckpt=build_per_sensor_model(str(checkpoint),torch.device("cpu"))
    require(int(ckpt["step"])==19500 and
            getattr(model,"full_model",None) is not None,
            "New V3 H9 best.pt not loaded correctly")
    observations={}
    for label,q in (
        ("measured",MEASURED_SEED),
        ("old_scalar_blocked",KNOWN_BLOCKED_QVIS),
        ("h9_s4_zero",h9_q_zero),
        ("h9_s4_candidate",h9_q),
        ("independent_geometry_witness",witness_q),
    ):
        actual=oracle.check(q,SENSOR)
        observations[label]=dict(
            q=[float(v) for v in q],
            geometry=actual,
            **model_diagnosis(model,q),
        )
    del model

    print("[CASE026 S4 DIAGNOSIS]",
          "H9 candidate g=",round(observations["h9_s4_candidate"]["geometry"]["min_conservative_g"],6),
          "occluded=",observations["h9_s4_candidate"]["geometry"]["any_primitive_self_occluded"],
          "witness f4=",round(observations["independent_geometry_witness"]["h9_s4_predicted_value"],6),
          flush=True)
    print("[CASE026 S4 ACTIVE JOINTS]",
          [DEFAULT_JOINT_NAMES[i] for i in active],
          "inactive=", [DEFAULT_JOINT_NAMES[i] for i in inactive],flush=True)

    start=time.monotonic()
    counts=dict(q_tested=0,s4_fov_pass=0,s4_los_blocked_after_fov=0,
                accepted=0)
    found=None

    def evaluate(label,q):
        nonlocal found
        q=np.asarray(q,dtype=np.float64).reshape(7)
        require(np.isfinite(q).all()
                and np.all(q>=Q_MIN-1e-9) and np.all(q<=Q_MAX+1e-9),
                "Rescue q is nonfinite or outside joint limits")
        counts["q_tested"]+=1
        # Use unchanged original Case026 predicate. This is deliberately
        # not a relaxed FOV margin or inferred neural feasibility.
        geom=oracle.check(q,SENSOR)
        if geom["min_conservative_g"]>=0.0:
            counts["s4_fov_pass"]+=1
            if geom["any_primitive_self_occluded"]:
                counts["s4_los_blocked_after_fov"]+=1
        if geom["accepted"]:
            found=dict(source=label,q=[float(x) for x in q],geometry=geom)
            counts["accepted"]+=1
            return True
        return False

    # H9 proposal is first. The other two seeds are explicitly named and
    # cannot be interpreted as H9-generated improvements.
    anchors=[
        ("H9_S4_candidate",h9_q),
        ("H9_S4_zero",h9_q_zero),
        ("old_scalar_blocked",KNOWN_BLOCKED_QVIS),
        ("measured",MEASURED_SEED),
    ]
    for label,q in anchors:
        if evaluate(label,q):
            break

    rng=np.random.default_rng(args.seed)
    radii=(0.15,0.4,0.9)
    if found is None:
        for i in range(args.local_samples):
            label,center=anchors[(i//len(radii))%len(anchors)]
            radius=radii[i%len(radii)]
            q=np.asarray(center,dtype=float).copy()
            q[active]=np.clip(
                q[active]+rng.normal(0,radius,len(active)),
                Q_MIN[active],Q_MAX[active],
            )
            if evaluate(f"local_{i:04d}_around_{label}_radius{radius}",q):
                break
    if found is None and args.global_samples:
        sobol=qmc.Sobol(d=len(active),scramble=True,seed=args.seed)
        m=int(math.ceil(math.log2(args.global_samples)))
        for i,unit in enumerate(sobol.random_base2(m=m)[:args.global_samples]):
            # Keep excluded wrist joints frozen to original measured q. That
            # isolates the ancestor joint search missed by original H9.
            q=np.asarray(MEASURED_SEED,dtype=float).copy()
            q[active]=Q_MIN[active]+(Q_MAX[active]-Q_MIN[active])*unit
            if evaluate(f"sobol_S4_ancestors_{i:04d}",q):
                break

    report=dict(
        format="careplanner_case026_s4_geometry_guided_rescue_v1",
        status="COMPLETE",
        verdict=("RESCUE_FOUND_S4_FOV_AND_PRIMITIVE_LOS" if found is not None
                 else "NO_RESCUE_WITHIN_BUDGET"),
        best_checkpoint=str(checkpoint),
        best_checkpoint_sha256=verification["checkpoint_sha256"],
        original_h9_case026_verdict=h9["verdict"],
        original_h9_case026_branch_count=len(h9["per_sensor"]["attempts"]),
        prior_geometry_witness_used_as_search_seed=False,
        old_8head_comparison_performed=False,
        geometry_gates_unchanged=True,
        target_xyz=[float(x) for x in TARGET],
        sensor_id=SENSOR,
        active_joint_indices=[int(i) for i in active],
        active_joint_names=[DEFAULT_JOINT_NAMES[i] for i in active],
        inactive_joint_names=[DEFAULT_JOINT_NAMES[i] for i in inactive],
        observations=observations,
        search=dict(seed=args.seed,local_samples=args.local_samples,
                    global_samples=args.global_samples,
                    moved_joints="S4 kinematic ancestors only",
                    reference_witness_never_a_search_seed=True),
        counts=counts,
        selected=found,
        elapsed_s=time.monotonic()-start,
        limitations=[
            "H9 distance regression values do not prove gradients point to LOS-clear solutions",
            "Conservative FOV and primitive LOS only; not trajectory/GCDF/VBC safe",
            "Geometry-guided random/Sobol rescue is not proof of superior network quality",
            "No old-8head checkpoint was available for paired new-versus-old evaluation",
            "Finite-budget search failure does not prove infeasibility",
        ],
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=output.parent,prefix="."+output.name+".")
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(report,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,output)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print("[CASE026 S4 REPAIR]",report["verdict"],flush=True)
    print("[CASE026 S4 REPAIR COUNTS]",json.dumps(counts,sort_keys=True),flush=True)
    if found is not None:
        print("[CASE026 S4 REPAIR CANDIDATE]",json.dumps(dict(
            q=found["q"],margin=found["geometry"]["min_conservative_g"],
            source=found["source"]),indent=2),flush=True)
    print("[CASE026 S4 REPAIR OUTPUT]",output,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--seed",type=int,default=260927)
    p.add_argument("--local-samples",type=int,default=256)
    p.add_argument("--global-samples",type=int,default=1024)
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    try:main()
    except (RuntimeError,FileNotFoundError,OSError,ValueError,KeyError) as exc:
        raise SystemExit(str(exc))
