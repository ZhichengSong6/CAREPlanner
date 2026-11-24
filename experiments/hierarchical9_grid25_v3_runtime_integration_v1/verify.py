#!/usr/bin/env python3
"""CPU/GPU proof that the formal Grid25 V3 H9 best.pt matches CARE runtime.

Read-only. Reuses the existing per_sensor_visibility_runtime.py loader and
original R1 training model architecture, checks 8-branch values and q-Jacobians.
No ROS/Gazebo, planner commands, dataset production or model training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME_SCRIPTS = REPO / "src/care_visibility_cdf/scripts"
MODEL_PATH = REPO / "experiments/hierarchical9_scratch50k_r012_v1/model.py"
sys.path.insert(0, str(RUNTIME_SCRIPTS))
from per_sensor_visibility_runtime import build_per_sensor_model  # noqa: E402

ROOT = Path(
    "/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/"
    "outputs/mainline_b/h9_grid25_data_v1"
)
Q = np.asarray([
    -0.23993935822373746, 0.7577102185084499, -0.301829051647589,
    -1.626700836259844, 0.19068393720573518, -0.18052088294007795,
    -0.3808012222262862,
], dtype=np.float32)
X = np.asarray([0.10, 0.05, 0.15], dtype=np.float32)


def require(condition, message):
    if not condition:
        raise RuntimeError("[RUNTIME CHECK FAILED] " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inputs(device):
    # Controlled perturbations of the measured Case026 seed. All 8 sensor
    # heads use the same x and q, unlike a min over sensor branches.
    delta = torch.tensor([
        [0.0]*7,
        [0.007, -0.003, 0.009, -0.005, 0.003, 0.004, -0.002],
        [-0.005, 0.008, -0.003, 0.009, -0.004, 0.006, 0.004],
        [0.002, 0.001, -0.008, -0.003, 0.004, -0.006, 0.005],
    ], device=device, dtype=torch.float32)
    q = torch.as_tensor(Q, device=device).reshape(1, 7)+delta
    x = torch.as_tensor(X, device=device).reshape(1, 3).expand(4, 3).clone()
    x[1, 0] += 0.007
    x[2, 1] -= 0.006
    x[3, 2] += 0.008
    return x, q


def run(root, device, out):
    if device.type == "cuda":
        require(torch.cuda.is_available(), "Requested CUDA but no CUDA available")
    p = root / "derived/grid25_v3_full_training_v1/run_r1_v3_full"
    checkpoint = p / "best.pt"
    run_info = json.loads((p / "run.json").read_text())
    require(run_info.get("status") == "COMPLETE"
            and run_info.get("best_step") == 19500
            and run_info.get("successful_updates") == 20000,
            "Expected completed formal 20,000-step run with best step 19,500")
    require(sha(MODEL_PATH) == run_info["model_source_sha256"],
            "Training reference model source changed")

    start = time.perf_counter()
    # Actual production runtime dispatch, including checkpoint provenance.
    sensor_view, ckpt = build_per_sensor_model(str(checkpoint), device)
    load_seconds = time.perf_counter()-start
    require(ckpt["step"] == run_info["best_step"],
            "Runtime loaded a checkpoint from the wrong step")
    require(hasattr(sensor_view, "full_model"),
            "Runtime did not return 9-head H9 wrapped as sensor view")
    full = sensor_view.full_model
    require(full.parameter_count() == run_info["model_architecture"]["total_parameters"],
            "Runtime R1 parameter count changed")

    spec=importlib.util.spec_from_file_location(
        "grid25_original_r1_model_runtime_verification_v1", MODEL_PATH)
    require(spec and spec.loader, "Original training R1 model unavailable")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    reference=module.PrivateTailCDF().to(device)
    reference.load_state_dict(ckpt["model_state"], strict=True)
    reference.eval().requires_grad_(False)
    x, q=inputs(device)
    inp=torch.cat((x,q),dim=1)
    with torch.no_grad():
        expected=reference(inp)
        actual_full=full(inp)
        actual_eight=sensor_view(inp)
        max_full=float((expected-actual_full).abs().max().item())
        max_sensor=float((expected[:,1:]-actual_eight).abs().max().item())
        require(expected.shape==(4,9) and actual_eight.shape==(4,8),
                "Union/sensor shape or offset is wrong")
        require(bool(torch.isfinite(actual_eight).all().item()),
                "Runtime produced nonfinite values")
        require(torch.allclose(expected,actual_full,rtol=1e-5,atol=5e-6)
                and torch.allclose(expected[:,1:],actual_eight,rtol=1e-5,atol=5e-6),
                "Runtime 8-head view is not exactly training heads 1..8")
        # A different q should give a different model value for at least one
        # sensor: this prevents accidentally caching stale q features.
        require(float((actual_eight[1]-actual_eight[0]).abs().max())>1e-7,
                "Model appears insensitive to changing x and q")

    branch_gradients=[]
    for sid in range(8):
        q_ref = q[0:1].detach().clone().requires_grad_(True)
        q_run = q[0:1].detach().clone().requires_grad_(True)
        inp_ref=torch.cat((x[0:1],q_ref),dim=1)
        inp_run=torch.cat((x[0:1],q_run),dim=1)
        value_ref=reference(inp_ref)[:,sid+1].sum()
        value_run=sensor_view(inp_run)[:,sid].sum()
        grad_ref=torch.autograd.grad(value_ref,q_ref)[0]
        grad_run=torch.autograd.grad(value_run,q_run)[0]
        err=float((grad_ref-grad_run).abs().max().item())
        require(bool(torch.isfinite(grad_run).all().item())
                and torch.allclose(grad_ref,grad_run,atol=2e-5,rtol=2e-4),
                f"Sensor S{sid} 7D q gradient differs from training reference")
        branch_gradients.append(dict(
            sensor=f"S{sid}",
            value=float(value_run.detach().item()),
            q_gradient_norm=float(torch.linalg.vector_norm(grad_run).item()),
            max_reference_gradient_error=err,
        ))

    report=dict(
        format="grid25_v3_h9_runtime_check_v1",
        status="PASS",
        diagnostic_only=True,
        no_automatic_planner_control=True,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),
        best_step=ckpt["step"],
        validation_rmse=float(ckpt["val_sensor_rmse"]),
        device=str(device),
        runtime_load_seconds=load_seconds,
        model_parameter_count=full.parameter_count(),
        max_nine_head_value_error=max_full,
        max_sensor_view_value_error=max_sensor,
        q_gradient_checks=branch_gradients,
        no_independent_fd_gradient_certificate=True,
        no_los_or_collision_certificate=True,
    )
    out=out.expanduser().resolve()
    require(not out.exists(),f"Report already exists: {out}; refuse overwrite")
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix="."+out.name+".",dir=out.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(report,f,indent=2,allow_nan=False)
            f.write("\n")
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,out)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    print("[GRID25 V3 H9 RUNTIME CHECK PASS]",out,flush=True)
    print(json.dumps({
        k:v for k,v in report.items() if k!="q_gradient_checks"
    },indent=2),flush=True)
    for row in branch_gradients:
        print(row,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    out=args.out or (
        args.root / "derived/grid25_v3_runtime_integration_v1" /
        f"runtime_check_{args.device}.json"
    )
    run(args.root.resolve(),torch.device(args.device),out)


if __name__ == "__main__":
    try:main()
    except (RuntimeError,ValueError,KeyError,FileNotFoundError,OSError) as e:
        raise SystemExit(str(e))
