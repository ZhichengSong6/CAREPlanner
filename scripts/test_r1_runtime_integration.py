#!/usr/bin/env python3
"""Pure offline real-R1 equivalence through the production model loader.

The pinned training reference is a test-only import. No ROS nodes, geometry
queries, optimization, or execution are started here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/care_visibility_cdf/scripts"))
sys.path.insert(0, str(REPO / "scripts/r1_model_validation"))

import per_sensor_visibility_runtime as runtime
import r1_visibility_model as r1
import reference_model_r012 as reference
from verify_r1_definition import run_checks, value_gradient


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(fn, error, message):
    try:
        fn()
    except error as exc:
        require(message in str(exc), str(exc))
        return str(exc)
    raise AssertionError("Expected " + message)


def weight_inventory():
    return {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for package in ("care_visibility_cdf", "care_collision_cdf")
        for p in sorted((REPO / "src" / package / "checkpoints").rglob("*.pt"))
    }


def run(checkpoint, device):
    before = weight_inventory()
    definition = run_checks(str(checkpoint), device)
    device = torch.device(device)
    # Fail if the R1 branch accidentally deserializes through the generic path,
    # or loads twice. Both wrappers still execute the actual production code.
    with patch.object(runtime, "torch_load_checkpoint",
                      side_effect=AssertionError("R1 used generic loader")), \
            patch.object(torch, "load", wraps=torch.load) as load:
        model, ckpt = runtime.build_per_sensor_model(str(checkpoint), device)
    require(load.call_count == 1, "R1 checkpoint must deserialize exactly once")
    require(isinstance(model, r1.HierarchicalSensorView), "Wrong returned view")
    require(isinstance(model.full_model, r1.PrivateTailCDF), "Wrong R1 class")
    require(model.full_model.parameter_count() == r1.R1_PARAMETERS,
            "Wrong parameter count")
    require(not model.training and not model.full_model.training, "Not eval")
    require(all(not p.requires_grad for p in model.parameters()), "Not frozen")
    require(ckpt["runtime_adapter"] == {
        "model_type": "hierarchical9_r1_sensor_view",
        "checkpoint_sha256": r1.R1_SHA256,
        "output_semantics": "per_sensor_signed_visibility_cdf",
        "num_sensor_outputs": 8,
    }, "Incorrect runtime metadata")
    ref = reference.build_model("R1")
    incompatible = ref.load_state_dict(ckpt["model_state"], strict=True)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys,
            "Reference strict load failed")
    ref = ref.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    require({k: tuple(v.shape) for k, v in ref.state_dict().items()} ==
            {k: tuple(v.shape) for k, v in model.full_model.state_dict().items()},
            "Key/shape mismatch")
    cases = []
    for seed in (0, 41, 260921):
        # Identical inputs to the handoff's standalone checks above.
        gen = torch.Generator(device="cpu").manual_seed(seed + 177)
        x = (torch.rand((17, 3), generator=gen) * .6 +
             torch.tensor([-.3, -.3, .1])).to(device)
        q = (torch.rand((1, 7), generator=gen) * 2 - 1).to(device)
        inp = torch.cat((x, q.expand(len(x), -1)), dim=-1)
        full, full_ref, eight = model.full_model(inp), ref(inp), model(inp)
        require(full.shape == (17, 9) and eight.shape == (17, 8), "Output shape")
        require(full.dtype == eight.dtype == torch.float32, "Output not FP32")
        torch.testing.assert_close(model(inp), eight, rtol=0, atol=0)
        value_error = (full - full_ref).abs().max().item()
        view_error = (eight - full_ref[:, 1:9]).abs().max().item()
        sensors = []
        for sensor in range(8):
            vr, gr = value_gradient(ref, x, q, sensor + 1)
            va, ga = value_gradient(model, x, q, sensor)
            require(ga.shape == (1, 7) and torch.isfinite(ga).all().item(),
                    "Invalid raw 7D q-gradient")
            error = (gr - ga).abs().max().item()
            require(error <= 3e-6 and (vr - va).abs().item() <= 2e-6,
                    "Runtime raw-gradient/min-value mismatch")
            sensors.append({"sensor": sensor, "min_value_abs_error":
                            (vr - va).abs().item(), "raw_q_gradient_max_abs_error":
                            error, "raw_q_gradient_norm": ga.norm().item()})
        q2 = q.clone()
        q2[:, 0] += .07
        delta = (model(torch.cat((x, q2.expand(len(x), -1)), dim=-1)) -
                 eight).abs().max().item()
        require(max(value_error, view_error) <= 2e-6 and delta > 1e-8,
                "Runtime output/recompute check failed")
        cases.append({"seed": seed, "full_output_max_abs_error": value_error,
                      "view_max_abs_error": view_error, "q_recompute_delta": delta,
                      "sensors": sensors})

    # Metadata/hash failures and missing weights must not be silently repaired.
    with patch.object(r1, "R1_SHA256", "0" * 64), \
            patch.object(torch, "load", side_effect=AssertionError("Unpickled")):
        expect_error(lambda: r1.load_r1_checkpoint(checkpoint, device),
                     ValueError, "SHA256 mismatch")
    with patch.object(runtime, "_checkpoint_sha256", return_value="0" * 64), \
            patch.object(runtime, "torch_load_checkpoint", return_value=ckpt):
        expect_error(lambda: runtime.build_per_sensor_model(str(checkpoint), device),
                     RuntimeError, "R1 checkpoint SHA256 mismatch")
    for key, value in (("arm", "R2"), ("completed", False), ("step", 49999)):
        expect_error(lambda: r1._validate_checkpoint(dict(ckpt, **{key: value})),
                     ValueError, "Invalid R1 metadata " + key)
    missing = dict(ckpt["model_state"])
    missing.pop("sensor_tails.0.0.weight")
    expect_error(lambda: r1.PrivateTailCDF().load_state_dict(missing, strict=True),
                 RuntimeError, "Missing key")

    legacy = []
    for folder, expected in (
        ("hierarchical9_scratch_seed0", "hierarchical9_v1_sensor_view"),
        ("per_sensor_e2e_fullbatch_seed0", "legacy_yiming_8head"),
    ):
        path = REPO / "src/care_visibility_cdf/checkpoints" / folder / "final.pt"
        old, old_ckpt = runtime.build_per_sensor_model(str(path), device)
        pred = old(inp)
        require(pred.shape == (17, 8) and torch.isfinite(pred).all().item(),
                "Legacy loader regression")
        require(old_ckpt["runtime_adapter"]["model_type"] == expected,
                "Legacy dispatch changed")
        legacy.append(old_ckpt["runtime_adapter"])
    require(weight_inventory() == before, "Checkpoint inventory changed")
    return {
        "status": "PASS", "checkpoint_path": str(checkpoint),
        "device": str(device), "torch": str(torch.__version__),
        "tf32": False, "amp": False, "dtype": "float32",
        "standalone_reference_checks": definition,
        "production_loader": {"metadata": ckpt["runtime_adapter"],
                              "deserialization_count": load.call_count,
                              "state_dict_tensor_count": len(ref.state_dict()),
                              "parameter_count": r1.R1_PARAMETERS, "cases": cases},
        "negative_checks": "PASS", "legacy_loader_smoke": legacy,
        "weights_unchanged": before,
        "qualification": "MODEL_EQUIVALENCE_ONLY; NOT R1_RUNTIME_QUALIFIED",
        "not_run": ["FOV/LOS", "solver", "ROS/Gazebo", "VBC/GCDF", "tracker",
                    "robot execution", "Q2"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO /
                        "src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse to overwrite report")
    report = run(args.checkpoint.resolve(), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "device": report["device"], "checkpoint": report["checkpoint_path"]}))


if __name__ == "__main__":
    main()
