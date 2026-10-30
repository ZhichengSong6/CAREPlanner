#!/usr/bin/env python3
"""A1 regression tests for legacy-8/H9 per-sensor runtime compatibility."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
VIS_SCRIPTS = REPO / "src" / "care_visibility_cdf" / "scripts"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(VIS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(VIS_SCRIPTS))

from experiments.hierarchical9_scratch_v1.model import (  # noqa: E402
    HierarchicalVisibilityCDF as TrainingHierarchicalVisibilityCDF,
)
from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    DEFAULT_SENSOR_FRAMES,
)
from per_sensor_visibility_runtime import (  # noqa: E402
    _HIERARCHICAL9_V1_SHA256,
    _validate_hierarchical9_v1_checkpoint,
    _validate_sensor_output,
    build_per_sensor_model,
    PerSensorVisibilityRuntime,
)


OLD8 = (
    REPO
    / "src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt"
)
H9 = (
    REPO
    / "src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt"
)
REFERENCE_URDF = REPO / "src/arm_description/urdf/Arm.urdf"
SELF_FILTER_URDF = (
    REPO / "src/arm_description/urdf/Arm_with_self_filter_collision.urdf"
)

TARGET = np.asarray(
    [0.10000000149011612, 0.05000000074505806, 0.15000000596046448],
    dtype=np.float64,
)
MEASURED_SEED = np.asarray(
    [
        -0.23993935822373746,
        0.7577102185084499,
        -0.301829051647589,
        -1.626700836259844,
        0.19068393720573518,
        -0.18052088294007795,
        -0.3808012222262862,
    ],
    dtype=np.float64,
)
KNOWN_BLOCKED_QVIS = np.asarray(
    [
        -0.26947852969169617,
        0.750813901424408,
        -0.2667731046676636,
        -1.8708069324493408,
        0.1902204304933548,
        -0.1728929728269577,
        -0.3792421519756317,
    ],
    dtype=np.float64,
)
Q_MIN = np.asarray(
    [-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20], dtype=np.float64
)
Q_MAX = np.asarray(
    [3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20], dtype=np.float64
)


class Results:
    def __init__(self) -> None:
        self.rows = []

    def run(self, name, fn):
        tic = time.perf_counter()
        try:
            detail = fn()
        except Exception as exc:  # continue so every independent check reports
            row = {
                "name": name,
                "status": "FAIL",
                "detail": "{}: {}".format(type(exc).__name__, exc),
                "elapsed_ms": 1000.0 * (time.perf_counter() - tic),
            }
        else:
            row = {
                "name": name,
                "status": "PASS",
                "detail": detail,
                "elapsed_ms": 1000.0 * (time.perf_counter() - tic),
            }
        self.rows.append(row)
        print("[{status}] {name}: {detail}".format(**row))
        return row

    def not_run(self, name, detail):
        row = {
            "name": name,
            "status": "NOT_RUN",
            "detail": detail,
            "elapsed_ms": 0.0,
        }
        self.rows.append(row)
        print("[NOT_RUN] {}: {}".format(name, detail))


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def expect_error(fn, error_type, contains):
    try:
        fn()
    except error_type as exc:
        require(contains in str(exc), "unexpected error text: {}".format(exc))
        return
    raise AssertionError("expected {} containing {!r}".format(error_type, contains))


def make_runtime(path, device, max_attempts=8):
    return PerSensorVisibilityRuntime(
        checkpoint_path=str(path),
        reference_urdf_path=str(REFERENCE_URDF),
        self_filter_urdf_path=str(SELF_FILTER_URDF),
        device=device,
        q_min=Q_MIN,
        q_max=Q_MAX,
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
        max_branch_attempts=max_attempts,
        min_conservative_g=0.0,
        require_primitive_los=True,
    )


def sample_tensors(device):
    points = torch.tensor(
        [
            TARGET.tolist(),
            [0.115, 0.041, 0.163],
            [0.083, 0.064, 0.142],
        ],
        device=device,
        dtype=torch.float32,
    )
    q = torch.tensor(
        MEASURED_SEED.reshape(1, 7), device=device, dtype=torch.float32
    )
    return points, q


def reference_value_grad(full_model, points, q, sensor_id):
    q_var = q.detach().clone().reshape(1, 7).requires_grad_(True)
    q_batch = q_var.expand(points.shape[0], -1)
    full = full_model(torch.cat([points, q_batch], dim=-1))
    values = full[:, 1 + int(sensor_id)]
    value = torch.min(values)
    grad = torch.autograd.grad(value, q_var)[0]
    return float(value.detach().item()), grad.detach(), values.detach()


def percentile_summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(values.size),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def timed(device, fn):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tic = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - tic)


def run_device(device, repeats, results):
    state = {}

    def load_models():
        old_model, old_ckpt = build_per_sensor_model(str(OLD8), device)
        h9_model, h9_ckpt = build_per_sensor_model(str(H9), device)
        training_reference = TrainingHierarchicalVisibilityCDF().to(
            device=device, dtype=torch.float32
        )
        training_reference.load_state_dict(h9_ckpt["model_state"], strict=True)
        training_reference.eval().requires_grad_(False)
        points, q = sample_tensors(device)
        inputs = torch.cat([points, q.expand(points.shape[0], -1)], dim=-1)
        old_output = old_model(inputs)
        h9_output = h9_model(inputs)
        require(tuple(old_output.shape) == (3, 8), "legacy output is not [3,8]")
        require(tuple(h9_output.shape) == (3, 8), "H9 view is not [3,8]")
        require(
            old_ckpt["runtime_adapter"]["model_type"] == "legacy_yiming_8head",
            "legacy dispatch mismatch",
        )
        require(
            h9_ckpt["runtime_adapter"]["model_type"]
            == "hierarchical9_v1_sensor_view",
            "H9 dispatch mismatch",
        )
        require(
            h9_ckpt["runtime_adapter"]["checkpoint_sha256"]
            == _HIERARCHICAL9_V1_SHA256,
            "H9 runtime SHA record mismatch",
        )
        require(
            all(not parameter.requires_grad for parameter in old_model.parameters()),
            "legacy parameters not frozen",
        )
        require(
            all(not parameter.requires_grad for parameter in h9_model.parameters()),
            "H9 parameters not frozen",
        )
        state.update(
            old_model=old_model,
            old_ckpt=old_ckpt,
            h9_model=h9_model,
            h9_ckpt=h9_ckpt,
            training_reference=training_reference,
        )
        return {"old_shape": [3, 8], "h9_shape": [3, 8]}

    results.run("{}: legacy/H9 load and old API".format(device), load_models)

    def output_mapping():
        points, q = sample_tensors(device)
        inputs = torch.cat([points, q.expand(points.shape[0], -1)], dim=-1)
        view = state["h9_model"](inputs)
        full = state["h9_model"].full_model(inputs)
        reference_full = state["training_reference"](inputs)
        require(tuple(full.shape) == (3, 9), "full H9 output is not [3,9]")
        torch.testing.assert_close(full, reference_full, rtol=0.0, atol=0.0)
        torch.testing.assert_close(view, full[:, 1:9], rtol=0.0, atol=0.0)
        torch.testing.assert_close(view[:, 0], full[:, 1], rtol=0.0, atol=0.0)
        torch.testing.assert_close(view[:, 7], full[:, 8], rtol=0.0, atol=0.0)
        return {"max_abs": 0.0, "S0_column": 1, "S7_column": 8}

    results.run("{}: H9 output[1:9] mapping".format(device), output_mapping)

    def gradients_and_aggregation():
        runtime = make_runtime(H9, device)
        state["h9_runtime"] = runtime
        points, q = sample_tensors(device)
        max_value_error = 0.0
        max_raw_grad_error = 0.0
        max_masked_grad_error = 0.0
        for sensor_id in range(8):
            for selected_points in (points[:1], points):
                value, grad = runtime.branch_value_and_grad(
                    selected_points, q, sensor_id
                )
                ref_value, raw_ref_grad, per_point = reference_value_grad(
                    state["training_reference"], selected_points, q, sensor_id
                )
                view_value, raw_view_grad, _ = reference_value_grad(
                    runtime.model.full_model, selected_points, q, sensor_id
                )
                require(tuple(grad.shape) == (1, 7), "gradient is not original 7D q")
                require(
                    tuple(raw_view_grad.shape) == (1, 7),
                    "unmasked view gradient is not original 7D q",
                )
                require(
                    abs(value - float(torch.min(per_point).item())) <= 1e-7,
                    "point aggregation is not min",
                )
                max_value_error = max(max_value_error, abs(value - ref_value))
                max_value_error = max(max_value_error, abs(value - view_value))
                max_raw_grad_error = max(
                    max_raw_grad_error,
                    float(
                        torch.max(torch.abs(raw_view_grad - raw_ref_grad)).item()
                    ),
                )
                expected_runtime_grad = raw_ref_grad * runtime.sensor_masks[
                    sensor_id : sensor_id + 1
                ]
                max_masked_grad_error = max(
                    max_masked_grad_error,
                    float(
                        torch.max(torch.abs(grad - expected_runtime_grad)).item()
                    ),
                )
        require(max_value_error <= 1e-7, "reference value mismatch")
        require(max_raw_grad_error <= 1e-6, "raw reference gradient mismatch")
        require(
            max_masked_grad_error <= 1e-6,
            "runtime kinematic-mask gradient mismatch",
        )
        return {
            "sensors": 8,
            "single_and_multi_point": True,
            "aggregation": "per-sensor pointwise_min",
            "max_value_error": max_value_error,
            "max_raw_7d_grad_error": max_raw_grad_error,
            "max_runtime_masked_grad_error": max_masked_grad_error,
        }

    results.run(
        "{}: eight raw-7D q gradients and min aggregation".format(device),
        gradients_and_aggregation,
    )

    def finite_difference_and_recompute():
        runtime = state["h9_runtime"]
        points, q = sample_tensors(device)
        value0, grad0 = runtime.branch_value_and_grad(points[:1], q, 7)
        direction = torch.tensor(
            [[0.31, -0.23, 0.17, 0.41, -0.29, 0.19, 0.37]],
            device=device,
            dtype=torch.float32,
        )
        direction = direction / torch.linalg.vector_norm(direction)
        epsilon = 1e-3
        plus = runtime.branch_score(points[:1], q + epsilon * direction, 7)
        minus = runtime.branch_score(points[:1], q - epsilon * direction, 7)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        autodiff = float(torch.sum(grad0 * direction).item())
        tolerance = max(5e-3, 0.03 * abs(autodiff))
        require(
            abs(finite_difference - autodiff) <= tolerance,
            "finite difference mismatch: fd={} ad={} tol={}".format(
                finite_difference, autodiff, tolerance
            ),
        )
        q_changed = q + 0.02 * direction
        value1, grad1 = runtime.branch_value_and_grad(points[:1], q_changed, 7)
        require(
            abs(value1 - value0) > 1e-6
            or not torch.allclose(grad1, grad0, rtol=1e-5, atol=1e-6),
            "q update appears to reuse stale value/gradient features",
        )
        return {
            "finite_difference": finite_difference,
            "autodiff": autodiff,
            "abs_error": abs(finite_difference - autodiff),
            "q_changed_recomputed": True,
            "parameters_frozen": True,
        }

    results.run(
        "{}: finite difference, frozen params, q recompute".format(device),
        finite_difference_and_recompute,
    )

    def invalid_and_metadata_guards():
        runtime = state["h9_runtime"]
        points, q = sample_tensors(device)
        expect_error(
            lambda: build_per_sensor_model(str(REPO / "missing-final.pt"), device),
            FileNotFoundError,
            "checkpoint not found",
        )
        expect_error(
            lambda: runtime.rank_sensors(points[:0], q),
            ValueError,
            "at least one",
        )
        q_nan = q.clone()
        q_nan[0, 0] = math.nan
        expect_error(
            lambda: runtime.branch_value_and_grad(points[:1], q_nan, 0),
            ValueError,
            "q must be finite",
        )
        expect_error(
            lambda: runtime.branch_score(points[:1], q, 8),
            ValueError,
            "sensor_id",
        )
        expect_error(
            lambda: _validate_sensor_output(
                torch.full((1, 8), math.nan, device=device), 1
            ),
            RuntimeError,
            "non-finite",
        )
        expect_error(
            lambda: _validate_hierarchical9_v1_checkpoint(
                state["h9_ckpt"], "0" * 64
            ),
            RuntimeError,
            "SHA256 mismatch",
        )
        mutated = copy.deepcopy(state["h9_ckpt"])
        mutated["step"] = 49999
        expect_error(
            lambda: _validate_hierarchical9_v1_checkpoint(
                mutated, _HIERARCHICAL9_V1_SHA256
            ),
            RuntimeError,
            "step",
        )
        return {
            "missing_model": "rejected",
            "empty_nan_bad_sensor_bad_output": "rejected",
            "sha_step_guards": "rejected",
        }

    results.run(
        "{}: invalid input/output and checkpoint guards".format(device),
        invalid_and_metadata_guards,
    )

    def bounded_failure_and_limits():
        runtime = make_runtime(H9, device, max_attempts=3)
        points, q = sample_tensors(device)
        seeds = []

        def fake_rank(_points, _q):
            return np.arange(8, dtype=np.float64), np.arange(7, -1, -1)

        def fake_optimize(_points, q_start, sensor_id, **_kwargs):
            seed = q_start.detach().cpu().numpy().reshape(7).astype(float)
            seeds.append(seed.copy())
            return {
                "sensor_id": int(sensor_id),
                "sensor_frame": DEFAULT_SENSOR_FRAMES[int(sensor_id)],
                "q_candidate": seed.tolist(),
            }

        def fake_geometry(_points, _q, _sensor_id, minimum_conservative_g=None):
            return {
                "accepted": False,
                "reject_reason": "synthetic_all_branch_failure",
            }

        runtime.rank_sensors = fake_rank
        runtime._optimize_branch = fake_optimize
        runtime._candidate_geometry = fake_geometry
        deliberately_different_q_zero = MEASURED_SEED + np.asarray(
            [0.4, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1], dtype=np.float64
        )
        result = runtime.generate(
            TARGET.reshape(1, 3),
            deliberately_different_q_zero,
            MEASURED_SEED,
        )
        require(not result["accepted"], "all-failed branches reported accepted")
        require(result["selected_q_vis"] is None, "all-failed result supplied q_vis")
        require(len(result["attempts"]) == 3, "max_branch_attempts not preserved")
        require(all(math.isfinite(a["branch_latency_ms"]) and a["branch_latency_ms"] >= 0
                    for a in result["attempts"]), "rejected branch latency missing")
        require(len(seeds) == 3, "wrong branch invocation count")
        for seed in seeds:
            np.testing.assert_allclose(seed, MEASURED_SEED, rtol=0.0, atol=1e-6)
            require(
                not np.allclose(seed, deliberately_different_q_zero),
                "branch incorrectly started from scalar q_zero",
            )

        outside = torch.tensor(
            [[9.0, -9.0, 9.0, -9.0, 9.0, -9.0, 9.0]],
            device=device,
            dtype=torch.float32,
        )
        clamped, changed = runtime._clamp(outside)
        require(changed, "joint limit clamp did not report a change")
        require(bool(torch.all(clamped <= runtime.q_max + 1e-7)), "q above max")
        require(bool(torch.all(clamped >= runtime.q_min - 1e-7)), "q below min")
        return {
            "attempts": 3,
            "independent_same_seed": True,
            "accepted": False,
            "caller_fallback_required": True,
            "joint_limits_clamped": True,
        }

    results.run(
        "{}: bounded all-branch failure and joint limits".format(device),
        bounded_failure_and_limits,
    )

    def replacement_mount_group_priority():
        runtime = make_runtime(H9, device, max_attempts=8)
        ranking = np.arange(8, dtype=np.int64)
        cosines = {sid: -0.99 + 0.10 * sid for sid in range(8)}

        def fake_rank(_points, _q):
            return np.arange(8, dtype=np.float64), ranking

        def fake_grad(_points, _q, sensor_id):
            cosine = cosines[int(sensor_id)]
            vector = [cosine, math.sqrt(1.0-cosine*cosine)] + [0.0] * 5
            return 0.0, torch.tensor(
                [vector], device=device, dtype=torch.float32)

        def fake_optimize(_points, q_start, sensor_id, **_kwargs):
            q_candidate = q_start.detach().cpu().numpy().reshape(7).copy()
            q_candidate[0] += 0.01 * (int(sensor_id) + 1)
            return {
                "sensor_id": int(sensor_id),
                "sensor_frame": DEFAULT_SENSOR_FRAMES[int(sensor_id)],
                "q_candidate": q_candidate.tolist(),
            }

        runtime.rank_sensors = fake_rank
        runtime.branch_value_and_grad = fake_grad
        runtime._optimize_branch = fake_optimize
        runtime._candidate_geometry = lambda *_args, **_kwargs: {
            "accepted": True, "reject_reason": "synthetic_accepted"}
        previous = np.zeros(7, dtype=np.float64)
        previous[0] = 1.0
        first_pass = runtime.generate(
            TARGET.reshape(1, 3), np.zeros(7), np.zeros(7),
            replacement={"excluded": [0], "tried_mount_groups": [0],
                         "previous_q": previous.tolist()})
        require(first_pass["selected_sensor_id"] == 2,
                "opposite S1 was selected before an untried mount group")
        require(first_pass["replacement_group_priority"]["phase"] ==
                "untried_mount_group", "wrong first-pass group phase")
        ee_independent = runtime.generate(
            TARGET.reshape(1, 3), np.zeros(7), np.zeros(7),
            replacement={"excluded": [0, 2, 4, 6],
                         "tried_mount_groups": [0, 1, 2, 3],
                         "previous_q": previous.tolist()})
        require(ee_independent["selected_sensor_id"] == 7,
                "S7 was incorrectly removed with independent EE sensor S6")
        require(ee_independent["replacement_group_priority"]["phase"] ==
                "untried_mount_group", "S7 was not treated as an untried group")
        paired = runtime.generate(
            TARGET.reshape(1, 3), np.zeros(7), np.zeros(7),
            replacement={"excluded": [0, 2, 4, 6, 7],
                         "tried_mount_groups": [0, 1, 2, 3, 4],
                         "previous_q": previous.tolist()})
        require(paired["selected_sensor_id"] == 1,
                "paired fallback did not restore the best opposite sensor")
        require(paired["replacement_group_priority"]["phase"] ==
                "paired_sensor_fallback", "wrong paired fallback phase")
        require(len(first_pass["attempts"]) == len(ee_independent["attempts"]) ==
                len(paired["attempts"]) == 1,
                "one reservation optimized more than one sensor")
        return {
            "first_pass_selected": 2,
            "ee_independent_selected": 7,
            "paired_fallback_selected": 1,
            "one_branch_per_reservation": True,
        }

    results.run(
        "{}: replacement mount-group priority".format(device),
        replacement_mount_group_priority,
    )

    def replacement_seed_score_gap():
        runtime = make_runtime(H9, device, max_attempts=8)
        scores = np.array([
            -1.0459, -0.8386, -1.4742, -1.0605,
            -0.6865, -0.6510, -1.9296, -1.8909,
        ])
        order = np.array([5, 1, 3, 4, 2, 0, 7, 6])
        runtime.rank_sensors = lambda _points, _q: (scores, order)
        runtime.branch_value_and_grad = lambda _points, _q, _sid: (
            0.0, torch.tensor([[1.0] + [0.0] * 6], device=device))
        runtime._optimize_branch = lambda _points, q_start, sid, **_kwargs: {
            "sensor_id": int(sid),
            "sensor_frame": DEFAULT_SENSOR_FRAMES[int(sid)],
            "q_candidate": q_start.detach().cpu().numpy().reshape(7).tolist(),
        }
        runtime._candidate_geometry = lambda *_args, **_kwargs: {
            "accepted": True, "reject_reason": "synthetic_accepted"}
        candidate = runtime.generate(
            TARGET.reshape(1, 3), np.zeros(7), np.zeros(7),
            replacement={"excluded": [5], "tried_mount_groups": [2],
                         "previous_q": [1.0] + [0.0] * 6,
                         "preferred_sensor_ids": [7, 6],
                         "ee_sensor_priority_first": True,
                         "score_gap_override": 0.75})
        require(candidate["selected_sensor_id"] == 1,
                "weak high-witness head displaced stronger untried mount group")
        require(candidate["replacement_score_override"]["original_sensor_id"] == 7,
                "original selection was not traced")
        require(len(candidate["attempts"]) == 1,
                "score override used more than one optimized branch")
        return candidate["replacement_score_override"]

    results.run(
        "{}: replacement measured-seed score gap".format(device),
        replacement_seed_score_gap,
    )

    def scalar_fallback_preserved():
        # Import the unchanged ROS caller without starting a node or planner.
        # NumPy/Torch are already imported from viscdf before system ROS paths
        # are appended, avoiding cross-Python binary package shadowing.
        extra_paths = [
            REPO / "devel/lib/python3/dist-packages",
            Path("/opt/ros/noetic/lib/python3/dist-packages"),
            Path("/usr/lib/python3/dist-packages"),
        ]
        for path in extra_paths:
            if str(path) not in sys.path:
                sys.path.append(str(path))
        source_path = str(REPO / "src")
        ros_paths = os.environ.get("ROS_PACKAGE_PATH", "").split(":")
        if source_path not in ros_paths:
            os.environ["ROS_PACKAGE_PATH"] = ":".join(
                [source_path, "/opt/ros/noetic/share"]
            )
        from vbc_deadline_waypoint_rolling_impl import (  # noqa: E402
            RollingVbcDeadlineWaypointNode,
        )

        class AllFailedRuntime:
            def generate(self, *_args, **_kwargs):
                return {
                    "accepted": False,
                    "ranking": list(range(8)),
                    "rejected_sensor_ids": list(range(8)),
                }

        node = object.__new__(RollingVbcDeadlineWaypointNode)
        node.per_sensor_hybrid_enabled = True
        node._per_sensor_runtime = AllFailedRuntime()
        scalar_q_vis = [float(v) for v in MEASURED_SEED]
        scalar_result = {
            "q_zero": [0.0] * 7,
            "q_deadline_nominal": [0.0] * 7,
            "q_vis": list(scalar_q_vis),
        }
        returned = node._maybe_apply_per_sensor_hybrid(
            TARGET.reshape(1, 3), scalar_result
        )
        require(returned["q_vis"] == scalar_q_vis, "scalar fallback was changed")
        require(
            returned["per_sensor_hybrid_used"] is False,
            "all-failed branch result marked hybrid used",
        )
        class ErrorRuntime:
            def generate(self, *_args, **_kwargs):
                raise RuntimeError("injected_branch_error")
        node._per_sensor_runtime = ErrorRuntime()
        errored = node._maybe_apply_per_sensor_hybrid(TARGET.reshape(1,3), copy.deepcopy(scalar_result))
        require(errored["q_vis"] == scalar_q_vis, "exception changed scalar fallback")
        require(errored["per_sensor_hybrid"]["compute_ms"] >= 0., "exception latency missing")
        require("injected_branch_error" in errored["per_sensor_hybrid"]["runtime_error"], "wrong failure stage")
        node._per_sensor_runtime = None
        # This CPU fixture does not initialize ROS time. Stub only throttled
        # logging, not the caller logic or fallback under test.
        from unittest.mock import patch
        with patch("vbc_deadline_waypoint_rolling_impl.rospy.logerr_throttle"):
            unavailable = node._maybe_apply_per_sensor_hybrid(TARGET.reshape(1,3), copy.deepcopy(scalar_result))
        require(unavailable["per_sensor_hybrid"]["runtime_error"] == "runtime_unavailable", "missing-runtime trace absent")
        return {
            "all_branches_failed": True,
            "scalar_q_vis_preserved": True,
            "hybrid_used": False,
            "ROS_node_started": False,
            "exception_and_unavailable_latency_recorded": True,
        }

    results.run(
        "{}: all-failed caller preserves scalar fallback".format(device),
        scalar_fallback_preserved,
    )

    def historical_geometry():
        runtime = state["h9_runtime"]
        geometry = runtime._candidate_geometry(
            TARGET.reshape(1, 3), KNOWN_BLOCKED_QVIS, 4
        )
        require(
            abs(float(geometry["min_conservative_g"]) - 0.018168168720235307)
            <= 1e-9,
            "historical conservative g changed",
        )
        require(
            geometry["any_primitive_self_occluded"],
            "historical S4 occlusion did not reproduce",
        )
        hit = geometry["per_point"][0]["primitive_hit"]
        require(hit is not None and hit["link"] == "link3", "wrong blocked link")
        return {
            "conservative_g": float(geometry["min_conservative_g"]),
            "self_occlusion": "FAIL",
            "hit_link": hit["link"],
        }

    results.run("{}: unchanged Case026 geometry".format(device), historical_geometry)

    def benchmarks():
        output = {}
        points, q = sample_tensors(device)
        for name, path in (("old8", OLD8), ("hierarchical9", H9)):
            runtime = make_runtime(path, device)
            warmup_ms = runtime.warmup()
            ranking = [
                timed(device, lambda: runtime.rank_sensors(points, q))
                for _ in range(repeats)
            ]
            value_grad = [
                timed(
                    device,
                    lambda: runtime.branch_value_and_grad(points, q, 7),
                )
                for _ in range(repeats)
            ]
            branch = [
                timed(device, lambda: runtime._optimize_branch(points, q, 4))
                for _ in range(max(3, repeats // 2))
            ]
            output[name] = {
                "warmup_ms": float(warmup_ms),
                "ranking": percentile_summary(ranking),
                "value_grad": percentile_summary(value_grad),
                "multi_point_branch_solver": percentile_summary(branch),
                "points": int(points.shape[0]),
            }
        return output

    results.run(
        "{}: warmup p50/p95 runtime benchmark".format(device), benchmarks
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--benchmark-repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        default=str(
            REPO
            / "outputs/phase_e_mainline_a_a1/runtime_hierarchical9_tests.json"
        ),
    )
    args = parser.parse_args()
    if args.benchmark_repeats < 3:
        raise ValueError("--benchmark-repeats must be >= 3")

    for required in (OLD8, H9, REFERENCE_URDF, SELF_FILTER_URDF):
        if not required.is_file():
            raise FileNotFoundError(required)

    results = Results()
    devices = []
    if args.device in ("auto", "cpu"):
        devices.append(torch.device("cpu"))
    if args.device in ("auto", "cuda"):
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        else:
            results.not_run(
                "cuda: full runtime suite",
                "torch.cuda.is_available() is false",
            )
            if args.device == "cuda":
                devices = []

    for device in devices:
        run_device(device, args.benchmark_repeats, results)

    report = {
        "diagnostic": "per_sensor_runtime_hierarchical9_a1",
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "benchmark_repeats": int(args.benchmark_repeats),
        "results": results.rows,
        "summary": {
            status: sum(row["status"] == status for row in results.rows)
            for status in ("PASS", "FAIL", "NOT_RUN")
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("[OUTPUT]", output)
    print("[SUMMARY]", report["summary"])
    if report["summary"]["FAIL"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
