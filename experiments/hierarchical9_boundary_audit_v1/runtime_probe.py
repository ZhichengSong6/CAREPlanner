"""Use the existing branch optimizer without constructing ROS or LOS machinery.

No solver reimplementation or parameter retuning: inherits _optimize_branch,
_projection_step, _refine_branch_root, _ascent_step, _clamp from repository runtime.
This offline adapter only supplies models/limits/masks and counts evaluations.
"""
from __future__ import annotations
import time
import torch


def make_probe(model, masks, lo, hi):
    from per_sensor_visibility_runtime import PerSensorVisibilityRuntime

    class Probe(PerSensorVisibilityRuntime):
        def __init__(self):
            # Deliberately do NOT call runtime __init__: no checkpoint reload,
            # raycast primitives, ROS publishers, or full framework execution.
            self.model, self.device = model, lo.device
            self.q_min, self.q_max = lo, hi
            self.sensor_masks = masks
            self.projection_iters = 10
            self.projection_damping = .5
            self.projection_epsilon_f = .03
            self.projection_max_step_norm = .25
            self.root_refine_iters = 12
            self.root_tolerance_f = .002
            self.branch_ascent_steps = 1
            self.branch_step_size = .05
            self.branch_max_step_norm = .25
            self.branch_fallback_ascent_steps = 8
            self.value_calls = self.gradient_calls = 0

        def branch_score(self, points, q, sensor_id):
            self.value_calls += 1
            return super().branch_score(points, q, sensor_id)

        def branch_value_and_grad(self, points, q, sensor_id):
            self.gradient_calls += 1
            return super().branch_value_and_grad(points, q, sensor_id)

    return Probe()


def run_probe(probe, oracle, x, q, s):
    if q.device.type == "cuda": torch.cuda.synchronize(q.device)
    probe.value_calls = probe.gradient_calls = 0
    start = time.perf_counter()
    out = probe._optimize_branch(x.reshape(1, 3), q.reshape(1, 7).clone(), s)
    if q.device.type == "cuda": torch.cuda.synchronize(q.device)
    out["solver_ms"] = 1000*(time.perf_counter()-start)
    out["value_calls"] = probe.value_calls
    out["gradient_calls"] = probe.gradient_calls
    # There is always a q_zero key, even on root-not-found best-effort fallback.
    # Do NOT mistake existence of that tensor or initial_positive for root success.
    out["predicted_bracket_found"] = out["root_source"] == "branch_sign_crossing_bisection"
    out["predicted_root_within_002"] = abs(out["f_zero"]) <= .002
    qz = torch.tensor(out["q_zero"], device=q.device, dtype=q.dtype)
    qc = torch.tensor(out["q_candidate"], device=q.device, dtype=q.dtype)
    out["initial_g_m"] = oracle.value(x, q, s)
    out["zero_g_m"] = oracle.value(x, qz, s)
    out["candidate_g_m"] = oracle.value(x, qc, s)
    out["candidate_in_limits"] = bool(((qc >= probe.q_min) & (qc <= probe.q_max)).all())
    out["fov_pass"] = out["candidate_in_limits"] and out["candidate_g_m"] >= 0
    out["fov_margin_001_pass"] = out["candidate_in_limits"] and out["candidate_g_m"] >= .01
    out["failure_stage"] = ("FOV_PASS_NOT_EXECUTION_CERTIFIED" if out["fov_pass"] else
        "ROOT_NOT_FOUND_AND_FOV_FAIL" if out["solution_mode"] == "branch_best_effort_ascent" else
        "LEARNED_CANDIDATE_BUT_FOV_FAIL")
    out["joint_clamp_events"] = sum(bool(h.get("joint_limit_clamped")) for key in
        ("projection_history", "ascent_history") for h in out[key])
    out["self_occlusion"] = out["trajectory_certification"] = out["actual_seen"] = "NOT_RUN"
    return out


def analytic_ascent(oracle, x, q, s, mask, lo, hi, steps=18):
    """Separate bounded control, NOT equal-objective/equal-time runtime comparison.

    18 normalized analytic-g gradient steps of 0.05 rad. Each iterate is checked;
    keep the best. Failure is not infeasibility. A local geometric witness exists
    only for samples constructed from a verified boundary and positive-side probe.
    """
    if q.device.type == "cuda": torch.cuda.synchronize(q.device)
    start = time.perf_counter()
    q = q.clone()
    best, best_g = q.clone(), oracle.value(x, q, s)
    hist = [{"iter": 0, "g_m": best_g, "q": q.tolist()}]
    calls = 0
    for k in range(steps):
        g, grad = oracle.value_grad(x, q, s); calls += 1
        grad = grad * mask
        if not torch.isfinite(grad).all() or grad.norm() < 1e-8:
            break
        raw = q + .05 * grad / grad.norm()
        q = torch.maximum(torch.minimum(raw, hi), lo).detach()
        g = oracle.value(x, q, s)
        if g > best_g: best, best_g = q.clone(), g
        hist.append({"iter": k+1, "g_m": g, "q": q.tolist(),
                     "joint_limit_clamped": not torch.equal(q, raw)})
    if q.device.type == "cuda": torch.cuda.synchronize(q.device)
    return {"best_g_m": best_g, "q_candidate": best.tolist(), "fov_pass": best_g >= 0,
            "gradient_calls": calls, "history": hist, "solver_ms": 1000*(time.perf_counter()-start),
            "meaning": "analytic FOV only; different objective/budget; failure is NOT infeasibility"}
