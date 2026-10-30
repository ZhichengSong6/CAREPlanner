#!/usr/bin/env python3
"""Offline exp81 high-point reachability and hard-witness diagnosis.

This script never starts ROS, publishes a trajectory, or changes a production
parameter.  R1/FOV/LOS endpoints and straight-line point-center clearances are
diagnostics, not execution certificates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import differential_evolution, linprog

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "src/care_visibility_cdf/scripts"
sys.path[:0] = [str(ROOT / "scripts"), str(VIS)]

from analyze_phase_e_observation_events import solver_point_key, string_rows
from diagnose_phase_e_case001_exact_witness import Geometry
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime
from validate_visibility_oracle import fk_transform

RID = "case_001_r1_owner_promotion_group_priority_20260922_exp81"
RUN = ROOT / "outputs/c5_5_vbc_gcdf_regime" / RID / "run"
LOG = ROOT / "logs/c5_5_vbc_gcdf_regime" / RID / "run/controlled.log"
TRACE = RUN / "projector_traces"
CHECKPOINT = ROOT / "src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt"
REFERENCE_URDF = ROOT / "src/arm_description/urdf/Arm.urdf"
SELF_FILTER_URDF = ROOT / "src/arm_description/urdf/Arm_with_self_filter_collision.urdf"
JOINTS = ["joint1", "joint2", "joint3", "joint4",
          "wrist_joint1", "wrist_joint2", "wrist_joint3"]
Q_MIN = np.array([-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20])
Q_MAX = np.array([3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20])
GROUPS = {"S0/S1": [0, 1], "S2/S3": [2, 3],
          "S4/S5": [4, 5], "S6/S7": [6, 7]}


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def numeric_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def vector(row, prefix, count=7):
    return np.array([float(row[f"{prefix}{j}"]) for j in range(count)])


def point_label(point):
    x, y, z = np.asarray(point)
    if x < 0:
        return "P5_O5_neg_x_z095"
    if abs(y) < 1e-5 and abs(z - .90) < 1e-5:
        return "P1a_O1_y0_z090"
    if y < -.01 and abs(z - .95) < 1e-5:
        return "P1b_O1_yneg_z095"
    if abs(y) < 1e-5 and abs(z - .95) < 1e-5:
        return "P1c_O1_y0_z095"
    if y < -.01 and abs(z - .90) < 1e-5:
        return "P1d_O1_yneg_z090"
    raise KeyError(point)


def runtime_from_trace(trace):
    config = trace["config"]
    return PerSensorVisibilityRuntime(
        checkpoint_path=str(CHECKPOINT),
        reference_urdf_path=str(REFERENCE_URDF),
        self_filter_urdf_path=str(SELF_FILTER_URDF),
        device=torch.device("cpu"), q_min=Q_MIN, q_max=Q_MAX,
        projection_iters=config["projection_iters"],
        projection_damping=config["projection_damping"],
        projection_epsilon_f=config["projection_epsilon_f"],
        projection_max_step_norm=config["projection_max_step_norm"],
        root_refine_iters=config["root_refine_iters"],
        root_tolerance_f=config["root_tolerance_f"],
        branch_ascent_steps=config["ascent_steps"],
        branch_step_size=config["ascent_step_size"],
        branch_max_step_norm=config["ascent_max_step_norm"],
        branch_fallback_ascent_steps=config["shared_fallback_ascent_steps"],
        max_branch_attempts=4, min_conservative_g=0.0,
        require_primitive_los=True)


def batch_scores(runtime, points, qs):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    qs = np.asarray(qs, dtype=np.float32).reshape(-1, 7)
    count, point_count = len(qs), len(points)
    x = torch.tensor(np.tile(points, (count, 1)), dtype=torch.float32)
    q = torch.tensor(np.repeat(qs, point_count, axis=0), dtype=torch.float32)
    with torch.no_grad():
        values = runtime.model(torch.cat([x, q], dim=-1))
    return values.reshape(count, point_count, 8).min(dim=1).values.numpy()


def q_distance(q0, q1, mask=None):
    delta = np.asarray(q1) - np.asarray(q0)
    if mask is not None:
        delta = delta[np.asarray(mask, dtype=bool)]
    return {"inf_rad": float(np.max(np.abs(delta))),
            "l2_rad": float(np.linalg.norm(delta)),
            "delta": delta.tolist()}


def optimize_branches(runtime, targets, q0):
    result = {}
    x = torch.tensor(np.asarray(targets, dtype=np.float32))
    seed = torch.tensor(np.asarray(q0, dtype=np.float32).reshape(1, 7))
    for sensor in range(8):
        branch = runtime._optimize_branch(x, seed, sensor)
        zero_geometry = runtime._candidate_geometry(targets, branch["q_zero"], sensor)
        final_geometry = runtime._candidate_geometry(targets, branch["q_candidate"], sensor)
        result[str(sensor)] = {
            "sensor": sensor,
            "mask": runtime.sensor_masks_np[sensor].tolist(),
            "initial_score": branch["initial_score"],
            "root_found": branch["root_found"],
            "root_score": branch["f_zero"],
            "final_score": branch["final_score"],
            "q_zero": branch["q_zero"],
            "q_candidate": branch["q_candidate"],
            "root_distance": q_distance(q0, branch["q_zero"], runtime.sensor_masks_np[sensor]),
            "final_distance": q_distance(q0, branch["q_candidate"], runtime.sensor_masks_np[sensor]),
            "root_geometry": zero_geometry,
            "final_geometry": final_geometry,
            "endpoint_accepted": bool(branch["root_found"] and final_geometry["accepted"]),
        }
    return result


def radius_scan(runtime, targets, q0, samples, radii):
    sobol = torch.quasirandom.SobolEngine(7, scramble=True, seed=8101)
    directions = 2.0 * sobol.draw(samples).numpy().astype(np.float64) - 1.0
    directions[0] = 0.0
    per_sensor = {}
    for sensor in range(8):
        mask = runtime.sensor_masks_np[sensor].astype(np.float64)
        records = []
        first_accepted = None
        for radius in radii:
            qs = np.clip(q0[None, :] + float(radius) * directions * mask[None, :], Q_MIN, Q_MAX)
            scores = batch_scores(runtime, targets, qs)[:, sensor]
            order = np.argsort(scores)[::-1]
            accepted = None
            # Geometry is checked only for the strongest positive samples.
            for index in order[:16]:
                if scores[index] < 0.0:
                    break
                geometry = runtime._candidate_geometry(targets, qs[index], sensor)
                if geometry["accepted"]:
                    accepted = {"score": float(scores[index]), "q": qs[index].tolist(),
                                "geometry": geometry, "sample_index": int(index)}
                    break
            record = {"radius_inf_rad": float(radius),
                      "best_score": float(scores[order[0]]),
                      "best_q": qs[order[0]].tolist(),
                      "accepted_positive": accepted}
            records.append(record)
            if accepted is not None and first_accepted is None:
                first_accepted = record
        per_sensor[str(sensor)] = {"sensor": sensor, "records": records,
                                   "first_sampled_accepted": first_accepted}
    by_group = {}
    for name, sensors in GROUPS.items():
        candidates = [per_sensor[str(sensor)]["first_sampled_accepted"] for sensor in sensors]
        candidates = [item for item in candidates if item is not None]
        by_group[name] = min(candidates, key=lambda item: item["radius_inf_rad"]) if candidates else None
    return {"samples_per_radius": samples, "radii_inf_rad": radii,
            "per_sensor": per_sensor, "first_sampled_accepted_by_group": by_group}


def path_diagnostic(runtime, geometry, targets, all_points, q0, branch, sensor, samples=129):
    q1 = np.asarray(branch["q_candidate"])
    alphas = np.linspace(0.0, 1.0, samples)
    qs = q0[None, :] + alphas[:, None] * (q1 - q0)[None, :]
    scores = batch_scores(runtime, targets, qs)[:, sensor]
    wrist_indices = [i for i, primitive in enumerate(geometry.primitives)
                     if primitive["link"].startswith("wrist_link")]
    wrist_chain = geometry.chains["wrist_link3"]
    start_wrist = fk_transform(wrist_chain, dict(zip(JOINTS, q0)))[:3, 3]
    rows = []
    for alpha, q, score in zip(alphas, qs, scores):
        sensor_geometry = runtime._candidate_geometry(targets, q, sensor)
        distances = geometry.distances(q, all_points)
        point_i, primitive_i = np.unravel_index(np.argmin(distances), distances.shape)
        wrist_distances = distances[:, wrist_indices]
        wrist_point_i, wrist_local_i = np.unravel_index(
            np.argmin(wrist_distances), wrist_distances.shape)
        wrist_primitive_i = wrist_indices[wrist_local_i]
        wrist_origin = fk_transform(wrist_chain, dict(zip(JOINTS, q)))[:3, 3]
        rows.append({
            "alpha": float(alpha), "q": q.tolist(), "sensor_score": float(score),
            "sensor_geometry_accepted": sensor_geometry["accepted"],
            "sensor_conservative_g": sensor_geometry["min_conservative_g"],
            "sensor_self_occluded": sensor_geometry["any_primitive_self_occluded"],
            "all_point_body_clearance_m": float(distances[point_i, primitive_i]),
            "closest_point": np.asarray(all_points)[point_i].tolist(),
            "closest_link": geometry.primitives[primitive_i]["link"],
            "wrist_point_clearance_m": float(wrist_distances[wrist_point_i, wrist_local_i]),
            "wrist_closest_point": np.asarray(all_points)[wrist_point_i].tolist(),
            "wrist_closest_link": geometry.primitives[wrist_primitive_i]["link"],
            "wrist_link3_origin_displacement_m": float(np.linalg.norm(wrist_origin - start_wrist)),
        })
    visible = next((row for row in rows if row["sensor_score"] >= 0.0
                    and row["sensor_geometry_accepted"]), None)
    contact = next((row for row in rows if row["all_point_body_clearance_m"] <= 0.0), None)
    wrist_contact = next((row for row in rows if row["wrist_point_clearance_m"] <= 0.0), None)
    minimum = min(rows, key=lambda row: row["all_point_body_clearance_m"])
    return {
        "sensor": sensor, "q_distance": q_distance(q0, q1),
        "first_visible_alpha": None if visible is None else visible["alpha"],
        "first_contact_alpha": None if contact is None else contact["alpha"],
        "first_wrist_contact_alpha": None if wrist_contact is None else wrist_contact["alpha"],
        "visible_before_contact": bool(visible and (not contact or visible["alpha"] < contact["alpha"])),
        "minimum_body_clearance_m": minimum["all_point_body_clearance_m"],
        "minimum_body_clearance_alpha": minimum["alpha"],
        "minimum_body_clearance_link": minimum["closest_link"],
        "endpoint": rows[-1], "rows": rows,
    }


def wrist_clearance_optimization(runtime, geometry, targets, all_points, q0, branch, sensor):
    q_base = np.asarray(branch["q_candidate"], dtype=float)
    wrist_indices = [i for i, primitive in enumerate(geometry.primitives)
                     if primitive["link"].startswith("wrist_link")]

    def metrics(wrist):
        q = q_base.copy(); q[4:] = wrist
        distances = geometry.distances(q, all_points)
        wrist_clearance = float(np.min(distances[:, wrist_indices]))
        full_clearance = float(np.min(distances))
        sensor_geometry = runtime._candidate_geometry(targets, q, sensor)
        return q, wrist_clearance, full_clearance, sensor_geometry

    baseline = metrics(q_base[4:])

    def objective(wrist):
        _, clearance, _, sensor_geometry = metrics(wrist)
        penalty = 0.0 if sensor_geometry["accepted"] else 2.0
        return -clearance + penalty

    solved = differential_evolution(
        objective, list(zip(Q_MIN[4:], Q_MAX[4:])), seed=8101 + sensor,
        maxiter=14, popsize=7, polish=True, workers=1, updating="immediate",
        atol=1e-5, tol=1e-4)
    optimized = metrics(solved.x)
    # S0-S5 analytic sensor pose is upstream of the wrist.  Any difference in
    # FOV/LOS can only be caused by wrist self-occlusion, not sensor aim.
    return {
        "sensor": sensor,
        "upstream_joint_count": int(np.sum(runtime.sensor_masks_np[sensor])),
        "baseline": {"q": baseline[0], "wrist_clearance_m": baseline[1],
                     "full_body_clearance_m": baseline[2], "geometry": baseline[3]},
        "optimized": {"q": optimized[0], "wrist_clearance_m": optimized[1],
                      "full_body_clearance_m": optimized[2], "geometry": optimized[3]},
        "wrist_clearance_gain_m": optimized[1] - baseline[1],
        "optimizer_success": bool(solved.success), "optimizer_message": str(solved.message),
        "qualification": "endpoint only; point centers; no self-collision/full-map/path certificate",
    }


def reconstruct_margins(outcomes, base, step, cap):
    margins = {}
    events = []
    for row in outcomes:
        if row.get("result") != "unsafe" or row.get("safety_gate") != "vbc":
            continue
        evidence = json.loads(row["vbc_evidence"])
        updates = []
        unique = {}
        for violation in evidence.get("violations", []):
            unique[solver_point_key(violation["point"])] = violation["point"]
        for key, point in unique.items():
            previous = margins.get(key, base)
            margins[key] = min(cap, previous + step)
            updates.append({"point": point, "previous": previous, "margin": margins[key]})
        events.append({"ros_s": row["ros_s"], "seq": int(row["seq"]), "updates": updates})
    return margins, events


def diagnostic_slack(rows, q0, margins, velocity, dt):
    A, b, detail = [], [], []
    for row in rows:
        point = json.loads(row["point"])
        key = solver_point_key(point)
        gradient = np.array(json.loads(row["gradient"]), dtype=float)
        qlin = np.array(json.loads(row["q_linearization"]), dtype=float)
        distance = float(row["distance"])
        at_start = float(distance + gradient @ (q0 - qlin))
        margin = margins[key]
        A.append(np.r_[-dt * gradient, -1.0])
        b.append(at_start - margin)
        maximum_growth = float(dt * np.abs(gradient) @ velocity)
        detail.append({"label": point_label(point), "point": point, "margin": margin,
                       "linearized_start": at_start, "maximum_first_step_growth": maximum_growth,
                       "maximum_first_step_value": at_start + maximum_growth,
                       "single_row_gap": margin - at_start - maximum_growth})
    bounds = list(zip(-velocity, velocity)) + [(0.0, None)]
    objective = np.r_[np.zeros(7), 1.0]
    solution = linprog(objective, A_ub=np.asarray(A), b_ub=np.asarray(b),
                       bounds=bounds, method="highs")
    return {"success": bool(solution.success), "status": int(solution.status),
            "message": solution.message,
            "minimum_common_slack": float(solution.x[-1]) if solution.success else None,
            "u_rad_s": solution.x[:7].tolist() if solution.success else None,
            "rows": detail}


def qp_subset_diagnostic(final_points):
    summaries = string_rows(RUN / "local_planner_summary.csv", {"start_ros_s": 0., "end_ros_s": 100.})
    selectors = string_rows(RUN / "local_witness_selector_diagnostics.csv",
                            {"start_ros_s": 0., "end_ros_s": 100.})
    outcomes = string_rows(RUN / "verification_outcome.csv", {"start_ros_s": 0., "end_ros_s": 100.})
    query_by_stamp = {row["field.header.stamp"]: row for row in numeric_rows(RUN / "local_gcdf_query.csv")}
    log = LOG.read_text(errors="replace")
    prefix = "/local_sparse_scp_planner_node/"

    def parameter(name):
        return re.search(re.escape(prefix + name) + r": ([^\n]+)", log)[1].strip()

    velocity = np.array([float(parameter("mpc/joint_velocity_limits/" + joint)) for joint in JOINTS])
    dt = float(parameter("local_planner/horizon_duration")) / int(parameter("local_planner/num_intervals"))
    base = float(parameter("local_planner/cdf/visibility_obligation_cdf_margin"))
    step = float(parameter("local_planner/cdf/vbc_feedback_cdf_margin_step"))
    cap = float(parameter("local_planner/cdf/vbc_feedback_cdf_margin_max"))
    margins, feedback = reconstruct_margins(outcomes, base, step, cap)
    failed = [row for row in summaries if row.get("event") == "scp_failed"
              and int(row.get("plan_seq", -1)) in range(156, 161)]
    plans = []
    for summary in failed:
        stamp = summary["query_stamp_ns"]
        query = query_by_stamp[stamp]
        q0 = vector(query, "field.points0.positions")
        rows = [row for row in selectors if row.get("query_stamp_ns") == stamp
                and row.get("timestep") == "1"]
        assert len(rows) == len(final_points) == 5
        singles = {point_label(json.loads(row["point"])):
                   diagnostic_slack([row], q0, margins, velocity, dt) for row in rows}
        plans.append({"plan_seq": int(summary["plan_seq"]), "query_stamp_ns": int(stamp),
                      "solver_status": summary.get("status"), "q0": q0,
                      "single_rows": singles,
                      "all_five": diagnostic_slack(rows, q0, margins, velocity, dt)})
    first = plans[0]
    stamp = str(first["query_stamp_ns"])
    q0 = np.asarray(first["q0"])
    first_rows = [row for row in selectors if row.get("query_stamp_ns") == stamp
                  and row.get("timestep") == "1"]
    subsets = []
    for size in range(1, 6):
        for indices in itertools.combinations(range(5), size):
            selected = [first_rows[index] for index in indices]
            diag = diagnostic_slack(selected, q0, margins, velocity, dt)
            subsets.append({"labels": [point_label(json.loads(row["point"])) for row in selected],
                            "size": size, "diagnostic": diag})
    critical = next(row for row in first_rows
                    if point_label(json.loads(row["point"])) == "P1a_O1_y0_z090")
    previous_margins = dict(margins)
    previous_margins[solver_point_key(json.loads(critical["point"]))] = .20
    return {
        "dt_s": dt, "velocity_limits_rad_s": velocity,
        "feedback": feedback,
        "final_margins": {point_label(point): margins[solver_point_key(point)] for point in final_points},
        "failed_plans": plans, "plan156_all_subsets": subsets,
        "critical_previous_margin_diagnostic": diagnostic_slack(
            [critical], q0, previous_margins, velocity, dt),
        "proof_scope": "timestep-1 hard witness rows + exact first-step dynamics/velocity bounds; all other constraints removed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--radius-samples", type=int, default=512)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    trace1_path = next(TRACE.glob("c46_obligation_001_*.json"))
    trace5_path = next(TRACE.glob("c46_obligation_005_*.json"))
    trace1 = json.loads(trace1_path.read_text())
    trace5 = json.loads(trace5_path.read_text())
    runtime = runtime_from_trace(trace5)
    geometry = Geometry()

    summaries = string_rows(RUN / "local_planner_summary.csv", {"start_ros_s": 0., "end_ros_s": 100.})
    plan156 = next(row for row in summaries if row.get("event") == "scp_failed"
                   and row.get("plan_seq") == "156")
    stamp = plan156["query_stamp_ns"]
    query = next(row for row in numeric_rows(RUN / "local_gcdf_query.csv")
                 if row["field.header.stamp"] == stamp)
    q0 = vector(query, "field.points0.positions")
    selector = string_rows(RUN / "local_witness_selector_diagnostics.csv",
                           {"start_ros_s": 0., "end_ros_s": 100.})
    point_rows = [row for row in selector if row.get("query_stamp_ns") == stamp
                  and row.get("timestep") == "1"]
    final_points = [json.loads(row["point"]) for row in point_rows]
    final_points.sort(key=point_label)
    targets = {
        **{point_label(point): [point] for point in final_points},
        "O1_high_four": [point for point in final_points if point[0] > 0],
        "O5_high_one": [point for point in final_points if point[0] < 0],
        "all_five": final_points,
    }

    radii = [0.0, .05, .10, .15, .20, .25, .30, .40, .60, .80, 1.00]
    radius_results = {}
    for name, points in targets.items():
        print("radius", name, flush=True)
        radius_results[name] = radius_scan(
            runtime, points, q0, args.radius_samples, radii)

    branch_results = {}
    path_results = {}
    wrist_results = {}
    for name in ("O1_high_four", "O5_high_one", "all_five"):
        points = targets[name]
        print("branches", name, flush=True)
        branches = optimize_branches(runtime, points, q0)
        branch_results[name] = branches
        path_results[name] = {}
        for sensor in (6, 7):
            print("path", name, "S" + str(sensor), flush=True)
            path_results[name][str(sensor)] = path_diagnostic(
                runtime, geometry, points, final_points, q0, branches[str(sensor)], sensor)
        if name in ("O1_high_four", "O5_high_one"):
            wrist_results[name] = {}
            for sensor in range(6):
                if not branches[str(sensor)]["root_found"]:
                    continue
                print("wrist", name, "S" + str(sensor), flush=True)
                wrist_results[name][str(sensor)] = wrist_clearance_optimization(
                    runtime, geometry, points, final_points, q0, branches[str(sensor)], sensor)

    qp = qp_subset_diagnostic(final_points)
    sources = [trace1_path, trace5_path, CHECKPOINT, REFERENCE_URDF, SELF_FILTER_URDF,
               RUN / "local_planner_summary.csv", RUN / "local_gcdf_query.csv",
               RUN / "local_witness_selector_diagnostics.csv", RUN / "verification_outcome.csv",
               LOG, Path(__file__).resolve()]
    result = {
        "run_id": RID, "qualification": "OFFLINE_DIAGNOSTIC_ONLY",
        "device": "CPU FP32", "plan_seq": 156, "query_stamp_ns": int(stamp),
        "q0": q0, "final_points": [{"label": point_label(point), "xyz_base_m": point}
                                      for point in final_points],
        "sensor_groups": GROUPS, "sensor_masks": runtime.sensor_masks_np,
        "radius_scan": radius_results, "production_branch_counterfactuals": branch_results,
        "s6_s7_straight_path_diagnostics": path_results,
        "upstream_sensor_wrist_clearance_endpoints": wrist_results,
        "qp_first_step_subset_diagnostic": qp,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "limits": [
            "No ROS, Gazebo, execution, production parameter change, or safety-gate bypass.",
            "Radius scan is deterministic sampled search and does not prove a global minimum radius.",
            "Straight paths are counterfactual joint interpolation, not Sparse-SCP trajectories.",
            "Point-center primitive clearance omits voxel extent, full map, and continuous certified execution.",
            "Wrist optimization is endpoint-only and does not certify self-collision or a continuous path.",
            "QP infeasibility proof applies to the recorded local linearized timestep-1 row, not the nonlinear planning problem.",
        ],
    }
    save(args.output_dir / "diagnostic.json", result)
    digest = {
        "critical_qp_row": qp["failed_plans"][0]["single_rows"]["P1a_O1_y0_z090"],
        "critical_previous_margin": qp["critical_previous_margin_diagnostic"],
        "s6_s7_paths": {name: {sensor: {key: value for key, value in path.items() if key != "rows"}
                                for sensor, path in paths.items()}
                         for name, paths in path_results.items()},
        "output": str(args.output_dir / "diagnostic.json"),
    }
    print(json.dumps(clean(digest), indent=2), flush=True)


if __name__ == "__main__":
    main()
