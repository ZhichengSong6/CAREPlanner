#!/usr/bin/env python3
"""Audit CASE006's recorded 20 mm VBC breach against the frozen startup prior.

This is a sampled, offline geometry audit. It does not reconstruct sensor
observations or certify continuous actuator motion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
from scipy.spatial import cKDTree
import yaml

from audit_case001_motion_prior import (
    CONFIG, JOINTS, ROOT, URDF, collision_sdf, distance_matrix,
    frame_placements, load_shapes,
)


CASES = ROOT / "outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json"
RUN = ROOT / "outputs/c5_5_vbc_gcdf_regime/phase_e_goal_006_r1_case006_bounded_owner_retry_fourth_20260924/run"
OUTPUT = ROOT / "outputs/phase_e_case006_prior_gap/audit.json"
BREACH_POINT = np.array([0.20, -0.15, 0.60])
VBC_MARGIN_M = 0.020


def rows(path: Path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def joint_states(path: Path) -> list[tuple[int, np.ndarray]]:
    result = []
    for row in rows(path):
        names = [row[f"field.name{i}"] for i in range(7)]
        positions = [float(row[f"field.position{names.index(j)}"]) for j in JOINTS]
        result.append((int(row["field.header.stamp"]), np.array(positions)))
    return result


def vbc_events(path: Path) -> tuple[dict, dict]:
    last_safe = first_unsafe = None
    # The ROS exporter leaves commas inside field.data unquoted (e.g. target
    # coordinates). Split only the timestamp so the evidence stays intact.
    for line in path.read_text().splitlines()[1:]:
        stamp, separator, message = line.partition(",")
        if not separator or not stamp.isdigit():
            continue
        t = int(stamp)
        if "trajectory_source=predicted" not in message or t > 21_300_000_000:
            continue
        if "has_violation=1" in message and first_unsafe is None:
            first_unsafe = dict(stamp_ns=t, message=message)
        elif "has_violation=0" in message and first_unsafe is None:
            last_safe = dict(stamp_ns=t, message=message)
    assert last_safe is not None and first_unsafe is not None
    return last_safe, first_unsafe


def field(message: str, name: str) -> str:
    match = re.search(r"(?:^| )" + re.escape(name) + r"=([^ ]+)", message)
    assert match is not None, name
    return match.group(1)


def committed_plan(path: Path, stamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
    row = next(r for r in rows(path) if r.get("field.header.stamp") == str(stamp_ns))
    times, positions = [], []
    index = 0
    while row.get(f"field.points{index}.time_from_start"):
        times.append(int(row[f"field.points{index}.time_from_start"]) / 1e9)
        positions.append([float(row[f"field.points{index}.positions{j}"]) for j in range(7)])
        index += 1
    assert len(times) >= 2 and all(b > a for a, b in zip(times, times[1:]))
    return np.array(times), np.array(positions)


def main() -> None:
    case = next(c for c in json.loads(CASES.read_text())["cases"]
                if c["case_id"] == "phase_e_goal_006")
    conf = yaml.safe_load(CONFIG.read_text())
    root = ET.parse(URDF).getroot()
    shapes = load_shapes(root)
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    q_indices = [model.joints[model.getJointId(j)].idx_q for j in JOINTS]
    q = np.zeros(model.nq)
    q[q_indices] = case["initial_q"]
    initial = frame_placements(model, data, shapes, q)

    grid = conf["confidence_map"]
    resolution = float(grid["resolution"])
    axes = [float(grid[f"{a}_min"]) + resolution * np.arange(
        math.floor((float(grid[f"{a}_max"]) - float(grid[f"{a}_min"])) / resolution) + 1)
        for a in "xyz"]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    radii = np.array([float(conf["current_body_prior"]["link_inflation_radius"][s["link"]])
                      for s in shapes])
    initial_distances = distance_matrix(points, shapes, initial)
    prior = (initial_distances <= radii).any(axis=1)
    assert len(shapes) == 26 and int(prior.sum()) == 1227
    tree = cKDTree(points)

    point_d0 = distance_matrix(BREACH_POINT[None, :], shapes, initial)[0]
    point_inside_prior = bool((point_d0 <= radii).any())
    assert not point_inside_prior
    nearest_initial_idx = int(np.argmin(point_d0))
    margin_to_prior = float(np.min(point_d0 - radii))

    states = joint_states(RUN / "joint_states.csv")
    last_safe, first_unsafe = vbc_events(RUN / "execution_vbc_summary.csv")
    first_unsafe_ns = first_unsafe["stamp_ns"]
    assert field(first_unsafe["message"], "target") == "[0.2,-0.15,0.6]"
    sampled = []
    crossing = None
    previous_slot = None
    first_outside = None
    for stamp, positions in states:
        if stamp > first_unsafe_ns + 100_000_000:
            break
        q[q_indices] = positions
        poses = frame_placements(model, data, shapes, q)
        d = distance_matrix(BREACH_POINT[None, :], shapes, poses)[0]
        closest = int(np.argmin(d))
        if crossing is None and d[closest] < VBC_MARGIN_M:
            crossing = dict(stamp_ns=stamp, min_distance_m=float(d[closest]),
                            link=shapes[closest]["link"], collision=shapes[closest]["name"])

        # A 100 ms sample tests whether the current 20 mm body shell lies
        # inside the *initial* raster. These are actual measured q samples,
        # not an interpolated or continuous swept-volume proof.
        slot = stamp // 100_000_000
        if slot == previous_slot:
            continue
        previous_slot = slot
        shell = np.zeros(len(points), dtype=bool)
        for shape, pose in zip(shapes, poses):
            nearby = tree.query_ball_point(np.asarray(pose.translation),
                                           shape["circumradius"] + VBC_MARGIN_M + 1e-9)
            if nearby:
                nearby = np.asarray(nearby, dtype=int)
                shell[nearby] |= collision_sdf(points[nearby], shape, pose) <= VBC_MARGIN_M
        outside = shell & ~prior
        entry = dict(stamp_ns=stamp, shell_centers=int(shell.sum()),
                     outside_initial_prior_centers=int(outside.sum()),
                     breach_point_distance_m=float(d[closest]))
        sampled.append(entry)
        if first_outside is None and outside.any():
            index = int(np.flatnonzero(outside)[0])
            first_outside = dict(**entry, example_grid_center_m=points[index].tolist())

    assert crossing is not None and sampled and first_outside is not None
    plan_stamp_ns = 20_821_000_000
    plan_times, plan_positions = committed_plan(RUN / "committed_trajectory.csv", plan_stamp_ns)
    nominal_distances = []
    # Production VBC uses linear knot sweeps. A 1 ms replay diagnoses this
    # particular point; it does not replace the VBC's full-grid certificate.
    for phase in np.arange(0, plan_times[-1] + 0.0005, 0.001):
        q[q_indices] = [np.interp(phase, plan_times, plan_positions[:, j]) for j in range(7)]
        poses = frame_placements(model, data, shapes, q)
        nominal_distances.append((float(distance_matrix(BREACH_POINT[None, :], shapes, poses).min()),
                                  float(phase)))
    nominal_min, nominal_min_phase = min(nominal_distances)
    phase_at_first_unsafe = (first_unsafe_ns - plan_stamp_ns) / 1e9
    q[q_indices] = [np.interp(phase_at_first_unsafe, plan_times, plan_positions[:, j])
                    for j in range(7)]
    nominal_at_first_unsafe = float(distance_matrix(
        BREACH_POINT[None, :], shapes, frame_placements(model, data, shapes, q)).min())
    tracker_before_breach = None
    for row in rows(RUN / "tracker_summary.csv"):
        if not row["%time"].isdigit():
            continue
        t = int(row["%time"])
        if t <= first_unsafe_ns and "source=active_trajectory" in row["field.data"]:
            tracker_before_breach = dict(stamp_ns=t,
                bound_m=float(field(row["field.data"], "primitive_tracking_bound_m")))
    assert tracker_before_breach is not None
    hold = None
    for row in rows(RUN / "tracker_summary.csv"):
        if not row["%time"].isdigit():
            continue
        t = int(row["%time"])
        if t >= first_unsafe_ns and "source=external_vbc_hold" in row["field.data"]:
            hold = t
            break
    assert hold is not None
    result = dict(
        scope="CASE006 recorded measured states through first VBC breach; 100 ms shell snapshots",
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (URDF, CONFIG, CASES, RUN / "joint_states.csv",
                                RUN / "committed_trajectory.csv",
                                RUN / "execution_vbc_summary.csv", RUN / "tracker_summary.csv")},
        grid_shape=[len(a) for a in axes], grid_resolution_m=resolution,
        prior_cells=int(prior.sum()), vbc_margin_m=VBC_MARGIN_M,
        breach_point=dict(xyz_m=BREACH_POINT.tolist(), inside_initial_prior=point_inside_prior,
                          startup_min_distance_m=float(point_d0[nearest_initial_idx]),
                          startup_nearest_link=shapes[nearest_initial_idx]["link"],
                          startup_gap_beyond_configured_prior_m=margin_to_prior),
        certified_plan=dict(stamp_ns=plan_stamp_ns, nominal_min_distance_m=nominal_min,
                            nominal_min_phase_s=nominal_min_phase,
                            nominal_distance_at_first_unsafe_m=nominal_at_first_unsafe,
                            nominal_margin_above_required_shell_m=nominal_min - VBC_MARGIN_M,
                            tracker_before_breach=tracker_before_breach,
                            conservative_required_nominal_margin_m=2 * VBC_MARGIN_M),
        timeline=dict(last_safe_audit_ns=last_safe["stamp_ns"],
                      last_safe_reason=field(last_safe["message"], "reason"),
                      first_unsafe_audit_ns=first_unsafe_ns,
                      first_unsafe_distance_m=float(field(first_unsafe["message"],
                                                            "primitive_signed_distance_m")),
                      first_measured_shell_crossing=crossing,
                      first_tracker_vbc_hold_ns=hold,
                      audit_to_hold_ms=(hold - first_unsafe_ns) / 1e6),
        shell_samples=dict(count=len(sampled), with_outside_initial_prior=sum(
            x["outside_initial_prior_centers"] > 0 for x in sampled),
            first_outside=first_outside, max_outside=max(
                sampled, key=lambda x: x["outside_initial_prior_centers"]),
            around_breach=[x for x in sampled if 20_700_000_000 <= x["stamp_ns"] <= 21_300_000_000]),
        limitations=["Startup raster reconstructed from nominal initial_q; no saved refresh-time TF snapshot.",
                     "Sampled shell centers do not certify between joint-state samples or full voxel cubes.",
                     "Sensor-derived confidence and actuator stopping distance are not reconstructed."])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "breach_point": result["breach_point"],
                      "certified_plan": result["certified_plan"],
                      "timeline": result["timeline"], "shell_samples": result["shell_samples"]}, indent=2))


if __name__ == "__main__":
    main()
