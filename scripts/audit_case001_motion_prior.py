#!/usr/bin/env python3
"""Offline audit of the CASE001 exact-primitive cold-start confidence prior.

Uses the same URDF primitive shapes, grid-center signed distances, and per-link
inflation rule as the confidence map. The 50-ms bound reproduces the analytic
triangle bound in VbcPrimitiveModel::displacementBounds; the corner replay is
an achievable velocity-box diagnostic, not a proof of sufficiency.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin
import yaml


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "src/arm_description/urdf/Arm_with_self_filter_collision.urdf"
CONFIG = ROOT / "src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml"
PLANNER = ROOT / "src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml"
CASES = ROOT / "src/egocentric_arm_planner/config/phase_c2_vbc_cases.json"
O6 = ROOT / "outputs/phase_e_owner_fov_followup/exp90_prior_proximity.json"
OUTPUT = ROOT / "outputs/phase_e_case001_motion_prior_audit/audit.json"
JOINTS = ["joint1", "joint2", "joint3", "joint4", "wrist_joint1", "wrist_joint2", "wrist_joint3"]
LINKS = ["base_link", "link1", "link2", "link3", "link4", "wrist_link1", "wrist_link2", "wrist_link3"]


def vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()], dtype=float)


def collision_sdf(points: np.ndarray, shape: dict, placement: pin.SE3) -> np.ndarray:
    xyz = (points - np.asarray(placement.translation)) @ np.asarray(placement.rotation)
    kind = shape["kind"]
    if kind == "sphere":
        return np.linalg.norm(xyz, axis=1) - shape["radius"]
    if kind == "cylinder":
        gap = np.column_stack((np.linalg.norm(xyz[:, :2], axis=1) - shape["radius"],
                               np.abs(xyz[:, 2]) - shape["half_length"]))
    else:
        gap = np.abs(xyz) - shape["half_size"]
    return np.linalg.norm(np.maximum(gap, 0), axis=1) + np.minimum(np.max(gap, axis=1), 0)


def load_shapes(root: ET.Element) -> list[dict]:
    shapes = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            xyz = vec(origin.attrib.get("xyz", "0 0 0")) if origin is not None else np.zeros(3)
            rpy = vec(origin.attrib.get("rpy", "0 0 0")) if origin is not None else np.zeros(3)
            geometry = collision.find("geometry")
            assert geometry is not None
            kind_element = next(iter(geometry))
            kind = kind_element.tag
            assert kind in {"box", "cylinder", "sphere"}, (name, kind)
            shape = dict(link=name, name=collision.attrib.get("name", ""), kind=kind,
                         local=pin.SE3(pin.rpy.rpyToMatrix(*rpy), xyz))
            if kind == "box":
                shape["half_size"] = vec(kind_element.attrib["size"]) / 2
                shape["circumradius"] = float(np.linalg.norm(shape["half_size"]))
            elif kind == "cylinder":
                shape["radius"] = float(kind_element.attrib["radius"])
                shape["half_length"] = float(kind_element.attrib["length"]) / 2
                shape["circumradius"] = float(np.hypot(shape["radius"], shape["half_length"]))
            else:
                shape["radius"] = float(kind_element.attrib["radius"])
                shape["circumradius"] = shape["radius"]
            shapes.append(shape)
    return shapes


def frame_placements(model: pin.Model, data: pin.Data, shapes: list[dict], q: np.ndarray) -> list[pin.SE3]:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return [data.oMf[model.getFrameId(s["link"])] * s["local"] for s in shapes]


def distance_matrix(points: np.ndarray, shapes: list[dict], placements: list[pin.SE3]) -> np.ndarray:
    return np.column_stack([collision_sdf(points, s, p) for s, p in zip(shapes, placements)])


def farthest_axis_distance_at_q0(shape: dict, pose: pin.SE3,
                                 axis_origin: np.ndarray, axis: np.ndarray) -> float:
    """Upper bound the farthest solid point from a revolute axis at q0."""
    if shape["kind"] == "sphere":
        return float(np.linalg.norm(np.cross(np.asarray(pose.translation) - axis_origin, axis))
                     + shape["radius"])
    if shape["kind"] == "box":
        h = shape["half_size"]
        local = np.array([np.array(signs) * h for signs in itertools.product((-1, 1), repeat=3)])
        arc_error = 0.0
    else:
        # A cylinder is the convex hull of its two cap circles. Distance to a
        # line is convex, so its maximum occurs on a cap circle. A 1024-point
        # angular net plus the 1-Lipschitz chord error is a rigorous upper bound.
        n = 1024
        theta = np.arange(n) * (2 * np.pi / n)
        ring = np.column_stack((shape["radius"] * np.cos(theta),
                                shape["radius"] * np.sin(theta)))
        local = np.vstack([np.column_stack((ring, np.full(n, z)))
                           for z in (-shape["half_length"], shape["half_length"])])
        arc_error = 2 * shape["radius"] * np.sin(np.pi / (2 * n))
    world = np.asarray(pose.translation) + local @ np.asarray(pose.rotation).T
    return float(np.linalg.norm(np.cross(world - axis_origin, axis), axis=1).max() + arc_error)


def analytic_bounds(shapes: list[dict], placements: list[pin.SE3], root: ET.Element,
                    model: pin.Model, data: pin.Data, travel: dict[str, float]) -> tuple[list[float], list[float]]:
    """Global VBC bound and a q0-specific bound with downstream-motion correction."""
    parent = {j.find("child").attrib["link"]: j for j in root.findall("joint")}
    global_bounds, local_bounds = [], []
    for s, pose in zip(shapes, placements):
        radius = float(np.linalg.norm(s["local"].translation)) + s["circumradius"]
        global_bound = local_bound = downstream_global_bound = 0.0
        link = s["link"]
        while link in parent:
            joint = parent[link]
            if joint.attrib["type"] != "fixed":
                assert joint.attrib["type"] in ("revolute", "continuous")
                name = joint.attrib["name"]
                delta = travel[name]
                global_bound += radius * delta
                jid = model.getJointId(name)
                origin = np.asarray(data.oMi[jid].translation)
                axis_local = vec(joint.find("axis").attrib.get("xyz", "0 0 1"))
                axis = np.asarray(data.oMi[jid].rotation) @ axis_local
                axis /= np.linalg.norm(axis)
                axis_radius_q0 = farthest_axis_distance_at_q0(s, pose, origin, axis)
                # Distances to an upstream axis can change by at most the
                # downstream primitive displacement accumulated so far.
                local_bound += delta * (axis_radius_q0 + downstream_global_bound)
                downstream_global_bound += radius * delta
            origin = joint.find("origin")
            radius += float(np.linalg.norm(vec(origin.attrib.get("xyz", "0 0 0")))) if origin is not None else 0
            link = joint.find("parent").attrib["link"]
        global_bounds.append(global_bound)
        local_bounds.append(local_bound)
    return global_bounds, local_bounds


def round6(x: float) -> float:
    return round(float(x), 6)


def surface_samples(shape: dict) -> np.ndarray:
    if shape["kind"] == "box":
        h = shape["half_size"]
        return np.array([np.array(s) * h for s in itertools.product((-1, 0, 1), repeat=3)
                         if any(abs(v) == 1 for v in s)])
    if shape["kind"] == "cylinder":
        a = np.arange(32) * (2 * np.pi / 32)
        ring = np.column_stack((shape["radius"] * np.cos(a), shape["radius"] * np.sin(a)))
        return np.array([[x, y, z] for z in (-shape["half_length"], 0, shape["half_length"])
                         for x, y in ring] + [[0, 0, -shape["half_length"]],
                                               [0, 0, shape["half_length"]]])
    a = np.arange(32) * (2 * np.pi / 32)
    polar = (np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    return shape["radius"] * np.array(
        [[np.sin(t) * np.cos(v), np.sin(t) * np.sin(v), np.cos(t)]
         for t in polar for v in a] + [[0, 0, -1], [0, 0, 1]])


def main() -> None:
    conf = yaml.safe_load(CONFIG.read_text())
    planner = yaml.safe_load(PLANNER.read_text())
    case = next(c for c in json.loads(CASES.read_text())["cases"] if c["case_id"] == "case_001")
    q0 = np.array(case["initial_q"], dtype=float)
    root = ET.parse(URDF).getroot()
    shapes = load_shapes(root)
    assert len(shapes) == 26
    assert {s["link"] for s in shapes} == set(LINKS)
    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()
    q_indices = [model.joints[model.getJointId(j)].idx_q for j in JOINTS]
    q = np.zeros(model.nq)
    q[q_indices] = q0
    initial = frame_placements(model, data, shapes, q)
    grid = conf["confidence_map"]
    resolution = float(grid["resolution"])
    # Match confidence_map_node.cpp: floor(range/resolution)+1 in binary64.
    # The current values yield 38 x 38 x 23, not an idealized 39 x 39 x 24.
    axes = [float(grid[f"{a}_min"]) + resolution * np.arange(
            math.floor((float(grid[f"{a}_max"]) - float(grid[f"{a}_min"])) / resolution) + 1)
            for a in "xyz"]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    d0 = distance_matrix(points, shapes, initial)
    static_min = d0.min(axis=1)
    radius = {link: float(conf["current_body_prior"]["link_inflation_radius"][link]) for link in LINKS}
    r = np.array([radius[s["link"]] for s in shapes])
    prior = (d0 <= r + 1e-12).any(axis=1)
    baseline = (d0 <= .1 + 1e-12).any(axis=1)
    assert int(prior.sum()) == 1227, f"initial raster mismatch: {prior.sum()}"
    assert int(baseline.sum()) == 731, f"baseline raster mismatch: {baseline.sum()}"

    dt = float(planner["local_planner"]["horizon_duration"]) / int(planner["local_planner"]["num_intervals"])
    assert abs(dt - .05) < 1e-12
    velocity = {j: float(planner["mpc"]["joint_velocity_limits"][j]) for j in JOINTS}
    travel = {j: dt * v for j, v in velocity.items()}
    bound, local_bound = analytic_bounds(shapes, initial, root, model, data, travel)
    cert = {}
    for link in LINKS:
        indices = [i for i, s in enumerate(shapes) if s["link"] == link]
        upper = max(bound[i] for i in indices)
        local_upper = max(local_bound[i] for i in indices)
        cert[link] = dict(primitives=len(indices), configured_radius_m=radius[link],
                          analytic_50ms_bound_m=round6(upper),
                          configured_minus_bound_m=round6(radius[link] - upper),
                          sufficient_by_analytic_bound=radius[link] >= upper,
                          q0_specific_analytic_bound_m=round6(local_upper),
                          sufficient_by_q0_bound=radius[link] >= local_upper)

    # Exact FK at every corner of the 50-ms velocity box gives an attainable
    # subset of the sweep. This is a diagnostic and cannot establish coverage.
    moved_union = np.zeros(len(points), dtype=bool)
    escaped_union = np.zeros(len(points), dtype=bool)
    per_link_max_static_distance = {link: -np.inf for link in LINKS}
    per_link_witness = {}
    first_escape = None
    surface_witness = None
    surface_max_gap = -np.inf
    surface_count = 0
    local_surfaces = [surface_samples(s) for s in shapes]
    max_surface_displacement = {link: 0.0 for link in LINKS}
    for signs in itertools.product((-1, 1), repeat=7):
        delta = np.array([signs[i] * travel[j] for i, j in enumerate(JOINTS)])
        q[q_indices] = q0 + delta
        poses = frame_placements(model, data, shapes, q)
        moved = distance_matrix(points, shapes, poses)
        occupied = (moved <= 1e-12).any(axis=1)
        moved_union |= occupied
        escaped = occupied & ~prior
        escaped_union |= escaped
        if escaped.any() and first_escape is None:
            idx = int(np.flatnonzero(escaped)[0])
            pi = int(np.argmin(moved[idx]))
            first_escape = dict(point_xyz_m=[round6(v) for v in points[idx]],
                                q_delta_rad=[round6(v) for v in delta],
                                moved_link=shapes[pi]["link"], moved_collision=shapes[pi]["name"],
                                moved_signed_distance_m=round6(moved[idx, pi]),
                                startup_min_signed_distance_m=round6(static_min[idx]))
        for link in LINKS:
            ix = [i for i, s in enumerate(shapes) if s["link"] == link]
            body = (moved[:, ix] <= 1e-12).any(axis=1)
            if not body.any():
                continue
            static_link_distance = d0[body][:, ix].min(axis=1)
            j = int(np.argmax(static_link_distance))
            value = float(static_link_distance[j])
            if value > per_link_max_static_distance[link]:
                idx = int(np.flatnonzero(body)[j])
                per_link_max_static_distance[link] = value
                per_link_witness[link] = dict(point_xyz_m=[round6(v) for v in points[idx]],
                                              q_delta_rad=[round6(v) for v in delta],
                                              startup_same_link_signed_distance_m=round6(value),
                                              inside_configured_global_prior=bool(prior[idx]))
        for shape, pose, initial_pose, samples in zip(shapes, poses, initial, local_surfaces):
            world = np.asarray(pose.translation) + samples @ np.asarray(pose.rotation).T
            world0 = np.asarray(initial_pose.translation) + samples @ np.asarray(initial_pose.rotation).T
            max_surface_displacement[shape["link"]] = max(
                max_surface_displacement[shape["link"]],
                float(np.linalg.norm(world - world0, axis=1).max()))
            initial_distances = distance_matrix(world, shapes, initial)
            gaps = (initial_distances - r).min(axis=1)
            surface_count += len(samples)
            k = int(np.argmax(gaps))
            if gaps[k] > surface_max_gap:
                surface_max_gap = float(gaps[k])
                surface_witness = dict(point_xyz_m=[round6(v) for v in world[k]],
                                       q_delta_rad=[round6(v) for v in delta],
                                       moved_link=shape["link"], moved_collision=shape["name"],
                                       distance_beyond_all_configured_dilations_m=round6(gaps[k]),
                                       startup_min_signed_distance_m=round6(initial_distances[k].min()))
    for link in LINKS:
        cert[link]["corner_replay_required_grid_center_radius_m"] = round6(max(0, per_link_max_static_distance[link]))
        cert[link]["corner_replay_max_surface_point_displacement_m"] = round6(max_surface_displacement[link])
        cert[link]["corner_replay_witness"] = per_link_witness[link]
        assert max_surface_displacement[link] <= cert[link]["q0_specific_analytic_bound_m"] + 2e-6

    old = json.loads(O6.read_text())
    o6 = []
    for row in old["rows"][:7]:
        p = np.array([row["point_xyz"]])
        ds = distance_matrix(p, shapes, initial)[0]
        covered = bool((ds <= r + 1e-12).any())
        o6.append(dict(point_xyz_m=row["point_xyz"], exact_signed_distance_m=round6(ds.min()),
                       covered_by_prior=covered, nearest_link=shapes[int(np.argmin(ds))]["link"]))
    assert len(o6) == 7 and all(x["covered_by_prior"] for x in o6)

    added = prior & ~baseline
    extra_free = prior & (static_min > 1e-12)
    added_idx = np.flatnonzero(added)
    farthest_idx = int(added_idx[np.argmax(static_min[added_idx])])
    result = dict(
        scope="Offline kinematic/voxel audit; no ROS, Gazebo, sensor packets, or actuator",
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (URDF, CONFIG, PLANNER, CASES, O6)},
        initial_q_rad=[float(x) for x in q0], grid_resolution_m=resolution,
        grid_shape=[len(x) for x in axes], primitive_count=len(shapes),
        dt_s=dt, joint_velocity_limits_rad_s=velocity,
        raster=dict(configured_prior_cells=int(prior.sum()), baseline_0p10_cells=int(baseline.sum()),
                    added_cells=int(added.sum()), static_body_center_cells=int((static_min <= 1e-12).sum()),
                    prior_cells_outside_static_body=int(extra_free.sum()),
                    added_cells_outside_static_body=int((added & extra_free).sum()),
                    added_cells_touched_at_velocity_box_corners=int((added & moved_union).sum()),
                    farthest_added_center=dict(point_xyz_m=[round6(v) for v in points[farthest_idx]],
                                               static_body_signed_distance_m=round6(static_min[farthest_idx]))),
        per_link=cert, o6=o6,
        corner_replay=dict(corners=128, reachable_grid_center_union=int(moved_union.sum()),
                           reachable_centers_outside_prior=int(escaped_union.sum()),
                           witness=first_escape, surface_samples_evaluated=surface_count,
                           worst_surface_gap_m=round6(surface_max_gap),
                           worst_surface_witness=surface_witness),
        interpretation="Analytic bounds are sufficient upper bounds, not required radii. Corner replay is a lower-bound counterexample if it escapes. Occupancy of bootstrap cells is not measured here.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "raster": result["raster"],
                      "corner_replay": result["corner_replay"],
                      "per_link": {k: {a: v for a, v in d.items() if a != "corner_replay_witness"}
                                   for k, d in cert.items()}}, indent=2))


if __name__ == "__main__":
    main()
