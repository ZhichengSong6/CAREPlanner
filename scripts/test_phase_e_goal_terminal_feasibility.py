#!/usr/bin/env python3
"""
Phase E Test 1: terminal goal feasibility in the obstacle world.

Question answered
-----------------
For one benchmark case (default: case_014), does there exist at least one
joint configuration q such that:

  1) FK(q) reaches the requested EE pose within tolerance;
  2) q respects URDF joint limits;
  3) the conservative CAREPlanner body-sphere model is collision-free with
     the static box obstacles in the Gazebo world?

This is intentionally OFFLINE. It does not start ROS, Gazebo, the planner,
confidence map, or any controller.

The collision test matches the current CAREPlanner GCDF body proxy:
  body_samples.yaml sphere radius + --body-inflation (default 0.015 m).

Outputs one JSON report plus a concise terminal summary with one of:
  NO_IK
  IK_EXISTS_BUT_ALL_COLLIDE
  COLLISION_FREE_IK_EXISTS
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import pinocchio as pin
except Exception as exc:
    raise SystemExit(
        "ERROR: Python package 'pinocchio' is required. "
        "Run this from a CAREPlanner environment where Pinocchio is installed. "
        f"Original error: {exc}"
    )


@dataclass
class BodySphere:
    link_name: str
    frame_name: str
    center_link: np.ndarray
    radius: float


@dataclass
class BoxObstacle:
    name: str
    center_world: np.ndarray
    rotation_world_box: np.ndarray
    size: np.ndarray


def _parse_vec(text: Optional[str], n: int, default: Sequence[float]) -> np.ndarray:
    if text is None or not text.strip():
        return np.asarray(default, dtype=float)
    vals = [float(v) for v in text.strip().split()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} numbers, got {len(vals)} in: {text!r}")
    return np.asarray(vals, dtype=float)


def _rpy_to_rotation(rpy: np.ndarray) -> np.ndarray:
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # SDF/URDF fixed-axis roll-pitch-yaw: Rz(yaw) Ry(pitch) Rx(roll).
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cr, -sr],
                   [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0],
                   [sy, cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def _pose_to_transform(text: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    pose = _parse_vec(text, 6, [0, 0, 0, 0, 0, 0])
    return _rpy_to_rotation(pose[3:6]), pose[0:3]


def _compose(
    Ra: np.ndarray, ta: np.ndarray,
    Rb: np.ndarray, tb: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return Ra @ Rb, ta + Ra @ tb


def load_world_boxes(world_path: Path) -> List[BoxObstacle]:
    root = ET.parse(str(world_path)).getroot()
    world = root.find("world")
    if world is None:
        raise RuntimeError(f"No <world> element in {world_path}")

    boxes: List[BoxObstacle] = []
    for model in world.findall("model"):
        model_name = model.get("name", "unnamed_model")
        Rm, tm = _pose_to_transform(
            model.findtext("pose", default="0 0 0 0 0 0"))

        for link in model.findall("link"):
            Rl, tl = _pose_to_transform(
                link.findtext("pose", default="0 0 0 0 0 0"))
            Rml, tml = _compose(Rm, tm, Rl, tl)

            for collision in link.findall("collision"):
                box = collision.find("./geometry/box")
                if box is None:
                    continue
                size_text = box.findtext("size")
                if size_text is None:
                    continue
                size = _parse_vec(size_text, 3, [0, 0, 0])

                Rc, tc = _pose_to_transform(
                    collision.findtext("pose", default="0 0 0 0 0 0"))
                Rw, tw = _compose(Rml, tml, Rc, tc)

                boxes.append(
                    BoxObstacle(
                        name=f"{model_name}/{collision.get('name', 'collision')}",
                        center_world=tw,
                        rotation_world_box=Rw,
                        size=size,
                    )
                )

    if not boxes:
        raise RuntimeError(f"No box collision geometries found in {world_path}")
    return boxes


def load_body_spheres(path: Path, body_inflation: float) -> List[BodySphere]:
    with path.open("r") as f:
        doc = yaml.safe_load(f)

    links = doc.get("body_sampling", {}).get("links", [])
    spheres: List[BodySphere] = []
    for item in links:
        if not bool(item.get("include_for_risk", False)):
            continue
        link_name = str(item["link_name"])
        frame_name = str(item.get("frame", link_name))
        for sample in item.get("samples", []):
            radius = float(sample["radius"]) + body_inflation
            spheres.append(
                BodySphere(
                    link_name=link_name,
                    frame_name=frame_name,
                    center_link=np.asarray(sample["center"], dtype=float),
                    radius=radius,
                )
            )

    if not spheres:
        raise RuntimeError(f"No risk body spheres loaded from {path}")
    return spheres


def signed_distance_point_to_box(
    p_world: np.ndarray, box: BoxObstacle
) -> float:
    """Signed Euclidean distance from a point to an oriented box volume."""
    p_local = box.rotation_world_box.T @ (p_world - box.center_world)
    q = np.abs(p_local) - 0.5 * box.size
    outside = np.maximum(q, 0.0)
    outside_dist = float(np.linalg.norm(outside))
    inside_term = min(float(np.max(q)), 0.0)
    return outside_dist + inside_term


def quaternion_xyzw_to_rotation(q_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q_xyzw]
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n <= 0.0:
        raise ValueError("Zero-norm goal quaternion")
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=float)


def pose_error(
    model, data, q: np.ndarray, frame_id: int,
    target_position: np.ndarray, target_rotation: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    M = data.oMf[frame_id]

    # Local-frame translational and rotational errors.
    p_err_local = M.rotation.T @ (target_position - M.translation)
    R_err_local = M.rotation.T @ target_rotation
    w_err_local = np.asarray(pin.log3(R_err_local)).reshape(3)

    err = np.concatenate([p_err_local, w_err_local])
    return err, float(np.linalg.norm(p_err_local)), float(np.linalg.norm(w_err_local))


def solve_ik_dls(
    model,
    frame_id: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    pos_tol: float,
    ori_tol: float,
    max_iters: int,
    damping: float,
    max_step_inf: float,
) -> Tuple[np.ndarray, bool, int, float, float]:
    data = model.createData()
    q = np.clip(np.asarray(q_seed, dtype=float).copy(), lower, upper)

    best_q = q.copy()
    best_score = float("inf")
    best_pos = float("inf")
    best_ori = float("inf")

    for it in range(max_iters):
        err, pos_err, ori_err = pose_error(
            model, data, q, frame_id, target_position, target_rotation)

        # Normalize the two tolerances so both matter in "best" tracking.
        score = (pos_err / max(pos_tol, 1e-12))**2 + (
            ori_err / max(ori_tol, 1e-12))**2
        if score < best_score:
            best_score = score
            best_q = q.copy()
            best_pos = pos_err
            best_ori = ori_err

        if pos_err <= pos_tol and ori_err <= ori_tol:
            return q, True, it + 1, pos_err, ori_err

        J = pin.computeFrameJacobian(
            model, data, q, frame_id, pin.ReferenceFrame.LOCAL)
        J = np.asarray(J, dtype=float)

        # Damped least squares in the frame-local 6D tangent space.
        A = J @ J.T + (damping * damping) * np.eye(6)
        try:
            v = J.T @ np.linalg.solve(A, err)
        except np.linalg.LinAlgError:
            break

        v = np.asarray(v).reshape(model.nv)
        v_inf = float(np.max(np.abs(v))) if v.size else 0.0
        if v_inf > max_step_inf:
            v *= max_step_inf / v_inf

        # Small backtracking line search on normalized pose error.
        accepted = False
        base_score = score
        for alpha in (1.0, 0.5, 0.25, 0.125):
            q_try = pin.integrate(model, q, alpha * v)
            q_try = np.clip(np.asarray(q_try).reshape(-1), lower, upper)
            _, p_try, o_try = pose_error(
                model, data, q_try, frame_id,
                target_position, target_rotation)
            score_try = (
                p_try / max(pos_tol, 1e-12))**2 + (
                o_try / max(ori_tol, 1e-12))**2
            if score_try < base_score:
                q = q_try
                accepted = True
                break
        if not accepted:
            # A tiny direct step helps escape numerical plateaus.
            q = pin.integrate(model, q, 0.05 * v)
            q = np.clip(np.asarray(q).reshape(-1), lower, upper)

    return best_q, False, max_iters, best_pos, best_ori


def transform_body_spheres(
    model, data, q: np.ndarray, spheres: Sequence[BodySphere]
) -> Iterable[Tuple[BodySphere, np.ndarray]]:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    frame_cache: Dict[str, object] = {}
    for sphere in spheres:
        if sphere.frame_name not in frame_cache:
            fid = model.getFrameId(sphere.frame_name)
            if fid >= len(model.frames):
                raise RuntimeError(
                    f"Body-sample frame {sphere.frame_name!r} not in Pinocchio model")
            frame_cache[sphere.frame_name] = data.oMf[fid]

        M = frame_cache[sphere.frame_name]
        center_world = M.translation + M.rotation @ sphere.center_link
        yield sphere, np.asarray(center_world, dtype=float)


def terminal_clearance(
    model,
    q: np.ndarray,
    spheres: Sequence[BodySphere],
    boxes: Sequence[BoxObstacle],
) -> Tuple[float, Dict[str, object]]:
    data = model.createData()
    best = float("inf")
    best_info: Dict[str, object] = {}

    for sphere, center_world in transform_body_spheres(
        model, data, q, spheres
    ):
        for box in boxes:
            d_center_box = signed_distance_point_to_box(center_world, box)
            clearance = d_center_box - sphere.radius
            if clearance < best:
                best = float(clearance)
                best_info = {
                    "link_name": sphere.link_name,
                    "frame_name": sphere.frame_name,
                    "sphere_center_world": center_world.tolist(),
                    "inflated_sphere_radius_m": float(sphere.radius),
                    "obstacle": box.name,
                    "obstacle_center_world": box.center_world.tolist(),
                    "obstacle_size": box.size.tolist(),
                    "clearance_m": float(clearance),
                }

    return best, best_info


def q_is_duplicate(q: np.ndarray, existing: Sequence[np.ndarray], tol: float) -> bool:
    return any(float(np.linalg.norm(q - x)) <= tol for x in existing)


def main() -> int:
    repo_default = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=repo_default)
    ap.add_argument("--case-id", default="case_014")
    ap.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help="Default: src/egocentric_arm_planner/config/phase_c2_vbc_cases.json",
    )
    ap.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Default: Arm_with_self_filter_collision.urdf",
    )
    ap.add_argument(
        "--body-samples",
        type=Path,
        default=None,
        help="Default: src/care_confidence_map/config/body_samples.yaml",
    )
    ap.add_argument(
        "--world",
        type=Path,
        default=None,
        help="Default: maixsense_obstacles.world",
    )
    ap.add_argument("--ee-frame", default="EE_link")
    ap.add_argument("--random-starts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--pos-tol", type=float, default=1e-3)
    ap.add_argument("--ori-tol", type=float, default=1e-2)
    ap.add_argument("--damping", type=float, default=2e-3)
    ap.add_argument("--max-step-inf", type=float, default=0.20)
    ap.add_argument(
        "--body-inflation",
        type=float,
        default=0.015,
        help="Meters added to every CAREPlanner risk-body sphere (default 15 mm).",
    )
    ap.add_argument(
        "--required-clearance",
        type=float,
        default=0.0,
        help="Required terminal clearance after body inflation, meters.",
    )
    ap.add_argument(
        "--dedup-q-l2",
        type=float,
        default=1e-3,
        help="Treat IK solutions within this q-space L2 distance as duplicates.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    repo = args.repo.resolve()
    cases_json = (args.cases_json or (
        repo / "src/egocentric_arm_planner/config/phase_c2_vbc_cases.json")).resolve()
    urdf = (args.urdf or (
        repo / "src/arm_description/urdf/Arm_with_self_filter_collision.urdf")).resolve()
    body_samples_path = (args.body_samples or (
        repo / "src/care_confidence_map/config/body_samples.yaml")).resolve()
    world = (args.world or (
        repo / "src/arm_description/worlds/maixsense_obstacles.world")).resolve()
    output_json = (args.output_json or (
        repo / "outputs/phase_e_test1" /
        f"{args.case_id}_terminal_feasibility.json")).resolve()

    for path in (cases_json, urdf, body_samples_path, world):
        if not path.is_file():
            raise SystemExit(f"ERROR: file not found: {path}")

    with cases_json.open("r") as f:
        cases_doc = json.load(f)
    case = next(
        (x for x in cases_doc.get("cases", []) if x.get("case_id") == args.case_id),
        None,
    )
    if case is None:
        raise SystemExit(f"ERROR: case_id={args.case_id!r} not found in {cases_json}")

    target_position = np.asarray(case["goal_position"], dtype=float)
    target_rotation = quaternion_xyzw_to_rotation(case["goal_orientation"])

    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 7 or model.nv != 7:
        raise SystemExit(
            f"ERROR: expected 7-DoF model, got nq={model.nq}, nv={model.nv}")

    frame_id = model.getFrameId(args.ee_frame)
    if frame_id >= len(model.frames):
        raise SystemExit(f"ERROR: EE frame {args.ee_frame!r} not found in URDF")

    lower = np.asarray(model.lowerPositionLimit, dtype=float).reshape(-1)
    upper = np.asarray(model.upperPositionLimit, dtype=float).reshape(-1)

    if not (
        np.all(np.isfinite(lower)) and
        np.all(np.isfinite(upper)) and
        np.all(upper > lower)
    ):
        raise SystemExit(
            "ERROR: invalid/non-finite joint limits in Pinocchio model")

    spheres = load_body_spheres(body_samples_path, args.body_inflation)
    boxes = load_world_boxes(world)

    # Deterministic seeds first because they are physically meaningful.
    seeds: List[Tuple[str, np.ndarray]] = []
    seeds.append(("neutral", np.asarray(pin.neutral(model), dtype=float).reshape(-1)))
    for key in ("initial_q", "q_nom_deadline", "q_vis"):
        if key in case:
            qv = np.asarray(case[key], dtype=float).reshape(-1)
            if qv.size == model.nq:
                seeds.append((key, qv))

    rng = np.random.default_rng(args.seed)
    for i in range(args.random_starts):
        q0 = rng.uniform(lower, upper)
        seeds.append((f"random_{i:04d}", q0))

    unique_solutions: List[np.ndarray] = []
    solutions: List[Dict[str, object]] = []
    attempts_converged = 0

    for seed_name, q0 in seeds:
        q_sol, ok, iters, pos_err, ori_err = solve_ik_dls(
            model=model,
            frame_id=frame_id,
            target_position=target_position,
            target_rotation=target_rotation,
            q_seed=q0,
            lower=lower,
            upper=upper,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            max_iters=args.max_iters,
            damping=args.damping,
            max_step_inf=args.max_step_inf,
        )
        if not ok:
            continue

        attempts_converged += 1
        if q_is_duplicate(q_sol, unique_solutions, args.dedup_q_l2):
            continue

        unique_solutions.append(q_sol.copy())
        min_clearance, closest = terminal_clearance(
            model, q_sol, spheres, boxes)

        collision_free = bool(
            min_clearance >= args.required_clearance)

        solutions.append({
            "seed": seed_name,
            "q": q_sol.tolist(),
            "position_error_m": float(pos_err),
            "orientation_error_rad": float(ori_err),
            "min_clearance_m": float(min_clearance),
            "collision_free": collision_free,
            "closest_pair": closest,
        })

    solutions.sort(key=lambda x: float(x["min_clearance_m"]), reverse=True)

    collision_free_solutions = [x for x in solutions if x["collision_free"]]

    if not solutions:
        verdict = "NO_IK"
    elif collision_free_solutions:
        verdict = "COLLISION_FREE_IK_EXISTS"
    else:
        verdict = "IK_EXISTS_BUT_ALL_COLLIDE"

    # Geometric sanity check of the goal point itself against world boxes.
    ee_goal_box = []
    for box in boxes:
        ee_goal_box.append({
            "obstacle": box.name,
            "ee_origin_signed_distance_to_box_m":
                float(signed_distance_point_to_box(target_position, box)),
        })
    ee_goal_box.sort(
        key=lambda x: x["ee_origin_signed_distance_to_box_m"])

    report = {
        "test": "phase_e_test1_terminal_goal_feasibility",
        "case_id": args.case_id,
        "verdict": verdict,
        "goal_position": target_position.tolist(),
        "goal_orientation_xyzw": list(case["goal_orientation"]),
        "ee_frame": args.ee_frame,
        "urdf": str(urdf),
        "body_samples": str(body_samples_path),
        "world": str(world),
        "body_inflation_m": float(args.body_inflation),
        "required_clearance_m": float(args.required_clearance),
        "ik": {
            "total_seed_attempts": len(seeds),
            "converged_attempts_including_duplicates": attempts_converged,
            "unique_ik_solutions": len(solutions),
            "collision_free_unique_solutions": len(collision_free_solutions),
            "pos_tol_m": float(args.pos_tol),
            "ori_tol_rad": float(args.ori_tol),
        },
        "ee_goal_point_vs_obstacles": ee_goal_box,
        "obstacle_boxes": [
            {
                "name": b.name,
                "center_world": b.center_world.tolist(),
                "size": b.size.tolist(),
            }
            for b in boxes
        ],
        "solutions": solutions,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(report, f, indent=2)

    print("=" * 78)
    print("PHASE E TEST 1 — TERMINAL GOAL FEASIBILITY")
    print("=" * 78)
    print(f"case_id              : {args.case_id}")
    print(f"goal_position        : {target_position.tolist()}")
    print(f"body inflation       : {args.body_inflation:.4f} m")
    print(f"seed attempts        : {len(seeds)}")
    print(f"converged attempts   : {attempts_converged}")
    print(f"unique IK solutions  : {len(solutions)}")
    print(f"collision-free IK    : {len(collision_free_solutions)}")
    print(f"VERDICT               : {verdict}")

    if ee_goal_box:
        nearest_goal = ee_goal_box[0]
        print(
            "EE-origin nearest box : {} signed_distance={:.6f} m".format(
                nearest_goal["obstacle"],
                nearest_goal["ee_origin_signed_distance_to_box_m"],
            )
        )

    if solutions:
        best = solutions[0]
        cp = best["closest_pair"]
        print(
            "best terminal clearance: {:.6f} m  collision_free={}".format(
                best["min_clearance_m"], int(bool(best["collision_free"])))
        )
        if cp:
            print(
                "closest geometry pair : link={} obstacle={} clearance={:.6f} m".format(
                    cp.get("link_name"),
                    cp.get("obstacle"),
                    cp.get("clearance_m"),
                )
            )
        print("best q                :", np.array(best["q"]))

    print(f"JSON report           : {output_json}")
    print("=" * 78)

    # Exit status is useful for scripting:
    #   0 = collision-free solution exists
    #   2 = IK exists but all terminal solutions collide
    #   3 = no IK solution found
    if verdict == "COLLISION_FREE_IK_EXISTS":
        return 0
    if verdict == "IK_EXISTS_BUT_ALL_COLLIDE":
        return 2
    return 3


if __name__ == "__main__":
    sys.exit(main())
