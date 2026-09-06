#!/usr/bin/env python3
"""Generate diverse random initial configurations for Phase-E obstacle tests.

Unlike the obstacle goal sampler, this script samples q directly in joint space.
It therefore does not inherit the fixed goal position/orientation bias of the
old terminal IK pool.

A candidate q0 is accepted only if:
  1. it lies inside shrunken URDF joint limits;
  2. the CAREPlanner risk-body spheres (+15 mm by default) are collision-free
     with the static obstacle world;
  3. the experiment's startup trusted-free body prior is obstacle-free
     (raw body by default: +0 mm);
  4. an approximate non-adjacent-link body-sphere self-collision check passes;
  5. the whole risk body remains inside the confidence-map workspace;
  6. its EE lies in a broad useful workspace;
  7. it is sufficiently different in q-space from already accepted q0s.

This is an offline sampler only. It does not launch ROS/Gazebo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pinocchio as pin
import yaml

from test_phase_e_goal_terminal_feasibility import (
    BodySphere,
    load_body_spheres,
    load_world_boxes,
    terminal_clearance,
    transform_body_spheres,
)


LINK_ORDER = {
    "base_link": 0,
    "link1": 1,
    "link2": 2,
    "link3": 3,
    "link4": 4,
    "wrist_link1": 5,
    "wrist_link2": 6,
    "wrist_link3": 7,
}


def load_all_body_spheres(path: Path, inflation_m: float) -> List[BodySphere]:
    doc = yaml.safe_load(path.read_text())
    out: List[BodySphere] = []
    for item in doc.get("body_sampling", {}).get("links", []):
        link = str(item["link_name"])
        frame = str(item.get("frame", link))
        for sample in item.get("samples", []):
            out.append(
                BodySphere(
                    link_name=link,
                    frame_name=frame,
                    center_link=np.asarray(sample["center"], dtype=float),
                    radius=float(sample["radius"]) + inflation_m,
                )
            )
    if not out:
        raise SystemExit("ERROR: no body spheres loaded")
    return out


def transformed(
    model, q: np.ndarray, spheres: Sequence[BodySphere]
) -> Tuple[List[BodySphere], np.ndarray]:
    data = model.createData()
    ss: List[BodySphere] = []
    cc: List[np.ndarray] = []
    for s, c in transform_body_spheres(model, data, q, spheres):
        ss.append(s)
        cc.append(np.asarray(c, dtype=float))
    return ss, np.vstack(cc)


def nonadjacent_self_clearance(
    spheres: Sequence[BodySphere],
    centers: np.ndarray,
) -> Tuple[float, Dict[str, object]]:
    """Approximate self clearance using non-adjacent body-sphere pairs.

    Same-link and directly adjacent-link pairs are intentionally ignored because
    their sphere covers overlap by construction around joints.
    """
    best = float("inf")
    info: Dict[str, object] = {}
    n = len(spheres)
    for i in range(n):
        ai = LINK_ORDER.get(spheres[i].link_name)
        if ai is None:
            continue
        for j in range(i + 1, n):
            aj = LINK_ORDER.get(spheres[j].link_name)
            if aj is None or abs(ai - aj) <= 1:
                continue
            d = float(np.linalg.norm(centers[i] - centers[j]))
            clearance = d - spheres[i].radius - spheres[j].radius
            if clearance < best:
                best = clearance
                info = {
                    "link_a": spheres[i].link_name,
                    "link_b": spheres[j].link_name,
                    "center_a": centers[i].tolist(),
                    "center_b": centers[j].tolist(),
                    "radius_a_m": float(spheres[i].radius),
                    "radius_b_m": float(spheres[j].radius),
                    "clearance_m": float(clearance),
                }
    return best, info


def body_inside_workspace(
    spheres: Sequence[BodySphere],
    centers: np.ndarray,
    bounds: Tuple[float, float, float, float, float, float],
    boundary_margin_m: float,
) -> bool:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    for s, c in zip(spheres, centers):
        r = float(s.radius) + boundary_margin_m
        if (
            c[0] - r < xmin or c[0] + r > xmax
            or c[1] - r < ymin or c[1] + r > ymax
            or c[2] - r < zmin or c[2] + r > zmax
        ):
            return False
    return True


def ee_position(model, frame_id: int, q: np.ndarray) -> np.ndarray:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.asarray(data.oMf[frame_id].translation, dtype=float).reshape(3)


def main() -> int:
    repo_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=repo_default)
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--body-samples", type=Path, default=None)
    ap.add_argument("--ee-frame", default="EE_link")

    ap.add_argument("--target-count", type=int, default=80)
    ap.add_argument("--max-samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260906)

    ap.add_argument("--joint-limit-margin-frac", type=float, default=0.05)
    ap.add_argument("--min-q-l2-separation", type=float, default=0.55)

    ap.add_argument("--risk-body-inflation", type=float, default=0.015)
    ap.add_argument("--min-obstacle-clearance", type=float, default=0.0)
    ap.add_argument("--startup-prior-inflation", type=float, default=0.0)
    ap.add_argument("--min-startup-prior-clearance", type=float, default=0.0)
    ap.add_argument("--min-self-clearance", type=float, default=0.0)

    # Match confidence_map_phase_e_ray.yaml, with a tiny internal margin.
    ap.add_argument(
        "--map-bounds",
        nargs=6,
        type=float,
        default=[-0.95, 0.95, -0.95, 0.95, 0.0, 1.15],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    ap.add_argument("--map-boundary-margin", type=float, default=0.005)

    # Broad useful EE region, deliberately much wider than the old goal sampler.
    ap.add_argument("--ee-x-range", nargs=2, type=float, default=[-0.55, 0.65])
    ap.add_argument("--ee-y-range", nargs=2, type=float, default=[-0.55, 0.55])
    ap.add_argument("--ee-z-range", nargs=2, type=float, default=[0.25, 1.00])

    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    world = (args.world or (
        repo / "src/arm_description/worlds/maixsense_obstacles.world"
    )).resolve()
    urdf = (args.urdf or (
        repo / "src/arm_description/urdf/Arm.urdf"
    )).resolve()
    body_path = (args.body_samples or (
        repo / "src/care_confidence_map/config/body_samples.yaml"
    )).resolve()
    out = args.output_json.resolve()

    for p in (world, urdf, body_path):
        if not p.is_file():
            raise SystemExit(f"ERROR: file not found: {p}")

    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 7 or model.nv != 7:
        raise SystemExit(
            f"ERROR: expected 7-DoF model, got nq={model.nq}, nv={model.nv}"
        )
    frame_id = model.getFrameId(args.ee_frame)
    if frame_id >= len(model.frames):
        raise SystemExit(f"ERROR: EE frame {args.ee_frame!r} not found")

    lower = np.asarray(model.lowerPositionLimit, dtype=float).reshape(-1)
    upper = np.asarray(model.upperPositionLimit, dtype=float).reshape(-1)
    span = upper - lower
    m = float(args.joint_limit_margin_frac)
    if not (0.0 <= m < 0.5):
        raise SystemExit("ERROR: --joint-limit-margin-frac must be in [0,0.5)")
    sample_lower = lower + m * span
    sample_upper = upper - m * span

    risk_spheres = load_body_spheres(
        body_path, body_inflation=float(args.risk_body_inflation))
    startup_spheres = load_all_body_spheres(
        body_path, inflation_m=float(args.startup_prior_inflation))
    raw_all_spheres = load_all_body_spheres(body_path, inflation_m=0.0)
    boxes = load_world_boxes(world)

    bounds = tuple(float(x) for x in args.map_bounds)
    ex0, ex1 = [float(x) for x in args.ee_x_range]
    ey0, ey1 = [float(x) for x in args.ee_y_range]
    ez0, ez1 = [float(x) for x in args.ee_z_range]

    rng = np.random.default_rng(args.seed)
    accepted: List[Dict[str, object]] = []
    accepted_q: List[np.ndarray] = []

    stats = {
        "sampled": 0,
        "rejected_obstacle": 0,
        "rejected_startup_prior": 0,
        "rejected_self_collision": 0,
        "rejected_map_bounds": 0,
        "rejected_ee_region": 0,
        "rejected_q_duplicate": 0,
        "accepted": 0,
    }

    for sample_idx in range(int(args.max_samples)):
        if len(accepted) >= int(args.target_count):
            break
        stats["sampled"] += 1
        q = rng.uniform(sample_lower, sample_upper)

        if any(
            float(np.linalg.norm(q - qprev)) < args.min_q_l2_separation
            for qprev in accepted_q
        ):
            stats["rejected_q_duplicate"] += 1
            continue

        obstacle_clear, obstacle_closest = terminal_clearance(
            model=model, q=q, spheres=risk_spheres, boxes=boxes)
        if obstacle_clear + 1e-12 < args.min_obstacle_clearance:
            stats["rejected_obstacle"] += 1
            continue

        startup_clear, startup_closest = terminal_clearance(
            model=model, q=q, spheres=startup_spheres, boxes=boxes)
        if startup_clear + 1e-12 < args.min_startup_prior_clearance:
            stats["rejected_startup_prior"] += 1
            continue

        raw_s, raw_centers = transformed(model, q, raw_all_spheres)
        self_clear, self_closest = nonadjacent_self_clearance(
            raw_s, raw_centers)
        if self_clear + 1e-12 < args.min_self_clearance:
            stats["rejected_self_collision"] += 1
            continue

        risk_s, risk_centers = transformed(model, q, risk_spheres)
        if not body_inside_workspace(
            risk_s,
            risk_centers,
            bounds,
            float(args.map_boundary_margin),
        ):
            stats["rejected_map_bounds"] += 1
            continue

        p_ee = ee_position(model, frame_id, q)
        if not (
            ex0 <= p_ee[0] <= ex1
            and ey0 <= p_ee[1] <= ey1
            and ez0 <= p_ee[2] <= ez1
        ):
            stats["rejected_ee_region"] += 1
            continue

        cid = f"phase_e_q0_{len(accepted):03d}"
        item = {
            "case_id": cid,
            # Compatibility with the cross-pair builder:
            "terminal_best_q": q.tolist(),
            "goal_position": p_ee.tolist(),
            "goal_orientation": [0.0, 0.0, 0.0, 1.0],
            "terminal_clearance_m": float(obstacle_clear),
            # q0-specific diagnostics:
            "q0": q.tolist(),
            "start_ee_position": p_ee.tolist(),
            "obstacle_clearance_m": float(obstacle_clear),
            "obstacle_closest_pair": obstacle_closest,
            "startup_prior_clearance_m": float(startup_clear),
            "startup_prior_closest_pair": startup_closest,
            "self_clearance_m": float(self_clear),
            "self_closest_pair": self_closest,
        }
        accepted.append(item)
        accepted_q.append(q.copy())
        stats["accepted"] = len(accepted)

        print(
            "[ACCEPT {:03d}/{:03d}] EE=[{:+.3f},{:+.3f},{:+.3f}] "
            "obs={:.3f}m startup={:.3f}m self={:.3f}m".format(
                len(accepted), int(args.target_count),
                p_ee[0], p_ee[1], p_ee[2],
                obstacle_clear, startup_clear, self_clear,
            )
        )

    complete = len(accepted) >= int(args.target_count)
    report = {
        "benchmark_name": "phase_e_random_initial_q_pool",
        "complete": bool(complete),
        "seed": int(args.seed),
        "semantics": (
            "direct joint-space q0 sampling; obstacle-world feasible; "
            "raw-body startup prior by default; broad workspace; "
            "non-adjacent sphere self-collision filter"
        ),
        "inputs": {
            "world": str(world),
            "urdf": str(urdf),
            "body_samples": str(body_path),
        },
        "sampling": {
            "target_count": int(args.target_count),
            "max_samples": int(args.max_samples),
            "joint_limit_margin_frac": float(args.joint_limit_margin_frac),
            "min_q_l2_separation": float(args.min_q_l2_separation),
            "risk_body_inflation_m": float(args.risk_body_inflation),
            "min_obstacle_clearance_m": float(args.min_obstacle_clearance),
            "startup_prior_inflation_m": float(args.startup_prior_inflation),
            "min_startup_prior_clearance_m": float(
                args.min_startup_prior_clearance),
            "min_self_clearance_m": float(args.min_self_clearance),
            "map_bounds": list(bounds),
            "map_boundary_margin_m": float(args.map_boundary_margin),
            "ee_x_range": [ex0, ex1],
            "ee_y_range": [ey0, ey1],
            "ee_z_range": [ez0, ez1],
        },
        "stats": stats,
        "selected_case_ids": [c["case_id"] for c in accepted],
        "cases": accepted,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("=" * 88)
    print("PHASE-E RANDOM INITIAL q0 POOL")
    print("=" * 88)
    print(json.dumps(stats, indent=2))
    print(f"complete : {complete}")
    print(f"output   : {out}")
    print("=" * 88)

    if not complete:
        raise SystemExit(
            f"ERROR: accepted only {len(accepted)} / {args.target_count} q0 "
            f"after {args.max_samples} samples"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
