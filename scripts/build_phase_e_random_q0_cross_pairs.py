#!/usr/bin/env python3
"""Build a Phase-E random-feasible-q0 cross-pair benchmark.

Targets and initial configurations are intentionally separated:

  target pool:
    the original 30 obstacle-selected Phase-E goals.

  q0 source pool:
    a larger obstacle-selected goal pool (normally the existing 200-goal pool).
    Each candidate contributes its terminal_best_q.

A q0 is eligible only if it passes BOTH:
  1) current CAREPlanner obstacle-body feasibility
       risk body sphere + body_inflation, clearance >= required_q0_clearance;
  2) the CURRENT Phase-E startup trusted-free assumption
       every startup-prior body sphere (raw radius + startup_prior_inflation)
       is non-overlapping with every static obstacle.

The second check is essential in an obstacle world. Otherwise the one-shot
startup body prior would mark a real obstacle region as trusted free before
planning even starts.

The builder then chooses a seeded, one-to-one assignment of distinct q0
candidates to all 30 targets, with:
  * no target paired with a source case of the same ID when IDs overlap;
  * a minimum FK start-EE to target-EE distance.

This script is offline and does not run ROS/Gazebo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin
import yaml

from test_phase_e_goal_terminal_feasibility import (
    BodySphere,
    load_body_spheres,
    load_world_boxes,
    terminal_clearance,
)


JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]


def case_list(doc: Dict[str, object], label: str) -> List[Dict[str, object]]:
    cases = doc.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"ERROR: {label} has no non-empty 'cases' list")
    out = [c for c in cases if isinstance(c, dict)]
    if len(out) != len(cases):
        raise SystemExit(f"ERROR: {label} contains malformed case entries")
    return out


def as_q(c: Dict[str, object]) -> np.ndarray:
    if "terminal_best_q" not in c:
        raise SystemExit(
            f"ERROR: {c.get('case_id')} has no terminal_best_q; "
            "use an obstacle-aware goal-pool JSON"
        )
    q = np.asarray(c["terminal_best_q"], dtype=float).reshape(-1)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise SystemExit(f"ERROR: invalid terminal_best_q in {c.get('case_id')}")
    return q


def as_goal_pos(c: Dict[str, object]) -> np.ndarray:
    p = np.asarray(c["goal_position"], dtype=float).reshape(-1)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise SystemExit(f"ERROR: invalid goal_position in {c.get('case_id')}")
    return p


def load_startup_prior_spheres(
    path: Path,
    inflation_m: float,
    risk_samples_only: bool,
) -> List[BodySphere]:
    """Mirror confidence_map current_body_prior sphere selection.

    Current Phase-E config uses risk_samples_only=false, therefore all configured
    body samples, including links excluded from trajectory risk, participate in
    the one-shot trusted-free startup envelope.
    """
    doc = yaml.safe_load(path.read_text())
    links = doc.get("body_sampling", {}).get("links", [])
    spheres: List[BodySphere] = []
    for item in links:
        include_for_risk = bool(item.get("include_for_risk", False))
        if risk_samples_only and not include_for_risk:
            continue
        link_name = str(item["link_name"])
        frame_name = str(item.get("frame", link_name))
        for sample in item.get("samples", []):
            spheres.append(
                BodySphere(
                    link_name=link_name,
                    frame_name=frame_name,
                    center_link=np.asarray(sample["center"], dtype=float),
                    radius=float(sample["radius"]) + inflation_m,
                )
            )
    if not spheres:
        raise SystemExit("ERROR: no startup-prior body spheres loaded")
    return spheres


def ee_position(model, frame_id: int, q: np.ndarray) -> np.ndarray:
    data = model.createData()
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.asarray(data.oMf[frame_id].translation, dtype=float).reshape(3)


def find_assignment(
    rng: np.random.Generator,
    targets: Sequence[Dict[str, object]],
    eligible: Sequence[Dict[str, object]],
    min_distance_m: float,
    attempts: int,
) -> List[int]:
    """Return eligible-index assignment, one distinct q0 per target."""
    n = len(targets)
    if len(eligible) < n:
        raise SystemExit(
            f"ERROR: only {len(eligible)} eligible q0 candidates for {n} targets"
        )

    target_pos = [as_goal_pos(c) for c in targets]
    best: Optional[List[int]] = None
    best_min = -math.inf

    eligible_indices = np.arange(len(eligible))
    for _ in range(attempts):
        chosen = rng.choice(eligible_indices, size=n, replace=False)
        rng.shuffle(chosen)

        ok = True
        dmin = math.inf
        for j, ei_raw in enumerate(chosen):
            ei = int(ei_raw)
            src = eligible[ei]
            target_id = str(targets[j].get("case_id", ""))
            source_id = str(src["source_case_id"])
            if source_id == target_id:
                ok = False
                break
            d = float(np.linalg.norm(
                np.asarray(src["start_ee_position"], dtype=float) - target_pos[j]
            ))
            dmin = min(dmin, d)
            if d + 1e-12 < min_distance_m:
                ok = False
                break

        if dmin > best_min:
            best_min = dmin
            best = [int(x) for x in chosen]
        if ok:
            return [int(x) for x in chosen]

    raise SystemExit(
        "ERROR: failed to find one-to-one q0 assignment satisfying constraints "
        f"after {attempts} attempts; best observed minimum EE separation="
        f"{best_min:.3f} m"
    )


def main() -> int:
    repo_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=repo_default)
    ap.add_argument("--target-pool", type=Path, required=True)
    ap.add_argument("--q0-source-pool", type=Path, required=True)
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--body-samples", type=Path, default=None)
    ap.add_argument("--ee-frame", default="EE_link")

    ap.add_argument("--body-inflation", type=float, default=0.015)
    ap.add_argument("--required-q0-clearance", type=float, default=0.06)

    ap.add_argument("--startup-prior-inflation", type=float, default=0.10)
    ap.add_argument(
        "--startup-prior-risk-samples-only",
        action="store_true",
        help=(
            "Use only risk samples for startup prior. Current Phase-E formal "
            "config leaves this false, so normally do NOT pass this flag."
        ),
    )
    ap.add_argument(
        "--required-startup-prior-clearance",
        type=float,
        default=0.0,
        help="Minimum obstacle clearance of the inflated startup-prior envelope.",
    )

    ap.add_argument("--min-start-goal-ee-distance", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--assignment-attempts", type=int, default=200000)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    target_pool = args.target_pool.resolve()
    q0_pool = args.q0_source_pool.resolve()
    world = (args.world or (
        repo / "src/arm_description/worlds/maixsense_obstacles.world"
    )).resolve()
    urdf = (args.urdf or (
        repo / "src/arm_description/urdf/Arm.urdf"
    )).resolve()
    body_samples_path = (args.body_samples or (
        repo / "src/care_confidence_map/config/body_samples.yaml"
    )).resolve()
    out = args.output_json.resolve()

    for p in (target_pool, q0_pool, world, urdf, body_samples_path):
        if not p.is_file():
            raise SystemExit(f"ERROR: file not found: {p}")

    target_doc = json.loads(target_pool.read_text())
    q0_doc = json.loads(q0_pool.read_text())
    targets = case_list(target_doc, "target pool")
    q0_cases = case_list(q0_doc, "q0 source pool")

    if len(targets) != 30:
        raise SystemExit(
            f"ERROR: target pool must contain exactly 30 cases, found {len(targets)}"
        )

    target_ids = [str(c.get("case_id", "")) for c in targets]
    if len(set(target_ids)) != 30 or any(not x for x in target_ids):
        raise SystemExit("ERROR: target case IDs are missing or non-unique")

    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 7 or model.nv != 7:
        raise SystemExit(f"ERROR: expected 7-DoF model, got nq={model.nq}, nv={model.nv}")
    lower = np.asarray(model.lowerPositionLimit, dtype=float).reshape(-1)
    upper = np.asarray(model.upperPositionLimit, dtype=float).reshape(-1)
    frame_id = model.getFrameId(args.ee_frame)
    if frame_id >= len(model.frames):
        raise SystemExit(f"ERROR: EE frame {args.ee_frame!r} not found")

    risk_spheres = load_body_spheres(
        body_samples_path, body_inflation=float(args.body_inflation))
    startup_spheres = load_startup_prior_spheres(
        body_samples_path,
        inflation_m=float(args.startup_prior_inflation),
        risk_samples_only=bool(args.startup_prior_risk_samples_only),
    )
    boxes = load_world_boxes(world)

    eligible: List[Dict[str, object]] = []
    rejected_joint_limit = 0
    rejected_collision_clearance = 0
    rejected_startup_prior_overlap = 0

    candidate_reports: List[Dict[str, object]] = []
    seen_q = []

    for c in q0_cases:
        q = as_q(c)
        cid = str(c.get("case_id", ""))

        if np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
            rejected_joint_limit += 1
            candidate_reports.append({
                "source_case_id": cid,
                "eligible": False,
                "reason": "joint_limit",
            })
            continue

        # De-duplicate effectively identical q0s if a larger pool contains
        # repeated IK solutions.
        if any(float(np.linalg.norm(q - qprev)) < 1e-6 for qprev in seen_q):
            candidate_reports.append({
                "source_case_id": cid,
                "eligible": False,
                "reason": "duplicate_q",
            })
            continue
        seen_q.append(q.copy())

        collision_clearance, collision_closest = terminal_clearance(
            model=model, q=q, spheres=risk_spheres, boxes=boxes)
        startup_clearance, startup_closest = terminal_clearance(
            model=model, q=q, spheres=startup_spheres, boxes=boxes)

        if collision_clearance + 1e-12 < args.required_q0_clearance:
            rejected_collision_clearance += 1
            candidate_reports.append({
                "source_case_id": cid,
                "eligible": False,
                "reason": "collision_clearance",
                "collision_clearance_m": float(collision_clearance),
                "startup_prior_clearance_m": float(startup_clearance),
            })
            continue

        if startup_clearance + 1e-12 < args.required_startup_prior_clearance:
            rejected_startup_prior_overlap += 1
            candidate_reports.append({
                "source_case_id": cid,
                "eligible": False,
                "reason": "startup_prior_overlap",
                "collision_clearance_m": float(collision_clearance),
                "startup_prior_clearance_m": float(startup_clearance),
                "startup_prior_closest_pair": startup_closest,
            })
            continue

        start_ee = ee_position(model, frame_id, q)
        item = {
            "source_case_id": cid,
            "q0": q.tolist(),
            "start_ee_position": start_ee.tolist(),
            "collision_clearance_m": float(collision_clearance),
            "collision_closest_pair": collision_closest,
            "startup_prior_clearance_m": float(startup_clearance),
            "startup_prior_closest_pair": startup_closest,
            "stored_terminal_clearance_m": c.get("terminal_clearance_m"),
        }
        eligible.append(item)
        candidate_reports.append({"eligible": True, **item})

    print("=" * 88)
    print("PHASE-E q0 ELIGIBILITY")
    print("=" * 88)
    print(f"q0 source pool size               : {len(q0_cases)}")
    print(f"eligible                           : {len(eligible)}")
    print(f"rejected joint limit               : {rejected_joint_limit}")
    print(f"rejected collision clearance       : {rejected_collision_clearance}")
    print(f"rejected startup-prior overlap     : {rejected_startup_prior_overlap}")
    if eligible:
        print(
            "eligible startup clearance min/max : "
            f"{min(float(x['startup_prior_clearance_m']) for x in eligible):.4f} / "
            f"{max(float(x['startup_prior_clearance_m']) for x in eligible):.4f} m"
        )
    print("=" * 88)

    if len(eligible) < 30:
        # Still write a diagnostic report before failing so BUILD_ONLY can tell
        # us exactly why the source pool is insufficient.
        report = {
            "benchmark_name": "phase_e_random_feasible_q0_cross_pairs",
            "complete": False,
            "reason": "insufficient_eligible_q0",
            "target_pool": str(target_pool),
            "q0_source_pool": str(q0_pool),
            "eligibility": {
                "source_count": len(q0_cases),
                "eligible_count": len(eligible),
                "rejected_joint_limit": rejected_joint_limit,
                "rejected_collision_clearance": rejected_collision_clearance,
                "rejected_startup_prior_overlap": rejected_startup_prior_overlap,
            },
            "q0_requirements": {
                "body_inflation_m": float(args.body_inflation),
                "required_collision_clearance_m": float(args.required_q0_clearance),
                "startup_prior_inflation_m": float(args.startup_prior_inflation),
                "startup_prior_risk_samples_only": bool(
                    args.startup_prior_risk_samples_only),
                "required_startup_prior_clearance_m": float(
                    args.required_startup_prior_clearance),
            },
            "candidate_reports": candidate_reports,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        raise SystemExit(
            f"ERROR: only {len(eligible)} q0 candidates satisfy current obstacle "
            "and startup-prior semantics; need at least 30"
        )

    rng = np.random.default_rng(args.seed)
    assignment = find_assignment(
        rng=rng,
        targets=targets,
        eligible=eligible,
        min_distance_m=float(args.min_start_goal_ee_distance),
        attempts=int(args.assignment_attempts),
    )

    out_cases = []
    pair_rows = []
    for target_idx, target_case in enumerate(targets):
        src = eligible[assignment[target_idx]]
        goal_p = as_goal_pos(target_case)
        start_p = np.asarray(src["start_ee_position"], dtype=float)
        ee_dist = float(np.linalg.norm(start_p - goal_p))

        item = dict(target_case)
        item["initial_q"] = list(src["q0"])
        item["initial_q_source_case_id"] = src["source_case_id"]
        item["initial_q_source_start_ee_position"] = list(src["start_ee_position"])
        item["initial_q_source_collision_clearance_m"] = float(
            src["collision_clearance_m"])
        item["initial_q_source_startup_prior_clearance_m"] = float(
            src["startup_prior_clearance_m"])
        item["cross_pair_start_goal_ee_distance_m"] = ee_dist
        out_cases.append(item)

        pair_rows.append({
            "target_case_id": target_case["case_id"],
            "initial_q_source_case_id": src["source_case_id"],
            "start_ee_position": list(src["start_ee_position"]),
            "goal_ee_position": goal_p.tolist(),
            "start_goal_ee_distance_m": ee_dist,
            "initial_q": list(src["q0"]),
            "q0_collision_clearance_m": float(src["collision_clearance_m"]),
            "q0_startup_prior_clearance_m": float(
                src["startup_prior_clearance_m"]),
        })

    used_sources = [r["initial_q_source_case_id"] for r in pair_rows]
    if len(set(used_sources)) != 30:
        raise RuntimeError("internal error: q0 sources are not one-to-one")

    distances = [float(r["start_goal_ee_distance_m"]) for r in pair_rows]
    collision_clearances = [
        float(r["q0_collision_clearance_m"]) for r in pair_rows]
    startup_clearances = [
        float(r["q0_startup_prior_clearance_m"]) for r in pair_rows]

    report = {
        "benchmark_name": "phase_e_random_feasible_q0_cross_pairs",
        "complete": True,
        "semantics": (
            "same 30 target goals; q0 sampled without replacement from a larger "
            "obstacle-selected terminal configuration pool; every q0 revalidated "
            "against both collision-body and current +10cm startup trusted-free "
            "envelope semantics"
        ),
        "seed": int(args.seed),
        "joint_names": JOINT_NAMES,
        "target_pool": str(target_pool),
        "q0_source_pool": str(q0_pool),
        "world": str(world),
        "urdf": str(urdf),
        "body_samples": str(body_samples_path),
        "eligibility": {
            "source_count": len(q0_cases),
            "eligible_count": len(eligible),
            "rejected_joint_limit": rejected_joint_limit,
            "rejected_collision_clearance": rejected_collision_clearance,
            "rejected_startup_prior_overlap": rejected_startup_prior_overlap,
        },
        "q0_requirements": {
            "body_inflation_m": float(args.body_inflation),
            "required_collision_clearance_m": float(args.required_q0_clearance),
            "startup_prior_inflation_m": float(args.startup_prior_inflation),
            "startup_prior_risk_samples_only": bool(
                args.startup_prior_risk_samples_only),
            "required_startup_prior_clearance_m": float(
                args.required_startup_prior_clearance),
        },
        "pairing": {
            "pair_count": 30,
            "one_to_one": True,
            "self_pairs_allowed": False,
            "minimum_required_start_goal_ee_distance_m": float(
                args.min_start_goal_ee_distance),
            "minimum_actual_start_goal_ee_distance_m": min(distances),
            "median_start_goal_ee_distance_m": float(np.median(distances)),
            "maximum_start_goal_ee_distance_m": max(distances),
            "minimum_q0_collision_clearance_m": min(collision_clearances),
            "minimum_q0_startup_prior_clearance_m": min(startup_clearances),
        },
        "selected_case_ids": [c["case_id"] for c in out_cases],
        "pairs": pair_rows,
        "eligible_q0_candidates": eligible,
        "candidate_reports": candidate_reports,
        "cases": out_cases,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("=" * 88)
    print("PHASE-E RANDOM FEASIBLE q0 CROSS-PAIR BENCHMARK")
    print("=" * 88)
    for r in pair_rows:
        print(
            f"{r['target_case_id']}  q0<-{r['initial_q_source_case_id']}  "
            f"EE_dist={r['start_goal_ee_distance_m']:.3f} m  "
            f"collision_clear={r['q0_collision_clearance_m']:.3f} m  "
            f"startup_clear={r['q0_startup_prior_clearance_m']:.3f} m"
        )
    print("-" * 88)
    print(f"eligible q0 count             : {len(eligible)} / {len(q0_cases)}")
    print(f"start-goal EE distance min    : {min(distances):.3f} m")
    print(f"start-goal EE distance median : {np.median(distances):.3f} m")
    print(f"start-goal EE distance max    : {max(distances):.3f} m")
    print(f"minimum collision clearance   : {min(collision_clearances):.3f} m")
    print(f"minimum startup prior clearance: {min(startup_clearances):.3f} m")
    print(f"output                        : {out}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
