#!/usr/bin/env python3
"""Build a 30-case Phase-E random-feasible-q0 cross-pair benchmark.

The source pool is the previously generated obstacle-aware goal pool. Each source
case provides:
  * target EE pose
  * terminal_best_q found in maixsense_obstacles.world
  * terminal clearance metadata

This script constructs a seeded random derangement:
  - every original goal is used exactly once as a target;
  - every terminal_best_q is used exactly once as an initial q0;
  - a case is never paired with its own terminal_best_q;
  - optionally require a minimum EE start/goal separation.

Before accepting a source q0, obstacle clearance is RECOMPUTED against the
current obstacle world using the same CAREPlanner body-sphere proxy.

This is a diagnostic benchmark generator, not a planner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pinocchio as pin

from test_phase_e_goal_terminal_feasibility import (
    load_body_spheres,
    load_world_boxes,
    terminal_clearance,
)


JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]


def get_cases(doc: Dict[str, object]) -> List[Dict[str, object]]:
    cases = doc.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit("ERROR: source pool has no non-empty 'cases' list")
    out = []
    for c in cases:
        if isinstance(c, dict):
            out.append(c)
    if not out:
        raise SystemExit("ERROR: source pool contains no valid case objects")
    return out


def as_q(c: Dict[str, object]) -> np.ndarray:
    if "terminal_best_q" not in c:
        raise SystemExit(
            f"ERROR: {c.get('case_id')} has no terminal_best_q; "
            "use the original obstacle-aware goal-pool JSON, not a stripped case file"
        )
    q = np.asarray(c["terminal_best_q"], dtype=float).reshape(-1)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise SystemExit(f"ERROR: invalid terminal_best_q in {c.get('case_id')}")
    return q


def as_pos(c: Dict[str, object]) -> np.ndarray:
    p = np.asarray(c["goal_position"], dtype=float).reshape(-1)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise SystemExit(f"ERROR: invalid goal_position in {c.get('case_id')}")
    return p


def find_derangement(
    rng: np.random.Generator,
    cases: Sequence[Dict[str, object]],
    min_distance_m: float,
    attempts: int,
) -> np.ndarray:
    n = len(cases)
    positions = [as_pos(c) for c in cases]
    base = np.arange(n)
    best = None
    best_min = -math.inf

    for _ in range(attempts):
        src_for_target = rng.permutation(n)
        if np.any(src_for_target == base):
            continue
        ds = np.asarray([
            np.linalg.norm(positions[int(src_for_target[j])] - positions[j])
            for j in range(n)
        ])
        dmin = float(np.min(ds))
        if dmin > best_min:
            best_min = dmin
            best = src_for_target.copy()
        if dmin + 1e-12 >= min_distance_m:
            return src_for_target

    if best is None:
        raise SystemExit("ERROR: failed to find any derangement")
    raise SystemExit(
        "ERROR: could not find a derangement satisfying "
        f"min_start_goal_ee_distance={min_distance_m:.3f} m after {attempts} attempts; "
        f"best minimum distance was {best_min:.3f} m. "
        "Lower --min-start-goal-ee-distance if this constraint is unnecessarily strict."
    )


def main() -> int:
    repo_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=repo_default)
    ap.add_argument("--source-pool", type=Path, required=True)
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--body-samples", type=Path, default=None)
    ap.add_argument("--body-inflation", type=float, default=0.015)
    ap.add_argument("--required-q0-clearance", type=float, default=0.06)
    ap.add_argument("--min-start-goal-ee-distance", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--derangement-attempts", type=int, default=200000)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    repo = args.repo.resolve()
    source_pool = args.source_pool.resolve()
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

    for p in (source_pool, world, urdf, body_samples_path):
        if not p.is_file():
            raise SystemExit(f"ERROR: file not found: {p}")

    doc = json.loads(source_pool.read_text())
    cases = get_cases(doc)
    if len(cases) != 30:
        raise SystemExit(
            f"ERROR: expected the 30-case source pool, found {len(cases)} cases"
        )

    ids = [str(c.get("case_id", "")) for c in cases]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        raise SystemExit("ERROR: source case IDs are missing or not unique")

    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 7 or model.nv != 7:
        raise SystemExit(f"ERROR: expected 7-DoF model, got nq={model.nq}, nv={model.nv}")
    lower = np.asarray(model.lowerPositionLimit, dtype=float).reshape(-1)
    upper = np.asarray(model.upperPositionLimit, dtype=float).reshape(-1)

    spheres = load_body_spheres(
        body_samples_path, body_inflation=float(args.body_inflation))
    boxes = load_world_boxes(world)

    q0_validation = []
    qs = []
    for c in cases:
        q = as_q(c)
        if np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
            raise SystemExit(f"ERROR: {c['case_id']} terminal_best_q violates joint limits")
        clearance, closest = terminal_clearance(
            model=model, q=q, spheres=spheres, boxes=boxes)
        if clearance + 1e-12 < args.required_q0_clearance:
            raise SystemExit(
                f"ERROR: {c['case_id']} q0 clearance={clearance:.6f} m "
                f"< required {args.required_q0_clearance:.6f} m"
            )
        qs.append(q)
        q0_validation.append({
            "source_case_id": c["case_id"],
            "q0": q.tolist(),
            "recomputed_obstacle_clearance_m": float(clearance),
            "closest_pair": closest,
            "stored_terminal_clearance_m": c.get("terminal_clearance_m"),
        })

    rng = np.random.default_rng(args.seed)
    src_for_target = find_derangement(
        rng=rng,
        cases=cases,
        min_distance_m=float(args.min_start_goal_ee_distance),
        attempts=int(args.derangement_attempts),
    )

    out_cases = []
    pair_rows = []
    for target_idx, target_case in enumerate(cases):
        src_idx = int(src_for_target[target_idx])
        src_case = cases[src_idx]
        start_p = as_pos(src_case)
        goal_p = as_pos(target_case)
        ee_dist = float(np.linalg.norm(start_p - goal_p))
        q_dist = float(np.linalg.norm(qs[src_idx] - as_q(target_case)))

        item = dict(target_case)
        # Keep the target case ID unchanged so results line up directly with the
        # historical zero-q0 qualification row for the same target goal.
        item["initial_q"] = qs[src_idx].tolist()
        item["initial_q_source_case_id"] = src_case["case_id"]
        item["initial_q_source_goal_position"] = start_p.tolist()
        item["initial_q_source_terminal_clearance_m"] = float(
            q0_validation[src_idx]["recomputed_obstacle_clearance_m"])
        item["cross_pair_start_goal_ee_distance_m"] = ee_dist
        item["cross_pair_q_to_target_terminal_q_l2_rad"] = q_dist
        out_cases.append(item)

        pair_rows.append({
            "target_case_id": target_case["case_id"],
            "initial_q_source_case_id": src_case["case_id"],
            "start_ee_position": start_p.tolist(),
            "goal_ee_position": goal_p.tolist(),
            "start_goal_ee_distance_m": ee_dist,
            "initial_q": qs[src_idx].tolist(),
            "source_q0_clearance_m": float(
                q0_validation[src_idx]["recomputed_obstacle_clearance_m"]),
        })

    used_sources = [r["initial_q_source_case_id"] for r in pair_rows]
    assert len(set(used_sources)) == 30
    assert all(
        r["target_case_id"] != r["initial_q_source_case_id"]
        for r in pair_rows
    )

    distances = [float(r["start_goal_ee_distance_m"]) for r in pair_rows]
    clearances = [float(r["source_q0_clearance_m"]) for r in pair_rows]

    report = {
        "benchmark_name": "phase_e_random_feasible_q0_cross_pairs",
        "semantics": (
            "same 30 obstacle-selected target goals; each target receives exactly "
            "one terminal_best_q from a different source goal via seeded random "
            "derangement; q0 obstacle clearance recomputed before use"
        ),
        "seed": int(args.seed),
        "joint_names": JOINT_NAMES,
        "source_pool": str(source_pool),
        "world": str(world),
        "urdf": str(urdf),
        "body_samples": str(body_samples_path),
        "q0_validation": {
            "body_inflation_m": float(args.body_inflation),
            "required_obstacle_clearance_m": float(args.required_q0_clearance),
            "note": (
                "This reproduces the existing CAREPlanner world-obstacle body-sphere "
                "feasibility model. It does not add a new self-collision model."
            ),
            "minimum_recomputed_clearance_m": min(clearances),
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
        },
        "selected_case_ids": [c["case_id"] for c in out_cases],
        "pairs": pair_rows,
        "source_q0_validation": q0_validation,
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
            f"q0_clearance={r['source_q0_clearance_m']:.3f} m"
        )
    print("-" * 88)
    print(f"minimum q0 obstacle clearance : {min(clearances):.3f} m")
    print(f"start-goal EE distance min    : {min(distances):.3f} m")
    print(f"start-goal EE distance median : {np.median(distances):.3f} m")
    print(f"start-goal EE distance max    : {max(distances):.3f} m")
    print(f"output                        : {out}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
