#!/usr/bin/env python3
"""
Generate an obstacle-aware pool of terminal-feasible Phase-E EE goals.

This is Stage 1 only:
  random EE position
    -> multi-start IK at fixed benchmark orientation
    -> terminal collision check against static obstacle boxes
    -> require conservative body clearance
    -> spatial de-duplication
    -> save accepted goal pool

It intentionally does NOT classify easy/medium/hard and does NOT prove
start-to-goal path feasibility. Those are later stages.

The script reuses the exact IK/world/body-sphere utilities from
test_phase_e_goal_terminal_feasibility.py so Test 1 and sampling cannot silently
drift apart.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

# scripts/ is on sys.path when this file is executed directly.
from test_phase_e_goal_terminal_feasibility import (
    load_body_spheres,
    load_world_boxes,
    quaternion_xyzw_to_rotation,
    signed_distance_point_to_box,
    solve_ik_dls,
    terminal_clearance,
)


def parse_bounds(values: Sequence[float], name: str) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} requires exactly two values")
    lo, hi = float(values[0]), float(values[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
        raise ValueError(f"Invalid {name}: [{lo}, {hi}]")
    return lo, hi


def goal_is_duplicate(
    p: np.ndarray,
    accepted: Sequence[Dict[str, object]],
    min_separation_m: float,
) -> bool:
    for item in accepted:
        q = np.asarray(item["goal_position"], dtype=float)
        if float(np.linalg.norm(p - q)) < min_separation_m:
            return True
    return False


def min_goal_origin_box_clearance(
    p: np.ndarray,
    boxes,
) -> Tuple[float, Optional[str]]:
    best = float("inf")
    best_name: Optional[str] = None
    for box in boxes:
        d = float(signed_distance_point_to_box(p, box))
        if d < best:
            best = d
            best_name = box.name
    return best, best_name


def build_seed_list(
    model,
    initial_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    random_starts: int,
    warm_q: Sequence[np.ndarray],
    warm_limit: int,
) -> List[Tuple[str, np.ndarray]]:
    seeds: List[Tuple[str, np.ndarray]] = []
    seeds.append(("initial_q", initial_q.copy()))
    seeds.append(("neutral", np.asarray(pin.neutral(model), dtype=float).reshape(-1)))

    # Recent accepted solutions are useful warm starts for nearby sampled goals,
    # but cap them so the sample distribution is still broad.
    for i, q in enumerate(list(warm_q)[-warm_limit:]):
        seeds.append((f"warm_{i:02d}", np.asarray(q, dtype=float).copy()))

    for i in range(random_starts):
        seeds.append((f"random_{i:03d}", rng.uniform(lower, upper)))
    return seeds


def find_terminal_feasible_ik(
    model,
    frame_id: int,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    initial_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spheres,
    boxes,
    rng: np.random.Generator,
    random_starts: int,
    warm_q: Sequence[np.ndarray],
    warm_limit: int,
    min_terminal_clearance_m: float,
    pos_tol: float,
    ori_tol: float,
    max_iters: int,
    damping: float,
    max_step_inf: float,
) -> Tuple[Optional[Dict[str, object]], Dict[str, int]]:
    counts = {
        "seed_attempts": 0,
        "ik_converged": 0,
        "ik_colliding": 0,
        "ik_below_clearance": 0,
    }

    seeds = build_seed_list(
        model=model,
        initial_q=initial_q,
        lower=lower,
        upper=upper,
        rng=rng,
        random_starts=random_starts,
        warm_q=warm_q,
        warm_limit=warm_limit,
    )

    best: Optional[Dict[str, object]] = None
    best_clearance = -float("inf")
    seen_q: List[np.ndarray] = []

    for seed_name, q0 in seeds:
        counts["seed_attempts"] += 1
        q_sol, ok, iters, pos_err, ori_err = solve_ik_dls(
            model=model,
            frame_id=frame_id,
            target_position=target_position,
            target_rotation=target_rotation,
            q_seed=q0,
            lower=lower,
            upper=upper,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            max_iters=max_iters,
            damping=damping,
            max_step_inf=max_step_inf,
        )
        if not ok:
            continue

        counts["ik_converged"] += 1

        # Avoid evaluating effectively identical solutions repeatedly.
        if any(float(np.linalg.norm(q_sol - q_prev)) < 1e-3 for q_prev in seen_q):
            continue
        seen_q.append(q_sol.copy())

        clearance, closest = terminal_clearance(
            model=model,
            q=q_sol,
            spheres=spheres,
            boxes=boxes,
        )

        if clearance < 0.0:
            counts["ik_colliding"] += 1
        elif clearance < min_terminal_clearance_m:
            counts["ik_below_clearance"] += 1

        if clearance > best_clearance:
            best_clearance = float(clearance)
            best = {
                "seed": seed_name,
                "q": q_sol.tolist(),
                "position_error_m": float(pos_err),
                "orientation_error_rad": float(ori_err),
                "iterations": int(iters),
                "terminal_clearance_m": float(clearance),
                "closest_pair": closest,
            }

        # Stage-1 generation only needs one certified terminal configuration.
        if clearance >= min_terminal_clearance_m:
            return best, counts

    return best, counts


def main() -> int:
    repo_default = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(
        description="Generate obstacle-aware terminal-feasible Phase-E EE goals."
    )
    ap.add_argument("--repo", type=Path, default=repo_default)
    ap.add_argument("--template-case", default="case_014")
    ap.add_argument("--cases-json", type=Path, default=None)
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--body-samples", type=Path, default=None)
    ap.add_argument("--world", type=Path, default=None)
    ap.add_argument("--ee-frame", default="EE_link")

    ap.add_argument("--target-count", type=int, default=100)
    ap.add_argument("--max-position-samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260904)

    # Deliberately centered on the useful front workspace rather than the entire
    # theoretical arm workspace. These can be widened from the CLI.
    ap.add_argument("--x-range", type=float, nargs=2, default=[0.18, 0.55])
    ap.add_argument("--y-range", type=float, nargs=2, default=[-0.35, 0.35])
    ap.add_argument("--z-range", type=float, nargs=2, default=[0.50, 0.85])

    ap.add_argument(
        "--min-ee-origin-clearance",
        type=float,
        default=0.05,
        help=(
            "Cheap pre-IK filter: EE goal origin must be at least this far "
            "from every static box volume, meters."
        ),
    )
    ap.add_argument(
        "--body-inflation",
        type=float,
        default=0.015,
        help="Meters added to every CAREPlanner risk-body sphere.",
    )
    ap.add_argument(
        "--min-terminal-clearance",
        type=float,
        default=0.06,
        help=(
            "Required whole-body terminal clearance after body inflation, "
            "meters. Default 6 cm."
        ),
    )
    ap.add_argument(
        "--min-goal-separation",
        type=float,
        default=0.07,
        help="Minimum Euclidean separation between accepted EE goals, meters.",
    )

    ap.add_argument("--ik-random-starts", type=int, default=48)
    ap.add_argument("--ik-warm-start-limit", type=int, default=4)
    ap.add_argument("--ik-max-iters", type=int, default=350)
    ap.add_argument("--pos-tol", type=float, default=1e-3)
    ap.add_argument("--ori-tol", type=float, default=1e-2)
    ap.add_argument("--damping", type=float, default=2e-3)
    ap.add_argument("--max-step-inf", type=float, default=0.20)

    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=25,
    )
    args = ap.parse_args()

    if args.target_count <= 0:
        raise SystemExit("ERROR: --target-count must be positive")
    if args.max_position_samples < args.target_count:
        raise SystemExit(
            "ERROR: --max-position-samples must be >= --target-count")
    if args.ik_random_starts < 0:
        raise SystemExit("ERROR: --ik-random-starts must be nonnegative")
    if args.body_inflation < 0.0:
        raise SystemExit("ERROR: --body-inflation must be nonnegative")
    if args.min_terminal_clearance < 0.0:
        raise SystemExit("ERROR: --min-terminal-clearance must be nonnegative")
    if args.min_goal_separation < 0.0:
        raise SystemExit("ERROR: --min-goal-separation must be nonnegative")

    x_lo, x_hi = parse_bounds(args.x_range, "x-range")
    y_lo, y_hi = parse_bounds(args.y_range, "y-range")
    z_lo, z_hi = parse_bounds(args.z_range, "z-range")

    repo = args.repo.resolve()
    cases_json = (args.cases_json or (
        repo / "src/egocentric_arm_planner/config/phase_c2_vbc_cases.json"
    )).resolve()
    urdf = (args.urdf or (
        repo / "src/arm_description/urdf/Arm.urdf"
    )).resolve()
    body_samples_path = (args.body_samples or (
        repo / "src/care_confidence_map/config/body_samples.yaml"
    )).resolve()
    world = (args.world or (
        repo / "src/arm_description/worlds/maixsense_obstacles.world"
    )).resolve()
    output_json = (args.output_json or (
        repo / "outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool.json"
    )).resolve()

    for p in (cases_json, urdf, body_samples_path, world):
        if not p.is_file():
            raise SystemExit(f"ERROR: file not found: {p}")

    with cases_json.open("r") as f:
        cases_doc = json.load(f)
    template = next(
        (c for c in cases_doc.get("cases", [])
         if c.get("case_id") == args.template_case),
        None,
    )
    if template is None:
        raise SystemExit(
            f"ERROR: template case {args.template_case!r} not found")

    goal_orientation = list(template["goal_orientation"])
    target_rotation = quaternion_xyzw_to_rotation(goal_orientation)
    initial_q = np.asarray(template["initial_q"], dtype=float).reshape(-1)

    model = pin.buildModelFromUrdf(str(urdf))
    if model.nq != 7 or model.nv != 7:
        raise SystemExit(
            f"ERROR: expected 7-DoF model, got nq={model.nq}, nv={model.nv}")
    if initial_q.size != model.nq:
        raise SystemExit(
            f"ERROR: template initial_q has {initial_q.size} values, "
            f"model expects {model.nq}")

    frame_id = model.getFrameId(args.ee_frame)
    if frame_id >= len(model.frames):
        raise SystemExit(
            f"ERROR: EE frame {args.ee_frame!r} not found in {urdf}")

    lower = np.asarray(model.lowerPositionLimit, dtype=float).reshape(-1)
    upper = np.asarray(model.upperPositionLimit, dtype=float).reshape(-1)
    if not (
        np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(upper > lower)
    ):
        raise SystemExit("ERROR: invalid/non-finite joint limits")

    spheres = load_body_spheres(
        body_samples_path, body_inflation=args.body_inflation)
    boxes = load_world_boxes(world)

    rng = np.random.default_rng(args.seed)
    accepted: List[Dict[str, object]] = []
    warm_q: List[np.ndarray] = []

    stats: Dict[str, int] = {
        "position_samples": 0,
        "rejected_ee_origin_clearance": 0,
        "rejected_duplicate_goal": 0,
        "rejected_no_ik": 0,
        "rejected_all_ik_collide_or_low_clearance": 0,
        "accepted": 0,
        "total_ik_seed_attempts": 0,
        "total_ik_converged": 0,
    }

    print("=" * 78)
    print("PHASE E OBSTACLE-AWARE GOAL SAMPLER — STAGE 1")
    print("=" * 78)
    print(f"target_count           : {args.target_count}")
    print(f"position bounds        : x=[{x_lo:.3f},{x_hi:.3f}] "
          f"y=[{y_lo:.3f},{y_hi:.3f}] z=[{z_lo:.3f},{z_hi:.3f}]")
    print(f"orientation xyzw       : {goal_orientation}")
    print(f"body inflation         : {args.body_inflation:.3f} m")
    print(f"terminal clearance min : {args.min_terminal_clearance:.3f} m")
    print(f"goal separation min    : {args.min_goal_separation:.3f} m")
    print(f"IK random starts       : {args.ik_random_starts}")
    print("=" * 78)

    for sample_idx in range(args.max_position_samples):
        if len(accepted) >= args.target_count:
            break

        stats["position_samples"] += 1

        p = np.array([
            rng.uniform(x_lo, x_hi),
            rng.uniform(y_lo, y_hi),
            rng.uniform(z_lo, z_hi),
        ], dtype=float)

        ee_origin_clearance, ee_origin_nearest = (
            min_goal_origin_box_clearance(p, boxes)
        )
        if ee_origin_clearance < args.min_ee_origin_clearance:
            stats["rejected_ee_origin_clearance"] += 1
            continue

        if goal_is_duplicate(p, accepted, args.min_goal_separation):
            stats["rejected_duplicate_goal"] += 1
            continue

        best, ik_counts = find_terminal_feasible_ik(
            model=model,
            frame_id=frame_id,
            target_position=p,
            target_rotation=target_rotation,
            initial_q=initial_q,
            lower=lower,
            upper=upper,
            spheres=spheres,
            boxes=boxes,
            rng=rng,
            random_starts=args.ik_random_starts,
            warm_q=warm_q,
            warm_limit=args.ik_warm_start_limit,
            min_terminal_clearance_m=args.min_terminal_clearance,
            pos_tol=args.pos_tol,
            ori_tol=args.ori_tol,
            max_iters=args.ik_max_iters,
            damping=args.damping,
            max_step_inf=args.max_step_inf,
        )

        stats["total_ik_seed_attempts"] += ik_counts["seed_attempts"]
        stats["total_ik_converged"] += ik_counts["ik_converged"]

        if best is None:
            stats["rejected_no_ik"] += 1
            continue

        if float(best["terminal_clearance_m"]) < args.min_terminal_clearance:
            stats["rejected_all_ik_collide_or_low_clearance"] += 1
            continue

        case_id = f"phase_e_goal_{len(accepted):03d}"
        item = {
            "case_id": case_id,
            "difficulty_bin": "unassigned",
            "goal_position": p.tolist(),
            "goal_orientation": goal_orientation,
            "initial_q": initial_q.tolist(),
            "terminal_best_q": best["q"],
            "terminal_clearance_m": float(best["terminal_clearance_m"]),
            "terminal_closest_pair": best["closest_pair"],
            "ee_origin_clearance_m": float(ee_origin_clearance),
            "ee_origin_nearest_obstacle": ee_origin_nearest,
            "ik_seed": best["seed"],
            "ik_position_error_m": float(best["position_error_m"]),
            "ik_orientation_error_rad": float(best["orientation_error_rad"]),
        }
        accepted.append(item)
        warm_q.append(np.asarray(best["q"], dtype=float))
        stats["accepted"] = len(accepted)

        print(
            "[ACCEPT {:03d}/{:03d}] p=[{:+.3f},{:+.3f},{:+.3f}] "
            "terminal_clearance={:.3f}m closest={}".format(
                len(accepted),
                args.target_count,
                p[0], p[1], p[2],
                float(best["terminal_clearance_m"]),
                best["closest_pair"].get("obstacle", "none")
                if best.get("closest_pair") else "none",
            )
        )

        if (
            args.progress_every > 0
            and stats["position_samples"] % args.progress_every == 0
        ):
            print(
                "[PROGRESS] sampled={} accepted={} no_ik={} "
                "low/collide={} point_filter={} dup={}".format(
                    stats["position_samples"],
                    stats["accepted"],
                    stats["rejected_no_ik"],
                    stats["rejected_all_ik_collide_or_low_clearance"],
                    stats["rejected_ee_origin_clearance"],
                    stats["rejected_duplicate_goal"],
                )
            )

    success = len(accepted) >= args.target_count

    clearances = [
        float(c["terminal_clearance_m"]) for c in accepted
    ]
    summary = {
        "num_accepted": len(accepted),
        "requested_count": int(args.target_count),
        "complete": bool(success),
        "terminal_clearance_min_m": (
            min(clearances) if clearances else None),
        "terminal_clearance_median_m": (
            float(np.median(clearances)) if clearances else None),
        "terminal_clearance_max_m": (
            max(clearances) if clearances else None),
    }

    report = {
        "benchmark_name": "phase_e_obstacle_goal_pool_stage1",
        "generator": "scripts/generate_phase_e_obstacle_goals.py",
        "seed": int(args.seed),
        "stage": 1,
        "semantics": (
            "terminal-feasible only; path feasibility and difficulty "
            "classification not yet evaluated"
        ),
        "template_case": args.template_case,
        "goal_orientation": goal_orientation,
        "initial_q": initial_q.tolist(),
        "sampling": {
            "x_range": [x_lo, x_hi],
            "y_range": [y_lo, y_hi],
            "z_range": [z_lo, z_hi],
            "max_position_samples": int(args.max_position_samples),
            "min_ee_origin_clearance_m":
                float(args.min_ee_origin_clearance),
            "body_inflation_m": float(args.body_inflation),
            "min_terminal_clearance_m":
                float(args.min_terminal_clearance),
            "min_goal_separation_m": float(args.min_goal_separation),
            "ik_random_starts": int(args.ik_random_starts),
            "ik_warm_start_limit": int(args.ik_warm_start_limit),
            "ik_max_iters": int(args.ik_max_iters),
            "pos_tol_m": float(args.pos_tol),
            "ori_tol_rad": float(args.ori_tol),
        },
        "inputs": {
            "urdf": str(urdf),
            "body_samples": str(body_samples_path),
            "world": str(world),
            "cases_json": str(cases_json),
        },
        "stats": stats,
        "summary": summary,
        "selected_case_ids": [c["case_id"] for c in accepted],
        "cases": accepted,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(report, f, indent=2)

    print("=" * 78)
    print("SAMPLING COMPLETE" if success else "SAMPLING INCOMPLETE")
    print("=" * 78)
    print(json.dumps(stats, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"output: {output_json}")
    print("=" * 78)

    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
