#!/usr/bin/env python3
"""Analyze link-wise kinematic motion envelopes for CAREPlanner body samples.

The script answers a specific geometry question:

    Under the planner's bounded joint motion over a short time window, how far
    can each link's CAREPlanner body-sphere centers move in Cartesian space?

For a body-sample center p attached to a link,

    delta_p ~= J_p(q) delta_q.

For each sampled configuration q and each requested time window dt, this script
uses the planner-level joint velocity limits to construct the local box

    delta_q_j in [-vmax_j * dt, +vmax_j * dt],

clipped by joint position limits. It then reports two first-order envelopes:

  exact_box:
      max ||J_p delta_q||_2 over all 2^7 corners of the local joint box.
      For the linearized model this is the exact maximum because the Euclidean
      norm is convex and a maximum over a box is attained at a vertex.

  triangle_bound:
      sum_j ||J_p[:,j]||_2 * max(|delta_q_j^-|, |delta_q_j^+|).
      This is a looser but directly interpretable conservative upper bound.

For every q, the link value is the maximum over all body samples belonging to
that link. The final table reports P50/P95/P99/MAX across random q.

This is a diagnostic only. It does NOT change the confidence map, startup prior,
VBC, GCDF, or planner.

Default configuration matches the current C5.5/Phase-E planner:
  - planner config: planner_c5_5_vbc_gcdf_regime.yaml
  - local knot dt: horizon_duration / num_intervals = 1.0 / 20 = 0.05 s
  - joint velocity limits: [2,2,2,2,2.5,2.5,2.5] rad/s

Example:
  conda run -n viscdf python scripts/analyze_link_kinematic_motion_envelope.py

Useful variants:
  # Only the native 50-ms planner knot:
  conda run -n viscdf python scripts/analyze_link_kinematic_motion_envelope.py \
      --windows 0.05

  # More random configurations:
  conda run -n viscdf python scripts/analyze_link_kinematic_motion_envelope.py \
      --num-q 10000

  # Explicit velocity limits instead of reading planner YAML:
  conda run -n viscdf python scripts/analyze_link_kinematic_motion_envelope.py \
      --velocity-limits 2 2 2 2 2.5 2.5 2.5
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pinocchio as pin
import yaml


DEFAULT_LINK_ORDER = [
    "link1",
    "link2",
    "link3",
    "link4",
    "wrist_link1",
    "wrist_link2",
    "wrist_link3",
]


@dataclass(frozen=True)
class BodyPoint:
    link_name: str
    frame_name: str
    center_link: np.ndarray
    sample_index: int


def _load_body_points(path: Path) -> List[BodyPoint]:
    doc = yaml.safe_load(path.read_text())
    links = doc.get("body_sampling", {}).get("links", [])
    out: List[BodyPoint] = []
    for item in links:
        link = str(item["link_name"])
        frame = str(item.get("frame", link))
        for idx, sample in enumerate(item.get("samples", [])):
            center = np.asarray(sample["center"], dtype=float).reshape(3)
            out.append(
                BodyPoint(
                    link_name=link,
                    frame_name=frame,
                    center_link=center,
                    sample_index=idx,
                )
            )
    if not out:
        raise SystemExit(f"ERROR: no body samples found in {path}")
    return out


def _planner_velocity_limits(
    planner_yaml: Path,
    joint_names: Sequence[str],
) -> Tuple[np.ndarray, str, float | None]:
    doc = yaml.safe_load(planner_yaml.read_text())

    # Prefer the local-planner/MPC limits because that is the trajectory whose
    # swept body is relevant to the current GCDF/VBC stack.
    candidate_blocks = [
        ("mpc/joint_velocity_limits", doc.get("mpc", {}).get("joint_velocity_limits")),
        (
            "task_generator/joint_velocity_limits",
            doc.get("task_generator", {}).get("joint_velocity_limits"),
        ),
    ]

    limits = None
    source = ""
    for name, block in candidate_blocks:
        if isinstance(block, dict) and all(j in block for j in joint_names):
            limits = np.asarray([float(block[j]) for j in joint_names], dtype=float)
            source = name
            break
    if limits is None:
        raise SystemExit(
            "ERROR: could not find complete joint_velocity_limits in "
            f"{planner_yaml}"
        )

    local = doc.get("local_planner", {})
    native_dt = None
    try:
        horizon = float(local["horizon_duration"])
        n = int(local["num_intervals"])
        if horizon > 0.0 and n > 0:
            native_dt = horizon / float(n)
    except (KeyError, TypeError, ValueError):
        native_dt = None

    return limits, source, native_dt


def _joint_indices(model: pin.Model, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx_q: List[int] = []
    idx_v: List[int] = []
    for name in joint_names:
        jid = model.getJointId(name)
        if jid == 0:
            raise SystemExit(f"ERROR: joint {name!r} not found in URDF")
        joint = model.joints[jid]
        if joint.nq != 1 or joint.nv != 1:
            raise SystemExit(
                f"ERROR: expected 1-DoF joint {name}, got nq={joint.nq}, nv={joint.nv}"
            )
        idx_q.append(int(joint.idx_q))
        idx_v.append(int(joint.idx_v))
    return np.asarray(idx_q, dtype=int), np.asarray(idx_v, dtype=int)


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = [float(a) for a in v]
    return np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _point_jacobian_world(
    model: pin.Model,
    data: pin.Data,
    point: BodyPoint,
    frame_id: int,
    controlled_v_indices: np.ndarray,
) -> np.ndarray:
    """Return 3xN translational Jacobian of a body point in world axes."""
    J_frame = pin.getFrameJacobian(
        model,
        data,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    # Pinocchio Motion/Jacobian convention: first 3 rows linear, last 3 angular.
    Jv = np.asarray(J_frame[:3, :], dtype=float)
    Jw = np.asarray(J_frame[3:, :], dtype=float)

    # Vector from frame origin to attached sample center, expressed in world.
    R_world_frame = np.asarray(data.oMf[frame_id].rotation, dtype=float)
    r_world = R_world_frame @ point.center_link

    # v_point = v_origin + omega x r = v_origin - [r]_x omega.
    J_point = Jv - _skew(r_world) @ Jw
    return J_point[:, controlled_v_indices]


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    return {
        "p50_m": float(np.percentile(values, 50.0)),
        "p95_m": float(np.percentile(values, 95.0)),
        "p99_m": float(np.percentile(values, 99.0)),
        "max_m": float(np.max(values)),
        "mean_m": float(np.mean(values)),
    }


def _fmt_cm(x_m: float) -> str:
    return f"{100.0 * x_m:7.2f}"


def _ceil_quantize(x: float, step: float) -> float:
    if step <= 0.0:
        return float("nan")
    return math.ceil(max(0.0, x) / step - 1e-12) * step


def main() -> int:
    repo_default = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Analyze link-wise CAREPlanner kinematic motion envelopes."
    )
    ap.add_argument("--repo", type=Path, default=repo_default)
    ap.add_argument("--urdf", type=Path, default=None)
    ap.add_argument("--body-samples", type=Path, default=None)
    ap.add_argument("--planner-config", type=Path, default=None)

    ap.add_argument("--num-q", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument(
        "--joint-limit-margin-frac",
        type=float,
        default=0.05,
        help="Random-q sampling margin as a fraction of each joint range.",
    )
    ap.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.25],
        help="Motion windows in seconds.",
    )
    ap.add_argument(
        "--velocity-limits",
        nargs=7,
        type=float,
        default=None,
        metavar=("V1", "V2", "V3", "V4", "VW1", "VW2", "VW3"),
        help="Optional explicit planner joint velocity limits [rad/s].",
    )
    ap.add_argument(
        "--links",
        nargs="+",
        default=DEFAULT_LINK_ORDER,
        help="Links to report. Must exist in body_samples.yaml.",
    )
    ap.add_argument(
        "--quantization-step",
        type=float,
        default=0.05,
        help="Diagnostic upward quantization step in meters (default: 5 cm).",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Default: outputs/link_kinematic_motion_envelope.json",
    )
    args = ap.parse_args()

    repo = args.repo.resolve()
    urdf = (args.urdf or (repo / "src/arm_description/urdf/Arm.urdf")).resolve()
    body_path = (
        args.body_samples
        or (repo / "src/care_confidence_map/config/body_samples.yaml")
    ).resolve()
    planner_yaml = (
        args.planner_config
        or (
            repo
            / "src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml"
        )
    ).resolve()
    out_json = (
        args.output_json
        or (repo / "outputs/link_kinematic_motion_envelope.json")
    ).resolve()

    for p in (urdf, body_path, planner_yaml):
        if not p.is_file():
            raise SystemExit(f"ERROR: file not found: {p}")

    if args.num_q <= 0:
        raise SystemExit("ERROR: --num-q must be positive")
    if not (0.0 <= args.joint_limit_margin_frac < 0.5):
        raise SystemExit("ERROR: --joint-limit-margin-frac must be in [0, 0.5)")
    windows = [float(x) for x in args.windows]
    if any(x <= 0.0 for x in windows):
        raise SystemExit("ERROR: every --windows entry must be positive")

    model = pin.buildModelFromUrdf(str(urdf))
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "wrist_joint1",
        "wrist_joint2",
        "wrist_joint3",
    ]
    q_idx, v_idx = _joint_indices(model, joint_names)

    if args.velocity_limits is None:
        velocity_limits, velocity_source, native_dt = _planner_velocity_limits(
            planner_yaml, joint_names
        )
    else:
        velocity_limits = np.asarray(args.velocity_limits, dtype=float)
        velocity_source = "command_line"
        _, _, native_dt = _planner_velocity_limits(planner_yaml, joint_names)

    if np.any(~np.isfinite(velocity_limits)) or np.any(velocity_limits <= 0.0):
        raise SystemExit(f"ERROR: invalid velocity limits: {velocity_limits.tolist()}")

    points_all = _load_body_points(body_path)
    requested_links = [str(x) for x in args.links]
    available_links = sorted({p.link_name for p in points_all})
    missing = [x for x in requested_links if x not in available_links]
    if missing:
        raise SystemExit(
            f"ERROR: requested links missing from body samples: {missing}; "
            f"available={available_links}"
        )
    points = [p for p in points_all if p.link_name in requested_links]

    frame_ids: Dict[str, int] = {}
    for p in points:
        if p.frame_name not in frame_ids:
            fid = model.getFrameId(p.frame_name)
            if fid >= model.nframes:
                raise SystemExit(
                    f"ERROR: body-sample frame {p.frame_name!r} not found in URDF"
                )
            frame_ids[p.frame_name] = int(fid)

    sample_counts = {
        link: sum(1 for p in points if p.link_name == link)
        for link in requested_links
    }

    lower_full = np.asarray(model.lowerPositionLimit, dtype=float).reshape(-1)
    upper_full = np.asarray(model.upperPositionLimit, dtype=float).reshape(-1)
    lower = lower_full[q_idx]
    upper = upper_full[q_idx]
    span = upper - lower
    margin = float(args.joint_limit_margin_frac)
    sample_lower = lower + margin * span
    sample_upper = upper - margin * span

    if np.any(sample_upper <= sample_lower):
        raise SystemExit("ERROR: shrunken joint sampling limits are invalid")

    # All sign patterns for the 7-D box. For an asymmetric box at a particular
    # q, -1 selects the clipped lower delta and +1 selects the clipped upper.
    sign_patterns = np.asarray(
        list(itertools.product([-1.0, 1.0], repeat=len(joint_names))),
        dtype=float,
    )
    choose_upper = sign_patterns > 0.0

    rng = np.random.default_rng(args.seed)

    # Per window / link / q values.
    exact_values: Dict[float, Dict[str, np.ndarray]] = {
        w: {link: np.zeros(args.num_q, dtype=float) for link in requested_links}
        for w in windows
    }
    triangle_values: Dict[float, Dict[str, np.ndarray]] = {
        w: {link: np.zeros(args.num_q, dtype=float) for link in requested_links}
        for w in windows
    }
    worst_records: Dict[float, Dict[str, Dict[str, object]]] = {
        w: {link: {} for link in requested_links} for w in windows
    }

    data = model.createData()

    print("================================================================")
    print("CAREPlanner link-wise kinematic motion envelope")
    print("================================================================")
    print(f"URDF            : {urdf}")
    print(f"body samples    : {body_path}")
    print(f"planner config  : {planner_yaml}")
    print(f"num random q    : {args.num_q}")
    print(f"seed            : {args.seed}")
    print(f"q margin frac   : {margin}")
    print(f"velocity source : {velocity_source}")
    print(
        "velocity limits : "
        + " ".join(
            f"{name}={v:.3f}" for name, v in zip(joint_names, velocity_limits)
        )
        + " rad/s"
    )
    if native_dt is not None:
        print(f"native local dt : {native_dt:.6f} s")
    print(f"windows         : {windows} s")
    print(
        "samples/link    : "
        + ", ".join(f"{k}={sample_counts[k]}" for k in requested_links)
    )
    print("")

    for qi in range(args.num_q):
        q7 = rng.uniform(sample_lower, sample_upper)
        q = pin.neutral(model)
        q = np.asarray(q, dtype=float).reshape(-1)
        q[q_idx] = q7

        pin.forwardKinematics(model, data, q)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        # Precompute the local delta-q corner matrices for this q and each dt,
        # respecting both velocity and joint-position bounds.
        corner_delta: Dict[float, np.ndarray] = {}
        max_abs_delta: Dict[float, np.ndarray] = {}
        for w in windows:
            nominal = velocity_limits * w
            d_lo = np.maximum(-nominal, lower - q7)
            d_hi = np.minimum(+nominal, upper - q7)
            corners = np.where(choose_upper, d_hi[None, :], d_lo[None, :])
            corner_delta[w] = corners
            max_abs_delta[w] = np.maximum(np.abs(d_lo), np.abs(d_hi))

        # Since arrays start at zero, taking max across body samples gives the
        # per-link envelope for this q directly.
        for p in points:
            Jp = _point_jacobian_world(
                model,
                data,
                p,
                frame_ids[p.frame_name],
                v_idx,
            )

            col_norms = np.linalg.norm(Jp, axis=0)

            for w in windows:
                # Exact maximum of the LINEARIZED displacement over the box.
                disp = corner_delta[w] @ Jp.T
                exact = float(np.max(np.linalg.norm(disp, axis=1)))

                # Triangle-inequality bound, also first-order.
                tri = float(np.dot(col_norms, max_abs_delta[w]))

                if exact > exact_values[w][p.link_name][qi]:
                    exact_values[w][p.link_name][qi] = exact

                if tri > triangle_values[w][p.link_name][qi]:
                    triangle_values[w][p.link_name][qi] = tri

                rec = worst_records[w][p.link_name]
                if not rec or exact > float(rec["exact_box_m"]):
                    rec.clear()
                    rec.update(
                        {
                            "exact_box_m": exact,
                            "triangle_bound_m": tri,
                            "sample_index": int(p.sample_index),
                            "frame_name": p.frame_name,
                            "center_link_m": p.center_link.tolist(),
                            "q_rad": q7.tolist(),
                        }
                    )

        if (qi + 1) % max(1, min(500, args.num_q)) == 0 or qi + 1 == args.num_q:
            print(f"[progress] {qi + 1}/{args.num_q} q samples")

    quant_step = float(args.quantization_step)
    result: Dict[str, object] = {
        "method": {
            "name": "first_order_link_body_point_motion_envelope",
            "exact_box_definition": "max ||J_p(q) delta_q|| over all 2^7 local box corners",
            "triangle_bound_definition": "sum_j ||J_p[:,j]|| * max_abs(delta_q_j)",
            "link_reduction": "maximum over CAREPlanner body-sample centers on the link",
            "configuration_reduction": "P50/P95/P99/MAX across random joint configurations",
            "note": (
                "These are first-order kinematic envelopes, not confidence-map FREE priors "
                "and not collision-clearance values."
            ),
        },
        "inputs": {
            "repo": str(repo),
            "urdf": str(urdf),
            "body_samples": str(body_path),
            "planner_config": str(planner_yaml),
            "num_q": int(args.num_q),
            "seed": int(args.seed),
            "joint_limit_margin_frac": margin,
            "joint_names": joint_names,
            "joint_lower_rad": lower.tolist(),
            "joint_upper_rad": upper.tolist(),
            "joint_velocity_limits_rad_s": velocity_limits.tolist(),
            "velocity_limit_source": velocity_source,
            "native_local_dt_s": native_dt,
            "windows_s": windows,
            "links": requested_links,
            "body_sample_count_per_link": sample_counts,
            "quantization_step_m": quant_step,
        },
        "windows": {},
    }

    print("")
    for w in windows:
        print("================================================================")
        native_tag = ""
        if native_dt is not None and abs(w - native_dt) <= 1e-9:
            native_tag = "  <-- native local-planner knot"
        print(f"WINDOW = {w:.3f} s{native_tag}")
        print("Values below are centimeters.")
        print(
            f"{'link':<14} "
            f"{'P50':>8} {'P95':>8} {'P99':>8} {'MAX':>8} "
            f"{'P95_tri':>10} {'P99_tri':>10} {'ceil5(P95)':>12}"
        )
        print("-" * 92)

        window_json: Dict[str, object] = {}
        for link in requested_links:
            exact = exact_values[w][link]
            tri = triangle_values[w][link]
            es = _percentiles(exact)
            ts = _percentiles(tri)
            q95 = _ceil_quantize(es["p95_m"], quant_step)

            print(
                f"{link:<14} "
                f"{_fmt_cm(es['p50_m']):>8} "
                f"{_fmt_cm(es['p95_m']):>8} "
                f"{_fmt_cm(es['p99_m']):>8} "
                f"{_fmt_cm(es['max_m']):>8} "
                f"{_fmt_cm(ts['p95_m']):>10} "
                f"{_fmt_cm(ts['p99_m']):>10} "
                f"{_fmt_cm(q95):>12}"
            )

            window_json[link] = {
                "exact_box": es,
                "triangle_bound": ts,
                "p95_upward_quantized_m": q95,
                "worst_exact_record": worst_records[w][link],
            }

        result["windows"][f"{w:.6f}"] = window_json
        print("")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print("================================================================")
    print(f"[OUTPUT] {out_json}")
    print("Send me either the terminal table or this JSON and I can interpret the tiers.")
    print("================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
