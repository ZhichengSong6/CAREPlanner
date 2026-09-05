#!/usr/bin/env python3
"""Offline Phase-E blocker A/B/C diagnostic.

A: Is a selected VBC blocker inside the raw body-sample sphere, or only inside
   the extra swept-volume margin shell?
B: Is that blocker geometrically visible from the real eight-sensor FOV at the
   measured seed, the learned q_vis, or a deterministic local q search?
C is produced by the wrapper by comparing two per-case reports.

This script is diagnostic-only. It never publishes ROS commands and never
changes planner/safety state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
Q_MIN = np.asarray([-3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20], dtype=np.float64)
Q_MAX = np.asarray([ 3.14,  2.30,  3.14,  2.65,  3.14,  3.14,  1.20], dtype=np.float64)


def token_rows(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    out: List[Dict[str, object]] = []
    with path.open(newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            return out
        ti = header.index("%time") if "%time" in header else 0
        di = header.index("field.data") if "field.data" in header else 1
        for row in rd:
            if len(row) <= di:
                continue
            text = ",".join(row[di:])
            d: Dict[str, object] = dict(TOKEN.findall(text))
            if not d:
                continue
            try:
                d["_t"] = float(row[ti]) / 1e9
            except Exception:
                d["_t"] = math.nan
            out.append(d)
    return out


def vec3(text: object) -> Optional[np.ndarray]:
    if text is None:
        return None
    s = str(text).strip().strip("[]()")
    try:
        vals = [float(v) for v in s.split(",")]
    except Exception:
        return None
    if len(vals) != 3 or not np.all(np.isfinite(vals)):
        return None
    return np.asarray(vals, dtype=np.float64)


def fnum(v: object, default: float = math.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def inum(v: object, default: int = -1) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def blocker_key(row: Dict[str, object]) -> Optional[Tuple[str, int, Tuple[float, float, float]]]:
    p = vec3(row.get("target"))
    if p is None:
        return None
    # target is already on the 5-cm map grid; round only to make JSON/text
    # formatting harmless.
    xyz = tuple(float(round(v, 4)) for v in p)
    return (str(row.get("link", "none")), inum(row.get("sample_index"), -1), xyz)


def blocker_record(row: Dict[str, object], count: int = 1) -> Dict[str, object]:
    p = vec3(row.get("target"))
    c = vec3(row.get("raw_sample_center_xyz"))
    raw_r = fnum(row.get("raw_sample_radius_m"))
    sweep_r = fnum(row.get("swept_radius_m"))
    d = fnum(row.get("blocker_center_distance_m"))
    return {
        "count": int(count),
        "t": row.get("_t"),
        "point_xyz": None if p is None else p.tolist(),
        "link": row.get("link"),
        "sample_index": inum(row.get("sample_index"), -1),
        "source_type": row.get("source_type"),
        "source_collision_index": inum(row.get("source_collision_index"), -1),
        "sample_center_xyz": None if c is None else c.tolist(),
        "raw_sample_radius_m": raw_r,
        "swept_radius_m": sweep_r,
        "swept_margin_m": (
            sweep_r - raw_r
            if math.isfinite(raw_r) and math.isfinite(sweep_r)
            else math.nan
        ),
        "point_center_distance_m": d,
        "inside_raw_body_sphere": inum(row.get("inside_raw_body_sphere"), 0),
        "margin_shell_only": inum(row.get("margin_shell_only"), 0),
        "confidence": fnum(row.get("confidence")),
        "nominally_visible_on_candidate": inum(row.get("nominally_visible"), 0),
        "see_time_s_on_candidate": fnum(row.get("see_time_s")),
        "sweep_time_s": fnum(row.get("sweep_time_s")),
    }


def load_traces(root: Path) -> List[Tuple[Path, Dict[str, object]]]:
    paths = sorted(set(root.rglob("c46_obligation_*.json")))
    out = []
    for p in paths:
        try:
            out.append((p, json.loads(p.read_text())))
        except Exception:
            pass
    return out


def trace_match_distance(trace: Dict[str, object], point: np.ndarray) -> float:
    try:
        pts = np.asarray(trace.get("active_set_points_xyz", []), dtype=np.float64).reshape(-1, 3)
    except Exception:
        return math.inf
    if pts.size == 0:
        return math.inf
    return float(np.min(np.linalg.norm(pts - point[None, :], axis=1)))


def build_oracle(repo: Path):
    script_dir = repo / "src" / "care_visibility_cdf" / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from evaluate_direct_vs_projection_ascent import (  # pylint: disable=import-error
        DEFAULT_JOINT_NAMES,
        DEFAULT_SENSOR_FRAMES,
        PinocchioFOVOracle,
        oracle_visibility_g,
    )

    urdf = repo / "src" / "arm_description" / "urdf" / "Arm.urdf"
    nominal = PinocchioFOVOracle(
        urdf_path=str(urdf),
        joint_names=list(DEFAULT_JOINT_NAMES),
        sensor_frames=list(DEFAULT_SENSOR_FRAMES),
        horizontal_fov_deg=55.0,
        vertical_fov_deg=72.0,
        z_min=0.15,
        z_max=0.75,
        delta=0.0,
        base_frame="base_link",
    )
    conservative = PinocchioFOVOracle(
        urdf_path=str(urdf),
        joint_names=list(DEFAULT_JOINT_NAMES),
        sensor_frames=list(DEFAULT_SENSOR_FRAMES),
        horizontal_fov_deg=50.0,
        vertical_fov_deg=66.0,
        z_min=0.20,
        z_max=0.70,
        delta=0.01,
        base_frame="base_link",
    )
    return nominal, conservative, oracle_visibility_g


def oracle_g(point: np.ndarray, q: np.ndarray, oracle, fn) -> float:
    x = torch.as_tensor(point.reshape(1, 3), dtype=torch.float32)
    qt = torch.as_tensor(q.reshape(1, 7), dtype=torch.float32)
    with torch.no_grad():
        g = fn(x, qt, oracle)
    return float(g.reshape(-1)[0].item())


def make_local_bank(q0: np.ndarray, qvis: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    bank: List[Tuple[str, np.ndarray]] = []
    bank.append(("measured_seed", q0.copy()))
    bank.append(("q_vis", qvis.copy()))
    for a in (0.10, 0.25, 0.50, 0.75):
        bank.append((f"line_to_qvis_{a:.2f}", np.clip(q0 + a * (qvis - q0), Q_MIN, Q_MAX)))

    for delta in (0.025, 0.05, 0.10, 0.20, 0.35):
        for j in range(7):
            for sign in (-1.0, 1.0):
                q = q0.copy()
                q[j] += sign * delta
                bank.append((f"joint{j+1}_{sign*delta:+.3f}", np.clip(q, Q_MIN, Q_MAX)))

    # Deterministic local coverage around current and around learned q_vis.
    rng = np.random.default_rng(20260905)
    for i in range(256):
        q = np.clip(q0 + rng.uniform(-0.45, 0.45, size=7), Q_MIN, Q_MAX)
        bank.append((f"rand_current_{i:03d}", q))
    for i in range(128):
        q = np.clip(qvis + rng.uniform(-0.25, 0.25, size=7), Q_MIN, Q_MAX)
        bank.append((f"rand_qvis_{i:03d}", q))
    return bank


def oracle_report(
    repo: Path,
    point: np.ndarray,
    traces: List[Tuple[Path, Dict[str, object]]],
) -> Dict[str, object]:
    if not traces:
        return {"status": "no_obligation_trace_matching_blocker"}

    ranked = sorted(
        ((trace_match_distance(t, point), p, t) for p, t in traces),
        key=lambda x: x[0],
    )
    dist, path, trace = ranked[0]
    if not math.isfinite(dist):
        return {"status": "trace_has_no_active_set_points"}

    q0 = np.asarray(
        trace.get("c4_6_measured_seed_q", trace.get("q_deadline_nominal", [])),
        dtype=np.float64,
    ).reshape(-1)
    qvis = np.asarray(trace.get("q_vis", []), dtype=np.float64).reshape(-1)
    if q0.shape != (7,) or qvis.shape != (7,):
        return {
            "status": "trace_missing_q",
            "trace_file": str(path),
            "match_distance_m": dist,
        }

    nominal, conservative, fn = build_oracle(repo)
    g0n = oracle_g(point, q0, nominal, fn)
    gvn = oracle_g(point, qvis, nominal, fn)
    g0c = oracle_g(point, q0, conservative, fn)
    gvc = oracle_g(point, qvis, conservative, fn)

    best = (-math.inf, "none", q0.copy())
    visible_count = 0
    bank = make_local_bank(q0, qvis)
    for label, q in bank:
        g = oracle_g(point, q, nominal, fn)
        if g >= 0.0:
            visible_count += 1
        if g > best[0]:
            best = (g, label, q.copy())

    region_pts = np.asarray(trace.get("active_set_points_xyz", []), dtype=np.float64).reshape(-1, 3)
    qvis_region_g: List[float] = []
    if region_pts.size:
        for p in region_pts:
            qvis_region_g.append(oracle_g(p, qvis, nominal, fn))

    return {
        "status": "ok",
        "trace_file": str(path),
        "trace_obligation_id": inum(trace.get("c4_6_obligation_id"), -1),
        "trace_point_match_distance_m": dist,
        "trace_active_set_size": int(region_pts.shape[0]),
        "q_measured_seed": q0.tolist(),
        "q_vis": qvis.tolist(),
        "trace_final_oracle_diagnostic": trace.get("final_oracle_diagnostic"),
        "blocker_nominal_g_at_measured_seed": g0n,
        "blocker_nominal_visible_at_measured_seed": bool(g0n >= 0.0),
        "blocker_nominal_g_at_q_vis": gvn,
        "blocker_nominal_visible_at_q_vis": bool(gvn >= 0.0),
        "blocker_conservative_g_at_measured_seed": g0c,
        "blocker_conservative_visible_at_measured_seed": bool(g0c >= 0.0),
        "blocker_conservative_g_at_q_vis": gvc,
        "blocker_conservative_visible_at_q_vis": bool(gvc >= 0.0),
        "local_search_candidate_count": len(bank),
        "local_search_visible_candidate_count": visible_count,
        "local_search_any_nominal_visible": bool(visible_count > 0),
        "local_search_best_nominal_g": float(best[0]),
        "local_search_best_source": best[1],
        "local_search_best_q": best[2].tolist(),
        "q_vis_region_min_nominal_g": (
            float(min(qvis_region_g)) if qvis_region_g else math.nan
        ),
        "q_vis_region_all_nominal_visible": bool(
            qvis_region_g and min(qvis_region_g) >= 0.0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--diag-root", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    root = Path(args.diag_root).resolve()
    cand_path = root / "artifacts" / "candidate_vbc_summary.csv"
    rows = token_rows(cand_path)
    unsafe = [
        r for r in rows
        if r.get("trajectory_source") == "predicted"
        and r.get("has_violation") == "1"
        and blocker_key(r) is not None
    ]

    counts = Counter(blocker_key(r) for r in unsafe)
    first = unsafe[0] if unsafe else None
    top_rows: List[Dict[str, object]] = []
    for key, count in counts.most_common(12):
        row = next(r for r in unsafe if blocker_key(r) == key)
        top_rows.append(blocker_record(row, count))

    top = top_rows[0] if top_rows else None
    traces = load_traces(root)
    b_report: Dict[str, object] = {"status": "no_top_blocker"}
    if top is not None and top.get("point_xyz") is not None:
        b_report = oracle_report(
            repo,
            np.asarray(top["point_xyz"], dtype=np.float64),
            traces,
        )

    first_record = blocker_record(first, 1) if first is not None else None
    report = {
        "test": "phase_e_blocker_abc_case",
        "case_id": args.case_id,
        "diag_root": str(root),
        "candidate_vbc_record_count": len(rows),
        "predicted_unsafe_with_blocker_count": len(unsafe),
        "unique_blocker_key_count": len(counts),
        "first_blocker": first_record,
        "top_blockers": top_rows,
        "top_blocker_fraction_of_unsafe": (
            float(top_rows[0]["count"]) / float(len(unsafe))
            if unsafe and top_rows else 0.0
        ),
        "A_raw_body_vs_margin": (
            None if top is None else {
                "point_xyz": top["point_xyz"],
                "link": top["link"],
                "sample_index": top["sample_index"],
                "raw_sample_radius_m": top["raw_sample_radius_m"],
                "swept_radius_m": top["swept_radius_m"],
                "swept_margin_m": top["swept_margin_m"],
                "point_center_distance_m": top["point_center_distance_m"],
                "inside_raw_body_sphere": top["inside_raw_body_sphere"],
                "margin_shell_only": top["margin_shell_only"],
                "source_type": top["source_type"],
                "source_collision_index": top["source_collision_index"],
            }
        ),
        "B_oracle_visibility": b_report,
        "obligation_trace_count": len(traces),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=True))

    print("\n========== BLOCKER ABC CASE REPORT ==========")
    print(json.dumps(report, indent=2, allow_nan=True))
    print("=============================================")


if __name__ == "__main__":
    main()
