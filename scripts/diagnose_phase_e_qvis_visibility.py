#!/usr/bin/env python3
"""Offline visibility validity diagnostic for Phase-E q_vis obligations.

Reads one generated q_vis, recomputes per-sensor analytic FOV margins, raycasts
the target segment against the dedicated self-filter collision primitives, and
optionally joins the real confidence-map observation.  It does not modify
planner/runtime semantics.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from urdf_parser_py.urdf import URDF

REPO_DEFAULT = "/home/zhicheng/Project/CAREPlanner"
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "src" / "care_visibility_cdf" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    PLANE_NAMES,
    find_chain_joints,
    fk_transform,
    sensor_margin,
)
from check_visibility_self_occlusion import (  # noqa: E402
    load_collision_primitives,
    q_row_to_map,
    raycast_self_occlusion,
)

TOKEN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def fnum(value, default=math.nan):
    try:
        return float(value)
    except Exception:
        return default


def inum(value, default=-1):
    try:
        return int(float(value))
    except Exception:
        return default


def read_acquisition_rows(path):
    if not path or not os.path.isfile(path):
        return []
    out = []
    with open(path, newline="", errors="replace") as f:
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
            d = dict(TOKEN.findall(text))
            if not d:
                continue
            d["_time_s"] = fnum(row[ti], math.nan) / 1e9
            out.append(d)
    return out


def select_obligation(acq_rows, requested_id):
    valid = []
    for row in acq_rows:
        oid = inum(row.get("active_obligation_id"), -1)
        if oid < 0:
            continue
        if requested_id is not None and oid != requested_id:
            continue
        if row.get("active_query_status") != "ok":
            continue
        qdist = fnum(row.get("active_q_distance_inf"))
        seen = inum(row.get("active_seen"), 0)
        if not math.isfinite(qdist):
            continue
        valid.append((seen, qdist, fnum(row.get("_time_s")), oid, row))
    if not valid:
        return None, None
    unseen = [v for v in valid if v[0] == 0]
    pool = unseen if unseen else valid
    pool.sort(key=lambda x: (x[1], -x[2]))
    best = pool[0]
    return best[3], best[4]


def load_trace_map(trace_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "c46_obligation_*.json"))):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        oid = inum(data.get("c4_6_obligation_id"), -1)
        if oid >= 0:
            out[oid] = (path, data)
    return out


def fmt_vec(values):
    return "[" + ", ".join(f"{float(v):+.5f}" for v in values) + "]"


def build_raycast_args(args):
    class A:
        pass
    out = A()
    out.min_ray_length = args.min_ray_length
    out.ray_start_offset = args.ray_start_offset
    out.point_end_offset = args.point_end_offset
    out.ignore_links = list(args.ignore_links)
    out.ignore_start_inside = args.ignore_start_inside
    out.min_hit_distance = args.min_hit_distance
    return out


def sensor_rows_for_point(
        point, q, self_filter_robot,
        reference_sensor_chains,
        collision_chains, primitives, args):
    q_map = q_row_to_map(DEFAULT_JOINT_NAMES, q)
    ray_args = build_raycast_args(args)
    rows = []

    for sensor_idx, frame in enumerate(DEFAULT_SENSOR_FRAMES):
        t_ref = fk_transform(reference_sensor_chains[sensor_idx], q_map)
        nominal_margin, nominal_plane, p_sensor = sensor_margin(
            point, t_ref,
            args.nominal_hfov_deg, args.nominal_vfov_deg,
            args.nominal_z_min, args.nominal_z_max)

        conservative_margin_raw, conservative_plane, _ = sensor_margin(
            point, t_ref,
            args.conservative_hfov_deg, args.conservative_vfov_deg,
            args.conservative_z_min, args.conservative_z_max)
        conservative_g = conservative_margin_raw - args.conservative_delta

        # The dedicated self-filter URDF intentionally contains body collision
        # primitives only; it does not duplicate the fixed ToF sensor frames.
        # Use the sensor pose from the reference Arm.urdf, while each occluding
        # body primitive is still transformed with the self-filter URDF chain
        # inside raycast_self_occlusion().
        occluded, hit = raycast_self_occlusion(
            self_filter_robot, collision_chains, primitives, t_ref,
            np.asarray(point, dtype=np.float64), q_map, ray_args)

        rows.append({
            "sensor_id": sensor_idx,
            "sensor_frame": frame,
            "point_sensor_xyz": [float(v) for v in p_sensor],
            "axial_z_m": float(p_sensor[2]),
            "range_euclidean_m": float(np.linalg.norm(p_sensor)),
            "nominal_margin_m": float(nominal_margin),
            "nominal_active_plane": PLANE_NAMES[int(nominal_plane)],
            "nominal_fov_visible": bool(nominal_margin >= 0.0),
            "conservative_raw_margin_m": float(conservative_margin_raw),
            "conservative_g_m": float(conservative_g),
            "conservative_active_plane": PLANE_NAMES[int(conservative_plane)],
            "conservative_fov_visible": bool(conservative_g >= 0.0),
            "self_occluded": bool(occluded),
            "occluding_link": hit["link"] if hit else None,
            "occluding_collision": hit["collision"] if hit else None,
            "occluding_type": hit["type"] if hit else None,
            "occluder_distance_from_sensor_m": (
                float(hit["distance"]) if hit else None),
            "nominal_visible_unoccluded": bool(
                nominal_margin >= 0.0 and not occluded),
            "conservative_visible_unoccluded": bool(
                conservative_g >= 0.0 and not occluded),
        })
    return rows


def classify(point_report, acquisition_row):
    rows = point_report["sensors"]
    any_nominal = any(r["nominal_fov_visible"] for r in rows)
    any_cons = any(r["conservative_fov_visible"] for r in rows)
    any_nominal_clear = any(r["nominal_visible_unoccluded"] for r in rows)
    any_cons_clear = any(r["conservative_visible_unoccluded"] for r in rows)
    nominal_visible = [r for r in rows if r["nominal_fov_visible"]]
    all_nominal_blocked = bool(nominal_visible) and all(
        r["self_occluded"] for r in nominal_visible)

    runtime_vis = (
        fnum(acquisition_row.get("active_max_current_visibility"))
        if acquisition_row else math.nan)
    runtime_conf = (
        fnum(acquisition_row.get("active_min_confidence"))
        if acquisition_row else math.nan)

    if not any_nominal:
        reason = "NO_NOMINAL_FOV__LEARNED_QVIS_FALSE_POSITIVE"
    elif all_nominal_blocked:
        reason = "NOMINAL_FOV_BUT_ALL_VISIBLE_SENSORS_SELF_OCCLUDED"
    elif any_nominal_clear and math.isfinite(runtime_vis) and runtime_vis <= 0.0:
        reason = "NOMINAL_FOV_AND_GEOMETRIC_LOS_CLEAR_BUT_RUNTIME_NOT_VISIBLE"
    elif any_nominal_clear:
        reason = "NOMINAL_FOV_AND_GEOMETRIC_LOS_CLEAR"
    else:
        reason = "MIXED_OR_UNRESOLVED"

    return {
        "classification": reason,
        "any_nominal_fov": any_nominal,
        "any_conservative_fov": any_cons,
        "any_nominal_visible_unoccluded": any_nominal_clear,
        "any_conservative_visible_unoccluded": any_cons_clear,
        "all_nominal_fov_sensors_self_occluded": all_nominal_blocked,
        "runtime_active_max_current_visibility": runtime_vis,
        "runtime_active_min_confidence": runtime_conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--acquisition-csv", default="")
    ap.add_argument("--obligation-id", type=int, default=None)
    ap.add_argument("--output-json", required=True)

    ap.add_argument("--reference-urdf", default="src/arm_description/urdf/Arm.urdf")
    ap.add_argument(
        "--self-filter-urdf",
        default="src/arm_description/urdf/Arm_with_self_filter_collision.urdf")

    ap.add_argument("--nominal-hfov-deg", type=float, default=55.0)
    ap.add_argument("--nominal-vfov-deg", type=float, default=72.0)
    ap.add_argument("--nominal-z-min", type=float, default=0.15)
    ap.add_argument("--nominal-z-max", type=float, default=0.75)

    ap.add_argument("--conservative-hfov-deg", type=float, default=50.0)
    ap.add_argument("--conservative-vfov-deg", type=float, default=66.0)
    ap.add_argument("--conservative-z-min", type=float, default=0.20)
    ap.add_argument("--conservative-z-max", type=float, default=0.70)
    ap.add_argument("--conservative-delta", type=float, default=0.01)

    ap.add_argument("--ray-start-offset", type=float, default=0.03)
    ap.add_argument("--point-end-offset", type=float, default=0.005)
    ap.add_argument("--min-hit-distance", type=float, default=0.0)
    ap.add_argument("--min-ray-length", type=float, default=1e-4)
    ap.add_argument("--ignore-start-inside", action="store_true", default=True)
    ap.add_argument("--count-start-inside", dest="ignore_start_inside", action="store_false")
    ap.add_argument("--ignore-links", nargs="*", default=[])
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    ref_urdf = (
        args.reference_urdf if os.path.isabs(args.reference_urdf)
        else os.path.join(repo, args.reference_urdf))
    sf_urdf = (
        args.self_filter_urdf if os.path.isabs(args.self_filter_urdf)
        else os.path.join(repo, args.self_filter_urdf))

    acq_rows = read_acquisition_rows(args.acquisition_csv)
    selected_oid, selected_acq = select_obligation(acq_rows, args.obligation_id)
    traces = load_trace_map(args.trace_dir)

    if args.obligation_id is not None:
        selected_oid = args.obligation_id
    if selected_oid is None:
        if not traces:
            raise SystemExit("No c46_obligation traces found.")
        selected_oid = max(traces.keys())
    if selected_oid not in traces:
        raise SystemExit(
            f"Selected obligation {selected_oid} has no c46 trace in {args.trace_dir}")

    trace_path, trace = traces[selected_oid]
    points = np.asarray(
        trace.get("active_set_points_xyz", []), dtype=np.float64).reshape(-1, 3)
    q_vis = np.asarray(trace.get("q_vis", []), dtype=np.float64).reshape(-1)
    if points.shape[0] < 1:
        raise SystemExit("Selected trace has no active_set_points_xyz")
    if q_vis.shape != (7,) or not np.all(np.isfinite(q_vis)):
        raise SystemExit("Selected trace has invalid q_vis")

    reference_robot = URDF.from_xml_file(ref_urdf)
    self_filter_robot = URDF.from_xml_file(sf_urdf)
    reference_sensor_chains = [
        find_chain_joints(reference_robot, "base_link", frame)
        for frame in DEFAULT_SENSOR_FRAMES]
    primitives = load_collision_primitives(self_filter_robot)
    collision_chains = [
        find_chain_joints(self_filter_robot, "base_link", p["link"])
        for p in primitives]

    point_reports = []
    for point_idx, point in enumerate(points):
        sensor_rows = sensor_rows_for_point(
            point, q_vis, self_filter_robot,
            reference_sensor_chains,
            collision_chains, primitives, args)
        p_report = {
            "point_index": point_idx,
            "point_base_xyz": [float(v) for v in point],
            "sensors": sensor_rows,
        }
        p_report.update(classify(p_report, selected_acq))
        point_reports.append(p_report)

    report = {
        "diagnostic": "phase_e_qvis_visibility_validity",
        "selected_obligation_id": selected_oid,
        "selected_trace": trace_path,
        "q_vis": [float(v) for v in q_vis],
        "learned_final_f_min": trace.get("final_f_min"),
        "learned_final_f_per_point": trace.get("final_f_per_point"),
        "learned_shared_solution_mode": trace.get("shared_solution_mode"),
        "trace_final_oracle_diagnostic": trace.get("final_oracle_diagnostic"),
        "runtime_acquisition_row": selected_acq,
        "self_occlusion_geometry": {
            "urdf": sf_urdf,
            "primitive_count": len(primitives),
            "ray_start_offset_m": args.ray_start_offset,
            "point_end_offset_m": args.point_end_offset,
            "ignore_start_inside": args.ignore_start_inside,
            "note": (
                "Raycast uses the dedicated exact/conservative self-filter "
                "box/cylinder/sphere primitives.")
        },
        "points": point_reports,
    }

    output = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("")
    print("================ Q_VIS VISIBILITY VALIDITY ================")
    print(f"obligation: {selected_oid}")
    print(f"trace:      {trace_path}")
    print(f"q_vis:      {fmt_vec(q_vis)}")
    print(f"learned final_f_min: {trace.get('final_f_min')}")
    if selected_acq:
        print(
            "runtime: q_dist_inf={} seen={} confidence={} current_visibility={}".format(
                selected_acq.get("active_q_distance_inf"),
                selected_acq.get("active_seen"),
                selected_acq.get("active_min_confidence"),
                selected_acq.get("active_max_current_visibility")))
    for p in point_reports:
        print("")
        print(
            f"point[{p['point_index']}]={fmt_vec(p['point_base_xyz'])} "
            f"classification={p['classification']}")
        print(
            "  any nominal FOV={} nominal+LOS={} conservative+LOS={}".format(
                int(p["any_nominal_fov"]),
                int(p["any_nominal_visible_unoccluded"]),
                int(p["any_conservative_visible_unoccluded"])))
        for s in p["sensors"]:
            hit = (
                f"{s['occluding_link']}/{s['occluding_collision']}@"
                f"{s['occluder_distance_from_sensor_m']:.4f}m"
                if s["self_occluded"] else "clear")
            print(
                "  S{sid} {frame:28s} p_s={ps} z={z:+.4f} "
                "nom={nm:+.4f}({nv}) cons_g={cg:+.4f}({cv}) "
                "self={occ} {hit}".format(
                    sid=s["sensor_id"], frame=s["sensor_frame"],
                    ps=fmt_vec(s["point_sensor_xyz"]), z=s["axial_z_m"],
                    nm=s["nominal_margin_m"],
                    nv=int(s["nominal_fov_visible"]),
                    cg=s["conservative_g_m"],
                    cv=int(s["conservative_fov_visible"]),
                    occ=int(s["self_occluded"]), hit=hit))
    print("")
    print(f"[OUTPUT] {output}")
    print("===========================================================")


if __name__ == "__main__":
    main()
