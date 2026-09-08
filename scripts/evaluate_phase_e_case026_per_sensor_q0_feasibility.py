#!/usr/bin/env python3
"""Case-026 per-sensor q0 feasibility diagnostic.

Purpose
-------
Before training an 8-output per-sensor VisCDF, test whether the *existing*
per-sensor zero-level-set library already contains useful alternative sensor
branches for the Case-026 failure.

For each sensor:
  1) take the nearest spatial grid points to the exact Case-026 target,
  2) collect valid per-sensor q0 boundary samples,
  3) rank them by motion from the obligation's measured seed q,
  4) keep inactive downstream joints at the measured seed,
  5) move a small distance inward using the analytic per-sensor FOV margin,
  6) require nominal FOV validity at the exact target,
  7) run zero-padding primitive sensor->target self-occlusion raycast.

This is a diagnostic only.  It does not modify planner, NCDF, or training.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from urdf_parser_py.urdf import URDF


REPO_DEFAULT = "/home/zhicheng/Project/CAREPlanner"
VIS_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "src" / "care_visibility_cdf" / "scripts"
if str(VIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(VIS_SCRIPT_DIR))

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


class RayArgs:
    min_ray_length = 1e-4
    ray_start_offset = 0.03
    point_end_offset = 0.005
    ignore_links = []
    ignore_start_inside = True
    min_hit_distance = 0.0


def fmt_vec(v, nd=4):
    return "[" + ", ".join(f"{float(x):+.{nd}f}" for x in v) + "]"


def load_trace(path):
    d = json.load(open(path))
    target_points = np.asarray(d["active_set_points_xyz"], dtype=np.float64).reshape(-1, 3)
    if len(target_points) != 1:
        # Case-026 obligation 7 is a single point.  Keep the centroid fallback
        # explicit so the script remains informative if the trace is regenerated.
        target = target_points.mean(axis=0)
    else:
        target = target_points[0]
    seed = np.asarray(d["c4_6_measured_seed_q"], dtype=np.float64).reshape(7)
    q_vis = np.asarray(d["q_vis"], dtype=np.float64).reshape(7)
    return d, target, seed, q_vis


def sensor_nominal_margin(point, q, chain, hfov, vfov, zmin, zmax):
    q_map = q_row_to_map(DEFAULT_JOINT_NAMES, q)
    T = fk_transform(chain, q_map)
    margin, plane, ps = sensor_margin(point, T, hfov, vfov, zmin, zmax)
    return float(margin), int(plane), np.asarray(ps, dtype=np.float64), T


def sensor_conservative_g(point, q, chain, hfov, vfov, zmin, zmax, delta):
    margin, plane, ps, _ = sensor_nominal_margin(
        point, q, chain, hfov, vfov, zmin, zmax)
    return float(margin - delta), plane, ps


def primitive_los(
        point, q, sensor_chain, self_filter_robot,
        collision_chains, primitives):
    q_map = q_row_to_map(DEFAULT_JOINT_NAMES, q)
    T_sensor = fk_transform(sensor_chain, q_map)
    occluded, hit = raycast_self_occlusion(
        self_filter_robot,
        collision_chains,
        primitives,
        T_sensor,
        np.asarray(point, dtype=np.float64),
        q_map,
        RayArgs(),
    )
    return bool(occluded), hit


def finite_difference_margin_grad(
        point, q, chain, active_mask,
        hfov, vfov, zmin, zmax, eps):
    grad = np.zeros(7, dtype=np.float64)
    for j in range(7):
        if active_mask[j] <= 0.5:
            continue
        qp = q.copy()
        qm = q.copy()
        qp[j] += eps
        qm[j] -= eps
        mp = sensor_nominal_margin(
            point, qp, chain, hfov, vfov, zmin, zmax)[0]
        mm = sensor_nominal_margin(
            point, qm, chain, hfov, vfov, zmin, zmax)[0]
        grad[j] = (mp - mm) / (2.0 * eps)
    return grad


def refine_inside(
        point, q0, chain, active_mask, q_min, q_max,
        hfov, vfov, zmin, zmax,
        target_margin, max_iters, step, fd_eps):
    q = np.asarray(q0, dtype=np.float64).copy()
    history = []
    for it in range(max_iters + 1):
        margin, plane, _, _ = sensor_nominal_margin(
            point, q, chain, hfov, vfov, zmin, zmax)
        history.append(float(margin))
        if margin >= target_margin:
            return q, True, it, history, plane
        if it == max_iters:
            break

        grad = finite_difference_margin_grad(
            point, q, chain, active_mask,
            hfov, vfov, zmin, zmax, fd_eps)
        gn = float(np.linalg.norm(grad))
        if not math.isfinite(gn) or gn < 1e-9:
            break

        direction = grad / gn
        old_margin = margin
        accepted = False
        alpha = step
        for _ in range(8):
            trial = np.clip(q + alpha * direction, q_min, q_max)
            new_margin = sensor_nominal_margin(
                point, trial, chain, hfov, vfov, zmin, zmax)[0]
            if new_margin > old_margin + 1e-8:
                q = trial
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    margin, plane, _, _ = sensor_nominal_margin(
        point, q, chain, hfov, vfov, zmin, zmax)
    return q, bool(margin >= target_margin), len(history) - 1, history, plane


def motion_metrics(q, ref):
    d = np.asarray(q) - np.asarray(ref)
    return {
        "linf": float(np.max(np.abs(d))),
        "l2": float(np.linalg.norm(d)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--trace", required=True)
    ap.add_argument(
        "--data",
        default="src/care_visibility_cdf/data/"
                "visibility_yiming_style_grid30_q20000_k500_fovonly.npz")
    ap.add_argument(
        "--reference-urdf",
        default="src/arm_description/urdf/Arm.urdf")
    ap.add_argument(
        "--self-filter-urdf",
        default="src/arm_description/urdf/Arm_with_self_filter_collision.urdf")
    ap.add_argument("--output-json", required=True)

    ap.add_argument("--blocked-sensor", type=int, default=4)
    ap.add_argument("--spatial-neighbors", type=int, default=8)
    ap.add_argument("--top-boundary-candidates", type=int, default=16)
    ap.add_argument("--target-nominal-margin", type=float, default=0.015)
    ap.add_argument("--max-refine-iters", type=int, default=20)
    ap.add_argument("--refine-step", type=float, default=0.04)
    ap.add_argument("--fd-eps", type=float, default=1e-4)

    ap.add_argument("--nominal-hfov-deg", type=float, default=55.0)
    ap.add_argument("--nominal-vfov-deg", type=float, default=72.0)
    ap.add_argument("--nominal-z-min", type=float, default=0.15)
    ap.add_argument("--nominal-z-max", type=float, default=0.75)
    ap.add_argument("--conservative-hfov-deg", type=float, default=50.0)
    ap.add_argument("--conservative-vfov-deg", type=float, default=66.0)
    ap.add_argument("--conservative-z-min", type=float, default=0.20)
    ap.add_argument("--conservative-z-max", type=float, default=0.70)
    ap.add_argument("--conservative-delta", type=float, default=0.01)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    data_path = args.data if os.path.isabs(args.data) else os.path.join(repo, args.data)
    ref_urdf = (
        args.reference_urdf if os.path.isabs(args.reference_urdf)
        else os.path.join(repo, args.reference_urdf))
    sf_urdf = (
        args.self_filter_urdf if os.path.isabs(args.self_filter_urdf)
        else os.path.join(repo, args.self_filter_urdf))

    trace, target, seed_q, blocked_q_vis = load_trace(args.trace)

    print(f"[data] opening {data_path}", flush=True)
    with np.load(data_path, allow_pickle=True) as d:
        # Keep the very large q library in its stored float32 form.  Only the
        # small candidate slices selected below are promoted to float64.
        x = d["x"].astype(np.float32, copy=False)
        qlib = d["q"].astype(np.float32, copy=False)
        valid = d["valid_fov"].astype(np.bool_, copy=False)
        sensor_masks = d["sensor_chain_masks"].astype(np.float32, copy=False)
        if "q_min" in d.files and "q_max" in d.files:
            q_min = d["q_min"].astype(np.float64)
            q_max = d["q_max"].astype(np.float64)
        else:
            q_min = np.full(7, -math.pi, dtype=np.float64)
            q_max = np.full(7, math.pi, dtype=np.float64)

    if qlib.ndim != 4 or qlib.shape[2:] != (7, len(DEFAULT_SENSOR_FRAMES)):
        raise RuntimeError(f"unexpected q library shape: {qlib.shape}")
    if valid.shape != (qlib.shape[0], qlib.shape[1], qlib.shape[3]):
        raise RuntimeError(f"unexpected valid_fov shape: {valid.shape}")

    spatial_d = np.linalg.norm(x - target[None, :], axis=1)
    neighbor_ids = np.argsort(spatial_d)[:args.spatial_neighbors]

    reference_robot = URDF.from_xml_file(ref_urdf)
    self_filter_robot = URDF.from_xml_file(sf_urdf)
    sensor_chains_ref = [
        find_chain_joints(reference_robot, "base_link", frame)
        for frame in DEFAULT_SENSOR_FRAMES]
    sensor_chains_sf = [
        # Sensor frames are absent from the dedicated self-filter URDF, so LOS
        # uses the reference sensor pose.  Primitive FK itself uses sf chains.
        None for _ in DEFAULT_SENSOR_FRAMES]
    primitives = load_collision_primitives(self_filter_robot)
    collision_chains = [
        find_chain_joints(self_filter_robot, "base_link", p["link"])
        for p in primitives]

    # Sanity: reproduce the known blocked S4 q_vis from the runtime trace.
    blocked_margin, blocked_plane, blocked_ps, blocked_T = sensor_nominal_margin(
        target, blocked_q_vis, sensor_chains_ref[args.blocked_sensor],
        args.nominal_hfov_deg, args.nominal_vfov_deg,
        args.nominal_z_min, args.nominal_z_max)
    blocked_q_map = q_row_to_map(DEFAULT_JOINT_NAMES, blocked_q_vis)
    blocked_occ, blocked_hit = raycast_self_occlusion(
        self_filter_robot, collision_chains, primitives, blocked_T,
        target, blocked_q_map, RayArgs())

    results = []
    any_alternative = False

    for s, frame in enumerate(DEFAULT_SENSOR_FRAMES):
        active_mask = sensor_masks[s]
        raw_candidates = []

        for xi in neighbor_ids:
            valid_k = np.flatnonzero(valid[xi, :, s])
            for k in valid_k:
                q0 = np.asarray(qlib[xi, k, :, s], dtype=np.float64)
                if not np.all(np.isfinite(q0)):
                    continue

                # Downstream joints that do not affect this sensor should stay at
                # the measured seed; this is both lower-motion and semantically
                # equivalent for the sensor pose.
                qc = seed_q.copy()
                use = active_mask > 0.5
                qc[use] = q0[use]
                mm = motion_metrics(qc, seed_q)
                raw_candidates.append({
                    "x_index": int(xi),
                    "k": int(k),
                    "grid_point": x[xi].copy(),
                    "grid_distance_m": float(spatial_d[xi]),
                    "q_boundary": qc,
                    "seed_linf": mm["linf"],
                    "seed_l2": mm["l2"],
                })

        raw_candidates.sort(key=lambda z: (z["seed_linf"], z["seed_l2"], z["grid_distance_m"]))
        raw_candidates = raw_candidates[:args.top_boundary_candidates]

        tested = []
        best_clear = None
        best_blocked = None

        for cand in raw_candidates:
            q_refined, reached, nit, hist, plane = refine_inside(
                target,
                cand["q_boundary"],
                sensor_chains_ref[s],
                active_mask,
                q_min, q_max,
                args.nominal_hfov_deg, args.nominal_vfov_deg,
                args.nominal_z_min, args.nominal_z_max,
                args.target_nominal_margin,
                args.max_refine_iters,
                args.refine_step,
                args.fd_eps,
            )

            nm, nplane, ps, T_sensor = sensor_nominal_margin(
                target, q_refined, sensor_chains_ref[s],
                args.nominal_hfov_deg, args.nominal_vfov_deg,
                args.nominal_z_min, args.nominal_z_max)
            cg, cplane, _ = sensor_conservative_g(
                target, q_refined, sensor_chains_ref[s],
                args.conservative_hfov_deg, args.conservative_vfov_deg,
                args.conservative_z_min, args.conservative_z_max,
                args.conservative_delta)

            q_map = q_row_to_map(DEFAULT_JOINT_NAMES, q_refined)
            occluded, hit = raycast_self_occlusion(
                self_filter_robot, collision_chains, primitives,
                T_sensor, target, q_map, RayArgs())

            mseed = motion_metrics(q_refined, seed_q)
            mblocked = motion_metrics(q_refined, blocked_q_vis)
            rec = {
                "x_index": cand["x_index"],
                "k": cand["k"],
                "grid_point": [float(v) for v in cand["grid_point"]],
                "grid_distance_m": cand["grid_distance_m"],
                "q_boundary": [float(v) for v in cand["q_boundary"]],
                "q_refined": [float(v) for v in q_refined],
                "refine_reached_target_margin": bool(reached),
                "refine_iters": int(nit),
                "nominal_margin_m": float(nm),
                "nominal_active_plane": PLANE_NAMES[int(nplane)],
                "conservative_g_m": float(cg),
                "conservative_active_plane": PLANE_NAMES[int(cplane)],
                "point_sensor_xyz": [float(v) for v in ps],
                "primitive_self_occluded": bool(occluded),
                "primitive_hit": hit,
                "motion_from_seed": mseed,
                "motion_from_blocked_qvis": mblocked,
            }
            tested.append(rec)

            if reached and nm >= args.target_nominal_margin:
                if not occluded:
                    if best_clear is None or (
                        mseed["linf"], mseed["l2"]) < (
                            best_clear["motion_from_seed"]["linf"],
                            best_clear["motion_from_seed"]["l2"]):
                        best_clear = rec
                else:
                    if best_blocked is None or (
                        mseed["linf"], mseed["l2"]) < (
                            best_blocked["motion_from_seed"]["linf"],
                            best_blocked["motion_from_seed"]["l2"]):
                        best_blocked = rec

        sensor_result = {
            "sensor_id": s,
            "sensor_frame": frame,
            "is_blocked_runtime_winner": bool(s == args.blocked_sensor),
            "active_joint_mask": [float(v) for v in active_mask],
            "raw_valid_boundary_candidates_considered": int(len(raw_candidates)),
            "tested_candidates": tested,
            "best_unoccluded": best_clear,
            "best_occluded": best_blocked,
            "alternative_feasible": bool(best_clear is not None and s != args.blocked_sensor),
        }
        if sensor_result["alternative_feasible"]:
            any_alternative = True
        results.append(sensor_result)

    report = {
        "diagnostic": "phase_e_case026_per_sensor_q0_feasibility",
        "trace": os.path.abspath(args.trace),
        "target": [float(v) for v in target],
        "measured_seed_q": [float(v) for v in seed_q],
        "blocked_q_vis": [float(v) for v in blocked_q_vis],
        "blocked_sensor_id": int(args.blocked_sensor),
        "blocked_sensor_frame": DEFAULT_SENSOR_FRAMES[args.blocked_sensor],
        "blocked_qvis_sanity": {
            "nominal_margin_m": float(blocked_margin),
            "nominal_active_plane": PLANE_NAMES[int(blocked_plane)],
            "point_sensor_xyz": [float(v) for v in blocked_ps],
            "primitive_self_occluded": bool(blocked_occ),
            "primitive_hit": blocked_hit,
        },
        "dataset": {
            "path": data_path,
            "x_shape": list(x.shape),
            "q_shape": list(qlib.shape),
            "valid_shape": list(valid.shape),
            "nearest_spatial_neighbors": [
                {
                    "x_index": int(i),
                    "point": [float(v) for v in x[i]],
                    "distance_m": float(spatial_d[i]),
                }
                for i in neighbor_ids
            ],
        },
        "config": {
            "spatial_neighbors": args.spatial_neighbors,
            "top_boundary_candidates": args.top_boundary_candidates,
            "target_nominal_margin": args.target_nominal_margin,
            "primitive_los_padding_m": 0.0,
        },
        "sensors": results,
        "any_nonblocked_sensor_feasible": bool(any_alternative),
    }

    out = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("")
    print("================ CASE 026 PER-SENSOR FEASIBILITY ================")
    print(f"target              : {fmt_vec(target, 5)}")
    print(f"measured seed q     : {fmt_vec(seed_q, 5)}")
    print(f"blocked q_vis       : {fmt_vec(blocked_q_vis, 5)}")
    print(
        f"blocked S{args.blocked_sensor} sanity : "
        f"margin={blocked_margin:+.5f} "
        f"self_occluded={int(blocked_occ)} "
        f"hit={blocked_hit}")
    print("")
    print("sensor | alt? | best seed Linf | nominal m | cons g | LOS | candidate")
    print("-------+------+----------------+-----------+--------+-----+----------------")
    for sr in results:
        best = sr["best_unoccluded"]
        if best is None:
            print(
                f"S{sr['sensor_id']}    |  no  |       --       |    --     |   --   | --  | "
                f"{sr['sensor_frame']}")
        else:
            alt = "YES" if sr["alternative_feasible"] else "win"
            print(
                f"S{sr['sensor_id']}    | {alt:4s} | "
                f"{best['motion_from_seed']['linf']:14.4f} | "
                f"{best['nominal_margin_m']:+9.4f} | "
                f"{best['conservative_g_m']:+6.3f} | "
                f"CLR | {sr['sensor_frame']}")
            print(f"       q={fmt_vec(best['q_refined'], 5)}")

    print("")
    print(
        "RESULT: any alternative sensor after removing S%d = %s" %
        (args.blocked_sensor, "YES" if any_alternative else "NO"))
    print(f"[OUTPUT] {out}")
    print("=================================================================")


if __name__ == "__main__":
    main()
