#!/usr/bin/env python3
"""A/B visualization: task-only nominal baseline vs full CAREPlanner execution.

Inputs are run directories containing rostopic CSV logs.  The comparison uses
measured /care_arm/joint_states for both runs and reconstructs the robot/EE
motion with the project URDF, so it compares actual executed motion rather than
planner intent alone.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
# Older Ubuntu/ROS matplotlib builds do not always auto-register the "3d"
# projection. Importing mplot3d explicitly registers it with matplotlib.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from urdf_parser_py.urdf import URDF


JOINTS = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]
TOK = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def finite_float(x, default=math.nan):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def indexed_cols(header, prefix):
    out = {}
    for i, h in enumerate(header):
        s = h.replace("field.", "").replace("[", "").replace("]", "")
        m = re.match(r"^([A-Za-z_]+)(\d+)$", s)
        if m and m.group(1) == prefix:
            out[int(m.group(2))] = i
    return out


def read_joint_states(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out_t, out_q = [], []
    need = set(JOINTS)
    with open(path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            raise RuntimeError("empty joint-state CSV: " + path)
        ti = header.index("%time") if "%time" in header else 0
        nc = indexed_cols(header, "name")
        pc = indexed_cols(header, "position")
        ids = sorted(set(nc) & set(pc))
        if not ids:
            raise RuntimeError("joint_states.csv has no flattened name*/position* columns")
        for row in rd:
            q = {}
            for k in ids:
                if nc[k] >= len(row) or pc[k] >= len(row):
                    continue
                name = row[nc[k]].strip()
                value = finite_float(row[pc[k]])
                if name in need and math.isfinite(value):
                    q[name] = value
            if not need.issubset(q):
                continue
            t = finite_float(row[ti]) / 1e9
            if math.isfinite(t):
                out_t.append(t)
                out_q.append([q[n] for n in JOINTS])
    if not out_t:
        raise RuntimeError("no complete 7-DoF samples in " + path)
    return np.asarray(out_t, dtype=float), np.asarray(out_q, dtype=float)


def read_token_csv(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd, [])
        if not header:
            return rows
        ti = header.index("%time") if "%time" in header else 0
        di = header.index("field.data") if "field.data" in header else 1
        for row in rd:
            if len(row) <= di:
                continue
            d = dict(TOK.findall(",".join(row[di:])))
            if not d:
                continue
            t = finite_float(row[ti]) / 1e9 if len(row) > ti else math.nan
            d["_t"] = t
            rows.append(d)
    return rows


def load_fk_helper(repo):
    path = os.path.join(
        repo, "src/egocentric_arm_planner/scripts/compute_ee_workspace_bounds.py")
    spec = importlib.util.spec_from_file_location("care_fk_helper", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def motion_start_time(t, q, threshold=1e-3):
    q0 = q[0]
    delta = np.max(np.abs(q - q0[None, :]), axis=1)
    idx = np.flatnonzero(delta >= threshold)
    return float(t[idx[0]]) if idx.size else float(t[0])


def normalize_motion_time(t, q):
    start = motion_start_time(t, q)
    return t - start, start


def resample(t, q, sample_t):
    qq = np.empty((len(sample_t), q.shape[1]), dtype=float)
    for j in range(q.shape[1]):
        qq[:, j] = np.interp(sample_t, t, q[:, j], left=q[0, j], right=q[-1, j])
    return qq


def fk_transform(helper, chain, q):
    qmap = dict(zip(JOINTS, q))
    T = np.eye(4)
    for joint in chain:
        T = (
            T
            @ helper.get_joint_origin_transform(joint)
            @ helper.joint_motion_transform(joint, qmap.get(joint.name, 0.0))
        )
    return T


def skeleton_points(helper, chain, q):
    qmap = dict(zip(JOINTS, q))
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for joint in chain:
        T = (
            T
            @ helper.get_joint_origin_transform(joint)
            @ helper.joint_motion_transform(joint, qmap.get(joint.name, 0.0))
        )
        pts.append(T[:3, 3].copy())
    return np.asarray(pts)


def ee_trace(helper, chain, q):
    out = np.empty((len(q), 3), dtype=float)
    for i, qi in enumerate(q):
        out[i] = fk_transform(helper, chain, qi)[:3, 3]
    return out


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def joint_path_length(q):
    if len(q) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


def regime_intervals(rows, care_motion_start_abs, end_rel):
    clean = []
    for r in rows:
        t = finite_float(r.get("_t"))
        state = r.get("state", "UNKNOWN")
        if math.isfinite(t):
            clean.append((t - care_motion_start_abs, state))
    if not clean:
        return []
    clean.sort(key=lambda x: x[0])
    intervals = []
    for i, (start, state) in enumerate(clean):
        end = clean[i + 1][0] if i + 1 < len(clean) else end_rel
        start = max(0.0, start)
        end = min(end_rel, end)
        if end > start:
            if intervals and intervals[-1][2] == state and abs(intervals[-1][1] - start) < 1e-9:
                intervals[-1] = (intervals[-1][0], end, state)
            else:
                intervals.append((start, end, state))
    return intervals


def state_at(intervals, t):
    for a, b, state in intervals:
        if a <= t < b:
            return state
    if intervals and t >= intervals[-1][1]:
        return intervals[-1][2]
    return "UNKNOWN"


def shade_regimes(ax, intervals):
    # Use the default matplotlib cycle rather than hardcoding paper colors.
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    state_color = {
        "NORMAL": cycle[0] if len(cycle) > 0 else "0.8",
        "REPAIR": cycle[1] if len(cycle) > 1 else "0.7",
        "PROBE_NORMAL": cycle[2] if len(cycle) > 2 else "0.6",
        "PROBE": cycle[2] if len(cycle) > 2 else "0.6",
    }
    seen = set()
    for a, b, state in intervals:
        color = state_color.get(state, "0.85")
        label = state if state not in seen else None
        ax.axvspan(a, b, alpha=0.09, color=color, linewidth=0, label=label)
        seen.add(state)


def equalize_3d_axes(ax, points):
    points = np.asarray(points)
    lo = np.nanmin(points, axis=0)
    hi = np.nanmax(points, axis=0)
    center = 0.5 * (lo + hi)
    radius = max(0.15, 0.55 * float(np.max(hi - lo)))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def save_ee_plot(outdir, ee_base, ee_care, goal):
    fig = plt.figure(figsize=(8.4, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_base[:, 0], ee_base[:, 1], ee_base[:, 2], label="Task-only baseline")
    ax.plot(ee_care[:, 0], ee_care[:, 1], ee_care[:, 2], label="CAREPlanner executed")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=110, label="EE goal")
    ax.scatter([ee_base[0, 0]], [ee_base[0, 1]], [ee_base[0, 2]], marker="o", s=36, label="Start")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Executed end-effector path")
    equalize_3d_axes(ax, np.vstack([ee_base, ee_care, goal[None, :]]))
    ax.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(outdir, "ee_path_3d.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_joint_plot(outdir, tb, qb, tc, qc, intervals):
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 11.5), sharex=True)
    axes = axes.ravel()
    for j, name in enumerate(JOINTS):
        ax = axes[j]
        shade_regimes(ax, intervals)
        ax.plot(tb, qb[:, j], label="Task-only baseline")
        ax.plot(tc, qc[:, j], label="CAREPlanner")
        ax.set_ylabel(name + " [rad]")
        ax.grid(True, alpha=0.22)
    axes[-1].axis("off")
    axes[0].legend(loc="best")
    axes[6].set_xlabel("Time from motion start [s]")
    axes[5].set_xlabel("Time from motion start [s]")
    fig.suptitle("Measured joint motion: task-only baseline vs CAREPlanner\n"
                 "CAREPlanner background shading indicates NORMAL / REPAIR / PROBE")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(outdir, "joint_motion_comparison.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_difference_plot(outdir, tb, qb, tc, qc, intervals):
    qbase_on_care = resample(tb, qb, tc)
    diff = qc - qbase_on_care
    diff_inf = np.max(np.abs(diff), axis=1)
    diff_l2 = np.linalg.norm(diff, axis=1)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    shade_regimes(ax, intervals)
    ax.plot(tc, diff_inf, label=r"$\|q_{CARE}-q_{task-only}\|_\infty$")
    ax.plot(tc, diff_l2, label=r"$\|q_{CARE}-q_{task-only}\|_2$")
    ax.set_xlabel("Time from motion start [s]")
    ax.set_ylabel("Joint-space deviation [rad]")
    ax.set_title("How much CAREPlanner changes the task-only arm configuration")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    path = os.path.join(outdir, "motion_difference.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, diff_inf, diff_l2


def save_animation(outdir, helper, chain, tb, qb, tc, qc, intervals,
                   goal, playback_speed, fps):
    end_t = max(float(tb[-1]), float(tc[-1]))
    playback_speed = max(0.1, float(playback_speed))
    fps = max(5, int(fps))
    dt_sim = playback_speed / fps
    frame_t = np.arange(0.0, end_t + 0.5 * dt_sim, dt_sim)
    # Bound pathological videos while preserving the full run.
    if len(frame_t) > 1800:
        frame_t = np.linspace(0.0, end_t, 1800)

    qb_f = resample(tb, qb, frame_t)
    qc_f = resample(tc, qc, frame_t)

    # Build global bounds from a sparse set of skeleton samples plus goal.
    idx = np.unique(np.linspace(0, len(frame_t) - 1, min(100, len(frame_t))).astype(int))
    bounds = [goal]
    for i in idx:
        bounds.extend(skeleton_points(helper, chain, qb_f[i]))
        bounds.extend(skeleton_points(helper, chain, qc_f[i]))
    bounds = np.asarray(bounds)

    fig = plt.figure(figsize=(12.8, 6.3))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for ax, title in ((ax1, "Task-only baseline"), (ax2, "CAREPlanner full")):
        equalize_3d_axes(ax, bounds)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(title)
        ax.scatter([goal[0]], [goal[1]], [goal[2]], marker="*", s=85, label="EE goal")

    p1 = skeleton_points(helper, chain, qb_f[0])
    p2 = skeleton_points(helper, chain, qc_f[0])
    line1, = ax1.plot(p1[:, 0], p1[:, 1], p1[:, 2], "-o", linewidth=2)
    line2, = ax2.plot(p2[:, 0], p2[:, 1], p2[:, 2], "-o", linewidth=2)
    trace1, = ax1.plot([], [], [], linewidth=1.5, alpha=0.75)
    trace2, = ax2.plot([], [], [], linewidth=1.5, alpha=0.75)
    ee1_hist, ee2_hist = [], []

    status = fig.text(
        0.5, 0.02, "", ha="center", va="bottom", fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.08))

    reduced = False
    try:
        reduced = bool(os.environ.get("CAREPLANNER_REDUCED_MOTION", ""))
    except Exception:
        pass

    def update(i):
        pb = skeleton_points(helper, chain, qb_f[i])
        pc = skeleton_points(helper, chain, qc_f[i])
        line1.set_data(pb[:, 0], pb[:, 1])
        line1.set_3d_properties(pb[:, 2])
        line2.set_data(pc[:, 0], pc[:, 1])
        line2.set_3d_properties(pc[:, 2])

        eb = pb[-1]
        ec = pc[-1]
        ee1_hist.append(eb.copy())
        ee2_hist.append(ec.copy())
        hb = np.asarray(ee1_hist)
        hc = np.asarray(ee2_hist)
        trace1.set_data(hb[:, 0], hb[:, 1])
        trace1.set_3d_properties(hb[:, 2])
        trace2.set_data(hc[:, 0], hc[:, 1])
        trace2.set_3d_properties(hc[:, 2])

        state = state_at(intervals, float(frame_t[i]))
        status.set_text(
            "t = {:.1f} s   |   CAREPlanner regime: {}   |   playback: {:.1f}x"
            .format(frame_t[i], state, playback_speed))
        return line1, line2, trace1, trace2, status

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_t), interval=1000.0 / fps,
        blit=False, repeat=False)

    mp4 = os.path.join(outdir, "robot_motion_comparison.mp4")
    gif = os.path.join(outdir, "robot_motion_comparison.gif")
    saved = None
    if animation.writers.is_available("ffmpeg"):
        ani.save(mp4, writer="ffmpeg", fps=fps, dpi=120)
        saved = mp4
    else:
        # Pillow is slower but keeps the visualization portable.
        ani.save(gif, writer="pillow", fps=min(fps, 15), dpi=95)
        saved = gif
    plt.close(fig)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run-dir", required=True)
    ap.add_argument("--care-run-dir", required=True)
    ap.add_argument("--case-id", default="case_014")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument(
        "--cases-json",
        default="src/egocentric_arm_planner/config/phase_c2_vbc_cases.json")
    ap.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--playback-speed", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--skip-video", action="store_true")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)

    # Fail early with a useful message on legacy matplotlib installations.
    try:
        test_fig = plt.figure()
        test_fig.add_subplot(111, projection="3d")
        plt.close(test_fig)
    except Exception as exc:
        raise RuntimeError(
            "Matplotlib 3D projection is unavailable even after importing "
            "mpl_toolkits.mplot3d. Check the system matplotlib installation."
        ) from exc

    baseline = os.path.abspath(args.baseline_run_dir)
    care = os.path.abspath(args.care_run_dir)
    outdir = os.path.abspath(
        args.output_dir or os.path.join(
            repo, "outputs/phase_d_visualization", args.case_id + "_ab"))
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(repo, args.cases_json)) as f:
        db = json.load(f)
    case = next((c for c in db["cases"] if c["case_id"] == args.case_id), None)
    if case is None:
        raise KeyError(args.case_id)
    goal = np.asarray(case["goal_position"], dtype=float)

    tb_abs, qb = read_joint_states(os.path.join(baseline, "joint_states.csv"))
    tc_abs, qc = read_joint_states(os.path.join(care, "joint_states.csv"))
    tb, base_start_abs = normalize_motion_time(tb_abs, qb)
    tc, care_start_abs = normalize_motion_time(tc_abs, qc)

    # Drop pre-motion samples so both trajectories start at physical motion t=0.
    mb = tb >= 0.0
    mc = tc >= 0.0
    tb, qb = tb[mb], qb[mb]
    tc, qc = tc[mc], qc[mc]

    helper = load_fk_helper(repo)
    robot = URDF.from_xml_file(os.path.join(repo, args.urdf))
    chain = helper.find_chain_joints(robot, "base_link", "EE_link")

    ee_base = ee_trace(helper, chain, qb)
    ee_care = ee_trace(helper, chain, qc)
    intervals = regime_intervals(
        read_token_csv(os.path.join(care, "regime_summary.csv")),
        care_start_abs, float(tc[-1]))

    ee_png = save_ee_plot(outdir, ee_base, ee_care, goal)
    joints_png = save_joint_plot(outdir, tb, qb, tc, qc, intervals)
    diff_png, diff_inf, diff_l2 = save_difference_plot(
        outdir, tb, qb, tc, qc, intervals)

    # Regime-duration summary is clipped to the measured CARE execution trace.
    regime_s = {}
    for a, b, state in intervals:
        regime_s[state] = regime_s.get(state, 0.0) + max(0.0, b - a)

    summary = {
        "case_id": args.case_id,
        "baseline_definition": (
            "same robot/EE goal/tracker; direct task_trajectory execution; "
            "CAREPlanner confidence/VBC/VisCDF/GCDF/Sparse-SCP/regime modules disabled"),
        "baseline_run_dir": baseline,
        "care_run_dir": care,
        "motion_time_alignment": "first |q-q_initial|_inf >= 1e-3 rad",
        "baseline_recorded_motion_duration_s": float(tb[-1]),
        "care_recorded_motion_duration_s": float(tc[-1]),
        "baseline_ee_path_length_m": path_length(ee_base),
        "care_ee_path_length_m": path_length(ee_care),
        "baseline_joint_path_length_rad_l2_accum": joint_path_length(qb),
        "care_joint_path_length_rad_l2_accum": joint_path_length(qc),
        "care_vs_task_only_joint_deviation_inf_rad": {
            "mean": float(np.mean(diff_inf)),
            "p95": float(np.percentile(diff_inf, 95)),
            "max": float(np.max(diff_inf)),
        },
        "care_vs_task_only_joint_deviation_l2_rad": {
            "mean": float(np.mean(diff_l2)),
            "p95": float(np.percentile(diff_l2, 95)),
            "max": float(np.max(diff_l2)),
        },
        "care_regime_time_s": regime_s,
        "outputs": {
            "ee_path_3d": ee_png,
            "joint_motion_comparison": joints_png,
            "motion_difference": diff_png,
        },
    }

    video = None
    if not args.skip_video:
        video = save_animation(
            outdir, helper, chain, tb, qb, tc, qc, intervals, goal,
            args.playback_speed, args.fps)
        summary["outputs"]["robot_motion_comparison"] = video

    summary_path = os.path.join(outdir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[VISUALIZATION COMPLETE]")
    print("[SUMMARY]", summary_path)
    for key, value in summary["outputs"].items():
        print("[{}] {}".format(key.upper(), value))


if __name__ == "__main__":
    main()
