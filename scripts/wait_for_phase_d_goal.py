#!/usr/bin/env python3
"""Wait until the measured EE pose stably reaches the Phase-D goal.

RUN_SECONDS remains a hard upper bound.  This watcher exits early when the
same pose tolerances used by evaluate_phase_d_run.py are satisfied
continuously for the configured hold duration.
"""

import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from urdf_parser_py.urdf import URDF


JOINTS = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]


def load_fk_helper(repo):
    path = os.path.join(
        repo,
        "src/egocentric_arm_planner/scripts/compute_ee_workspace_bounds.py",
    )
    spec = importlib.util.spec_from_file_location("care_fk_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_rot(q):
    x, y, z, w = map(float, q)
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n <= 0.0:
        raise ValueError("goal quaternion has zero norm")
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*z+y*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def fk_T(helper, chain, qmap):
    T = np.eye(4)
    for joint in chain:
        T = (
            T
            @ helper.get_joint_origin_transform(joint)
            @ helper.joint_motion_transform(
                joint, qmap.get(joint.name, 0.0))
        )
    return T


def pose_error(T, p_goal, R_goal):
    pos = float(np.linalg.norm(T[:3, 3] - p_goal))
    c = (float(np.trace(R_goal.T @ T[:3, :3])) - 1.0) / 2.0
    rot = math.acos(max(-1.0, min(1.0, c)))
    return pos, rot


def write_status(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--urdf", default="src/arm_description/urdf/Arm.urdf")
    ap.add_argument("--joint-topic", default="/care_arm/joint_states")
    ap.add_argument("--timeout-s", type=float, required=True)
    ap.add_argument("--position-tolerance-m", type=float, default=0.02)
    ap.add_argument("--orientation-tolerance-rad", type=float, default=0.20)
    ap.add_argument("--hold-s", type=float, default=0.10)
    ap.add_argument("--settle-velocity-inf-rad-s", type=float, default=0.05)
    ap.add_argument("--settle-timeout-s", type=float, default=1.0)
    ap.add_argument("--post-success-record-s", type=float, default=0.0)
    ap.add_argument("--goal-position", type=float, nargs=3, required=True)
    ap.add_argument("--goal-orientation", type=float, nargs=4, required=True)
    ap.add_argument("--status-json", default="")
    args = ap.parse_args()

    if args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be positive")
    if args.position_tolerance_m <= 0.0:
        raise ValueError("--position-tolerance-m must be positive")
    if args.orientation_tolerance_rad <= 0.0:
        raise ValueError("--orientation-tolerance-rad must be positive")
    if (args.hold_s < 0.0 or args.post_success_record_s < 0.0 or
            args.settle_velocity_inf_rad_s < 0.0 or
            args.settle_timeout_s < 0.0):
        raise ValueError("hold/settle/post-success values must be non-negative")

    repo = os.path.abspath(args.repo)
    helper = load_fk_helper(repo)
    robot = URDF.from_xml_file(os.path.join(repo, args.urdf))
    chain = helper.find_chain_joints(robot, "base_link", "EE_link")
    p_goal = np.array(args.goal_position, dtype=float)
    R_goal = quat_rot(args.goal_orientation)

    rospy.init_node(
        "phase_d_goal_stop_watch",
        anonymous=True,
        disable_signals=True,
    )

    wall_start = time.monotonic()
    inside_since_ros = None
    success_wall_s = None
    success_latched_wall = None
    last_pos = math.nan
    last_rot = math.nan
    last_speed_inf = math.nan
    sample_count = 0
    goal_reached = False
    settled_before_exit = False
    last_log_wall = -math.inf

    print(
        "[GOAL WATCH] max_timeout={:.3f}s pos_tol={:.4f}m "
        "rot_tol={:.4f}rad hold={:.3f}s".format(
            args.timeout_s,
            args.position_tolerance_m,
            args.orientation_tolerance_rad,
            args.hold_s,
        )
    )

    while not rospy.is_shutdown():
        wall_elapsed = time.monotonic() - wall_start
        remaining = args.timeout_s - wall_elapsed
        if remaining <= 0.0:
            break

        try:
            msg = rospy.wait_for_message(
                args.joint_topic,
                JointState,
                timeout=min(0.20, max(0.01, remaining)),
            )
        except rospy.ROSException:
            continue

        if len(msg.name) != len(msg.position):
            continue
        qmap = {
            str(name): float(value)
            for name, value in zip(msg.name, msg.position)
        }
        if not all(name in qmap for name in JOINTS):
            continue

        dqmap = {}
        if len(msg.velocity) == len(msg.name):
            dqmap = {
                str(name): float(value)
                for name, value in zip(msg.name, msg.velocity)
            }
        if all(name in dqmap for name in JOINTS):
            last_speed_inf = max(abs(dqmap[name]) for name in JOINTS)
        else:
            last_speed_inf = math.nan

        sample_count += 1
        T = fk_T(helper, chain, qmap)
        last_pos, last_rot = pose_error(T, p_goal, R_goal)
        inside = (
            last_pos <= args.position_tolerance_m
            and last_rot <= args.orientation_tolerance_rad
        )

        stamp_s = msg.header.stamp.to_sec()
        now_ros = stamp_s if stamp_s > 0.0 else rospy.Time.now().to_sec()

        if success_wall_s is None:
            if inside:
                if inside_since_ros is None:
                    inside_since_ros = now_ros
                if now_ros - inside_since_ros >= args.hold_s:
                    success_wall_s = time.monotonic() - wall_start
                    success_latched_wall = time.monotonic()
                    print(
                        "[GOAL WATCH] benchmark success latched: elapsed={:.3f}s "
                        "position_error={:.6f}m orientation_error={:.6f}rad "
                        "speed_inf={:.6f}rad/s".format(
                            success_wall_s, last_pos, last_rot,
                            last_speed_inf
                        )
                    )
            else:
                inside_since_ros = None
        else:
            speed_ok = (
                math.isfinite(last_speed_inf)
                and last_speed_inf <= args.settle_velocity_inf_rad_s
            )
            if inside and speed_ok:
                goal_reached = True
                settled_before_exit = True
                print(
                    "[GOAL WATCH] settled after success: speed_inf="
                    "{:.6f}rad/s".format(last_speed_inf)
                )
                break
            if (success_latched_wall is not None and
                    time.monotonic() - success_latched_wall >=
                    args.settle_timeout_s):
                goal_reached = True
                print(
                    "[GOAL WATCH] success settle timeout reached; "
                    "ending without extending task metric"
                )
                break

        now_wall = time.monotonic()
        if now_wall - last_log_wall >= 2.0:
            last_log_wall = now_wall
            print(
                "[GOAL WATCH] elapsed={:.2f}s position_error={:.4f}m "
                "orientation_error={:.4f}rad inside={}".format(
                    now_wall - wall_start,
                    last_pos,
                    last_rot,
                    int(inside),
                )
            )

    if goal_reached and args.post_success_record_s > 0.0:
        remaining = max(0.0, args.timeout_s - (time.monotonic() - wall_start))
        time.sleep(min(args.post_success_record_s, remaining))

    elapsed = time.monotonic() - wall_start
    reason = "goal_tolerance_stable" if goal_reached else "max_timeout"
    status = {
        "goal_reached": bool(goal_reached),
        "reason": reason,
        "elapsed_wall_s": elapsed,
        "success_elapsed_wall_s": success_wall_s,
        "max_timeout_s": args.timeout_s,
        "position_tolerance_m": args.position_tolerance_m,
        "orientation_tolerance_rad": args.orientation_tolerance_rad,
        "required_hold_s": args.hold_s,
        "settle_velocity_inf_rad_s": args.settle_velocity_inf_rad_s,
        "settle_timeout_s": args.settle_timeout_s,
        "settled_before_exit": bool(settled_before_exit),
        "post_success_record_s": args.post_success_record_s,
        "last_speed_inf_rad_s": (
            last_speed_inf if math.isfinite(last_speed_inf) else None),
        "last_position_error_m": (
            last_pos if math.isfinite(last_pos) else None),
        "last_orientation_error_rad": (
            last_rot if math.isfinite(last_rot) else None),
        "sample_count": sample_count,
    }
    write_status(args.status_json, status)

    if goal_reached:
        print("[GOAL WATCH] early-stop success; ending this case")
    else:
        print("[GOAL WATCH] maximum runtime reached; ending this case")


if __name__ == "__main__":
    main()
