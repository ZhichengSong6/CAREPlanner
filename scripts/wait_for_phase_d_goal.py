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
from phase_e_evaluation_common import quat_rot, pose_error, seconds_to_ns


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
    ap.add_argument("--arm-token", default="")
    ap.add_argument("--required-recorder", nargs="*", default=[])
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

    arm_pub = None
    if args.arm_token:
        from phase_e_benchmark_arm import prepare_arm
        try:
            arm_pub = prepare_arm(args.required_recorder)
        except Exception as exc:
            write_status(args.status_json, dict(goal_reached=False,
                benchmark_window={"valid": False}, startup_error=str(exc)))
            raise

    wall_start = time.monotonic()
    start_ros_ns = rospy.Time.now().to_nsec()
    required_hold_ns = seconds_to_ns(args.hold_s, duration=True)
    window = {"schema_version": 3, "valid": False,
              "start_wall_unix_s": time.time(), "start_monotonic_s": wall_start,
              "start_ros_s": start_ros_ns / 1e9, "start_ros_ns": start_ros_ns,
              "watchdog_start_wall_unix_s": time.time(),
              "watchdog_start_monotonic_s": wall_start,
              "joint_names": JOINTS, "first_measured_q": None,
              "end_ros_s": None, "end_ros_ns": None, "end_measured_q": None}
    window["protocol"] = "pre_goal_arm_v1" if arm_pub is not None else "legacy_post_gate_v2"
    window["required_recorders"] = list(args.required_recorder)
    previous_stamp = None
    inside_since_ros_ns = None
    success_hold_start_ns = None
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

        # A message returned just after the wall deadline must not create a
        # late success or extend the benchmark window.
        if time.monotonic() - wall_start >= args.timeout_s:
            break

        if len(msg.name) != len(msg.position):
            continue
        qmap = {
            str(name): float(value)
            for name, value in zip(msg.name, msg.position)
        }
        if not all(name in qmap and math.isfinite(qmap[name]) for name in JOINTS):
            continue

        stamp_ns = msg.header.stamp.to_nsec()
        now_ros_ns = stamp_ns if stamp_ns > 0 else rospy.Time.now().to_nsec()
        now_ros = now_ros_ns / 1e9  # display/legacy fields only
        if now_ros_ns < window["start_ros_ns"]:
            continue
        if previous_stamp is not None and now_ros_ns <= previous_stamp:
            inside_since_ros_ns = None
            continue
        previous_stamp = now_ros_ns
        if success_wall_s is None:
            measured_q = [qmap[name] for name in JOINTS]
            if window["first_measured_q"] is None:
                # /clock is often still zero immediately after init_node.
                # Anchor all three clocks to the first accepted measured
                # sample; the watchdog still starts at the original wall time.
                window["start_ros_s"] = now_ros
                window["start_ros_ns"] = now_ros_ns
                window["start_wall_unix_s"] = time.time()
                window["start_monotonic_s"] = time.monotonic()
                window["first_measured_q"] = measured_q
                window["first_sample_ros_s"] = now_ros
                window["first_sample_ros_ns"] = now_ros_ns
                if arm_pub is not None:
                    # Capture measured q BEFORE the broker sends the first goal.
                    # Planning/gate wait is inside the unchanged task budget.
                    wall_start = time.monotonic()
                    window["watchdog_start_monotonic_s"] = wall_start
                    window["watchdog_start_wall_unix_s"] = time.time()
                    window["arm_token"] = args.arm_token
                    arm_pub.publish(args.arm_token)
                    print("[GOAL WATCH] armed after measured q and recorder readiness")
            window.update(valid=True, end_ros_s=now_ros, end_ros_ns=now_ros_ns, end_measured_q=measured_q,
                          end_wall_unix_s=time.time(), end_monotonic_s=time.monotonic())

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

        if success_wall_s is None:
            if inside:
                if inside_since_ros_ns is None:
                    inside_since_ros_ns = now_ros_ns
                if now_ros_ns - inside_since_ros_ns >= required_hold_ns:
                    success_hold_start_ns = inside_since_ros_ns
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
                inside_since_ros_ns = None
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
    # Success is latched independently of settling, including a success very
    # close to the hard watchdog limit. Settling never extends task metrics.
    goal_reached = success_wall_s is not None
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
        "required_hold_ns": required_hold_ns,
        "goal_hold_start_ros_ns": success_hold_start_ns,
        "goal_hold_end_ros_ns": window['end_ros_ns'] if success_wall_s is not None else None,
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
        "benchmark_window": window,
    }
    write_status(args.status_json, status)

    if goal_reached:
        print("[GOAL WATCH] early-stop success; ending this case")
    else:
        print("[GOAL WATCH] maximum runtime reached; ending this case")


if __name__ == "__main__":
    main()
