#!/usr/bin/env python3
"""Move the Gazebo arm to the fixed Case-026 S7 pose and test real visibility.

No planner, GCDF, VBC, NCDF, or visibility obligation is involved.
The arm is moved with the normal JointGroupVelocityController from q=0 so
Gazebo link/sensor poses remain physically updated (no -J / SetModelConfiguration).

After the arm settles, query the real confidence-map service repeatedly for the
fixed target and report whether current_visibility becomes positive.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import rospy

from care_confidence_map.srv import QueryConfidence, QueryConfidenceRequest
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "wrist_joint1", "wrist_joint2", "wrist_joint3",
]

S7_Q = np.asarray([
    -0.8144537806510925,
     0.5605988502502441,
     1.0171856880187988,
    -2.180335760116577,
     0.12358028441667557,
    -1.7687498331069946,
     1.1331826448440552,
], dtype=np.float64)

TARGET = np.asarray([
    0.10000000149011612,
    0.05000000074505806,
    0.15000000596046448,
], dtype=np.float64)


class StateCache:
    def __init__(self):
        self.q = None
        self.stamp = None

    def cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        if not all(n in idx for n in JOINT_NAMES):
            return
        self.q = np.asarray(
            [float(msg.position[idx[n]]) for n in JOINT_NAMES],
            dtype=np.float64)
        self.stamp = time.monotonic()


def publish_velocity(pub, values):
    msg = Float64MultiArray()
    msg.data = [float(v) for v in values]
    pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", required=True)
    ap.add_argument(
        "--joint-topic", default="/care_arm/joint_states")
    ap.add_argument(
        "--command-topic",
        default="/care_arm/arm_group_velocity_controller/command")
    ap.add_argument(
        "--query-service", default="/care_planner/confidence_map/query")
    ap.add_argument("--kp", type=float, default=0.9)
    ap.add_argument("--max-velocity", type=float, default=0.35)
    ap.add_argument("--move-timeout", type=float, default=20.0)
    ap.add_argument("--settle-tolerance", type=float, default=0.01)
    ap.add_argument("--settle-seconds", type=float, default=1.0)
    ap.add_argument("--query-seconds", type=float, default=5.0)
    ap.add_argument("--query-rate", type=float, default=20.0)
    args = ap.parse_args()

    rospy.init_node("phase_e_s7_static_visibility_test", anonymous=True)

    state = StateCache()
    rospy.Subscriber(args.joint_topic, JointState, state.cb, queue_size=5)
    pub = rospy.Publisher(
        args.command_topic, Float64MultiArray, queue_size=1)

    deadline = time.monotonic() + 10.0
    while not rospy.is_shutdown() and state.q is None:
        if time.monotonic() > deadline:
            raise RuntimeError("timed out waiting for joint state")
        time.sleep(0.02)

    q_start = state.q.copy()
    initial_error = float(np.max(np.abs(S7_Q - q_start)))
    print("[S7 STATIC] initial q =", q_start.tolist(), flush=True)
    print("[S7 STATIC] target  q =", S7_Q.tolist(), flush=True)
    print(
        "[S7 STATIC] initial q_dist_inf = %.6f rad" % initial_error,
        flush=True)

    # Slow P velocity servo.  The purpose is only to establish a physically
    # consistent Gazebo pose; this is not a planning experiment.
    rate = rospy.Rate(50.0)
    move_start = time.monotonic()
    reached_at = None
    while not rospy.is_shutdown():
        q = state.q.copy()
        err = S7_Q - q
        err_inf = float(np.max(np.abs(err)))
        if err_inf <= args.settle_tolerance:
            if reached_at is None:
                reached_at = time.monotonic()
            if time.monotonic() - reached_at >= args.settle_seconds:
                break
        else:
            reached_at = None

        if time.monotonic() - move_start > args.move_timeout:
            publish_velocity(pub, np.zeros(7))
            raise RuntimeError(
                "failed to reach S7 within %.1fs; q_dist_inf=%.6f"
                % (args.move_timeout, err_inf))

        cmd = np.clip(args.kp * err, -args.max_velocity, args.max_velocity)
        publish_velocity(pub, cmd)
        rate.sleep()

    # Stop and let Gazebo/sensors settle.
    for _ in range(50):
        publish_velocity(pub, np.zeros(7))
        rate.sleep()

    q_final = state.q.copy()
    final_error = float(np.max(np.abs(S7_Q - q_final)))
    print("[S7 STATIC] reached q =", q_final.tolist(), flush=True)
    print(
        "[S7 STATIC] final q_dist_inf = %.6f rad" % final_error,
        flush=True)

    rospy.wait_for_service(args.query_service, timeout=10.0)
    query = rospy.ServiceProxy(
        args.query_service, QueryConfidence, persistent=False)

    rows = []
    n_queries = max(1, int(round(args.query_seconds * args.query_rate)))
    dt = 1.0 / max(args.query_rate, 1e-6)
    for i in range(n_queries):
        req = QueryConfidenceRequest()
        req.points = [
            Point(
                x=float(TARGET[0]),
                y=float(TARGET[1]),
                z=float(TARGET[2]))
        ]
        try:
            res = query(req)
            confidence = (
                float(res.confidence[0])
                if len(res.confidence) == 1 else math.nan)
            current_visibility = (
                float(res.current_visibility[0])
                if len(res.current_visibility) == 1 else math.nan)
            inside = (
                bool(res.inside_map[0])
                if len(res.inside_map) == 1 else False)
            status = "ok"
        except Exception as exc:
            confidence = math.nan
            current_visibility = math.nan
            inside = False
            status = "error:" + str(exc)

        rows.append({
            "index": i,
            "time_s": float(rospy.Time.now().to_sec()),
            "status": status,
            "inside_map": bool(inside),
            "confidence": confidence,
            "current_visibility": current_visibility,
        })
        if i == 0 or i == n_queries - 1 or current_visibility > 0.0:
            print(
                "[QUERY %03d] inside=%d confidence=%s current_visibility=%s"
                % (
                    i,
                    int(inside),
                    ("nan" if not math.isfinite(confidence)
                     else "%.6f" % confidence),
                    ("nan" if not math.isfinite(current_visibility)
                     else "%.6f" % current_visibility)),
                flush=True)
        time.sleep(dt)

    valid_vis = [
        r["current_visibility"] for r in rows
        if r["status"] == "ok"
        and math.isfinite(r["current_visibility"])
    ]
    valid_conf = [
        r["confidence"] for r in rows
        if r["status"] == "ok"
        and math.isfinite(r["confidence"])
    ]
    positive = [v for v in valid_vis if v > 0.0]

    verdict = (
        "S7_STATIC_ACTUAL_VISIBILITY_POSITIVE"
        if positive else
        "S7_STATIC_ACTUAL_VISIBILITY_ZERO"
    )

    report = {
        "diagnostic": "phase_e_case026_s7_static_visibility",
        "planner": "off",
        "gcdf": "off",
        "vbc": "off",
        "ncdf": "off",
        "target_xyz": TARGET.tolist(),
        "target_q_s7": S7_Q.tolist(),
        "initial_q": q_start.tolist(),
        "final_q": q_final.tolist(),
        "initial_q_dist_inf": initial_error,
        "final_q_dist_inf": final_error,
        "query_count": len(rows),
        "positive_current_visibility_count": len(positive),
        "max_current_visibility": max(valid_vis) if valid_vis else None,
        "max_confidence": max(valid_conf) if valid_conf else None,
        "final_confidence": valid_conf[-1] if valid_conf else None,
        "verdict": verdict,
        "queries": rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2, allow_nan=True)

    print("")
    print("================ S7 STATIC VISIBILITY =================")
    print("final q_dist_inf          : %.6f" % final_error)
    print("positive visibility rows  :", len(positive), "/", len(rows))
    print("max current_visibility    :", report["max_current_visibility"])
    print("max confidence            :", report["max_confidence"])
    print("VERDICT                   :", verdict)
    print("[OUTPUT]", os.path.abspath(args.output_json))
    print("=======================================================")

    publish_velocity(pub, np.zeros(7))


if __name__ == "__main__":
    main()
