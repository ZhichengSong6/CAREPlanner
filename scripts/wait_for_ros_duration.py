#!/usr/bin/env python3
"""Wait for a fixed amount of ROS/Gazebo simulation time.

This helper intentionally uses rospy.Time.now() with /use_sim_time enabled so
benchmark duration is independent of host load / Gazebo real-time factor.
"""

import argparse
import time

import rospy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, required=True)
    ap.add_argument("--stall-timeout-wall-s", type=float, default=60.0)
    ap.add_argument("--progress-period-s", type=float, default=5.0)
    args = ap.parse_args()

    duration = float(args.duration_s)
    if duration < 0.0:
        raise SystemExit("--duration-s must be non-negative")

    rospy.init_node(
        "careplanner_fixed_sim_runtime_wait",
        anonymous=True,
        disable_signals=True,
    )

    if not bool(rospy.get_param("/use_sim_time", False)):
        raise SystemExit(
            "[ERROR] fixed sim-time runtime requested but /use_sim_time is false"
        )

    start_wait_wall = time.monotonic()
    start = rospy.Time.now().to_sec()
    while start <= 0.0 and not rospy.is_shutdown():
        if time.monotonic() - start_wait_wall > args.stall_timeout_wall_s:
            raise SystemExit("[ERROR] ROS /clock did not start")
        time.sleep(0.01)
        start = rospy.Time.now().to_sec()

    last_sim = start
    last_progress_wall = time.monotonic()
    next_report = max(0.0, float(args.progress_period_s))

    print(
        "[SIM TIME WAIT] start={:.6f}s duration={:.3f}s".format(
            start, duration
        ),
        flush=True,
    )

    while not rospy.is_shutdown():
        now = rospy.Time.now().to_sec()
        if now + 1e-9 < last_sim:
            raise SystemExit(
                "[ERROR] ROS simulation clock moved backwards "
                "({:.6f} -> {:.6f})".format(last_sim, now)
            )

        if now > last_sim + 1e-9:
            last_sim = now
            last_progress_wall = time.monotonic()
        elif time.monotonic() - last_progress_wall > args.stall_timeout_wall_s:
            raise SystemExit(
                "[ERROR] ROS simulation clock stalled for {:.1f}s wall time".format(
                    args.stall_timeout_wall_s
                )
            )

        elapsed = now - start
        if next_report > 0.0 and elapsed + 1e-9 >= next_report:
            print(
                "[SIM TIME WAIT] elapsed={:.3f}/{:.3f}s".format(
                    elapsed, duration
                ),
                flush=True,
            )
            next_report += max(0.1, float(args.progress_period_s))

        if elapsed + 1e-9 >= duration:
            print(
                "[SIM TIME WAIT] complete elapsed={:.3f}s".format(elapsed),
                flush=True,
            )
            return

        # Wall sleep is intentional: rospy.sleep() itself follows sim time and
        # would make it harder to detect a stalled /clock.
        time.sleep(0.01)

    raise SystemExit("[ERROR] ROS shutdown before fixed simulation duration completed")


if __name__ == "__main__":
    main()
