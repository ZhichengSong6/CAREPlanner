#!/usr/bin/env python3
"""Minimal ROS service smoke test for the collision CDF server."""

import argparse

import rospy
from geometry_msgs.msg import Point

from care_collision_cdf.srv import QueryCollisionCDF, QueryCollisionCDFRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service", default="/care_planner/collision_cdf/query"
    )
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("query_collision_cdf_smoke", anonymous=True)
    rospy.wait_for_service(args.service, timeout=10.0)
    client = rospy.ServiceProxy(args.service, QueryCollisionCDF)

    req = QueryCollisionCDFRequest()
    # Non-empty generic test batch. This is an interface smoke test, not a
    # semantic collision-accuracy test.
    req.points = [
        Point(x=0.40, y=0.00, z=0.40),
        Point(x=0.45, y=0.10, z=0.45),
        Point(x=0.45, y=-0.10, z=0.45),
    ]
    req.num_q = 2
    req.q_flat = [0.0] * 14
    res = client(req)
    print("success:", res.success)
    print("message:", res.message)
    print("distance:", list(res.min_distance))
    print("gradient_flat_len:", len(res.min_gradient_flat))
    print("argmin_point_index:", list(res.argmin_point_index))
    print("inference_ms:", res.inference_ms)
    if not res.success:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
