#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from care_confidence_map.srv import QueryConfidence, QueryConfidenceRequest


def make_point(x, y, z):
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p


def print_result(points, response):
    print("")
    print("========== QueryConfidence Result ==========")

    for i, p in enumerate(points):
        confidence = response.confidence[i]
        current_visibility = response.current_visibility[i]
        inside_map = response.inside_map[i]

        print(
            "[{:02d}] p = [{:+.3f}, {:+.3f}, {:+.3f}] | "
            "inside_map = {} | "
            "confidence = {:.3f} | "
            "current_visibility = {:.3f}".format(
                i,
                p.x,
                p.y,
                p.z,
                int(inside_map),
                confidence,
                current_visibility,
            )
        )

    print("============================================")
    print("")


def main():
    rospy.init_node("test_query_confidence", anonymous=True)

    service_name = rospy.get_param(
        "~service_name",
        "/care_planner/confidence_map/query"
    )

    rospy.loginfo("Waiting for service: %s", service_name)
    rospy.wait_for_service(service_name)
    rospy.loginfo("Service is available.")

    query_confidence = rospy.ServiceProxy(service_name, QueryConfidence)

    # These points are all in confidence_map frame.
    # Current setup uses base_link as confidence_map frame.
    #
    # Some points may or may not be currently visible depending on the
    # current arm pose and ToF sensor directions.
    #
    # Map bounds are currently:
    #   x [-0.9, 0.9]
    #   y [-0.9, 0.9]
    #   z [ 0.0, 1.1]
    points = [
        # Near map center / robot workspace.
        make_point(0.00, 0.00, 0.30),
        make_point(0.20, 0.00, 0.30),
        make_point(0.30, 0.00, 0.40),
        make_point(0.40, 0.00, 0.50),

        # Other valid in-map points.
        make_point(0.00, 0.30, 0.40),
        make_point(0.00, -0.30, 0.40),
        make_point(0.50, 0.20, 0.60),
        make_point(-0.50, -0.20, 0.60),

        # Boundary-ish in-map points.
        make_point(0.90, 0.00, 0.50),
        make_point(-0.90, 0.00, 0.50),
        make_point(0.00, 0.90, 0.50),
        make_point(0.00, -0.90, 0.50),
        make_point(0.00, 0.00, 1.10),

        # Clearly outside map.
        make_point(1.50, 0.00, 0.50),
        make_point(0.00, 1.50, 0.50),
        make_point(0.00, 0.00, -0.20),
        make_point(0.00, 0.00, 1.50),
    ]

    req = QueryConfidenceRequest()
    req.points = points

    try:
        response = query_confidence(req)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", str(e))
        return

    if not (
        len(response.confidence) == len(points)
        and len(response.current_visibility) == len(points)
        and len(response.inside_map) == len(points)
    ):
        rospy.logerr(
            "Response size mismatch: points=%d, confidence=%d, visibility=%d, inside=%d",
            len(points),
            len(response.confidence),
            len(response.current_visibility),
            len(response.inside_map),
        )
        return

    print_result(points, response)

    # Simple sanity checks.
    outside_indices = [13, 14, 15, 16]
    outside_ok = True
    for idx in outside_indices:
        if response.inside_map[idx] != 0:
            outside_ok = False
            rospy.logwarn(
                "Expected point %d to be outside map, but inside_map=%d",
                idx,
                response.inside_map[idx],
            )

    inside_indices = list(range(13))
    inside_ok = True
    for idx in inside_indices:
        if response.inside_map[idx] != 1:
            inside_ok = False
            rospy.logwarn(
                "Expected point %d to be inside map, but inside_map=%d",
                idx,
                response.inside_map[idx],
            )

    if inside_ok and outside_ok:
        rospy.loginfo("Basic inside/outside sanity check passed.")
    else:
        rospy.logwarn("Basic inside/outside sanity check failed.")

    visible_count = sum(1 for v in response.current_visibility if v > 0.5)
    confident_count = sum(1 for c in response.confidence if c > 1e-4)

    rospy.loginfo(
        "Queried %d points: visible_now=%d, confident=%d",
        len(points),
        visible_count,
        confident_count,
    )


if __name__ == "__main__":
    main()