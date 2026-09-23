#!/usr/bin/env python3
"""Killable confidence-service fixture for transport failure/restart tests only."""
import rospy
from care_confidence_map.srv import QueryConfidence, QueryConfidenceResponse


def query(req):
    mode = rospy.get_param('/primitive_test/mode', 'invalid')
    if mode == 'error':
        raise rospy.ServiceException('deliberate transport-test service failure')
    if mode == 'invalid':
        return QueryConfidenceResponse([], [], [])
    n = len(req.points)
    return QueryConfidenceResponse([1. if mode == 'known' else 0.]*n, [0.]*n, [1]*n)


if __name__ == '__main__':
    rospy.init_node('primitive_test_confidence_server')
    service = rospy.Service('/primitive_test/query', QueryConfidence, query)
    rospy.spin()
