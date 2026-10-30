#!/usr/bin/env python3
"""Small synthetic tests: no runtime, checkpoint, ROS, or GPU dependencies."""
import unittest

from diagnose_phase_e_vbc_rejection_loops import (
    endpoint_geometry, in_window, numeric_range, point_key,
)


class RejectionForensicsTests(unittest.TestCase):
    def test_exact_solver_precision_not_nearest_point(self):
        self.assertEqual(point_key([.1, -.2, .8]),
                         point_key([.10000000149011612, -.20000000298023224, .800000011920929]))
        self.assertNotEqual(point_key([.1, -.2, .8]), point_key([.1, -.2, .75]))

    def test_window_excludes_teardown_and_pre_schedule(self):
        rows = [dict(ros_s=t) for t in [9, 10, 11, 12, 13]]
        self.assertEqual([r['ros_s'] for r in in_window(rows, 10, 12)], [10, 11, 12])

    def test_unknown_range_not_reported_as_zero(self):
        self.assertIsNone(numeric_range([dict(value='nan')], 'value'))
        self.assertEqual(numeric_range([dict(value='1'), dict(value='3'), dict(value='nan')], 'value'), [1, 3])

    def test_union_coverage_does_not_require_one_sensor_for_every_point(self):
        class Runtime:
            def _candidate_geometry(self, points, q, sensor):
                return dict(accepted=sensor == int(points[0][0]))
        result = endpoint_geometry(Runtime(), [[0, 0, 0], [4, 0, 0]], [0]*7)
        self.assertTrue(result['every_point_has_clear_sensor'])
        self.assertEqual([r['passing_sensors'] for r in result['points']], [[0], [4]])
        self.assertEqual(len(result['points'][0]['sensors']), 8)

    def test_one_uncovered_point_fails_union_coverage(self):
        class Runtime:
            def _candidate_geometry(self, points, q, sensor):
                return dict(accepted=points[0][0] == 0 and sensor == 0)
        self.assertFalse(endpoint_geometry(Runtime(), [[0, 0, 0], [1, 0, 0]], [0]*7)[
            'every_point_has_clear_sensor'])


if __name__ == '__main__':
    unittest.main()
