#!/usr/bin/env python3
"""Offline exact-solid and observational logger regression; no ROS master."""
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_confidence_map/scripts'))
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Transform
from tf.transformations import quaternion_from_euler, quaternion_matrix
from primitive_point_geometry import Primitive, load_primitives, rotation_rpy, target_in_link
from executed_vbc_logger_node import ExecutedVBCLogger

URDF = ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'


class GeometryTests(unittest.TestCase):
    def test_legacy_yaml_loader_remains_available(self):
        logger = ExecutedVBCLogger.__new__(ExecutedVBCLogger)
        logger.ignored_links = {'base_link', 'link1'}
        samples = logger._load_samples(ROOT/'src/care_confidence_map/config/body_samples.yaml')
        self.assertTrue(samples)
        self.assertTrue(all(samples.values()))

    def test_actual_model_counts(self):
        self.assertEqual(sum(map(len, load_primitives(URDF).values())), 26)
        model = load_primitives(URDF, ['base_link', 'link1'])
        self.assertEqual(sum(map(len, model.values())), 21)
        self.assertEqual(len(model), 6)

    def test_exact_signed_distance(self):
        identity = rotation_rpy((0, 0, 0))
        box = Primitive('x', 0, 'box', (0, 0, 0), identity, (2, 2, 2))
        cylinder = Primitive('x', 1, 'cylinder', (0, 0, 0), identity, (1, 2))
        sphere = Primitive('x', 2, 'sphere', (0, 0, 0), identity, (1,))
        for p in [box, cylinder, sphere]:
            self.assertAlmostEqual(p.distance((0, 0, 0)), -1)
            self.assertAlmostEqual(p.distance((1, 0, 0)), 0)
        self.assertAlmostEqual(box.distance((2, 2, 2)), math.sqrt(3))
        self.assertAlmostEqual(cylinder.distance((2, 0, 2)), math.sqrt(2))
        self.assertAlmostEqual(sphere.distance((1, 1, 1)), math.sqrt(3)-1)

    def test_rotated_origin_and_tf_reference(self):
        rng = np.random.RandomState(71)
        for kind, dims in [('box', (.2, .4, .6)), ('cylinder', (.15, .3)), ('sphere', (.2,))]:
            for _ in range(100):
                rpy, center, point = rng.randn(3), rng.randn(3), rng.randn(3)
                rotation = quaternion_matrix(quaternion_from_euler(*rpy))[:3, :3]
                primitive = Primitive('x', 0, kind, tuple(center), rotation_rpy(rpy), dims)
                local = rotation.T.dot(point-center)
                if kind == 'box':
                    gap = np.abs(local)-np.array(dims)/2
                elif kind == 'cylinder':
                    gap = np.array([np.linalg.norm(local[:2])-dims[0], abs(local[2])-dims[1]/2])
                else:
                    gap = np.array([np.linalg.norm(local)-dims[0]])
                expected = np.linalg.norm(np.maximum(gap, 0))+min(np.max(gap), 0)
                self.assertAlmostEqual(primitive.distance(point), expected, places=12)
                transform = Transform()
                transform.translation.x, transform.translation.y, transform.translation.z = center
                transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w = quaternion_from_euler(*rpy)
                np.testing.assert_allclose(target_in_link(point, transform), local, atol=1e-12)

    def test_malformed_models_fail_without_partial_geometry(self):
        shapes = ['<mesh filename="bad.stl"/>', '<box size="1 0 1"/>', '<sphere radius="nan"/>',
                  '<cylinder radius="1" length="-1"/>', '<box size="1 1"/>', '<sphere radius="1"/><box size="1 1 1"/>']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'fixture.urdf'
            for shape in shapes:
                path.write_text('<robot name="test"><link name="a"><collision><geometry><sphere radius="1"/></geometry></collision>'
                                '<collision><geometry>'+shape+'</geometry></collision></link></robot>')
                with self.assertRaises((ValueError, KeyError)):
                    load_primitives(path)
            path.write_text('<robot><link name="a"/><link name="a"/></robot>')
            with self.assertRaises(ValueError):
                load_primitives(path)
            with self.assertRaises(ValueError):
                load_primitives(URDF, list(load_primitives(URDF)))

    def test_invalid_tf_and_target(self):
        transform = Transform()
        with self.assertRaises(ValueError):
            target_in_link((0, 0, 0), transform)
        transform.rotation.w = 1
        with self.assertRaises(ValueError):
            target_in_link((math.nan, 0, 0), transform)


class LoggerTests(unittest.TestCase):
    def test_inflight_old_epoch_cannot_latch(self):
        logger = ExecutedVBCLogger.__new__(ExecutedVBCLogger)
        logger._lock = threading.Lock(); logger._target = PointStamped(); logger._target_time = 1.; logger._target_epoch = 1
        logger._sequence = 0; logger._target_in_base = lambda p: p
        logger._compute_min_clearance = lambda p: (-1., 'link', 0)
        def delayed_query(_):
            logger._target_epoch += 1
            return 1., 1., 1
        logger._query_visibility = delayed_query
        with patch.object(rospy.Time, 'now', return_value=rospy.Time(2)):
            logger._timer_callback(None)
        self.assertEqual(logger._sequence, 0)

    def test_frame_change_resets_epoch(self):
        logger = ExecutedVBCLogger.__new__(ExecutedVBCLogger)
        logger._lock = threading.Lock(); logger._target = PointStamped(); logger._target.header.frame_id = 'base'
        logger._target_xyz = (0., 0., 0.); logger._target_epoch = 4; logger.target_change_tolerance = 1e-4
        msg = PointStamped(); msg.header.frame_id = 'other'
        with patch.object(rospy.Time, 'now', return_value=rospy.Time(2)), patch.object(rospy, 'logwarn'):
            logger._target_callback(msg)
        self.assertEqual(logger._target_epoch, 5)
        self.assertIsNone(logger._sweep_time); self.assertIsNone(logger._see_time)

    def test_invalid_confidence_rejected(self):
        logger = ExecutedVBCLogger.__new__(ExecutedVBCLogger)
        logger.query_service = 'unused'; logger.query_timeout = .01
        for c, v, inside in [(math.nan, 1., 1), (1., math.inf, 1), (1., 1.1, 1), (1., 1., 2)]:
            logger.query_client = Mock(return_value=Mock(confidence=[c], current_visibility=[v], inside_map=[inside]))
            with patch.object(rospy, 'wait_for_service'):
                self.assertIsNone(logger._query_visibility(PointStamped()))


if __name__ == '__main__':
    unittest.main()
