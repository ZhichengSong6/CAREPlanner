#!/usr/bin/env python3
"""Deterministic production-method tests; no ROS master, model, or robot.

CARE_TIMER_INSTALL_PREFIX optionally tests the installed copies. Extract only
class methods to avoid loading the waypoint model during a lifecycle unit test.
"""
import ast
import os
from pathlib import Path
import threading
import types
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = os.environ.get('CARE_TIMER_INSTALL_PREFIX')


def load_methods(package, filename, classname, methods, namespace):
    if PREFIX:
        base = Path(PREFIX)/('share' if package == 'care_visibility_cdf' else 'lib')/package
        path = base/('scripts' if package == 'care_visibility_cdf' else '')/filename
    else:
        path = ROOT/'src'/package/'scripts'/filename
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert len(selected) == len(methods), (path, methods)
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return type(classname, (), {n.name: namespace[n.name] for n in selected})


class RosError(Exception):
    pass


def message(**kwargs):
    return types.SimpleNamespace(**kwargs)


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.ros = types.SimpleNamespace(
            core=types.SimpleNamespace(is_shutdown_requested=Mock(return_value=False)),
            is_shutdown=Mock(return_value=False), ROSException=RosError,
            logerr_throttle=Mock(), Time=types.SimpleNamespace(now=Mock(return_value=0)))
        ns = dict(rospy=self.ros, np=np, Bool=message, String=message,
                  Float64MultiArray=message)
        rate = load_methods('egocentric_arm_planner', 'joint_velocity_rate_limiter_node.py',
                            'JointVelocityRateLimiter',
                            ['_on_shutdown', '_shutdown_requested', '_timer_cb', '_timer_tick',
                             '_publish_summary'], dict(ns))
        base = load_methods('care_visibility_cdf', 'vbc_deadline_waypoint_node.py',
                            'VbcDeadlineWaypointNode',
                            ['_on_shutdown', '_shutdown_requested', '_run_timer_callback'], dict(ns))
        acq = load_methods('care_visibility_cdf', 'vbc_visibility_acquisition_impl.py',
                           'VisibilityAcquisitionWaypointNode', ['_timer_callback'], dict(ns))
        self.rate = rate()
        self.acq = type('Acquisition', (acq, base), {})()
        for node in [self.rate, self.acq]:
            node._stopping = False
            node.timer = Mock()
            node._lock = threading.Lock()
        self.acq._maybe_generate = Mock()
        self.acq._publish_state = Mock()
        self.acq.acquisition_complete_pub = Mock()
        self.acq._timer_exception_count = 0
        self.acq._acquisition_complete = True
        self.acq._generation_success = True
        self.acq._summary = 'before'

    def test_shutdown_hook_and_late_ticks(self):
        self.rate._timer_tick = Mock()
        for node, callback in [(self.rate, self.rate._timer_cb),
                               (self.acq, self.acq._run_timer_callback)]:
            node._on_shutdown()
            callback(None)
            node._on_shutdown()  # harmless repeated shutdown
            self.assertEqual(node.timer.shutdown.call_count, 2)
        self.rate._timer_tick.assert_not_called()
        self.acq._maybe_generate.assert_not_called()

    def test_requested_before_hook_skips_ticks(self):
        self.ros.core.is_shutdown_requested.return_value = True
        self.rate._timer_tick = Mock()
        self.rate._timer_cb(None)
        self.acq._run_timer_callback(None)
        self.rate._timer_tick.assert_not_called()
        self.acq._maybe_generate.assert_not_called()

    def test_inflight_closed_topic_race_both_nodes(self):
        for node, method, callback in [
                (self.rate, '_timer_tick', self.rate._timer_cb),
                (self.acq, '_publish_state', self.acq._run_timer_callback)]:
            with self.subTest(node=type(node).__name__):
                entered, release = threading.Event(), threading.Event()
                errors = []
                def pending(_event=None):
                    entered.set()
                    if not release.wait(2):
                        raise AssertionError('test barrier timed out')
                    raise RosError('publish() to a closed topic')
                def run():
                    try:
                        callback(None)
                    except Exception as exc:
                        errors.append(exc)
                setattr(node, method, pending)
                worker = threading.Thread(target=run)
                worker.start()
                try:
                    self.assertTrue(entered.wait(2))
                    node._on_shutdown()  # must not join the pending callback
                finally:
                    release.set()
                    worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
        self.assertEqual(self.acq._timer_exception_count, 0)
        self.acq.acquisition_complete_pub.publish.assert_not_called()
        self.ros.logerr_throttle.assert_not_called()

    def test_runtime_rate_ros_error_still_raises(self):
        self.rate._timer_tick = Mock(side_effect=RosError('unexpected closed publisher'))
        with self.assertRaises(RosError):
            self.rate._timer_cb(None)

    def test_non_ros_bug_during_shutdown_not_suppressed(self):
        def broken(_event):
            self.rate._on_shutdown()
            raise ValueError('bug')
        self.rate._timer_tick = broken
        with self.assertRaises(ValueError):
            self.rate._timer_cb(None)

    def test_acquisition_runtime_fail_closed_preserved(self):
        for error in [RosError('publisher failure'), ValueError('projection failure')]:
            with self.subTest(error=type(error).__name__):
                self.acq._maybe_generate.side_effect = error
                self.acq._run_timer_callback(None)
                self.assertFalse(self.acq._acquisition_complete)
                self.assertFalse(self.acq._generation_success)
                self.assertEqual(self.acq._summary, 'c47_timer_exception_fail_closed')
                self.assertFalse(self.acq.acquisition_complete_pub.publish.call_args[0][0].data)
        self.assertEqual(self.acq._timer_exception_count, 2)

    def test_shutdown_after_projection_skips_publication(self):
        self.acq._maybe_generate.side_effect = self.acq._on_shutdown
        self.acq._run_timer_callback(None)
        self.acq._publish_state.assert_not_called()

    def test_rate_stale_ramp_and_limits_unchanged(self):
        node = self.rate
        node.dof, node.rate, node.input_timeout = 7, 100., .1
        node._desired = np.ones(7)
        node._desired_received = node._last_tick = None
        node._last_output = np.full(7, .1)
        node.velocity_limits, node.acceleration_limits = np.full(7, 2.), np.full(7, 4.)
        node._input_count = node._output_count = node._limited_cycle_count = 0
        node._max_accel_ratio = 0.
        node._vbc_hold = node._gcdf_hold = False
        node._held_input_count = 0
        node.pub, node.summary_pub = Mock(), Mock()
        node.stamped_pub = None
        node._timer_cb(None)
        np.testing.assert_allclose(node.pub.publish.call_args[0][0].data, np.full(7, .06))
        self.assertAlmostEqual(node._max_accel_ratio, 1.)
        self.assertEqual(node._output_count, 1)


if __name__ == '__main__':
    unittest.main()
