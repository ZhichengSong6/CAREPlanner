#!/usr/bin/env python3
"""Exercise production limiter callbacks with delayed commands and hold races."""
import threading
import types
import unittest
from unittest.mock import Mock

import numpy as np

from test_phase_e_shutdown_timers import load_methods, message


class Stamp(float):
    def __sub__(self, other):
        return Stamp(float(self)-float(other))

    def to_sec(self):
        return float(self)


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.clock = 1.
        ros = types.SimpleNamespace(
            Time=types.SimpleNamespace(now=lambda:Stamp(self.clock)),
            core=types.SimpleNamespace(is_shutdown_requested=lambda:False),
            is_shutdown=lambda:False, logwarn_throttle=Mock())
        cls=load_methods('egocentric_arm_planner','joint_velocity_rate_limiter_node.py',
            'JointVelocityRateLimiter', ['_set_hold','_vbc_hold_cb','_gcdf_hold_cb',
            '_desired_cb','_timer_tick','_publish_summary','_shutdown_requested'],
            dict(np=np, rospy=ros, Float64MultiArray=message, String=message,
                 JointState=lambda:message(header=message())))
        n=self.node=cls()
        n._lock=threading.Lock(); n._stopping=False
        n.dof=7; n.rate=100.; n.input_timeout=.1
        n._desired=np.zeros(7); n._desired_received=None; n._last_tick=None
        n._last_output=np.array([.12,-.12,0,0,0,0,0])
        n.velocity_limits=np.full(7,2.); n.acceleration_limits=np.full(7,3.)
        n._vbc_hold=n._gcdf_hold=False
        n._input_count=n._output_count=n._limited_cycle_count=n._held_input_count=0
        n._max_accel_ratio=0.
        n.pub=Mock(); n.summary_pub=Mock(); n.stamped_pub=None
        n.joint_names=['j'+str(i) for i in range(7)]

    def tick(self):
        self.clock += .01
        self.node._timer_tick(None)
        return np.array(self.node.pub.publish.call_args[0][0].data)

    def test_hold_brakes_despite_repeated_delayed_nonzero_commands(self):
        n=self.node
        n._desired_cb(message(data=[2.]*7))
        n._vbc_hold_cb(message(data=True))
        previous=n._last_output.copy()
        for _ in range(5):
            n._desired_cb(message(data=[2.]*7))
            current=self.tick()
            self.assertTrue(np.all(np.abs(current)<=np.abs(previous)+1e-12))
            self.assertTrue(np.all(np.abs(current-previous)<=.030000001))
            previous=current
        np.testing.assert_allclose(current,np.zeros(7),atol=1e-12)
        self.assertEqual(n._held_input_count,5)

    def test_both_holds_must_release_and_require_new_desired(self):
        n=self.node
        n._last_output=np.zeros(7)
        n._vbc_hold_cb(message(data=True)); n._gcdf_hold_cb(message(data=True))
        n._vbc_hold_cb(message(data=False))
        n._desired_cb(message(data=[2.]*7))
        np.testing.assert_array_equal(self.tick(),np.zeros(7))
        n._gcdf_hold_cb(message(data=False))
        np.testing.assert_array_equal(self.tick(),np.zeros(7))
        n._desired_cb(message(data=[1.]*7))
        np.testing.assert_allclose(self.tick(),np.full(7,.03))

    def test_repeated_false_does_not_erase_a_fresh_command(self):
        n=self.node; n._last_output=np.zeros(7)
        n._desired_cb(message(data=[1.]*7))
        n._gcdf_hold_cb(message(data=False))
        np.testing.assert_allclose(self.tick(),np.full(7,.03))

    def test_pre_hold_command_is_not_reused_after_release(self):
        n=self.node; n._last_output=np.zeros(7)
        n._desired_cb(message(data=[1.]*7))
        n._vbc_hold_cb(message(data=True)); n._vbc_hold_cb(message(data=False))
        np.testing.assert_array_equal(self.tick(),np.zeros(7))

    def test_stale_request_still_ramps_to_zero(self):
        n=self.node
        n._desired_cb(message(data=[2.]*7)); self.clock += .2
        np.testing.assert_allclose(self.tick(),[.09,-.09,0,0,0,0,0])
        self.assertTrue(n._last_status.startswith('stale_ramp_to_zero'))

    def test_stamped_output_has_generation_time_names_and_actual_limited_velocity(self):
        n=self.node; n.stamped_pub=Mock()
        n._vbc_hold_cb(message(data=True))
        output=self.tick()
        msg=n.stamped_pub.publish.call_args[0][0]
        self.assertEqual(msg.header.stamp,Stamp(self.clock))
        self.assertEqual(msg.header.seq,1)
        self.assertEqual(msg.name,n.joint_names)
        np.testing.assert_array_equal(msg.velocity,output)


if __name__=='__main__':
    unittest.main()
