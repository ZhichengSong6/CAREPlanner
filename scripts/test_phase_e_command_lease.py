#!/usr/bin/env python3
import copy
from pathlib import Path
import sys
import types
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/egocentric_arm_planner/scripts'))
from execution_command_lease import LeasedCommandLimiter

class Time(float):
    def to_sec(self):return float(self)

class TransportTests(unittest.TestCase):
    def setUp(self):
        self.n=LeasedCommandLimiter(['j'+str(i) for i in range(7)],[2.]*7,[3.]*7)
        self.state=types.SimpleNamespace(header=types.SimpleNamespace(stamp=Time(1.)),controller_session=71,
            execution_lease_required=True,clock_fault=False,retired_execution_through_ns=0,
            velocity_limits=[2.]*7,acceleration_limits=[3.]*7)
        self.n.on_state(self.state,1.)
        self.msg=types.SimpleNamespace(header=types.SimpleNamespace(stamp=Time(1.)),source_sequence=1,
            controller_session=71,execution_stamp_ns=100,lease_sequence=3,valid_until=Time(1.1),
            certificate_sha256='a'*64,joint_names=list(self.n.names),velocity=[.6]*7,stop=False)

    def test_limiter_preserves_every_identity_and_original_time(self):
        self.assertTrue(self.n.receive(self.msg,1.))
        out=self.n.tick(1.01,.01)
        for key in vars(self.msg):
            if key!='velocity':self.assertEqual(vars(out)[key],vars(self.msg)[key])
        np.testing.assert_allclose(out.velocity,[.03]*7)

    def test_repeated_ticks_neither_renew_nor_increase_command(self):
        self.n.receive(self.msg,1.)
        a=self.n.tick(1.01,.01);b=self.n.tick(1.02,.01)
        self.assertEqual(vars(a),vars(b))

    def test_delayed_replayed_future_and_wrong_session_rejected(self):
        for field,value in [('controller_session',72),('valid_until',Time(.9)),('source_sequence',0),('joint_names',[])]:
            m=copy.deepcopy(self.msg);setattr(m,field,value)
            self.assertFalse(self.n.receive(m,1.01))
        self.assertFalse(self.n.receive(self.msg,1.2))
        m=copy.deepcopy(self.msg);m.header.stamp=Time(1.02)
        self.assertFalse(self.n.receive(m,1.01))
        self.assertTrue(self.n.receive(self.msg,1.01))
        self.assertFalse(self.n.receive(self.msg,1.01))

    def test_hold_and_release_cannot_resurrect_same_execution(self):
        self.n.receive(self.msg,1.);self.n.stop()
        self.assertTrue(self.n.tick(1.01,.01).stop)
        m=copy.deepcopy(self.msg);m.source_sequence=2
        self.assertFalse(self.n.receive(m,1.02))
        m.execution_stamp_ns=101
        self.assertTrue(self.n.receive(m,1.02))
        self.assertFalse(self.n.tick(1.02,.01).stop)

    def test_deadline_and_stale_application_state_force_terminal_stop(self):
        self.n.receive(self.msg,1.)
        self.assertTrue(self.n.tick(1.051,.01).stop)
        self.state.header.stamp=Time(1.1);self.n.on_state(self.state,1.1)
        self.assertTrue(self.n.tick(1.1,.01).stop)

    def test_session_restart_clears_old_transport_and_replay(self):
        self.n.receive(self.msg,1.);self.n.tick(1.01,.01)
        self.state.controller_session=72;self.state.header.stamp=Time(1.02)
        self.n.on_state(self.state,1.02)
        self.assertIsNone(self.n.tick(1.02,.01))
        self.assertFalse(self.n.receive(self.msg,1.02))

    def test_controller_limit_mismatch_cannot_supply_fresh_application_state(self):
        self.state.header.stamp=Time(1.01);self.state.acceleration_limits=[6.]*7
        self.assertFalse(self.n.on_state(self.state,1.01))
        self.assertEqual(self.n.state_time,1.)

    def test_delayed_previous_session_status_cannot_roll_back_identity(self):
        old=copy.deepcopy(self.state)
        self.state.controller_session=72;self.state.header.stamp=Time(1.02)
        self.assertTrue(self.n.on_state(self.state,1.02))
        self.assertFalse(self.n.on_state(old,1.021))
        self.assertEqual(self.n.session,72)

    def test_bound_range_is_preserved_and_illegal_target_rejected(self):
        self.msg.target_range_enforced=True;self.msg.target_abs_velocity=[.6,0.,0.,0.,0.,0.,0.]
        self.msg.velocity=[.6,0.,0.,0.,0.,0.,0.]
        self.assertTrue(self.n.receive(self.msg,1.))
        out=self.n.tick(1.01,.01)
        self.assertEqual(out.target_abs_velocity,self.msg.target_abs_velocity)
        self.assertTrue(out.target_range_enforced)
        bad=copy.deepcopy(self.msg);bad.source_sequence=2;bad.velocity[1]=.001
        self.assertFalse(self.n.receive(bad,1.02))

    def test_smaller_range_does_not_emit_old_limiter_target(self):
        self.n.receive(self.msg,1.);self.n.tick(1.01,.01)
        self.msg.source_sequence=2;self.msg.header.stamp=Time(1.02)
        self.msg.target_range_enforced=True;self.msg.target_abs_velocity=[0.]*7;self.msg.velocity=[0.]*7
        self.assertTrue(self.n.receive(self.msg,1.02))
        np.testing.assert_array_equal(self.n.tick(1.02,.001).velocity,[0.]*7)

    def test_malformed_range_is_rejected(self):
        self.msg.target_range_enforced=True
        for cap in [[-1.]*7,[float('nan')]*7,[2.1]*7,[0.]*6]:
            self.msg.target_abs_velocity=cap
            self.assertFalse(self.n.receive(self.msg,1.))

if __name__=='__main__':unittest.main()
