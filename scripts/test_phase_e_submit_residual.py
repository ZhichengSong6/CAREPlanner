#!/usr/bin/env python3
"""Held-state proposals and independently checked measured-stop contracts."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
from handoff_schedule import predict_held_motion
from initial_shell_recovery import commit_check
from recovery_motion import residual_motion_bound
from test_phase_e_handoff_schedule import ScheduleTests


class PredictionTests(unittest.TestCase):
    def data(self):
        gain=2.5;dt=.01;q=[-.001]*7;v=[.0025]*7
        old=(1000000000,q,v)
        new=(1010000000,[x*math.exp(-gain*dt) for x in q],
             [x*math.exp(-gain*dt) for x in v])
        cert=dict(joint_motion_radii=[1.]*7,motion_horizon_s=.4,
                  certified_residual_m=.00016,handoff_latency_s=.105)
        return old,new,list(new[2]),cert,[3.]*7,gain

    def test_forecast_matches_static_reference_and_preserves_real_state(self):
        args=self.data();saved=copy.deepcopy(args);p=predict_held_motion(*args)
        self.assertIsNotNone(p)
        for a,b in zip(p['q'],args[1][1]):self.assertAlmostEqual(a,b*math.exp(-2.5*p['latency_s']))
        self.assertEqual(args,saved)
        self.assertEqual(p['source_stamp_ns'],args[1][0])
        self.assertLess(p['estimated_error_m'],args[3]['certified_residual_m'])

    def test_measured_ramp_or_applied_velocity_change_is_not_a_hold(self):
        for index in [1,2]:
            args=list(self.data())
            if index==1:args[1]=(args[1][0],args[1][1],[.006]*7)
            else:args[2]=[.006]*7
            self.assertIsNone(predict_held_motion(*args))

    def test_invalid_delay_stale_samples_and_nonfinite_state_are_rejected(self):
        for delay in [-1.,0.,.200001,float('nan'),float('inf')]:
            args=list(self.data());args[3]['handoff_latency_s']=delay
            self.assertIsNone(predict_held_motion(*args))
        for stamp in [1000000000,999000000,1051000000]:
            args=list(self.data());args[1]=(stamp,args[1][1],args[1][2])
            self.assertIsNone(predict_held_motion(*args))
        args=list(self.data());args[2][0]=float('nan')
        self.assertIsNone(predict_held_motion(*args))


class StopContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ledger=json.loads((ROOT/'outputs/phase_e_observation_identity_20261017/online_ledger_002_a.json').read_text())
        r=next(r for r in ledger['terminal_failures'] if r['raw_candidate_stamp_ns']=='27423000000')
        cls.e=json.loads(r['shell_recovery_evidence']);cls.o=json.loads(r['shell_recovery_onset_detail'])
        cls.names=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']

    def check(self,e,o=None):
        o=copy.deepcopy(self.o if o is None else o);d={}
        ok=commit_check(json.dumps(e),int(e['trajectory_stamp_ns']),int(o['decision_ros_ns']),
            int(o['live_stamp_ns']),self.names,dict(zip(self.names,o['live_q'])),
            dict(zip(self.names,o['live_velocity'])),[3.,3.,4.,4.,6.,6.,6.],.05,20,.01,
            diagnostics=d,feedback_gain=2.5,actuator_velocity=dict(zip(self.names,o['actuator_velocity'])))
        return ok,d

    def modern(self):
        e=copy.deepcopy(self.e)
        e.update(measured_stop_horizon_s=.05,
                 measured_stop_contract='ramp_stop_and_linear_start_bridge_v1')
        return e

    def test_legacy_state_rejection_is_exactly_preserved(self):
        ok,d=self.check(self.e)
        self.assertFalse(ok)
        self.assertAlmostEqual(d['motion_bound_m'],self.o['motion_bound_m'])

    def test_stop_horizon_does_not_shorten_planned_initial_or_actuator_check(self):
        ok,d=self.check(self.modern())
        self.assertFalse(ok)
        self.assertLess(d['measured_motion_bound_m'],self.o['measured_motion_bound_m'])
        self.assertEqual(d['initial_motion_bound_m'],self.o['initial_motion_bound_m'])
        self.assertEqual(d['actuator_motion_bound_m'],self.o['actuator_motion_bound_m'])
        self.assertEqual(d['motion_bound_m'],self.o['motion_bound_m'])

    def test_measured_stop_metadata_cannot_supply_arbitrary_short_horizon(self):
        for value in [0.,.049,.051,float('nan')]:
            e=self.modern();e['measured_stop_horizon_s']=value
            ok,d=self.check(e);self.assertFalse(ok)
            self.assertEqual(d['failure_kind'],'invalid_measured_stop_contract')
        e=self.modern();e['measured_stop_contract']='unproved'
        self.assertEqual(self.check(e)[1]['failure_kind'],'invalid_measured_stop_contract')
        e=self.modern();del e['measured_stop_horizon_s']
        self.assertEqual(self.check(e)[1]['failure_kind'],'invalid_measured_stop_contract')

    def test_stop_contract_checks_applied_velocity_too(self):
        e=self.modern();o=copy.deepcopy(self.o)
        # A matching planned initial boundary still cannot hide a changed
        # measured-stop branch or an applied command from another state.
        e['initial_q']=list(o['live_q']);e['initial_velocity']=list(o['live_velocity'])
        self.assertTrue(self.check(e,o)[0])
        o['actuator_velocity'][0]+=.02
        ok,d=self.check(e,o);self.assertFalse(ok)
        self.assertGreater(d['measured_actuator_motion_bound_m'],e['certified_residual_m'])

    def test_stop_displacement_difference_is_bounded_across_velocity_signs(self):
        # The native proof uses the exact50ms clipped/trapezoid stop. Compare
        # two such stops, including reversals and different stop durations.
        for first_v in [-1.8,-.151,-.004,0.,.004,.151,1.8]:
            for second_v in [-1.8,-.004,0.,.004,1.8]:
                v0,v1=first_v,second_v
                q0,q1=0.,.0001;a=3.;r=.8
                bound=residual_motion_bound([q0],[v0],[q1],[v1],[r],[a],.05,2.5)
                for _ in range(20):
                    self.assertLessEqual(r*abs(q1-q0),bound+1e-12)
                    n0=math.copysign(max(0.,abs(v0)-a*.05),v0)
                    n1=math.copysign(max(0.,abs(v1)-a*.05),v1)
                    q0+=.025*(v0+n0);q1+=.025*(v1+n1);v0,v1=n0,n1


class ReceiverPredictionTests(ScheduleTests):
    def test_supported_forecast_builds_a_proposal_without_overwriting_measurement(self):
        self.waiting();self.n.tracking_feedback_gain=2.5
        self.n.handoff_prediction_enabled=True
        self.n._handoff_certificate=(self.n._handoff_scope(self.entry),dict(
            joint_motion_radii=[1.]*7,motion_horizon_s=.4,
            certified_residual_m=.00015,handoff_latency_s=.08))
        start=self.now
        q=None
        for i in range(1,16):
            self.advance(.01)
            offset=-.001*math.exp(-2.5*(self.now-start))
            q=[x+offset for x in self.data['measured_q']]
            self.measured(velocity=-2.5*offset,position=q);self.tracker(.3)
            if self.n.final_gcdf_query_pub.messages:break
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
        candidate=self.n.final_gcdf_query_pub.messages[-1]
        self.assertNotEqual(candidate.points[0].positions,q)
        self.assertEqual(self.n._latest_measured_q_by_name,dict(zip(self.raw.joint_names,q)))
        self.assertEqual(self.n._gcdf_outstanding_diag['handoff_start_prediction'],1)
        self.assertEqual(self.entry['deadline'],self.deadline)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.gcdf(False)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)


def load_tests(loader,tests,pattern):
    return unittest.TestSuite([loader.loadTestsFromTestCase(PredictionTests),
        loader.loadTestsFromTestCase(StopContractTests),
        ReceiverPredictionTests('test_supported_forecast_builds_a_proposal_without_overwriting_measurement')])


if __name__=='__main__':unittest.main(verbosity=2)
