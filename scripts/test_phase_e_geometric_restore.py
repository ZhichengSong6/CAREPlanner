"""Alternate recovery must retain content identity and require native proof."""
import copy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
from geometric_restore import validate, permits_unknown_conflicts
from repair_prefix_contract import parse
import test_phase_e_selected_prefix as prefix
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header,String
from care_collision_cdf.msg import CollisionCDFConstraintBatch

def intent(stamp=100):
    return dict(schema=1,raw_candidate_stamp_ns=str(stamp),query_stamp_ns=str(stamp-1),
                mode_epoch='2',observation_token='care_obs_v1_1_fixture',points=[[.05,.2,.55]])

class IntentTests(unittest.TestCase):
    def batch(self,source=0,point=None):
        return SimpleNamespace(num_pairs=1,source_type=[source],point_flat=point or [.05,.2,.55],SOURCE_UNKNOWN=0)
    def test_only_declared_unknown_conflict_can_request_geometry(self):
        self.assertTrue(permits_unknown_conflicts(intent(),self.batch(),[-.1],.01))
        self.assertFalse(permits_unknown_conflicts(intent(),self.batch(1),[-.1],.01))
        self.assertFalse(permits_unknown_conflicts(intent(),self.batch(point=[.1,.2,.55]),[-.1],.01))
    def test_missing_source_or_point_identity_fails_closed(self):
        b=self.batch();b.source_type=[]
        self.assertFalse(permits_unknown_conflicts(intent(),b,[-.1],.01))
        b=self.batch();b.point_flat=[]
        self.assertFalse(permits_unknown_conflicts(intent(),b,[-.1],.01))
    def test_nonfinite_distance_cannot_be_overridden(self):
        self.assertFalse(permits_unknown_conflicts(intent(),self.batch(),[float('nan')],.01))
    def test_uniquified_query_identity_uses_separate_physical_clock(self):
        r=intent();r.update(schema=2,query_stamp_ns='101',query_ros_ns='99')
        self.assertEqual(validate(r),r)
        r['query_ros_ns']='101'
        with self.assertRaises(ValueError):validate(r)
    def test_rejects_malformed_or_foreign_intent(self):
        for key,value in [('schema',True),('mode_epoch','0'),('query_stamp_ns','101'),
                          ('observation_token','ordinary'),('points',[[float('nan'),0,0]]),
                          ('points',[[.05,.2,.55],[.05,.2,.55]])]:
            r=intent();r[key]=value
            with self.assertRaises(ValueError):validate(r)

class RecoveryAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.b=prefix.PrefixTests(methodName='runTest');self.b.setUp();self.addCleanup(self.b.doCleanups)
        self.n=self.b.n;self.n.repair_motion_consistency_enabled=True;self.n.initial_shell_execution_margin_m=.01
        self.measure()
        self.r=intent(self.b.raw.header.stamp.to_nsec())
        self.contract=dict(self.b.contract,geometric_restore=self.r)
    def measure(self):
        state=JointState(header=Header(stamp=rospy.Time.from_sec(self.b.now)),name=self.b.raw.joint_names,
                         position=self.b.data['measured_q'],velocity=[0.]*7)
        self.n._joint_state_cb(state);self.n._actuator_state_cb(state)
    def dispatch(self,point=None,source=0):
        self.b.start(json.dumps(self.contract))
        batch=CollisionCDFConstraintBatch(header=Header(stamp=self.n.final_gcdf_query_pub.messages[-1].header.stamp),
            num_pairs=1,distance=[-.1],point_flat=point or self.r['points'][0],source_type=[source],
            original_timestep=[0],approx_body_clearance_m=[.1])
        self.n._final_gcdf_batch_cb(batch)
    def evidence(self):
        c=self.n._outstanding
        return dict(schema=2,trajectory_stamp_ns=str(c.header.stamp.to_nsec()),
            measured_stamp_ns=str(self.n._latest_measured_stamp_ns),map_snapshot_stamp_ns=str(self.n._latest_measured_stamp_ns),
            execution_floor_m=.01,actual_min_distance_m=.03,onset_clearance_budget_m=.009,
            certified_residual_m=.005,motion_horizon_s=c.points[-1].time_from_start.to_sec(),
            measured_q=c.points[0].positions,measured_velocity=[0.]*7,initial_q=c.points[0].positions,
            initial_velocity=c.points[0].velocities,joint_motion_radii=[1.]*7,contract='geometric_restore',
            restoration_points=self.r['points'],points=[dict(point=self.r['points'][0],floor_m=.02,start_m=.03,end_m=.04)])
    def verdict(self,e,count=1):
        c=self.n._outstanding;d=self.n._outstanding_diag;key=d['transfer_key'];p=d['planned_prefix']
        self.n._global_summary_cb(String(data='trajectory_source=predicted trajectory_stamp_ns={} has_violation=0 '
            'shell_recovery_count={} shell_recovery_evidence={} context_request_session={} context_request_sequence={} '
            'context_prefix_session={} context_prefix_revision={} context_prefix_s={}'.format(
                c.header.stamp.to_nsec(),count,json.dumps(e,separators=(',',':')),key[0],key[1],p['session'],p['revision'],p['prefix_s'])))
    def test_declared_conflict_requires_independent_certificate(self):
        self.dispatch();self.assertFalse(self.n.committed_pub.messages)
        self.assertFalse(self.n._outstanding_diag['final_gcdf_passed'])
        self.assertTrue(self.n._outstanding_diag['geometry_override_pending'])
        ctx=json.loads(self.n.shell_context_pub.messages[-1].data)
        self.assertEqual(ctx['kind'],'geometric_restore');self.assertEqual(ctx['geometric_restore'],self.r)
        self.verdict(self.evidence());self.assertEqual(len(self.n.committed_pub.messages),1)
    def test_zero_certificate_count_cannot_commit(self):
        self.dispatch();self.verdict('none',0);self.assertFalse(self.n.committed_pub.messages)
    def test_shell_certificate_cannot_authorize_geometric_restore(self):
        self.dispatch();e=self.evidence();e.pop('contract');self.verdict(e)
        self.assertFalse(self.n.committed_pub.messages)
    def test_certificate_for_other_points_cannot_commit(self):
        self.dispatch();e=self.evidence();e['restoration_points']=[[0.,0.,0.]];self.verdict(e)
        self.assertFalse(self.n.committed_pub.messages)
    def test_other_unknown_or_occupied_conflict_stays_rejected(self):
        self.dispatch(point=[.1,.2,.55]);self.assertFalse(self.n.verification_pub.messages)
    def test_foreign_raw_identity_rejected_before_query(self):
        self.contract['geometric_restore']['raw_candidate_stamp_ns']=str(self.b.raw.header.stamp.to_nsec()+1)
        self.b.start(json.dumps(self.contract));self.assertFalse(self.n.final_gcdf_query_pub.messages)
    def test_unchanged_prefix_can_carry_bound_recovery_intent(self):
        self.assertEqual(parse(json.dumps(self.contract))['geometric_restore'],self.r)
        self.dispatch();self.assertEqual(self.n._outstanding_diag['planned_prefix'],self.b.contract)
    def test_stale_certificate_refresh_preserves_exact_motion(self):
        self.dispatch();e=self.evidence();first=copy.deepcopy(self.n._outstanding)
        self.b.advance(.051);self.measure();self.verdict(e)
        self.assertEqual(len(self.n.verification_pub.messages),2)
        self.assertEqual(self.n._outstanding,first)
        self.assertFalse(self.n.committed_pub.messages)
        self.assertTrue(self.n._outstanding_diag['shell_refresh_requested'])
        self.verdict(self.evidence());self.assertEqual(len(self.n.committed_pub.messages),1)
    def test_second_stale_reply_ends_without_geometric_failure(self):
        self.dispatch();e=self.evidence();self.b.advance(.051);self.measure();self.verdict(e)
        self.verdict(e);self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(len(self.n.verification_pub.messages),2)
        self.assertIn('safety_gate=state_evidence',self.n.verification_event_pub.messages[-1].data)
        self.assertEqual(self.n._verification_unsafe_count,0)
    def test_unavailable_actuator_waits_only_within_original_deadline(self):
        self.n._latest_actuator_stamp_ns=0
        self.b.start(json.dumps(self.contract))
        self.assertEqual(self.n._preflight_reason,'actuator_state_unavailable')
        key=self.n._preflight_wait_key
        self.assertIsNotNone(key)
        deadline=self.n._admission_entries[key]['deadline']
        self.n._resume_preflight()
        self.assertEqual(self.n._admission_entries[key]['deadline'],deadline)
        self.assertFalse(self.n.final_gcdf_query_pub.messages or self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.b.advance(.751,ros=False);self.n._admission_tick();self.b.pump()
        self.assertEqual(self.n._admission_entries[key]['stage'],'expired')
        self.assertIn('audit_started=0',self.n._admission_entries[key]['status'].data)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)

if __name__=='__main__':unittest.main()
