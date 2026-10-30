"""Saved state failures through the production commit and receiver callbacks."""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from test_phase_e_candidate_admission import AdmissionTests,Pub,String,rospy,JointTrajectoryPoint
ROOT=Path(__file__).resolve().parents[1]

class HandoffFlowTests(AdmissionTests):
    def recorded(self,tag,audit):
        rows=json.loads((ROOT/'outputs/phase_e_handoff_detailed_analysis_20261016/handoff_fixtures.json').read_text())
        row=next(r for r in rows if r['tag']==tag and r['audit']==audit)
        return copy.deepcopy(next(m for m in row['matches'] if m['matching_actuators']))

    def prepare(self,tag='002',audit='18873000000'):
        self.start();self.gcdf();m=self.recorded(tag,audit);e=m['certificate'];c=m['context']
        self.n.initial_shell_recovery_enabled=True;self.n.tracking_feedback_gain=2.5
        self.n.initial_shell_execution_margin_m=.01
        stamp=self.n._outstanding.header.stamp.to_nsec()
        self.n._outstanding.points=[JointTrajectoryPoint(positions=q,velocities=v,time_from_start=rospy.Duration(t))
            for q,v,t in zip(c['q'],c['v'],c['times_s'])]
        self.n._outstanding_repair=True;self.n._outstanding_probe=False
        self.n._outstanding_view='repair_prefix_brake_hold'
        restore=copy.deepcopy(c['geometric_restore']);restore['raw_candidate_stamp_ns']=str(self.raw.header.stamp.to_nsec())
        restore['query_stamp_ns']=restore['query_ros_ns']=str(self.raw.header.stamp.to_nsec()-1000)
        self.n._outstanding_diag.update(geometric_restore=restore,final_gcdf_passed=True)
        self.n._latest_measured_q_by_name=dict(zip(c['joint_names'],m['live_q']))
        self.n._latest_measured_velocity_by_name=dict(zip(c['joint_names'],m['live_v']))
        self.n._latest_actuator_velocity=dict(zip(c['joint_names'],m['matching_actuators'][-1]['velocity']))
        e['trajectory_stamp_ns']=str(stamp)
        delta=rospy.Time.now().to_nsec()-m['now_ns']
        for key in ['measured_stamp_ns','map_snapshot_stamp_ns']:e[key]=str(int(e[key])+delta)
        self.n._latest_measured_stamp_ns=m['live_stamp_ns']+delta
        self.n._latest_actuator_stamp_ns=rospy.Time.now().to_nsec()
        d=m['certificate_diagnostics'][0]
        e.update(residual_stage=d['residual_stage'],residual_floor_m=float(d['residual_floor_m']),
            residual_point=[float(x) for x in d['residual_point'].split(',')],
            residual_ordinary_unknown=float(d['residual_floor_m'])==.02,residual_interval_index=2)
        return m,e,stamp

    def summary(self,e,stamp):
        key=self.n._outstanding_diag['transfer_key']
        return String(data='trajectory_source=predicted trajectory_stamp_ns={} has_violation=0 shell_recovery_count=1 '
            'shell_recovery_evidence={} context_request_session={} context_request_sequence={}'.format(
                stamp,json.dumps(e,separators=(',',':')),key[0],key[1]))

    def test_large_old_start_mismatch_skips_refresh_and_emits_real_typed_failure(self):
        m,e,stamp=self.prepare();count=len(self.n.verification_pub.messages)
        with patch.object(rospy,'loginfo'):
            self.n._global_summary_cb(self.summary(e,stamp))
        self.assertEqual(len(self.n.verification_pub.messages),count)
        event=self.n.verification_event_pub.messages[-1].data
        self.assertIn('result=timeout',event);self.assertIn('safety_gate=state_evidence',event)
        self.assertIn('handoff_residual_evidence={',event)
        self.assertIn('"requires_reconstruction":true',event)
        self.assertEqual(self.n.committed_pub.messages,[])

    def test_attainable_residual_refresh_preserves_exact_candidate_deadline_and_quality_identity(self):
        m,e,stamp=self.prepare('018_b','32160000000');candidate=copy.deepcopy(self.n._outstanding)
        deadline=self.n._outstanding_sent
        with patch.object(rospy,'loginfo'):
            self.n._global_summary_cb(self.summary(e,stamp))
        self.assertEqual(self.n._outstanding,candidate)
        self.assertEqual(self.n._outstanding_sent,deadline)
        hint=json.loads(self.n.shell_quality_pub.messages[-1].data)
        self.assertEqual(hint['trajectory_stamp_ns'],str(stamp))
        self.assertAlmostEqual(hint['required_residual_m'],.00032700400736078916)
        self.assertEqual(self.n.committed_pub.messages,[])
        # Only the original single certificate refresh; the second failure closes the request.
        count=len(self.n.verification_pub.messages)
        with patch.object(rospy,'loginfo'):
            self.n._global_summary_cb(self.summary(e,stamp))
        self.assertEqual(len(self.n.verification_pub.messages),count)
        self.assertIn('handoff_residual_evidence={',self.n.verification_event_pub.messages[-1].data)

    def test_decomposed_bound_matches_saved_zero_measurement_delta_and_full_gate(self):
        m,e,stamp=self.prepare();d={}
        self.assertFalse(self.n._shell_commit_valid_locked(json.dumps(e),stamp,rospy.Time.now(),d))
        self.assertEqual(d['measured_motion_bound_m'],0.)
        self.assertAlmostEqual(d['motion_bound_m'],.26128431051181594)
        self.assertTrue(d['requires_reconstruction'])
        self.assertEqual(d['handoff_residual_evidence']['raw_candidate_stamp_ns'],str(self.raw.header.stamp.to_nsec()))

def load_tests(loader,tests,pattern):return loader.loadTestsFromTestCase(HandoffFlowTests)
if __name__=='__main__':unittest.main()
