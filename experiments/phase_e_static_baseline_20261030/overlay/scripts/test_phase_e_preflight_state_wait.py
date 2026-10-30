#!/usr/bin/env python3
"""Actual transport/receiver callbacks with controlled clocks and publishers."""
import copy,json,unittest
from unittest.mock import patch
from test_phase_e_candidate_admission import AdmissionTests,JointState,String,Bool,rospy

class PreflightWaitTests(AdmissionTests):
    def state(self, stamp=None, actuator=False):
        return JointState(header=type(self.raw.header)(stamp=rospy.Time.from_sec(self.now if stamp is None else stamp)),
                          name=self.raw.joint_names,position=[] if actuator else self.data['measured_q'],velocity=[0.]*7)
    def waiting(self, offset=-.051):
        self.n.repair_motion_consistency_enabled=True
        self.g.sender.replan_count=1  # Original one transport recovery is already spent.
        self.n._joint_state_cb(self.state(self.now+offset))
        self.n._actuator_state_cb(self.state(actuator=True))
        self.start()
        self.assertIsNotNone(self.n._pending_raw)
        self.assertIsNotNone(self.n._preflight_wait_key)
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
        self.assertEqual(self.n.verification_pub.messages,[])
        self.assertEqual(self.n.committed_pub.messages,[])
        self.key=self.n._pending_transfer_key
        self.deadline=self.n._admission_entries[self.key]['deadline']
    def test_resource_evidence_crosses_real_protocol_outcome_only_for_current_audit(self):
        self.start();self.gcdf()
        self.n.initial_shell_recovery_enabled=True
        stamp=self.n._outstanding.header.stamp.to_nsec()
        evidence=json.dumps(dict(schema=1,kind='mandatory_proof_resource_exhausted',trajectory_stamp_ns=str(stamp)),separators=(',',':'))
        def message(identity):return String(data='trajectory_stamp_ns={} state=invalid reason=geometric_restore_proof_resource_exhausted proof_resource_evidence={}'.format(identity,evidence))
        count=len(self.n.verification_event_pub.messages)
        self.n._verification_protocol_cb(message(stamp+1))
        self.assertEqual(len(self.n.verification_event_pub.messages),count)
        self.n._verification_protocol_cb(message(stamp))
        event=self.n.verification_event_pub.messages[-1].data
        self.assertIn('proof_resource_evidence='+evidence,event)
        self.assertIn('vbc_protocol_reason=geometric_restore_proof_resource_exhausted',event)
        self.assertIn('raw_candidate_stamp_ns='+str(self.raw.header.stamp.to_nsec()),event)
        self.assertIn('audited_trajectory_stamp_ns='+str(stamp),event)
        self.assertEqual(self.n.committed_pub.messages,[])
    def test_stale_to_fresh_same_identity_dispatches_once_after_transport_retry_spent(self):
        self.waiting();raw=self.n._pending_raw.header.stamp.to_nsec();digest=self.n._admission_entries[self.key]['digest']
        self.n._joint_state_cb(self.state());self.pump()
        self.assertIsNone(self.n._preflight_wait_key)
        self.assertIsNone(self.n._pending_raw)
        self.assertEqual(self.n._preflight_resume_count,1)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
        self.assertEqual(self.n._gcdf_outstanding_diag['raw_candidate_stamp_ns'],raw)
        self.assertEqual(self.n._admission_entries[self.key]['digest'],digest)
        self.assertEqual(self.n._admission_entries[self.key]['deadline'],self.deadline)
        self.n._joint_state_cb(self.state());self.n._actuator_state_cb(self.state(actuator=True));self.n._admission_tick()
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
        self.assertTrue(self.g.sender.replan_count)
        self.assertEqual(self.n.committed_pub.messages,[])
    def test_future_timestamp_waits_until_clock_catches_up(self):
        self.waiting(offset=.001);self.advance(.002);self.n._admission_tick();self.pump()
        self.assertEqual(self.n._preflight_resume_count,1)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
        self.assertEqual(self.n._admission_entries[self.key]['deadline'],self.deadline)
    def test_missing_actuator_waits_for_actual_state(self):
        self.n.repair_motion_consistency_enabled=True
        self.n._joint_state_cb(self.state());self.start()
        self.assertEqual(self.n._preflight_reason,'actuator_state_unavailable')
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
        self.n._actuator_state_cb(self.state(actuator=True));self.pump()
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
    def test_never_fresh_expires_original_lease(self):
        self.waiting();self.advance(1.0,ros=False);self.n._admission_tick();self.pump()
        self.assertEqual(self.n._admission_entries[self.key]['stage'],'expired')
        self.assertEqual(self.n._admission_entries[self.key]['deadline'],self.deadline)
        self.n._joint_state_cb(self.state());self.n._actuator_state_cb(self.state(actuator=True))
        self.assertIsNone(self.n._preflight_wait_key)
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
    def test_cancel_wait_cannot_resurrect_on_fresh_state(self):
        self.waiting();self.g._emit_admission_actions(self.g.sender.invalidate(self.wall));self.pump()
        self.n._joint_state_cb(self.state())
        self.assertIsNone(self.n._preflight_wait_key)
        self.assertEqual(self.n._admission_entries[self.key]['stage'],'cancelled')
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
    def test_goal_change_cancels_wait(self):
        from geometry_msgs.msg import PoseStamped
        self.waiting();self.n._admission_goal_cb(PoseStamped());self.n._joint_state_cb(self.state())
        self.assertEqual(self.n._admission_entries[self.key]['stage'],'cancelled')
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
    def test_mode_change_cancels_wait(self):
        self.waiting();self.n._repair_active_cb(Bool(data=True));self.n._joint_state_cb(self.state())
        self.assertEqual(self.n._admission_entries[self.key]['stage'],'cancelled')
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
    def test_owner_change_cancels_wait(self):
        self.n._preflight_owner_token='care_obs_v1_old'
        self.waiting();self.n._preflight_owner_cb(String(data='active_q_vis_token=care_obs_v1_new'))
        self.n._joint_state_cb(self.state())
        self.assertEqual(self.n._admission_entries[self.key]['stage'],'cancelled')
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
    def test_nonfinite_fresh_state_is_rejected(self):
        self.waiting();msg=self.state();msg.position[0]=float('nan');self.n._joint_state_cb(msg)
        self.assertEqual(self.n._admission_entries[self.key]['stage'],'rejected')
        self.assertEqual(self.n.final_gcdf_query_pub.messages,[])
    def test_clock_ages_between_preflight_and_construction_preserves_request(self):
        self.waiting()
        build=self.n._repair_prefix_with_braking_tail
        def fail_once(*a,**kw):
            self.n._last_prefix_build_reason='live_state_stale';return None
        with patch.object(self.n,'_repair_prefix_with_braking_tail',side_effect=fail_once):
            self.n._joint_state_cb(self.state())
        self.assertEqual(self.n._pending_transfer_key,self.key)
        self.assertEqual(self.n._admission_entries[self.key]['stage'],'received')
        self.assertEqual(self.n._admission_entries[self.key]['deadline'],self.deadline)
        self.n._admission_tick();self.pump()
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)

def load_tests(loader, tests, pattern):
    return loader.loadTestsFromTestCase(PreflightWaitTests)

if __name__=='__main__':unittest.main()
