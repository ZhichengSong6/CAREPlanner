"""Real sender/receiver transaction controls for pre-audit expiry and thin points."""
import copy
import json
import unittest
from unittest.mock import patch
from test_phase_e_handoff_schedule import ScheduleTests, String, rospy


class PreauditTests(ScheduleTests):
    def expired(self):
        self.waiting()
        self.g.sender.replan_count = 1
        self.advance(.751, ros=False)
        self.n._admission_tick()
        self.pump()

    def test_terminal_after_transport_once_has_exact_payload_and_no_audit(self):
        # These tests use the bound transport just like the production planner.
        self.expired_bound()
        events=self.g.admission_terminal_pub.messages
        self.assertEqual(len(events),1)
        e=events[0]
        self.assertEqual(e.action,'preaudit_expired')
        self.assertEqual(e.remaining_budget_s,0.)
        self.assertEqual(e.trajectory,self.raw)
        self.assertEqual(e.planned_prefix_context,self.context)
        self.assertEqual(e.payload_sha256,self.entry['digest'])
        self.assertTrue(self.g.sender.replan_count)
        self.assertFalse(self.g.replan_request_pub.messages)
        self.assertFalse(self.n.final_gcdf_query_pub.messages or self.n.committed_pub.messages)
        self.g._admission_status_cb(copy.deepcopy(self.entry['status']))
        self.assertEqual(len(events),1)

    def expired_bound(self):
        import optimized_trajectory_continuity_node as receiver
        import probe_single_flight_gate_node as sender
        from test_phase_e_candidate_admission import JointState
        self.params['~repair_prefix_binding_enabled']=True
        self.params['~initial_shell_recovery_enabled']=True
        self.params['~repair_projection_consistent_dispatch_enabled']=True
        self.n=receiver.OptimizedTrajectoryContinuityNode()
        self.g=sender.ProbeSingleFlightGate()
        self.n._joint_state_cb(JointState(name=self.raw.joint_names,position=self.data['measured_q']))
        self.g._admission_status_cb(String(data='action=pong ready_nonce=test receiver_boot_id='+self.n._admission_boot_id))
        original=self.g._candidate_cb
        def submit(raw):
            c=copy.deepcopy(self.n._prefix_context)
            stamp=raw.header.stamp.to_nsec()
            c['geometric_restore']=dict(schema=2,raw_candidate_stamp_ns=str(stamp),
                query_stamp_ns=str(stamp-1),query_ros_ns=str(stamp-1),mode_epoch='3',
                observation_token='care_obs_v1_schedule',points=[[.25,0.,.95]])
            self.context=json.dumps(c,separators=(',',':'))
            original(raw,self.context)
        with patch.object(self.g,'_candidate_cb',side_effect=submit):self.waiting()
        self.g.sender.replan_count=1
        self.advance(.751,ros=False);self.n._admission_tick();self.pump()

    def test_postaudit_expiry_cannot_become_preaudit(self):
        self.start();self.g.sender.active['prefix_context']='bound'
        self.assertEqual(next(iter(self.n._admission_entries.values()))['audit_started'],True)
        self.g.sender.replan_count=1
        self.advance(.751,ros=False);self.n._admission_tick();self.pump()
        self.assertFalse(self.g.admission_terminal_pub.messages)

    def test_next_bound_proposal_rebuilds_and_still_needs_both_audits(self):
        from egocentric_arm_planner.msg import CandidateTransfer
        self.expired_bound()
        self.advance(.01);self.measured();self.tracker(.4,source='trajectory_complete_hold',complete=True)
        raw=copy.deepcopy(self.raw);raw.header.stamp=rospy.Time.from_sec(self.now)
        context=json.loads(self.context);context['geometric_restore']['raw_candidate_stamp_ns']=str(raw.header.stamp.to_nsec())
        self.g._planned_candidate_cb(CandidateTransfer(action='planned',trajectory=raw,
            planned_prefix_context=json.dumps(context,separators=(',',':'))))
        self.pump()
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.gcdf();self.assertEqual(len(self.n.verification_pub.messages),1)
        self.assertFalse(self.n.committed_pub.messages)
        d=self.n._outstanding_diag;c=d['planned_prefix'];key=d['transfer_key']
        self.n._global_summary_cb(String(data='trajectory_source=predicted trajectory_stamp_ns={} '
            'has_violation=1 context_request_session={} context_request_sequence={} '
            'context_prefix_session={} context_prefix_revision={} context_prefix_s={}'.format(
                self.n._outstanding.header.stamp.to_nsec(),*key,c['session'],c['revision'],c['prefix_s'])))
        self.pump();self.assertFalse(self.n.committed_pub.messages)
        self.assertTrue(self.g.sender.replan_count)

    def test_wrong_or_duplicate_receipt_cannot_release(self):
        self.waiting();self.g.sender.active['prefix_context']='{"geometric_restore":{}}'
        self.g.sender.replan_count=1
        self.advance(.751,ros=False);self.n._admission_tick()
        status=self.n.admission_status_pub.messages[-1]
        for suffix in [' session=bad',' sequence=0',' payload_sha256=bad']:
            self.g._admission_status_cb(String(data=status.data+suffix))
        self.assertFalse(self.g.admission_terminal_pub.messages)
        self.pump();self.assertEqual(len(self.g.admission_terminal_pub.messages),1)

    def thin(self):
        self.waiting()
        self.n._handoff_certificate=(self.n._handoff_scope(self.entry),dict(
            joint_motion_radii=[1.]*7,motion_horizon_s=.45,
            certified_residual_m=4.129723483400563e-6,handoff_latency_s=.094))
        self.advance(.01);self.measured(velocity=.004);self.tracker(.3)

    def test_thin_high_point_needs_usable_variation_or_supported_forecast(self):
        self.thin()
        from handoff_schedule import state_variation_ready
        start=self.now
        for i in range(1,13):
            self.advance(.01)
            q=[x+.00004*i for x in self.data['measured_q']]
            self.measured(velocity=.004,position=q);self.tracker(.3)
        self.assertLess(self.now-start,.2)
        # Constant drift with no supported held-reference decay cannot use a
        # fixed100ms low-speed override for a4-micrometre old reserve.
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        for _ in range(2):
            self.advance(.01)
            self.measured(position=q);self.tracker(.3)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
        self.assertEqual(self.entry['deadline'],self.deadline)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.gcdf(False)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)

    def test_changing_or_fast_state_cannot_use_settle_override(self):
        self.thin()
        for i in range(1,25):
            self.advance(.01);self.measured(velocity=.02,position=[x+.001*i for x in self.data['measured_q']]);self.tracker(.3)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_moving_old_prefix_still_cannot_use_settle_override(self):
        self.thin()
        for i in range(1,15):
            self.advance(.01);self.measured(velocity=.004,position=[x+.00004*i for x in self.data['measured_q']]);self.tracker(.1)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)


def load_tests(loader,tests,pattern):
    names=[n for n in loader.getTestCaseNames(PreauditTests) if not n.startswith('test_schedule_') and n not in loader.getTestCaseNames(ScheduleTests.__bases__[0])]
    return unittest.TestSuite(PreauditTests(n) for n in names)

if __name__=='__main__':unittest.main()
