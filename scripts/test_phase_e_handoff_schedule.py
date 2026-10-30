#!/usr/bin/env python3
"""Real receiver callbacks: phase identity, original lease and rebuilt motion."""
import copy
import json
import unittest
from unittest.mock import patch

from test_phase_e_candidate_admission import AdmissionTests, JointState, String, Bool, rospy
from trajectory_msgs.msg import JointTrajectoryPoint
from handoff_schedule import terminal_hold_start, state_variation_ready


class ScheduleTests(AdmissionTests):
    def setUp(self):
        super().setUp()
        self.addCleanup(patch.stopall)
        patch.object(rospy, 'loginfo').start()

    def measured(self, velocity=0., position=None):
        q = self.data['measured_q'] if position is None else position
        self.n._joint_state_cb(JointState(header=type(self.raw.header)(stamp=rospy.Time.from_sec(self.now)),
            name=self.raw.joint_names, position=q, velocity=[velocity]*7))
        self.n._actuator_state_cb(JointState(header=type(self.raw.header)(stamp=rospy.Time.from_sec(self.now)),
            name=self.raw.joint_names, velocity=[velocity]*7))

    def tracker(self, phase, token=None, source='active_trajectory', aborted=False, complete=False):
        token = self.master.header.stamp.to_nsec() if token is None else token
        self.n._execution_tracker_cb(String(data=(
            'execution_stamp_ns={} phase_s={} source={} execution_aborted={} complete={}').format(
                token, phase, source, int(aborted), int(complete))))

    def waiting(self):
        self.n.repair_motion_consistency_enabled = True
        self.n.initial_shell_recovery_enabled = True
        self.n._repair_active_cb(Bool(data=True))
        self.measured()
        self.master = copy.deepcopy(self.raw)
        self.master.header.stamp = rospy.Time.from_sec(self.now-.1)
        q = self.data['measured_q']
        self.master.points = [JointTrajectoryPoint(positions=q, velocities=[.1]*7,
            time_from_start=rospy.Duration(0.)), JointTrajectoryPoint(positions=q, velocities=[0.]*7,
            time_from_start=rospy.Duration(.3)), JointTrajectoryPoint(positions=q, velocities=[0.]*7,
            time_from_start=rospy.Duration(.4))]
        self.n._committed_master = copy.deepcopy(self.master)
        self.tracker(.1)
        self.g._candidate_cb(copy.deepcopy(self.raw))
        self.transfer()
        self.key = next(iter(self.n._admission_entries))
        self.entry = self.n._admission_entries[self.key]
        self.entry['geometric_restore'] = dict(schema=2, raw_candidate_stamp_ns=str(self.raw.header.stamp.to_nsec()),
            query_stamp_ns=str(self.raw.header.stamp.to_nsec()-1), query_ros_ns=str(self.raw.header.stamp.to_nsec()-1),
            mode_epoch='3', observation_token='care_obs_v1_schedule', points=[[.25, 0., .95]])
        self.deadline = self.entry['deadline']
        self.pump()
        self.assertIsNotNone(self.n._pending_raw)
        self.assertEqual(self.n._preflight_reason, 'handoff_certified_brake_pending')
        self.assertFalse(self.n.final_gcdf_query_pub.messages or self.n.committed_pub.messages)

    def test_schedule_phase_event_rebuilds_same_raw_once_without_new_lease(self):
        self.waiting()
        fresh = [x+.0001 for x in self.data['measured_q']]
        self.advance(.01);self.measured(position=fresh);self.tracker(.3)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)
        candidate = self.n.final_gcdf_query_pub.messages[-1]
        self.assertEqual(candidate.points[0].positions, fresh)
        self.assertEqual(candidate.points[0].velocities, [0.]*7)
        self.assertEqual(self.entry['deadline'], self.deadline)
        self.assertEqual(self.n._gcdf_outstanding_diag['transfer_key'], self.key)
        self.assertEqual(self.n._gcdf_outstanding_diag['raw_candidate_stamp_ns'], self.raw.header.stamp.to_nsec())
        self.assertEqual(self.n._handoff_resume_count, 1)
        self.tracker(.35);self.n._admission_tick()
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        # The new motion still has to pass the original final-GCDF gate.
        self.gcdf(False)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)

    def test_schedule_wrong_token_cannot_release_even_with_matching_wall_time(self):
        self.waiting();self.advance(.01);self.measured()
        self.tracker(.4, token=self.master.header.stamp.to_nsec()+1, source='trajectory_complete_hold', complete=True)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.assertEqual(self.n._preflight_reason, 'handoff_tracker_identity')

    def test_schedule_stale_tracker_cannot_release(self):
        self.waiting();self.advance(.051);self.measured()
        self.assertEqual(self.n._preflight_reason, 'handoff_tracker_unavailable')
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_schedule_expiry_and_late_completion_cannot_resurrect(self):
        self.waiting();self.advance(.751, ros=False);self.n._admission_tick();self.pump()
        self.assertEqual(self.entry['stage'], 'expired')
        self.assertEqual(self.entry['deadline'], self.deadline)
        self.measured();self.tracker(.4, source='trajectory_complete_hold', complete=True)
        self.assertFalse(self.n.final_gcdf_query_pub.messages or self.n.committed_pub.messages)

    def test_schedule_mode_change_cancels_wait(self):
        self.waiting();self.n._repair_active_cb(Bool(data=False));self.tracker(.4)
        self.assertEqual(self.entry['stage'], 'cancelled')
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_schedule_owner_change_cancels_wait(self):
        self.waiting();self.n._preflight_owner_cb(String(data='active_q_vis_token=care_obs_v1_other'))
        self.tracker(.4)
        self.assertEqual(self.entry['stage'], 'cancelled')
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_schedule_stopped_reference_still_waits_for_changing_real_state(self):
        self.waiting();self.advance(.01);self.measured(velocity=.1);self.tracker(.3)
        self.assertEqual(self.n._preflight_reason, 'handoff_state_changing')
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.advance(.01);self.measured();self.tracker(.3)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)

    def test_schedule_completed_token_is_retained_for_last_reference_hold(self):
        self.waiting();self.advance(.01);self.measured(velocity=.1)
        self.tracker(.4, source='trajectory_complete_hold', complete=True)
        self.tracker(0., token=0, source='last_reference_hold')
        self.measured()
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)

    def test_schedule_last_reference_uses_real_state_instead_of_old_endpoint(self):
        self.waiting();self.measured(velocity=.1, position=[x+.01 for x in self.data['measured_q']])
        self.tracker(0., token=0, source='last_reference_hold')
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.measured(position=[x+.01 for x in self.data['measured_q']])
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)

    def test_schedule_duplicate_tracker_fields_do_not_release(self):
        self.waiting();self.n._execution_tracker_cb(String(data='execution_stamp_ns=0 execution_stamp_ns=1 phase_s=.4'))
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_schedule_external_vbc_pause_rebuilds_without_waiting_for_frozen_tail(self):
        self.waiting();self.advance(.01);self.measured()
        self.tracker(.1, source='external_vbc_hold')
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)
        self.assertEqual(self.entry['deadline'], self.deadline)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.gcdf(False)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)

    def test_schedule_external_gcdf_pause_still_needs_full_vbc_after_gcdf(self):
        self.waiting();self.advance(.01);self.measured()
        self.tracker(.1, source='external_gcdf_hard_hold')
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)
        self.gcdf()
        self.assertEqual(len(self.n.verification_pub.messages), 1)
        self.assertFalse(self.n.committed_pub.messages)
        key=self.n._outstanding_diag['transfer_key'];stamp=self.n._outstanding.header.stamp.to_nsec()
        self.n._global_summary_cb(String(data='trajectory_source=predicted trajectory_stamp_ns={} has_violation=1 '
            'context_request_session={} context_request_sequence={}'.format(stamp,key[0],key[1])))
        self.assertFalse(self.n.committed_pub.messages)

    def test_schedule_recorded_certificate_controls_state_variation(self):
        self.waiting()
        e=dict(joint_motion_radii=[1.]*7, motion_horizon_s=.5, certified_residual_m=.002)
        self.n._handoff_certificate = (self.n._handoff_scope(self.entry), e)
        self.advance(.01);self.measured(velocity=.004);self.tracker(.3)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.advance(.01);self.measured(velocity=.004)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)
        self.assertEqual(self.entry['deadline'], self.deadline)


class FormulaTests(unittest.TestCase):
    def test_observed_latency_projects_change_beyond_one_sample(self):
        e=dict(joint_motion_radii=[1.]*7,motion_horizon_s=.45,
               certified_residual_m=.0002,handoff_latency_s=.094)
        previous=(1000000000,[0.]*7,[.001]*7)
        current=(1010000000,[.000005]*7,[.001]*7)
        # One10ms change fits; projecting it over the recorded94ms does not.
        self.assertTrue(state_variation_ready(previous,current,[.001]*7,
            dict(e,handoff_latency_s=.01),[3.]*7,2.5))
        self.assertFalse(state_variation_ready(previous,current,[.001]*7,e,[3.]*7,2.5))
        stationary=(1010000000,[0.]*7,[.001]*7)
        self.assertTrue(state_variation_ready(previous,stationary,[.001]*7,e,[3.]*7,2.5))

    def test_latency_metadata_is_finite_and_does_not_weaken_short_sample_check(self):
        e=dict(joint_motion_radii=[1.]*7,motion_horizon_s=.45,certified_residual_m=.0002)
        previous=(1000000000,[0.]*7,[.001]*7);current=(1010000000,[.0001]*7,[.001]*7)
        for delay in [-1.,float('nan'),float('inf'),0.,.001]:
            self.assertFalse(state_variation_ready(previous,current,[.001]*7,
                dict(e,handoff_latency_s=delay),[3.]*7,2.5))

    def test_no_synthetic_zero_marker_counts_as_hold(self):
        p=lambda q,v,t:JointTrajectoryPoint(positions=[q]*7,velocities=[v]*7,time_from_start=rospy.Duration(t))
        self.assertIsNone(terminal_hold_start([p(0., .1, 0.), p(.1, 0., .5)]))
        self.assertEqual(terminal_hold_start([p(0., .1, 0.), p(.1, 0., .5), p(.1, 0., .6)]), .5)

    def test_state_formula_rejects_actuator_mismatch_and_stale_or_repeated_samples(self):
        e=dict(joint_motion_radii=[1.]*7,motion_horizon_s=.5,certified_residual_m=.002)
        q=[0.]*7;previous=(1000000000,q,q);current=(1010000000,q,q)
        self.assertTrue(state_variation_ready(previous,current,q,e,[3.]*7,2.5))
        self.assertFalse(state_variation_ready(previous,current,[.01]*7,e,[3.]*7,2.5))
        self.assertFalse(state_variation_ready(previous,previous,q,e,[3.]*7,2.5))
        self.assertFalse(state_variation_ready(previous,(1060000000,q,q),q,e,[3.]*7,2.5))


def load_tests(loader, tests, pattern):
    selected=[name for name in loader.getTestCaseNames(ScheduleTests) if name.startswith('test_schedule_')]
    return unittest.TestSuite([ScheduleTests(name) for name in selected]+[loader.loadTestsFromTestCase(FormulaTests)])


if __name__ == '__main__':
    unittest.main()
