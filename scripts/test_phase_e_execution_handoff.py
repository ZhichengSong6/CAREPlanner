#!/usr/bin/env python3
"""Immutable staging, deadline consumption and absence of motion authority."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/egocentric_arm_planner/scripts'),str(ROOT/'src/care_visibility_cdf/scripts')]
import rospy
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from care_velocity_controller.msg import CommandStatus
from candidate_admission import payload_sha256
from execution_handoff import ExecutionHandoffShadow,QUALIFICATION_GAPS
from execution_preparation import ExecutionPreparationLedger
from task_observation_binding import ObservationTaskOwner,canonical
from task_recovery_events import TaskEventSource
from test_phase_e_task_context import command

NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
VELOCITY=[2.]*7
ACCELERATION=[3.,3.,4.,4.,6.,6.,6.]

def fixture(now=19000000000):
    owner=ObservationTaskOwner('handoff-owner')
    context=owner.command(command(),now)[-1]
    control=TaskEventSource('regime').event('regime_control',context,dict(mode='NORMAL',mode_revision=1,
        ready=True,clear=False,replan_seq=0,reference_refresh_seq=0,force_bootstrap=False,qp_retry=''),now)
    curve=JointTrajectory();curve.header.frame_id='base_link';curve.header.stamp=rospy.Time(0,now);curve.joint_names=NAMES
    for t,q,v in [(0.,0.,0.),(.1,.002,.04),(.2,.004,0.)]:
        p=JointTrajectoryPoint();p.positions=[0.]*7;p.positions[1]=q;p.velocities=[0.]*7;p.velocities[1]=v
        p.time_from_start=rospy.Duration(t);curve.points.append(p)
    ledger=ExecutionPreparationLedger()
    prepared=ledger.prepare(curve,context['candidate_binding'],'regime:1:NORMAL',('source',1),
        payload_sha256(curve,'',context['candidate_binding'],'regime:1:NORMAL'),1,now,now+200000000)
    state=CommandStatus();state.header.stamp=rospy.Time(0,now);state.controller_session=11
    state.execution_lease_required=True;state.velocity_limits=VELOCITY;state.acceleration_limits=ACCELERATION
    state.integral_valid=True;state.max_execution_lease_s=.1;state.max_braking_step_s=.02
    return context,control,prepared,state


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.now=19000000000;self.wall=1.
        self.context,self.control,self.prepared,self.state=fixture(self.now)
        self.shadow=ExecutionHandoffShadow(NAMES,VELOCITY,ACCELERATION,wall_clock=lambda:self.wall)
        self.shadow.update_context(self.context,self.now)
        self.shadow.update_control(canonical(self.control),self.now)
        self.shadow.observe_controller(self.state,self.now)

    def rehash(self,message):
        message.payload_sha256=payload_sha256(message.trajectory,message.planned_prefix_context,message.task_context,message.task_regime)
        return message

    def test_valid_preparation_stages_but_all_qualifications_remain_unmet(self):
        staged=self.shadow.accept(self.prepared,self.now)
        self.assertEqual(staged.payload_sha256,self.prepared.payload_sha256)
        report=self.shadow.report(self.now)
        self.assertTrue(report['staged']);self.assertFalse(report['execution_authorized'])
        self.assertFalse(report['runtime_admission']);self.assertFalse(report['production_issuer_enabled'])
        self.assertEqual(set(report['missing_qualification']),set(QUALIFICATION_GAPS))

    def test_repetition_preserves_first_ros_and_wall_deadline(self):
        self.shadow.accept(self.prepared,self.now);pending=copy.deepcopy(self.shadow.pending)
        self.wall+=.01
        self.assertIsNone(self.shadow.accept(self.prepared,self.now+10000000))
        self.assertEqual(self.shadow.pending['expires_ns'],pending['expires_ns'])
        self.assertEqual(self.shadow.pending['wall_until'],pending['wall_until'])

    def test_repeat_after_application_does_not_need_waiting_again(self):
        self.shadow.accept(self.prepared,self.now)
        self.state.mode=1;self.state.header.stamp=rospy.Time(0,self.now+1)
        self.shadow.observe_controller(self.state,self.now+1)
        self.assertIsNone(self.shadow.accept(self.prepared,self.now+1))

    def test_raw_committed_action_is_rejected(self):
        self.prepared.action='execution_activate'
        with self.assertRaisesRegex(ValueError,'identity'):self.shadow.accept(self.prepared,self.now)

    def test_curve_mutation_without_new_digest_is_rejected(self):
        self.prepared.trajectory.points[1].positions[1]+=.001
        with self.assertRaisesRegex(ValueError,'digest'):self.shadow.accept(self.prepared,self.now)

    def test_same_transport_identity_cannot_change_verified_curve(self):
        self.shadow.accept(self.prepared,self.now)
        self.prepared.trajectory.points[0].positions[0]+=.01
        self.prepared.trajectory.points[1].positions[0]+=.01
        self.prepared.trajectory.points[2].positions[0]+=.01
        with self.assertRaisesRegex(ValueError,'mutated'):self.shadow.accept(self.rehash(self.prepared),self.now)

    def test_wrong_task_is_rejected_even_with_matching_payload_digest(self):
        self.prepared.task_context=self.prepared.task_context.replace('goal-A','goal-B')+' '
        with self.assertRaisesRegex(ValueError,'identity'):self.shadow.accept(self.rehash(self.prepared),self.now)

    def test_wrong_regime_is_rejected_even_with_matching_payload_digest(self):
        self.prepared.task_regime='regime:2:REPAIR'
        with self.assertRaisesRegex(ValueError,'identity'):self.shadow.accept(self.rehash(self.prepared),self.now)

    def test_cancel_preserves_preparation_replay_history(self):
        self.shadow.accept(self.prepared,self.now);self.shadow.cancel('cancelled')
        with self.assertRaisesRegex(ValueError,'retired'):self.shadow.accept(self.prepared,self.now)

    def test_ros_expiry_cannot_be_refreshed_by_retransmission(self):
        self.shadow.accept(self.prepared,self.now)
        self.shadow.report(self.now+200000000)
        self.assertIsNone(self.shadow.pending)
        with self.assertRaises(ValueError):self.shadow.accept(self.prepared,self.now+200000001)

    def test_paused_ros_clock_does_not_extend_wall_window(self):
        self.shadow.accept(self.prepared,self.now);self.wall+=.201
        self.shadow.report(self.now);self.assertIsNone(self.shadow.pending)
        with self.assertRaisesRegex(ValueError,'retired'):self.shadow.accept(self.prepared,self.now)

    def test_clock_reversal_cannot_reopen_preparation_session(self):
        self.shadow.accept(self.prepared,self.now)
        self.shadow.report(self.now-1)
        with self.assertRaisesRegex(ValueError,'clock_reversed'):self.shadow.accept(self.prepared,self.now+1)

    def test_mode_revision_invalidates_pending_and_late_offer(self):
        self.shadow.accept(self.prepared,self.now)
        e=copy.deepcopy(self.control);e['event_seq']+=1;e['payload']['mode_revision']=2
        self.shadow.update_control(canonical(e),self.now)
        self.assertIsNone(self.shadow.pending)
        with self.assertRaises(ValueError):self.shadow.accept(self.prepared,self.now)

    def test_controller_restart_invalidates_pending_and_rejects_old_session(self):
        self.shadow.accept(self.prepared,self.now)
        fresh=copy.deepcopy(self.state);fresh.controller_session=12
        self.assertTrue(self.shadow.observe_controller(fresh,self.now))
        self.assertIsNone(self.shadow.pending)
        self.assertFalse(self.shadow.observe_controller(self.state,self.now))

    def test_nonzero_or_stale_controller_state_prevents_staging(self):
        self.state.applied_velocity[0]=.001;self.state.header.stamp=rospy.Time(0,self.now+1)
        self.shadow.observe_controller(self.state,self.now+1)
        with self.assertRaisesRegex(ValueError,'fresh_waiting_rest'):self.shadow.accept(self.prepared,self.now+1)

    def test_pending_cannot_be_relabelled_qualified(self):
        meta=json.loads(self.prepared.planned_prefix_context);meta['execution_qualification']='QUALIFIED'
        self.prepared.planned_prefix_context=canonical(meta)
        with self.assertRaisesRegex(ValueError,'metadata'):self.shadow.accept(self.rehash(self.prepared),self.now)

    def test_preparation_cannot_outlive_captured_control(self):
        meta=json.loads(self.prepared.planned_prefix_context);meta['expires_ns']=self.now+250000001
        self.prepared.planned_prefix_context=canonical(meta)
        with self.assertRaisesRegex(ValueError,'deadline'):self.shadow.accept(self.rehash(self.prepared),self.now)

    def test_extra_nonfinite_curve_fields_are_rejected(self):
        self.prepared.trajectory.points[0].accelerations=[float('nan')]*7
        with self.assertRaisesRegex(ValueError,'accelerations'):self.shadow.accept(self.rehash(self.prepared),self.now)

    def test_source_capacity_does_not_evict_replay_history(self):
        for i in range(16):
            message=copy.deepcopy(self.prepared);message.session='session'+str(i)
            self.shadow.accept(message,self.now)
        self.prepared.session='overflow'
        with self.assertRaisesRegex(ValueError,'capacity'):self.shadow.accept(self.prepared,self.now)
        self.assertEqual(len(self.shadow.source_sequences),16)

    def test_older_same_regime_heartbeat_caps_local_window_without_mutating_payload(self):
        self.control['event_seq']+=1;self.control['valid_until_ns']=self.now+150000000
        self.shadow.update_control(canonical(self.control),self.now)
        value=self.shadow.accept(self.prepared,self.now)
        self.assertEqual(value.payload_sha256,self.prepared.payload_sha256)
        self.assertEqual(json.loads(value.planned_prefix_context)['expires_ns'],self.now+200000000)
        self.assertEqual(self.shadow.pending['expires_ns'],self.now+150000000)

    def test_later_heartbeat_cannot_postpone_locally_capped_first_deadline(self):
        self.control['event_seq']+=1;self.control['valid_until_ns']=self.now+150000000
        self.shadow.update_control(canonical(self.control),self.now)
        self.shadow.accept(self.prepared,self.now)
        self.control['event_seq']+=1;self.control['valid_until_ns']=self.now+250000000
        self.shadow.update_control(canonical(self.control),self.now)
        self.assertIsNone(self.shadow.accept(self.prepared,self.now))
        self.assertEqual(self.shadow.pending['expires_ns'],self.now+150000000)
        self.shadow.report(self.now+150000000)
        self.assertIsNone(self.shadow.pending)


if __name__=='__main__':unittest.main()
