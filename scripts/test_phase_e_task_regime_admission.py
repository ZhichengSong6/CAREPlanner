#!/usr/bin/env python3
"""Actual gate/continuity callbacks with bound automatic regime revisions."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.append('/usr/lib/python3/dist-packages')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
import rospy
from std_msgs.msg import Bool, String, Header
from sensor_msgs.msg import JointState
from care_velocity_controller.msg import CommandStatus
from task_recovery_events import TaskEventSource, regime_identity
from task_observation_binding import canonical
from candidate_admission import payload_sha256
import test_phase_e_task_context as task_fixture
from test_phase_e_candidate_admission import Pub
from test_phase_e_task_recovery import snapshot


class RegimeAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=task_fixture.AdmissionBindingTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.old=self.fixture.old;self.g=self.old.g;self.n=self.old.n
        self.context=self.fixture.first;self.binding=self.context['candidate_binding']
        self.raw=copy.deepcopy(self.old.raw)
        self.g.require_bound_regime=self.n.require_bound_regime=True
        self.n.require_execution_preparation=True
        self.n.execution_prepared_pub=Pub();self.n.execution_prepared_status_pub=Pub()
        self.g.task_status_pub=Pub()
        self.source=TaskEventSource('regime-unit')
        self.control()
        self.state()

    def state(self,session=1,applied=None):
        message=CommandStatus(header=Header(stamp=rospy.Time.now()),
            execution_lease_required=True,controller_session=session,
            applied_velocity=applied or [0.]*7,velocity_limits=[2.]*7)
        self.n._controller_state_cb(message)
        return message

    def control(self,mode='NORMAL',revision=1):
        event=self.source.event('regime_control',self.context,
            dict(mode=mode,mode_revision=revision,ready=True,clear=False,replan_seq=0,
                 reference_refresh_seq=0,force_bootstrap=False,qp_retry=''),
            rospy.Time.now().to_nsec())
        self.key=regime_identity(event)
        msg=String(data=canonical(event))
        self.g._task_control_cb(msg);self.n._task_control_cb(msg)

    def start(self):
        self.g._candidate_cb(self.raw,'',self.binding,self.key)
        self.old.pump()

    def test_current_regime_still_requires_and_passes_both_geometry_gates(self):
        self.start()
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
        self.assertFalse(self.n.committed_pub.messages)
        self.old.certify()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(len(self.n.execution_prepared_pub.messages),1)
        self.assertEqual(self.n.execution_prepared_pub.messages[0].action,'execution_prepare')
        self.assertIn('UNMET',self.n.execution_prepared_pub.messages[0].planned_prefix_context)

    def test_same_trajectory_different_regime_has_different_digest(self):
        self.assertNotEqual(payload_sha256(self.raw,'',self.binding,self.key),
                            payload_sha256(self.raw,'',self.binding,'other:1:NORMAL'))

    def test_late_old_mode_candidate_is_rejected_inside_same_task(self):
        previous=self.key
        self.control('REPAIR',2)
        self.g._candidate_cb(self.raw,'',self.binding,previous)
        self.assertIsNone(self.g.sender.active)
        self.assertFalse(self.g.transfer_pub.messages)

    def test_changing_regime_without_changing_hash_is_rejected_at_receiver(self):
        self.g._candidate_cb(self.raw,'',self.binding,self.key)
        transfer=self.g.transfer_pub.messages.pop()
        transfer.task_regime='different:2:REPAIR'
        self.n._transfer_cb(transfer)
        entry=self.n._admission_entries[(transfer.session,transfer.sequence)]
        self.assertEqual(entry['stage'],'rejected')
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_expired_control_cannot_forward_or_commit(self):
        self.start();self.old.advance(.251)
        self.g._admission_tick();self.n._admission_expire_locked()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertTrue(all(e['stage'] in ('cancelled','expired','rejected','terminal')
                            for e in self.n._admission_entries.values()))
        before=len(self.g.transfer_pub.messages)
        self.g._candidate_cb(self.raw,'',self.binding,self.key)
        self.assertEqual(len(self.g.transfer_pub.messages),before)

    def test_raw_mode_bools_cannot_override_bound_normal(self):
        self.g._repair_active_cb(Bool(data=True));self.g._probe_active_cb(Bool(data=True))
        self.n._repair_active_cb(Bool(data=True));self.n._probe_active_cb(Bool(data=True))
        self.assertFalse(self.g._repair_active);self.assertFalse(self.g.probe_active)
        self.assertFalse(self.n._repair_active);self.assertFalse(self.n._probe_active)

    def test_mode_change_while_waiting_vbc_discards_late_safe_verdict(self):
        self.start();self.old.gcdf()
        self.assertTrue(self.n.verification_pub.messages)
        self.control('REPAIR',2)
        self.old.vbc()
        self.assertFalse(self.n.committed_pub.messages)

    def test_raw_observation_owner_cannot_change_preflight(self):
        self.n._preflight_owner_cb(String(data='active_q_vis_token=care_obs_v1_99_'+'a'*24))
        self.assertIsNone(self.n._preflight_owner_token)

    def test_bound_observation_owner_uses_same_task_snapshot(self):
        event=TaskEventSource('scheduler-unit').event('recovery_snapshot',self.context,snapshot(),
                                                    rospy.Time.now().to_nsec())
        self.n._task_recovery_cb(String(data=canonical(event)))
        self.assertEqual(self.n._preflight_owner_token,snapshot()['target']['observation_token'])

    def test_command_request_topic_cannot_impersonate_applied_controller_state(self):
        self.n._actuator_state_cb(JointState(header=Header(stamp=rospy.Time.now()),
            name=self.raw.joint_names,velocity=[1.]*7))
        self.assertEqual(list(self.n._latest_actuator_velocity.values()),[0.]*7)

    def test_retired_controller_session_cannot_replace_new_applied_receipt(self):
        self.old.advance(.001);self.state(2,[.1]*7)
        self.old.advance(.001);self.state(1,[.2]*7)
        self.assertEqual(self.n._controller_session,2)
        self.assertEqual(list(self.n._latest_actuator_velocity.values()),[.1]*7)

    def test_faulted_controller_receipt_does_not_refresh_preflight_age(self):
        stamp=self.n._latest_actuator_stamp_ns
        self.old.advance(.001)
        message=CommandStatus(header=Header(stamp=rospy.Time.now()),clock_fault=True,
            execution_lease_required=True,controller_session=1,applied_velocity=[0.]*7,velocity_limits=[2.]*7)
        self.n._controller_state_cb(message)
        self.assertEqual(self.n._latest_actuator_stamp_ns,stamp)

    def test_pending_candidate_metadata_is_an_atomic_copy(self):
        receiver=self.g.task_context
        now=rospy.Time.now().to_nsec()
        self.raw.header.stamp=rospy.Time(0,now)
        receiver.offer_candidate(self.raw,self.binding,now,dict(prefix_context='first',task_regime=self.key))
        first=receiver.candidate_envelope(now)
        newer=copy.deepcopy(self.raw);newer.points[0].positions[0]+=.001
        receiver.offer_candidate(newer,self.binding,now,dict(prefix_context='second',task_regime='new:2:REPAIR'))
        received=[]
        self.g._candidate_cb=lambda *args:received.append(args)
        self.g._consume_task_candidate(first)
        self.assertEqual(received[0][1],'first');self.assertEqual(received[0][3],self.key)
        self.assertNotEqual(received[0][0].points[0].positions[0],newer.points[0].positions[0])


if __name__=='__main__':unittest.main()
