#!/usr/bin/env python3
"""Task events and actual C4 callbacks, including retained execution safety."""
import copy
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.append('/usr/lib/python3/dist-packages')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src/care_visibility_cdf/scripts'),
               str(ROOT/'src/egocentric_arm_planner/scripts')]
import rospy
from std_msgs.msg import Bool, String
from task_observation_binding import ObservationTaskOwner, TaskContextReceiver, canonical
from task_recovery_events import (TaskEventSource, TaskEventReceiver, RecoverySnapshotSource,
                                  validate_recovery_snapshot, validate_regime_control)
import c4_4_verified_regime_manager_node as regime
from test_phase_e_task_context import command
from test_phase_e_execution_vbc_hold import Pub


def target(oid=1):
    return dict(obligation_id=oid, observation_token='care_obs_v1_'+str(oid)+'_'+'a'*24,
                region_token='care_region_v1_'+str(oid)+'_'+'b'*24, q_vis=[0.]*7,
                sensor_id=-1,
                joint_mask=[1.]*7, points=[[.3, 0., .5]], deadline_ns=20000000000)


def snapshot(episode='c'*32, complete=False, pending=1, pose=True):
    return dict(episode_id=episode, started=True, complete=complete,
                pending_count=pending, target=target() if pose else None)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.now = 19000000000
        self.owner = ObservationTaskOwner('owner')
        self.context = self.owner.command(command(), self.now)[-1]
        self.source = TaskEventSource('scheduler')
        self.receiver = TaskEventReceiver()

    def event(self, kind='recovery_snapshot', payload=None):
        return self.source.event(kind, self.context, payload or snapshot(), self.now)

    def test_event_cannot_cross_goal_even_with_later_source_stamp(self):
        event = self.event()
        other = self.owner.command(command('B'), self.now+1)[-1]
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            self.receiver.parse(canonical(event), other, self.now+1, 'recovery_snapshot')

    def test_duplicate_does_not_refresh_event_age_or_sequence(self):
        event = self.receiver.parse(canonical(self.event()), self.context, self.now, 'recovery_snapshot')
        self.receiver.remember(event)
        with self.assertRaisesRegex(ValueError, 'replayed'):
            self.receiver.parse(canonical(event), self.context, self.now+1, 'recovery_snapshot')
        with self.assertRaisesRegex(ValueError, 'expired'):
            TaskEventReceiver().parse(canonical(event), self.context, self.now+250000001, 'recovery_snapshot')

    def test_future_event_cannot_be_consumed_before_clock(self):
        event = self.event(); event['stamp_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'future'):
            self.receiver.parse(canonical(event), self.context, self.now, 'recovery_snapshot')

    def test_sequence_order_is_independent_between_connection_kinds(self):
        first = self.event('first', dict(value=1))
        second = self.event('second', dict(value=2))
        self.receiver.remember(self.receiver.parse(canonical(second), self.context, self.now, 'second'))
        self.receiver.remember(self.receiver.parse(canonical(first), self.context, self.now, 'first'))

    def test_malformed_payload_does_not_consume_source_sequence(self):
        event = self.event(); event['payload']['pending_count'] = -1
        parsed = self.receiver.parse(canonical(event), self.context, self.now, 'recovery_snapshot')
        with self.assertRaises(ValueError): validate_recovery_snapshot(parsed['payload'])
        self.assertFalse(self.receiver.sequences)

    def test_completion_requires_started_empty_ledger_and_no_target(self):
        for changes in ({'started':False}, {'pending_count':1}, {'target':target()}):
            p = dict(snapshot(complete=True, pending=0, pose=False), **changes)
            with self.assertRaises(ValueError): validate_recovery_snapshot(p)

    def test_snapshot_rejects_wrong_region_mask_and_nonfinite_pose(self):
        for changes in ({'region_token':'care_region_v1_2_'+'b'*24},
                        {'joint_mask':[0.]*7}, {'q_vis':[float('nan')]*7},
                        {'points':[[0., 0.]]}):
            p = snapshot(); p['target'].update(changes)
            with self.assertRaises(ValueError): validate_recovery_snapshot(p)

    def test_source_cancellation_and_replacement_do_not_complete_episode(self):
        source = RecoverySnapshotSource()
        first = source.snapshot(self.context, 1, target(), self.now)
        source.reset()
        other = self.owner.command(command('B'), self.now+1)[-1]
        empty = source.snapshot(other, 0, None, self.now+1)
        self.assertFalse(empty['payload']['started']); self.assertFalse(empty['payload']['complete'])
        self.assertNotEqual(first['task_context'], empty['task_context'])

    def test_reopened_demand_has_new_episode_and_preserves_target_deadline(self):
        source = RecoverySnapshotSource()
        first = source.snapshot(self.context, 1, target(), self.now)
        complete = source.snapshot(self.context, 0, None, self.now+1)
        reopened = source.snapshot(self.context, 1, target(), self.now+2)
        self.assertTrue(complete['payload']['complete'])
        self.assertNotEqual(first['payload']['episode_id'], reopened['payload']['episode_id'])
        self.assertEqual(first['payload']['target']['deadline_ns'], reopened['payload']['target']['deadline_ns'])

    def test_replay_protection_has_explicit_source_capacity(self):
        for n in range(16):
            event = TaskEventSource('source'+str(n)).event('recovery_snapshot',self.context,snapshot(),self.now)
            self.receiver.remember(self.receiver.parse(canonical(event),self.context,self.now,'recovery_snapshot'))
        event = TaskEventSource('overflow').event('recovery_snapshot',self.context,snapshot(),self.now)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            self.receiver.parse(canonical(event),self.context,self.now,'recovery_snapshot')

    def test_control_requires_exact_modes_and_boolean_types(self):
        good = dict(mode='NORMAL',mode_revision=1,ready=True,clear=False,replan_seq=0,
                    reference_refresh_seq=0,force_bootstrap=False,qp_retry='')
        validate_regime_control(good)
        for changes in ({'mode':'RECOVERY_HOLD'}, {'ready':1}, {'replan_seq':-1}):
            with self.assertRaises(ValueError): validate_regime_control(dict(good, **changes))


class RegimeTests(unittest.TestCase):
    def setUp(self):
        self.now = 19.
        params = dict(enable_observation_context=True)
        for mock in [patch.object(rospy, 'get_param', side_effect=lambda k,d=None:params.get(k.lstrip('~'),d)),
                     patch.object(rospy, 'Publisher', Pub), patch.object(rospy, 'Subscriber'),
                     patch.object(rospy, 'Timer'), patch.object(rospy, 'logwarn'),
                     patch.object(rospy, 'loginfo'),
                     patch.object(rospy.Time, 'now', side_effect=lambda:rospy.Time.from_sec(self.now))]:
            mock.start(); self.addCleanup(mock.stop)
        self.m = regime.C44VerifiedRegimeManager()
        self.m.execution_ready = True
        self.owner = ObservationTaskOwner('owner')
        self.a = copy.deepcopy(self.owner.command(command(), self.ns())[-1])
        self.source = TaskEventSource('scheduler')
        self.context(self.a)

    def ns(self): return rospy.Time.from_sec(self.now).to_nsec()
    def context(self, event): self.m._task_context_cb(String(data=canonical(event)))
    def send(self, payload=None, context=None, kind='recovery_snapshot'):
        if kind in ('candidate_outcome', 'planner_status'):
            payload = dict(payload, task_regime=self.m.task_event_source.session+':'+
                           str(self.m.task_mode_revision)+':'+self.m.state)
        event = self.source.event(kind, context or self.a, payload or snapshot(), self.ns())
        self.m._task_event_cb(String(data=canonical(event)), kind)
        return event
    def replace(self):
        self.now += .001
        events = self.owner.command(command('B'), self.ns())
        for event in events: self.context(event)
        return copy.deepcopy(events[-1])

    def test_actual_snapshot_automatically_enters_repair(self):
        self.send()
        self.assertEqual(self.m.state, self.m.REPAIR)
        self.assertEqual(self.m.observation_takeover_count,1)
        control = json.loads(self.m.task_control_pub.messages[-1].data)
        self.assertEqual(control['task_context'],self.a['candidate_binding'])
        self.assertEqual(control['payload']['mode'],'REPAIR')

    def test_actual_same_episode_completion_enters_probe(self):
        self.send(); self.send(snapshot(complete=True,pending=0,pose=False))
        self.assertEqual(self.m.state,self.m.PROBE_NORMAL)
        self.assertEqual(self.m.repair_completion_event_count,1)

    def test_completion_for_another_episode_cannot_clear_repair(self):
        self.send(); self.send(snapshot(episode='d'*32,complete=True,pending=0,pose=False))
        self.assertEqual(self.m.state,self.m.REPAIR)

    def test_raw_latched_completion_cannot_clear_task_repair(self):
        self.send(); self.m._repair_completion_cb(Bool(data=True))
        self.assertEqual(self.m.state,self.m.REPAIR)

    def test_callback_dispatch_permission_is_local_to_the_receiving_thread(self):
        self.send()
        self.m.task_event_dispatch.active = True
        worker = threading.Thread(target=lambda:self.m._repair_completion_cb(Bool(data=True)))
        worker.start(); worker.join(timeout=1.)
        self.m.task_event_dispatch.active = False
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.m.state,self.m.REPAIR)

    def test_empty_initial_snapshot_does_not_complete_task(self):
        self.send(dict(episode_id='',started=False,complete=False,pending_count=0,target=None))
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.assertEqual(self.m.repair_completion_event_count,0)

    def test_heartbeat_does_not_reset_episode_or_deadline(self):
        self.send(); deadline=self.m.physical_deadline_abs
        self.now += .1; self.context(self.owner.tick(self.ns())[0])
        self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertEqual(self.m.task_observation_episode,'c'*32)
        self.assertEqual(self.m.physical_deadline_abs,deadline)

    def test_replacement_clears_old_mode_and_old_completion_cannot_revive(self):
        self.send(); b=self.replace()
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.send(snapshot(complete=True,pending=0,pose=False), context=self.a)
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.send(context=b)
        self.assertEqual(self.m.state,self.m.REPAIR)

    def test_duplicate_event_cannot_reopen_finished_episode(self):
        old=self.send(); self.send(snapshot(complete=True,pending=0,pose=False))
        self.m._task_event_cb(String(data=canonical(old)),'recovery_snapshot')
        self.assertEqual(self.m.state,self.m.PROBE_NORMAL)

    def test_new_episode_can_take_over_idle_probe(self):
        self.send(); self.send(snapshot(complete=True,pending=0,pose=False))
        self.send(snapshot(episode='d'*32))
        self.assertEqual(self.m.state,self.m.REPAIR)

    def test_context_expiry_clears_planning_without_fake_completion(self):
        self.send(); self.now += .801; self.m._timer_cb(None)
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.assertIsNone(self.m.task_binding)
        self.assertEqual(self.m.repair_completion_event_count,0)

    def test_snapshot_before_context_is_bounded_and_later_consumed(self):
        self.now += .001
        b=copy.deepcopy(self.owner.command(command('B'),self.ns())[-1])
        self.send(context=b)
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.context(b)
        self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertEqual(self.m.task_binding,b['candidate_binding'])

    def test_old_candidate_even_with_higher_seq_cannot_modify_new_task(self):
        b=self.replace()
        self.send(dict(text='seq=999 result=unsafe committed=0'),context=self.a,kind='candidate_outcome')
        self.assertEqual(self.m.candidate_event_count,0)
        self.send(dict(text='seq=1 result=safe committed=0'),context=b,kind='candidate_outcome')
        self.assertEqual(self.m.candidate_event_count,1)

    def test_old_execution_audit_still_stops_old_token_after_task_replacement(self):
        token=self.ns()
        self.send(dict(execution_stamp_ns=token,trajectory_sha256='a'*64),kind='execution_binding')
        self.m._tracker_summary_cb(String(data='active=1 complete=0 execution_stamp_ns='+str(token)))
        b=self.replace()
        audit=token+1
        self.m._execution_audit_identity_cb(String(data='audit_trajectory_stamp_ns='+str(audit)+' execution_stamp_ns='+str(token)))
        for _ in range(2):
            self.m._execution_summary_cb(String(data='success=1 trajectory_source=predicted has_violation=1 trajectory_stamp_ns='+str(audit)))
        self.assertTrue(self.m.execution_vbc_stop_pub.messages)
        self.assertIn(str(token),self.m.execution_vbc_stop_pub.messages[-1].data)
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.assertEqual(self.m.task_binding,b['candidate_binding'])


if __name__ == '__main__': unittest.main()
