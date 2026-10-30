#!/usr/bin/env python3
"""Fault injection through real candidate/GCDF/VBC/commit callbacks.

Uses the recorded CASE007 raw payload. Synthetic certification responses verify
protocol ownership, not the geometry or original TCP failure.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/egocentric_arm_planner/scripts'))
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from care_collision_cdf.msg import CollisionCDFConstraintBatch
from egocentric_arm_planner.msg import CandidateTransfer
import optimized_trajectory_continuity_node as receiver_module
import probe_single_flight_gate_node as sender_module
from candidate_admission import payload_sha256


class Pub:
    def __init__(self, *a, **kw):
        self.messages = []
    def publish(self, msg):
        self.messages.append(copy.deepcopy(msg))
    def get_num_connections(self):
        return 1


def load_fixture():
    data = json.loads((ROOT/'scripts/fixtures/phase_e_case007_candidate.json').read_text())
    raw = JointTrajectory()
    raw.header.stamp = rospy.Time(0, data['raw_stamp_ns'])
    raw.header.frame_id = data['frame_id']
    raw.joint_names = data['joint_names']
    for fields in data['points']:
        p = JointTrajectoryPoint(time_from_start=rospy.Duration(0, fields['time_ns']))
        for field in ('positions', 'velocities', 'accelerations', 'effort'):
            setattr(p, field, fields[field])
        raw.points.append(p)
    return data, raw


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.wall = 100.0
        self.now = 8.453
        self.params = {'~candidate_admission_enabled': True, '~input_topic': '/candidate',
                       '~output_topic': '/candidate', '~final_gcdf_enabled': True,
                       '~final_gcdf_safety_margin': .01,
                       '~execution_audit_enabled': True,
                       '~smooth_handoff_rebase_enabled': True,
                       '~repair_prefix_verification_enabled': True,
                       '~repair_motion_consistency_enabled': False,
                       '~repair_execution_prefix_s': .25}
        patches = [patch.object(rospy, 'get_param', side_effect=lambda k,d=None:self.params.get(k,d)),
                   patch.object(rospy, 'Publisher', Pub), patch.object(rospy, 'Subscriber'),
                   patch.object(rospy, 'Timer'), patch.object(rospy, 'on_shutdown'),
                   patch.object(rospy, 'logwarn'), patch.object(rospy, 'logerr_throttle'),
                   patch.object(rospy.Time, 'now', side_effect=lambda:rospy.Time.from_sec(self.now)),
                   patch.object(receiver_module.time, 'monotonic', side_effect=lambda:self.wall),
                   patch.object(receiver_module, 'WallWorker'),
                   patch.object(sender_module, 'WallWorker')]
        for p in patches:
            p.start(); self.addCleanup(p.stop)
        self.n = receiver_module.OptimizedTrajectoryContinuityNode()
        self.g = sender_module.ProbeSingleFlightGate()
        self.data, self.raw = load_fixture()
        self.n._joint_state_cb(JointState(name=self.raw.joint_names, position=self.data['measured_q']))
        self.g._admission_status_cb(String(data='action=pong ready_nonce=test receiver_boot_id='+self.n._admission_boot_id))

    def advance(self, dt, ros=True):
        self.wall += dt
        if ros:
            self.now += dt

    def transfer(self):
        msg = self.g.transfer_pub.messages.pop(0)
        self.n._transfer_cb(msg)
        return msg

    def ack(self):
        msg = self.n.admission_status_pub.messages.pop(0)
        self.g._admission_status_cb(msg)
        return msg

    def pump(self):
        for _ in range(50):
            if self.g.transfer_pub.messages:
                self.transfer()
            elif self.n.admission_status_pub.messages:
                self.ack()
            else:
                return
        self.fail('Protocol failed to quiesce')

    def start(self):
        self.g._candidate_cb(copy.deepcopy(self.raw))
        self.pump()

    def gcdf(self, safe=True):
        msg = CollisionCDFConstraintBatch()
        msg.header.stamp = self.n.final_gcdf_query_pub.messages[-1].header.stamp
        if not safe:
            msg.num_pairs = 1; msg.distance = [-.1]
            msg.original_timestep = [0]; msg.point_flat = [0.,0.,0.]
        self.n._final_gcdf_batch_cb(msg)

    def vbc(self, safe=True, stamp=None):
        stamp = stamp or self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        self.n._global_summary_cb(String(data=(
            'trajectory_source=predicted trajectory_stamp_ns={} has_violation={} vbc_evidence=fixed'.format(
                stamp, int(not safe)))))

    def certify(self):
        self.advance(.001); self.gcdf()
        self.advance(.001); self.vbc()
        self.pump()

    def test_recorded_candidate_waits_for_both_gates(self):
        self.start()
        self.assertEqual(len(self.raw.points), 21)
        self.assertEqual(self.n._raw_input_count, 1)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.gcdf(); self.assertFalse(self.n.committed_pub.messages)
        self.vbc(); self.pump()
        self.assertEqual(len(self.n.committed_pub.messages), 1)

    def test_first_deliver_lost_recovers_with_one_resend(self):
        self.g._candidate_cb(self.raw)
        offer = self.transfer(); self.assertEqual(offer.action, 'offer')
        self.ack()
        lost = self.g.transfer_pub.messages.pop(0)
        self.assertEqual(lost.action, 'deliver')
        self.advance(.251); self.g._admission_tick(); self.pump()
        self.assertEqual(self.g.sender.resend_count, 1)
        self.assertEqual(self.n._raw_input_count, 1)
        self.certify(); self.assertEqual(len(self.n.committed_pub.messages), 1)

    def test_first_offer_lost_recovers(self):
        self.g._candidate_cb(self.raw); self.g.transfer_pub.messages.clear()
        self.advance(.251); self.g._admission_tick(); self.pump(); self.certify()
        self.assertEqual(len(self.n.committed_pub.messages), 1)

    def test_offer_ack_lost_duplicate_offer_keeps_first_deadline(self):
        self.g._candidate_cb(self.raw); self.transfer()
        entry = next(iter(self.n._admission_entries.values())); deadline = entry['deadline']
        self.n.admission_status_pub.messages.clear()
        self.advance(.251); self.g._admission_tick(); self.pump()
        self.assertEqual(entry['deadline'], deadline)
        self.certify(); self.assertEqual(self.n._raw_input_count, 1)

    def test_receipt_ack_lost_duplicate_deliver_does_not_query_twice(self):
        self.g._candidate_cb(self.raw); self.transfer(); self.ack(); self.transfer()
        self.n.admission_status_pub.messages.clear()
        self.advance(.251); self.g._admission_tick(); self.pump()
        self.assertEqual(self.n._raw_input_count, 1)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages), 1)
        self.assertEqual(self.n._admission_duplicate_count, 1)
        self.certify(); self.assertEqual(len(self.n.committed_pub.messages), 1)

    def test_duplicate_after_commit_only_returns_receipt(self):
        self.g._candidate_cb(self.raw); self.transfer(); self.ack()
        deliver = self.transfer(); self.pump(); self.certify()
        self.n._transfer_cb(deliver); self.pump()
        self.assertEqual(self.n._raw_input_count, 1)
        self.assertEqual(len(self.n.committed_pub.messages), 1)

    def test_delayed_first_offer_after_sender_timeout_cannot_dispatch(self):
        self.g._candidate_cb(self.raw)
        late = self.g.transfer_pub.messages.pop(0)
        self.advance(.501); self.g._admission_tick()
        cancel = self.g.transfer_pub.messages.pop(0)
        self.n._transfer_cb(late); self.pump()
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.n._transfer_cb(cancel); self.pump()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(len(self.g.replan_request_pub.messages), 1)

    def test_cancel_before_offer_prevents_late_admission(self):
        self.g._candidate_cb(self.raw); late = self.g.transfer_pub.messages.pop(0)
        self.advance(.501); self.g._admission_tick(); self.transfer(); self.pump()
        self.n._transfer_cb(late)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.assertEqual(len(self.g.replan_request_pub.messages), 1)

    def test_paused_ros_lease_expires_and_late_safe_cannot_commit(self):
        self.start(); self.gcdf(); stamp = self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        self.advance(.751, ros=False)
        self.vbc(stamp=stamp); self.pump()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(self.n._admission_expiry_count, 1)

    def test_persistent_failure_is_bounded_without_replanning_unconfirmed_work(self):
        self.g._candidate_cb(self.raw)
        self.g.transfer_pub.messages.clear()
        for dt in (.251,.25,.251,.25,1.,1.):
            self.advance(dt, ros=False); self.g._admission_tick()
        actions = [m.action for m in self.g.transfer_pub.messages]
        self.assertEqual(actions, ['offer','cancel','cancel'])
        self.assertTrue(self.g.sender.fault)
        self.assertFalse(self.g.replan_request_pub.messages or self.n.committed_pub.messages)

    def test_each_distinct_transport_expiry_can_replan(self):
        for _ in range(2):
            self.raw.header.stamp = rospy.Time.from_sec(self.now)
            self.g._candidate_cb(self.raw); self.g.transfer_pub.messages.clear()
            self.advance(.501); self.g._admission_tick(); self.pump()
        self.assertEqual(len(self.g.replan_request_pub.messages), 2)

    def test_gcdf_unsafe_remains_rejected(self):
        self.start(); self.gcdf(False); self.pump()
        self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.assertEqual(self.n._final_gcdf_unsafe_count, 1)

    def test_vbc_unsafe_remains_rejected(self):
        self.start(); self.gcdf(); self.vbc(False); self.pump()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(self.n._verification_unsafe_count, 1)

    def test_start_continuity_gate_remains_rejected(self):
        self.start(); self.gcdf()
        self.n._joint_state_cb(JointState(name=self.raw.joint_names, position=[.2]*7))
        self.vbc(); self.pump()
        self.assertFalse(self.n.committed_pub.messages)
        self.assertIn('safety_gate=start_continuity', self.n.verification_event_pub.messages[-1].data)

    def test_raw_diagnostic_cannot_bypass_admission(self):
        self.n._raw_cb(self.raw)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_same_identity_changed_payload_is_rejected(self):
        self.g._candidate_cb(self.raw); offer=self.transfer(); self.ack()
        changed = self.g.transfer_pub.messages.pop(0)
        changed.trajectory.points[-1].positions[0] += .01
        self.n._transfer_cb(changed); self.pump()
        self.n._transfer_cb(offer)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_header_seq_is_not_identity(self):
        self.start(); raw = copy.deepcopy(self.raw); raw.header.seq=999
        self.assertEqual(payload_sha256(raw), payload_sha256(self.raw))

    def test_mode_change_invalidates_old_candidate_and_late_ack(self):
        self.start(); self.gcdf(); stamp=self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        self.g._probe_active_cb(Bool(data=True)); self.n._probe_active_cb(Bool(data=True))
        self.pump(); self.vbc(stamp=stamp); self.pump()
        self.assertFalse(self.n.committed_pub.messages)

    def test_new_goal_invalidates_current_candidate(self):
        self.start(); self.gcdf(); stamp=self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        goal=PoseStamped();goal.pose.position.x=.4
        self.g._goal_cb(goal); self.n._admission_goal_cb(goal); self.pump()
        self.vbc(stamp=stamp); self.assertFalse(self.n.committed_pub.messages)

    def test_existing_replacement_fence_cancels_transfer(self):
        self.start(); self.gcdf(); stamp=self.n.verification_pub.messages[-1].header.stamp.to_nsec()
        text='request_id=6 granted=1 mode_epoch=2 query_stamp_ns=8451000000 deadline_ns=10853000000'
        self.n._replacement_grant_cb(String(data=text)); self.pump()
        self.vbc(stamp=stamp); self.assertFalse(self.n.committed_pub.messages)

    def test_probe_full_safety_prefix_keeps_single_flight_ownership(self):
        self.g._probe_active_cb(Bool(data=True)); self.n._probe_active_cb(Bool(data=True))
        self.start(); self.assertEqual(self.g.phase, self.g.VERIFYING)
        self.certify(); self.assertEqual(self.g.phase, self.g.EXECUTING)
        self.assertEqual(len(self.n.committed_pub.messages), 1)
        self.g._candidate_cb(self.raw); self.assertIsNotNone(self.g.buffered_candidate)
        self.assertEqual(self.n._raw_input_count, 1)

    def test_commit_wins_cancel_race_receipt_reports_existing_commit(self):
        self.start(); self.certify()
        entry = next(iter(self.n._admission_entries.values()))
        msg = CandidateTransfer(action='cancel', session=entry['session'],sequence=entry['sequence'],
            receiver_boot_id=self.n._admission_boot_id,payload_sha256=entry['digest'],trajectory=self.raw)
        self.n._transfer_cb(msg)
        self.assertIn('committed=1',self.n.admission_status_pub.messages[-1].data)
        self.assertEqual(len(self.n.committed_pub.messages),1)

    def test_receiver_restart_never_replays_into_new_process(self):
        self.start(); self.g._admission_status_cb(String(data=(
            'action=pong ready_nonce=test receiver_boot_id='+'a'*32)))
        self.assertTrue(self.g.sender.fault)
        self.advance(.3); self.g._admission_tick(); self.assertFalse(self.g.transfer_pub.messages)

    def test_readiness_ping_is_roundtrip_without_candidate_or_commit(self):
        self.g._admission_ping_cb(String(data='startup_nonce')); self.pump()
        self.assertIn('ready_nonce=startup_nonce',self.g.admission_ready_pub.messages[-1].data)
        self.assertEqual(self.n._raw_input_count,0)
        self.assertFalse(self.n.committed_pub.messages)

    def test_cancel_confirmation_cannot_overtake_committed_publication(self):
        self.start(); self.gcdf(); self.pump()
        entry = next(iter(self.n._admission_entries.values()))
        cancel = CandidateTransfer(action='cancel',session=entry['session'],sequence=entry['sequence'],
            receiver_boot_id=self.n._admission_boot_id,payload_sha256=entry['digest'],trajectory=self.raw)
        entered=threading.Event();release=threading.Event();errors=[]
        original=self.n.committed_pub.publish
        def publish(msg):
            entered.set();release.wait(2);original(msg)
        self.n.committed_pub.publish=publish
        def run(fn):
            try:fn()
            except BaseException as exc:errors.append(str(exc))
        first=threading.Thread(target=lambda:run(self.vbc));first.start()
        self.assertTrue(entered.wait(1))
        second=threading.Thread(target=lambda:run(lambda:self.n._transfer_cb(cancel)));second.start()
        self.assertFalse(self.n.admission_status_pub.messages)
        release.set();first.join(2);second.join(2)
        self.assertFalse(errors)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(len(self.n.committed_pub.messages),1)
        self.assertTrue(all('committed=1' in m.data for m in self.n.admission_status_pub.messages))

    def test_before_after_same_lost_delivery(self):
        before=ROOT/'outputs/phase_e_case007_admission_fix_20261010/before/src/egocentric_arm_planner/scripts'
        if not before.exists():
            self.skipTest('Before snapshots belong to the CASE007 evidence archive')
        def load(name,path):
            spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module);return module
        old_receiver=load('before_receiver',before/'optimized_trajectory_continuity_node.py')
        old_gate=load('before_sender',before/'probe_single_flight_gate_node.py')
        n=old_receiver.OptimizedTrajectoryContinuityNode();g=old_gate.ProbeSingleFlightGate()
        n._joint_state_cb(JointState(name=self.raw.joint_names,position=self.data['measured_q']))
        g._candidate_cb(self.raw)
        self.assertEqual(len(g.output_pub.messages),1)
        g.output_pub.messages.clear()  # identical first delivery lost
        self.advance(.251);n._timer_cb(None)
        self.assertFalse(n.final_gcdf_query_pub.messages or n.committed_pub.messages)
        self.assertFalse(g.output_pub.messages or g.replan_request_pub.messages)
        self.g._candidate_cb(self.raw);self.transfer();self.ack()
        self.g.transfer_pub.messages.pop(0)  # same recorded payload, first deliver lost
        self.advance(.251);self.g._admission_tick();self.pump();self.certify()
        self.assertEqual(len(self.n.committed_pub.messages),1)
        type(self).counterfactual=dict(
            old=dict(raw_receipts=n._raw_input_count,gcdf_queries=len(n.final_gcdf_query_pub.messages),commits=len(n.committed_pub.messages)),
            new=dict(raw_receipts=self.n._raw_input_count,gcdf_queries=len(self.n.final_gcdf_query_pub.messages),commits=len(self.n.committed_pub.messages)),
            qualification='Same recorded payload, measured q and synthetic certification responses; first delivery dropped. Mechanism evidence, not proof of original TCP trigger.')

    def test_paused_ros_new_candidate_cannot_accept_old_gcdf_reply(self):
        self.start();old=copy.deepcopy(self.n.final_gcdf_query_pub.messages[-1])
        with self.g._lock:actions=self.g.sender.cancel(self.wall)
        self.g._emit_admission_actions(actions);self.pump()
        self.g._candidate_cb(copy.deepcopy(self.raw));self.pump()
        new=self.n.final_gcdf_query_pub.messages[-1]
        self.assertNotEqual(old.header.stamp.to_nsec(),new.header.stamp.to_nsec())
        late=CollisionCDFConstraintBatch();late.header.stamp=old.header.stamp
        self.n._final_gcdf_batch_cb(late)
        self.assertFalse(self.n.verification_pub.messages)
        self.gcdf();self.vbc();self.pump()
        self.assertEqual(len(self.n.committed_pub.messages),1)

    def test_paused_ros_execution_tokens_remain_distinct(self):
        self.start();self.gcdf();self.vbc();self.pump()
        self.g._candidate_cb(copy.deepcopy(self.raw));self.pump()
        self.gcdf();self.vbc();self.pump()
        stamps=[m.header.stamp.to_nsec() for m in self.n.committed_pub.messages]
        self.assertEqual(len(stamps),2)
        self.assertGreater(stamps[1],stamps[0])

    def test_missing_receiver_readiness_is_bounded(self):
        self.g.sender.receiver_boot_id=''
        self.g._candidate_cb(self.raw);self.advance(.501);self.g._admission_tick()
        self.assertTrue(self.g.sender.fault)
        self.assertFalse(self.n.final_gcdf_query_pub.messages or self.g.replan_request_pub.messages)

    def test_ledger_remains_bounded_and_retired_messages_cannot_rearm(self):
        self.g._candidate_cb(self.raw);old=self.transfer()
        for sequence in range(2,70):
            msg=copy.deepcopy(old);msg.sequence=sequence;self.n._transfer_cb(msg)
        self.assertEqual(len(self.n._admission_entries),64)
        late=copy.deepcopy(old);late.action='deliver';self.n._transfer_cb(late)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-json',type=Path)
    args=ap.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AdmissionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        args.output_json.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            tests=result.testsRun, failures=[str(t) for t,_ in result.failures],
            errors=[str(t) for t,_ in result.errors],
            counterfactual=getattr(AdmissionTests,'counterfactual',None),
            fixture='scripts/fixtures/phase_e_case007_candidate.json',
            qualification='Production callbacks with recorded CASE007 payload and injected transport faults. Synthetic verdicts test gate ownership; not physical collision or original TCP-cause reproduction.'),indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())
