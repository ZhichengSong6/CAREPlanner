#!/usr/bin/env python3
"""Reserved tokens, immutable preparation identity and first-deadline expiry."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.append('/usr/lib/python3/dist-packages')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/egocentric_arm_planner/scripts'),str(ROOT/'src/care_visibility_cdf/scripts')]
from execution_preparation import ExecutionPreparationLedger
from candidate_admission import payload_sha256
from task_observation_binding import ObservationTaskOwner
from test_phase_e_task_context import command
from test_phase_e_candidate_admission import load_fixture


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.raw=load_fixture()[1]
        self.now=self.raw.header.stamp.to_nsec()
        self.context=ObservationTaskOwner('owner').command(command(),self.now)[-1]['candidate_binding']
        self.regime='regime:1:NORMAL'
        self.digest=payload_sha256(self.raw,'',self.context,self.regime)
        self.ledger=ExecutionPreparationLedger()

    def prepare(self,**changes):
        args=dict(trajectory=self.raw,task_context=self.context,task_regime=self.regime,
            source_key=('source',1),source_digest=self.digest,verification_seq=1,
            now_ns=self.now,expires_ns=self.now+100000000)
        args.update(changes)
        return self.ledger.prepare(**args)

    def test_preparation_binds_motion_task_mode_and_qualification_requirement(self):
        message=self.prepare()
        self.assertEqual(message.action,'execution_prepare')
        self.assertEqual(message.payload_sha256,payload_sha256(message.trajectory,
            message.planned_prefix_context,message.task_context,message.task_regime))
        self.assertEqual(json.loads(message.planned_prefix_context)['execution_qualification'],'UNMET')
        self.assertFalse(self.ledger.status()['execution_authorized'])

    def test_repeated_proposal_cannot_postpone_first_deadline(self):
        first=self.prepare()
        later=self.prepare(now_ns=self.now+10000000,expires_ns=self.now+200000000)
        self.assertEqual(first.sequence,later.sequence)
        self.assertEqual(first.trajectory.header.stamp,later.trajectory.header.stamp)
        self.assertEqual(json.loads(later.planned_prefix_context)['expires_ns'],self.now+100000000)
        self.assertLess(later.remaining_budget_s,first.remaining_budget_s)

    def test_mutated_payload_cannot_reuse_verified_source(self):
        self.prepare();raw=copy.deepcopy(self.raw);raw.points[0].positions[0]+=.001
        with self.assertRaisesRegex(ValueError,'mutated'):self.prepare(trajectory=raw)

    def test_expired_proposal_does_not_revive_with_a_new_deadline(self):
        self.prepare()
        self.assertIsNone(self.ledger.snapshot(self.now+100000000,self.context,self.regime))
        with self.assertRaisesRegex(ValueError,'retired'):
            self.prepare(now_ns=self.now+100000001,expires_ns=self.now+200000000)

    def test_cancel_does_not_erase_source_replay_protection(self):
        self.prepare();self.ledger.cancel('goal_cancelled')
        with self.assertRaisesRegex(ValueError,'retired'):self.prepare()
        self.assertFalse(self.ledger.status()['execution_authorized'])

    def test_mode_replacement_cancels_pending_without_relabelling(self):
        self.prepare()
        self.assertIsNone(self.ledger.snapshot(self.now,self.context,'regime:2:REPAIR'))
        with self.assertRaisesRegex(ValueError,'retired'):self.prepare(task_regime='regime:2:REPAIR')

    def test_new_verified_source_gets_a_new_preparation_identity(self):
        first=self.prepare()
        second=self.prepare(source_key=('source',2),verification_seq=2)
        self.assertGreater(second.sequence,first.sequence)
        self.assertEqual(second.session,first.session)
        self.assertNotEqual(first.payload_sha256,second.payload_sha256)

    def test_caller_mutation_cannot_change_pending_trajectory(self):
        message=self.prepare();message.trajectory.points[0].positions[0]+=.2
        pending=self.ledger.snapshot(self.now,self.context,self.regime)
        self.assertNotEqual(message.trajectory.points[0].positions[0],pending.trajectory.points[0].positions[0])

    def test_clock_reversal_closes_preparation_session(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'clock_reversed'):
            self.ledger.snapshot(self.now-1,self.context,self.regime)
        with self.assertRaisesRegex(ValueError,'clock_reversed'):
            self.prepare(source_key=('source',2),now_ns=self.now+1)
        self.assertIsNone(self.ledger.pending)

    def test_source_capacity_does_not_drop_old_replay_history(self):
        for i in range(16):self.prepare(source_key=('source'+str(i),1))
        with self.assertRaisesRegex(ValueError,'capacity'):self.prepare(source_key=('overflow',1))

    def test_oversized_future_window_and_missing_mode_are_rejected(self):
        for changes in [dict(expires_ns=self.now+250000001),dict(task_regime=''),dict(source_digest='')]:
            with self.assertRaises(ValueError):self.prepare(**changes)


if __name__=='__main__':unittest.main()
