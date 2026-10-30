#!/usr/bin/env python3
"""Actual admission callbacks: immutable prefix intent, not geometry truth."""
import copy
import json
import unittest
from unittest.mock import patch
import test_phase_e_candidate_admission as base
from candidate_admission import CandidateSender, payload_sha256
from repair_prefix_contract import make, parse
from egocentric_arm_planner.msg import CandidateTransfer
from std_msgs.msg import String

class PrefixTests(unittest.TestCase):
    advance=base.AdmissionTests.advance
    transfer=base.AdmissionTests.transfer
    ack=base.AdmissionTests.ack
    pump=base.AdmissionTests.pump
    gcdf=base.AdmissionTests.gcdf
    def setUp(self):
        base.AdmissionTests.setUp(self)
        self.n.prefix_binding_enabled=True
        self.n.initial_shell_recovery_enabled=True
        self.n.repair_projection_consistent_dispatch_enabled=True
        self.n.prefix_selection_pub=base.Pub()
        self.n.shell_context_pub=base.Pub()
        self.g.prefix_binding_enabled=True
        self.n._repair_active=True
        self.n.cycle_recovery_enabled=True
        self.n._sync_repair_prefix_locked()
        self.contract=copy.deepcopy(self.n._prefix_context)
        self.raw_context=json.dumps(self.contract,separators=(',',':'))
    def start(self,context=None):
        self.g._planned_candidate_cb(CandidateTransfer(action='planned',trajectory=self.raw,
            planned_prefix_context=self.raw_context if context is None else context))
        self.pump()
    def vbc(self,contract=None,safe=True):
        d=self.n._outstanding_diag;c=contract or d['planned_prefix'];key=d['transfer_key']
        stamp=self.n._outstanding.header.stamp.to_nsec()
        self.n._global_summary_cb(String(data='trajectory_source=predicted trajectory_stamp_ns={} '
            'has_violation={} shell_recovery_count=0 context_request_session={} context_request_sequence={} '
            'context_prefix_session={} context_prefix_revision={} context_prefix_s={}'.format(
            stamp,int(not safe),*key,c['session'],c['revision'],c['prefix_s'])))
        self.pump()
    def change_prefix(self):
        self.n._cycle_recovery_active=True
        self.n._cycle_recovery_prefix_index=1
        self.n._sync_repair_prefix_locked()
    def test_exact_contract_constructs_and_requires_both_gates(self):
        self.start()
        self.assertEqual(self.n._gcdf_outstanding_diag['planned_prefix'],self.contract)
        self.assertFalse(self.n.committed_pub.messages)
        self.gcdf();self.assertFalse(self.n.committed_pub.messages)
        ctx=json.loads(self.n.shell_context_pub.messages[-1].data)
        self.assertEqual(ctx['planned_prefix'],self.contract)
        self.vbc();self.assertEqual(len(self.n.committed_pub.messages),1)
    def test_missing_context_and_raw_diagnostic_cannot_dispatch(self):
        self.g._candidate_cb(self.raw)
        self.assertEqual(self.g.input_count,0)
        self.start('');self.assertFalse(self.n.final_gcdf_query_pub.messages)
    def test_old_revision_boot_and_parameters_cannot_dispatch(self):
        for key,value in [('revision',2),('session','a'*32),('prefix_s',.05),('brake_dt_s',.1),('acceleration',[1.]*7)]:
            c=dict(self.contract);c[key]=value
            entry={'view':'repair_prefix_brake_hold'}
            self.assertFalse(self.n._offer_prefix_valid_locked(entry,json.dumps(c)))
    def test_context_hash_mutation_cancels_offer(self):
        self.g._planned_candidate_cb(CandidateTransfer(action='planned',trajectory=self.raw,planned_prefix_context=self.raw_context))
        offer=self.g.transfer_pub.messages.pop(0);offer.planned_prefix_context=offer.planned_prefix_context.replace('.25','.05')
        self.n._transfer_cb(offer)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)
        self.assertEqual(next(iter(self.n._admission_entries.values()))['stage'],'rejected')
    def test_version_change_before_gcdf_cancels_without_new_query(self):
        self.start();stamp=self.n._gcdf_outstanding.header.stamp
        self.change_prefix();self.assertIsNone(self.n._gcdf_outstanding)
        self.assertEqual(self.n._prefix_revision,2)
        self.gcdf();self.assertFalse(self.n.verification_pub.messages or self.n.committed_pub.messages)
        self.assertEqual(len(self.n.final_gcdf_query_pub.messages),1)
    def test_old_safe_certificate_after_change_cannot_commit(self):
        self.start();self.gcdf();old=self.n._outstanding.header.stamp.to_nsec()
        self.change_prefix()
        self.n._global_summary_cb(String(data='trajectory_source=predicted trajectory_stamp_ns={} has_violation=0'.format(old)))
        self.assertFalse(self.n.committed_pub.messages)
    def test_echo_mismatch_closes_without_geometric_witness(self):
        self.start();self.gcdf();c=dict(self.contract);c['revision']=99
        self.vbc(c);self.assertFalse(self.n.committed_pub.messages)
        self.assertEqual(self.n._verification_unsafe_count,0)
        self.assertIsNone(self.n._outstanding)
    def test_repeated_selection_does_not_change_budget_or_version(self):
        before=(self.n._prefix_revision,self.n._cycle_recovery_backoff_count,self.n._cycle_recovery_prefix_index)
        self.n._sync_repair_prefix_locked();self.n._sync_repair_prefix_locked()
        self.assertEqual(before,(self.n._prefix_revision,self.n._cycle_recovery_backoff_count,self.n._cycle_recovery_prefix_index))
    def test_normal_and_probe_keep_empty_context_protocol(self):
        self.assertTrue(self.n._offer_prefix_valid_locked({'view':'full_horizon'},''))
        self.assertTrue(self.n._offer_prefix_valid_locked({'view':'probe_prefix_brake_hold'},''))
        self.assertFalse(self.n._offer_prefix_valid_locked({'view':'full_horizon'},self.raw_context))
    def test_waiting_and_retransmitted_payload_keep_contract(self):
        s=CandidateSender();s.forward(self.raw,100.,self.raw_context)
        offer=s.pong(self.n._admission_boot_id,100.)[0][1]
        self.assertEqual(offer.planned_prefix_context,self.raw_context)
        self.assertEqual(offer.payload_sha256,payload_sha256(self.raw,self.raw_context))
        retransmit=s.tick(100.26)[0][1]
        self.assertEqual(retransmit.planned_prefix_context,self.raw_context)
    def test_duplicate_and_invalid_numeric_context_rejected(self):
        for raw in [self.raw_context[:-1]+',"revision":1}',self.raw_context.replace('0.25','NaN'),self.raw_context.replace('"schema":1','"schema":true')]:
            with self.assertRaises(ValueError):parse(raw)

if __name__=='__main__':unittest.main()
