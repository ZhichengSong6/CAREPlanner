#!/usr/bin/env python3
"""Protocol fixtures, never physical qualification or production admission."""
import copy
from dataclasses import FrozenInstanceError
import unittest
from unittest import mock
import numpy as np

from phase_e_leased_stop_evidence import query_adaptive_stopping_snapshot
from phase_e_serial_lease_ledger import SerialLeaseLedger, LedgerReject


class Fixture:
    def __init__(self):
        self.now=10000000000;self.wall=0.;self.calls=[];self.levels=[]
        self.unknown_coarse=False;self.outside_coarse=False;self.always_unknown=False
        self.changed=False;self.hook=None
        self.geometry=self

    def raster(self,q,travel,meta,margin,subdivisions,check_budget):
        check_budget();self.levels.append(subdivisions)
        ids=[0,1,2] if subdivisions==1 else [0,1]
        result=dict(grid_indices=ids,points=np.array([[i*.05,0.,0.] for i in ids]),
                    inside=not (self.outside_coarse and subdivisions==1))
        if hasattr(self,'reference_ids'):result['reference_pose_grid_indices']=self.reference_ids
        return result

    def query(self,points,horizon,expected):
        self.calls.append(copy.deepcopy(points))
        if self.hook:self.hook(points,expected)
        n=len(points)
        result=dict(valid=True,instance='fixture_map',epoch=1,revision=2+(1 if expected and self.changed else 0),
            frame='base_link',resolution=.05,map_min=[0.,0.,0.],map_max=[1.,1.,1.],grid_size=[21]*3,
            snapshot_ns=self.now,clock_fault=False,horizon_s=horizon,inside_map=[1]*n,
            confidence_at_horizon=[.99]*n,occupancy=[0.]*n,provenance=[1]*n,model_free_at_horizon=[1]*n)
        if n and (self.always_unknown or (self.unknown_coarse and n==3)):
            result['confidence_at_horizon'][-1]=0.;result['model_free_at_horizon'][-1]=0
        return result

    def adaptive(self,**kw):
        budgets={k:dict(command_travel_rad=[t]*7,required_until_s=11.)
                 for k,t in [('withdraw_now',.1),('continue_then_stop',.2)]}
        return query_adaptive_stopping_snapshot(budgets=budgets,q=[0.]*7,geometry=self,
            query=self.query,now_ns=lambda:self.now,measured_ns=10000000000,
            authorization_until_ns=10100000000,enclosure_field='command_travel_rad',
            wall_clock=lambda:self.wall,**kw)

    def state(self,mode=0,grant=None):
        row=dict(controller_session=71,t_ns=self.now,mode=mode,clock_fault=False,
            integral_valid=True,lease_required=True,q=[0.]*7,applied=[0.]*7,measured=[0.]*7,
            max_update_gap=.001,max_braking_step_s=.02,max_execution_lease_s=.1,
            velocity_limits=[2.]*7,acceleration_limits=[3.,3.,4.,4.,6.,6.,6.],execution_stamp_ns=0)
        if grant:row.update(grant)
        return row


class AdaptiveTests(unittest.TestCase):
    def setUp(self):self.f=Fixture()

    def test_coarse_free_enclosure_finishes_without_unnecessary_refinement(self):
        r=self.f.adaptive();self.assertTrue(r['diagnostic_accepted'])
        self.assertEqual(r['adaptive']['selected_subdivisions'],1)
        self.assertEqual([len(p) for p in self.f.calls],[0,3,0])
        self.assertFalse(r['runtime_admission'])

    def test_unknown_coarse_set_is_queried_in_full_before_refining(self):
        self.f.unknown_coarse=True;self.f.reference_ids=[0];r=self.f.adaptive()
        self.assertTrue(r['diagnostic_accepted']);self.assertEqual(r['adaptive']['selected_subdivisions'],2)
        self.assertEqual([len(p) for p in self.f.calls],[0,3,0,2,0])
        self.assertEqual(len(r['adaptive']['attempts']),2)

    def test_outside_coarse_enclosure_refines_before_spending_map_query(self):
        self.f.outside_coarse=True;r=self.f.adaptive()
        self.assertTrue(r['diagnostic_accepted'])
        self.assertEqual([len(p) for p in self.f.calls],[0,2,0])

    def test_unknown_at_all_levels_is_explicit_rejection(self):
        self.f.always_unknown=True;r=self.f.adaptive()
        self.assertFalse(r['diagnostic_accepted']);self.assertFalse(r['complete'])
        self.assertEqual(len(r['adaptive']['attempts']),4)
        self.assertEqual(r['adaptive']['reason'],'unknown_or_decayed_stopping_enclosure')

    def test_age_and_wall_budgets_are_independent(self):
        for kind in ['ros','wall']:
            f=Fixture()
            def delay(points,expected):
                if points:
                    if kind=='ros':f.now+=51000000
                    else:f.wall+=.051
            f.hook=delay;r=f.adaptive()
            self.assertFalse(r['diagnostic_accepted'])
            self.assertIn('state_age' if kind=='ros' else 'wall',r['adaptive']['reason'])

    def test_late_evidence_serialization_cannot_return_success(self):
        r=self.f.adaptive(evidence_sink=lambda value:setattr(self.f,'wall',.051))
        self.assertFalse(r['diagnostic_accepted'])
        self.assertEqual(r['adaptive']['reason'],'query_wall_budget_exhausted')

    def test_version_change_is_rejected_without_retrying_new_content(self):
        self.f.changed=True;r=self.f.adaptive()
        self.assertFalse(r['diagnostic_accepted']);self.assertEqual(len(r['adaptive']['attempts']),1)

    def test_ros_time_regression_and_expired_authorization_reject(self):
        self.f.hook=lambda points,expected:setattr(self.f,'now',self.f.now-1)
        r=self.f.adaptive();self.assertEqual(r['adaptive']['reason'],'query_ros_time_regressed')
        self.f=Fixture();self.f.now+=100000000
        r=self.f.adaptive();self.assertFalse(r['diagnostic_accepted'])

    def test_exhausted_later_level_cannot_reuse_previous_complete_result(self):
        self.f.unknown_coarse=True
        def delay(points,expected):
            if len(points)==2:self.f.wall=.051
        self.f.hook=delay;r=self.f.adaptive()
        self.assertFalse(r['complete']);self.assertFalse(r['diagnostic_accepted'])

    def test_unknown_in_required_reference_margin_rejects_without_futile_refinement(self):
        self.f.unknown_coarse=True;self.f.reference_ids=[2]
        r=self.f.adaptive();self.assertFalse(r['diagnostic_accepted'])
        self.assertEqual(r['adaptive']['reason'],'uncovered_reference_pose_margin')
        self.assertEqual(len(r['adaptive']['attempts']),1)
        self.assertEqual(r['reference_pose_uncovered_grid_indices'],[2])

    def test_metadata_reuse_never_reuses_coarse_free_evidence(self):
        self.f.unknown_coarse=True;self.f.reference_ids=[0];original=self.f.query
        def changed(points,horizon,expected):
            result=original(points,horizon,expected)
            if len(points)==2:
                result['confidence_at_horizon'][0]=0.;result['model_free_at_horizon'][0]=0
            return result
        self.f.query=changed;r=self.f.adaptive()
        self.assertFalse(r['diagnostic_accepted']);self.assertEqual(r['adaptive']['reason'],'uncovered_reference_pose_margin')
        self.assertEqual([len(p) for p in self.f.calls],[0,3,0,2,0])

    def test_metadata_reuse_rejects_map_restart_or_geometry_change(self):
        for field,value in [('instance','restarted'),('epoch',9),('resolution',.025)]:
            f=Fixture();f.unknown_coarse=True;original=f.query
            def changed(points,horizon,expected):
                result=original(points,horizon,expected)
                if len(points)==2:result[field]=value
                return result
            f.query=changed;r=f.adaptive()
            self.assertFalse(r['diagnostic_accepted']);self.assertEqual(r['adaptive']['reason'],'map_changed_or_invalid')


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.f=Fixture();self.ledger=SerialLeaseLedger('f'*64)
        self.ledger.observe(self.f.state())

    def test_diagnostic_history_can_be_drained_without_losing_any_live_right(self):
        self.tracking([.2]*7)
        rights=copy.deepcopy(self.ledger.rights);sequence=self.ledger.sequence
        drained=self.ledger.drain_events()
        self.assertTrue(drained);self.assertEqual(self.ledger.events,[])
        self.assertEqual(self.ledger.rights,rights);self.assertEqual(self.ledger.sequence,sequence)
        grant=self.commit(self.begin([.2]*7))
        self.assertGreater(grant['lease_sequence'],sequence)

    def begin(self,cap=None,token=9,duration=100000000):
        return self.ledger.begin(now_ns=self.f.now,token=token,
            target_abs_velocity=cap or [0.]*7,duration_ns=duration)

    def check(self,p):
        return self.ledger.check(p,geometry=self.f,query=self.f.query,now_ns=lambda:self.f.now)

    def commit(self,p):
        proof=self.check(p)
        return self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)['grant']

    def tracking(self,cap=None):
        grant=self.commit(self.begin());self.f.now+=1000000
        self.ledger.observe(self.f.state(1,grant))
        if cap:
            grant=self.commit(self.begin(cap));self.f.now+=1000000
            self.ledger.observe(self.f.state(1,grant))
        return grant

    def test_initial_authorization_needs_rest_zero_and_a_complete_query(self):
        with self.assertRaisesRegex(LedgerReject,'rest_and_zero'):self.begin([.3]*7)
        p=self.begin();grant=self.commit(p)
        self.assertEqual(grant['target_abs_velocity'],[0.]*7)
        self.assertFalse(grant['requires_explicit_rearm'])
        self.assertEqual(grant['valid_until_ns'],p.record()['candidate']['valid_until_ns'])

    def test_proposal_and_returned_records_cannot_mutate_checked_inputs(self):
        p=self.begin();r=p.record();r['candidate']['valid_until_ns']+=1
        self.assertNotEqual(r,p.record())
        with self.assertRaises(FrozenInstanceError):p.payload=b'forged'
        self.assertEqual(self.commit(p)['valid_until_ns'],10100000000)

    def test_only_one_proposal_and_failed_query_does_not_extend_old_deadline(self):
        old=self.tracking();p=self.begin([.3]*7)
        with self.assertRaisesRegex(LedgerReject,'in_flight'):self.begin()
        self.f.always_unknown=True;proof=self.check(p)
        with self.assertRaisesRegex(LedgerReject,'unknown'):self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)
        self.assertEqual(self.ledger.rights[old['lease_sequence']]['valid_until_ns'],old['valid_until_ns'])
        self.assertIsNone(self.ledger.proposal)

    def test_unacknowledged_earlier_wider_grant_remains_in_union(self):
        self.tracking();wide=self.commit(self.begin([1.]*7,duration=40000000))
        self.f.now+=1000000
        # No applied receipt for wide: it may nevertheless already be executing.
        p=self.begin([.1]*7)
        self.assertIn(wide,p.record()['outstanding'])
        self.assertEqual(p.record()['enclosed_target_abs_velocity'],[1.]*7)
        self.assertGreater(p.record()['enclosed_until_ns'],wide['valid_until_ns'])

    def test_same_grant_source_updates_do_not_invalidate_proof(self):
        old=self.tracking([.3]*7);p=self.begin([.3]*7);proof=self.check(p)
        self.f.now+=1000000;s=self.f.state(1,old);s.update(source_sequence=99,generated_ns=self.f.now)
        self.ledger.observe(s)
        result=self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)
        self.assertFalse(result['runtime_admission'])

    def test_many_legal_renewals_make_continuous_progress(self):
        old=self.tracking([.3]*7);distance=0.
        for _ in range(100):
            self.f.now+=20000000;s=self.f.state(1,old)
            distance+=.3*.02;s['q']=[distance]*7;s['applied']=s['measured']=[.3]*7
            self.ledger.observe(s);old=self.commit(self.begin([.3]*7))
        self.assertGreater(distance,.59)
        self.assertEqual(len([e for e in self.ledger.events if e['kind']=='diagnostic_commit']),102)

    def test_expiry_during_check_cannot_be_repaired_by_sliding_deadline(self):
        old=self.tracking();self.f.now=old['valid_until_ns']-1000000
        self.ledger.observe(self.f.state(1,old));p=self.begin([.3]*7)
        self.f.hook=lambda points,expected:setattr(self.f,'now',self.f.now+2000000) if points else None
        proof=self.check(p)
        with self.assertRaisesRegex(LedgerReject,'expired'):self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)

    def test_final_map_version_recheck_and_commit_age_are_mandatory(self):
        for kind in ['version','age']:
            self.setUp();p=self.begin();proof=self.check(p)
            if kind=='version':self.f.changed=True
            else:self.f.now+=51000000
            with self.assertRaises(LedgerReject):self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)
            self.assertEqual(self.ledger.rights,{})

    def test_revoke_restart_and_braking_invalidate_inflight_proposals(self):
        for kind in ['revoke','restart','braking']:
            self.setUp();old=self.tracking();p=self.begin();proof=self.check(p)
            if kind=='revoke':self.ledger.revoke()
            else:
                s=self.f.state(2 if kind=='braking' else 0)
                if kind=='restart':s['controller_session']=72
                else:s['execution_stamp_ns']=old['execution_stamp_ns']
                self.ledger.observe(s)
            with self.assertRaises(LedgerReject):self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)

    def test_latched_recovery_requires_new_token_zero_and_explicit_rearm(self):
        old=self.tracking();self.f.now+=100000000
        s=self.f.state(3,old);self.ledger.observe(s)
        with self.assertRaisesRegex(LedgerReject,'retired'):self.begin(token=9)
        with self.assertRaisesRegex(LedgerReject,'rest_and_zero'):self.begin([.1]*7,token=10)
        p=self.begin(token=10)
        self.f.now+=1000000;self.ledger.observe(self.f.state(3,old))
        g=self.commit(p);self.assertTrue(g['requires_explicit_rearm'])

    def test_receipt_for_unknown_grant_and_clock_regression_fault_session(self):
        old=self.tracking();s=self.f.state(1,old);s['lease_sequence']+=100
        with self.assertRaisesRegex(LedgerReject,'unrecorded'):self.ledger.observe(s)
        with self.assertRaises(LedgerReject):self.begin()
        self.setUp();self.begin();self.f.now-=1
        with self.assertRaisesRegex(LedgerReject,'clock_regressed'):self.begin()

    def test_changed_limits_during_query_cannot_be_committed(self):
        old=self.tracking();p=self.begin();proof=self.check(p)
        s=self.f.state(1,old);s['acceleration_limits']=[1.]*7;self.ledger.observe(s)
        with self.assertRaisesRegex(LedgerReject,'contract_changed'):
            self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)

    def test_nested_check_cannot_create_parallel_proof_or_new_issuance(self):
        p=self.begin();observed=[]
        def nested(points,expected):
            if points:
                with self.assertRaisesRegex(LedgerReject,'already_started'):self.check(p)
                with self.assertRaisesRegex(LedgerReject,'still_in_flight'):self.begin()
                observed.append(True)
        self.f.hook=nested;proof=self.check(p)
        self.assertEqual(observed,[True])
        with self.assertRaisesRegex(LedgerReject,'already_started'):self.check(p)
        self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)

    def test_initial_rest_must_still_hold_at_commit(self):
        p=self.begin();proof=self.check(p);s=self.f.state();s['measured']=[.002]*7
        self.ledger.observe(s)
        with self.assertRaisesRegex(LedgerReject,'no_longer_at_rest'):
            self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)

    def test_commit_and_queue_time_share_wall_budget(self):
        p=self.begin();proof=self.check(p)
        with mock.patch('phase_e_serial_lease_ledger.time.monotonic',return_value=proof.record()['protocol_wall_deadline_s']+1.):
            with self.assertRaisesRegex(LedgerReject,'wall_budget_exhausted_at_commit'):
                self.ledger.commit(proof,query=self.f.query,now_ns=lambda:self.f.now)
        p=self.begin()
        with mock.patch('phase_e_serial_lease_ledger.time.monotonic',return_value=p.record()['created_wall_s']+1.):
            with self.assertRaisesRegex(LedgerReject,'wall_budget_exhausted_before_check'):self.check(p)


if __name__=='__main__':unittest.main()
