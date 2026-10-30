#!/usr/bin/env python3
"""Ownership, replay, cancellation and actual producer/scheduler race regressions."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import threading
import unittest
from types import SimpleNamespace as NS
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_visibility_cdf/scripts'))
import test_phase_e_observation_prefix as old
import test_phase_e_trajectory_prefix as curves
from stopping_observation_context import (ObservationContext, ContextObservationDemandQueue,
    context_key, context_path, validate_context_report)
from stopping_observation_demands import validate_report, exact_region_owner
from stopping_trajectory_prefix import validate_trajectory_report, trajectory_sha256, apply_absolute_observation_deadline


def event(now=10_000_000_000, generation=1, seq=1, active=True):
    return dict(schema=1, owner_session='owner', context_id='goal-recovery', generation=generation,
        event_seq=seq, active=active, stamp_ns=now, valid_until_ns=now+1_000_000_000,
        region_min=[-1.]*3, region_max=[1.]*3)


def report(f=None, ctx=None):
    f = f or old.Fixture()
    r = f.scan(trajectory_stamp_ns=10_000_000_000)
    r.update(observation_context=ctx or ObservationContext().update(event(), f.now),
        trajectory_sha256='a'*64, evidence_kind='context_observation_demand', execution_eligible=False)
    return r


class ContextTests(unittest.TestCase):
    def test_heartbeat_preserves_start_and_cancelled_epoch_cannot_resume(self):
        c=ObservationContext();a=c.update(event(),10_000_000_000)
        b=c.update(event(10_100_000_000,seq=2),10_100_000_000)
        self.assertEqual(a['started_ns'],b['started_ns'])
        self.assertIsNone(c.update(event(10_200_000_000,seq=3,active=False),10_200_000_000))
        with self.assertRaisesRegex(ValueError,'closed'):c.update(event(10_300_000_000,seq=4),10_300_000_000)
        self.assertIsNotNone(c.update(event(10_300_000_000,generation=2,seq=4),10_300_000_000))

    def test_expiry_or_clock_reverse_needs_new_generation(self):
        for now in (11_100_000_000,9_999_999_999):
            c=ObservationContext();c.update(event(),10_000_000_000)
            self.assertIsNone(c.snapshot(now))
            with self.assertRaises(ValueError):c.update(event(now,seq=2),now)

    def test_replay_owner_and_mutated_relevance_cannot_reactivate_work(self):
        c=ObservationContext();c.update(event(),10_000_000_000)
        with self.assertRaisesRegex(ValueError,'regressed'):c.update(event(),10_000_000_000)
        e=event(seq=2);e['region_min'][0]=-.5
        with self.assertRaisesRegex(ValueError,'mutated'):c.update(e,10_000_000_000)
        e=event(10_100_000_000);e['owner_session']='new-owner';c.update(e,10_100_000_000)
        with self.assertRaisesRegex(ValueError,'old_owner'):c.update(event(10_200_000_000,seq=3),10_200_000_000)

    def test_invalid_or_unbounded_heartbeat_is_rejected(self):
        for change in ({'active':1},{'valid_until_ns':14_000_000_000},{'region_max':[float('inf')]*3}):
            c=ObservationContext();e=event();e.update(change)
            with self.assertRaises(ValueError):c.update(e,10_000_000_000)


class LedgerTests(unittest.TestCase):
    def test_context_epoch_is_explicit_without_moving_absolute_deadline(self):
        r=apply_absolute_observation_deadline({},dict(stopping_observation_deadline_abs_s=15.,
            stopping_observation_epoch_s=10.,stopping_observation_epoch_source='observation_context_started'),
            'stopping_prefix_lookahead')
        self.assertEqual(r['deadline_absolute_ros_s'],15.)
        self.assertEqual(r['deadline_epoch_source'],'observation_context_started')
    def test_candidate_changes_merge_cells_and_never_slide_earliest_deadline(self):
        q=ContextObservationDemandQueue();r=report();ctx=r['observation_context'];q.update(r,ctx,10.)
        r['path_id']='different-candidate';r['trajectory_sha256']='b'*64
        for k in ('first_use_s','required_until_s','observation_deadline_s','latest_start_s'):r['cells'][0][k]+=5.
        cells=q.update(r,ctx,10.)
        self.assertEqual(len(cells),1);self.assertEqual(cells[0]['observation_deadline_s'],15.)
        self.assertEqual(cells[0]['required_until_s'],21.)
        self.assertEqual(cells[0]['source_count'],2)

    def test_only_new_sensor_with_full_accumulated_horizon_completes(self):
        f=old.Fixture();q=ContextObservationDemandQueue();r=report(f);ctx=r['observation_context'];q.update(r,ctx,10.)
        f.now+=100_000_000;f.provenance=2;f.confidence=1.
        self.assertEqual(len(q.update(report(f,ctx),ctx,10.1)),1)
        f.provenance=1;f.last=10.1
        self.assertEqual(q.update(report(f,ctx),ctx,10.1),[]);self.assertEqual(q.resolved,1)

    def test_capacity_is_transactional_and_outside_owner_region_is_ignored(self):
        q=ContextObservationDemandQueue(1);r=report();ctx=r['observation_context'];q.update(r,ctx,10.)
        other=copy.deepcopy(r);other['cells'][0]['grid_index']=2
        with self.assertRaisesRegex(ValueError,'capacity'):q.update(other,ctx,10.)
        self.assertEqual(len(q.pending),1)
        other['cells'][0]['point']=[2.,0.,0.]
        self.assertEqual(len(q.update(other,ctx,10.)),1)

    def test_map_restart_closes_ledger_and_old_instance_cannot_revive_it(self):
        q=ContextObservationDemandQueue();r=report();ctx=r['observation_context'];q.update(r,ctx,10.)
        new=copy.deepcopy(r);new['map_identity'][0]='restarted'
        with self.assertRaisesRegex(ValueError,'map_changed'):q.update(new,ctx,10.)
        with self.assertRaisesRegex(ValueError,'map_changed'):q.update(r,ctx,10.)
        self.assertFalse(q.pending)

    def test_old_candidate_report_is_never_candidate_execution_evidence(self):
        with self.assertRaisesRegex(ValueError,'not_candidate_evidence'):
            validate_trajectory_report(report(),curves.trajectory(),10.)

    def test_requery_includes_cells_omitted_from_new_prefix(self):
        f=old.Fixture();r=report(f);q=ContextObservationDemandQueue();ctx=r['observation_context'];q.update(r,ctx,10.)
        retained=q.ordered();f.provenance=1;f.last=10.;f.confidence=1.
        rr=f.scan(segments=[],retained_cells=retained,retained_map_scope=q.map_scope)
        self.assertEqual(rr['cells'][0]['grid_index'],1)
        self.assertEqual(rr['cells'][0]['action'],'covered')
        f.revision=8;self.assertEqual(f.scan(segments=[],retained_cells=retained,retained_map_scope=q.map_scope)['map_identity'][2],8)


class ProducerTests(unittest.TestCase):
    def fixture(self):
        legacy=curves.ObserverTests();o=legacy.fixture()
        o.context_enabled=True;o.offer_context(event())
        t=curves.trajectory();t.header.stamp=curves.duration(10_000_000_000);o.offer(t)
        return o,legacy,t

    def test_candidate_replacement_during_query_retains_only_context_demand(self):
        o,f,t=self.fixture();original=o.query
        def query(*args):
            new=copy.deepcopy(t);new.points[1].positions[0]=.01;o.offer(new)
            return original(*args)
        o.query=query;o.tick()
        self.assertEqual(len(f.sent),1);r=f.sent[0]
        self.assertFalse(r['candidate_current_at_publish']);self.assertFalse(r['execution_eligible'])
        validate_context_report(r,o.observation_context.snapshot(f.f.now),10.)

    def test_cancel_during_query_rejects_result(self):
        o,f,t=self.fixture();original=o.query;cancelled=[]
        def query(*args):
            if not cancelled:
                o.offer_context(event(seq=2,active=False));cancelled.append(True)
            return original(*args)
        o.query=query;o.tick();self.assertFalse(f.sent)
        self.assertIn('context_changed',f.rows[-1]['reason'])

    def test_expired_candidate_can_only_refresh_retained_work(self):
        o,f,t=self.fixture();o.tick();f.f.now+=600_000_000;o.offer_state(curves.state(f.f.now))
        o.tick();self.assertEqual(len(f.sent),2);self.assertTrue(f.sent[-1]['retained_refresh_only'])
        self.assertEqual(len(f.f.boxes),1)

    def test_new_context_rejects_old_business_stamp(self):
        o,f,t=self.fixture();f.f.now+=100_000_000;o.offer_context(event(f.f.now,generation=2,seq=2))
        o.offer(t);o.offer_state(curves.state(f.f.now));o.tick();self.assertFalse(f.sent)

    def test_old_query_failure_cannot_invalidate_a_new_context(self):
        o,f,t=self.fixture()
        def query(*args):
            o.offer_context(event(generation=2,seq=2))
            raise ValueError('observation_context_map_changed')
        o.query=query;o.tick()
        self.assertFalse(f.sent)
        self.assertEqual(o.observation_context.snapshot(f.f.now)['generation'],2)


class SchedulerTests(unittest.TestCase):
    def fixture(self):
        adapter=old.ProductionAdapterTests();n=adapter.node();clock=adapter.clock
        path=ROOT/'src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py'
        cls=next(c for c in ast.parse(path.read_text()).body if isinstance(c,ast.ClassDef))
        names={'_clear_stopping_context','_expire_stopping_context','_stopping_context_callback','_process_context_observation_demands'}
        ns=dict(np=np,json=json,context_key=context_key,context_path=context_path,
            exact_region_owner=exact_region_owner,validate_context_report=validate_context_report,
            rospy=NS(Time=NS(now=lambda:NS(to_sec=lambda:clock[0],to_nsec=lambda:round(clock[0]*1e9))),logwarn=lambda *a:None))
        exec(compile(ast.Module(body=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in names],type_ignores=[]),str(path),'exec'),ns)
        for name in names:setattr(n,name,ns[name].__get__(n))
        n._schedule_publish_lock=threading.RLock();n._stopping_context_enabled=True
        n._stopping_context=ObservationContext();n._stopping_context_queue=ContextObservationDemandQueue()
        n._stopping_context_worker=threading.Lock();n._stopping_context_key=None
        n._stopping_context_callback(NS(data=json.dumps(event())))
        return n,clock

    def test_actual_scheduler_survives_candidate_change_and_resolves_sensor_update(self):
        n,clock=self.fixture();n._stopping_prefix_pending=report();n._process_context_observation_demands()
        self.assertEqual(len(n._obligations),1);first=n._obligations[0]['id']
        n.trajectory.header.stamp=curves.duration(10_100_000_000)
        rr=report();rr['path_id']='another';n._stopping_prefix_pending=rr;n._process_context_observation_demands()
        self.assertEqual(n._obligations[0]['id'],first)
        f=old.Fixture();f.now=10_100_000_000;f.provenance=1;f.last=10.1;f.confidence=1.;clock[0]=10.1
        n._stopping_prefix_pending=report(f);n._process_context_observation_demands();self.assertFalse(n._obligations)

    def test_cancel_during_generation_cannot_append_late_obligation(self):
        n,clock=self.fixture();original=n._generate_new_obligation
        def generate(*args):
            value=original(*args);n._stopping_context_callback(NS(data=json.dumps(event(seq=2,active=False))));return value
        n._generate_new_obligation=generate;n._stopping_prefix_pending=report();n._process_context_observation_demands()
        self.assertFalse(n._obligations)

    def test_map_invalidation_message_immediately_clears_schedule(self):
        n,clock=self.fixture();r=report();n._stopping_prefix_pending=r;n._process_context_observation_demands()
        # Use the actual callback from the adapter, with a clock exposing ns.
        method=n._stopping_prefix_callback.__func__
        method.__globals__['context_key']=context_key
        method.__globals__['rospy'].Time.now=lambda:NS(to_sec=lambda:clock[0],to_nsec=lambda:round(clock[0]*1e9))
        n._stopping_prefix_callback(NS(data=json.dumps(dict(kind='observation_context_invalidated',
            observation_context=r['observation_context'],reason='observation_context_map_changed'))))
        self.assertFalse(n._obligations);self.assertTrue(n._stopping_context.closed)

    def test_heartbeat_expiry_clears_without_new_report(self):
        n,clock=self.fixture();n._stopping_prefix_pending=report();n._process_context_observation_demands()
        clock[0]=11.1
        with n._schedule_publish_lock:n._expire_stopping_context()
        self.assertFalse(n._obligations)

    def test_cancellation_publication_cannot_be_followed_by_old_snapshot(self):
        n,clock=self.fixture();n._stopping_prefix_pending=report();n._process_context_observation_demands()
        path=ROOT/'src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py'
        cls=next(c for c in ast.parse(path.read_text()).body if isinstance(c,ast.ClassDef))
        method=next(m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name=='_publish_schedule')
        ns={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
        n._publish_schedule=ns[method.name].__get__(n)
        entered=threading.Event();release=threading.Event();published=[];errors=[]
        def snapshot():
            value=bool(n._obligations)
            if not published:entered.set();release.wait(2.)
            published.append(value)
        n._publish_schedule_snapshot=snapshot
        def cancel():
            try:n._stopping_context_callback(NS(data=json.dumps(event(seq=2,active=False))))
            except Exception as exc:errors.append(exc)
        first=threading.Thread(target=n._publish_schedule);first.start();self.assertTrue(entered.wait(2.))
        second=threading.Thread(target=cancel);second.start();release.set()
        first.join(2.);second.join(2.)
        self.assertFalse(first.is_alive() or second.is_alive());self.assertFalse(errors)
        self.assertEqual(published,[True,False])


if __name__=='__main__':unittest.main()
