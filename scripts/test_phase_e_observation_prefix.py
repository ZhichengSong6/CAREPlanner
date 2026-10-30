#!/usr/bin/env python3
import copy
import ast
import json
import math
import re
import threading
import time
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from stopping_observation_demands import (query_prefix_observation_demands, ObservationDemandQueue,
    validate_report,exact_region_owner,inherit_stopping_request)
from stopping_trajectory_prefix import trajectory_sha256, validate_trajectory_report


class Fixture:
    def __init__(self):
        self.now=10_000_000_000;self.wall=0.;self.revision=4;self.calls=[];self.boxes=[]
        self.provenance=0;self.last=-1.;self.confidence=0.;self.occupancy=0.;self.changed=False
        self.configuration_limits=[(-3.,3.)]*7
    def raster(self,q,travel,meta,margin,subdivisions,check_budget):
        check_budget();self.boxes.append((q,travel,margin))
        return dict(inside=True,grid_indices=[1],points=[[.05,0.,0.]])
    def query(self,points,horizon,expected):
        self.calls.append((points,horizon,expected));n=len(points)
        r=dict(valid=True,clock_fault=False,instance='map',epoch=0,revision=self.revision,
            frame='base_link',resolution=.05,map_min=[0.,0.,0.],map_max=[1.,1.,1.],grid_size=[21]*3,
            snapshot_ns=self.now,temporal_decay_time=100.,inside_map=[1]*n,provenance=[self.provenance]*n,
            confidence_now=[self.confidence]*n,occupancy=[self.occupancy]*n,last_seen_s=[self.last]*n,
            model_free_at_horizon=[int(self.confidence>=.8 and 0<=self.occupancy<=.5)]*n)
        if expected and self.changed:r['revision']+=1
        return r
    def scan(self,**kw):
        segments=[dict(q_low=[.5]*7,q_high=[.7]*7,stop_travel_rad=[.02]*7,first_use_s=15.,required_until_s=16.)]
        values=dict(segments=segments,geometry=self,query=self.query,now_ns=lambda:self.now,path_id='path',
            trajectory_stamp_ns=8_000_000_000,observation_timing=dict(motion_s=2.,acquisition_s=.1,verification_s=.05,stop_reserve_s=.2),
            wall_clock=lambda:self.wall)
        values.update(kw);return query_prefix_observation_demands(**values)


class PrefixTests(unittest.TestCase):
    def test_observation_and_task_boxes_respect_actual_joint_limits(self):
        f=Fixture();f.configuration_limits[-1]=(-.5,.5)
        with self.assertRaisesRegex(ValueError,'joint_limit'):f.scan()
    def test_future_cells_generate_requests_before_current_stop_uses_them(self):
        f=Fixture();r=f.scan();c=r['cells'][0]
        self.assertEqual(c['action'],'first_observation');self.assertEqual(c['observation_deadline_s'],15.)
        self.assertAlmostEqual(c['latest_start_s'],12.65);self.assertFalse(c['scheduling_overdue'])
        self.assertFalse(r['runtime_admission']);self.assertTrue(r['planning_only'])
    def test_entire_segment_box_and_stopping_travel_are_enclosed(self):
        f=Fixture();f.scan();q,travel,margin=f.boxes[0]
        for x in q:self.assertAlmostEqual(x,.6)
        for x in travel:self.assertAlmostEqual(x,.12)
        self.assertEqual(margin,.020)
    def test_currently_valid_sensor_evidence_can_require_proactive_refresh(self):
        f=Fixture();f.provenance=1;f.last=0.;f.confidence=math.exp(-.1)
        segment=dict(q_low=[0.]*7,q_high=[0.]*7,stop_travel_rad=[.01]*7,first_use_s=10.,required_until_s=30.)
        r=f.scan(segments=[segment]);c=r['cells'][0]
        self.assertEqual(c['action'],'refresh_observation');self.assertGreater(c['confidence_now'],.8)
        self.assertAlmostEqual(c['latest_start_s'],-100*math.log(.8)-.2-2.35)
    def test_frozen_prior_and_sensor_memory_remain_separate(self):
        f=Fixture();f.provenance=2;f.confidence=1.
        self.assertEqual(f.scan()['cells'][0]['action'],'covered')
        f.provenance=1;f.last=9.;self.assertEqual(f.scan()['cells'][0]['action'],'covered')
    def test_future_tf_or_oracle_cannot_satisfy_observation_request(self):
        f=Fixture();f.provenance=1;f.last=11.;f.confidence=1.
        with self.assertRaisesRegex(ValueError,'future_or_invalid'):f.scan()
        f.provenance=5;f.last=9.
        self.assertEqual(f.scan()['cells'][0]['action'],'first_observation')
    def test_occupied_cells_are_explicit_blockers(self):
        f=Fixture();f.provenance=1;f.last=9.;f.confidence=1.;f.occupancy=1.
        self.assertEqual(f.scan()['cells'][0]['action'],'blocked_occupied')
    def test_planning_occupancy_threshold_matches_the_complete_query_contract(self):
        f=Fixture();f.provenance=1;f.last=9.;f.confidence=1.;f.occupancy=.4
        self.assertEqual(f.scan()['cells'][0]['action'],'covered')
    def test_version_change_and_late_work_are_rejected(self):
        f=Fixture();f.changed=True
        with self.assertRaisesRegex(ValueError,'version_changed'):f.scan()
        f=Fixture();original=f.raster
        def delayed(*args,**kw):
            result=original(*args,**kw);f.wall=.6;return result
        f.raster=delayed
        with self.assertRaisesRegex(ValueError,'planning_budget_exhausted'):f.scan()
    def test_report_is_bound_to_actual_path_stamp_and_fresh_snapshot(self):
        f=Fixture();r=f.scan()
        with self.assertRaisesRegex(ValueError,'unbound'):validate_report(r,10.,1)
        with self.assertRaisesRegex(ValueError,'unbound'):validate_report(r,11.)
    def test_missing_route_time_is_not_replaced_by_a_fixed_warning_second(self):
        with self.assertRaisesRegex(ValueError,'route_timing'):Fixture().scan(observation_timing={})


class QueueTests(unittest.TestCase):
    def test_repeated_old_observation_cannot_postpone_the_request(self):
        f=Fixture();q=ObservationDemandQueue();r=f.scan();q.update(r,10.)
        later=copy.deepcopy(r)
        for k in ('first_use_s','required_until_s','observation_deadline_s','latest_start_s'):later['cells'][0][k]+=5.
        pending=q.update(later,10.)
        self.assertEqual(pending[0]['observation_deadline_s'],15.)
        self.assertAlmostEqual(pending[0]['latest_start_s'],12.65)
    def test_disappearance_from_a_candidate_does_not_resolve_a_request(self):
        f=Fixture();q=ObservationDemandQueue();r=f.scan();q.update(r,10.)
        other=copy.deepcopy(r);other['cells'][0]['grid_index']=2;other['cells'][0]['point']=[.1,0.,0.]
        self.assertEqual(len(q.update(other,10.)),2)
    def test_only_a_new_real_sensor_observation_with_adequate_horizon_resolves(self):
        f=Fixture();q=ObservationDemandQueue();q.update(f.scan(),10.)
        f.now+=100_000_000;f.provenance=2;f.confidence=1.
        self.assertEqual(len(q.update(f.scan(),10.1)),1)
        f.provenance=1;f.last=10.1
        self.assertEqual(q.update(f.scan(),10.1),[])
    def test_capacity_and_map_restart_do_not_drop_existing_requests(self):
        f=Fixture();q=ObservationDemandQueue(1);r=f.scan();q.update(r,10.)
        other=copy.deepcopy(r);other['cells'][0]['grid_index']=2
        with self.assertRaisesRegex(ValueError,'capacity'):q.update(other,10.)
        self.assertEqual(len(q.pending),1)
        other=copy.deepcopy(r);other['map_identity'][0]='restarted'
        with self.assertRaisesRegex(ValueError,'identity_changed'):q.update(other,10.)
        self.assertEqual(len(q.pending),1)
    def test_explicit_retirement_is_separate_from_observation_completion(self):
        f=Fixture();q=ObservationDemandQueue();q.update(f.scan(),10.)
        q.retire('other');self.assertEqual(len(q.pending),1)
        q.retire('path');self.assertEqual(len(q.pending),0)


class ProductionAdapterTests(unittest.TestCase):
    def node(self):
        source=Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py'
        cls=next(c for c in ast.parse(source.read_text()).body if isinstance(c,ast.ClassDef))
        names={'_stopping_prefix_callback','_process_stopping_prefix_demands','_candidate_vbc_summary_callback'}
        methods=[m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name in names]
        self.clock=[10.];clock=self.clock
        namespace=dict(json=json,math=math,np=np,String=object,validate_report=validate_report,
            trajectory_sha256=trajectory_sha256,validate_trajectory_report=validate_trajectory_report,
            exact_region_owner=exact_region_owner,
            _TOKEN=re.compile(r'([A-Za-z0-9_]+)=([^\s]+)'),
            rospy=NS(Time=NS(now=lambda:NS(to_sec=lambda:clock[0])),logwarn=lambda *a:None))
        exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),namespace)
        n=type('ActualAdapterMethods',(),{k:namespace[k] for k in names})()
        n._stopping_prefix_lock=threading.RLock();n._stopping_prefix_queue=ObservationDemandQueue()
        n._stopping_prefix_pending=None;n._stopping_prefix_path=None;n._lock=threading.Lock()
        n._obligation_lock=threading.Lock();n._obligations=[];n.max_obligations=12
        n._schedule_clear_count=0;n._publish_schedule=lambda:None
        n.trajectory=NS(header=NS(stamp=NS(to_nsec=lambda:8_000_000_000),seq=1,frame_id='base_link'),
            joint_names=['j'+str(i) for i in range(7)],points=[NS(positions=[0.]*7,
                velocities=[],accelerations=[],effort=[],time_from_start=NS(to_nsec=lambda:0))])
        n._preferred_trajectory_locked=lambda:(n.trajectory,None,'test')
        n._cluster_regions=lambda points:[dict(points=points)]
        n._match_existing=lambda region:next((o for o in n._obligations if np.array_equal(o['points'],region['points'])),None)
        n._generate_new_obligation=lambda region,*args:dict(id=1,points=region['points'],deadline_abs_s=100.)
        return n
    def publish(self,n,r):
        r=copy.deepcopy(r);r['trajectory_sha256']=trajectory_sha256(n.trajectory)
        n._stopping_prefix_callback(NS(data=json.dumps(r)));n._process_stopping_prefix_demands()
    def test_actual_scheduler_generates_requests_and_safe_current_prefix_does_not_clear_them(self):
        n=self.node();self.publish(n,Fixture().scan())
        self.assertEqual(len(n._obligations),1);self.assertEqual(n._obligations[0]['deadline_abs_s'],15.)
        n._candidate_vbc_summary_callback(NS(data='trajectory_source=predicted has_violation=0'))
        self.assertEqual(len(n._obligations),1)
        f=Fixture();f.now=10_100_000_000;f.provenance=1;f.last=10.1;f.confidence=1.;self.clock[0]=10.1
        self.publish(n,f.scan());self.assertEqual(n._obligations,[])
    def test_legacy_matching_obligation_survives_resolution_until_its_own_safe_verdict(self):
        n=self.node();n._obligations=[dict(id=9,points=np.array([[.05,0.,0.]]),deadline_abs_s=100.)]
        self.publish(n,Fixture().scan());self.assertEqual(n._obligations[0]['id'],9)
        f=Fixture();f.now=10_100_000_000;f.provenance=1;f.last=10.1;f.confidence=1.;self.clock[0]=10.1
        self.publish(n,f.scan());self.assertEqual(len(n._obligations),1)
        self.assertNotIn('stopping_prefix_request',n._obligations[0])
        n._candidate_vbc_summary_callback(NS(data='trajectory_source=predicted has_violation=0'))
        self.assertEqual(n._obligations,[])
    def test_wrong_current_trajectory_or_generation_race_cannot_append_a_request(self):
        n=self.node();n.trajectory.header.stamp.to_nsec=lambda:999
        self.publish(n,Fixture().scan());self.assertEqual(n._obligations,[])
        n=self.node();original=n._generate_new_obligation
        def changed(*args):
            value=original(*args);n.trajectory.header.stamp.to_nsec=lambda:999;return value
        n._generate_new_obligation=changed
        self.publish(n,Fixture().scan());self.assertEqual(n._obligations,[])

    def test_nearby_region_cannot_own_a_different_required_cell(self):
        wrong=dict(points=[[.051,0.,0.]])
        right=dict(points=[[.05,0.,0.],[.1,0.,0.]])
        self.assertIs(exact_region_owner([wrong,right],[[.05,0.,0.]]),right)
        self.assertIsNone(exact_region_owner([wrong],[[.05,0.,0.]]))
    def test_spatial_refinement_keeps_only_the_childs_required_cells_and_deadline(self):
        parent=dict(points=[[.05,0.,0.],[.1,0.,0.]],deadline_abs_s=15.,stopping_prefix_owned=True,
            stopping_prefix_request=dict(path_id='p',points=[[.05,0.,0.]]))
        child=inherit_stopping_request(parent,dict(points=np.array([[.05,0.,0.]]),deadline_abs_s=20.))
        self.assertEqual(child['deadline_abs_s'],15.);self.assertTrue(child['stopping_prefix_owned'])
        other=inherit_stopping_request(parent,dict(points=np.array([[.1,0.,0.]]),deadline_abs_s=20.))
        self.assertNotIn('stopping_prefix_request',other)
    def test_current_visibility_does_not_clear_a_prefix_request_with_a_longer_horizon(self):
        source=Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py'
        method=next(m for c in ast.parse(source.read_text()).body if isinstance(c,ast.ClassDef)
            for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='_update_actual_visibility_completion')
        namespace=dict(np=np,observation_region_token=lambda ob:str(ob['id']),
            rospy=NS(Time=NS(now=lambda:NS(to_sec=lambda:10.)),logwarn=lambda *a:None))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        n=NS(_c47_ready=True,_last_visibility_check_s=0.,visibility_check_rate=10.,_acquisition_started=True,
            _obligation_lock=threading.Lock(),_schedule_publish_lock=threading.Lock(),_seen_obligation_count=0,
            _trace_seen_samples={},_obligations=[dict(id=1,stopping_prefix_request={'path_id':'p'}),dict(id=2)])
        n._query_region_seen=lambda ob:(True,1.,1.,1.,1.,1.,[0.,0.,0.],0.,dict(q_distance_inf=0.))
        n._record_seen_query=lambda *a:None;n._obligation_geometry_diagnostics=lambda ob:{}
        n._trace_observation=lambda *a,**k:None;n._publish_schedule=lambda:None
        n._ordered_obligations=lambda:n._obligations;n._set_acquisition_completion=lambda value:setattr(n,'complete',value)
        n._publish_acquisition_summary=lambda *a:None
        namespace[method.name](n)
        self.assertEqual([o['id'] for o in n._obligations],[1]);self.assertFalse(n.complete)


class PersistentTransportTests(unittest.TestCase):
    def transport(self,fn):
        from phase_e_stop_shadow_node import PersistentServiceTransport
        self.created=[];self.closed=[]
        def factory():
            obj=NS()
            obj.close=lambda:self.closed.append(obj)
            class Proxy:
                def __call__(inner,request):return fn(request)
                def close(inner):obj.close()
            proxy=Proxy();self.created.append(proxy);return proxy
        return PersistentServiceTransport(factory,lambda x:x,lambda x:x)
    def test_connection_is_reused_but_every_response_belongs_to_its_request(self):
        t=self.transport(lambda x:x+1)
        self.assertEqual([t(1),t(2)],[2,3]);self.assertEqual(len(self.created),1)
    def test_transport_error_is_not_retried_and_the_next_request_reconnects(self):
        requests=[]
        def call(x):
            requests.append(x)
            if x==1:raise RuntimeError('transport failed')
            return x
        t=self.transport(call)
        with self.assertRaisesRegex(RuntimeError,'transport failed'):t(1)
        self.assertEqual(requests,[1]);self.assertEqual(len(self.closed),1)
        self.assertEqual(t(2),2);self.assertEqual(len(self.created),2)
    def test_late_response_is_discarded_before_reusing_the_connection(self):
        from phase_e_stop_shadow_node import BoundedQuery
        ready=threading.Event();release=threading.Event()
        def call(x):
            if x==1:ready.set();release.wait(1.)
            return x
        t=self.transport(call);q=BoundedQuery(t,.01)
        with self.assertRaisesRegex(RuntimeError,'timeout'):q(1)
        self.assertTrue(ready.is_set())
        with self.assertRaisesRegex(RuntimeError,'busy'):q(2)
        release.set();q.thread.join(1.)
        self.assertEqual(q(3),3);self.assertEqual(len(self.created),1)


if __name__=='__main__':unittest.main()
