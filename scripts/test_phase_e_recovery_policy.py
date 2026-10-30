#!/usr/bin/env python3
"""Bound recovery snapshots: identity, actual C4 adoption and producer races."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/care_visibility_cdf/scripts'),str(ROOT/'scripts')]
from recovery_policy import (build_policy,selection_record,validate_selection,target_sha,content_sha)
from task_recovery_events import (RecoverySnapshotSource,TaskEventSource,TaskEventReceiver,
                                  validate_recovery_snapshot,validate_regime_control)
from task_observation_binding import canonical
from stopping_observation_context import context_key
from observation_identity import observation_token,observation_region_token
import test_phase_e_task_recovery as fixtures
target=fixtures.target


class PolicyTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.EventTests();f.setUp();self.context=f.context;self.now=f.now
        self.target=target();self.target['sensor_id']=2
        self.result=dict(observation_token=self.target['observation_token'],
            region_token=self.target['region_token'],q_eval=np.zeros(7),measured_stamp=19.,
            target=np.full(7,.03),q_vis_joint_mask=np.ones(7),
            frontier_weight_scale=1.,qvis_weight_scale=0.)
        self.source=TaskEventSource('regime')
        self.control=self.source.event('regime_control',self.context,
            dict(mode='REPAIR',mode_revision=2,ready=True,clear=False,replan_seq=0,
                 reference_refresh_seq=0,force_bootstrap=False,qp_retry='',
                 recovery_target_sha256=target_sha(self.target)),self.now)

    def policy(self):return build_policy(self.target,self.result,self.context,self.control,self.now)

    def test_target_version_is_stable_across_heartbeat(self):
        source=RecoverySnapshotSource()
        a=source.snapshot(self.context,1,self.target,self.now)
        b=source.snapshot(self.context,1,self.target,self.now+1)
        self.assertEqual(a['payload']['selection'],b['payload']['selection'])
        self.assertEqual(a['payload']['target']['deadline_ns'],b['payload']['target']['deadline_ns'])

    def test_content_change_changes_version_even_when_observation_token_is_same(self):
        other=copy.deepcopy(self.target);other['q_vis'][0]=.01
        self.assertNotEqual(target_sha(other),target_sha(self.target))

    def test_no_control_is_target_only_bootstrap(self):
        self.assertIsNone(build_policy(self.target,self.result,self.context,None,self.now))
        self.assertIsNone(validate_selection(selection_record(self.target),self.target))

    def test_control_must_acknowledge_exact_target_content(self):
        self.control['payload']['recovery_target_sha256']='a'*64
        self.assertIsNone(self.policy())

    def test_control_must_be_current_task_ready_repair(self):
        for key,value in [('mode','NORMAL'),('ready',False)]:
            old=self.control['payload'][key];self.control['payload'][key]=value
            self.assertIsNone(self.policy());self.control['payload'][key]=old
        self.control['task_context']='other'
        self.assertIsNone(self.policy())

    def test_policy_expiry_is_measurement_anchored(self):
        a=self.policy();self.now+=50000000
        self.control=self.source.event('regime_control',self.context,self.control['payload'],self.now)
        b=self.policy()
        self.assertEqual(a['valid_until_ns'],19000000000+200000000)
        self.assertEqual(a['valid_until_ns'],b['valid_until_ns'])
        self.now=19200000000
        self.assertIsNone(self.policy())

    def test_fresh_soft_objective_does_not_extend_original_demand_deadline(self):
        self.target['deadline_ns']=self.now-1
        self.control['payload']['recovery_target_sha256']=target_sha(self.target)
        p=self.policy()
        e=RecoverySnapshotSource().snapshot(self.context,1,self.target,self.now,policy=p)
        self.assertEqual(e['payload']['target']['deadline_ns'],self.now-1)
        self.assertFalse(e['payload']['complete'])
        self.assertEqual(p['valid_until_ns'],self.now+200000000)

    def test_future_or_pre_task_measurement_cannot_supply_policy(self):
        for stamp in (18.999,19.001,float('nan')):
            self.result['measured_stamp']=stamp
            self.assertIsNone(self.policy())

    def test_policy_roundtrip_keeps_measurement_and_binding(self):
        p=self.policy();s=selection_record(self.target,p)
        self.assertEqual(validate_selection(s,self.target,self.context['candidate_binding']),p)
        self.assertEqual(p['measured_ns'],self.now)

    def test_tampered_policy_bytes_or_target_are_rejected(self):
        s=selection_record(self.target,self.policy())
        altered=copy.deepcopy(s);altered['policy_content']+=' '
        with self.assertRaises(ValueError):validate_selection(altered,self.target)
        other=copy.deepcopy(self.target);other['joint_mask'][0]=0.
        with self.assertRaises(ValueError):validate_selection(s,other)

    def test_valid_digest_cannot_relabel_policy_to_another_target(self):
        p=self.policy();p['target_sha256']='0'*64
        s=selection_record(self.target);s['policy_content']=canonical(p)
        s['policy_sha256']=content_sha('policy',s['policy_content'])
        with self.assertRaises(ValueError):validate_selection(s,self.target)

    def test_policy_cannot_cross_task_even_with_valid_content_digest(self):
        s=selection_record(self.target,self.policy())
        with self.assertRaises(ValueError):validate_selection(s,self.target,'other')

    def test_guidance_must_share_pose_time_mask_sensor_and_target(self):
        guide=dict(sequence=1,sensor_id=2,value=.1,q_eval=np.zeros(7),target=np.full(7,.04),
            gradient=np.ones(7),joint_mask=np.ones(7),measured_stamp=19.,
            observation_token=self.target['observation_token'],region_token=self.target['region_token'],
            drive_enabled=True,geometry_valid=True,geometry_signature='geometry',conservative_fov_margin=.1)
        self.result['guidance']=guide
        self.assertTrue(self.policy()['guidance']['drive_enabled'])
        for key,value in [('measured_stamp',18.99),('observation_token','other'),('q_eval',np.ones(7))]:
            old=guide[key];guide[key]=value;self.assertIsNone(self.policy());guide[key]=old
        guide['sensor_id']=3
        with self.assertRaises(ValueError):self.policy()

    def test_invalid_frontier_does_not_become_a_policy(self):
        for key,value in [('target',[float('nan')]*7),('frontier_weight_scale',-1.)]:
            old=self.result[key];self.result[key]=value
            with self.assertRaises(ValueError):self.policy()
            self.result[key]=old

    def test_snapshot_policy_does_not_clear_or_complete_demand(self):
        e=RecoverySnapshotSource().snapshot(self.context,1,self.target,self.now,policy=self.policy())
        validate_recovery_snapshot(e['payload'],e['task_context'])
        self.assertFalse(e['payload']['complete']);self.assertEqual(e['payload']['pending_count'],1)

    def test_source_sequence_replay_does_not_refresh_policy(self):
        e=RecoverySnapshotSource().snapshot(self.context,1,self.target,self.now,policy=self.policy())
        r=TaskEventReceiver();r.remember(r.parse(canonical(e),self.context,self.now,'recovery_snapshot'))
        with self.assertRaises(ValueError):r.parse(canonical(e),self.context,self.now+1,'recovery_snapshot')


class ActualC4PolicyTests(fixtures.RegimeTests):
    # Inherit only fixture methods; inherited tests are removed below so IDs
    # remain distinct and the parent suite is not silently counted twice.
    def test_c4_acknowledges_target_and_revises_same_token_content_change(self):
        t=target();source=RecoverySnapshotSource()
        self.send(source.snapshot(self.a,1,t,self.ns())['payload'])
        a=json.loads(self.m.task_control_pub.messages[-1].data)
        self.assertEqual(a['payload']['recovery_target_sha256'],target_sha(t))
        self.now+=.001
        self.send(source.snapshot(self.a,1,t,self.ns())['payload'])
        b=json.loads(self.m.task_control_pub.messages[-1].data)
        self.assertEqual(a['payload']['mode_revision'],b['payload']['mode_revision'])
        t['q_vis'][0]=.01
        self.send(source.snapshot(self.a,1,t,self.ns())['payload'])
        c=json.loads(self.m.task_control_pub.messages[-1].data)
        self.assertGreater(c['payload']['mode_revision'],b['payload']['mode_revision'])
        self.assertEqual(c['payload']['recovery_target_sha256'],target_sha(t))

    def test_task_replacement_clears_adopted_target(self):
        self.send(RecoverySnapshotSource().snapshot(self.a,1,target(),self.ns())['payload'])
        self.replace()
        c=json.loads(self.m.task_control_pub.messages[-1].data)
        self.assertEqual(c['payload']['recovery_target_sha256'],'')

    def test_first_target_after_repair_started_creates_new_control_revision(self):
        self.m.state=self.m.REPAIR;self.m._observation_waypoint=None
        before=self.m.task_mode_revision
        self.send(RecoverySnapshotSource().snapshot(self.a,1,target(),self.ns())['payload'])
        c=json.loads(self.m.task_control_pub.messages[-1].data)
        self.assertGreater(c['payload']['mode_revision'],before)
        self.assertEqual(c['payload']['recovery_target_sha256'],target_sha(target()))


# Copy fixture helpers onto a fresh TestCase instead of retaining test inheritance.
ActualC4PolicyTests=type('ActualC4PolicyTests',(unittest.TestCase,),{
    **{n:getattr(fixtures.RegimeTests,n) for n in ('setUp','ns','context','send','replace')},
    **{n:v for n,v in ActualC4PolicyTests.__dict__.items() if n.startswith('test_')}})


class ProducerTests(PolicyTests):
    def producer(self):
        path=ROOT/'src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py'
        tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
        methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in (
            '_task_target','_task_current_obligations','_publish_task_recovery_snapshot','_task_policy_control_cb')]
        cls=ast.ClassDef(name='Producer',bases=[],keywords=[],decorator_list=[],body=methods)
        import rospy
        scope=dict(np=np,canonical=canonical,context_key=context_key,
            observation_token=observation_token,observation_region_token=observation_region_token,
            q_vis_joint_mask=lambda ob:np.asarray(ob.get('q_vis_joint_mask',[1.]*7)),
            build_policy=build_policy,validate_regime_control=validate_regime_control,
            String=lambda **k:NS(**k),rospy=NS(Time=NS(now=lambda:NS(to_nsec=lambda:self.now))))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),scope)
        p=scope['Producer']();p._require_task_binding=True
        p._schedule_publish_lock=threading.RLock();p._stopping_prefix_lock=threading.RLock()
        p._obligation_lock=threading.Lock();p._task_policy_control=self.control
        p._task_policy_events=TaskEventReceiver();p._task_recovery_source=RecoverySnapshotSource()
        p._stopping_context=NS(snapshot=lambda now:self.context)
        p._stopping_context_queue=NS(pending=[object()])
        ob=dict(id=1,q_vis=np.zeros(7),q_vis_joint_mask=np.ones(7),points=np.array([[.3,0.,.5]]),
            deadline_abs_s=20.,per_sensor_selected_sensor_id=2,
            stopping_prefix_request=dict(context_key=list(context_key(self.context))))
        p._obligations=[ob];p._ordered_obligations=lambda:copy.deepcopy(p._obligations)
        p.messages=[];p._task_recovery_pub=NS(publish=lambda msg:p.messages.append(json.loads(msg.data)))
        return p

    def test_frozen_target_replacement_during_compute_cannot_publish(self):
        p=self.producer();frozen=copy.deepcopy(p._obligations)
        p._obligations[0]['q_vis'][0]=.1
        p._publish_task_recovery_snapshot(obligations=frozen,frontier=None)
        self.assertEqual(p.messages,[])

    def test_frozen_selection_is_not_reordered_during_publication(self):
        p=self.producer();frozen=copy.deepcopy(p._obligations)
        p._ordered_obligations=lambda:self.fail('selection was recomputed')
        p._publish_task_recovery_snapshot(obligations=frozen,frontier=None)
        self.assertEqual(len(p.messages),1)
        self.assertEqual(p.messages[0]['payload']['target']['deadline_ns'],20000000000)

    def test_derived_shared_pose_is_checked_against_its_stored_source(self):
        p=self.producer();frozen=copy.deepcopy(p._obligations)
        frozen[0]['_task_storage_target']=p._task_target(frozen[0])
        frozen[0]['q_vis'][0]=.1
        p._publish_task_recovery_snapshot(obligations=frozen)
        self.assertEqual(len(p.messages),1)
        self.assertEqual(p.messages[0]['payload']['target']['q_vis'][0],.1)

    def test_old_context_selection_cannot_publish_into_new_task(self):
        p=self.producer();frozen=copy.deepcopy(p._obligations)
        p._obligations=[]
        p._publish_task_recovery_snapshot(obligations=frozen)
        self.assertEqual(p.messages,[])

    def test_task_filter_removes_unbound_or_old_task_before_selecting_frontier(self):
        p=self.producer();current=p._obligations[0]
        old=copy.deepcopy(current);old['stopping_prefix_request']['context_key']=['old']
        bare=copy.deepcopy(current);bare.pop('stopping_prefix_request')
        selected=p._task_current_obligations([old,bare,current])
        self.assertEqual(len(selected),1);self.assertIs(selected[0],current)

    def test_duplicate_and_mutated_same_revision_control_are_rejected(self):
        p=self.producer();p._task_policy_control=None
        p._task_policy_control_cb(NS(data=canonical(self.control)))
        accepted=p._task_policy_control
        p._task_policy_control_cb(NS(data=canonical(self.control)))
        other=self.source.event('regime_control',self.context,
            dict(self.control['payload'],recovery_target_sha256='b'*64),self.now)
        p._task_policy_control_cb(NS(data=canonical(other)))
        self.assertIs(p._task_policy_control,accepted)


ProducerTests=type('ProducerTests',(unittest.TestCase,),{
    'setUp':PolicyTests.setUp,'producer':ProducerTests.producer,
    **{n:v for n,v in ProducerTests.__dict__.items() if n.startswith('test_')}})

if __name__=='__main__':unittest.main()
