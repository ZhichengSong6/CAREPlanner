#!/usr/bin/env python3
"""Exercise actual regime callbacks and transition on goal-stop evidence."""
import ast
import os
from pathlib import Path
import re
import threading
from types import MethodType,SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/phase_e_static_baseline_20261030/overlay/src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py'


class Stamp:
    def __init__(self,t=1.):self.t=t
    def to_sec(self):return self.t
    def to_nsec(self):return round(self.t*1e9)
    def __sub__(self,other):return Stamp(self.t-other.t)


class HandoffTests(unittest.TestCase):
    def fixture(self):
        tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        names=['_repair_final_vbc_feedback_cb','_transition_locked','_maybe_enter_observation_repair_locked']
        functions=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
        scope=dict(os=os,_tokens=lambda s:dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',s)),
            _as_int=lambda x,default=0:int(x) if x is not None else default,
            _as_bool=lambda x:True if x=='1' else False if x=='0' else None,
            rospy=SimpleNamespace(Time=SimpleNamespace(now=lambda:Stamp()),logwarn=lambda *args:None),Bool=SimpleNamespace,math=__import__('math'))
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),scope)
        messages=[]
        node=SimpleNamespace(_lock=threading.RLock(),repair_goal_stopping=True,state='REPAIR',
            NORMAL='NORMAL',REPAIR='REPAIR',PROBE_NORMAL='PROBE_NORMAL',execution_ready=True,
            tracker_summary_time=Stamp(.99),tracker_execution_stamp_ns=123,last_tracker_aborted_stamp_ns=0,
            vbc_unsafe_execution_stamp_ns=0,tracker_source='active_trajectory',input_timeout=.25,
            _retire_qp_retry_locked=lambda:None,blocker_rediscovery_force_bootstrap=False,
            normal_entry_count=0,repair_entry_count=1,probe_entry_count=0,
            trigger_pub=SimpleNamespace(publish=messages.append),probe_active_pub=SimpleNamespace(publish=messages.append),
            _observation_demand=('existing',Stamp()),visibility_waypoint_active=True)
        for name in names:setattr(node,name,MethodType(scope[name],node))
        return node,messages

    def message(self,**fields):
        values=dict(final_vbc_hold='0',goal_stop_execution_stamp_ns='123',goal_stop_sample_ros_ns='1000000000')
        values.update(fields)
        return SimpleNamespace(data=' '.join(k+'='+v for k,v in values.items()))

    def test_actual_transition_preserves_observation_and_changes_modes(self):
        node,messages=self.fixture();node._repair_final_vbc_feedback_cb(self.message())
        self.assertEqual(node.state,'NORMAL');self.assertEqual(node.last_transition_reason,'certified_goal_stop_handoff')
        self.assertEqual(node._observation_demand[0],'existing');self.assertTrue(node.visibility_waypoint_active)
        self.assertEqual([m.data for m in messages],[False,False])
        self.assertFalse(node._maybe_enter_observation_repair_locked(Stamp(),'test'))

    def test_invalid_evidence_cannot_change_repair_state(self):
        changes=[('stale_sample',{},dict(goal_stop_sample_ros_ns='900000000')),
                 ('wrong_execution',{},dict(goal_stop_execution_stamp_ns='124')),
                 ('stale_tracker',dict(tracker_summary_time=Stamp(.5)),{}),
                 ('aborted',dict(last_tracker_aborted_stamp_ns=123),{}),
                 ('unsafe',dict(vbc_unsafe_execution_stamp_ns=123),{}),
                 ('tracker_source',dict(tracker_source='stopped_hold'),{}),
                 ('disabled',dict(repair_goal_stopping=False),{})]
        for name,attrs,message in changes:
            with self.subTest(name=name):
                node,_=self.fixture()
                for key,value in attrs.items():setattr(node,key,value)
                node._repair_final_vbc_feedback_cb(self.message(**message))
                self.assertEqual(node.state,'REPAIR')

    def test_zero_certificate_releases_scheduling_latch(self):
        node,_=self.fixture();node._repair_final_vbc_feedback_cb(self.message())
        node._repair_final_vbc_feedback_cb(self.message(goal_stop_execution_stamp_ns='0',goal_stop_sample_ros_ns='0'))
        self.assertEqual(node._goal_stop_execution,0)


if __name__=='__main__':unittest.main()
