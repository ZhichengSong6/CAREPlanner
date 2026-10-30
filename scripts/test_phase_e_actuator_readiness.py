#!/usr/bin/env python3
"""Receiver identity, counters, zero stream and original startup deadline."""
import sys
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import phase_e_actuator_readiness as readiness
import phase_e_benchmark_arm as arm

CONFIG=dict(tracker_node='/tracker',limiter_node='/limiter',controller_node='/controller',
            desired_topic='/desired',command_topic='/command')


class Tests(unittest.TestCase):
    def rpc_result(self, result, method='getBusStats'):
        from unittest.mock import MagicMock
        proxy=MagicMock();proxy.__enter__.return_value=proxy
        getattr(proxy,method).return_value=result
        with patch.object(readiness.xmlrpc.client,'ServerProxy',return_value=proxy):
            return readiness.rpc('http://receiver',method,readiness.time.monotonic()+1.,'/watcher')

    def test_gazebo_raw_stats_and_wrapped_rospy_stats(self):
        # Topic-table subset of the saved host Gazebo reply; the real C++
        # receiver regression separately exercises live receipt counters.
        raw=[[['/clock',[[2,0,0,0,0]]]],
             [['/clock',[[1,1274952,106246,39,0]]],
              ['/care_arm/arm_group_velocity_controller/command',[]]]]
        self.assertEqual(len(raw),2)
        self.assertEqual(self.rpc_result(raw),raw)
        payload=[[],[['/command',[[7,100,5,-1,True]]]],[]]
        self.assertEqual(self.rpc_result([1,'',payload]),payload)
        cpp=[[],[['/command',[[7,100,5,0,0]]]]]
        self.assertEqual(self.rpc_result(cpp),cpp)
        self.assertEqual(self.rpc_result(cpp+[None]),cpp+[None])
        info=[[7,'/sender_uri','i','TCPROS','/command',True]]
        self.assertEqual(readiness.received_count(info,self.rpc_result(cpp),'/command','/sender','/sender_uri'),(7,5))

    def test_malformed_stats_and_raw_other_methods_fail(self):
        for value in ([0,'failed',[]],[True,'',[]],[[],[['/command',[[7,100,'5',0,0]]]]],
                      [[],[['/command',[[7,100,5,0,1]]]]],[[],['bad row']], [[],[],'invalid services']):
            with self.subTest(value=value),self.assertRaises(RuntimeError):self.rpc_result(value)
        with self.assertRaises(RuntimeError):self.rpc_result([[],[]],'getBusInfo')

    def check(self, mode):
        clock=NS(t=0.,count=10)
        node=readiness.ActuatorReadiness.__new__(readiness.ActuatorReadiness)
        node.config=CONFIG;node.rospy=NS(get_name=lambda:'/watcher');node.master_uri='master'
        node.lock=threading.Lock();node.zero_time=0.;node.previous=None;node.last={}
        def rpc(uri,method,deadline,*args):
            if mode=='exception':raise TimeoutError('receiver RPC')
            if method=='lookupNode':return args[-1]+'_uri'
            controller=uri=='/controller_uri';topic='/command' if controller else '/desired'
            peer='/limiter_uri' if controller else '/tracker_uri'
            if mode=='wrong_peer' and controller:peer='/recorder_uri'
            if method=='getBusInfo':return [[1,peer,'i','TCPROS',topic,mode!='disconnected']]
            count=clock.count if mode!='stagnant' else 10
            return [[],[[topic,[[1,100,count,-1,0 if mode=='roscpp' and controller else True]]]],[]]
        with patch.object(readiness,'rpc',rpc),patch.object(readiness.time,'monotonic',lambda:clock.t):
            self.assertFalse(node.check(20.))
            clock.t=.05;clock.count+=5
            if mode=='nonzero':node.command(NS(data=[.1]+[0.]*6))
            if mode=='stale_zero':node.zero_time=-1.
            return node.check(20.),node.last

    def test_actual_receiver_counters_and_zero_stream_can_ready(self):self.assertTrue(self.check('healthy')[0])
    def test_roscpp_placeholder_requires_real_counter_progress(self):self.assertTrue(self.check('roscpp')[0])
    def test_registration_or_echo_cannot_substitute_for_receiver(self):self.assertFalse(self.check('wrong_peer')[0])
    def test_connected_but_no_received_messages_cannot_ready(self):self.assertFalse(self.check('stagnant')[0])
    def test_disconnected_bus_cannot_ready(self):self.assertFalse(self.check('disconnected')[0])
    def test_nonzero_stream_cannot_arm(self):self.assertFalse(self.check('nonzero')[0])
    def test_stale_zero_cannot_arm(self):self.assertFalse(self.check('stale_zero')[0])
    def test_rpc_failure_cannot_arm(self):self.assertFalse(self.check('exception')[0])

    def test_drop_estimate_is_not_the_connected_flag(self):
        info=[[1,'sender','i','TCPROS','topic',True]]
        self.assertEqual(readiness.received_count(info,[[],[['topic',[[1,100,5,0,True]]]],[]],'topic','sender','uri'),(1,5))
        self.assertIsNone(readiness.received_count(info,[[],[['topic',[[1,100,5,-1,False]]]],[]],'topic','sender','uri'))
        self.assertEqual(readiness.received_count(info,[[],[['topic',[[1,100,5,0,0]]]],[]],'topic','sender','uri'),(1,5))

    def test_original_deadline_and_no_arm_when_actuator_missing(self):
        clock=NS(t=0.);events=[]
        class Pub:
            resolved_name=arm.ARM_TOPIC
            def get_num_connections(self):return 1
        checker=NS(check=lambda deadline:False,last={'ready':False},close=lambda:events.append('close'))
        rospy=NS(Publisher=lambda *a,**kw:Pub(),get_name=lambda:'/watcher',is_shutdown=lambda:False,
            get_param=lambda key,default=None:True if key=='/care_planner/benchmark/actuator_readiness_required' else CONFIG if key=='/care_planner/benchmark/actuator_readiness' else default)
        master=NS(master_uri='master',getSystemState=lambda:([],[('/data',['/recorder'])],[]))
        modules={'rospy':rospy,'rosgraph':NS(Master=lambda _:master),'std_msgs.msg':NS(String=object)}
        with patch.dict(sys.modules,modules),patch.object(readiness,'ActuatorReadiness',lambda *a:checker),\
             patch.object(arm.time,'monotonic',lambda:clock.t),patch.object(arm.time,'sleep',lambda dt:setattr(clock,'t',clock.t+dt)):
            with self.assertRaisesRegex(RuntimeError,'actuator_evidence'):
                arm.prepare_arm(['/recorder'],timeout_s=.2,admission_ping_topic='')
        self.assertLessEqual(clock.t,.2)
        self.assertEqual(events,['close'])


if __name__=='__main__':unittest.main(verbosity=2)
