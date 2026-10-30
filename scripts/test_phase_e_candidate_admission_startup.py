#!/usr/bin/env python3
"""Startup requires the matching roundtrip without extending its deadline."""
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import phase_e_benchmark_arm as arm


class StartupTests(unittest.TestCase):
    def run_barrier(self, response):
        clock=NS(t=0.);events=[];subscribers={}
        class String:
            _type='std_msgs/String'
            def __init__(self,data=''):self.data=data
        class Pub:
            def __init__(self,topic,*a,**kw):self.topic=topic;self.resolved_name=topic
            def get_num_connections(self):return 1
            def unregister(self):events.append(('unregister',self.topic))
            def publish(self,msg):
                events.append(('publish',self.topic))
                if self.topic=='/ping' and response=='fresh':
                    subscribers['/ping/ready'](String(data='ready_nonce='+msg.data+' receiver_boot_id=abc'))
                if self.topic=='/ping' and response=='stale':
                    subscribers['/ping/ready'](String(data='ready_nonce=old receiver_boot_id=abc'))
        def subscribe(topic,cls,cb,**kw):
            subscribers[topic]=cb
            return NS(unregister=lambda:events.append(('unregister',topic)))
        modules={'rospy':NS(Publisher=Pub,Subscriber=subscribe,get_name=lambda:'/test',is_shutdown=lambda:False),
            'rosgraph':NS(Master=lambda name:NS(getSystemState=lambda:([],[('/data',['/recorder'])],[]))),
            'std_msgs.msg':NS(String=String)}
        with patch.dict(sys.modules,modules),patch.object(arm.time,'monotonic',lambda:clock.t), \
                patch.object(arm.time,'sleep',lambda dt:setattr(clock,'t',clock.t+dt)):
            if response=='fresh':
                arm.prepare_arm(['/recorder'],timeout_s=.2,admission_ping_topic='/ping')
            else:
                with self.assertRaisesRegex(RuntimeError,'admission_ready=False'):
                    arm.prepare_arm(['/recorder'],timeout_s=.2,admission_ping_topic='/ping')
        self.assertLessEqual(clock.t,.2)
        self.assertNotIn(('publish',arm.ARM_TOPIC),events)
        self.assertIn(('unregister','/ping/ready'),events)

    def test_registered_topics_without_response_cannot_arm(self):self.run_barrier('none')
    def test_stale_response_cannot_arm(self):self.run_barrier('stale')
    def test_matching_roundtrip_can_arm_without_sending_goal(self):self.run_barrier('fresh')


if __name__=='__main__':unittest.main(verbosity=2)
