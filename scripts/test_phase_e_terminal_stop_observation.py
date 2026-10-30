"""Fault controls for read-only stopped-tail evidence and original score/watchdog."""
import ast
import argparse
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from phase_e_terminal_stop_observation import TerminalStopObservation, zero_hold_tail
from phase_e_evaluation_common import seconds_to_ns, quat_rot, pose_error


class ObservationTests(unittest.TestCase):
    def ready(self):
        s=TerminalStopObservation();s.commit(100,1100000000,True)
        s.tracker(1000000000,100,1100000000,'trajectory_complete_hold',True,False)
        return s

    def sample(self,s,t,inside=True,v=.01):
        for k in ['execution_vbc','regime','actuator']:
            s.evidence_update(k,t,True)
        s.tracker(t,0,0,'last_reference_hold',False,False)
        return s.sample(t,t,inside,v)

    def hold(self,s,start=1000000000):
        result=False
        for t in range(start,start+100000001,10000000):result=self.sample(s,t)
        return result

    def test_requires_full_100ms_after_complete(self):
        s=self.ready()
        for t in range(1000000000,1100000000,10000000):self.assertFalse(self.sample(s,t))
        self.assertTrue(self.sample(s,1100000000))
        self.assertEqual(s.snapshot()['hold_end_ros_ns']-s.snapshot()['hold_start_ros_ns'],100000000)

    def test_low_speed_before_complete_does_not_count(self):
        s=TerminalStopObservation();s.commit(100,1100000000,True)
        self.assertFalse(self.hold(s));self.assertFalse(s.snapshot()['tail_completed'])

    def test_delayed_measurement_and_lagging_receiver_clock_cannot_start_early(self):
        s=TerminalStopObservation();s.commit(100,1100000000,True)
        s.tracker(1000000000,100,1100000000,'trajectory_complete_hold',True,False,1005000000)
        self.assertFalse(self.sample(s,1000000000))
        self.assertEqual(s.reason,'measured_sample_precedes_tail_completion')
        self.assertFalse(self.sample(s,1010000000))
        self.assertEqual(s.hold_start_ns,1010000000)
        for t in range(1020000000,1110000000,10000000):self.assertFalse(self.sample(s,t))
        self.assertTrue(self.sample(s,1110000000))

    def test_pose_and_speed_reset(self):
        for inside,v in [(False,.01),(True,.051),(True,math.nan)]:
            s=self.ready();self.sample(s,1000000000);self.sample(s,1040000000,inside,v)
            self.assertFalse(self.sample(s,1080000000))
            self.assertEqual(s.snapshot()['hold_start_ros_ns'],1080000000)

    def test_execution_replacement_resets_completion(self):
        s=self.ready();self.sample(s,1000000000);s.commit(200,1100000000,True)
        self.assertFalse(self.hold(s,1010000000));self.assertFalse(s.tail_completed)

    def test_old_and_duplicate_commits_do_not_replace_current(self):
        s=self.ready();s.commit(99,1,False);s.commit(100,1,False)
        self.assertEqual(s.commit_stamp,100);self.assertTrue(s.tail_completed)

    def test_wrong_token_and_abort_fail_closed(self):
        for token,abort in [(99,False),(100,True)]:
            s=self.ready();s.tracker(1000000000,token,1100000000,'trajectory_complete_hold',True,abort)
            self.assertFalse(self.hold(s))

    def test_missing_or_stale_execution_audit_cannot_complete(self):
        s=self.ready()
        for t in range(1000000000,1200000001,10000000):
            s.tracker(t,0,0,'last_reference_hold',False,False)
            for k in ['regime','actuator']:s.evidence_update(k,t,True)
            self.assertFalse(s.sample(t,t,True,.01))
        self.assertEqual(s.reason,'missing_or_stale_execution_vbc')

    def test_invalid_evidence_and_clock_reversal_reset(self):
        for kind in ['tracker','execution_vbc','regime','actuator']:
            s=self.ready();self.sample(s,1000000000)
            s.evidence_update(kind,1010000000,False)
            self.assertIsNone(s.hold_start_ns)
        s=self.ready();self.sample(s,1000000000);s.evidence_update('regime',999999999,True)
        self.assertIsNone(s.hold_start_ns)

    def test_duplicate_out_of_order_or_large_gap_is_not_continuous(self):
        for t in [1000000000,999999999,1050000001]:
            s=self.ready();self.sample(s,1000000000)
            self.assertFalse(self.sample(s,t));self.assertIsNone(s.hold_start_ns)

    def test_stale_measured_state_is_not_retimestamped(self):
        s=self.ready();self.sample(s,1000000000)
        self.assertFalse(s.sample(1010000000,1060000001,True,.01))
        self.assertEqual(s.reason,'stale_measured_sample')

    def test_malformed_measured_sample_restarts_hold(self):
        s=self.ready();self.sample(s,1000000000);s.invalid_measured_sample()
        self.assertIsNone(s.hold_start_ns)
        self.assertFalse(self.sample(s,1010000000))

    def test_zero_tail_rejects_fake_zero_end_marker(self):
        def p(t,q=0,v=0):
            return SimpleNamespace(time_from_start=SimpleNamespace(to_nsec=lambda:t),positions=[q]*7,velocities=[v]*7)
        t=SimpleNamespace(joint_names=['j%d'%i for i in range(7)],points=[p(0,v=1),p(1000000000),p(1100000000)])
        self.assertTrue(zero_hold_tail(t,100000000))
        t.points[-2]=p(1000000000,v=.1);self.assertFalse(zero_hold_tail(t,100000000))
        t.points[-2]=p(1000000000,q=.01);self.assertFalse(zero_hold_tail(t,100000000))
        t.points[-2]=p(1000000001);self.assertFalse(zero_hold_tail(t,100000000))


class WatcherTests(unittest.TestCase):
    def run_watcher(self,samples):
        # Run production main with deterministic wall/ROS clocks and transport.
        import phase_e_terminal_stop_observation as observation
        tree=ast.parse((Path(__file__).parent/'wait_for_phase_d_goal.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        clock=SimpleNamespace(wall=0.,ros=100.);result={};names=['j%d'%i for i in range(7)];states=[]
        def stamp(t):return SimpleNamespace(to_nsec=lambda:round(t*1e9))
        def subscribe(ros,state):
            states.append(state);state.commit(100,1100000000,True);return []
        def wait(*a,**kw):
            w,t,x,v,complete=samples.pop(0) if samples else (clock.wall+.05,clock.ros+.01,0.,.01,True)
            clock.wall=w;clock.ros=t
            s=states[0]
            s.tracker(round(t*1e9),100,1100000000 if complete else 900000000,
                      'trajectory_complete_hold' if complete else 'active_trajectory',complete,False)
            for k in ['execution_vbc','regime','actuator']:s.evidence_update(k,round(t*1e9),True)
            return SimpleNamespace(name=names,position=[x]+[0]*6,velocity=[v]*7,header=SimpleNamespace(stamp=stamp(t)))
        def fk(h,c,q):T=np.eye(4);T[0,3]=q[names[0]];return T
        scope=dict(argparse=argparse,os=os,math=math,np=np,JOINTS=names,
            time=SimpleNamespace(monotonic=lambda:clock.wall,time=lambda:clock.wall+1000,sleep=lambda s:None),
            rospy=SimpleNamespace(init_node=lambda *a,**k:None,Time=SimpleNamespace(now=lambda:stamp(clock.ros)),
                is_shutdown=lambda:False,wait_for_message=wait,ROSException=RuntimeError),
            JointState=object,load_fk_helper=lambda _:SimpleNamespace(find_chain_joints=lambda *a:[]),
            URDF=SimpleNamespace(from_xml_file=lambda _:None),quat_rot=quat_rot,pose_error=pose_error,fk_T=fk,
            seconds_to_ns=seconds_to_ns,write_status=lambda p,d:result.update(d))
        exec(compile(ast.Module(body=[main],type_ignores=[]),'watcher','exec'),scope)
        argv=['watcher','--timeout-s','2','--settle-timeout-s','.01','--observe-terminal-stop',
              '--goal-position','0','0','0','--goal-orientation','0','0','0','1']
        with patch.object(sys,'argv',argv),patch.object(observation,'subscribe_terminal_observation',subscribe):scope['main']()
        return result

    def test_slow_ros_tail_hold_uses_remaining_budget_and_frozen_score(self):
        rows=[(.1,100.01,0,.5,False),(.3,100.11,0,.5,False),(.5,100.12,1,.1,False)]
        rows += [(1+i*.02,100.3+i*.01,0,.01,True) for i in range(11)]
        r=self.run_watcher(rows)
        self.assertTrue(r['goal_reached']);self.assertTrue(r['settled_before_exit'])
        self.assertEqual(r['goal_hold_end_ros_ns'],100110000000)
        self.assertGreater(r['elapsed_wall_s'],r['success_elapsed_wall_s']+.01)
        self.assertTrue(r['terminal_stop_observation']['completed'])

    def test_watchdog_does_not_wait_for_tail_or_create_late_success(self):
        r=self.run_watcher([(.1,100.01,0,.5,False),(.3,100.11,0,.5,False),(2.001,100.12,0,0,True)])
        self.assertTrue(r['goal_reached']);self.assertFalse(r['settled_before_exit'])
        self.assertFalse(r['terminal_stop_observation']['completed'])
        self.assertEqual(r['benchmark_window']['end_ros_ns'],100110000000)


if __name__=='__main__':unittest.main(verbosity=2)
