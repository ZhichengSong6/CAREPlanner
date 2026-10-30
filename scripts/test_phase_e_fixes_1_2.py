#!/usr/bin/env python3
"""Offline regression of shared pose metrics, bounded windows, and counters."""
import json
import math
import ast
import argparse
import os
import sys
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import evaluate_phase_d_run as evaluator
from phase_e_evaluation_common import quat_rot, pose_error, window_rows, window_joint_samples, load_window, rejection_counts, seconds_to_ns, parse_stamp_ns


class EvaluationTests(unittest.TestCase):
    def test_so3_and_exact_goal(self):
        rng = np.random.RandomState(10)
        for q in [[0,0,math.sqrt(.5),math.sqrt(.5)], *rng.normal(size=(50,4))]:
            R = quat_rot(q)
            np.testing.assert_allclose(R.T@R, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R),1)
            T=np.eye(4); T[:3,:3]=R
            p,r=pose_error(T,np.zeros(3),R)
            self.assertEqual(p,0); self.assertLess(r,5e-8)

    def test_quaternion_rejects_bad_values(self):
        for q in [[0]*4,[0,0,0,math.nan],[0,0,0,math.inf]]:
            with self.assertRaises(ValueError): quat_rot(q)

    def test_nonfinite_pose_fails_closed(self):
        T=np.eye(4); T[0,0]=math.nan
        self.assertEqual(pose_error(T,np.zeros(3),np.eye(3)),(math.inf,math.inf))

    def test_counter_disjointness(self):
        self.assertEqual(rejection_counts(1,0,1,0)["commit_gate_rejection_count"],1)
        self.assertEqual(rejection_counts(67,0,69,0)["exact_vbc_unsafe_rejection_count"],2)
        self.assertEqual(rejection_counts(0,1,0,1)["post_gcdf_timeout_count"],0)
        with self.assertRaises(ValueError): rejection_counts(2,0,1,0)

    def test_window_state_carry_and_teardown(self):
        rows=[{"_t":1,"state":"REPAIR"},{"_t":3,"state":"NORMAL"},{"_t":100,"state":"PROBE"}]
        window={"start_ros_s":2,"end_ros_s":4}
        chosen=window_rows(rows,window,True)
        self.assertEqual([r["_t"] for r in chosen],[2,3])
        self.assertEqual(evaluator.regime_times(chosen,4),
                         {"normal_time_s":1,"repair_time_s":1,"probe_time_s":0})
        self.assertEqual(len(window_rows(rows,window)),1)
        self.assertEqual(rows[0]["_t"],1)

    def test_legacy_and_required_window(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(load_window(d))
            with self.assertRaises(ValueError): load_window(d,True)
            p=Path(d)/"goal_stop_status.json"
            p.write_text(json.dumps({"benchmark_window":{"valid":False}}))
            with self.assertRaises(ValueError): load_window(d)
            w={"valid":True,"start_ros_s":1,"end_ros_s":2,"end_measured_q":[0]*7}
            p.write_text(json.dumps({"benchmark_window":w}))
            self.assertEqual(load_window(d,True),w)

    def _goal(self, samples):
        def fake_fk(helper,chain,qmap):
            T=np.eye(4); T[0,3]=qmap["q"]; return T
        with patch.object(evaluator,"fk_T",fake_fk):
            return evaluator.goal_eval(samples,None,None,["q"],
                {"goal_position":[0,0,0],"goal_orientation":[0,0,0,1]},.02,.2,.1)

    def test_late_success_is_excluded(self):
        rows=[{"_t":1,"q":[1]},{"_t":2,"q":[1]},
              {"_t":10,"q":[0]},{"_t":10.2,"q":[0]}]
        w={"start_ros_s":1,"end_ros_s":2}
        self.assertTrue(self._goal([(r["_t"],r["q"]) for r in rows])["task_success"])
        result=self._goal([(r["_t"],r["q"]) for r in window_rows(rows,w)])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["best_position_error_m"],1)

    def test_success_then_teardown_drift(self):
        rows=[{"_t":1,"q":[0]},{"_t":1.2,"q":[0]},{"_t":10,"q":[1]}]
        result=self._goal([(r["_t"],r["q"]) for r in window_rows(rows,{"start_ros_s":1,"end_ros_s":1.2})])
        self.assertTrue(result["task_success"])
        self.assertEqual(result["final_position_error_m"],0)

    def test_recorder_missing_boundary_frames(self):
        w={"start_ros_s":1,"end_ros_s":1.2,"first_sample_ros_s":1,
           "first_measured_q":[0]*7,"end_measured_q":[0]*7}
        samples=window_joint_samples([(0,[1]*7),(1.1,[0]*7),(10,[1]*7)],w)
        self.assertEqual([s[0] for s in samples],[1,1.1,1.2])
        self.assertTrue(self._goal([(t,[q[0]]) for t,q in samples])["task_success"])

    def _watcher(self, samples, arm=False):
        # Execute the real watcher main, replacing only ROS/time/URDF IO.
        tree=ast.parse((Path(__file__).parent/"wait_for_phase_d_goal.py").read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        clock=SimpleNamespace(t=0.0)
        names=["j%d"%i for i in range(7)]
        result={}
        arm_calls=[]
        def stamp(t): return SimpleNamespace(to_sec=lambda: t, to_nsec=lambda: seconds_to_ns(t))
        def wait(*args,**kwargs):
            t,x=samples.pop(0) if samples else (clock.t+.2,1.0)
            clock.t=t
            return SimpleNamespace(name=names, position=[x]+[0]*6, velocity=[0]*7,
                                   header=SimpleNamespace(stamp=stamp(100+t)))
        def fk(helper,chain,qmap):
            T=np.eye(4); T[0,3]=qmap[names[0]]; return T
        helper=SimpleNamespace(find_chain_joints=lambda *a: [])
        scope={"argparse":argparse,"os":os,"math":math,"np":np,"JOINTS":names,
               "time":SimpleNamespace(monotonic=lambda:clock.t,time=lambda:1000+clock.t,
                                      sleep=lambda duration:None),
               "rospy":SimpleNamespace(init_node=lambda *a,**kw:None,
                   Time=SimpleNamespace(now=lambda:stamp(0 if clock.t==0 else 100+clock.t)),
                   is_shutdown=lambda:False,wait_for_message=wait,ROSException=RuntimeError),
               "JointState":object,"load_fk_helper":lambda repo:helper,
               "URDF":SimpleNamespace(from_xml_file=lambda path:None),
               "quat_rot":quat_rot,"pose_error":pose_error,"fk_T":fk,"seconds_to_ns":seconds_to_ns,
               "write_status":lambda path,payload:result.update(payload)}
        exec(compile(ast.Module(body=[main],type_ignores=[]),"watcher_main","exec"),scope)
        argv=["watcher","--timeout-s","1","--goal-position","0","0","0",
              "--goal-orientation","0","0","0","1"]
        if arm: argv += ["--arm-token", "run_nonce"]
        arm_module=SimpleNamespace(prepare_arm=lambda recorders: SimpleNamespace(
            publish=lambda token:arm_calls.append((token,clock.t))))
        with patch.object(sys,"argv",argv), patch.dict(sys.modules,{"phase_e_benchmark_arm":arm_module}):
            scope["main"]()
        result['_arm_calls']=arm_calls
        return result

    def test_watcher_arms_only_after_finite_measured_q(self):
        result=self._watcher([(.1,math.nan),(.2,1),(.3,1)],arm=True)
        self.assertEqual(result['_arm_calls'],[('run_nonce',.2)])
        self.assertEqual(result['benchmark_window']['first_measured_q'],[1]+[0]*6)
        self.assertEqual(result['benchmark_window']['watchdog_start_monotonic_s'],.2)
        self.assertEqual(result['benchmark_window']['protocol'],'pre_goal_arm_v1')

    def test_watcher_freezes_window_at_success_before_settle(self):
        result=self._watcher([(.1,0),(.3,0),(.4,1)])
        self.assertTrue(result["goal_reached"])
        self.assertAlmostEqual(result["benchmark_window"]["end_ros_s"],100.3)
        self.assertEqual(result["benchmark_window"]["end_measured_q"],[0]*7)

    def test_watcher_cannot_accept_message_after_watchdog(self):
        result=self._watcher([(.2,1),(.8,1),(1.1,0)])
        self.assertFalse(result["goal_reached"])
        self.assertAlmostEqual(result["benchmark_window"]["end_ros_s"],100.8)
        self.assertAlmostEqual(result["benchmark_window"]["start_ros_s"],100.2)
        self.assertEqual(result["benchmark_window"]["watchdog_start_monotonic_s"],0)

    def test_watcher_latched_success_survives_watchdog_before_settle(self):
        result=self._watcher([(.7,0),(.9,0),(1.1,0)])
        self.assertTrue(result["goal_reached"])
        self.assertFalse(result["settled_before_exit"])
        self.assertAlmostEqual(result["benchmark_window"]["end_ros_s"],100.9)

    def test_integer_hold_boundary_and_large_epoch(self):
        for origin in (15351000000, 1789092000123456789):
            for delta in (99999999, 100000000, 100000001):
                with patch.object(evaluator, 'fk_T', lambda *a: np.eye(4)):
                    result = evaluator.goal_eval([(origin,[0]), (origin+delta,[0])],
                        None, None, ['q'], {'goal_position':[0,0,0], 'goal_orientation':[0,0,0,1]},
                        .02, .2, .1, timestamps_ns=True)
                self.assertEqual(result['task_success'], delta >= 100000000)
                self.assertEqual(result['required_hold_ns'], 100000000)

    def test_exact_ns_window_excludes_one_ns_outside(self):
        start = 1789092000123456789; end = start + 100000000
        window = dict(start_ros_s=start/1e9, end_ros_s=end/1e9,
                      start_ros_ns=start, end_ros_ns=end, end_measured_q=[0]*7)
        samples = [(start-1,[1]*7), (start,[0]*7), (end,[0]*7), (end+1,[1]*7)]
        self.assertEqual([t for t,q in window_joint_samples(samples,window,True)], [start,end])

    def test_timestamp_parsing_and_duration_never_shortened(self):
        self.assertEqual(parse_stamp_ns('1789092000123456789'),1789092000123456789)
        self.assertEqual(parse_stamp_ns('1.5351e10'),15351000000)
        self.assertEqual(seconds_to_ns('0.1000000001',duration=True),100000001)
        for value in ('NaN','Infinity','-1','0.1'):
            with self.assertRaises(ValueError): parse_stamp_ns(value)
        for value in ('NaN','Infinity','-1'):
            with self.assertRaises(ValueError): seconds_to_ns(value,duration=True)

    def test_csv_keeps_integer_stamp_beyond_float_precision(self):
        origin = 1789092000123456789
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'joints.csv'
            path.write_text('field.header.stamp,field.name0,field.position0\n'
                            + str(origin)+',q,0\n'+str(origin+99999999)+',q,0\n')
            actual = evaluator.joint_states(str(path), ['q'], timestamps_ns=True)
            self.assertEqual([t for t,q in actual], [origin, origin+99999999])
            self.assertIsInstance(evaluator.joint_states(str(path), ['q'])[0][0], float)

    def test_watcher_exact_100ms_matches_evaluator(self):
        result=self._watcher([(.1,0),(.2,0),(.3,1)])
        self.assertTrue(result['goal_reached'])
        self.assertEqual(result['goal_hold_end_ros_ns']-result['goal_hold_start_ros_ns'],100000000)
        self.assertEqual(result['benchmark_window']['schema_version'],3)

    def test_watcher_one_ns_short_is_not_success(self):
        result=self._watcher([(.1,0),(.199999999,0),(.2,1)])
        self.assertFalse(result['goal_reached'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
