#!/usr/bin/env python3
"""Real production publication/completion methods, actual Lock/RLock types.

Only ROS transport, model proposals and auxiliary summary/frontier generation
are fixtures. Publication, ordering, token history and observation completion
are NOT stubbed. Every scenario has a bounded worker timeout.
"""
import ast
from collections import OrderedDict
import copy
import json
import math
import threading
import time
from types import SimpleNamespace as NS
from typing import Dict, List
import unittest

import numpy as np
import test_phase_e_observation_dependency as dependency_tests
from test_phase_e_path_target_lock import REPO, observation_region_token, observation_token, region_key


def production_class(path, names, base, scope):
    cls = next(c for c in ast.parse(path.read_text()).body if isinstance(c, ast.ClassDef))
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in cls.body} == set(names)
    cls.bases = [ast.Name(id='Parent', ctx=ast.Load())]
    scope = dict(scope, Parent=base)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(path), 'exec'), scope)
    return scope[cls.name]


class PublishChainTests(unittest.TestCase):
    def node(self):
        fixture = dependency_tests.ObservationDependencyTests()
        old = fixture.node()
        clock = NS(now=1.)
        vector = lambda q: NS(data=list(q), layout=NS(dim=[]))
        scope = dict(np=np, math=math, time=time, json=json, threading=threading,
            Dict=Dict, List=List, observation_token=observation_token,
            observation_region_token=observation_region_token, region_key=region_key,
            Float64=NS, Bool=NS, Float64MultiArray=lambda: vector([]),
            MultiArrayDimension=NS, QueryConfidenceRequest=NS, Point=NS, _vector_msg=vector,
            q_vis_joint_mask=lambda ob: np.asarray(
                ob.get('q_vis_joint_mask', np.ones(7)), dtype=float),
            rospy=NS(Time=NS(now=lambda: NS(to_sec=lambda: clock.now)),
                     logwarn=lambda *args: None, logwarn_throttle=lambda *args: None))
        base = production_class(REPO/'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py',
            ['_publish_schedule', '_traced_waypoint', '_update_actual_visibility_completion',
             '_query_region_seen', '_record_seen_query'], type(old), scope)
        cls = production_class(REPO/'src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py',
            ['_publish_schedule', '_traced_waypoint', '_update_actual_visibility_completion'], base, scope)
        n = cls(); n.__dict__.update(old.__dict__)
        del n._publish_schedule  # use both actual class overrides, including super()
        n._obligation_lock = threading.Lock()  # exactly the production base lock
        n._lock = threading.Lock()
        n._schedule_publish_lock = threading.RLock()  # intentionally reentrant in production
        n._c47_ready = n._c49_ready = True
        n._trace_last_target = None
        n._trace_seen_samples = {}
        n._trace_measured = (np.zeros(7), 1.)
        n._final_recovery_target_history = OrderedDict()
        n._final_recovery_target_history_capacity = 64
        n._current_obligation_points_key = None
        n._current_obligation_points_seq = 0
        n._coherent_bundle_lock = threading.Lock()
        n._coherent_bundle_enabled = True
        n._coherent_bundle_pending_nonempty = False
        n._last_visibility_check_s = -math.inf
        n.visibility_check_rate = 20.
        n.seen_threshold = .8
        n.required_seen_fraction = 1.
        n._seen_obligation_count = 0
        n._acquisition_started = False
        n._obligation_geometry_diagnostics = lambda ob: {}
        n._publish_acquisition_summary = lambda diag: None
        n._publish_visibility_frontier = lambda: None  # no learned-field inference
        n.seen_ids = set()
        n.confidence_client = lambda req: NS(
            confidence=[float(int(p.x) in n.seen_ids) for p in req.points],
            inside_map=bytes([1]*len(req.points)), current_visibility=[1.]*len(req.points))
        n.messages = []; n.trace_events = []
        n._trace_observation = lambda stage, **kw: n.trace_events.append(dict(stage=stage, **kw))
        def emit(topic, msg):
            # Publication must not occur under either non-reentrant state lock.
            for lock in [n._obligation_lock, n._lock]:
                self.assertTrue(lock.acquire(blocking=False), 'state lock held during publication')
                lock.release()
            n.messages.append((topic, copy.deepcopy(msg)))
        for attr, topic in [('waypoint_pub','q'), ('zero_pub','zero'), ('deadline_pub','deadline'),
                            ('current_obligation_points_pub','points'), ('acquisition_complete_pub','complete')]:
            setattr(n, attr, NS(publish=lambda msg, topic=topic: emit(topic,msg)))
        return n, fixture, clock

    def bounded(self, work):
        errors = []
        def run():
            try: work()
            except BaseException as exc: errors.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        worker.start(); worker.join(2.)
        self.assertFalse(worker.is_alive(), 'production publish/completion chain deadlocked')
        if errors: raise errors[0]

    def test_existing_child_publishes_coherent_target_and_history(self):
        n, fixture, _ = self.node()
        def work():
            fixture.event(n); n._process_observation_dependency()
            self.assertEqual(n._repair_stack, [1,2])
            self.assertFalse(n._dependency_processing)
            self.assertEqual([t for t,m in n.messages], ['q','zero','deadline','points'])
            self.assertEqual(n.messages[0][1].data, [2.]*7)
            token = n.messages[0][1].layout.dim[0].label
            self.assertEqual(token, n._trace_published_target['observation_token'])
            self.assertIn(token, n._final_recovery_target_history)
            self.assertEqual(n.messages[-1][1].data[1:], [2.,1.,2.,0.,.5])
            n._publish_schedule()  # second publication also finishes; point sequence deduplicates
            self.assertEqual(n._current_obligation_points_seq, 1)
        self.bounded(work)

    def test_generated_child_uses_same_real_publication_chain(self):
        n, fixture, _ = self.node()
        def work():
            fixture.event(n, points=[[4.,0.,.5]])
            n._process_observation_dependency()
            self.assertEqual(n._repair_stack,[1,4])
            self.assertEqual(n.messages[0][1].data,[4.]*7)
            self.assertEqual(n.messages[-1][1].data[1],4.)
        self.bounded(work)

    def test_actual_seen_child_resumes_original_published_parent(self):
        n, fixture, clock = self.node()
        original = np.arange(7)+10.
        fixture.publish(n,1,original)
        def work():
            fixture.event(n); n._process_observation_dependency()
            n._update_actual_visibility_completion()  # all confidence still zero
            self.assertEqual(n._repair_stack,[1,2])
            n.seen_ids.add(2); clock.now += .1
            n._update_actual_visibility_completion()  # production confidence query/removal/pop/publish
            self.assertEqual(n._repair_stack,[1])
            self.assertEqual(n._dependency_resumes,1)
            self.assertEqual(n._seen_obligation_count,1)
            np.testing.assert_equal(n._trace_published_target['q_vis'],original)
            self.assertEqual([m.data[1] for t,m in n.messages if t=='points'],[2.,1.])
        self.bounded(work)

    def test_concurrent_schedule_publisher_cannot_interleave_transaction(self):
        n, fixture, _ = self.node()
        entered=threading.Event(); release=threading.Event(); started=threading.Event(); done=threading.Event()
        errors=[]
        emit=n.waypoint_pub.publish
        def pause(msg):
            emit(msg)
            if not entered.is_set():
                entered.set()
                if not release.wait(1.): raise AssertionError('test release timeout')
        n.waypoint_pub.publish=pause
        def competitor():
            started.set()
            try: n._publish_schedule()
            except BaseException as exc: errors.append(exc)
            finally: done.set()
        def work():
            fixture.event(n)
            def observer():
                if not entered.wait(1.):
                    errors.append(AssertionError('publication never reached')); release.set(); return
                other=threading.Thread(target=competitor,daemon=True); other.start()
                started.wait(1.)
                if done.wait(.05): errors.append(AssertionError('publication transaction interleaved'))
                release.set(); other.join(1.)
                if other.is_alive(): errors.append(AssertionError('competing publisher deadlocked'))
            observer_thread=threading.Thread(target=observer,daemon=True); observer_thread.start()
            n._process_observation_dependency()
            observer_thread.join(1.)
            self.assertFalse(observer_thread.is_alive())
            if errors: raise errors[0]
            self.assertTrue(done.is_set())
            self.assertEqual([t for t,m in n.messages], ['q','zero','deadline','points','q','zero','deadline'])
        try: self.bounded(work)
        finally: release.set()

    def test_publication_exception_releases_locks_and_processing_owner(self):
        n, fixture, _ = self.node()
        def work():
            emit=n.waypoint_pub.publish
            def fail(msg): raise RuntimeError('fixture transport error')
            n.waypoint_pub.publish=fail
            fixture.event(n)
            with self.assertRaisesRegex(RuntimeError,'fixture transport error'):
                n._process_observation_dependency()
            self.assertFalse(n._dependency_processing)
            self.assertTrue(n._obligation_lock.acquire(blocking=False)); n._obligation_lock.release()
            n.waypoint_pub.publish=emit
            n._publish_schedule()
            self.assertEqual(n.messages[0][1].data,[2.]*7)
        self.bounded(work)


if __name__ == '__main__':
    unittest.main(verbosity=2)
