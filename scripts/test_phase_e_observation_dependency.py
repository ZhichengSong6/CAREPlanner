#!/usr/bin/env python3
"""Run production scheduler methods without ROS/model startup.

These test scheduling/identity, not execution safety or CASE001 success.
"""
import json
import ast
import math
import threading
from types import SimpleNamespace as NS
from typing import Dict
import unittest

import numpy as np
import test_phase_e_path_target_lock as lock_tests
from test_phase_e_path_target_lock import (
    obligation, observation_region_token, observation_token, REPO)


class ObservationDependencyTests(unittest.TestCase):
    def node(self):
        node = lock_tests.PathTargetLockTests()._node([
            '_observation_dependency_cb', '_process_observation_dependency',
            '_process_observation_dependency_event',
            '_process_pending_refinement',
            '_prune_or_initialize_stack', '_ordered_obligations', '_consider_active_layer'])
        # Match the production base classes, not permissive reentrant doubles.
        node._obligation_lock = threading.Lock()
        node._lock = threading.Lock()
        node._schedule_publish_lock = threading.RLock()
        node._dependency_lock = threading.Lock()
        node._dependency_pending = None
        node._dependency_latest = (-1, -1, -1)
        node._dependency_attempts = {}
        node._dependency_pins = {}
        node._dependency_edges = {}
        node._dependency_reason = 'startup'
        node._dependency_pushes = node._dependency_resumes = 0
        node.max_obligations = 12
        node._schedule_new_obligations = 0
        node._preferred_trajectory_locked = lambda: (NS(), NS(), 'unit')
        node._publish_schedule = lambda: None
        node._publish_blocker_stack_summary = lambda: None
        node._progressive_priority_order = lambda by_id, active: [active]+[i for i in by_id if i!=active]
        node._shared_path_candidate_ids = lambda by_id, active: set(by_id)
        node._compute_progressive_shared_target = lambda *args: self.fail('pinned target rewritten')
        node._cluster_regions = lambda pts: [dict(points=pts)]
        node._generate_new_obligation = lambda *args: obligation(4)
        self.publish(node, 1)
        return node

    def publish(self, node, oid, q=None):
        ob = next(ob for ob in node._obligations if ob['id']==oid)
        node._trace_published_target = dict(observation_token=observation_token(ob),
            region_token=observation_region_token(ob),
            q_vis=ob['q_vis'].copy() if q is None else np.asarray(q).copy())

    def event(self, node, points=None, seq=1, **kwargs):
        data = dict(version=1, mode_epoch=1, plan_seq=seq, query_stamp_ns=seq,
            query_ros_s=1., observation_token=node._trace_published_target['observation_token'],
            points=points if points is not None else [[2., 0., .5]])
        data.update(kwargs)
        node._observation_dependency_cb(NS(data=json.dumps(data)))

    def test_push_and_observed_return_preserves_actual_published_parent_target(self):
        n = self.node()
        original = np.arange(7)+10.  # published shared pose != stored individual pose
        self.publish(n, 1, original)
        self.event(n); n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1, 2])
        self.assertEqual(n._dependency_edges, {2: 1})
        n._prune_or_initialize_stack()  # motion/more bundles cannot pop unseen child
        self.assertEqual(n._repair_stack, [1, 2])
        np.testing.assert_equal(n._ordered_obligations()[0]['q_vis'], np.full(7, 2.))
        n._obligations = [ob for ob in n._obligations if ob['id'] != 2]  # actual observation path
        n._prune_or_initialize_stack()
        self.assertEqual(n._repair_stack, [1])
        self.assertEqual(n._dependency_resumes, 1)
        np.testing.assert_equal(n._ordered_obligations()[0]['q_vis'], original)
        self.assertEqual({ob['id'] for ob in n._obligations}, {1, 3})

    def test_stale_token_does_not_preempt(self):
        n = self.node(); self.event(n, observation_token='care_obs_v1_old')
        n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1])
        self.assertFalse(n._dependency_attempts)

    def test_old_malformed_and_duplicate_messages_are_rejected(self):
        n = self.node()
        for kwargs in [dict(query_ros_s=0.), dict(query_ros_s=2.),
                       dict(points=[[float('nan'), 0., .5]]), dict(points=[]),
                       dict(points=[[1., 2.]]), dict(version=2)]:
            self.event(n, **kwargs)
            self.assertIsNone(n._dependency_pending)
        n._observation_dependency_cb(NS(data='{"version":1,"version":1}'))
        self.assertIsNone(n._dependency_pending)
        self.event(n); n._dependency_pending = None
        self.event(n)
        self.assertIsNone(n._dependency_pending)

    def test_self_and_ancestor_cannot_create_cycle_or_reset_target(self):
        n = self.node(); self.event(n, points=[[1., 0., .5]])
        n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1])
        self.event(n, seq=2); n._process_observation_dependency()
        self.publish(n, 2)
        self.event(n, points=[[1., 0., .5]], seq=3); n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1, 2])
        n._consider_active_layer([1], [], .1, urgent_gcdf_recovery=True)
        self.assertEqual(n._repair_stack, [1, 2])
        self.assertEqual(n._path_association_reason, 'observation_dependency_cycle_hold')

    def test_nested_dependencies_resume_in_order(self):
        n = self.node(); self.event(n); n._process_observation_dependency()
        self.publish(n, 2)
        self.event(n, points=[[3., 0., .5]], seq=2); n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1, 2, 3])
        n._obligations = [ob for ob in n._obligations if ob['id'] != 3]
        n._prune_or_initialize_stack()
        self.assertEqual(n._repair_stack, [1, 2])
        n._obligations = [ob for ob in n._obligations if ob['id'] != 2]
        n._prune_or_initialize_stack()
        self.assertEqual(n._repair_stack, [1])

    def test_unowned_unknown_generates_child_not_new_parent_pose(self):
        n = self.node(); self.event(n, points=[[4., 0., .5]])
        n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1, 4])
        self.assertEqual(n._schedule_new_obligations, 1)
        np.testing.assert_equal(n._obligations[0]['q_vis'], np.ones(7))

    def test_generation_rechecks_parent_and_does_not_append_stale_result(self):
        n = self.node()
        def generate(*args):
            n._trace_published_target['observation_token'] = 'care_obs_v1_changed'
            return obligation(4)
        n._generate_new_obligation = generate
        self.event(n, points=[[4., 0., .5]]); n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1])
        self.assertEqual(len(n._obligations), 3)
        self.assertEqual(n._dependency_reason, 'dependency_changed_during_generation')

    def test_failed_generation_budget_survives_motion_and_token_change(self):
        n = self.node(); calls = []
        def generate(*args):
            calls.append(1)
            raise ValueError('no candidate')
        n._generate_new_obligation = generate
        for seq in range(1, 6):
            n._latest_measured_q += .02
            n._trace_published_target['observation_token'] = 'care_obs_v1_'+str(seq)
            self.event(n, points=[[4., 0., .5]], seq=seq)
            n._process_observation_dependency()
        self.assertEqual(len(calls), 2)
        self.assertEqual(n._dependency_reason, 'dependency_budget_exhausted')
        self.assertEqual(n._repair_stack, [1])

    def test_invalid_generated_pose_never_enters_obligation_list(self):
        n = self.node()
        child = obligation(4); child['q_vis'][0] = np.nan
        n._generate_new_obligation = lambda *args: child
        self.event(n, points=[[4., 0., .5]]); n._process_observation_dependency()
        self.assertEqual(len(n._obligations), 3)
        self.assertEqual(n._repair_stack, [1])

    def test_sensing_queue_has_one_generation_owner(self):
        n = self.node(); self.event(n)
        n._dependency_processing = True
        n._process_observation_dependency()
        self.assertIsNotNone(n._dependency_pending)
        self.assertEqual(n._repair_stack, [1])
        n._dependency_processing = False
        n._process_observation_dependency()
        self.assertEqual(n._repair_stack, [1, 2])

    def test_legacy_cycle_refinement_cannot_replace_unseen_dependency(self):
        n = self.node(); self.event(n); n._process_observation_dependency()
        n._adaptive_refinement_lock = threading.Lock()
        n._pending_refinement_ids = [1, 2]
        n._adaptive_refinement_skip_count = 0
        self.assertFalse(n._process_pending_refinement(None, None, 'test', .1))
        self.assertEqual(n._repair_stack, [1, 2])
        self.assertEqual(n._adaptive_refinement_last_reason, 'observation_dependency_in_flight')


class ActualObservationTests(unittest.TestCase):
    def test_bytes_array_does_not_mark_outside_points_seen(self):
        path = REPO/'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py'
        method = next(m for c in ast.parse(path.read_text()).body if isinstance(c, ast.ClassDef)
                      for m in c.body if isinstance(m, ast.FunctionDef) and m.name=='_query_region_seen')
        scope = dict(np=np, math=math, Dict=Dict, QueryConfidenceRequest=NS, Point=NS,
                     rospy=NS(logwarn_throttle=lambda *args: None))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     str(path), 'exec'), scope)
        node = NS(seen_threshold=.8, required_seen_fraction=1.)
        ob = dict(points=[[1.,0.,0.],[2.,0.,0.]])
        for flags in [bytes([1,0]), bytearray([1,0]), [1,0], (True,False)]:
            node.confidence_client = lambda req: NS(confidence=[1.,1.],
                inside_map=flags, current_visibility=[1.,1.])
            result = scope['_query_region_seen'](node, ob)
            self.assertFalse(result[0])
            self.assertEqual(result[1], .5)
            self.assertEqual(result[-1]['outside_count'], 1)
        node.confidence_client = lambda req: NS(confidence=[1.,1.],
            inside_map=bytes([1,1]), current_visibility=[1.,1.])
        self.assertTrue(scope['_query_region_seen'](node, ob)[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
