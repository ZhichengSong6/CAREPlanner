#!/usr/bin/env python3
"""Exercise production acquisition methods without ROS services or actuators."""
import ast
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
from typing import Dict, List
import unittest
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src/care_visibility_cdf/scripts'))
from observation_event_trace import observation_token, observation_region_token


class LineageTests(unittest.TestCase):
    def setUp(self):
        source = REPO / 'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py'
        names = {'_record_seen_query', '_traced_waypoint', '_query_region_seen',
                 '_update_actual_visibility_completion', '_set_acquisition_completion'}
        tree = ast.parse(source.read_text())
        methods = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
                   if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = dict(np=np, math=math, time=time, Dict=Dict, List=List,
            observation_token=observation_token, observation_region_token=observation_region_token,
            QueryConfidenceRequest=NS, Point=lambda **kw: NS(**kw), Bool=lambda **kw: NS(**kw),
            MultiArrayDimension=lambda **kw: NS(**kw),
            q_vis_joint_mask=lambda ob: np.asarray(
                ob.get('q_vis_joint_mask', np.ones(7)), dtype=float),
            _vector_msg=lambda q: NS(data=list(q), layout=NS(dim=[])),
            rospy=NS(Time=NS(now=lambda: NS(to_sec=lambda: 10.)),
                     logwarn=lambda *a: None, logwarn_throttle=lambda *a: None))
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                     str(source), 'exec'), scope)
        cls = type('Acquisition', (), {name: scope[name] for name in names})
        self.node = n = cls()
        n._schedule_publish_lock = threading.RLock()
        n._obligation_lock = threading.RLock()
        n._trace_published_target = n._trace_last_target = None
        n._trace_seen_samples = {}
        n._trace_measured = (np.ones(7), 9.95)
        n._c47_ready = True; n._acquisition_started = False
        n._last_visibility_check_s = -math.inf; n.visibility_check_rate = 20.
        n._seen_obligation_count = 0; n.seen_threshold = .5; n.required_seen_fraction = 1.
        self.events = []; n._trace_observation = lambda stage, **kw: self.events.append(dict(stage=stage, **kw))
        n.acquisition_complete_pub = NS(publish=lambda m: None)
        n._publish_acquisition_summary = lambda d: None
        n._publish_schedule = lambda: None
        n._ordered_obligations = lambda: list(n._obligations)
        n._obligation_geometry_diagnostics = lambda ob: {}
        self.ob = dict(id=1, generation_event_id='g', points=np.array([[1., 2., 3.]]), q_vis=np.zeros(7))
        n._obligations = [dict(self.ob)]
        n.confidence_client = lambda req: NS(confidence=[1.], inside_map=[True], current_visibility=[1.])

    def test_region_identity_ignores_pose_but_not_points_or_generation(self):
        token = observation_region_token(self.ob)
        self.assertEqual(token, observation_region_token(dict(self.ob, q_vis=np.ones(7))))
        self.assertEqual(token, observation_region_token(dict(
            self.ob, q_vis_joint_mask=[1, 1, 1, 1, 0, 0, 0])))
        self.assertNotEqual(
            observation_token(self.ob),
            observation_token(dict(
                self.ob, q_vis_joint_mask=[1, 1, 1, 1, 0, 0, 0])))
        for change in (dict(points=[[1., 2., 3.000000000001]]), dict(generation_event_id='new')):
            self.assertNotEqual(token, observation_region_token(dict(self.ob, **change)))

    def test_shared_target_queries_same_region_with_actual_published_q(self):
        n = self.node; shared = dict(self.ob, q_vis=np.ones(7))
        message = n._traced_waypoint(shared, shared['q_vis'])
        self.assertEqual(message.data, [1.] * 7)  # legacy interface unchanged
        self.assertEqual(message.layout.dim[0].label, observation_token(shared))
        self.assertEqual(message.layout.dim[1].label, 'care_qmask_v1_1111111')
        diag = dict(measured_q=np.ones(7), measured_ros_s=9.95, status='ok')
        n._record_seen_query(self.ob, n._trace_published_target, False, 0., diag)
        seen = [e for e in self.events if e['stage'] == 'seen_query']
        self.assertEqual(len(seen), 2)
        self.assertEqual([e['q_distance_inf'] for e in seen], [1., 0.])
        self.assertTrue(all(not e['actual_seen'] for e in seen))
        self.assertEqual(seen[1]['evidence_relation'], 'same_exact_region_published')
        self.assertEqual(seen[1]['causal_head_attribution'], 'unknown')

    def test_changed_published_target_during_query_is_not_attributed(self):
        n = self.node; shared = dict(self.ob, q_vis=np.ones(7))
        n._traced_waypoint(shared, shared['q_vis']); old = n._trace_published_target
        new = dict(shared, q_vis=np.ones(7) * 2)
        n._traced_waypoint(new, new['q_vis'])
        n._record_seen_query(self.ob, old, True, 1., dict(status='ok'))
        self.assertEqual(len([e for e in self.events if e['stage'] == 'seen_query']), 1)

    def test_different_region_never_inherits_seen(self):
        n = self.node; shared = dict(self.ob, points=[[4., 5., 6.]], q_vis=np.ones(7))
        n._traced_waypoint(shared, shared['q_vis'])
        n._record_seen_query(self.ob, n._trace_published_target, True, 1., dict(status='ok'))
        self.assertEqual(len([e for e in self.events if e['stage'] == 'seen_query']), 1)

    def test_stale_seen_response_cannot_delete_refreshed_region(self):
        n = self.node
        def confidence(req):
            n._obligations[0]['points'] = np.array([[4., 5., 6.]])
            return NS(confidence=[1.], inside_map=[True], current_visibility=[1.])
        n.confidence_client = confidence
        n._update_actual_visibility_completion()
        self.assertEqual(len(n._obligations), 1)
        self.assertFalse(n._acquisition_complete)
        self.assertEqual(n._seen_obligation_count, 0)
        self.assertTrue(any(e['stage'] == 'seen_stale_discarded' for e in self.events))

    def test_pose_change_does_not_invalidate_real_seen_of_same_region(self):
        n = self.node
        def confidence(req):
            n._obligations[0]['q_vis'] = np.ones(7)
            return NS(confidence=[1.], inside_map=[True], current_visibility=[1.])
        n.confidence_client = confidence
        n._update_actual_visibility_completion()
        self.assertEqual(n._obligations, [])
        self.assertTrue(n._acquisition_complete)

    def test_bad_confidence_cannot_clear_obligation(self):
        n = self.node
        for response in (NS(confidence=[math.nan], inside_map=[True], current_visibility=[1.]),
                         NS(confidence=[], inside_map=[True], current_visibility=[1.])):
            n._last_visibility_check_s = -math.inf
            n.confidence_client = lambda req: response
            n._update_actual_visibility_completion()
            self.assertEqual(len(n._obligations), 1)
            self.assertFalse(n._acquisition_complete)


if __name__ == '__main__':
    unittest.main(verbosity=2)
