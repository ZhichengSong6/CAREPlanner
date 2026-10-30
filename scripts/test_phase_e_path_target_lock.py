#!/usr/bin/env python3
"""Unit checks for blocker-aware q_vis target persistence."""

import ast
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
from typing import List
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/care_visibility_cdf/scripts"))
from observation_identity import observation_region_token, observation_token
from observation_identity import observation_target_state, restore_observation_target_state
from bounded_visibility_recovery import region_key


class _FakeTorchTensor:
    """Small tensor shim for the bounded-frontier unit test."""
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float64)

    def __getitem__(self, index):
        return _FakeTorchTensor(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeTorch:
    float32 = object()

    @staticmethod
    def tensor(value, **_kwargs):
        return _FakeTorchTensor(value)


def production_methods(names):
    source = REPO / "src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py"
    tree = ast.parse(source.read_text())
    nodes = [node for cls in tree.body if isinstance(cls, ast.ClassDef)
             for node in cls.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(
        math=__import__("math"), json=__import__("json"), np=np, threading=threading, List=List,
        torch=_FakeTorch(),
        observation_region_token=observation_region_token,
        observation_token=observation_token,
        observation_target_state=observation_target_state,
        restore_observation_target_state=restore_observation_target_state,
        region_key=region_key,
        rospy=NS(
            logwarn=lambda *args: None,
            logwarn_throttle=lambda *args: None,
            Time=NS(now=lambda: NS(to_sec=lambda: 1.0))),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(source), "exec"), scope)
    return type("Subject", (), {name: scope[name] for name in names})()


def obligation(oid):
    return dict(
        id=oid,
        points=np.asarray([[float(oid), 0.0, 0.5]]),
        q_vis=np.full(7, float(oid)),
        q_zero=np.zeros(7),
        keys=((oid, 0, 0),),
        deadline_abs_s=float(oid),
        final_f_min=0.1,
    )


class PathTargetLockTests(unittest.TestCase):
    def _node(self, names):
        node = production_methods(names)
        node._obligation_lock = threading.RLock()
        node._latest_measured_q = np.zeros(7)
        node._repair_stack = [1]
        node._obligations = [obligation(i) for i in (1, 2, 3)]
        node._path_co_plan_ids = []
        node._path_associated_ids = []
        node._path_association_reason = "startup"
        node._path_associated_count = 0
        node._path_queued_count = 0
        node._last_active_layer_ids = []
        node._last_active_layer_sweep_s = np.nan
        node._stack_pop_count = 0
        node._stack_push_count = 0
        node._pending_blocker_id = None
        node._pending_blocker_count = 0
        node._last_switch_reason = "startup"
        node.blocker_push_max_sweep_s = 0.30
        node.blocker_confirmations = 2
        node.progressive_shared_path_association_only = True
        node.progressive_shared_repair_enabled = True
        node.progressive_shared_max_regions = 3
        node._progressive_shared_last_mode = "startup"
        node._progressive_shared_last_considered_ids = []
        node._progressive_shared_last_kept_ids = []
        node._progressive_shared_last_dropped_ids = []
        node._progressive_shared_last_slacks = {}
        node._progressive_shared_cache_key = None
        node._progressive_shared_cache = None
        node._progressive_shared_lock = threading.RLock()
        node._progressive_shared_accept_f_min = 0.0
        node._active_qvis_target_lock_enabled = True
        node._safety_union_merge_count = 0
        node._cell_key = lambda p: tuple(
            np.rint(np.asarray(p, dtype=np.float64) / 0.05).astype(int))
        node._select_nearest_qvis_id = lambda ids: min(ids)
        node._request_cycle_refinement = lambda *args: False
        node._trace_observation = lambda *args, **kwargs: None
        node.rospy = NS(
            logwarn=lambda *args: None,
            logwarn_throttle=lambda *args: None,
            Time=NS(now=lambda: NS(to_sec=lambda: 1.0)))
        return node

    def test_non_path_region_stays_queued_and_cannot_enter_shared_solve(self):
        node = self._node({"_shared_path_candidate_ids",
                           "_progressive_priority_order",
                           "_ordered_obligations"})
        calls = []
        node._compute_progressive_shared_target = (
            lambda by_id, active, order: calls.append(list(order)) or None)
        ordered = node._ordered_obligations()
        self.assertEqual(ordered[0]["id"], 1)
        self.assertEqual(calls, [[1]])

    def test_confirmed_path_blocker_is_the_only_new_shared_member(self):
        node = self._node({"_shared_path_candidate_ids",
                           "_progressive_priority_order",
                           "_ordered_obligations",
                           "_prune_or_initialize_stack",
                           "_consider_active_layer"})
        # A later layer is queued and cannot change the active target.
        node._consider_active_layer([1, 2], [2], 0.50)
        self.assertEqual(node._repair_stack, [1])
        self.assertEqual(node._path_co_plan_ids, [])
        self.assertEqual(node._path_association_reason, "queued_non_path_layer")

        # The same near-term path blocker must be confirmed before it can
        # preempt/co-plan with the current target.
        node._consider_active_layer([1, 2], [2], 0.10)
        self.assertEqual(node._repair_stack, [1])
        node._consider_active_layer([1, 2], [], 0.10)
        self.assertEqual(node._repair_stack, [1, 2])
        self.assertEqual(node._path_co_plan_ids, [1, 2])

        calls = []
        node._compute_progressive_shared_target = (
            lambda by_id, active, order: calls.append(list(order)) or None)
        node._ordered_obligations()
        self.assertEqual(calls, [[2, 1]])

    def test_gcdf_recovery_blocker_preempts_without_second_confirmation(self):
        node = self._node({"_prune_or_initialize_stack",
                           "_consider_active_layer"})
        # The rejected candidate's blocker is at 0.35 s, just outside the
        # ordinary VBC earliest-layer window.  GCDF recovery is already hard
        # evidence, so it must immediately become the active target.
        node._consider_active_layer([1, 2], [2], 0.35,
                                    urgent_gcdf_recovery=True)
        self.assertEqual(node._repair_stack, [1, 2])
        self.assertEqual(node._path_co_plan_ids, [1, 2])
        self.assertEqual(node._path_association_reason,
                         "gcdf_recovery_blocker_pushed")

    def test_matched_vbc_points_form_monotone_safety_union(self):
        node = self._node({"_update_matched_geometry_diagnostics"})
        matched = obligation(1)
        matched["points"] = np.asarray([
            [0.10, 0.00, 0.50],
            [0.15, 0.00, 0.50],
        ])
        matched["keys"] = tuple(node._cell_key(p) for p in matched["points"])
        matched["centroid"] = np.mean(matched["points"], axis=0)
        region = {
            "points": np.asarray([
                [0.15, 0.00, 0.50],
                [0.20, 0.00, 0.50],
            ]),
            "keys": tuple(node._cell_key(p) for p in [
                [0.15, 0.00, 0.50], [0.20, 0.00, 0.50]]),
            "centroid": np.asarray([0.175, 0.00, 0.50]),
        }
        original_qvis = matched["q_vis"].copy()
        node._update_matched_geometry_diagnostics(
            matched, region, "candidate_vbc_active_set")
        self.assertEqual(matched["points"].shape, (3, 3))
        self.assertEqual(len(matched["keys"]), 3)
        self.assertTrue(np.array_equal(matched["q_vis"], original_qvis))
        self.assertEqual(node._safety_union_merge_count, 1)

        # A later smaller bundle must not remove the earlier point.
        node._update_matched_geometry_diagnostics(
            matched,
            {"points": np.asarray([[0.10, 0.00, 0.50]]),
             "keys": (node._cell_key([0.10, 0.00, 0.50]),),
             "centroid": np.asarray([0.10, 0.00, 0.50])},
            "candidate_vbc_active_set")
        self.assertEqual(matched["points"].shape, (3, 3))
        self.assertEqual(node._safety_union_merge_count, 1)

    def test_single_obligation_uses_bounded_vbc_frontier_step(self):
        node = self._node({"_shared_path_candidate_ids",
                           "_compute_visibility_frontier",
                           "_dependency_cycle_active"})
        node._obligations = [obligation(1)]
        node._obligations[0]["q_vis_joint_mask"] = np.asarray(
            [1., 1., 1., 1., 0., 0., 0.])
        node._repair_stack = [1]
        node._latest_measured_q = np.zeros(7)
        node.frontier_steering_enabled = True
        node.vbc_gated_frontier_step_enabled = True
        node.frontier_step_inf = 0.05
        node.frontier_gradient_eps = 1e-6
        node.device = "cpu"
        node._clamp = lambda tensor: (tensor, None)
        node._frontier_lock = threading.RLock()
        node._frontier_cache_key = None
        node._frontier_cache = None
        node._frontier_compute_count = 0
        node._frontier_last_mode = "startup"
        node._frontier_last_considered_ids = []
        node._frontier_last_f_values = {}
        node._frontier_last_softmin_weights = {}
        node._frontier_last_direction_norm_inf = np.nan
        node._frontier_last_target_shift_inf = np.nan
        node._frontier_last_weight_scale = 0.0
        node._frontier_last_qvis_weight_scale = 1.0
        node._frontier_last_cycle_active = False
        node._frontier_feedback_failures = 0
        node.frontier_vbc_escalation_enabled = False
        node.frontier_vbc_escalation_after = 3
        node.frontier_escalated_step_inf = 0.10
        node._schedule_publish_lock = threading.RLock()
        node._trace_published_target = None

        result = node._compute_visibility_frontier()
        self.assertIsNotNone(result)
        self.assertEqual(result["mode"],
                         "vbc_gated_single_obligation_frontier_step")
        self.assertEqual(result["considered_ids"], [1])
        self.assertLessEqual(result["target_shift_inf"], 0.05 + 1e-12)
        self.assertAlmostEqual(result["target"][0], 0.05, places=12)
        np.testing.assert_array_equal(
            result["q_vis_joint_mask"], [1., 1., 1., 1., 0., 0., 0.])
        np.testing.assert_array_equal(result["target"][4:], [0., 0., 0.])


if __name__ == "__main__":
    unittest.main(verbosity=2)
