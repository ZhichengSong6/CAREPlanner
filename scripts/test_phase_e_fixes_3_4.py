#!/usr/bin/env python3
"""Run the actual regime-manager callbacks without ROS, sockets or actuators."""
import ast
from collections import deque
import math
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace as NS
import unittest
from summarize_phase_e_fix34_regression import bound_conflicts

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py"


class Publisher:
    def __init__(self): self.messages = []
    def publish(self, msg): self.messages.append(msg)


class StallTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        methods = {"_task_stall_cb", "_finish_task_stall_locked", "_begin_blocker_rediscovery_locked",
                   "_candidate_vbc_summary_cb", "_visibility_waypoint_active_cb", "_timer_cb",
                   "_task_uncertified_cb", "_candidate_outcome_cb", "_retire_qp_retry_locked", "_pump_qp_retry_locked",
                   "_maybe_enter_observation_repair_locked"}
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign))]
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            nodes.extend(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods)
        ns = dict(math=math, re=re, time=time, Bool=lambda **kw: NS(**kw),
                  String=lambda **kw: NS(**kw), Float64=lambda **kw: NS(**kw), rospy=NS(Time=NS(now=lambda: 10.)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
        cls = type("Callbacks", (), {k: ns[k] for k in methods})
        self.m = cls(); m = self.m
        m._lock = threading.RLock()
        m.NORMAL = "NORMAL"; m.PROBE_NORMAL = "PROBE_NORMAL"; m.REPAIR = "REPAIR"
        m.state = m.NORMAL; m.execution_ready = True
        m.qp_retry_token = ""; m.qp_retry_mode = ""; m.qp_retry_last_send = -math.inf
        m.qp_retry_retired = deque(maxlen=64); m.qp_retry_send_count = 0
        m.qp_retry_pub = Publisher()
        m.task_stall_stamp_ns = 0; m.task_stall_deadline = None; m.task_stall_confirmed = False
        m.task_stall_outcome = "none"; m.blocker_rediscovery_origin = "none"
        m.blocker_rediscovery_pending = False; m.blocker_rediscovery_force_bootstrap = False
        m.blocker_rediscovery_count = m.blocker_rediscovery_vbc_safe_count = m.blocker_rediscovery_vbc_unsafe_count = 0
        m.force_vbc_bootstrap_pub = Publisher(); m.replan_request_pub = Publisher()
        m.vbc_unsafe_execution_stamp_ns = 0
        m.visibility_waypoint_active = False
        m._probe_single_flight_busy_locked = lambda: False
        m.transitions = []
        m._transition_locked = lambda *args: m.transitions.append(args)

    def stall(self): self.m._task_stall_cb(NS(data="status=stalled query_stamp_ns=123 failures=5"))
    def verdict(self, stamp=123, unsafe=False, success=True):
        self.m._candidate_vbc_summary_cb(NS(data="trajectory_source=scp_rediscovery trajectory_stamp_ns={} success={} has_violation={}".format(stamp,int(success),int(unsafe))))

    def test_no_blocker_is_terminal_not_replan(self):
        self.stall(); self.verdict()
        self.assertEqual(self.m.task_stall_outcome, "QP_NO_PROGRESS_NO_VBC_BLOCKER")
        self.assertFalse(self.m.replan_request_pub.messages)
        self.assertFalse(self.m.transitions)

    def test_old_or_invalid_verdict_cannot_clear_stall(self):
        self.stall(); self.verdict(stamp=122); self.verdict(success=False)
        self.assertTrue(self.m.blocker_rediscovery_pending)

    def test_unknown_rows_do_not_create_obligation(self):
        self.stall(); self.verdict(unsafe=True)
        self.assertFalse(self.m.transitions)
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertEqual(len(self.m.transitions), 1)

    def test_old_waypoint_alone_does_not_enter_repair(self):
        self.stall(); self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertFalse(self.m.transitions)
        self.verdict(unsafe=True)
        self.assertEqual(len(self.m.transitions), 1)

    def test_certified_normal_stall_uses_live_obligation_after_exact_vbc(self):
        self.m.visibility_waypoint_active = True
        self.m._task_stall_cb(NS(data=(
            "status=stalled classification=NORMAL_CERTIFIED_NO_MEASURED_PROGRESS "
            "query_stamp_ns=123 failures=5")))
        self.assertFalse(self.m.transitions)
        self.verdict(stamp=122, unsafe=False)
        self.assertFalse(self.m.transitions)
        self.verdict(stamp=123, unsafe=False)
        self.assertEqual(len(self.m.transitions), 1)
        self.assertEqual(self.m.transitions[0][0], self.m.REPAIR)
        self.assertEqual(self.m.task_stall_outcome,
                         "qp_stall_existing_visibility_obligation")

    def test_duplicate_does_not_extend_deadline(self):
        self.stall(); deadline = self.m.task_stall_deadline; self.stall()
        self.assertEqual(self.m.task_stall_deadline, deadline)
        self.assertEqual(self.m.blocker_rediscovery_count, 1)

    def test_early_unknown_stall_still_requires_same_query_and_live_obligation(self):
        self.m.visibility_waypoint_active = True
        self.m._task_stall_cb(NS(data=(
            "status=stalled classification=NORMAL_UNKNOWN_CONSTRAINED_NO_MEASURED_PROGRESS "
            "query_stamp_ns=123 unknown_evidence_query_stamp_ns=122 active_execution_stamps=3 duration_s=2")))
        self.assertFalse(self.m.transitions)
        self.verdict(stamp=122, unsafe=False)
        self.assertFalse(self.m.transitions)
        self.verdict(stamp=123, unsafe=False)
        self.assertEqual(len(self.m.transitions), 1)
        self.assertEqual(self.m.transitions[0][0], self.m.REPAIR)

    def test_early_unknown_stall_without_live_obligation_stays_normal(self):
        self.m.visibility_waypoint_active = False
        self.m._task_stall_cb(NS(data=(
            "status=stalled classification=NORMAL_UNKNOWN_CONSTRAINED_NO_MEASURED_PROGRESS "
            "query_stamp_ns=123")))
        self.verdict(stamp=123, unsafe=False)
        self.assertFalse(self.m.transitions)
        self.assertEqual(self.m.task_stall_outcome, "QP_NO_PROGRESS_NO_VBC_BLOCKER")

    def test_late_obligation_after_same_query_safe_resumes_once(self):
        self.stall(); self.verdict()
        self.m._visibility_waypoint_active_cb(NS(data=False))
        self.assertFalse(self.m.transitions)
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertEqual(len(self.m.transitions), 1)
        self.assertEqual(self.m.transitions[0][0], self.m.REPAIR)
        self.assertEqual(self.m.task_stall_outcome, "qp_stall_late_visibility_obligation")
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertEqual(len(self.m.transitions), 1)
        self.assertFalse(self.m.replan_request_pub.messages)
        self.assertFalse(self.m.task_stall_confirmed)

    def test_late_obligation_requires_matching_successful_verdict(self):
        self.stall(); self.verdict(stamp=122); self.verdict(success=False)
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertFalse(self.m.transitions)

    def test_late_obligation_cannot_revive_reset_stall(self):
        self.stall(); self.verdict()
        self.m._task_stall_cb(NS(data="status=reset reason=fresh_reference"))
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertFalse(self.m.transitions)

    def test_late_obligation_waits_for_execution_gate(self):
        self.stall(); self.verdict(); self.m.execution_ready = False
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertFalse(self.m.transitions)
        self.m.execution_ready = True
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertEqual(len(self.m.transitions), 1)

    def test_late_obligation_does_not_interrupt_probe_flight(self):
        self.m.state = self.m.PROBE_NORMAL
        self.stall(); self.verdict()
        self.m._probe_single_flight_busy_locked = lambda: True
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertFalse(self.m.transitions)
        self.m._probe_single_flight_busy_locked = lambda: False
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertEqual(len(self.m.transitions), 1)

    def test_diagnostic_timeout_is_not_same_query_safe(self):
        self.stall()
        self.m._finish_task_stall_locked("QP_NO_PROGRESS_EXACT_DIAGNOSTIC_TIMEOUT")
        self.m._visibility_waypoint_active_cb(NS(data=True))
        self.assertFalse(self.m.transitions)

    def test_new_reference_resets_diagnostics(self):
        self.stall(); self.m._task_stall_cb(NS(data="status=reset reason=fresh_reference"))
        self.assertEqual(self.m.task_stall_stamp_ns, 0)
        self.assertFalse(self.m.blocker_rediscovery_pending)

    def test_late_uncertified_cannot_restart(self):
        self.stall(); self.m._task_uncertified_cb(NS(data=True))
        self.assertFalse(self.m.replan_request_pub.messages)

    def test_primal_failure_is_retry_not_visibility_evidence(self):
        self.m._task_stall_cb(NS(data="status=retry classification=QP_PRIMAL_INFEASIBLE_UNATTRIBUTED token=fixture mode=NORMAL"))
        self.assertEqual(len(self.m.qp_retry_pub.messages), 1)
        self.assertFalse(self.m.transitions)
        self.assertFalse(self.m.blocker_rediscovery_pending)
        self.stall()
        self.m._task_stall_cb(NS(data="status=retry classification=QP_PRIMAL_INFEASIBLE_UNATTRIBUTED token=fixture mode=NORMAL"))
        self.assertEqual(len(self.m.qp_retry_pub.messages), 1)

    def test_timer_timeout_is_finite(self):
        self.stall(); m = self.m; m.task_stall_deadline = time.monotonic()-1
        for key in ("clear_until", "execution_summary_time", "tracker_summary_time", "execution_ready_time", "physical_deadline_abs"):
            setattr(m,key,None)
        m.input_timeout = 1.; m.tracker_active = False
        m._fresh = lambda *a: False; m._publish_summary = lambda: None
        for key in ("trigger_pub", "probe_active_pub", "clear_pub", "hold_pub"):
            setattr(m,key,Publisher())
        m._timer_cb(None)
        self.assertEqual(m.task_stall_outcome, "QP_NO_PROGRESS_EXACT_DIAGNOSTIC_TIMEOUT")
        self.assertFalse(m.blocker_rediscovery_pending)

    def test_vbc_rejection_holds_only_the_old_execution(self):
        m = self.m
        m.task_stall_deadline = None
        m.execution_ready_time = None
        m.execution_summary_time = m.tracker_summary_time = None
        m.physical_deadline_abs = None
        m.input_timeout = 1.
        m.tracker_active = True
        m.tracker_execution_stamp_ns = 42
        m.vbc_unsafe_execution_stamp_ns = 42
        m._fresh = lambda *a: True
        m._publish_summary = lambda: None
        for key in ("trigger_pub", "probe_active_pub", "clear_pub", "hold_pub"):
            setattr(m, key, Publisher())
        m._timer_cb(None)
        self.assertTrue(m.hold_pub.messages[-1].data)
        m.tracker_execution_stamp_ns = 43
        m._timer_cb(None)
        self.assertFalse(m.hold_pub.messages[-1].data)

    def test_repair_candidate_rejection_preserves_execution_verdict(self):
        m = self.m
        m.state = m.REPAIR
        m.tracker_active = True
        m.tracker_execution_stamp_ns = 42
        m.last_candidate_seq = 0
        m.candidate_event_count = 0
        m.candidate_unsafe_count = 0
        m.candidate_unsafe_streak = 0
        m._candidate_outcome_cb(NS(data=(
            "seq=1 result=unsafe committed=0 safety_gate=vbc "
            "verification_view=repair_prefix_brake_hold")))
        self.assertEqual(m.vbc_unsafe_execution_stamp_ns, 0)
        self.assertEqual(len(m.replan_request_pub.messages), 1)

    def test_candidate_rejection_cannot_bind_execution_during_startup(self):
        m = self.m
        m.execution_ready = False
        m.tracker_active = True
        m.tracker_execution_stamp_ns = 42
        m.last_candidate_seq = 0
        m.candidate_event_count = 0
        m.candidate_unsafe_count = 0
        m.candidate_unsafe_streak = 0
        m._candidate_outcome_cb(NS(data=(
            "seq=1 result=unsafe committed=0 safety_gate=vbc "
            "verification_view=repair_prefix_brake_hold")))
        self.assertEqual(m.vbc_unsafe_execution_stamp_ns, 0)
        m.task_stall_deadline = None
        class GraceStart:
            def __rsub__(self, now): return NS(to_sec=lambda: 0.5)
        m.execution_ready_time = GraceStart()
        m.execution_summary_time = m.tracker_summary_time = None
        m.physical_deadline_abs = None
        m.input_timeout = 1.
        m._fresh = lambda *a: True
        m._publish_summary = lambda: None
        for key in ("trigger_pub", "probe_active_pub", "clear_pub", "hold_pub"):
            setattr(m, key, Publisher())
        m._timer_cb(None)
        self.assertFalse(m.hold_pub.messages[-1].data)


class SnapshotBoundTests(unittest.TestCase):
    def test_independent_bound_certificate(self):
        # x in [-1,1], 2*x >= 3 is impossible; 2*x >= 1 is not disproven.
        d = dict(G=dict(coo=[[0,0,2.]]), x_l=[-1.],x_u=[1.],h_l=[3.],
                 selected_pairs=[[0,0]],original_timestep=[1],source_type=[0])
        self.assertEqual(bound_conflicts(d)[0]["deficit"],1.)
        d["h_l"] = [1.]; self.assertEqual(bound_conflicts(d),[])
        d["G"]["coo"][0][2] = -2.; d["h_l"] = [3.]
        self.assertEqual(bound_conflicts(d)[0]["deficit"],1.)


if __name__ == "__main__": unittest.main()
