#!/usr/bin/env python3
"""Exact audit identity, legacy compatibility, temporal evidence and routing."""
import ast
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
from types import SimpleNamespace as NS
from typing import List
import unittest
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src/care_visibility_cdf/scripts'))
from observation_event_trace import vbc_bundle_identity, observation_region_token
from analyze_phase_e_observation_events import attach_vbc_evidence, tokens, string_rows, solver_point_key
from pending_vbc_blockers import PendingVbcBlockers


def methods(path, names, scope):
    tree = ast.parse((REPO / path).read_text())
    nodes = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, 'exec'), scope)
    return type('Subject', (), {name: scope[name] for name in names})()


class VbcEvidenceTests(unittest.TestCase):
    def test_same_fp32_solver_point_not_nearby_voxel(self):
        a = [.2, -.15, .55]
        b = np.asarray(a, dtype=np.float32).astype(float).tolist()
        self.assertNotEqual(a, b)
        self.assertEqual(solver_point_key(a), solver_point_key(b))
        self.assertNotEqual(solver_point_key(a), solver_point_key([.25, -.15, .55]))

    def test_ros_unquoted_commas_and_standard_csv_are_both_lossless(self):
        import csv
        evidence = dict(schema=1, trajectory_stamp_ns='100', bundle_seq=7, violations=[])
        payload = 'target=[1,2,3] audited_trajectory_stamp_ns=100 vbc_evidence=' + json.dumps(evidence, separators=(',', ':'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'string.csv'
            for quoted in (False, True):
                with path.open('w') as stream:
                    if quoted:
                        writer = csv.writer(stream)
                        writer.writerow(['%time', 'field.data']); writer.writerow([1000000000, payload])
                    else:
                        stream.write('%time,field.data\n1000000000,' + payload + '\n')
                rows = string_rows(path, dict(start_ros_s=0, end_ros_s=2))
                self.assertEqual(rows[0]['target'], '[1,2,3]')
                self.assertEqual(json.loads(rows[0]['vbc_evidence']), evidence)

    def test_cpp_serialization_roundtrip_and_negative_deadline(self):
        evidence = json.loads(subprocess.check_output(['/tmp/care_vbc_evidence_test'], text=True))
        self.assertEqual(evidence['trajectory_stamp_ns'], '9007199254740993')
        a, b = evidence['violations']
        self.assertEqual(a['point'], [.015, -.025, .035])
        self.assertEqual(a['sweeping_link'], 'link 4"\\')
        self.assertIsNone(a['see_s'])
        self.assertEqual(a['refinement_status'], 'sampled_intersection')
        self.assertEqual(a['refinement_fk_before'], 256)
        self.assertEqual(a['refinement_fk_after'], 256)
        self.assertEqual(a['candidate_margin_m'], .02)
        self.assertLess(a['latest_see_s'], 0)
        self.assertEqual(b['layer'], 1)
        self.assertTrue(b['nominally_visible'])

    def test_bundle_metadata_is_optional_and_exact(self):
        msg = NS(layout=NS(dim=[NS(label='care_vbc_v1_123456789000000001_7')]))
        self.assertEqual(vbc_bundle_identity(msg, 7)['trajectory_stamp_ns'], '123456789000000001')
        self.assertIsNone(vbc_bundle_identity(msg, 8))
        self.assertIsNone(vbc_bundle_identity(NS(data=[1., .1, 0.]), 1))
        msg.layout.dim[0].label = 'care_vbc_v1_0_7'
        self.assertIsNone(vbc_bundle_identity(msg, 7))

    def test_near_stamp_or_periodic_refresh_never_inherits_route(self):
        evidence = dict(schema=1, trajectory_stamp_ns='100', bundle_seq=7,
                        violations=[dict(latest_see_s=-.15)])
        encoded = json.dumps(evidence, separators=(',', ':'))
        outcomes = [dict(safety_gate='vbc', audited_trajectory_stamp_ns=s, vbc_evidence=encoded)
                    for s in ('100', '101')]
        outcomes += [dict(safety_gate='vbc'), dict(safety_gate='gcdf_unknown')]
        events = [dict(stage='blocker_route', vbc_identity=dict(trajectory_stamp_ns=s, bundle_seq=b))
                  for s, b in [('100', 8), ('101', 7)]]
        counts = attach_vbc_evidence(outcomes, events)
        self.assertEqual(counts['exact_audits'], 1)
        self.assertEqual(counts['invalid_or_mismatched'], 1)
        self.assertEqual(counts['missing'], 1)
        self.assertEqual(outcomes[0]['vbc_route_events'], [])
        events.append(dict(stage='blocker_route', vbc_identity=dict(trajectory_stamp_ns='100', bundle_seq=7)))
        attach_vbc_evidence(outcomes, events)
        self.assertEqual(len(outcomes[0]['vbc_route_events']), 1)

    def test_production_vbc_callback_copies_only_matching_audit(self):
        class Stamp:
            def __init__(self, ns): self.ns = ns
            def to_nsec(self): return self.ns
            def to_sec(self): return self.ns / 1e9
            def __sub__(self, other): return Stamp(self.ns - other.ns)
        pub = NS(publish=lambda msg: None)
        n = methods('src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py',
            {'_global_summary_cb'}, dict(copy=copy, _tokens=tokens,
                _as_bool=lambda v: {'1': True, '0': False}.get(v),
                rospy=NS(Time=NS(now=lambda: Stamp(200)))))
        n._lock = threading.RLock()
        n.initial_shell_recovery_enabled = False
        n._sync_repair_prefix_locked = lambda: None
        n._selector_cycle_count = 0; n._outstanding_dispatch_cycle = -1
        n._outstanding = NS(header=NS(stamp=Stamp(100)))
        n._outstanding_sent = Stamp(100); n._outstanding_raw_received = Stamp(90)
        n._outstanding_seq = 1; n._outstanding_repair = True; n._outstanding_probe = False
        n._outstanding_view = 'repair_prefix_brake_hold'; n._outstanding_diag = {'raw_candidate_stamp_ns': 90}
        n._exact_vbc_stamp_miss_count = n._verification_outcome_count = 0
        n._verification_unsafe_count = n._repair_prefix_unsafe_count = 0
        n._cycle_recovery_active = False
        n._publish_summary_locked = lambda: None
        n._dispatch_pending_locked = lambda now: (None, None)
        n.verification_pub = n.final_gcdf_query_pub = pub
        events = []
        n._make_verification_event = lambda *a, **kw: NS(diag=kw['diag'])
        n.verification_event_pub = NS(publish=events.append)
        n._global_summary_cb(NS(data='trajectory_source=predicted trajectory_stamp_ns=101 has_violation=1 vbc_evidence=wrong'))
        self.assertFalse(events)
        n._global_summary_cb(NS(data='trajectory_source=predicted trajectory_stamp_ns=100 has_violation=1 vbc_evidence=exact'))
        self.assertEqual(events[0].diag['audited_trajectory_stamp_ns'], 100)
        self.assertEqual(events[0].diag['vbc_evidence'], 'exact')
        self.assertEqual(events[0].diag['raw_candidate_stamp_ns'], 90)

    def test_production_route_uses_captured_bundle_not_refreshed_one(self):
        n = methods('src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py',
            {'_process_new_active_set'}, dict(math=math, np=np, List=List,
                observation_region_token=observation_region_token,
                rospy=NS(logerr=lambda *a: None, logerr_throttle=lambda *a: None,
                         logwarn_throttle=lambda *a: None)))
        n._c49_ready = n._coherent_bundle_enabled = True
        n._coherent_bundle_urgent_recovery = False
        n._coherent_bundle_source = 'candidate_vbc_active_set'
        n._coherent_bundle_source_seq = 7
        n._process_gcdf_recovery_event = lambda: None
        n._process_final_recovery = lambda: None
        n._coherent_bundle_lock = n._obligation_lock = n._lock = n._schedule_publish_lock = threading.RLock()
        n._coherent_bundle_seq = 7; n._coherent_bundle_sweep_s = .15
        n._trace_coherent_bundle_identity = dict(trajectory_stamp_ns='100', bundle_seq=7)
        n._coherent_bundle_points = np.array([[.015, .025, .035]])
        n._pending_vbc_blockers = PendingVbcBlockers()
        n._pending_vbc_blockers.add(dict(generation=7, source='candidate_vbc_active_set',
            source_seq=7, identity=n._trace_coherent_bundle_identity,
            points=n._coherent_bundle_points.copy(), sweep=.15))
        n._processed_active_set_serial = 0; n._process_attempt_count = n._process_success_count = 0
        n._coherent_bundle_pending_nonempty = True; n._coherent_bundle_processed_count = 0
        n._preferred_trajectory_locked = lambda: ('trajectory', 0, 'predicted')
        n._latest_measured_q = np.zeros(7); n._repair_stack = [1]
        n._cluster_regions = lambda raw: [dict(points=raw)]
        n._absorb_refined_partition_region = lambda *a: None
        n._match_existing = lambda region: None
        n._obligations = []; n.max_obligations = 2
        n._schedule_new_obligations = n._schedule_matched_obligations = 0
        def generate(*a):
            n._coherent_bundle_seq = 8
            n._trace_coherent_bundle_identity = dict(trajectory_stamp_ns='101', bundle_seq=8)
            n._pending_vbc_blockers.add(dict(generation=8, source='candidate_vbc_active_set',
                source_seq=8, identity=n._trace_coherent_bundle_identity,
                points=n._coherent_bundle_points.copy(), sweep=.15))
            return dict(id=2, points=n._coherent_bundle_points.copy(), generation_event_id='new')
        n._generate_new_obligation = generate
        n._prune_or_initialize_stack = n._publish_schedule = n._publish_blocker_stack_summary = lambda: None
        n._consider_active_layer = lambda *a, **kw: n._repair_stack.append(2)
        n._process_pending_refinement = lambda *a: None
        events = []; n._trace_observation = lambda stage, **kw: events.append(kw)
        n._process_new_active_set()
        self.assertEqual(events[0]['vbc_identity']['bundle_seq'], 7)
        self.assertEqual(events[0]['stack_before'], [1])
        self.assertEqual(events[0]['stack_after'], [1, 2])
        self.assertEqual(events[0]['routes'][0]['obligation_id'], 2)
        self.assertTrue(n._coherent_bundle_pending_nonempty)


if __name__ == '__main__': unittest.main(verbosity=2)
