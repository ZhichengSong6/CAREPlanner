#!/usr/bin/env python3
"""Actual production methods with controlled generation/map, optionally real ROS."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_visibility_cdf/scripts'))
from pending_vbc_blockers import PendingVbcBlockers
from observation_identity import observation_region_token, vbc_bundle_identity

REL = 'src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py'
BASE = 'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py'
OLD = ROOT/'outputs/phase_e_case001_blocker_handoff_fix_20261012/before'

def subject(version='current', publish=None):
    scope = dict(np=np, math=math, Float64MultiArray=NS, Bool=NS, List=List,
        observation_region_token=observation_region_token, vbc_bundle_identity=vbc_bundle_identity,
        rospy=NS(logerr=lambda *a: None, logerr_throttle=lambda *a: None,
                 logwarn=lambda *a: None, logwarn_throttle=lambda *a: None,
                 Time=NS(now=lambda: NS(to_sec=lambda: 100.))))
    names = {'_active_set_bundle_cb', '_process_new_active_set', '_set_acquisition_completion'}
    path = (OLD if version == 'before' else ROOT)/REL
    tree = ast.parse(path.read_text())
    nodes = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), scope)
    methods = {n.name: scope[n.name] for n in nodes}
    # Exercise actual map-driven obligation retirement, including the new hook.
    base_path = (OLD if version == 'before' else ROOT)/BASE
    tree = ast.parse(base_path.read_text())
    nodes = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body
             if isinstance(n, ast.FunctionDef) and n.name == '_update_actual_visibility_completion']
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(base_path), 'exec'), scope)
    methods['check_seen'] = scope['_update_actual_visibility_completion']
    n = type('Subject', (), methods)()
    for key in ['seq', 'generation', 'last_execution_seq', 'last_candidate_seq',
                'source_seq', 'received_count', 'processed_count', 'drop_count']:
        setattr(n, '_coherent_bundle_'+key, 0)
    for key in ['raw_active_set_serial', 'processed_active_set_serial',
                'visibility_episode_reopen_count', 'process_attempt_count', 'process_success_count']:
        setattr(n, '_'+key, 0)
    for key in ['coherent_bundle_lock', 'obligation_lock', 'lock', 'schedule_publish_lock']:
        setattr(n, '_'+key, threading.RLock())
    n._pending_vbc_blockers = PendingVbcBlockers()
    n._coherent_bundle_enabled = n._c49_ready = n._c47_ready = True
    n._coherent_bundle_pending_nonempty = False
    n._coherent_bundle_source = 'none'
    n._coherent_bundle_points = np.zeros((0, 3))
    n._coherent_bundle_sweep_s = math.nan
    n._obligations = []; n._repair_stack = []
    n._acquisition_complete = n._acquisition_started = False
    n._cell_key = lambda p: tuple(np.rint(np.asarray(p)/.05).astype(int))
    n.publications = []
    def emitted(msg):
        n.publications.append(msg.data)
        if publish: publish(msg.data)
    n.acquisition_complete_pub = NS(publish=emitted)
    n._process_final_recovery = n._process_gcdf_recovery_event = lambda: None
    n._preferred_trajectory_locked = lambda: (NS(), NS(), 'controlled')
    n._latest_measured_q = np.zeros(7)
    n._cluster_regions = lambda raw: [dict(points=raw)]
    n._absorb_refined_partition_region = lambda *a: None
    def match(region):
        return next((ob for ob in n._obligations if np.array_equal(ob['points'], region['points'])), None)
    n._match_existing = match
    n._update_matched_geometry_diagnostics = lambda *a: None
    n.max_obligations = 8
    n._schedule_new_obligations = n._schedule_matched_obligations = n._schedule_generation_failures = 0
    n.after_generation = lambda: None
    def generate(region, *args):
        ob = dict(id=len(n._obligations)+1, points=region['points'].copy(), generation_event_id='controlled')
        n.after_generation()
        return ob
    n._generate_new_obligation = generate
    n._prune_or_initialize_stack = n._publish_schedule = n._publish_blocker_stack_summary = lambda: None
    n._consider_active_layer = lambda *a, **kw: None
    n._process_pending_refinement = lambda *a: None
    n.events = []
    n._trace_observation = lambda stage, **kw: n.events.append(dict(stage=stage, **kw))
    n._last_visibility_check_s = -math.inf; n.visibility_check_rate = 20.
    n._seen_obligation_count = 0; n._trace_seen_samples = {}
    n.known = False
    n._query_region_seen = lambda ob: (n.known, float(n.known), float(n.known),
        float(n.known), float(n.known), len(ob['points']), ob['points'][0], 0., {'q_distance_inf': 0.})
    n._record_seen_query = lambda *a: None
    n._obligation_geometry_diagnostics = lambda ob: {}
    n._ordered_obligations = lambda: n._obligations
    n._publish_acquisition_summary = lambda *a: None
    return n

def message(seq=85, stamp=20470000000, points=None, sweep=0.):
    points = [[0., -.25, .85]] if points is None else points
    return NS(data=[float(seq), sweep if points else math.nan, float(len(points))]+np.asarray(points).reshape(-1).tolist(),
              layout=NS(dim=[NS(label=f'care_vbc_v1_{stamp}_{seq}')]))

def state(n):
    return dict(pending=n._coherent_bundle_pending_nonempty,
        inbox_count=len(n._pending_vbc_blockers.records),
        capacity_fault=n._pending_vbc_blockers.overflow,
        enqueued=n._pending_vbc_blockers.enqueued,
        materialized=n._pending_vbc_blockers.materialized,
        received=n._coherent_bundle_received_count,
        obligations=[dict(id=ob['id'], points=ob['points'].tolist()) for ob in n._obligations],
        complete=n._acquisition_complete, seen=n._seen_obligation_count,
        process_reason=getattr(n, '_last_process_reason', None), events=n.events)

def controls():
    report = dict(status='PASS', checks=[], historical_control={})
    for version in ['before', 'current']:
        n = subject(version)
        n._active_set_bundle_cb(message())
        n._active_set_bundle_cb(message(518, 20471000000, []), urgent_recovery=True)
        n._process_new_active_set()
        report['historical_control'][version] = state(n)
        assert len(n._obligations) == (0 if version == 'before' else 1)
        if version == 'current':
            assert n.events[-1]['vbc_identity']['trajectory_stamp_ns'] == '20470000000'
            n.check_seen(); assert not n._acquisition_complete
            n.known = True; n._last_visibility_check_s = -math.inf
            n.check_seen(); assert n._acquisition_complete and n._seen_obligation_count == 1
    report['checks'].append('same recorded candidate/other-source empty: before loses, current materializes; actual seen alone clears')
    n = subject(); n._active_set_bundle_cb(message())
    n._active_set_bundle_cb(message(86, 20472000000, []))
    n._process_new_active_set(); assert len(n._obligations) == 1
    report['checks'].append('same-source newer empty preserves work')
    n = subject(); n._active_set_bundle_cb(message())
    n._active_set_bundle_cb(message(86, 20470000000))
    assert len(n._pending_vbc_blockers.records) == 1
    n._active_set_bundle_cb(message(84, 20470000000, []))
    assert n._pending_vbc_blockers.pending
    report['checks'].append('same audit/geometry coalesces; old sequence cannot erase')
    n = subject(); n._active_set_bundle_cb(message())
    n.after_generation = lambda: n._active_set_bundle_cb(message(86, 20472000000, [[.2,.2,.2]]))
    n._process_new_active_set(); assert n._coherent_bundle_pending_nonempty
    n.after_generation = lambda: None
    n._process_new_active_set(); assert len(n._obligations) == 2
    assert not n._coherent_bundle_pending_nonempty
    report['checks'].append('arrival during generation survives acknowledgement of captured work')
    for kind in ['generation_failure', 'capacity_before', 'capacity_after']:
        n = subject(); n._active_set_bundle_cb(message())
        if kind == 'generation_failure':
            def fail(*a): raise RuntimeError('controlled inference failure')
            n._generate_new_obligation = fail
        elif kind == 'capacity_before': n.max_obligations = 0
        else: n.after_generation = lambda: setattr(n, 'max_obligations', 0)
        n._process_new_active_set()
        assert n._pending_vbc_blockers.pending and not n._obligations
        n.check_seen(); assert not n._acquisition_complete
        report['checks'].append(kind+' retains pending and blocks completion')
    n = subject(); n._pending_vbc_blockers = PendingVbcBlockers(capacity=1)
    n._active_set_bundle_cb(message())
    n._active_set_bundle_cb(message(86, 20472000000, [[.2,.2,.2]]))
    n._process_new_active_set(); n.known = True; n.check_seen()
    assert n._pending_vbc_blockers.overflow and not n._acquisition_complete
    assert True not in n.publications
    report['checks'].append('bounded inbox overflow explicitly holds even after stored obligation is seen')
    return report

def worker(version):
    import rospy
    from std_msgs.msg import Bool, Float64MultiArray, String
    rospy.init_node('blocker_handoff_consumer')
    output = rospy.Publisher('/blocker_test/state', String, queue_size=32, latch=True)
    complete = rospy.Publisher('/blocker_test/complete', Bool, queue_size=32)
    n = subject(version, publish=lambda value: complete.publish(Bool(data=value)))
    lock = threading.RLock()
    def snapshot(): output.publish(String(data=json.dumps(state(n))))
    def bundle(msg, urgent=False):
        with lock:
            n._active_set_bundle_cb(msg, urgent); snapshot()
    def command(msg):
        with lock:
            if msg.data == 'process': n._process_new_active_set()
            elif msg.data == 'seen':
                n.known = True; n._last_visibility_check_s = -math.inf; n.check_seen()
            elif msg.data == 'unseen':
                n.known = False; n._last_visibility_check_s = -math.inf; n.check_seen()
            snapshot()
    subs = [rospy.Subscriber('/blocker_test/candidate', Float64MultiArray, bundle, queue_size=32),
            rospy.Subscriber('/blocker_test/execution', Float64MultiArray, lambda msg: bundle(msg, True), queue_size=32),
            rospy.Subscriber('/blocker_test/command', String, command, queue_size=32)]
    snapshot(); rospy.spin()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--worker', choices=['before', 'current'])
    args = ap.parse_args()
    if args.worker: return worker(args.worker)
    report = controls()
    report['qualification'] = 'Actual production callback/process and actual map-driven retirement with controlled generation and map outcomes; no learned model, robot, or safety certification.'
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], checks=report['checks']), indent=2))

if __name__ == '__main__': main()
