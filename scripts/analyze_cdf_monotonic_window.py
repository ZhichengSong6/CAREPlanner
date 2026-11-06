#!/usr/bin/env python3
"""Reconcile a complete CDF session and select the unchanged score window."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analyze_cdf_input_audit import analyze
from cdf_monotonic_clock import task_bounds


def select(records, window):
    lo, hi = task_bounds(records, window)
    requests = [r for r in records if r['event'] == 'request']
    results = {r['sequence']: r for r in records if r['event'] == 'result'}
    groups = {'before': [], 'task': [], 'after': []}
    for r in requests:
        groups['before' if r['monotonic_ns'] < lo else 'after' if r['monotonic_ns'] > hi else 'task'].append(r)
    summary = {}
    for name, rows in groups.items():
        summary[name] = dict(requests=len(rows), pairs=sum(r['pairs'] for r in rows),
            outside_tolerance_rows=sum(r['outside_tolerance_rows'] for r in rows),
            sequences=[r['sequence'] for r in rows],
            xyz_min_m=[min(r['xyz_min_m'][i] for r in rows) for i in range(3)] if rows else None,
            xyz_max_m=[max(r['xyz_max_m'][i] for r in rows) for i in range(3)] if rows else None)
    return dict(status='MONOTONIC_TASK_PARTITION', groups=summary,
        requests_started_inside_finished_after=[r['sequence'] for r in groups['task'] if results[r['sequence']]['monotonic_ns'] > hi],
        wall_clock_reversals=[dict(previous_sequence=a.get('sequence'),next_sequence=b.get('sequence'),
            backward_ns=a['wall_time_ns']-b['wall_time_ns']) for a,b in zip(records,records[1:]) if b['wall_time_ns'] < a['wall_time_ns']],
        task_bounds_ns=[lo,hi], scope='Request record emission time, inclusive score bounds; wall time never determines membership.')


def report(audit, attestation, window_path, selector=None):
    session = analyze(audit, attestation)
    records = [json.loads(line) for line in Path(audit).read_text().splitlines()]
    window = json.loads(Path(window_path).read_text())
    window = window.get('benchmark_window', window)
    result = dict(session=session, **select(records, window))
    paths = [Path(audit), Path(attestation), Path(window_path)]
    if selector:
        paths.append(Path(selector))
        calls = [r for r in map(json.loads, Path(selector).read_text().splitlines())
                 if r.get('retained_pair_count',0)>0 and r.get('timing_ms',{}).get('worker_total_ms',0)>0]
        requests = [r for r in records if r['event']=='request']
        if len(calls)!=len(requests) or any(c['retained_pair_count']!=q['pairs'] for c,q in zip(calls,requests)):
            raise ValueError('Shared synchronous caller/worker batches differ')
        task = set(result['groups']['task']['sequences'])
        result['task_channels_by_request_clock'] = dict(Counter(c['channel'] for c,q in zip(calls,requests) if q['sequence'] in task))
        result['caller_alignment'] = 'Ordered single shared synchronous client; no caller wall-clock window filtering.'
    result['input_sha256'] = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audit','attestation','window','output'): p.add_argument('--'+key,required=True)
    p.add_argument('--selector');a=p.parse_args()
    destination=Path(a.output).resolve()
    if any((p/'artifact_sha256.json').exists() for p in destination.parents):raise ValueError('Sealed output')
    value=report(a.audit,a.attestation,a.window,a.selector)
    with destination.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(value['status'])
