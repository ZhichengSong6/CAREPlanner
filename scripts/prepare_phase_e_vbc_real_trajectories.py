#!/usr/bin/env python3
"""Extract exact recorded JointTrajectory payloads; never synthesize rejected candidates."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from phase_e_evaluation_common import parse_stamp_ns, window_time_ns
from run_phase_e_repeatability import fingerprint, save


def extract(path, window):
    start, end = (window_time_ns(window, k) for k in ('start_ros', 'end_ros'))
    rows = []
    with path.open() as stream:
        for line, r in enumerate(csv.DictReader(stream), 2):
            assert None not in r, (path, line, 'extra unlabelled columns')
            receipt = parse_stamp_ns(r['%time'])
            if receipt > end:
                continue
            indices = sorted(int(m[1]) for key, value in r.items()
                             if value not in (None, '') and
                             (m := re.fullmatch(r'field.points(\d+)\.positions0', key)))
            assert indices == list(range(len(indices))) and indices, (path, line)
            names = [r['field.joint_names'+str(i)] for i in range(7)]
            assert len(set(names)) == 7
            points = []
            for k in indices:
                prefix = 'field.points'+str(k)
                q = [float(r[prefix+'.positions'+str(i)]) for i in range(7)]
                assert all(math.isfinite(v) for v in q)
                points.append(dict(q=q, time_ns=parse_stamp_ns(r[prefix+'.time_from_start'])))
            rows.append(dict(receipt_ns=receipt, stamp_ns=parse_stamp_ns(r['field.header.stamp']),
                             source_line=line, joint_names=names, points=points))
    # Include the last pre-window state only as explicit carry-in, like the
    # real selector's latched trajectory; do not relabel its original time.
    before = [r for r in rows if r['receipt_ns'] < start]
    selected = ([dict(before[-1], carry_in=True)] if before else [])
    selected += [dict(r, carry_in=False) for r in rows if r['receipt_ns'] >= start]
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads(args.summary.read_text())
    items, sources = [], {}
    for case in summary['cases']:
        for role in ('task', 'committed'):
            path = Path(case['run'])/(role+'_trajectory.csv')
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in extract(path, case['benchmark_window']):
                items.append(dict(row, case_id=case['case_id'], recorded_backend=case['geometry_backend'],
                                  role=role, source=str(path), id=len(items)))
    assert items
    save(out/'inputs.json', dict(schema_version=1, trajectories=items, source_sha256=sources,
        summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        limitations='Recorded task and committed payloads only; rejected candidate payloads unavailable. '
                    'No map/service/ROS scheduling or downstream full-VBC latency replay.',
        grid=dict(min=[-.95, -.95, 0], max=[.95, .95, 1.15], resolution=.05),
        max_eval_timesteps=50, fallback_dt=.05, margins=[0., .01, .02]))
    save(out/'source_fingerprint.json', fingerprint())
    print('PASS extracted', len(items), 'trajectories from', len(sources), 'CSV files')


if __name__ == '__main__':
    main()
