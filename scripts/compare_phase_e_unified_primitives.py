#!/usr/bin/env python3
"""Strict P50/P95 non-regression of combined ROS receipt latencies (no tolerance)."""
import argparse
import json
from pathlib import Path


def compare(old, new):
    assert old['status'] == new['status'] == 'PASS', 'functional acceptance must pass first'
    assert old['backend'] == 'samples' and new['backend'] == 'primitive'
    assert not old['instrumented'] and not new['instrumented'], 'strace timings are not comparable'
    assert old.get('knots', 2) == new.get('knots', 2), 'different horizon workloads'
    assert old.get('map_query_dedicated_queue', False) == new.get('map_query_dedicated_queue', False), 'different map query scheduling'
    assert old.get('execution_event_driven', False) == new.get('execution_event_driven', False), 'different scheduling policies'
    assert old.get('persistent_confidence', False) == new.get('persistent_confidence', False), 'different service transports'
    assert old.get('measurement_schema') == new.get('measurement_schema'), 'different output identity bindings'
    assert old.get('shadow_event_driven', False) == new.get('shadow_event_driven', False), 'different shadow policies'
    assert old.get('input_period_s', 0.) == new.get('input_period_s', 0.), 'different input cadence'
    assert old.get('fixture_tcp_nodelay', False) == new.get('fixture_tcp_nodelay', False), 'different fixture transports'
    assert old.get('verification_tcp_nodelay', False) == new.get('verification_tcp_nodelay', False), 'different production verification transports'
    assert old.get('unknown_soak', False) == new.get('unknown_soak', False), 'different map workloads'
    assert old.get('observer_history_limit', 0) == new.get('observer_history_limit', 0), 'different observation retention'
    assert len(old['latency_rows_ms']) == len(new['latency_rows_ms']) >= 20
    common = old['sha256'].keys() & new['sha256'].keys()
    assert common and all(old['sha256'][k] == new['sha256'][k] for k in common), 'shared input hash mismatch'
    assert old['latency_ms'].keys() == new['latency_ms'].keys()
    rows = {}
    for path in old['latency_ms']:
        a, b = old['latency_ms'][path], new['latency_ms'][path]
        rows[path] = dict(samples_ms=a, primitive_ms=b,
            ratio={q: b[q]/a[q] for q in ['p50', 'p95']},
            status='PASS' if all(b[q] <= a[q] for q in ['p50', 'p95']) else 'FAIL')
    return dict(status='PASS' if all(r['status'] == 'PASS' for r in rows.values()) else 'FAIL',
                scope='Sequential isolated ROS fixture runs; input publication to stamp-matched output receipt. '
                      'Includes timer phase, scheduling and transport; not hard-real-time or a full planner/tracker benchmark.',
                tolerance=0, repetitions=len(old['latency_rows_ms']), metrics=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--primitive', type=Path, required=True)
    ap.add_argument('--output-file', type=Path, required=True)
    args = ap.parse_args()
    report = compare(json.loads(args.samples.read_text()), json.loads(args.primitive.read_text()))
    report['inputs'] = dict(samples=str(args.samples.resolve()), primitive=str(args.primitive.resolve()))
    with args.output_file.open('x') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
