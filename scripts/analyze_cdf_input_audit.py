#!/usr/bin/env python3
"""Reconcile complete worker-domain records; never infer safety from gaps."""
import argparse
import hashlib
import json
from pathlib import Path


def analyze(audit_path, attestation_path):
    model_raw = Path(attestation_path).read_bytes()
    model = json.loads(model_raw)
    with Path(audit_path).open() as stream:
        rows = [json.loads(line) for line in stream]
    assert rows and rows[0]['event'] == 'session'
    session = rows[0]
    assert session['attestation_sha256'] == hashlib.sha256(model_raw).hexdigest()
    assert session['pid'] == model['pid']
    assert session['domain_min_m'] == [-.8, -.8, 0.]
    assert session['domain_max_m'] == [.8, .8, 1.]
    active = None
    requested = completed = outside = strict = nonfinite = requests = 0
    input_times = []
    lower, upper = None, None
    for row in rows[1:-1]:
        if row['event'] == 'request':
            assert active is None
            requests += 1
            assert row['sequence'] == requests and row['pairs'] > 0
            assert len(row['payload_sha256']) == 64
            active = row
            requested += row['pairs']
            outside += row['outside_tolerance_rows']
            strict += row['strict_outside_rows']
            nonfinite += row['nonfinite_xyz_rows'] + row['nonfinite_q_rows']
            if row['xyz_min_m'] is not None:
                lower = row['xyz_min_m'] if lower is None else list(map(min, zip(lower, row['xyz_min_m'])))
                upper = row['xyz_max_m'] if upper is None else list(map(max, zip(upper, row['xyz_max_m'])))
        elif row['event'] == 'result':
            assert active is not None and row['sequence'] == active['sequence']
            assert row['pairs'] == active['pairs'] and row['finite_output']
            completed += row['pairs']
            input_times.append(row['input_audit_ms'])
            active = None
        else:
            raise ValueError('Audit contains an error/unexpected event: '+row['event'])
    close = rows[-1]
    assert close['event'] == 'close', 'Missing clean worker closure; recording incomplete'
    assert active is None and close['unfinished_sequence'] is None
    assert close['failed_requests'] == 0
    assert close['requests'] == requests
    assert close['started_pairs'] == close['finished_pairs'] == requested == completed
    assert close['outside_pairs'] == outside
    assert nonfinite == 0
    times = sorted(input_times)
    return dict(status='COMPLETE_WITH_DOMAIN_EXCEEDANCES' if outside else 'COMPLETE_RECORDED_INPUTS_IN_DOMAIN',
                requests=requests, pairs=requested, strict_outside_rows=strict,
                outside_tolerance_rows=outside, xyz_min_m=lower, xyz_max_m=upper,
                input_audit_p95_ms=times[min(len(times)-1, int(.95*len(times)))] if times else None,
                checkpoint_sha256=model['checkpoint_sha256'], activation=model['activation'],
                scope='Recorded CQ01 worker inputs only; no independent collision or global model-accuracy claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--attestation', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = analyze(a.audit, a.attestation)
    with Path(a.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))
