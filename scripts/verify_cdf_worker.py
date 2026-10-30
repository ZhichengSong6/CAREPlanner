#!/usr/bin/env python3
"""Verify actual model binding and passive recorder before arming a task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def verify(attestation, audit, expected_sha, activation, socket_path):
    raw = Path(attestation).read_bytes()
    model = json.loads(raw)
    assert model['checkpoint_sha256'] == expected_sha
    assert model['activation'] == activation == model['architecture']['activation']
    assert not model['diagnostic_cpu'], 'CPU diagnostic is not an online GPU worker'
    assert hashlib.sha256(Path(model['checkpoint']).read_bytes()).hexdigest() == expected_sha
    os.kill(model['pid'], 0)
    assert stat.S_ISSOCK(Path(socket_path).stat().st_mode)
    with Path(audit).open() as f:
        session = json.loads(next(f))
    assert session['event'] == 'session' and session['pid'] == model['pid']
    assert session['attestation_sha256'] == hashlib.sha256(raw).hexdigest()
    assert session['domain_min_m'] == [-.8, -.8, 0.]
    assert session['domain_max_m'] == [.8, .8, 1.]
    return dict(status='MODEL_AND_INPUT_AUDIT_READY', pid=model['pid'],
                checkpoint_sha256=expected_sha, activation=activation)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('attestation', 'audit', 'expected-sha', 'activation', 'socket'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.attestation, a.audit, a.expected_sha, a.activation, a.socket)))
