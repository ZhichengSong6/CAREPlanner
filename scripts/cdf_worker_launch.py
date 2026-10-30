#!/usr/bin/env python3
"""Convert explicit model environment declarations to safe worker argv."""
import json
import os
from pathlib import Path
import re
import sys


def command(env, repo, python=sys.executable):
    required = ['CHECKPOINT', 'CDF_ACTIVATION', 'CDF_EXPECTED_SHA256', 'CDF_MODEL_ATTESTATION',
                'CDF_INPUT_AUDIT', 'GPU_SOCKET', 'GPU_DEVICE']
    if any(not env.get(k) for k in required):
        raise ValueError('Missing model/audit declaration: '+', '.join(k for k in required if not env.get(k)))
    if env['CDF_ACTIVATION'] not in ('gelu', 'relu'):
        raise ValueError('Activation must be explicitly gelu or relu')
    if not re.fullmatch('[0-9a-f]{64}', env['CDF_EXPECTED_SHA256']):
        raise ValueError('Expected checkpoint SHA256 must be a lowercase digest')
    checkpoint = Path(env['CHECKPOINT']).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError('Checkpoint is not a file')
    for key in ('CDF_MODEL_ATTESTATION', 'CDF_INPUT_AUDIT'):
        path = Path(env[key]).resolve()
        if path.exists():
            raise ValueError('Refusing to overwrite '+str(path))
        for ancestor in path.parents:
            if (ancestor/'artifact_sha256.json').exists():
                raise ValueError('Refusing to write under sealed study '+str(ancestor))
    return [python, '-u', str(Path(repo)/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py'),
            '--checkpoint', str(checkpoint), '--activation', env['CDF_ACTIVATION'],
            '--checkpoint-key', env.get('CDF_CHECKPOINT_KEY', 'latest'),
            '--expected-checkpoint-sha256', env['CDF_EXPECTED_SHA256'],
            '--model-attestation', env['CDF_MODEL_ATTESTATION'], '--domain-audit', env['CDF_INPUT_AUDIT'],
            '--device', env['GPU_DEVICE'], '--socket', env['GPU_SOCKET'],
            '--max-pairs', '8000', '--warmup-pairs', '2048']


if __name__ == '__main__':
    argv = command(os.environ, Path(__file__).resolve().parents[1])
    if '--print-command' in sys.argv:
        print(json.dumps(argv))
    else:
        os.execv(sys.executable, argv)
