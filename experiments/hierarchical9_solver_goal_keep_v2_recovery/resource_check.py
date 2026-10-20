#!/usr/bin/env python3
"""Resource-only recovery checks. No optimizer, label or original-source edits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

GIB = 1024 ** 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def assess_memory(rows, minimum_gib=20.0, expected_devices=4):
    """Admission rule, NOT a guarantee that all future training peaks will fit."""
    require(math.isfinite(minimum_gib) and minimum_gib > 0, 'Bad minimum free memory')
    require(len(rows) == expected_devices, f'Expected {expected_devices} visible GPUs, got {len(rows)}')
    bad = []
    for index, row in enumerate(rows):
        free, total = row['free_bytes'], row['total_bytes']
        require(isinstance(free, int) and isinstance(total, int) and 0 <= free <= total and total > 0,
                f'Invalid CUDA memory report for logical GPU {index}')
        if free < minimum_gib * GIB:
            bad.append(index)
    return {'ok': not bad, 'minimum_free_gib': minimum_gib, 'insufficient_logical_devices': bad,
            'devices': rows, 'meaning': 'Global free memory at admission, not future peak-fit certification'}


def validate_existing(root, code):
    root, code = Path(root).resolve(strict=True), Path(code).resolve(strict=True)
    out = root / 'formal'
    require(code.parent == root / 'sources', 'Not the original content-addressed source directory')
    job_path = out / 'job.json'
    job = json.loads(job_path.read_text())
    require(Path(job['out']).resolve(strict=True) == out, 'Frozen job points to a different output')
    require(Path(job['deps']).resolve(strict=True) == out / 'dependencies', 'Unexpected frozen dependency root')
    require(not (out / 'training_complete.json').exists(), 'Training already complete; use the original eval entry')
    require((code / 'run.py').is_file() and (code / 'run.sh').is_file(), 'Original runner is missing')
    mod_path = code / 'k2_common.py'
    spec = importlib.util.spec_from_file_location('k2_recovery_frozen_common', mod_path)
    require(spec is not None and spec.loader is not None, 'Cannot load original verification module')
    common = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = common
    spec.loader.exec_module(common)
    # The original checker validates the original source hashes, protocol,
    # dependency snapshots, source data identities, and frozen split/test inputs.
    common.verify(job)
    stages = {}
    for relative in ('reference_inputs', 'reference_mined/rank0', 'reference_mined/rank1',
                     'reference_mined/rank2', 'reference_mined/rank3', 'replay'):
        folder = out / relative
        if (folder / 'complete.json').is_file():
            common.stage(folder)
            stages[relative] = 'COMPLETE_HASH_VERIFIED_REUSE'
        elif folder.exists():
            raise ValueError(f'Existing incomplete final stage: {folder}; preserve it for inspection')
        else:
            stages[relative] = 'NOT_YET_PUBLISHED'
    for arm in common.ARMS:
        folder = out / arm
        if (folder / 'complete.json').is_file():
            common.stage(folder)
            stages[arm] = 'COMPLETE_HASH_VERIFIED_REUSE'
        elif (folder / 'latest.pt').is_file():
            stages[arm] = 'RESUME_CHECKPOINT_VALIDATED_BY_ORIGINAL_TRAINER'
        else:
            # This failure occurred at the first batch; the original trainer
            # can initialize from the fixed teacher with an empty arm folder.
            stages[arm] = 'NO_SAVED_UPDATE_START_FIXED_TEACHER'
    print(json.dumps({'root': str(root), 'frozen_source': str(code), 'stages': stages},
                     indent=2, ensure_ascii=False), flush=True)
    print('[PASS] original job/source/data identities; no source, labels, or checkpoints modified', flush=True)


def gpu_check():
    require(bool(os.environ.get('SLURM_JOB_ID')), 'GPU admission check must run inside the allocated Slurm job')
    # Print process identifiers for diagnosis only. Never kill or alter them.
    for cmd in (['nvidia-smi'],
                ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv']):
        print('[diagnostic] ' + ' '.join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=False, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print('[diagnostic unavailable]', str(exc), flush=True)
    import torch
    require(torch.cuda.is_available(), 'CUDA is unavailable in this allocation')
    rows = []
    for device in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device)
        row = {'logical_gpu': device, 'name': torch.cuda.get_device_name(device),
               'free_bytes': int(free), 'total_bytes': int(total),
               'free_gib': free / GIB, 'total_gib': total / GIB}
        rows.append(row)
        print(f'[GPU] logical={device} name={row["name"]} free={free/GIB:.2f}/{total/GIB:.2f} GiB', flush=True)
    report = assess_memory(rows)
    report.update(host=socket.gethostname(), slurm_job_id=os.environ['SLURM_JOB_ID'],
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'), pid=os.getpid())
    print(json.dumps(report, indent=2), flush=True)
    require(report['ok'], 'Allocated GPUs have insufficient free memory; NOT starting training. '
            'Inspect the printed process list; do not kill unknown processes or change batch size blindly.')
    print('[GPU ADMISSION PASS] four visible GPUs each have >=20 GiB free; checker exits before training', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('validate', 'gpu'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--code', type=Path)
    args = parser.parse_args()
    if args.mode == 'validate':
        require(args.root is not None and args.code is not None, '--root and --code required')
        validate_existing(args.root, args.code)
    else:
        gpu_check()


if __name__ == '__main__':
    main()
