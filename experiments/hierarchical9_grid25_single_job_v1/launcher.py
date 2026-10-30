#!/usr/bin/env python3
"""One Slurm allocation (one or two nodes) around the unchanged frozen Q0 collector."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ('launcher.py', 'run.sh', 'test_single_job.py')
ALLOWED_NODES = ('3090node2', '3090node3')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def parse_nodes(text):
    nodes = text.split(',')
    require(1 <= len(nodes) <= 2 and len(set(nodes)) == len(nodes)
            and all(n in ALLOWED_NODES for n in nodes),
            'NODES must be 3090node2, 3090node3, or 3090node2,3090node3')
    return nodes


def ensure_account_idle():
    user = pwd.getpwuid(os.getuid()).pw_name
    r = subprocess.run(['squeue', '-h', '-u', user, '-o', '%i %T %j'],
                       capture_output=True, text=True)
    require(r.returncode == 0, 'Cannot read your queue; no job submitted: ' + r.stderr.strip())
    require(not r.stdout.strip(), 'Existing queued/running job; no additional job submitted:\n' + r.stdout.strip())


def source_hashes(directory):
    return {name: sha(Path(directory) / name) for name in SOURCE_FILES}


def freeze(root):
    hashes = source_hashes(HERE)
    ident = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    parent = Path(root) / 'single_job_sources'
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / ident
    if not dest.exists():
        tmp = Path(tempfile.mkdtemp(prefix='.building.', dir=parent))
        try:
            for name in SOURCE_FILES + ('README_zh.md',):
                shutil.copyfile(HERE / name, tmp / name)
            write(tmp / 'launcher_manifest.json', {'sha256': hashes})
            os.rename(tmp, dest)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    require(source_hashes(dest) == hashes, 'Existing launcher snapshot changed')
    return dest


def verify_launcher():
    manifest = HERE / 'launcher_manifest.json'
    require(manifest.is_file(), 'Run via run.sh: immutable launcher snapshot required')
    require(read(manifest)['sha256'] == source_hashes(HERE), 'Launcher snapshot corrupted')


def verify_data(code, root):
    """Use the original snapshot's own integrity checker; never edit its hashes."""
    code, root = Path(code).resolve(), Path(root).resolve()
    path = code / 'g25_common.py'
    spec = importlib.util.spec_from_file_location('single_job_frozen_g25_common', path)
    require(spec is not None and spec.loader is not None, 'Frozen collector missing')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    job_path = root / 'formal/job.json'
    job = common.read(job_path)
    require(Path(job['out']).resolve() == root / 'formal', 'Collector output path mismatch')
    common.verify_job(job)
    return job_path


def sbatch_command(root, repo, nodes, limit):
    count = len(parse_nodes(nodes))
    return ['sbatch', '--parsable', '--partition=GPU', f'--nodes={count}',
            f'--ntasks={count}', '--ntasks-per-node=1', '--cpus-per-task=16',
            '--gres=gpu:3090:4', '--exclusive', '--no-requeue', f'--time={limit}',
            f'--nodelist={nodes}', f'--chdir={repo}', '--export=ALL',
            '--job-name=vis_grid25', f'--output={root}/logs/collect_%j.out',
            f'--error={root}/logs/collect_%j.out', str(HERE / 'run.sh'), '_batch']


def srun_command(root, code, nodes):
    count = len(parse_nodes(nodes))
    return ['srun', '--mpi=none', f'--nodes={count}', f'--ntasks={count}',
            '--ntasks-per-node=1', '--cpus-per-task=16', '--gres=gpu:3090:4',
            '--kill-on-bad-exit=1', '--cpu-bind=none', '--export=ALL',
            f'--output={root}/logs/node_%j_%N.log', f'--error={root}/logs/node_%j_%N.log',
            sys.executable, '-u', str(HERE / 'launcher.py'), 'node',
            '--root', str(root), '--code', str(code), '--nodes', nodes]


def submit(root, repo, code, nodes, action):
    root, repo, code = (Path(p).resolve() for p in (root, repo, code))
    count = len(parse_nodes(nodes))
    ensure_account_idle()
    verify_launcher()
    job_path = verify_data(code, root)
    require(not (root / 'formal/dataset/complete.json').exists(), 'Dataset already complete; use verify/summary')
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(G25_SINGLE_CODE=str(HERE), G25_CODE=str(code), G25_ROOT=str(root),
               G25_NODES=nodes, G25_JOB=str(job_path))
    cmd = sbatch_command(root, repo, nodes, os.environ.get('TIME_LIMIT', '3-00:00:00'))
    # test-only creates no job/allocation; it is not a promise that the job can run now.
    check = subprocess.run([cmd[0], '--test-only'] + cmd[1:], capture_output=True, text=True, env=env)
    print('[Slurm test-only]\n' + (check.stdout + check.stderr).strip(), flush=True)
    require(check.returncode == 0,
            'Slurm resource check rejected/failed. NO job submitted; no fallback and no second submission.')
    ensure_account_idle()
    stamp = time.time_ns()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    write(root / f'logs/submission_attempt_{stamp}.json', dict(
        command=cmd, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        nodes=nodes, action=action, launcher=str(HERE), collector=str(code)))
    require(result.returncode == 0, 'sbatch failed; no automatic retry: ' + result.stderr.strip())
    raw = result.stdout.strip().split(';')[0]
    require(re.fullmatch(r'\d+', raw) is not None,
            'Ambiguous sbatch reply: inspect squeue before retrying; a job may exist: ' + result.stdout)
    jid = int(raw)
    record = dict(action=action, submitted_ns=stamp, nodes_requested=nodes,
                  single_allocation=True, launcher=str(HERE), collector=str(code),
                  jobs=[dict(id=jid, node=nodes, nodes=count, gpus=4 * count,
                             log=str(root / f'logs/collect_{jid}.out'))])
    write(root / 'latest_submission.json', record)
    write(root / f'logs/submission_{stamp}.json', record)
    print(f'[submitted] job={jid} jobs=1 nodes={count} gpus={4 * count} nodelist={nodes}', flush=True)
    print('[out]', root / 'formal', flush=True)
    print('[policy] A pending node/QOS limit must be resolved before retrying; no automatic fallback.', flush=True)


def wait_file(path, timeout=120.):
    deadline = time.monotonic() + timeout
    while not Path(path).is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Shared-storage/lock handshake timed out: {path}')
        time.sleep(.1)


def shared_lock_check(root, jobid, rank, count):
    """Check that both allocated nodes see the same file and respect its flock."""
    if count == 1:
        return
    folder = Path(root) / 'locks' / f'single_job_{jobid}'
    folder.mkdir(parents=True, exist_ok=True)
    held, ack, done = (folder / n for n in ('held.json', 'ack.json', 'done.json'))
    with (folder / 'probe.lock').open('a+') as f:
        if rank == 0:
            require(not held.exists(), 'Stale job handshake; do not requeue this job ID')
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                write(held, {'job': jobid})
                wait_file(ack)
                require(read(ack).get('job') == jobid, 'Wrong shared-lock peer')
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            write(done, {'job': jobid})
        else:
            wait_file(held)
            require(read(held).get('job') == jobid, 'Wrong shared-lock owner')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
                raise RuntimeError('Shared filesystem does not enforce cross-node lock; STOP')
            write(ack, {'job': jobid})
            wait_file(done)
    print('[SHARED LOCK PASS] node rank=' + str(rank), flush=True)


def worker_spec(root, code, job_path, jobid, node_rank, local_rank, gpu):
    slot = 4 * node_rank + local_rank
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    log = Path(root) / 'logs' / f'worker_{jobid}_{slot}.log'
    cmd = [sys.executable, '-u', str(Path(code) / 'run.py'), 'worker',
           '--job', str(job_path), '--slot', str(slot)]
    return cmd, env, log


def stop_children(processes):
    for proc in processes:
        if proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
    for proc in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def node(root, code, nodes):
    verify_launcher()
    job_path = verify_data(code, root)
    count = len(parse_nodes(nodes))
    jobid = os.environ.get('SLURM_JOB_ID', '')
    rank = int(os.environ.get('SLURM_PROCID', '-1'))
    require(jobid.isdigit() and 0 <= rank < count, 'Must run as a task inside the one srun step')
    require(int(os.environ.get('SLURM_LOCALID', '-1')) == 0, 'Exactly one coordinator per node required')
    subprocess.run(['nvidia-smi'], check=False)
    subprocess.run([sys.executable, str(Path(code) / 'run.py'), 'admission', '--job', str(job_path)], check=True)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(visible) == len(set(visible)) == 4 and all(visible), 'Exactly four allocated GPU identifiers required')
    shared_lock_check(root, jobid, rank, count)
    processes, handles = [], []
    def interrupted(*_):
        raise KeyboardInterrupt('Allocated node interrupted; completed shards are retained')
    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        for local_rank, gpu in enumerate(visible):
            cmd, env, log = worker_spec(root, code, job_path, jobid, rank, local_rank, gpu)
            handle = log.open('a')
            handles.append(handle)
            processes.append(subprocess.Popen(cmd, env=env, stdout=handle,
                              stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            require(all(p.poll() in (None, 0) for p in processes), 'Collector failed; inspect worker logs')
            time.sleep(.5)
        require(all(p.returncode == 0 for p in processes), 'Collector failed; inspect worker logs')
    finally:
        stop_children(processes)
        for handle in handles:
            handle.close()
        signal.signal(signal.SIGTERM, old)
    print(f'[NODE COMPLETE] rank={rank} slots={4 * rank}..{4 * rank + 3}', flush=True)


def batch(root, code, nodes):
    verify_launcher()
    job_path = verify_data(code, root)
    count = len(parse_nodes(nodes))
    allocated = int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', '0')))
    require(os.environ.get('SLURM_JOB_ID') and allocated == count, 'Allocated node count differs from request')
    print(f'[ONE JOB] nodes={count}, GPUs/node=4, total={4 * count}', flush=True)
    proc = subprocess.Popen(srun_command(root, code, nodes), start_new_session=True)
    def interrupted(*_):
        raise KeyboardInterrupt('Batch interrupted; retain prepared job and complete shards')
    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        require(proc.wait() == 0, 'Node step failed; no merge; inspect node/worker logs')
    finally:
        stop_children([proc])
        signal.signal(signal.SIGTERM, old)
    # srun has joined all allocated nodes. Merge happens exactly once, after collection.
    for mode in ('merge', 'verify'):
        subprocess.run([sys.executable, str(Path(code) / 'run.py'), mode, '--job', str(job_path)], check=True)
    print('[COMPLETE] single allocation collected and verified the dataset', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('policy', 'freeze', 'submit', 'resume', 'batch', 'node'))
    parser.add_argument('--nodes', default='3090node2')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--code', type=Path)
    args = parser.parse_args()
    parse_nodes(args.nodes)
    if args.mode == 'policy':
        ensure_account_idle()
    elif args.mode == 'freeze':
        print(freeze(args.root))
    elif args.mode in ('submit', 'resume'):
        submit(args.root, args.repo, args.code, args.nodes, args.mode)
    else:
        {'batch': batch, 'node': node}[args.mode](args.root, args.code, args.nodes)


if __name__ == '__main__':
    main()
