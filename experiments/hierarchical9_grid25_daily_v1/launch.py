#!/usr/bin/env python3
"""One 24h allocation, multiple original Grid25 segments; frozen engine unchanged."""
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
FILES = ('launch.py', 'run.sh', 'test_daily.py')
SOFT_SECONDS = 23 * 3600 + 30 * 60
HARD_LIMIT = '1-00:00:00'
ALLOWED_NODES = ('3090node1', '3090node2')
ENGINE_BLOB = 'e695e1f357669bd91918ca47546d12bc93b4f46c'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def submission_lock(root):
    # Same lock as the existing four-hour launcher, not a separate job stream.
    with (Path(root) / '.fast_segment_entry.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another collection entry is active') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def empty_queue():
    user = pwd.getpwuid(os.getuid()).pw_name
    r = subprocess.run(['squeue', '-h', '-u', user, '-o', '%i %T %j'],
                       capture_output=True, text=True)
    require(r.returncode == 0, 'Cannot read Slurm queue; no submission')
    require(not r.stdout.strip(), 'Existing job; leave it running, submit after exit:\n' + r.stdout)


def previous_done(root):
    terminal = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
    for folder in ('fast_segments_v1', 'daily_v1'):
        p = Path(root) / folder / 'latest_submission.json'
        if not p.exists():
            continue
        jid = str(read(p)['job_id'])
        r = subprocess.run(['sacct', '-X', '-n', '-P', '-j', jid,
                            '--format=JobIDRaw,State%40'], capture_output=True, text=True)
        states = [row.split('|')[1] for row in r.stdout.splitlines()
                  if row.split('|')[0].strip() == jid and len(row.split('|')) >= 2]
        require(r.returncode == 0 and states and
                all(any(s.strip().startswith(t) for t in terminal) for s in states),
                f'Cannot confirm job {jid} has exited; no duplicate job')


def load_engine(root):
    root = Path(root).resolve()
    path = root / 'fast_segments_v1/manifest.json'
    require(path.is_file(), 'Existing fast manifest required; do not delete or recreate data')
    m = read(path)
    require(Path(m['root']).resolve() == root, 'Wrong data root')
    code = Path(m['source']).resolve()
    require(code.is_relative_to(root / 'fast_sources'), 'Fast snapshot outside expected root')
    require(blob(code / 'run.py') == ENGINE_BLOB, 'Unreviewed fast collector version')
    for name, digest in m['source_sha256'].items():
        require(Path(name).name == name and sha(code / name) == digest,
                'Frozen fast file changed: ' + name)
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('grid25_daily_frozen_engine', code / 'run.py')
    require(spec is not None and spec.loader is not None, 'Cannot load frozen engine')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    checked, common, _, _, job = engine.verify_manifest(path)
    return engine, checked, common, job


def own_hashes(directory):
    return {name: sha(Path(directory) / name) for name in FILES}


def freeze(root):
    hashes = own_hashes(HERE)
    ident = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    parent = Path(root) / 'daily_sources'
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / ident
    if not dest.exists():
        temp = Path(tempfile.mkdtemp(prefix='.building.', dir=parent))
        try:
            for name in FILES + ('README_zh.md',):
                shutil.copyfile(HERE / name, temp / name)
            write(temp / 'launcher_manifest.json', {'files': hashes})
            os.rename(temp, dest)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    require(own_hashes(dest) == hashes, 'Daily source snapshot mismatch')
    return dest


def sbatch_command(root, repo, source, node):
    require(node in ALLOWED_NODES, 'Use 3090node1 or 3090node2 only')
    return ['sbatch', '--parsable', '--partition=GPU', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=16', '--gres=gpu:3090:4', '--exclusive', '--no-requeue',
            '--time=' + HARD_LIMIT, '--signal=B:USR1@1800', '--nodelist=' + node,
            '--chdir=' + str(repo), '--export=ALL', '--job-name=g25_day',
            '--output=' + str(Path(root) / 'logs/daily_%j.out'),
            '--error=' + str(Path(root) / 'logs/daily_%j.out'),
            str(Path(source) / 'run.sh'), '_batch']


def submit(root, repo, node):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    require(node in ALLOWED_NODES and root.is_dir(), 'Invalid node or unprepared root')
    with submission_lock(root):
        empty_queue(); previous_done(root)
        engine, m, common, job = load_engine(root)
        if engine.choose_segment(m, common, job) is None:
            print('[READY TO MERGE] Nothing left to collect; no job submitted.', flush=True)
            return
        source = freeze(root)
        spec = root / 'daily_v1/submissions' / f'request_{time.time_ns()}.json'
        write(spec, dict(root=str(root), source=str(source), node=node,
                         launcher_hashes=own_hashes(source), soft_seconds=SOFT_SECONDS,
                         manifest_sha256=sha(root / 'fast_segments_v1/manifest.json'),
                         scope='Scheduling only; frozen sampling and existing data unchanged'))
        (root / 'logs').mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(G25D_SOURCE=str(source), G25D_SPEC=str(spec))
        cmd = sbatch_command(root, repo, source, node)
        check = subprocess.run([cmd[0], '--test-only'] + cmd[1:],
                               capture_output=True, text=True, env=env)
        print('[Slurm test-only]\n' + check.stdout + check.stderr, flush=True)
        require(check.returncode == 0, 'Resource test failed; no job, no fallback')
        empty_queue()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        attempt = dict(command=cmd, returncode=result.returncode, stdout=result.stdout,
                       stderr=result.stderr, spec=str(spec), node=node)
        write(spec.with_name(spec.stem + '_reply.json'), attempt)
        require(result.returncode == 0, 'Submission failed; no retry: ' + result.stderr)
        raw = result.stdout.strip().split(';')[0]
        require(re.fullmatch(r'\d+', raw) is not None,
                'Ambiguous reply: inspect squeue before retry; job may exist')
        record = dict(attempt, job_id=int(raw), log=str(root / f'logs/daily_{raw}.out'))
        write(root / 'daily_v1/latest_submission.json', record, replace=True)
        print(f'[submitted] job={raw} jobs=1 nodes=1 gpus=4 node={node} limit=24h', flush=True)
        print('[mode] Continue original segments in this allocation; stop new shards at 23h30m.', flush=True)


def worker_command(m, manifest, segment, rank, request):
    # Execute the exact existing worker, including its per-GPU parity gate.
    return [sys.executable, '-u', str(Path(m['source']) / 'run.py'), 'worker',
            '--manifest', str(manifest), '--segment', str(segment),
            '--rank', str(rank), '--request', str(request)]


def launch_segment(engine, m, common, job, folder, segment, deadline, stop, visible):
    folder.mkdir(parents=True, exist_ok=False)
    request = folder / 'request.json'
    manifest = Path(m['root']) / 'fast_segments_v1/manifest.json'
    write(request, dict(segment=segment, stop_file=str(stop), deadline_unix=deadline,
                        gpu_type='3090', manifest=str(manifest)))
    processes, handles = [], []
    try:
        for rank, gpu in enumerate(visible):
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
            handle = (folder / f'worker_{rank}.log').open('a'); handles.append(handle)
            processes.append(subprocess.Popen(
                worker_command(m, manifest, segment, rank, request), env=env,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            require(all(p.poll() in (None, 0) for p in processes),
                    'Worker or parity gate failed; no next segment, no merge')
            if time.time() >= deadline:
                stop.touch(exist_ok=True)
            time.sleep(1)
        require(all(p.returncode == 0 for p in processes), 'Worker failed; inspect daily ranks')
    finally:
        engine.terminate(processes)
        for handle in handles:
            handle.close()
    present = [s for s in m['segments'][segment] if engine.stage_exists(common, job, s)]
    engine.engine_integrity(common, job, present, m)
    complete = len(present) == len(m['segments'][segment])
    result = dict(status='SEGMENT_COMPLETE' if complete else 'SEGMENT_PAUSED',
                  segment=segment, complete_shards=len(present),
                  expected_shards=len(m['segments'][segment]), job_id=os.environ['SLURM_JOB_ID'])
    write(folder / 'result.json', result)
    print(json.dumps(result), flush=True)
    return complete


def run_window(engine, m, common, job, folder, visible, deadline, stop):
    completed = []
    while not stop.exists() and time.time() < deadline:
        segment = engine.choose_segment(m, common, job)
        if segment is None:
            break
        print(f'[SEGMENT START] {segment + 1}/{len(m["segments"])}; same Slurm job', flush=True)
        complete = launch_segment(engine, m, common, job, folder / f'segment_{segment:03d}',
                                  segment, deadline, stop, visible)
        if not complete:
            require(stop.exists() or time.time() >= deadline,
                    'Unexpected incomplete segment; refusing silent retry')
            break
        completed.append(segment)
    all_done = engine.choose_segment(m, common, job) is None
    result = dict(status='READY_TO_MERGE' if all_done else 'WINDOW_PAUSED',
                  completed_segments_this_job=completed, job_id=os.environ['SLURM_JOB_ID'])
    write(folder / 'result.json', result)
    print(json.dumps(result), flush=True)
    print('[EXIT] Published shards retained. No next Slurm job or merge submitted.', flush=True)
    return result


def batch(spec_path):
    started = time.time()  # Runtime budget starts now, never at submission/queue time.
    spec = read(spec_path); root = Path(spec['root'])
    require(spec['soft_seconds'] == SOFT_SECONDS, 'Unexpected runtime budget')
    require(Path(spec['source']).resolve() == HERE and own_hashes(HERE) == spec['launcher_hashes'],
            'Run the frozen daily launcher; source changed')
    require(read(HERE / 'launcher_manifest.json')['files'] == spec['launcher_hashes'],
            'Corrupt launcher manifest')
    require(sha(root / 'fast_segments_v1/manifest.json') == spec['manifest_sha256'],
            'Original segment manifest changed')
    jid = os.environ.get('SLURM_JOB_ID', '')
    require(jid.isdigit() and int(os.environ.get('SLURM_JOB_NUM_NODES',
            os.environ.get('SLURM_NNODES', '0'))) == 1, 'One Slurm node required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(visible) == len(set(visible)) == 4 and all(visible), 'Four allocated GPUs required')
    engine, m, common, job = load_engine(root)
    folder = root / 'daily_v1/runs' / jid
    folder.mkdir(parents=True, exist_ok=False)
    stop = folder / 'STOP'
    deadline = started + SOFT_SECONDS
    write(folder / 'window.json', dict(start_unix=started, deadline_unix=deadline,
                                      spec=str(spec_path), hard_limit=HARD_LIMIT))
    def pause(*_):
        stop.touch(exist_ok=True)
        print('[STOP REQUESTED] Finish current shards; do not claim new shards.', flush=True)
    handlers = {sig: signal.signal(sig, pause) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        print('[ONE JOB] 3090 x4; cross segment boundaries; 24h allocation / 23h30m claim deadline', flush=True)
        return run_window(engine, m, common, job, folder, visible, deadline, stop)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def diagnostics(root, mode):
    root = Path(root)
    engine, m, common, job = load_engine(root)
    if mode in ('summary', 'verify'):
        rows = [common.stage(job, s, mode == 'verify')
                for ids in m['segments'] for s in ids if engine.stage_exists(common, job, s)]
        points = sum(r['points'] for r in rows); total = job['cfg']['grid_points']
        finished = sum(all(engine.stage_exists(common, job, s) for s in ids) for ids in m['segments'])
        print(f'Completed shards: {len(rows)}/{sum(map(len,m["segments"]))}; points={points}/{total}')
        print(f'Segments complete: {finished}/{len(m["segments"])}; one daily job may advance multiple segments')
        fast = [r for r in rows if r.get('producer', {}).get('engine') == 'target_only_adam_reuse_forward_v1']
        if fast:
            sp = sum(r['elapsed_seconds'] for r in fast) / sum(r['points'] for r in fast)
            print(f'Fast mean={sp:.3f} seconds/point/worker; remaining sampling@4={(total-points)*sp/14400:.2f} h (no queue/merge)')
        if mode == 'verify':
            ids = [s for part in m['segments'] for s in part if engine.stage_exists(common, job, s)]
            engine.engine_integrity(common, job, ids, m)
            print('[VERIFIED] Published shards and original engine provenance')
        return
    record_path = root / 'daily_v1/latest_submission.json'
    require(record_path.exists(), 'No daily job submitted')
    record = read(record_path); jid = str(record['job_id'])
    if mode == 'status':
        subprocess.run(['squeue', '-j', jid, '-o', '%.12i %.12T %.14M %.14l %.24R'], check=False)
        subprocess.run(['sacct', '-X', '-j', jid, '--format=JobID,State,ExitCode,Elapsed,NodeList'], check=False)
        print('[log]', record['log']); return
    if mode == 'log':
        paths = [Path(record['log'])]
    elif mode == 'ranks':
        parts = sorted((root / 'daily_v1/runs' / jid).glob('segment_*'))
        paths = sorted(parts[-1].glob('worker_*.log')) if parts else []
        if not paths:
            print('[worker logs not created yet]')
    else:
        import tarfile
        dest = root / f'daily_review_{time.time_ns()}.tar.gz'
        with tarfile.open(dest, 'x:gz') as archive:
            for directory in (root/'daily_v1', root/'daily_sources', root/'fast_segments_v1', root/'fast_sources', root/'logs'):
                for p in sorted(directory.rglob('*')):
                    if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.py','.sh','.md','.out','.log'):
                        archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
        print('[pack]', dest, sha(dest)); return
    for path in paths:
        print('===', path, '===', flush=True)
        if path.exists():
            subprocess.run(['tail', '-n', '30', str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('policy', 'next', 'submit', 'resume', 'batch',
                                         'status', 'summary', 'verify', 'ranks', 'log', 'pack'))
    parser.add_argument('--root', type=Path); parser.add_argument('--repo', type=Path)
    parser.add_argument('--node', default='3090node2'); parser.add_argument('--spec', type=Path)
    args = parser.parse_args()
    if args.mode == 'policy':
        require(args.node in ALLOWED_NODES, 'Only 3090node1 or 3090node2'); empty_queue()
    elif args.mode in ('next', 'submit', 'resume'):
        submit(args.root, args.repo, args.node)
    elif args.mode == 'batch':
        batch(args.spec)
    else:
        diagnostics(args.root, args.mode)


if __name__ == '__main__':
    main()
