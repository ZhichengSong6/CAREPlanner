#!/usr/bin/env python3
"""Recover interrupted Grid25 output; one allocation, two nodes, eight GPUs.

Only orchestration changes. Published shards, numerical engine, seeds and
qualification reports remain immutable. Cleanup quarantines, never deletes.
"""
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
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
FILES = ('launch.py', 'run.sh', 'test_recovery.py')
STATE = 'two_node_v1'
ENGINE_BLOB = 'e695e1f357669bd91918ca47546d12bc93b4f46c'
SOFT_SECONDS = 23 * 3600 + 30 * 60
TERMINAL = ('COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + p.name + '.', dir=p.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(name, p)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def lock(p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Live lock; do not unlink it: ' + str(p)) from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def idle_locks(root):
    # Interoperate with all previous submitters. Lock files are not deleted.
    with contextlib.ExitStack() as stack:
        for name in ('.fast_segment_entry.lock', '.single_job_entry.lock',
                     '.submit.lock', '.grid25_allocation.lock'):
            stack.enter_context(lock(Path(root) / name))
        yield


def nodes_arg(text):
    nodes = text.split(',')
    require(len(nodes) == 2 and len(set(nodes)) == 2 and
            all(re.fullmatch(r'3090node[1-3]', n) for n in nodes),
            'NODES requires exactly two distinct 3090 nodes; default 3090node1,3090node2')
    return nodes


def empty_queue():
    user = pwd.getpwuid(os.getuid()).pw_name
    r = subprocess.run(['squeue', '-h', '-u', user, '-o', '%i %T %j'],
                       capture_output=True, text=True)
    require(r.returncode == 0, 'Cannot read Slurm queue; no cleanup/submission: ' + r.stderr)
    require(not r.stdout.strip(), 'Jobs still queued/running; NOT cancelling or cleaning:\n' + r.stdout)


def recorded_jobs(root):
    """Preserve history. Missing accounting after reboot is not a completed job."""
    ids = set()
    for rel in ('latest_submission.json', 'fast_segments_v1/latest_submission.json',
                'daily_v1/latest_submission.json', STATE + '/latest_submission.json'):
        p = Path(root) / rel
        if not p.is_file():
            continue
        record = read(p)
        if 'job_id' in record:
            ids.add(int(record['job_id']))
        ids.update(int(j['id']) for j in record.get('jobs', []))
    history = []
    for jid in sorted(ids):
        r = subprocess.run(['sacct', '-X', '-n', '-P', '-j', str(jid),
                            '--format=JobIDRaw,State%40'], capture_output=True, text=True)
        states = [line.split('|')[1].strip() for line in r.stdout.splitlines()
                  if len(line.split('|')) > 1 and line.split('|')[0].strip() == str(jid)]
        # Fail on explicitly nonterminal accounting, not on an absent old record.
        require(not states or all(any(s.startswith(t) for t in TERMINAL) for s in states),
                f'Inconsistent nonterminal accounting for {jid}: {states}; inspect before recovery')
        history.append(dict(job_id=jid, states=states, returncode=r.returncode,
                            note='' if states else 'ACCOUNTING_UNAVAILABLE; queue and file locks checked separately'))
    return history


def load_engine(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Existing real data root required; never recreate it')
    root = root.resolve()
    p = root / 'fast_segments_v1/manifest.json'; m = read(p)
    require(Path(m['root']).resolve() == root, 'Wrong root in original manifest')
    code = Path(m['source']).resolve()
    require(code.is_relative_to(root / 'fast_sources'), 'Unexpected frozen engine path')
    b = (code / 'run.py').read_bytes()
    require(hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == ENGINE_BLOB,
            'Unreviewed engine; not patching sampling code')
    for name, digest in m['source_sha256'].items():
        require(Path(name).name == name and sha(code / name) == digest, 'Frozen engine changed: ' + name)
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location('grid25_recovery_engine', code / 'run.py')
    require(spec is not None and spec.loader is not None, 'Cannot load engine')
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    checked, common, _, _, job = engine.verify_manifest(p)
    require(Path(job['out']).resolve() == root / 'formal', 'Wrong original job output')
    return engine, checked, common, job


def probe_shard_locks(root, total):
    for sid in range(total):
        with lock(Path(root) / 'formal/locks' / f'shard_{sid:05d}.lock'):
            pass


def scan_shards(root, engine, m, c, job):
    """Validate every published shard; only recognizable unpublished dirs move."""
    root = Path(root); total = c.nshards(job['cfg']); good = []; candidates = []
    for p in sorted((root / 'formal/shards').iterdir()):
        require(not p.is_symlink(), 'Symlink in shards; manual inspection required: ' + str(p))
        if re.fullmatch(r'\d{5}', p.name):
            sid = int(p.name)
            require(p.is_dir() and 0 <= sid < total, 'Unexpected canonical shard: ' + str(p))
            if not (p / 'complete.json').is_file():
                candidates.append((p, sid, 'unpublished canonical directory'))
            else:
                require(not (p/'data.npz').is_symlink() and not (p/'complete.json').is_symlink(), 'Symlink in completed shard')
                good.append(sid)
        elif re.fullmatch(r'\.\d{5}\.[A-Za-z0-9_-]+', p.name):
            sid = int(p.name.split('.')[1])
            require(p.is_dir() and 0 <= sid < total, 'Unexpected temporary shard')
            candidates.append((p, sid, 'unpublished temporary directory'))
        else:
            print('[untouched unknown entry]', p, flush=True)
    # Corrupt published data stop recovery BEFORE moving anything.
    engine.engine_integrity(c, job, good, m)
    for p in sorted((root / 'formal').glob('.dataset.*')):
        require(p.is_dir() and not p.is_symlink(), 'Unexpected merge temporary entry')
        candidates.append((p, None, 'unpublished merge directory'))
    points = sum(c.stage(job, sid, False)['points'] for sid in good)
    return good, points, candidates


def quarantine(root, candidates):
    """Atomic rename within the same filesystem; no rm and no released-lock unlink."""
    root = Path(root); dest = root / STATE / 'recovery' / str(time.time_ns())
    moved = []
    for p, sid, reason in candidates:
        p = Path(p)
        require(not p.is_symlink() and p.is_dir(), 'Cleanup target changed')
        expected = root / 'formal' if sid is None else root / 'formal/shards'
        require(p.parent.resolve() == expected.resolve(), 'Cleanup target escaped its directory')
        with lock(root/'formal/locks' / ('merge.lock' if sid is None else f'shard_{sid:05d}.lock')):
            if p.name.isdigit():
                require(not (p/'complete.json').exists(), 'Shard became complete; do not move it')
            dest.mkdir(parents=True, exist_ok=True)
            target = dest / p.name
            require(not target.exists(), 'Quarantine name collision')
            os.rename(p, target)
            moved.append(dict(from_path=str(p), to_path=str(target), reason=reason))
            write(dest/'moves.json', moved)
    return moved


def recover(root):
    root = Path(root).resolve()
    with idle_locks(root):
        empty_queue(); history = recorded_jobs(root)
        engine, m, c, job = load_engine(root)
        probe_shard_locks(root, c.nshards(job['cfg']))
        good, points, candidates = scan_shards(root, engine, m, c, job)
        empty_queue()
        moved = quarantine(root, candidates)
        report = dict(status='RECOVERED', timestamp_ns=time.time_ns(), preserved_shards=len(good),
                      preserved_points=points, quarantined=moved, jobs=history,
                      manifest_sha256=sha(root/'fast_segments_v1/manifest.json'),
                      base_job_sha256=sha(root/'formal/job.json'),
                      note='No published data, logs, gates, checkpoints, manifests or lock files deleted')
        write(root/STATE/'recovery_latest.json', report)
    print(f'[RECOVERED] preserved={len(good)} shards/{points} points; quarantined={len(moved)}; deleted=0', flush=True)


def own_hashes(directory=HERE):
    return {n: sha(Path(directory)/n) for n in FILES}


def freeze(root):
    hashes = own_hashes(); ident = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    parent = Path(root)/'two_node_sources'; parent.mkdir(parents=True, exist_ok=True)
    dest = parent/ident
    if not dest.exists():
        tmp = Path(tempfile.mkdtemp(prefix='.building.', dir=parent))
        try:
            for n in FILES + ('README_zh.md',):
                shutil.copyfile(HERE/n, tmp/n)
            write(tmp/'launcher_manifest.json', hashes)
            os.rename(tmp, dest)
        finally:
            if tmp.exists(): shutil.rmtree(tmp)
    require(own_hashes(dest) == hashes, 'Launcher snapshot changed')
    return dest


def sbatch_command(root, repo, source, nodes):
    nodes_arg(nodes)
    return ['sbatch', '--parsable', '--partition=GPU', '--nodes=2', '--ntasks=2',
            '--ntasks-per-node=1', '--cpus-per-task=16', '--gres=gpu:3090:4',
            '--exclusive', '--no-requeue', '--time=1-00:00:00', '--signal=B:USR1@1800',
            '--nodelist='+nodes, '--chdir='+str(repo), '--export=ALL', '--job-name=g25_two',
            '--output='+str(Path(root)/'logs/two_%j.out'),
            '--error='+str(Path(root)/'logs/two_%j.out'), str(Path(source)/'run.sh'), '_batch']


def srun_command(spec_path, step):
    return ['srun', '--mpi=none', '--nodes=2', '--ntasks=2', '--ntasks-per-node=1',
            '--cpus-per-task=16', '--gres=gpu:3090:4', '--cpu-bind=none',
            '--kill-on-bad-exit=1', '--export=ALL',
            '--output='+str(Path(step)/'node_%N.log'), '--error='+str(Path(step)/'node_%N.log'),
            sys.executable, '-u', str(HERE/'launch.py'), 'node',
            '--spec', str(spec_path), '--step', str(step)]


def submit(root, repo, nodes, draft=False):
    root, repo = Path(root).resolve(), Path(repo).resolve(); nodes_arg(nodes)
    with idle_locks(root):
        empty_queue(); recorded_jobs(root)
        engine, m, c, job = load_engine(root)
        require((root/STATE/'recovery_latest.json').is_file(), 'Run recover first; do not clear the data root')
        recovery = read(root/STATE/'recovery_latest.json')
        require(recovery['base_job_sha256'] == sha(root/'formal/job.json') and
                recovery['manifest_sha256'] == sha(root/'fast_segments_v1/manifest.json'), 'Recovery identity changed')
        probe_shard_locks(root, c.nshards(job['cfg']))
        # Recheck published hashes/provenance; never silently accept a corrupt completed shard.
        _, _, leftovers = scan_shards(root, engine, m, c, job)
        require(not leftovers, 'Unpublished data remain; run recover after all jobs have exited')
        if engine.choose_segment(m, c, job) is None:
            print('[READY TO MERGE] No collection submitted', flush=True); return
        source = freeze(root); cmd = sbatch_command(root, repo, source, nodes)
        if draft:
            print('[DRAFT ONLY] one sbatch; two nodes x four GPUs; run next to submit')
            print(shlex.join(cmd)); return
        path = root/STATE/'submissions'/f'request_{time.time_ns()}.json'
        spec = dict(root=str(root), source=str(source), nodes=nodes, launcher_hashes=own_hashes(source),
                    fast_manifest_sha256=sha(root/'fast_segments_v1/manifest.json'), soft_seconds=SOFT_SECONDS)
        write(path, spec); (root/'logs').mkdir(exist_ok=True)
        env = os.environ.copy(); env.update(G25R_SOURCE=str(source), G25R_SPEC=str(path), G25R_SPEC_SHA=sha(path))
        check = subprocess.run([cmd[0], '--test-only']+cmd[1:], capture_output=True, text=True, env=env)
        print('[Slurm test-only]\n'+check.stdout+check.stderr, flush=True)
        require(check.returncode == 0, 'Resource check failed; no real submission, no fallback')
        empty_queue()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        record = dict(spec=str(path), command=cmd, returncode=result.returncode,
                      stdout=result.stdout, stderr=result.stderr, nodes=nodes)
        write(path.with_name(path.stem+'_reply.json'), record)
        require(result.returncode == 0, 'sbatch failed; no automatic retry: '+result.stderr)
        raw = result.stdout.strip().split(';')[0]
        require(raw.isdigit(), 'Ambiguous reply; inspect squeue before retrying: '+result.stdout)
        record.update(job_id=int(raw), log=str(root/f'logs/two_{raw}.out'))
        write(root/STATE/'latest_submission.json', record)
        print(f'[submitted] job={raw} jobs=1 nodes=2 gpus=8 nodelist={nodes} limit=24h', flush=True)


def check_spec(path):
    spec = read(path); root = Path(spec['root'])
    require(Path(spec['source']).resolve() == HERE and own_hashes() == spec['launcher_hashes'], 'Launcher changed')
    require(read(HERE/'launcher_manifest.json') == spec['launcher_hashes'], 'Snapshot identity changed')
    require(sha(root/'fast_segments_v1/manifest.json') == spec['fast_manifest_sha256'], 'Original manifest changed')
    require(spec['soft_seconds'] == SOFT_SECONDS, 'Window changed')
    nodes_arg(spec['nodes'])
    return spec, root


def wait_file(path, timeout=180.):
    until = time.monotonic()+timeout
    while not Path(path).exists():
        require(time.monotonic() < until, 'Shared-filesystem handshake timed out: '+str(path))
        time.sleep(.1)


def shared_probe(folder, rank, token):
    """Prove cross-host file visibility AND actual flock exclusion before workers."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    held, ack, done = (folder/n for n in ('held.json', 'ack.json', 'done.json'))
    with (folder/'probe.lock').open('a+') as f:
        if rank == 0:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                write(held, {'token': token}); wait_file(ack)
                require(read(ack)['token'] == token, 'Wrong handshake peer')
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
            write(done, {'token': token})
        else:
            wait_file(held); require(read(held)['token'] == token, 'Wrong handshake owner')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
                raise RuntimeError('Cross-node flock exclusion failed; no data collection')
            write(ack, {'token': token}); wait_file(done)
    print(f'[SHARED LOCK PASS] node_rank={rank}', flush=True)


def worker_spec(m, root, step, node_rank, local_rank, gpu, request):
    slot = 4*node_rank+local_rank
    cmd = [sys.executable, '-u', str(Path(m['source'])/'run.py'), 'worker',
           '--manifest', str(Path(root)/'fast_segments_v1/manifest.json'),
           '--segment', str(read(request)['segment']), '--rank', str(local_rank), '--request', str(request)]
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpu
    return cmd, env, Path(step)/f'node_{node_rank}'/f'worker_{slot}.log'


def node(spec_path, step):
    spec, root = check_spec(spec_path); engine, m, c, job = load_engine(root)
    rank = int(os.environ.get('SLURM_PROCID', '-1')); jid = os.environ.get('SLURM_JOB_ID', '')
    require(rank in (0, 1) and os.environ.get('SLURM_LOCALID') == '0' and jid.isdigit(), 'One coordinator per allocated node required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    require(len(visible) == len(set(visible)) == 4 and all(visible), 'Four Slurm-visible GPUs per node required')
    step = Path(step); info = read(step/'step.json')
    shared_probe(step/'shared_probe', rank, jid+':'+str(info['segment']))
    folder = step/f'node_{rank}'; folder.mkdir(exist_ok=False)
    request = folder/'request.json'
    write(request, dict(segment=info['segment'], stop_file=info['stop_file'],
                        deadline_unix=info['deadline_unix'], gpu_type='3090',
                        manifest=str(root/'fast_segments_v1/manifest.json')))
    processes, handles = [], []
    stop = Path(info['stop_file'])
    def pause(*_): stop.touch(exist_ok=True)
    old = signal.signal(signal.SIGTERM, pause)
    try:
        for local, gpu in enumerate(visible):
            cmd, env, log = worker_spec(m, root, step, rank, local, gpu, request)
            handle = log.open('a'); handles.append(handle)
            processes.append(subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            require(all(p.poll() in (None, 0) for p in processes), 'Worker failed; preserve completed shards')
            if time.time() >= info['deadline_unix']: stop.touch(exist_ok=True)
            time.sleep(.5)
        require(all(p.returncode == 0 for p in processes), 'Worker failed; no next segment')
    finally:
        engine.terminate(processes)
        for handle in handles: handle.close()
        signal.signal(signal.SIGTERM, old)
    print(f'[NODE DONE] node_rank={rank} global_slots={4*rank}..{4*rank+3}', flush=True)


def window(spec_path, folder, deadline, stop, engine, m, c, job):
    completed = []
    while not stop.exists() and time.time() < deadline:
        seg = engine.choose_segment(m, c, job)
        if seg is None: break
        step = folder/f'segment_{seg:03d}'; step.mkdir()
        write(step/'step.json', dict(segment=seg, deadline_unix=deadline, stop_file=str(stop)))
        print(f'[SEGMENT START] {seg+1}/{len(m["segments"])}; SAME JOB, TWO NODES, EIGHT GPUs', flush=True)
        process = subprocess.Popen(srun_command(spec_path, step), start_new_session=True)
        try:
            require(process.wait() == 0, 'srun failed; no next step or merge; published shards preserved')
        finally:
            engine.terminate([process])
        present = [sid for sid in m['segments'][seg] if engine.stage_exists(c, job, sid)]
        engine.engine_integrity(c, job, present, m)
        complete = len(present) == len(m['segments'][seg])
        write(step/'result.json', dict(status='SEGMENT_COMPLETE' if complete else 'SEGMENT_PAUSED',
                                      segment=seg, complete_shards=len(present)))
        if not complete:
            require(stop.exists() or time.time() >= deadline, 'Unexpected missing shards; no silent retry')
            break
        completed.append(seg)
    result = dict(status='READY_TO_MERGE' if engine.choose_segment(m, c, job) is None else 'WINDOW_PAUSED',
                  completed_segments_this_job=completed, job_id=os.environ['SLURM_JOB_ID'])
    write(folder/'result.json', result); print(json.dumps(result), flush=True)


def batch(spec_path):
    start = time.time(); spec, root = check_spec(spec_path)
    require(sha(spec_path) == os.environ.get('G25R_SPEC_SHA'), 'Submission spec changed')
    jid = os.environ.get('SLURM_JOB_ID', '')
    require(jid.isdigit() and int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', '0'))) == 2,
            'Expected exactly one allocation containing two nodes')
    with lock(root/'.grid25_allocation.lock'):
        engine, m, c, job = load_engine(root)
        folder = root/STATE/'runs'/jid; folder.mkdir(parents=True, exist_ok=False)
        stop = folder/'STOP'; deadline = start + SOFT_SECONDS
        write(folder/'window.json', dict(start_unix=start, deadline_unix=deadline, nodes=spec['nodes'], gpus=8))
        def pause(*_):
            stop.touch(exist_ok=True)
            print('[STOP REQUESTED] Finish current shards; no new shards', flush=True)
        old = {s: signal.signal(s, pause) for s in (signal.SIGUSR1, signal.SIGTERM)}
        try:
            print('[ONE JOB] nodes=2 GPUs/node=4 total=8; original frozen fast engine', flush=True)
            window(spec_path, folder, deadline, stop, engine, m, c, job)
        finally:
            for s, h in old.items(): signal.signal(s, h)


def diagnostics(root, mode):
    root = Path(root); engine, m, c, job = load_engine(root)
    if mode in ('summary', 'verify'):
        ids = [s for part in m['segments'] for s in part if engine.stage_exists(c, job, s)]
        rows = [c.stage(job, s, False) for s in ids]
        n = sum(r['points'] for r in rows); total = job['cfg']['grid_points']
        print(f'Completed shards: {len(ids)}/{c.nshards(job["cfg"])}; points={n}/{total}')
        fast = [r for r in rows if r.get('producer', {}).get('engine') == 'target_only_adam_reuse_forward_v1']
        if fast:
            seconds = sum(r['elapsed_seconds'] for r in fast)/sum(r['points'] for r in fast)
            print(f'Mean={seconds:.3f} sec/point/worker; remaining@8={(total-n)*seconds/28800:.2f}h (assumes sustained 8 workers; no queue/merge)')
        if mode == 'verify':
            engine.engine_integrity(c, job, ids, m)
            print('[VERIFIED] published shards; this is not final merged-dataset verification')
        return
    rec = read(root/STATE/'latest_submission.json'); jid = str(rec['job_id'])
    if mode == 'status':
        subprocess.run(['squeue', '-j', jid, '-o', '%.12i %.12T %.14M %.14l %.6D %.40R'], check=False)
        subprocess.run(['sacct', '-X', '-j', jid, '--format=JobID,State,ExitCode,Elapsed,NodeList'], check=False)
        return
    paths = [Path(rec['log'])] if mode == 'log' else []
    if mode == 'ranks':
        steps = sorted((root/STATE/'runs'/jid).glob('segment_*'))
        if steps: paths = sorted(steps[-1].glob('node_*/worker_*.log')) + sorted(steps[-1].glob('node_*.log'))
    for p in paths:
        print('===', p, '===', flush=True)
        if p.is_file(): subprocess.run(['tail', '-n', '18', str(p)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('recover', 'draft', 'next', 'resume', 'batch', 'node', 'status', 'summary', 'verify', 'ranks', 'log'))
    parser.add_argument('--root', type=Path); parser.add_argument('--repo', type=Path)
    parser.add_argument('--nodes', default='3090node1,3090node2')
    parser.add_argument('--spec', type=Path); parser.add_argument('--step', type=Path)
    a = parser.parse_args()
    if a.mode == 'recover': recover(a.root)
    elif a.mode in ('draft', 'next', 'resume'): submit(a.root, a.repo, a.nodes, a.mode == 'draft')
    elif a.mode == 'batch': batch(a.spec)
    elif a.mode == 'node': node(a.spec, a.step)
    else: diagnostics(a.root, a.mode)


if __name__ == '__main__':
    main()
