#!/usr/bin/env python3
"""Grid25: one Slurm allocation, 1-2 nodes, four single-GPU tasks per node.

Reuse the reviewed recovery code and frozen collector. Never change old shards,
manifests, seeds, sampling parameters or numerical code. No DDP/NCCL required.
"""
from __future__ import annotations
import argparse
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
import socket
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
STATE = 'slurm_tasks_v2'
BASE_BLOB = 'bd10c9bd07f0b021c9aa99b2c083f44eaa2c0d90'
DEFAULT_ROOT = '/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_grid25_data_v1'
BLOCKED = {'AssocMaxNodePerJobLimit', 'QOSMaxNodePerJobLimit', 'PartitionNodeLimit'}
# Practical sentinel for the immutable worker's numeric deadline_unix field.
# Slurm --time=0 and the batch loop control actual unlimited operation.
UNLIMITED_DEADLINE_UNIX = 253402300799.0  # 9999-12-31T23:59:59 UTC



def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def call(args, **kwargs):
    return subprocess.run(args, capture_output=True, text=True, timeout=60, **kwargs)


def base():
    path = HERE / ('recovery.py' if (HERE/'recovery.py').exists() else 'launch.py')
    content = path.read_bytes()
    digest = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    require(digest == BASE_BLOB, 'Recovery source changed; review it instead of bypassing the hash')
    spec = importlib.util.spec_from_file_location('grid25_reviewed_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fields(text):
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)', text.strip()))


def gpu_token(visible, local):
    tokens = visible.split(',')
    require(0 <= local < 4 and all(tokens) and len(tokens) == len(set(tokens)), 'Invalid allocated GPU list')
    require(len(tokens) in (1, 4), 'Expected one task-bound GPU or four node-visible allocated GPUs')
    return tokens[0] if len(tokens) == 1 else tokens[local]


def resources(mode, nodelist='', cap=None):
    require(mode in ('auto', '1', '2'), 'GRID25_NODES must be auto, 1 or 2')
    selected = ('1' if cap == 1 else '1-2') if mode == 'auto' else mode
    if nodelist:
        nodes = nodelist.split(',')
        require(mode != 'auto', 'A required nodelist forces all listed nodes; omit GRID25_NODELIST in auto mode')
        require(len(nodes) == int(mode) and len(set(nodes)) == len(nodes) and
                all(re.fullmatch(r'3090node[1-3]', n) for n in nodes), 'Wrong explicit 3090 nodelist')
    return selected


def association_hint(account, partition):
    """A conservative hint, NOT a full resolver for account/QOS inheritance."""
    user = pwd.getpwuid(os.getuid()).pw_name
    cmd = ['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user='+user,
           'format=Cluster%100,Account%100,Partition%100,MaxTRESPJ%500']
    try:
        result = call(cmd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, {'warning': str(exc), 'command': cmd}
    caps = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            row = [x.strip() for x in line.split('|')]
            if len(row) < 4 or (account and row[1] != account) or row[2] not in ('', partition):
                continue
            match = re.search(r'(?:^|,)node=(\d+)(?:,|$)', row[3])
            if match:
                caps.append(int(match.group(1)))
    require(not caps or min(caps) > 0, 'Visible association permits zero nodes; no submission')
    hint = 1 if caps and min(caps) == 1 else None
    return hint, dict(command=cmd, returncode=result.returncode, stdout=result.stdout,
                      stderr=result.stderr, conservative_single_node=hint == 1,
                      note='Strictest visible matching association only; not proof of permission or start time')


def batch_text(source, repo, root, nodes, nodelist, account, qos, partition, unlimited=False):
    # CPU count follows the supplied site example: 4 * 16 = 64 per node.
    opts = ['--job-name=g25_tasks', '--nodes='+nodes, '--ntasks-per-node=4',
            '--cpus-per-task=16', '--gres=gpu:3090:4', '--partition='+partition,
            '--time=0' if unlimited else '--time=1-00:00:00',
            '--no-requeue',
            '--chdir='+shlex.quote(str(repo)), '--output='+shlex.quote(str(root/'logs/tasks_%j.out')),
            '--error='+shlex.quote(str(root/'logs/tasks_%j.err'))]
    if not unlimited:
        opts.append('--signal=B:USR1@1800')
    for key, value in (('nodelist', nodelist), ('account', account), ('qos', qos)):
        if value:
            require(re.fullmatch(r'[A-Za-z0-9_,.-]+', value), 'Invalid '+key)
            opts.append('--'+key+'='+value)
    require(re.fullmatch(r'[A-Za-z0-9_.-]+', partition), 'Invalid partition')
    return ('#!/bin/bash\n'+'\n'.join('#SBATCH '+x for x in opts)+'\nset -euo pipefail\n'
            ': "${VIS_PYTHON:?}" "${G25T_SPEC:?}"\n'
            'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1\n'
            'exec "$VIS_PYTHON" -u '+shlex.quote(str(source/'slurm_tasks.py'))+' batch --spec "$G25T_SPEC"\n')


def step_command(source, spec, step, nodes):
    require(nodes in (1, 2), 'Invalid actual node count')
    return ['srun', '--mpi=none', '--nodes='+str(nodes), '--ntasks='+str(4*nodes),
            '--ntasks-per-node=4', '--cpus-per-task=16', '--gres=gpu:3090:4',
            '--gpu-bind=none', '--cpu-bind=cores', '--kill-on-bad-exit=1', '--export=ALL',
            '--output='+str(step/'rank_%t.out'), '--error='+str(step/'rank_%t.err'),
            sys.executable, '-u', str(source/'slurm_tasks.py'), 'task', '--spec', str(spec), '--step', str(step)]


def freeze(b, root):
    inputs = {'slurm_tasks.py': HERE/'slurm_tasks.py', 'recovery.py': Path(b.__file__)}
    hashes = {name: b.sha(p) for name, p in inputs.items()}
    ident = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    parent = root/'slurm_tasks_sources'; parent.mkdir(exist_ok=True)
    dest = parent/ident
    if not dest.exists():
        tmp = Path(tempfile.mkdtemp(prefix='.building.', dir=parent))
        try:
            for name, path in inputs.items(): shutil.copyfile(path, tmp/name)
            b.write(tmp/'launcher_manifest.json', hashes)
            os.rename(tmp, dest)
        finally:
            if tmp.exists(): shutil.rmtree(tmp)
    require(all(b.sha(dest/name) == h for name, h in hashes.items()), 'Launcher snapshot changed')
    return dest, hashes


def load_spec(b, path):
    spec = b.read(path); root = Path(spec['root'])
    require(Path(spec['source']).resolve() == HERE, 'Run from the frozen launcher')
    require(b.read(HERE/'launcher_manifest.json') == spec['hashes'], 'Snapshot manifest changed')
    require(all(b.sha(HERE/n) == h for n, h in spec['hashes'].items()), 'Snapshot changed')
    require(b.sha(root/'fast_segments_v1/manifest.json') == spec['engine_manifest_sha'], 'Engine manifest changed')
    require(b.sha(path) == os.environ.get('G25T_SPEC_SHA'), 'Submission spec changed')
    return spec, root


def history(b, root):
    old = b.STATE
    try:
        b.recorded_jobs(root)
        b.STATE = STATE
        b.recorded_jobs(root)
    finally:
        b.STATE = old


def status(b, root):
    record = b.read(root/STATE/'latest_submission.json'); jid = str(record['job_id'])
    r = call(['squeue', '-h', '-j', jid, '-o', '%i|%T|%R'])
    print('[queue]\n'+r.stdout+r.stderr, flush=True)
    if any(reason in r.stdout for reason in BLOCKED):
        print('[BLOCKED BY POLICY] Not running. No automatic cancellation or second submission. '
              'Use cancel-blocked, then GRID25_NODES=1 next; do not change account/QOS without authorization.', flush=True)
    elif 'PENDING' in r.stdout:
        print('[PENDING] Submitted is not running; inspect the displayed reason.', flush=True)
    if r.returncode or not r.stdout.strip():
        r = call(['sacct', '-X', '-j', jid, '--format=JobID,State,ExitCode,Elapsed,NodeList'])
        print(r.stdout+r.stderr, flush=True)


def cancel_blocked(b, root, repo):
    """Only an explicitly requested cancellation of the latest matching pending job."""
    user = pwd.getpwuid(os.getuid()).pw_name
    with b.idle_locks(root):
        records = [root/STATE/'latest_submission.json', root/'two_node_v1/latest_submission.json']
        seen = set()
        for path in records:
            if not path.is_file(): continue
            jid = str(b.read(path)['job_id'])
            require(jid.isdigit(), 'Invalid recorded job ID')
            if jid in seen: continue
            seen.add(jid)
            q = call(['squeue', '-h', '-u', user, '-o', '%i|%T|%R'])
            require(q.returncode == 0, 'Cannot read queue; no cancellation: '+q.stderr)
            if not any(line.split('|', 1)[0].strip() == jid for line in q.stdout.splitlines()): continue
            info = call(['scontrol', 'show', 'job', '-o', jid])
            require(info.returncode == 0, 'Cannot verify job identity; no cancellation')
            f = fields(info.stdout)
            require(f.get('UserId', '').split('(')[0] == user and
                    f.get('JobName') in ('g25_two', 'g25_tasks') and
                    Path(f.get('WorkDir', '/nonexistent')).resolve() == repo and
                    any(f.get('StdOut', '').startswith(str(root/'logs'/p)) for p in ('two_', 'tasks_')),
                    'Recorded job identity mismatch; NOT cancelling '+jid)
            require(f.get('JobState') == 'PENDING' and f.get('Reason') in BLOCKED,
                    'Only node-limit-blocked PENDING jobs may be cancelled; NOT cancelling '+jid)
            # State filter also prevents cancelling a job that starts after the identity read.
            r = call(['scancel', '--state=PENDING', '--user='+user, jid])
            require(r.returncode == 0, 'Cancellation failed: '+r.stderr)
            b.write(root/STATE/('cancel_'+jid+'.json'), dict(job_id=jid, fields=f, time_ns=time.time_ns()))
            print('[cancel-requested] '+jid+'; data untouched', flush=True)
        b.empty_queue()  # No new work if a job started meanwhile or is still completing.


def submit(b, a):
    root, repo = a.root.resolve(strict=True), a.repo.resolve(strict=True)
    with b.idle_locks(root):
        b.empty_queue(); history(b, root)
        engine, m, c, job = b.load_engine(root)
        recovery = b.read(root/'two_node_v1/recovery_latest.json')
        require(recovery['base_job_sha256'] == b.sha(root/'formal/job.json') and
                recovery['manifest_sha256'] == b.sha(root/'fast_segments_v1/manifest.json'), 'Run recover first')
        b.probe_shard_locks(root, c.nshards(job['cfg']))
        good, points, leftovers = b.scan_shards(root, engine, m, c, job)
        require(not leftovers, 'Run recover; incomplete output is never silently overwritten')
        print(f'[resume] preserved={len(good)} shards/{points} points', flush=True)
        if engine.choose_segment(m, c, job) is None:
            print('[READY TO MERGE] No job submitted'); return
        cap, evidence = association_hint(a.account, a.partition) if a.nodes == 'auto' else (None, {})
        nodes = resources(a.nodes, a.nodelist, cap)
        print(f'[resources] requested_mode={a.nodes} nodes={nodes}; 4 tasks/node, 16 CPUs/task, 4 GPUs/node', flush=True)
        if cap == 1:
            print('[auto] Visible association node=1; requesting one node, not bypassing policy')
        elif a.nodes == 'auto':
            print('[auto] No confirmed node=1 hint; scheduler resolves 1-2. This is not a permission guarantee.')
            if evidence.get('warning') or evidence.get('returncode', 0):
                print('[policy-query unavailable] '+str(evidence.get('warning', evidence.get('stderr', ''))))
        source, hashes = freeze(b, root)
        unlimited = bool(getattr(a, 'unlimited', False))
        text = batch_text(source, repo, root, nodes, a.nodelist, a.account, a.qos, a.partition,
                          unlimited=unlimited)
        print('[duration] '+('UNLIMITED: --time=0; run until all shards complete' if unlimited
                              else '24h allocation / 23h30m soft collection window'), flush=True)
        if a.mode == 'draft':
            print('[DRAFT ONLY] no sbatch; generated batch script:\n'+text); return
        python = Path(os.environ.get('VIS_PYTHON', sys.executable)).absolute()
        require(python.is_file() and os.access(python, os.X_OK), 'VIS_PYTHON must be an executable absolute path')
        root.joinpath('logs').mkdir(exist_ok=True)
        request = root/STATE/'submissions'/str(time.time_ns()); request.mkdir(parents=True)
        spec_path = request/'spec.json'; script = request/'batch.sbatch'
        script.write_text(text)
        b.write(spec_path, dict(root=str(root), source=str(source), hashes=hashes, requested_nodes=nodes,
                               engine_manifest_sha=b.sha(root/'fast_segments_v1/manifest.json'),
                               policy_hint=evidence, batch_sha=b.sha(script), unlimited=unlimited))
        env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
        env.update(VIS_PYTHON=str(python), G25T_SPEC=str(spec_path), G25T_SPEC_SHA=b.sha(spec_path),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2',
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        cmd = ['sbatch', '--parsable', str(script)]
        check = call(['sbatch', '--test-only', '--parsable', str(script)], env=env)
        b.write(request/'test_only.json', dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr))
        print('[test-only; NOT proof of permission/start]\n'+check.stdout+check.stderr, flush=True)
        require(check.returncode == 0, 'Precheck rejected; zero real submissions')
        b.empty_queue()
        result = call(cmd, env=env)
        rec = dict(command=cmd, spec=str(spec_path), nodes=nodes, stdout=result.stdout,
                   stderr=result.stderr, returncode=result.returncode)
        b.write(request/'reply.json', rec)
        require(result.returncode == 0, 'sbatch failed; no automatic retry: '+result.stderr)
        jid = result.stdout.strip().split(';')[0]
        require(jid.isdigit(), 'Ambiguous reply: inspect squeue before retrying; do not submit again')
        rec.update(job_id=int(jid), log=str(root/f'logs/tasks_{jid}.out'), error=str(root/f'logs/tasks_{jid}.err'))
        b.write(root/STATE/'latest_submission.json', rec)
        print(f'[submitted] ONE job={jid}; requested nodes={nodes}; actual GPU count printed after start', flush=True)
    status(b, root)


def validate_identities(rows, nodes):
    require(len(rows) == 4*nodes, 'Missing task identities')
    require({r['rank'] for r in rows} == set(range(4*nodes)), 'Wrong global ranks')
    hosts = set()
    for node in range(nodes):
        group = [r for r in rows if r['node'] == node]
        require(len(group) == 4 and {r['local'] for r in group} == set(range(4)), 'Wrong per-node task placement')
        require(len({r['host'] for r in group}) == 1 and len({r['gpu_uuid'] for r in group}) == 4,
                'Tasks share a GPU or span wrong hosts; no collection')
        hosts.add(group[0]['host'])
    require(len(hosts) == nodes, 'Expected distinct physical hosts')


def task(b, spec_path, step):
    spec, root = load_spec(b, spec_path); info = b.read(step/'step.json')
    rank, local, node = [int(os.environ.get(k, '-1')) for k in ('SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NODEID')]
    require(0 <= rank < 4*info['nodes'] and 0 <= node < info['nodes'] and 0 <= local < 4, 'Unexpected task ranks')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_token(os.environ.get('CUDA_VISIBLE_DEVICES', ''), local)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Each task must see one allocated GPU')
    prop = torch.cuda.get_device_properties(0)
    require('3090' in prop.name, 'Not an RTX3090: '+prop.name)
    uuid = str(getattr(prop, 'uuid', ''))
    require(uuid and uuid != 'None', 'Cannot verify unique GPU UUID; update GPU probe before collecting')
    identity = dict(rank=rank, local=local, node=node, host=socket.gethostname(), gpu_uuid=uuid)
    b.write(step/f'identity_{rank}.json', identity)
    for i in range(4*info['nodes']): b.wait_file(step/f'identity_{i}.json')
    validate_identities([b.read(step/f'identity_{i}.json') for i in range(4*info['nodes'])], info['nodes'])
    if info['nodes'] == 2:
        if local == 0: b.shared_probe(step/'shared_probe', node, os.environ['SLURM_JOB_ID'])
        b.wait_file(step/'shared_probe/done.json')
    engine, m, _, _ = b.load_engine(root)
    folder = step/f'task_{rank}'; folder.mkdir(exist_ok=False)
    request = folder/'request.json'
    b.write(request, dict(segment=info['segment'], stop_file=info['stop_file'], deadline_unix=info['deadline'],
                         gpu_type='3090', manifest=str(root/'fast_segments_v1/manifest.json')))
    print('[TASK] '+json.dumps(identity)+'; original worker local_rank='+str(local), flush=True)
    # One Slurm task -> one original worker. No extra four-worker supervisor.
    os.execv(sys.executable, [sys.executable, '-u', str(Path(m['source'])/'run.py'), 'worker',
             '--manifest', str(root/'fast_segments_v1/manifest.json'), '--segment', str(info['segment']),
             '--rank', str(local), '--request', str(request)])


def batch(b, spec_path):
    start = time.time(); spec, root = load_spec(b, spec_path)
    nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', '0')))
    require(nodes in (1, 2) and (spec['requested_nodes'] == '1-2' or nodes == int(spec['requested_nodes'])),
            'Actual allocation does not match request')
    jid = os.environ.get('SLURM_JOB_ID', '')
    require(jid.isdigit(), 'Slurm allocation required')
    with b.lock(root/'.grid25_allocation.lock'):
        engine, m, c, job = b.load_engine(root)
        folder = root/STATE/'runs'/jid; folder.mkdir(parents=True, exist_ok=False)
        stop = folder/'STOP'
        unlimited = spec.get('unlimited', False)
        require(isinstance(unlimited, bool), 'Invalid unlimited flag in frozen submission')
        deadline = UNLIMITED_DEADLINE_UNIX if unlimited else start+b.SOFT_SECONDS
        attempts = {}
        b.write(folder/'allocation.json', dict(job_id=jid, nodes=nodes, gpus=4*nodes,
                  nodelist=os.environ.get('SLURM_JOB_NODELIST', ''), deadline=deadline,
                  unlimited=unlimited))
        def pause(*_): stop.touch(exist_ok=True)
        old = {s: signal.signal(s, pause) for s in (signal.SIGUSR1, signal.SIGTERM)}
        try:
            print(f'[RUNNING] ONE JobID={jid}; nodes={nodes}; tasks={4*nodes}; GPUs={4*nodes}; '
                  f'unlimited={unlimited}', flush=True)
            while not stop.exists() and (unlimited or time.time() < deadline):
                seg = engine.choose_segment(m, c, job)
                if seg is None: break
                attempt = attempts.get(seg, 0)
                step = folder/(f'segment_{seg:03d}' if attempt == 0 else
                               f'segment_{seg:03d}_retry_{attempt:03d}')
                step.mkdir()
                before = sum(engine.stage_exists(c, job, sid) for sid in m['segments'][seg])
                b.write(step/'step.json', dict(nodes=nodes, segment=seg, stop_file=str(stop), deadline=deadline))
                process = subprocess.Popen(step_command(HERE, spec_path, step, nodes), start_new_session=True)
                try:
                    require(process.wait() == 0, 'srun failed; no next segment; published shards preserved')
                finally:
                    engine.terminate([process])
                present = [sid for sid in m['segments'][seg] if engine.stage_exists(c, job, sid)]
                engine.engine_integrity(c, job, present, m)
                complete = len(present) == len(m['segments'][seg])
                b.write(step/'result.json', dict(complete=complete, complete_shards=len(present)))
                if not complete:
                    if unlimited and not stop.exists():
                        # A worker may skip a shard already locked by another task.
                        # Explicitly retry only if published-shard count increased.
                        require(len(present) > before,
                                'Incomplete segment without new published shards; '
                                'stopping rather than looping indefinitely')
                        attempts[seg] = attempt + 1
                        print(f'[UNLIMITED CONTINUE] segment={seg} '
                              f'published={len(present)}/{len(m["segments"][seg])}; '
                              'retrying remaining shards in this JobID', flush=True)
                        continue
                    require(stop.exists() or time.time() >= deadline,
                            'Unexpected missing shards; stop rather than retry')
                    break
            outcome = 'READY_TO_MERGE' if engine.choose_segment(m, c, job) is None else 'WINDOW_PAUSED'
            b.write(folder/'result.json', dict(status=outcome, job_id=jid))
            print('['+outcome+'] No automatic submission, merge or training', flush=True)
        finally:
            for s, handler in old.items(): signal.signal(s, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('recover', 'draft', 'next', 'status', 'summary', 'verify', 'log', 'ranks', 'cancel-blocked', 'batch', 'task'))
    p.add_argument('--root', type=Path, default=Path(os.environ.get('GRID25_ROOT', DEFAULT_ROOT)))
    p.add_argument('--repo', type=Path, default=HERE.parents[1])
    p.add_argument('--nodes', default=os.environ.get('GRID25_NODES', 'auto'))
    p.add_argument('--nodelist', default=os.environ.get('GRID25_NODELIST', ''))
    p.add_argument('--account', default=os.environ.get('GRID25_ACCOUNT', ''))
    p.add_argument('--qos', default=os.environ.get('GRID25_QOS', ''))
    p.add_argument('--partition', default=os.environ.get('GRID25_PARTITION', 'GPU'))
    p.add_argument('--unlimited', action='store_true',
                   default=os.environ.get('GRID25_UNLIMITED', '0') == '1',
                   help='Next submission only: Slurm --time=0; collect until all shards finish')
    p.add_argument('--spec', type=Path); p.add_argument('--step', type=Path)
    a = p.parse_args(); b = base()
    if a.mode in ('batch', 'task'):
        return batch(b, a.spec) if a.mode == 'batch' else task(b, a.spec, a.step)
    root, repo = a.root.resolve(strict=True), a.repo.resolve(strict=True)
    if a.mode in ('recover', 'draft', 'next', 'cancel-blocked'):
        branch = call(['git', '-C', str(repo), 'branch', '--show-current'])
        require(branch.returncode == 0 and branch.stdout.strip() == 'mainline-b/h9-scratch50k-r012-v1', 'Wrong branch')
        for flag in ([], ['--cached']):
            r = call(['git', '-C', str(repo), 'diff', '--quiet']+flag)
            require(r.returncode == 0, 'Tracked edits; no reset or stash performed')
    if a.mode == 'recover':
        history(b, root)
        b.recover(root)
    elif a.mode in ('draft', 'next'): submit(b, a)
    elif a.mode == 'cancel-blocked': cancel_blocked(b, root, repo)
    elif a.mode == 'status': status(b, root)
    elif a.mode in ('summary', 'verify'):
        b.diagnostics(root, a.mode)
        print('[note] remaining@8 above assumes 8 workers; four-worker estimate is twice that, not a guarantee')
    else:
        r = b.read(root/STATE/'latest_submission.json'); paths = [Path(r['log']), Path(r['error'])]
        if a.mode == 'ranks':
            steps = sorted((root/STATE/'runs'/str(r['job_id'])).glob('segment_*'))
            paths = sorted(steps[-1].glob('rank_*.*')) if steps else []
        for path in paths:
            print('=== '+str(path)+' ===', flush=True)
            if path.is_file(): print('\n'.join(path.read_text(errors='replace').splitlines()[-24:]))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print('[STOP] '+str(exc), file=sys.stderr)
        sys.exit(2)
