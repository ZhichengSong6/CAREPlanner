#!/usr/bin/env python3
"""Frozen R1/B convergence characterization; original test inputs, no training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
from ie_common import (HERE, ARMS, require, sha, read, lines, write, finish, stage,
                       code_hashes, verify, model_digest, validate_starts, load_runtime)


def check_full(got, expected):
    require(got['fov_pass'] == expected['fov_pass'], 'FULL FOV decision differs from frozen reference')
    require(abs(got['candidate_g_m'] - expected['candidate_g_m']) <= 2e-5, 'FULL margin differs')
    require(np.allclose(got['q_candidate'], expected['q_candidate'], atol=2e-5, rtol=0), 'FULL candidate differs')
    for key in ('gradient_calls', 'value_calls', 'root_source', 'solution_mode'):
        require(got[key] == expected[key], f'FULL {key} differs')


def prepare(k2root, out):
    kroot = Path(k2root).resolve() / 'formal'; out = Path(out).resolve()
    require(not out.exists(), 'Existing experiment: use resume, never overwrite')
    require(not out.is_relative_to(kroot.parent) and not kroot.parent.is_relative_to(out), 'Output overlaps Keep-v2')
    cfg = read(HERE / 'protocol.json'); old = read(kroot / 'job.json')
    require(cfg['models'] == list(ARMS) and cfg['budgets'] == [1, 2, 3, 5, 10], 'Unexpected protocol')
    require(read(kroot / 'training_complete.json')['status'] == 'COMPLETE', 'Keep-v2 training incomplete')
    source_inputs = kroot / 'confirmation/inputs'; source_report = kroot / 'confirmation/report'
    stage(source_inputs); stage(source_report)
    require(sha(source_inputs / 'starts.jsonl') == cfg['starts_sha256'], 'Expected the frozen 22518-case suite')
    starts = list(lines(source_inputs / 'starts.jsonl'))
    validate_starts(starts, cfg, np.asarray(old['lo']), np.asarray(old['hi']))
    old_report = read(source_report / 'report.json')
    b_run = read(kroot / 'B_keep/run.json'); b_stage = read(kroot / 'B_keep/complete.json')
    require(b_stage['status'] == 'COMPLETE' and b_stage['additional_updates'] == 20000, 'B is incomplete')
    require(b_run['runtime_sha256'] == b_stage['files']['runtime.pt'] == cfg['b_sha256'], 'Wrong B identity')
    models = {'R1': {'path': old['r1'], 'sha256': cfg['r1_sha256']},
              'B_keep': {'path': str(kroot / 'B_keep/runtime.pt'), 'sha256': cfg['b_sha256']}}
    for spec in models.values():
        p = Path(spec['path']).resolve()
        require(not p.is_relative_to(out) and not out.is_relative_to(p.parent), 'Output overlaps weights')
        require(sha(p) == spec['sha256'], f'Weight identity mismatch: {p}')
    reference = [None] * len(starts)
    ref_source_hashes = {}
    for rank in range(4):
        folder = kroot / 'confirmation/evaluation' / f'rank{rank}'; manifest = stage(folder)
        for name in ARMS:
            require(manifest['checkpoints'][name] == models[name]['sha256'], 'Previous result used different weights')
        ref_source_hashes[str(folder / 'solves.jsonl')] = sha(folder / 'solves.jsonl')
        for row in lines(folder / 'solves.jsonl'):
            i = row['case_id']; require(type(i) is int and 0 <= i < len(starts), 'Invalid reference case ID')
            require(reference[i] is None and row['start'] == starts[i], 'Duplicate/misaligned reference case')
            reference[i] = dict(case_id=i, models={n: {key: row['models'][n][key] for key in
                ('q_candidate', 'candidate_g_m', 'fov_pass', 'gradient_calls', 'value_calls', 'root_source', 'solution_mode')}
                for n in ARMS})
    require(all(r is not None for r in reference), 'Missing reference case')
    for group in ('local', 'uniform', 'all'):
        ids = [i for i, row in enumerate(starts) if group == 'all' or row['cohort'] == group]
        for name in ARMS:
            count = sum(reference[i]['models'][name]['fov_pass'] for i in ids)
            require(abs(count / len(ids) - old_report['groups'][group]['rates'][name]) < 1e-12, 'Reference summary disagrees with raw cases')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.prepare.', dir=out.parent))
    try:
        deps = Path(old['deps'])
        for rel, digest in old['dep_hashes'].items():
            source = (deps / rel).resolve()
            require(source.is_relative_to(deps.resolve()) and sha(source) == digest, f'Frozen upstream corrupted: {rel}')
            target = tmp / 'dependencies' / rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in ('starts.jsonl', 'coverage.jsonl'):
            shutil.copyfile(source_inputs / name, tmp / name)
        with (tmp / 'reference.jsonl').open('x') as f:
            for row in reference:
                f.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
        job = dict(cfg=cfg, out=str(out), source_keep_v2=str(kroot), models=models,
                   lo=old['lo'], hi=old['hi'], masks=old['masks'],
                   code_hashes=code_hashes(), dep_hashes=old['dep_hashes'],
                   frozen_hashes={f: sha(tmp / f) for f in ('starts.jsonl', 'coverage.jsonl', 'reference.jsonl')},
                   prior_result_hashes=ref_source_hashes)
        # Actual parent weights, exact deployed single-step formulas and actual
        # URDF are checked before requesting GPUs; no optimizer is constructed.
        import torch
        from budget_probe import instrument, execute, geometric_result
        torch.set_num_threads(2)
        tempjob = dict(job, out=str(tmp))
        mods, probes, oracle = load_runtime(tempjob, torch.device('cpu'))
        checks = []
        for s in range(8):
            i = next(i for i, row in enumerate(starts) if row['sensor'] == s)
            row = starts[i]; x = torch.tensor(row['x']); q = torch.tensor(row['q_init'])
            require(oracle.value(x, q, s) < 0 and oracle.value(x, torch.tensor(row['witness_q']), s) > 0, 'Actual input geometry changed')
            for name in ARMS:
                probe = instrument(probes[name])
                raw = probes[name]._optimize_branch(x.reshape(1, 3), q.reshape(1, 7), s)
                full, _ = execute(probe, x, q, s, None)
                capped, _ = execute(probe, x, q, s, 18)
                require(np.allclose(raw['q_candidate'], full['q_candidate'], atol=1e-7, rtol=0)
                        and full['q_candidate'] == capped['q_candidate'], 'Instrumentation changed FULL result')
                one, _ = execute(probe, x, q, s, 1)
                require(one['gradient_calls'] <= 1, 'One-step budget overrun')
                checks.append(dict(case_id=i, model=name, full_calls=full['gradient_calls']))
        del mods, probes, oracle
        job['preflight'] = dict(status='PASS', device='cpu', real_weight_geometry_checks=checks,
                               no_training=True, gpu_reference_comparison='per-case gate during formal evaluation')
        write(tmp / 'job.json', job)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print('[prepared] frozen R1/B, 22518 unchanged starts; CPU budget/FULL parity PASS', flush=True)


def annotate(result, oracle, x, s, device, job):
    import torch
    from budget_probe import geometric_result
    q = torch.tensor(result['q_candidate'], device=device, dtype=torch.float32)
    margin = oracle.value(x, q, s)
    return dict(result, **geometric_result(result['q_candidate'], margin, job['lo'], job['hi']))


def evaluate(job, rank):
    import torch
    from budget_probe import instrument, execute, first_hit
    verify(job); out = Path(job['out']); parent = out / 'evaluation'
    parent.mkdir(exist_ok=True); final = parent / f'rank{rank}'
    require(not final.exists(), 'Existing shard; coordinator must verify before reuse')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One visible GPU per evaluation worker')
    torch.set_num_threads(2); torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); models, raw, oracle = load_runtime(job, device)
    probes = {n: instrument(raw[n]) for n in ARMS}; before = {n: model_digest(m) for n, m in models.items()}
    starts = list(lines(out / 'starts.jsonl')); refs = list(lines(out / 'reference.jsonl'))
    budgets = list(job['cfg']['budgets']) + [None]; mine = starts[rank::job['cfg']['world']]
    if mine:
        sample = mine[0]; x = torch.tensor(sample['x'], device=device); q = torch.tensor(sample['q_init'], device=device)
        for _ in range(2):
            for name in ARMS:
                for budget in budgets:
                    execute(probes[name], x, q, sample['sensor'], budget)
    tmp = Path(tempfile.mkdtemp(prefix=f'.rank{rank}.', dir=parent))
    with (tmp / 'results.jsonl').open('x') as f:
        for index, row in enumerate(mine):
            x = torch.tensor(row['x'], device=device); q = torch.tensor(row['q_init'], device=device); s = row['sensor']
            require(oracle.value(x, q, s) < 0, 'Frozen outside start no longer outside')
            result = dict(case_id=row['case_id'], point_id=row['point_id'], sensor=s, cohort=row['cohort'], models={})
            order = ARMS if index % 2 == 0 else ARMS[::-1]
            shifted = budgets[index % len(budgets):] + budgets[:index % len(budgets)]
            for name in order:
                all_results = {}
                for budget in shifted:
                    values, trace = execute(probes[name], x, q, s, budget)
                    values = annotate(values, oracle, x, s, device, job)
                    key = str(budget) if budget is not None else 'full'
                    if budget is None:
                        check_full(values, refs[row['case_id']]['models'][name])
                        margins = [oracle.value(x, torch.tensor(t['q'], device=device), s) for t in trace]
                        hit, observed = first_hit(trace, margins, job['lo'], job['hi'])
                        values['first_hit_gradient_query_diagnostic'] = hit
                        values['trace_diagnostic'] = observed
                    all_results[key] = values
                result['models'][name] = all_results
            f.write(json.dumps(result, allow_nan=False, sort_keys=True) + '\n')
            if (index + 1) % 100 == 0:
                f.flush(); print(f'[budget rank{rank}] {index+1}/{len(mine)}; FULL reference reproduced', flush=True)
        f.flush(); os.fsync(f.fileno())
    require(all(model_digest(m) == before[n] for n, m in models.items()), 'Frozen model mutated')
    finish(tmp, cases=len(mine), models=job['models'], inputs_sha256=sha(out / 'starts.jsonl'),
           warmup_calls_per_model=2*len(budgets) if mine else 0, full_reference_parity='PASS')
    os.rename(tmp, final)
    print(f'[budget rank{rank}] COMPLETE cases={len(mine)}', flush=True)


def admission(job):
    import torch
    subprocess.run(['nvidia-smi'], check=False)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4, 'Four visible GPUs required')
    for i in range(4):
        free, total = torch.cuda.mem_get_info(i)
        print(f'[GPU {i}] free={free/(1<<30):.2f} GiB total={total/(1<<30):.2f} GiB', flush=True)
        require(free >= job['cfg']['minimum_free_gib']*(1<<30), 'Insufficient free GPU memory; no process killed')


def worker(job):
    verify(job); out = Path(job['out']); (out / 'logs').mkdir(exist_ok=True)
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    require(len(gpus) == 4, 'Exactly four GPU assignments required')
    subprocess.run([sys.executable, str(HERE / 'evaluate.py'), 'admission', '--job', str(out / 'job.json')], check=True)
    processes = []; handles = []
    def stop(*_):
        raise KeyboardInterrupt('Job interrupted; complete shards are resumable')
    signal.signal(signal.SIGTERM, stop)
    try:
        for rank in range(4):
            dest = out / 'evaluation' / f'rank{rank}'
            if dest.exists():
                stage(dest); print(f'[reuse] complete rank{rank}', flush=True); continue
            log = (out / 'logs' / f'rank{rank}_{time.time_ns()}.log').open('x'); handles.append(log)
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = gpus[rank]
            p = subprocess.Popen([sys.executable, '-u', str(HERE / 'evaluate.py'), 'rank', '--job', str(out / 'job.json'), '--rank', str(rank)],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((p, rank, log.name))
        while processes:
            for p, rank, path in list(processes):
                status = p.poll()
                if status is None:
                    continue
                require(status == 0, f'Rank {rank} failed: {path}')
                processes.remove((p, rank, path))
            if processes:
                time.sleep(.5)
    finally:
        for p, _, _ in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p, _, _ in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
        for handle in handles:
            handle.close()
    verify(job)
    from report import merge
    merge(job)
    print('[COMPLETE] fixed-weight iteration report; no training or model promotion', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'worker', 'rank', 'admission', 'check', 'verify'))
    p.add_argument('--k2root', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--job', type=Path); p.add_argument('--rank', type=int, default=0)
    args = p.parse_args()
    if args.mode == 'prepare':
        require(args.k2root is not None and args.out is not None, 'Missing prepare paths')
        prepare(args.k2root, args.out); return
    require(args.job is not None, '--job required'); job = read(args.job)
    if args.mode == 'worker':
        worker(job)
    elif args.mode == 'rank':
        require(0 <= args.rank < 4, 'Bad rank'); evaluate(job, args.rank)
    elif args.mode == 'admission':
        admission(job)
    elif args.mode == 'check':
        verify(job); print('[VERIFIED] immutable evaluation job', flush=True)
    else:
        verify(job); out = Path(job['out']); stage(out / 'report')
        for rank in range(4):
            stage(out / 'evaluation' / f'rank{rank}')
        print('[VERIFIED] iteration report, checkpoints, starts and all shards', flush=True)


if __name__ == '__main__':
    main()
