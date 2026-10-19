#!/usr/bin/env python3
"""Stage coordinator: reference mining -> A20k -> B20k; separate final confirmation."""
from __future__ import annotations
import argparse,os,signal,subprocess,sys,time
from pathlib import Path
from k2_common import HERE,ARMS,require,read,sha,stage,verify,prepare,write

def launch_workers(job,kind):
    out=Path(job['out']);gpus=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    require(len(gpus)==4,'Exactly four assigned GPU identifiers required')
    dirs={'starts':out/'reference_inputs','mine':out/'reference_mined',
          'generate':out/'confirmation/geometry','evaluate':out/'confirmation/evaluation'}
    processes=[];handles=[];(out/'logs').mkdir(exist_ok=True)
    try:
        for rank in range(4):
            p=dirs[kind]/f'rank{rank}'
            if p.exists():stage(p);print(f'[reuse] {kind} rank{rank}',flush=True);continue
            f=(out/'logs'/f'{kind}_rank{rank}_{time.time_ns()}.log').open('x');handles.append(f)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='' if kind in ('starts','generate') else gpus[rank],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            child=subprocess.Popen([sys.executable,'-u',str(HERE/'run.py'),kind,'--job',str(out/'job.json'),'--rank',str(rank)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((child,rank,f.name))
        while processes:
            for p,rank,f in list(processes):
                rc=p.poll()
                if rc is None:continue
                require(rc==0,f'{kind} rank{rank} failed rc={rc}; log={f}')
                processes.remove((p,rank,f))
            if processes:time.sleep(.5)
    finally:
        for p,_,_ in processes:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p,_,_ in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
        for f in handles:f.close()

def launch_arm(job,arm):
    p=Path(job['out'])/arm
    if (p/'complete.json').exists():stage(p);print(f'[reuse] {arm} complete',flush=True);return
    proc=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','--max_restarts=0',str(HERE/'train.py'),'--job',str(Path(job['out'])/'job.json'),'--arm',arm],start_new_session=True)
    try:require(proc.wait()==0,f'{arm} training failed; use resume after inspecting log')
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def verify_training(job):
    out=Path(job['out']);a=stage(out/'A_continue');b=stage(out/'B_keep')
    require(a['base_stream']==b['base_stream'],'A/B base sample stream mismatch; experiment invalid')
    require(a['additional_updates']==b['additional_updates']==20000,'Training incomplete')
    if not (out/'training_complete.json').exists():
        write(out/'training_complete.json',dict(status='COMPLETE',matched_base_stream=a['base_stream'],
           A_runtime_sha256=sha(out/'A_continue/runtime.pt'),B_runtime_sha256=sha(out/'B_keep/runtime.pt'),
           evaluation='NOT_RUN; no checkpoint promotion'))
    print('[VERIFIED] A/B each20k; SAME base sample stream; reference teacher unchanged',flush=True)

def worker(job,action):
    import torch
    verify(job);require(torch.cuda.is_available() and torch.cuda.device_count()==4,'Formal allocation must contain four GPUs')
    def interrupt(*_):raise KeyboardInterrupt('Stopped; completed stages are reusable')
    signal.signal(signal.SIGTERM,interrupt)
    if action in ('train','resume'):
        from replay import freeze_inputs,merge
        launch_workers(job,'starts');freeze_inputs(job);launch_workers(job,'mine');merge(job)
        for arm in ARMS:launch_arm(job,arm)
        verify_training(job)
    else:
        verify_training(job)
        from evaluate import freeze,report
        launch_workers(job,'generate');freeze(job);launch_workers(job,'evaluate');report(job)
        print('[COMPLETE] new-space final confirmation; no automatic promotion',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','worker','starts','mine','generate','evaluate','verify'))
    p.add_argument('--repo',type=Path);p.add_argument('--sgroot',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--job',type=Path);p.add_argument('--rank',type=int,default=0);p.add_argument('--action',choices=('train','resume','eval'),default='train')
    a=p.parse_args()
    if a.mode=='prepare':
        require(all(x is not None for x in (a.repo,a.sgroot,a.out)),'Missing prepare inputs');prepare(a.repo,a.sgroot,a.out);return
    require(a.job is not None and 0<=a.rank<4,'Missing job or invalid rank');job=read(a.job)
    if a.mode=='worker':worker(job,a.action)
    elif a.mode=='verify':
        verify(job);verify_training(job)
        out=Path(job['out'])/'confirmation'
        if (out/'report/complete.json').exists():
            stage(out/'report');stage(out/'inputs')
            for rank in range(4):stage(out/'evaluation'/f'rank{rank}')
            print('[VERIFIED] completed confirmation hashes',flush=True)
    elif a.mode in ('starts','mine'):
        from replay import create_starts,mine
        (create_starts if a.mode=='starts' else mine)(job,a.rank)
    else:
        from evaluate import generate,evaluate
        (generate if a.mode=='generate' else evaluate)(job,a.rank)
if __name__=='__main__':main()
