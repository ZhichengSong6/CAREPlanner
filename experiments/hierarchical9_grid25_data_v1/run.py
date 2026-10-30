#!/usr/bin/env python3
"""25 mm Q0 acquisition coordinator. Two single-node jobs share immutable shards."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
from g25_common import (HERE,SOURCE_DIR,PINNED_BLOBS,URDF_REL,URDF_BLOB,ROW_KEYS,
    require,read,write,sha,blob,grid,lock,reference_metadata,check_urdf,load_generator,
    verify_job,args_for,nshards)


def preflight_geometry(job):
    """CPU geometry parity only; not a GPU pilot and no sampled training data saved."""
    import torch
    torch.set_num_threads(2)
    g=load_generator(job);a=args_for(job);r=g.URDF.from_xml_file(a.urdf)
    import validate_visibility_oracle as v
    limits=g.get_joint_limits(r,a.joint_names);q=g.sample_q_batch(np.random.default_rng(731025),limits,a.joint_names,4)
    specs=g.prepare_chain_specs(r,a.base_frame,a.sensor_frames,a.joint_names,torch.device('cpu'))
    require(np.array_equal(g.SENSOR_CHAIN_MASKS,job['reference']['sensor_chain_masks']),'Generator masks differ')
    x=np.array([.1,.15,.55],np.float32)
    with torch.no_grad():_,m,_,_=g.visibility_g_batch(torch.tensor(x),torch.tensor(q),specs,a.horizontal_fov_deg,a.vertical_fov_deg,a.z_min,a.z_max,a.delta)
    errors=[]
    for s,frame in enumerate(a.sensor_frames):
        chain=v.find_chain_joints(r,a.base_frame,frame)
        for i,row in enumerate(q):
            T=v.fk_transform(chain,dict(zip(a.joint_names,map(float,row))))
            gt,_,_=v.sensor_margin(x,T,a.horizontal_fov_deg,a.vertical_fov_deg,a.z_min,a.z_max)
            errors.append(abs(gt-float(m[i,s])))
    require(max(errors)<2e-6,'FP32/NumPy upstream FK-FOV parity failed')
    return {'status':'PASS','queries':len(errors),'max_abs_margin_error_m':max(errors),'device':'cpu'}


def prepare(repo,root,reference):
    repo,root,reference=(Path(p).resolve() for p in (repo,root,reference));out=root/'formal'
    if out.exists():
        job=read(out/'job.json');verify_job(job)
        require(Path(job['reference_path'])==reference,'Reference path changed')
        print('[reuse] immutable collection job; no old labels copied',flush=True);return job
    cfg=read(HERE/'protocol.json');ref=reference_metadata(reference,cfg)
    for p in (repo,reference.parent):
        require(not (out==p or out.is_relative_to(p) or p.is_relative_to(out)),f'Output overlaps protected source: {p}')
    require(sha(reference)==cfg['reference_sha256'],'Reference dataset checksum mismatch; do not guess profile')
    for name,gitsha in PINNED_BLOBS.items():require(blob(repo/SOURCE_DIR/name)==gitsha,f'Original collector dependency changed: {name}')
    require(blob(repo/URDF_REL)==URDF_BLOB,'URDF changed: needs explicit review, not silent reuse')
    check_urdf(repo/URDF_REL,cfg,ref['sensor_chain_masks'])
    # Upper bound for both compact shards and the merged file; no compressed-size guess.
    per_slot=7*8*4+8+8*4+8*8*4+8*8*2
    upper=cfg['grid_points']*ref['q0_per_sensor']*per_slot+cfg['grid_points']*100
    require(shutil.disk_usage(root).free>=2*upper+(4<<30),f'Need conservative free-disk budget {(2*upper+(4<<30))/(1<<30):.1f} GiB for shards + merge')
    tmp=Path(tempfile.mkdtemp(prefix='.prepare.',dir=root))
    try:
        deps=tmp/'dependencies';paths=[SOURCE_DIR+'/'+k for k in PINNED_BLOBS]+[URDF_REL]
        for rel in paths:
            dest=deps/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/rel,dest)
        np.save(tmp/'x.npy',grid(cfg),allow_pickle=False)
        job=dict(cfg=cfg,out=str(out),deps=str(out/'dependencies'),reference_path=str(reference),reference=ref,
            reference_sha256=cfg['reference_sha256'],reference_bytes=reference.stat().st_size,
            x_sha256=sha(tmp/'x.npy'),deps_sha256={rel:sha(deps/rel) for rel in paths},
            code_hashes={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.sh','.json')},
            compact_raw_upper_bytes=upper,scope='Q0_COLLECTION_ONLY; no CASE files; no neural checkpoint loaded',
            split_status='Full grid retained. Train/validation/test split must be frozen before subsequent training.')
        testjob=dict(job,deps=str(deps));job['preflight']=preflight_geometry(testjob)
        write(tmp/'job.json',job);(tmp/'shards').mkdir();(tmp/'locks').mkdir()
        os.rename(tmp,out)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print('[prepared] grid=65x65x41=173225, spacing=25mm, frame=base_link; original per-point sampler inherited',flush=True)
    print('[inherited]',json.dumps(ref,sort_keys=True),flush=True)
    print('[out]',out,flush=True);return job


def parse_nodes(text):
    ns=text.split(',')
    require(len(ns) in (1,2) and len(set(ns))==len(ns) and all(n in ('3090node2','3090node3') for n in ns),
            'NODES must be 3090node2, 3090node3, or 3090node2,3090node3')
    return ns


def sbatch_args(root,repo,node,code,limit):
    return ['sbatch','--parsable','--partition=GPU','--nodes=1','--ntasks=1','--gres=gpu:3090:4',
       '--cpus-per-task=16','--exclusive',f'--time={limit}',f'--nodelist={node}',f'--chdir={repo}',
       '--export=ALL','--job-name=vis_grid25',f'--output={root}/logs/collect_%j.out',
       f'--error={root}/logs/collect_%j.out',str(code/'run.sh'),'_node']


def previous_jobs(root):
    p=root/'latest_submission.json';return read(p)['jobs'] if p.exists() else []


def ensure_idle(root):
    terminal=('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE')
    for j in previous_jobs(root):
        q=subprocess.run(['squeue','-h','-j',str(j['id']),'-o','%T'],capture_output=True,text=True)
        require(not q.stdout.strip(),f'Job {j["id"]} is still active; do not duplicate submission')
        a=subprocess.run(['sacct','-X','-n','-j',str(j['id']),'--format=State'],capture_output=True,text=True)
        state=a.stdout.strip().split()[0] if a.stdout.strip() else ''
        require(a.returncode==0 and any(state.startswith(s) for s in terminal),f'Cannot confirm terminal Slurm state for {j["id"]}: {state!r}')


def submit(root,repo,nodes,action):
    root,repo=Path(root).resolve(),Path(repo).resolve();parse_nodes(nodes)
    ensure_idle(root);require(not (root/'formal/dataset/complete.json').exists(),'Dataset complete; use verify/summary, not submit')
    require((root/'formal/job.json').is_file(),'Prepare data protocol first')
    verify_job(read(root/'formal/job.json'))
    stamp=time.time_ns();record={'action':action,'jobs':[],'submitted_ns':stamp,'nodes_requested':nodes}
    env=os.environ.copy();env.update(G25_CODE=str(HERE),G25_ROOT=str(root),G25_JOB=str(root/'formal/job.json'))
    for idx,node in enumerate(parse_nodes(nodes)):
        env['G25_SLOT_BASE']=str(4*idx)
        try:
            r=subprocess.run(sbatch_args(root,repo,node,HERE,os.environ.get('TIME_LIMIT','3-00:00:00')),
                             capture_output=True,text=True,env=env)
            require(r.returncode==0,f'sbatch failed for {node}: {r.stderr.strip()}')
            raw=r.stdout.strip().split(';')[0];require(re.fullmatch(r'\d+',raw) is not None,f'Unrecognized sbatch output; inspect squeue before retrying: {r.stdout}')
            jobid=int(raw);record['jobs'].append({'id':jobid,'node':node,'log':str(root/f'logs/collect_{jobid}.out')})
            write(root/'latest_submission.json',record,replace=True)
            write(root/f'logs/submission_{stamp}.json',record,replace=True)
            print(f'[submitted] job={jobid} node={node} gpus=4 exclusive; shared dynamic Q0 shards',flush=True)
        except BaseException:
            if record['jobs']: print('[PARTIAL SUBMISSION] accepted jobs remain active; they can collect the entire grid. Do not duplicate or delete.',flush=True)
            raise


def admission(job):
    import torch
    require(os.environ.get('SLURM_JOB_ID'),'GPU check only on allocated node')
    require(torch.cuda.is_available() and torch.cuda.device_count()==4,'Exactly four Slurm-visible CUDA GPUs required')
    for d in range(4):
        with torch.cuda.device(d):free,total=torch.cuda.mem_get_info()
        print(f'[GPU {d}] {torch.cuda.get_device_name(d)} free={free/(1<<30):.2f}/{total/(1<<30):.2f} GiB',flush=True)
        require(free>=job['cfg']['min_free_gpu_gib']*(1<<30),'GPU memory admission failed; inspect processes, do not kill others')
    print('[GPU ADMISSION PASS] 4 GPUs; allocation is not a guarantee against external processes',flush=True)


def node(job,slotbase):
    from merge import merge
    verify_job(job);root=Path(job['out']).parent
    done=Path(job['out'])/'dataset/complete.json'
    if done.exists():
        m=read(done);require(m.get('status')=='COMPLETE' and m.get('job_sha256')==sha(Path(job['out'])/'job.json'),'Invalid completion marker')
        print('[reuse] another node already published complete dataset; no new GPU workers',flush=True);return
    subprocess.run(['nvidia-smi'],check=False)
    # Exit the checker process before workers so its CUDA contexts are freed.
    subprocess.run([sys.executable,str(HERE/'run.py'),'admission','--job',str(Path(job['out'])/'job.json')],check=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');require(len(visible)==4 and all(visible),'Need four Slurm CUDA identifiers')
    processes=[];handles=[]
    def interrupted(*_):raise KeyboardInterrupt('Worker interrupted; published shards retained')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        for rank,gpu in enumerate(visible):
            p=root/'logs'/f'worker_{os.environ["SLURM_JOB_ID"]}_{rank}.log';f=p.open('a');handles.append(f)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            child=subprocess.Popen([sys.executable,'-u',str(HERE/'run.py'),'worker','--job',str(Path(job['out'])/'job.json'),'--slot',str(slotbase+rank)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((child,str(p)))
        while processes:
            for proc,p in list(processes):
                rc=proc.poll()
                if rc is None:continue
                require(rc==0,f'Collector failed rc={rc}; inspect {p}')
                processes.remove((proc,p))
            if processes:time.sleep(1)
    finally:
        for proc,_ in processes:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for proc,_ in processes:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
        for f in handles:f.close()
    merge(job)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','submit','resume','node','worker','admission','merge','verify','summary','status','log','ranks','pack'))
    p.add_argument('--repo',type=Path);p.add_argument('--root',type=Path);p.add_argument('--reference',type=Path)
    p.add_argument('--nodes',default='3090node2,3090node3');p.add_argument('--job',type=Path);p.add_argument('--slot',type=int,default=0)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.repo,a.root,a.reference);return
    if a.mode in ('submit','resume'):submit(a.root,a.repo,a.nodes,a.mode);return
    if a.mode in ('status','log','ranks','pack'):
        root=a.root
        if a.mode=='status':
            for j in previous_jobs(root):
                subprocess.run(['squeue','-j',str(j['id']),'-o','%.18i %.12T %.12M %.30R'],check=False)
                subprocess.run(['sacct','-X','-j',str(j['id']),'--format=JobID,State,ExitCode,Elapsed,NodeList'],check=False)
                print('[log]',j['log'])
        elif a.mode=='log':
            for j in previous_jobs(root):
                print('===',j['log'],'===',flush=True)
                if Path(j['log']).exists():subprocess.run(['tail','-n','50',j['log']],check=True)
        elif a.mode=='ranks':
            for j in previous_jobs(root):
                for f in sorted((root/'logs').glob(f'worker_{j["id"]}_*.log')):
                    print('===',f,'===',flush=True);subprocess.run(['tail','-n','8',str(f)],check=True)
        else:
            import tarfile
            dest=root/f'review_{time.time_ns()}.tar.gz'
            with tarfile.open(dest,'x:gz') as t:
                for f in sorted(root.rglob('*')):
                    if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.md','.py','.sh','.out','.log','.env'):
                        t.add(f,arcname=str(f.relative_to(root)),recursive=False)
            print('[pack]',dest);print(sha(dest))
        return
    require(a.job is not None,'--job required');job=read(a.job)
    if a.mode=='node':node(job,a.slot)
    elif a.mode=='admission':admission(job)
    elif a.mode=='worker':
        from collect import worker
        worker(job,a.slot)
    else:
        from merge import merge,verify_dataset,summary
        {'merge':merge,'verify':verify_dataset,'summary':summary}[a.mode](job)
if __name__=='__main__':main()
