#!/usr/bin/env python3
"""Single-job four-GPU segmented collector around immutable legacy Grid25 inputs."""
from __future__ import annotations
import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib
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
FILES = ('run.py', 'fast_core.py', 'run.sh', 'test_fast_segments.py')
SEGMENT_SHARDS = 64
PLAN_SEED = 731026


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@contextlib.contextmanager
def locked(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another entry is active; no concurrent job')
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def empty_queue():
    name = pwd.getpwuid(os.getuid()).pw_name
    p = subprocess.run(['squeue','-h','-u',name,'-o','%i %T %j'],capture_output=True,text=True)
    require(p.returncode == 0, 'Cannot read Slurm queue; fail closed')
    require(not p.stdout.strip(), 'Existing job; no second submission.\n'+p.stdout.strip())


def legacy_code(root):
    env = Path(root)/'source.env'
    require(env.is_file(), 'Existing Grid25 source.env required; do not recreate or delete old data')
    found = []
    for line in env.read_text().splitlines():
        parts = shlex.split(line)
        if len(parts)==2 and parts[0]=='export' and parts[1].startswith('CODE='):
            found.append(Path(parts[1][5:]).resolve())
    require(len(found)==1 and (found[0]/'g25_common.py').is_file(), 'Invalid frozen collector pointer')
    return found[0]


def modules(code):
    code = Path(code).resolve()
    sys.path.insert(0, str(code))
    result = []
    for name in ('g25_common','collect','merge'):
        if name in sys.modules:
            require(Path(sys.modules[name].__file__).resolve().parent==code, 'Conflicting module: '+name)
        result.append(importlib.import_module(name))
    return result


def base(root):
    root = Path(root).resolve(); code = legacy_code(root)
    common, collector, merger = modules(code)
    job = read(root/'formal/job.json')
    require(Path(job['out']).resolve()==root/'formal', 'Wrong base output')
    common.verify_job(job)
    ref = job['reference']
    require(ref['zero_level_optimizer']=='adam' and ref['max_iter']>0, 'Expected inherited Adam profile')
    require(ref['has_occlusion_labels'] is False, 'FOV-only profile required')
    return code, common, collector, merger, job


def source_hashes(directory=HERE):
    return {name:sha(Path(directory)/name) for name in FILES}


def plan(total, size=SEGMENT_SHARDS):
    import numpy as np
    order = np.random.default_rng(PLAN_SEED).permutation(total).astype(int).tolist()
    return [order[k:k+size] for k in range(0,total,size)]


def prepare(root):
    """New sidecar protocol only. Never edit legacy job/source/shards."""
    root = Path(root).resolve(); code,c,_,_,job = base(root)
    fp = root/'fast_segments_v1'; path = fp/'manifest.json'
    hashes = source_hashes()
    ident = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    if path.exists():
        m=read(path)
        require(m['base_job_sha256']==sha(root/'formal/job.json'), 'Original job identity changed')
        require(m['source_sha256']==hashes, 'Fast source changed; do not silently mix engines')
        require(source_hashes(m['source'])==hashes, 'Frozen fast source changed')
        require(m['segments']==plan(c.nshards(job['cfg'])), 'Segment plan changed')
        return path,m
    fp.mkdir(parents=True,exist_ok=True)
    dest=root/'fast_sources'/ident; dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        tmp=Path(tempfile.mkdtemp(prefix='.freeze.',dir=dest.parent))
        try:
            for name in FILES+('README_zh.md','TEST_RESULTS.md'):
                shutil.copyfile(HERE/name,tmp/name)
            os.rename(tmp,dest)
        finally:
            if tmp.exists(): shutil.rmtree(tmp)
    require(source_hashes(dest)==hashes,'Snapshot mismatch')
    m=dict(format='grid25_fast_segments_v1',source=str(dest),source_sha256=hashes,
           collector=str(code),root=str(root),base_job_sha256=sha(root/'formal/job.json'),
           segments=plan(c.nshards(job['cfg'])),segment_shards=SEGMENT_SHARDS,plan_seed=PLAN_SEED,
           scope='Same canonical data layout and labels; per-shard producer recorded. No training/LOS.',
           float_claim='Sampled same-device tolerance check, NOT cross-device bitwise equality')
    write(path,m)
    print(f'[prepared fast] {len(m["segments"])} segments; at most {SEGMENT_SHARDS} shards/segment; original data retained',flush=True)
    return path,m


def verify_manifest(path):
    m=read(path); root=Path(m['root'])
    require(source_hashes()==m['source_sha256'],'Run from frozen fast snapshot; source mismatch')
    require(sha(root/'formal/job.json')==m['base_job_sha256'],'Base job changed')
    require(legacy_code(root)==Path(m['collector']),'Legacy source pointer changed')
    _,c,collector,merger,job=base(root)
    require(m['segments']==plan(c.nshards(job['cfg'])),'Frozen segment plan changed')
    return m,c,collector,merger,job


def stage_exists(c,job,sid):
    path=Path(job['out'])/'shards'/f'{sid:05d}'
    if not path.exists(): return False
    c.stage(job,sid,False)  # Incorrect or partially published directory fails loudly.
    return True


def choose_segment(m,c,job):
    for i,ids in enumerate(m['segments']):
        if any(not stage_exists(c,job,s) for s in ids): return i
    return None


def node_spec(name):
    require(re.fullmatch(r'(3090|4090)node[1-3]',name) is not None,
            'NODE must be one explicit 3090node1..3 or 4090node1..3; never two nodes')
    return name[:4]


def sbatch_command(root,repo,source,node,mode):
    typ=node_spec(node)
    return ['sbatch','--parsable','--partition=GPU','--nodes=1','--ntasks=1',
            '--cpus-per-task=16',f'--gres=gpu:{typ}:4','--exclusive','--no-requeue',
            '--time=1-00:00:00','--signal=B:USR1@900',f'--nodelist={node}',
            f'--chdir={repo}','--export=ALL','--job-name=g25_fast',
            f'--output={root}/logs/fast_%j.out',f'--error={root}/logs/fast_%j.out',
            str(Path(source)/'run.sh'),'_batch']


def previous_done(root):
    p=Path(root)/'fast_segments_v1/latest_submission.json'
    if not p.exists(): return
    jid=read(p)['job_id']
    r=subprocess.run(['sacct','-X','-n','-j',str(jid),'--format=State'],capture_output=True,text=True)
    terminal=('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE')
    rows=r.stdout.split()
    require(r.returncode==0 and rows and all(any(x.startswith(t) for t in terminal) for x in rows),
            'Cannot confirm previous job is terminal; do not duplicate submission')


def submit(root,repo,node,mode):
    empty_queue(); previous_done(root); node_spec(node)
    path,m=prepare(root); _,c,_,_,job=base(root)
    require(not (Path(root)/'formal/dataset/complete.json').exists(),'Dataset already complete; use verify')
    segment=choose_segment(m,c,job)
    if mode=='merge': require(segment is None,'All segments must finish before merge')
    elif segment is None:
        print('[READY TO MERGE] All shards present. Run run.sh merge; no job submitted.');return
    else: mode='collect'
    env=os.environ.copy(); env.update(G25F_SOURCE=m['source'],G25F_MANIFEST=str(path),
        G25F_SEGMENT=str(-1 if segment is None else segment),G25F_MODE=mode,G25F_GPU=node_spec(node))
    cmd=sbatch_command(root,repo,m['source'],node,mode)
    check=subprocess.run([cmd[0],'--test-only']+cmd[1:],capture_output=True,text=True,env=env)
    print('[Slurm test-only]\n'+check.stdout+check.stderr,flush=True)
    require(check.returncode==0,'test-only failed; no real submission/fallback')
    empty_queue()
    r=subprocess.run(cmd,capture_output=True,text=True,env=env)
    stamp=time.time_ns(); fp=Path(root)/'fast_segments_v1'
    attempt=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,mode=mode,
                 segment=segment,node=node,source=m['source'],time_ns=stamp)
    write(fp/f'submissions/attempt_{stamp}.json',attempt)
    require(r.returncode==0,'Submission failed; no automatic retry: '+r.stderr)
    raw=r.stdout.strip().split(';')[0]
    require(raw.isdigit(),'Ambiguous scheduler reply; inspect squeue before retrying')
    record=dict(attempt,job_id=int(raw),log=str(Path(root)/f'logs/fast_{raw}.out'))
    write(fp/'latest_submission.json',record,replace=True)
    print(f'[submitted] job={raw} jobs=1 nodes=1 gpus=4 node={node} mode={mode} segment={segment}',flush=True)
    print('[manual continuation] One segment only; no job arrays/dependencies/automatic next job.',flush=True)


def engine_integrity(c,job,ids,m):
    """Old completed shards remain valid; new shards must name this exact engine."""
    result={}
    for sid in ids:
        record=c.stage(job,sid,True)
        producer=record.get('producer')
        if producer is None:
            result[str(sid)]={'engine':'legacy_original','data_sha256':record['data_sha256']}
        else:
            require(producer['engine']=='target_only_adam_reuse_forward_v1','Unknown shard engine')
            require(producer['source_sha256']==m['source_sha256'],'Shard fast source mismatch')
            gate_path=Path(producer['gate_path'])
            require(sha(gate_path)==producer['gate_sha256'] and read(gate_path)['status']=='PASS','Missing/corrupt parity gate')
            result[str(sid)]=dict(producer,data_sha256=record['data_sha256'])
    return result


def collect_one(job,sid,gen,specs,lo,hi,limits,device,c,collector,producer):
    """Canonical legacy arrays and validation; only imported refinement is patched."""
    import numpy as np
    import torch
    ids=c.point_ids(job['cfg'],sid)
    points=np.load(Path(job['out'])/'x.npy',mmap_mode='r')[ids]
    a=c.args_for(job); n=len(ids); data=gen.allocate_yiming_style_arrays(n,a.q0_per_sensor,7,8)
    data['x'][:]=points; data['k'][:]=ids
    start=time.monotonic(); maximum=0.
    for i,(pid,x) in enumerate(zip(ids,points)):
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,int(pid)]))
        qi=gen.sample_q_batch(rng,limits,a.joint_names,a.q_init_per_p)
        results=gen.generate_for_one_point(x,qi,specs,lo,hi,None,a,device)
        sa=copy.copy(a);sa.seed=a.seed+1000003*(int(pid)-i)
        for s,r in enumerate(results):
            if r is None:continue
            gen.store_sensor_block(data,i,int(pid),x,s,r['q0'],r['g0'],r['sensor_margins0'],r['active_planes0'],r['occluded'],sa)
            count=int(data['num_sensor_q0'][i,s])
            if count:
                with torch.no_grad():
                    _,margin,_,_=gen.visibility_g_batch(torch.tensor(x,device=device),
                      torch.tensor(data['q'][i,:count,:,s],device=device),specs,
                      a.horizontal_fov_deg,a.vertical_fov_deg,a.z_min,a.z_max,a.delta)
                gs=(margin[:,s]-a.delta).cpu().numpy()
                err=float(np.max(np.abs(gs-data['g'][i,:count,s])));maximum=max(maximum,err)
                require(err<=2e-6 and np.all(np.abs(gs)<=a.epsilon+2e-6),'Retained Q0 recheck failed')
        if (i+1)%8==0 or i+1==n:
            print(f'[fast shard={sid}] {i+1}/{n} points; elapsed={time.monotonic()-start:.1f}s',flush=True)
    data['valid_point_fov']=data['valid_fov'].any(axis=(1,2))
    compact={k:data[k] for k in c.ROW_KEYS};stats=collector.validate_arrays(compact,ids,points,job)
    root=Path(job['out']);tmp=Path(tempfile.mkdtemp(prefix=f'.{sid:05d}.',dir=root/'shards'))
    try:
        np.savez_compressed(tmp/'data.npz',**compact)
        with np.load(tmp/'data.npz',allow_pickle=False) as z:
            collector.validate_arrays({k:z[k] for k in c.ROW_KEYS},ids,points,job)
        c.write(tmp/'complete.json',dict(status='COMPLETE',shard_id=sid,point_ids=ids.tolist(),
           job_sha256=c.sha(root/'job.json'),data_sha256=c.sha(tmp/'data.npz'),
           elapsed_seconds=time.monotonic()-start,fov_recheck_max_abs_m=maximum,producer=producer,**stats))
        os.rename(tmp,root/'shards'/f'{sid:05d}')
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print(f'[FAST SHARD COMPLETE] id={sid} points={n} valid_q0={stats["valid_q0"]}',flush=True)


def worker(manifest,segment,rank,request):
    import numpy as np
    import torch
    from fast_core import installed,parity_gate,ENGINE
    m,c,collector,_,job=verify_manifest(manifest); req=read(request)
    require(0<=rank<4 and req['segment']==segment,'Worker request mismatch')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Worker must see one assigned GPU')
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=torch.cuda.get_device_name(0)
    require(req['gpu_type'] in gpu,'Allocated GPU differs from requested type: '+gpu)
    free,total=torch.cuda.mem_get_info()
    require(free>=job['cfg']['min_free_gpu_gib']*(1<<30),'GPU memory admission failed; no fallback')
    gen=c.load_generator(job);ex=importlib.import_module('extract_visibility_zero_level_sets')
    a=c.args_for(job);device=torch.device('cuda:0');robot=gen.URDF.from_xml_file(a.urdf)
    limits=gen.get_joint_limits(robot,a.joint_names)
    lo=gen.torch_tensor(np.asarray(job['cfg']['q_min'],np.float32),device)
    hi=gen.torch_tensor(np.asarray(job['cfg']['q_max'],np.float32),device)
    specs=gen.prepare_chain_specs(robot,a.base_frame,a.sensor_frames,a.joint_names,device)
    require(np.array_equal(gen.SENSOR_CHAIN_MASKS,job['reference']['sensor_chain_masks']),'Mask mismatch')
    points=np.load(Path(job['out'])/'x.npy',mmap_mode='r')
    # Fixed, task-independent grid points; identical qualification on all devices.
    gate_ids=[(36*65+38)*41+22,(18*65+47)*41+30]
    print('[gate] real same-input old/fast comparisons; production sampling not started',flush=True)
    gate=parity_gate(gen,ex,specs,lo,hi,limits,a,device,points[gate_ids],gate_ids)
    gate.update(source_sha256=m['source_sha256'],base_job_sha256=m['base_job_sha256'],rank=rank)
    gp=Path(request).parent/f'gate_rank{rank}.json';write(gp,gate)
    producer=dict(engine=ENGINE,source_sha256=m['source_sha256'],gate_path=str(gp),gate_sha256=sha(gp),
                  gpu=gpu,torch=str(torch.__version__),cuda=torch.version.cuda,job_id=os.environ['SLURM_JOB_ID'])
    print(f'[PARITY PASS] rank={rank} refine-only speedup={gate["refinement_speedup"]:.2f}x; full throughput not yet measured',flush=True)
    stop=Path(req['stop_file'])
    def pause(*_): stop.touch(exist_ok=True)
    signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGUSR1,pause)
    deadline=float(req['deadline_unix']);done=0
    with installed(gen,ex):
        for sid in np.random.default_rng(PLAN_SEED+rank).permutation(m['segments'][segment]).tolist():
            if stop.exists() or time.time()>=deadline:break
            if stage_exists(c,job,sid):continue
            with c.lock(Path(job['out'])/'locks'/f'shard_{sid:05d}.lock') as acquired:
                if not acquired:continue
                if stage_exists(c,job,sid):continue
                collect_one(job,sid,gen,specs,lo,hi,limits,device,c,collector,producer);done+=1
    print(f'[WORKER PAUSED/DONE] rank={rank} new_shards={done}; completed shards preserved',flush=True)


def terminate(processes):
    for p in processes:
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGTERM)
    for p in processes:
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):os.killpg(p.pid,signal.SIGKILL)
            p.wait()


def batch(manifest,segment,mode):
    m,c,_,merger,job=verify_manifest(manifest);root=Path(m['root']);jid=os.environ.get('SLURM_JOB_ID','')
    require(jid.isdigit(),'Slurm allocation required')
    require(int(os.environ.get('SLURM_JOB_NUM_NODES',os.environ.get('SLURM_NNODES','0')))==1,'Exactly ONE node required')
    if mode=='merge':
        require(choose_segment(m,c,job) is None,'Incomplete collection; no final merge')
        prov=engine_integrity(c,job,range(c.nshards(job['cfg'])),m)
        write(root/'fast_segments_v1/dataset_provenance.json',prov,replace=True)
        merger.merge(job);merger.verify_dataset(job)
        print('[DATASET COMPLETE] original NPZ layout; provenance in fast_segments_v1',flush=True);return
    require(0<=segment<len(m['segments']),'Bad segment')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    require(len(visible)==len(set(visible))==4 and all(visible),'Exactly four allocated GPU identifiers required')
    folder=root/'fast_segments_v1/runs'/jid;folder.mkdir(parents=True,exist_ok=False)
    request=folder/'request.json';stop=folder/'STOP'
    write(request,dict(segment=segment,stop_file=str(stop),deadline_unix=time.time()+20*3600,
          gpu_type=os.environ['G25F_GPU'],manifest=str(manifest)))
    procs=[];handles=[]
    def pause(*_):
        stop.touch(exist_ok=True)
        print('[STOP REQUESTED] finish current shards; no new shard claimed',flush=True)
    old_usr=signal.signal(signal.SIGUSR1,pause);old_term=signal.signal(signal.SIGTERM,pause)
    subprocess.run(['nvidia-smi'],check=False)
    try:
        for rank,gpu in enumerate(visible):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            handle=(folder/f'worker_{rank}.log').open('a');handles.append(handle)
            procs.append(subprocess.Popen([sys.executable,'-u',str(HERE/'run.py'),'worker',
                '--manifest',str(manifest),'--segment',str(segment),'--rank',str(rank),'--request',str(request)],
                env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in procs):
            require(all(p.poll() in (None,0) for p in procs),'Worker/parity failure; inspect logs, no merge')
            time.sleep(1)
        require(all(p.returncode==0 for p in procs),'Worker failed')
    finally:
        terminate(procs)
        for h in handles:h.close()
        signal.signal(signal.SIGUSR1,old_usr);signal.signal(signal.SIGTERM,old_term)
    present=[s for s in m['segments'][segment] if stage_exists(c,job,s)]
    engine_integrity(c,job,present,m)
    complete=len(present)==len(m['segments'][segment])
    result=dict(status='SEGMENT_COMPLETE' if complete else 'SEGMENT_PAUSED',segment=segment,
                complete_shards=len(present),expected_shards=len(m['segments'][segment]),job_id=jid)
    write(folder/'result.json',result)
    print(json.dumps(result),flush=True)
    print('[NEXT] No task queued automatically. After exit use run.sh next.',flush=True)


def summary(root):
    root=Path(root);mp=root/'fast_segments_v1/manifest.json'
    require(mp.exists(),'Fast plan not prepared')
    m=read(mp);_,c,_,merger,job=base(root);merger.summary(job)
    rows=[];finished=0
    for seg,ids in enumerate(m['segments']):
        n=sum(stage_exists(c,job,s) for s in ids);finished+=int(n==len(ids))
        if 0<n<len(ids):print(f'Partial segment {seg+1}/{len(m["segments"])}: {n}/{len(ids)} shards')
    print(f'Segments complete: {finished}/{len(m["segments"])}; one submission advances one segment')
    for sid in range(c.nshards(job['cfg'])):
        if stage_exists(c,job,sid):rows.append(c.stage(job,sid,False))
    new=[r for r in rows if r.get('producer',{}).get('engine')=='target_only_adam_reuse_forward_v1']
    if new:
        sp=sum(r['elapsed_seconds'] for r in new)/sum(r['points'] for r in new)
        remaining=job['cfg']['grid_points']-sum(r['points'] for r in rows)
        print(f'Fast completed-shard mean={sp:.3f} seconds/point/worker; remaining sampling@4={remaining*sp/14400:.2f} h (no queue/merge)')
    print('Legacy/fast completed shards:',len(rows)-len(new),len(new))


def diagnostics(root,mode):
    root=Path(root);fp=root/'fast_segments_v1';p=fp/'latest_submission.json'
    if mode=='summary':summary(root);return
    if mode=='verify':
        mp,m=prepare(root);_,c,_,merger,job=base(root)
        present=[s for s in range(c.nshards(job['cfg'])) if stage_exists(c,job,s)]
        engine_integrity(c,job,present,m)
        print(f'[SHARDS VERIFIED] {len(present)}/{c.nshards(job["cfg"])}')
        if (Path(job['out'])/'dataset/complete.json').exists():merger.verify_dataset(job)
        return
    require(p.exists(),'No fast job submitted')
    r=read(p)
    if mode=='status':
        subprocess.run(['squeue','-j',str(r['job_id']),'-o','%.12i %.12T %.14M %.24R'],check=False)
        subprocess.run(['sacct','-X','-j',str(r['job_id']),'--format=JobID,State,ExitCode,Elapsed,NodeList'],check=False)
        print(r['log']);return
    if mode=='log':paths=[Path(r['log'])]
    elif mode=='ranks':paths=sorted((fp/'runs'/str(r['job_id'])).glob('worker_*.log'))
    else:
        import tarfile
        dest=root/f'fast_review_{time.time_ns()}.tar.gz'
        with tarfile.open(dest,'x:gz') as t:
            for directory in (fp,root/'fast_sources',root/'logs'):
                for f in sorted(directory.rglob('*')):
                    if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.py','.sh','.md','.log','.out'):
                        t.add(f,arcname=str(f.relative_to(root)),recursive=False)
        print('[pack]',dest,sha(dest));return
    for path in paths:
        print('===',path,'===',flush=True)
        if path.exists():subprocess.run(['tail','-n','25',str(path)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('policy','prepare','next','submit','resume','merge','batch','worker','status','summary','log','ranks','verify','pack'))
    p.add_argument('--root',type=Path);p.add_argument('--repo',type=Path)
    p.add_argument('--node',default='3090node3');p.add_argument('--manifest',type=Path)
    p.add_argument('--segment',type=int,default=-1);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--request',type=Path);p.add_argument('--action',default='collect')
    a=p.parse_args()
    if a.mode=='policy':node_spec(a.node);empty_queue()
    elif a.mode=='batch':batch(a.manifest,a.segment,a.action)
    elif a.mode=='worker':worker(a.manifest,a.segment,a.rank,a.request)
    elif a.mode in ('next','submit','resume','merge'):
        with locked(a.root/'.fast_segment_entry.lock'):submit(a.root,a.repo,a.node,a.mode)
    elif a.mode=='prepare':
        empty_queue()
        with locked(a.root/'.fast_segment_entry.lock'):prepare(a.root)
    else:diagnostics(a.root,a.mode)


if __name__=='__main__':main()
