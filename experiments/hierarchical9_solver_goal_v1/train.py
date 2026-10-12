#!/usr/bin/env python3
"""Independent R1-scratch + differentiable short-update geometry experiment.

Uses original R012 trainer/400k stream/Adam/DDP/AMP/scheduler WITHOUT source edits.
Keeps inference architecture/format compatible; lineage recorded separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import torch.distributed as dist
from goal import load, WitnessPool, paired_planes, geometry_objective, weight_at, inject_once

HERE = Path(__file__).resolve().parent
R1_SHA = '4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002'
CACHE_SHA = '060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc'
PINNED = {
'experiments/hierarchical9_scratch50k_r012_v1/train.py':'8092a791585dff8ac6ab5e562d158450b1095df4',
'experiments/hierarchical9_scratch50k_r012_v1/model.py':'0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b',
'experiments/hierarchical9_scratch_v1/train.py':'88d365045bb1c53cf71f5b8fbd9f8db74a5c2ae9',
'experiments/hierarchical9_scratch_v1/model.py':'33580154af2e9cda22f9c64f15cd5b4f5417a86f',
'experiments/hierarchical9_scratch_v1/objective.py':'a53539d27fb003d6b06d46da80ca0c7a93b1230c',
'experiments/hierarchical9_boundary_audit_v1/oracle.py':'de6d93c02753da9f347a314643c2feca0748629e',
'experiments/hierarchical9_paired_slope_scratch_v1/data.py':'c5fc55461f6794960bc319a5c24f798b105f7e19'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); os.replace(tmp,p)


def sources(repo):
    for rel, expect in PINNED.items():
        b = (repo/rel).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if actual != expect: raise ValueError(f'Frozen dependency changed: {rel}')
    scripts = repo/'src/care_visibility_cdf/scripts'
    deps = list(PINNED) + [str(p.relative_to(repo)) for p in scripts.glob('*.py')]
    return {**{rel:sha(repo/rel) for rel in sorted(set(deps))},
            **{'solver_goal/'+p.name:sha(p) for p in sorted(HERE.iterdir())
               if p.suffix in ('.py','.sh','.json')}}


def open_cache(repo, path):
    mod = load('sg_cache',repo/'experiments/hierarchical9_paired_slope_scratch_v1/data.py')
    return mod.Cache(path, verify=False)


def oracle_for(repo, urdf, device):
    scripts = str(repo/'src/care_visibility_cdf/scripts')
    if scripts not in sys.path: sys.path.insert(0,scripts)
    api = __import__('train_signed_visibility_cdf_pairwise_replace')
    mod = load('sg_sensor_oracle',repo/'experiments/hierarchical9_boundary_audit_v1/oracle.py')
    return mod.SensorOracle(urdf, device, api.DEFAULT_JOINT_NAMES, api.DEFAULT_SENSOR_FRAMES)


def splits(data, cfg):
    with np.load(data, allow_pickle=False) as bank:
        x = np.asarray(bank['x'],np.float32)
        ids = np.flatnonzero(np.asarray(bank['valid_fov'],bool).any(axis=(1,2)))
        mask = np.asarray(bank['sensor_chain_masks'],np.float32)
        lo, hi = np.asarray(bank['q_min'],np.float32), np.asarray(bank['q_max'],np.float32)
    np.random.default_rng(cfg['baseline']['seed']).shuffle(ids)
    n = min(cfg['baseline']['val_count'],max(1,len(ids)//10))
    if mask.shape != (8,7) or not np.isin(mask,[0,1]).all(): raise ValueError('Invalid sensor masks')
    return x, ids[n:], ids[:n], mask, lo, hi


def checked_files(root, specs, is_cache):
    for rel, spec in specs.items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root) or not p.is_file(): raise ValueError(f'Unsafe/missing manifest file {p}')
        expect=spec['sha256'] if is_cache else spec
        if (is_cache and p.stat().st_size != spec['bytes']) or sha(p)!=expect:
            raise ValueError(f'Corrupt cache/pairs file: {p}')


def prepare(a):
    cfg = read(HERE/'config.json'); src = sources(a.repo)
    old_dir=a.r012/'formal/R1'; cp_path=old_dir/'final.pt'
    run=read(old_dir/'run.json')
    if (sha(cp_path)!=R1_SHA or run.get('final_sha256')!=R1_SHA
            or run.get('status')!='COMPLETE' or run.get('successful_updates')!=50000):
        raise ValueError('Original R1 identity/completion failed')
    cp=torch.load(cp_path,map_location='cpu',weights_only=False)
    for key,value in cfg['baseline'].items():
        if key in ('log_every','save_every','val_every'): continue
        if cp['args'].get(key)!=value: raise ValueError(f'Not identical baseline config: {key}')
    data=Path(cp['args']['data']).resolve(); urdf=a.repo/'src/arm_description/urdf/Arm.urdf'
    if cp['metadata']['urdf_sha256'] != sha(urdf): raise ValueError('R1 URDF changed')
    cache=open_cache(a.repo,a.cache)
    if cache.identity!=CACHE_SHA: raise ValueError('Wrong cache')
    checked_files(a.cache,cache.manifest['files'],True)
    pm=read(a.pairs/'manifest.json'); checked_files(a.pairs,pm['fields'],False)
    x,tr,va,masks,lo,hi=splits(data,cfg)
    if x.shape!=cache.x.shape or not np.allclose(x,cache.x,rtol=0,atol=1e-6):
        raise ValueError('R1 and witness x indexing mismatch')
    # Also verify the original dataset against its existing production manifest.
    prod=Path(cache.manifest['source']); prod_m=read(prod/'manifest.json')
    if sha(prod/'manifest.json')!=cache.manifest['source_manifest_sha256']:
        raise ValueError('Production/cache manifest mismatch')
    data_sha=sha(data)
    if data_sha!=prod_m['original_data_sha256']: raise ValueError('Original dataset SHA mismatch')
    pool=WitnessPool(cache,a.pairs,tr,va,cfg)
    oracle=oracle_for(a.repo,urdf,torch.device('cpu'))
    parity=0.
    for sp in ('train','val'):
        batch=pool.draw(sp,1,0,torch.device('cpu'))
        for s,(xx,qm,qp) in enumerate(batch):
            for q in (qm,qp):
                pred=paired_planes(oracle,xx,q,s)
                ref=torch.cat([oracle.planes(xx[i],q[i:i+1],s) for i in range(len(q))],0)
                parity=max(parity,float((ref-pred).abs().max()))
            if not (paired_planes(oracle,xx,qm,s).min(-1).values<0).all(): raise ValueError('Bad negative witness')
            if not (paired_planes(oracle,xx,qp,s).min(-1).values>0).all(): raise ValueError('Bad positive witness')
        oracle.verify_against_upstream(batch[0][0][0],batch[0][2][:3])
    if parity>2e-6: raise ValueError('Differentiable paired FOV mismatch')
    # Real production-input, real R1 model and exact FK end-to-end gradient gate.
    # This is a read-only CPU check before GPU allocation; never an optimizer step.
    arch=load('sg_preflight_arch',a.repo/'experiments/hierarchical9_scratch50k_r012_v1/model.py')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['baseline']['seed']); model=arch.build_model('R1')
        example=pool.draw('train',1,0,torch.device('cpu'))
        loss,met=geometry_objective(model,example,oracle,torch.tensor(masks),torch.tensor(lo),torch.tensor(hi),cfg)
        grads=torch.autograd.grad(loss,tuple(model.parameters()),allow_unused=True)
        live=[g for g in grads if g is not None]
        if not live or not all(torch.isfinite(g).all() for g in live):
            raise ValueError('Real production FOV -> q-update -> R1 parameter gradient is invalid')
        grad_norm=float(torch.stack([g.detach().float().square().sum() for g in live]).sum().sqrt())
        if grad_norm==0: raise ValueError('No trainable geometry signal on real preflight batch')
    del model,loss,met,grads,live
    print(f'[preflight] REAL model/FK witness gradient gate PASS, raw parameter norm={grad_norm:.6g}',flush=True)
    job={'cfg':cfg,'source':src,'repo':str(a.repo),'r012':str(a.r012),'cache':str(a.cache),
         'pairs':str(a.pairs),'data':str(data),'urdf':str(urdf),'out':str(a.out),
         'cache_sha':cache.identity,'pair_sha':sha(a.pairs/'manifest.json'),
         'data_sha':data_sha,'data_bytes':data.stat().st_size,'data_mtime_ns':data.stat().st_mtime_ns,
         'r1_sha':R1_SHA,'base_stream':run['training_stream_sha256'],
         'r1_args':cp['args'],'witness_strata':pool.counts,'paired_geometry_max_error':parity}
    write(a.job,job)
    print('[preflight] original source/data/checkpoint and existing witness files VERIFIED',flush=True)
    print('[preflight] R1 train/val x isolated; true +/- witnesses; upstream FOV agreement',flush=True)
    print('[job]',a.job,flush=True)


def verify_job(job):
    repo=Path(job['repo'])
    if sources(repo)!=job['source'] or read(HERE/'config.json')!=job['cfg']:
        raise ValueError('Job source/config changed')
    data=Path(job['data'])
    if data.stat().st_size!=job['data_bytes'] or data.stat().st_mtime_ns!=job['data_mtime_ns']:
        raise ValueError('Original bank changed since preflight')
    if sha(Path(job['pairs'])/'manifest.json')!=job['pair_sha']:
        raise ValueError('Witness manifest changed')
    if sha(Path(job['cache'])/'manifest.json')!=job['cache_sha']:
        raise ValueError('Cache manifest changed')


def train(a):
    job=read(a.job); verify_job(job)
    repo,out,cfg=Path(job['repo']),Path(job['out']),job['cfg']
    resume=out/'latest.pt'
    start=0
    if a.resume:
        if (out/'final.pt').exists(): raise ValueError('Already finished')
        cp=torch.load(resume,map_location='cpu',weights_only=False)
        if (cp.get('metadata',{}).get('solver_goal_v1')!=job
                or cp.get('source_sha256')!=job['source']): raise ValueError('Resume lineage mismatch')
        start=int(cp['step'])
    elif out.exists() and any(out.iterdir()): raise FileExistsError('No overwrite of training output')
    core=load('sg_original_trainer',repo/'experiments/hierarchical9_scratch50k_r012_v1/train.py')
    def fingerprint():
        verify_job(job)
        return job['source']
    core.source_fingerprints=fingerprint
    prep0,run0=core.base.prepare_batch,core.base.run_prepared_batch
    holder={'step':start}
    def prepared(api,dataset,oracle,args,device,split):
        if 'pool' not in holder:
            cache=open_cache(repo,Path(job['cache']))
            if cache.x.shape!=tuple(dataset.x_cpu.shape) or not np.allclose(cache.x,dataset.x_cpu.numpy(),rtol=0,atol=1e-6):
                raise ValueError('Live R1 dataset and paired witness x differ')
            holder['pool']=WitnessPool(cache,Path(job['pairs']),dataset.train_indices_np,dataset.val_indices_np,cfg)
            holder['oracle']=oracle_for(repo,Path(job['urdf']),device)
            holder['masks']=dataset.sensor_masks(device)
            holder['lo'],holder['hi']=dataset.q_limits(device)
        return prep0(api,dataset,oracle,args,device,split)
    core.base.prepare_batch=prepared
    def run(ddp,batches,counts,weights,args,optimizer=None,scaler=None):
        training=optimizer is not None
        if training: holder['step']+=1
        step=holder['step']; w=weight_at(step,cfg)
        monitor=(not training) or step==1 or step%args.log_every==0 or step==cfg['steps']
        if w==0 and training:
            result=run0(ddp,batches,counts,weights,args,optimizer,scaler)
            if monitor: result['solver_goal']={'weight':0.,'stage':'baseline_warmup'}
            return result
        batch=holder['pool'].draw('train' if training else 'val',step if training else 0,dist.get_rank(),counts.device)
        def extra(model):
            loss,met=geometry_objective(model,batch,holder['oracle'],holder['masks'],holder['lo'],holder['hi'],cfg)
            return loss*w,met.detach()
        if training:
            result,met=inject_once(core.base,run0,ddp,batches,counts,weights,args,optimizer,scaler,extra)
        else:
            result=run0(ddp,batches,counts,weights,args)
            _,met=extra(ddp.module)
        if monitor:
            dist.all_reduce(met); met/=dist.get_world_size()
            result['solver_goal']={'weight':w,'surrogate_only':True,
                **{key:met[:,i].cpu().tolist() for i,key in enumerate((
                    'loss','candidate_margin_m','outside_start_fov_pass','inside_start_fov_pass',
                    'witness_margin_m','initial_outside_margin_m','target_margin_m','displacement_rad'))}}
            core.log(f'[goal] step={step} weight={w:.5f} loss={met[:,0].mean().item():.5f} '+
                f'outside_surrogate={met[:,2].mean().item():.4f} inside_surrogate={met[:,3].mean().item():.4f}')
        return result
    core.base.run_prepared_batch=run
    save0=core.save_checkpoint
    def save(path,ddp,opt,sched,scaler,args,step,best,last,stream,metadata):
        return save0(path,ddp,opt,sched,scaler,args,step,best,last,stream,
                     dict(metadata,solver_goal_v1=job))
    core.save_checkpoint=save
    argv=['R1-original','--repo',str(repo),'--data',job['data'],'--urdf',job['urdf'],
          '--out-dir',str(out),'--steps',str(cfg['steps'])]
    for key,v in cfg['baseline'].items(): argv += ['--'+key.replace('_','-'),str(v)]
    if a.resume: argv += ['--resume-training',str(resume)]
    sys.argv=argv
    core.main()
    if int(os.environ.get('LOCAL_RANK','0'))!=0:return
    r=read(out/'run.json')
    if r.get('training_stream_sha256')!=job['base_stream'] or r.get('successful_updates')!=cfg['steps']:
        write(out/'solver_goal_complete.json',{'status':'INVALID_BASE_STREAM'})
        raise RuntimeError('Not identical original R1 sample stream')
    write(out/'solver_goal_complete.json',{'status':'COMPLETE','base_stream_match':True,
        'experiment':cfg['name'],'steps':cfg['steps'],'final_sha256':sha(out/'final.pt'),
        'original_r1_sha':R1_SHA,'solver_evaluation':'NOT_RUN'})
    print('[done] scratch 50k finished, original R1 x/q stream EXACT MATCH; solver NOT_RUN',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('prepare','train'))
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    for key in ('repo','r012','cache','pairs','out'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.mode=='prepare':
        for key in ('repo','r012','cache','pairs','out'):
            if getattr(a,key) is None:p.error('--'+key+' required for prepare')
            setattr(a,key,getattr(a,key).resolve())
        protected=[a.r012,a.cache,a.pairs]
        if any(a.out==s or a.out.is_relative_to(s) or s.is_relative_to(a.out) for s in protected):
            raise ValueError('Output overlaps protected input')
        prepare(a)
    else:train(a)

if __name__=='__main__':main()
