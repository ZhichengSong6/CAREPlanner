#!/usr/bin/env python3
"""Run an isolated, original-R1-compatible scratch experiment; not a system change."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from support import (HERE, BASE, R1_SHA, STREAM, FORMAT, sha, read, write,
                     load, fingerprint, Pairs, Geometry, check_solver_parity)
from engine import Step

ROOT_DEFAULT='/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b'

def config():
    c=read(HERE/'protocol.json')
    if (c['format']!=FORMAT or c['steps']!=50000 or c['world']!=4 or
        not 0<c['weight']<=.1 or not 0<=c['warmup']<c['steps'] or
        c['ramp']<=0 or c['pairs_per_sensor_per_rank']<=0):raise ValueError('Invalid protocol')
    return c

def original_checkpoint(paths):
    p=Path(paths['r012'])/'formal/R1/final.pt'
    r=read(p.parent/'run.json')
    if (sha(p)!=R1_SHA or r.get('status')!='COMPLETE' or r.get('successful_updates')!=50000 or
        r.get('training_stream_sha256')!=STREAM):raise ValueError('Original R1 identity incomplete/wrong')
    cp=torch.load(p,map_location='cpu',weights_only=False)
    if cp.get('arm')!='R1' or cp.get('step')!=50000:raise ValueError('Wrong R1 checkpoint')
    return cp

def preflight(a):
    repo=a.repo.resolve(); paths={'r012':str(a.r012.resolve()),'cache':str(a.cache.resolve()),
                                 'pairs':str(a.pairs.resolve())}
    cp=original_checkpoint(paths);old=cp['args'];cfg=config()
    required={'seed':0,'stream_seed':190915,'amp':'fp16','lr':.001,
              'global_batch_x':4000,'batch_q':100,'microbatch_x':250,
              'weight_sdf':5.,'weight_grad':.1,'weight_eikonal':.01,'weight_tension':.01,
              'weight_union_objective':1.,'weight_sensor_objective':1.,'weight_consistency':.1}
    for k,v in required.items():
        if old.get(k)!=v:raise ValueError(f'Original R1 configuration mismatch: {k}={old.get(k)}')
    paths['data']=str(Path(old['data']).resolve())
    urdf=repo/'src/arm_description/urdf/Arm.urdf';paths['urdf']=str(urdf)
    if sha(urdf)!=cp['metadata']['urdf_sha256']:raise ValueError('Original URDF changed')
    sources=fingerprint(repo)
    pairs=Pairs(paths,verify=True)
    # Check model/geometry and actual runtime on real pairs before allocation.
    trainer=load('outcome_preflight_R012',repo/BASE/'train.py')
    api=trainer.base.load_repo_api(repo)
    geom=Geometry(repo,urdf,torch.device('cpu'),api['DEFAULT_JOINT_NAMES'],api['DEFAULT_SENSOR_FRAMES'])
    import numpy as np
    with np.load(paths['data'],allow_pickle=True) as z:
        lo=torch.as_tensor(np.array(z['q_min']),dtype=torch.float32)
        hi=torch.as_tensor(np.array(z['q_max']),dtype=torch.float32)
        masks=torch.as_tensor(np.array(z['sensor_chain_masks']),dtype=torch.float32)
    if masks.shape!=(8,7):raise ValueError('Original mask must be [8,7]')
    model=trainer.models.build_model('R1').eval().requires_grad_(False)
    model.load_state_dict(cp['model_state'],strict=True)
    if model.parameter_count()!=2184329:raise ValueError('Not exact R1 architecture')
    batch=pairs.batch('train',0,0,torch.device('cpu'),1)
    gp=geom.parity(batch)
    rp=check_solver_parity(repo,model,batch,masks,lo,hi)
    with torch.no_grad():
        x,qm,qp,s=batch
        valid=(geom(x,qm,s)<-1e-5)&(geom(x,qp,s)>1e-5)
        if not valid.all():raise ValueError('Preflight train pairs lack robust local witness')
        bx,bm,bp,bs=pairs.batch('val',0,0,torch.device('cpu'),1)
        if not ((geom(bx,bm,bs)<-1e-5)&(geom(bx,bp,bs)>1e-5)).all():raise ValueError('Preflight validation witness invalid')
    # Full raw-bank hash prevents a changed online-label dataset from going unnoticed.
    source_manifest=read(Path(read(a.cache/'manifest.json')['source'])/'manifest.json')
    if sha(paths['data'])!=source_manifest['original_data_sha256']:raise ValueError('Original bank hash mismatch')
    files=pairs.source_hashes | {paths['data']:source_manifest['original_data_sha256'],
                                paths['urdf']:sha(urdf)}
    spec={'format':FORMAT,'protocol':cfg,'paths':paths,'base_args':old,'source':sources,
          'inputs':files,'pair_counts':pairs.counts,'pair_manifest_sha256':pairs.identity,
          'reference_R1_sha256':R1_SHA,'reference_stream':STREAM,
          'geometry_error_m':gp,'solver_q_error_rad':rp}
    if a.spec.exists():
        if read(a.spec)!=spec:raise ValueError('Existing frozen job specification differs; use a new root')
    else:write(a.spec,spec)
    print('[preflight] exact R1 + real pair schema + geometry + original-solver value parity PASS',flush=True)
    print('[preflight] 400k original pairs/update; scratch 50k; outcome warmup 10k, ramp 5k, weight .01',flush=True)

def validate_spec(repo,spec, *, full=True):
    if spec['format']!=FORMAT or spec['protocol']!=config() or fingerprint(repo)!=spec['source']:
        raise ValueError('Frozen experiment/source mismatch')
    if full:
        for p,h in spec['inputs'].items():
            if sha(p)!=h:raise ValueError('Frozen input changed: '+p)

def launch(a):
    repo=a.repo.resolve();spec=read(a.spec);cfg=spec['protocol'];paths=spec['paths']
    validate_spec(repo,spec,full=True)
    out=a.out.resolve();start=0
    if a.resume:
        p=out/'latest.pt'
        cp=torch.load(p,map_location='cpu',weights_only=False)
        if cp.get('format')!=FORMAT or cp.get('metadata',{}).get('outcome_spec')!=spec:
            raise ValueError('Resume only this identical outcome run')
        start=int(cp['step'])
        if not 0<start<cfg['steps']:raise ValueError('No interrupted steps left')
    trainer=load('outcome_original_trainer',repo/BASE/'train.py')
    trainer.FORMAT=FORMAT
    def fingerprints():
        actual=fingerprint(repo)
        if actual!=spec['source']:raise ValueError('Source changed after submission')
        return actual
    trainer.source_fingerprints=fingerprints
    original_prepare,original_step=trainer.base.prepare_batch,trainer.base.run_prepared_batch
    holder={}
    def prepare(api,dataset,oracle,args,device,split):
        if not holder:
            if dataset.x_cpu.shape[1]!=3:raise ValueError('Bad R1 dataset')
            import numpy as np
            cachex=np.load(Path(paths['cache'])/'x.npy',mmap_mode='r')
            if cachex.shape!=tuple(dataset.x_cpu.shape) or not np.allclose(cachex,dataset.x_cpu.numpy(),atol=1e-6,rtol=0):
                raise ValueError('R1/V4 x mismatch')
            ps=Pairs(paths,{'train':dataset.train_indices_np,'val':dataset.val_indices_np},verify=False)
            if ps.identity!=spec['pair_manifest_sha256'] or ps.counts!=spec['pair_counts']:
                raise ValueError('Pair split/identity changed')
            geometry=Geometry(repo,Path(paths['urdf']),device,api['DEFAULT_JOINT_NAMES'],api['DEFAULT_SENSOR_FRAMES'])
            lo,hi=dataset.q_limits(device);masks=dataset.sensor_masks(device)
            first=ps.batch('train',0,dist.get_rank(),device,1);geometry.parity(first)
            holder['step']=Step(original_step,trainer.base,ps,geometry,masks,lo,hi,cfg,start)
            trainer.base.run_prepared_batch=holder['step']
        return original_prepare(api,dataset,oracle,args,device,split)
    trainer.base.prepare_batch=prepare
    old_save=trainer.save_checkpoint
    def save(path,ddp,opt,scheduler,scaler,args,step,best,stats,stream,metadata):
        if holder['step'].step!=step:raise RuntimeError('Auxiliary/update counter mismatch')
        return old_save(path,ddp,opt,scheduler,scaler,args,step,best,stats,stream,
                        dict(metadata,outcome_spec=spec))
    trainer.save_checkpoint=save
    args=dict(spec['base_args']);args.update(repo=str(repo),data=paths['data'],urdf=paths['urdf'],
          out_dir=str(out),steps=cfg['steps'],resume_training=str(out/'latest.pt') if a.resume else None)
    # Reuse the exact baseline Namespace, not guessed renamed flags.
    trainer.parse_args=lambda:argparse.Namespace(**args)
    trainer.main()
    if int(os.environ.get('LOCAL_RANK',0))!=0:return
    r=read(out/'run.json')
    if r['training_stream_sha256']!=STREAM or r['successful_updates']!=50000:
        raise ValueError('Base x/q stream mismatch; do not promote')
    validate_spec(repo,spec,full=True)
    cp=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    # Deployment format is the existing architecture's supported runtime format.
    # Provenance remains explicitly this new experiment, NOT historical R1.
    runtime={k:cp[k] for k in ('model_state','arm','step','architecture','initialization','args','routing','metadata')}
    runtime.update(format='care_h9_scratch50k_r012_v1',training_format=FORMAT,completed=True,frozen_parameters=0,
                   training_checkpoint_sha256=sha(out/'final.pt'),source_sha256=spec['source'],
                   output_semantics='hierarchical_union_plus_per_sensor_signed_visibility_cdf',out_dim=9)
    tmp=out/'runtime.pt.tmp';torch.save(runtime,tmp);os.replace(tmp,out/'runtime.pt')
    from per_sensor_visibility_runtime import build_per_sensor_model
    view,_=build_per_sensor_model(str(out/'runtime.pt'),torch.device('cpu'))
    full=trainer.models.build_model('R1');full.load_state_dict(cp['model_state'],strict=True)
    z=torch.zeros(3,10,requires_grad=True)
    one,two=full(z)[:,1:],view(z)
    verr=float((one-two).detach().abs().max())
    g1=torch.autograd.grad(one.sum(),z)[0];g2=torch.autograd.grad(two.sum(),z)[0]
    gerr=float((g1-g2).abs().max())
    if verr>1e-6 or gerr>1e-6:raise ValueError('Runtime export changed value/gradient')
    r.update(runtime_value_error=verr,runtime_gradient_error=gerr,experiment=FORMAT,baseline_stream_exact_match=True,
             runtime_sha256=sha(out/'runtime.pt'),solver_evaluation='NOT_RUN')
    write(out/'run.json',r)
    print('[done] scratch50k and original stream verified; runtime.pt saved; superiority NOT_TESTED',flush=True)

def verify(a):
    spec=read(a.spec);validate_spec(a.repo.resolve(),spec,full=False)
    r=read(a.out/'run.json')
    if (r.get('status')!='COMPLETE' or r.get('experiment')!=FORMAT or r.get('successful_updates')!=50000
        or r.get('training_stream_sha256')!=STREAM or not r.get('baseline_stream_exact_match')
        or sha(a.out/'final.pt')!=r['final_sha256'] or sha(a.out/'runtime.pt')!=r['runtime_sha256']):
        raise ValueError('Outcome training not fully verified')
    print('[verified] complete 50k original stream; final.pt + runtime.pt checksums PASS',flush=True)
    print(json.dumps(r,indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=('preflight','train','verify'))
    ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--r012',type=Path,default=Path(ROOT_DEFAULT)/'h9_scratch50k_r012_v1')
    ap.add_argument('--cache',type=Path,default=Path(ROOT_DEFAULT)/'h9_v4_scratch_v1/training_cache')
    ap.add_argument('--pairs',type=Path,default=Path(ROOT_DEFAULT)/'h9_paired_slope_scratch_v1/formal/pairs')
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2)
    if a.mode=='preflight':preflight(a)
    elif a.mode=='train':launch(a)
    else:verify(a)
if __name__=='__main__':main()
