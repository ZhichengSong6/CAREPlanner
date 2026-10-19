"""Matched two-arm fine tuning: fresh Adam in BOTH arms, 20k successful updates."""
from __future__ import annotations
from dataclasses import asdict
from datetime import timedelta
import argparse,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from k2_common import (FORMAT,TEACHER_SHA,ARMS,require,read,write,sha,stage,modules,
                       load_model,verify,model_digest,finish)
from keep import ReplayBank,objective as keep_objective

def learning_rate(step,cfg):
    t=(step-1)/max(1,cfg['updates']-1)
    return cfg['lr_end']+.5*(cfg['lr_start']-cfg['lr_end'])*(1+math.cos(math.pi*t))

def save(path,job,arm,model,opt,scaler,step,stream,base,replay_sha):
    rng=[None]*dist.get_world_size();dist.all_gather_object(rng,base.rng_state())
    if dist.get_rank()==0:
        payload=dict(format='care_h9_scratch50k_r012_v1',arm='R1',experiment_format=FORMAT,
          variant=arm,step=50000+step,stage_step=step,initialization='finetune_from_solver_goal_50000',
          parent_sha256=TEACHER_SHA,model_state=model.module.state_dict(),optimizer_state=opt.state_dict(),
          scaler_state=scaler.state_dict(),rng_by_rank=rng,job=job,replay_sha256=replay_sha,
          training_stream_sha256=stream.chain,training_stream_updates=stream.count,
          architecture=model.module.architecture(),args=job['parent']['r1_args'],
          metadata={'keep_v2':{'job':job,'arm':arm},'urdf_sha256':sha(job['urdf'])})
        base.atomic_save(payload,path)
    dist.barrier()

def checked_checkpoint(path,job,arm,replay_sha):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    require(cp.get('job')==job and cp.get('variant')==arm and cp.get('experiment_format')==FORMAT,'Resume lineage mismatch')
    require(cp.get('parent_sha256')==TEACHER_SHA and cp.get('replay_sha256')==replay_sha,'Resume parent/replay mismatch')
    require(0<cp.get('stage_step',-1)<=job['cfg']['updates'],'Wrong stage update')
    return cp

def train(job,arm):
    require(arm in ARMS,'Unknown arm');verify(job)
    require(torch.cuda.is_available() and 'LOCAL_RANK' in os.environ,'Use four-rank torchrun')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    try:
        require(dist.get_world_size()==4,'Four ranks required')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        cfg=job['cfg'];out=Path(job['out'])/arm;out.mkdir(exist_ok=True)
        core,goal,st=modules(job);base=core.base
        replay=ReplayBank(Path(job['out'])/'replay',TEACHER_SHA)
        replay_sha=sha(Path(job['out'])/'replay/complete.json')
        if (out/'complete.json').exists(): stage(out);return
        base.seed_everything(0)
        api=base.load_repo_api(Path(job['deps']))
        args=argparse.Namespace(**job['parent']['cfg']['baseline'])
        args.steps=cfg['updates'];args.log_every=cfg['log_every'];args.val_every=cfg['val_every'];args.save_every=cfg['save_every']
        args.data=job['data'];args.urdf=job['urdf'];args.repo=job['deps']
        dataset=api['VisibilityQ0Dataset'](path=job['data'],val_count=args.val_count,seed=args.seed)
        with np.load(Path(job['out'])/'splits.npz',allow_pickle=False) as z:
            require(np.array_equal(np.sort(dataset.train_indices_np),np.sort(z['train'])) and np.array_equal(np.sort(dataset.val_indices_np),np.sort(z['val'])),'Actual original dataset split differs')
        original_oracle=api['PinocchioFOVOracle'](urdf_path=job['urdf'],joint_names=api['DEFAULT_JOINT_NAMES'],
             sensor_frames=api['DEFAULT_SENSOR_FRAMES'],horizontal_fov_deg=args.horizontal_fov_deg,
             vertical_fov_deg=args.vertical_fov_deg,z_min=args.z_min,z_max=args.z_max,delta=args.delta)
        oracle=st.oracle_for(Path(job['deps']),Path(job['urdf']),device)
        cache=st.open_cache(Path(job['deps']),Path(job['cache']))
        pool=goal.WitnessPool(cache,Path(job['pairs']),dataset.train_indices_np,dataset.val_indices_np,job['parent']['cfg'])
        masks=dataset.sensor_masks(device);lo,hi=dataset.q_limits(device)
        model,parent=load_model(core,job['teacher'],TEACHER_SHA,device)
        require(model_digest(model)==job['initial_model_digest'],'Initial tensor mismatch')
        del parent
        ddp=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False)
        opt=torch.optim.Adam(ddp.parameters(),lr=cfg['lr_start'])
        scaler=torch.cuda.amp.GradScaler(enabled=args.amp=='fp16',init_scale=65536.)
        weights=core.obj.LossWeights(**{k:getattr(args,'weight_'+k) for k in asdict(core.obj.LossWeights())})
        base.seed_everything(cfg['stream_seed'])
        start=0;chain='0'*64;count=0
        if (out/'latest.pt').exists():
            cp=checked_checkpoint(out/'latest.pt',job,arm,replay_sha)
            start=cp['stage_step'];ddp.module.load_state_dict(cp['model_state'],strict=True)
            opt.load_state_dict(cp['optimizer_state']);scaler.load_state_dict(cp['scaler_state'])
            base.restore_rng(cp['rng_by_rank'][rank]);chain=cp['training_stream_sha256'];count=cp['training_stream_updates']
            del cp
        elif any(p.name not in ('metrics.jsonl',) for p in out.iterdir()):
            raise RuntimeError('Unfinished output without resumable checkpoint; preserve for review')
        stream=core.StreamChain(api,chain,count);stream.install()
        # Crash-safe metric append: discard only entries AFTER the restored checkpoint,
        # preserving the previous log in a uniquely named backup.
        if rank==0 and (out/'metrics.jsonl').exists():
            import json
            p=out/'metrics.jsonl';saved=p.read_text();retained=[]
            for line in saved.splitlines():
                try:r=json.loads(line)
                except ValueError:break
                if r['step']<=start:retained.append(line)
            os.rename(p,out/f'metrics.before_resume.{time.time_ns()}.jsonl')
            p.write_text('\n'.join(retained)+ ('\n' if retained else ''))
        dist.barrier()
        if rank==0: print(f'[train] {arm} from teacher50k, new Adam, start={start}, additional={cfg["updates"]}',flush=True)
        for step in range(start+1,cfg['updates']+1):
            t0=time.perf_counter()
            for g in opt.param_groups:g['lr']=learning_rate(step,cfg)
            batches,counts=base.prepare_batch(api,dataset,original_oracle,args,device,'train')
            geom_batch=pool.draw('train',50000+step,rank,device)
            rb=replay.draw('train',step,rank,cfg,device) if arm=='B_keep' else None
            def extra(m):
                gl,gm=goal.geometry_objective(m,geom_batch,oracle,masks,lo,hi,job['parent']['cfg'])
                kl=gl*0.;km=gm.new_zeros((8,4))
                if rb is not None:kl,km=keep_objective(m,rb,masks,lo,hi,cfg,True)
                return .01*gl+(cfg['keep_weight']*kl if rb is not None else kl),torch.cat((gm.detach(),km.detach()),1)
            stats,met=goal.inject_once(base,base.run_prepared_batch,ddp,batches,counts,weights,args,opt,scaler,extra)
            del batches,geom_batch,rb
            if step==1 or step%cfg['log_every']==0 or step==cfg['updates']:
                dist.all_reduce(met);met/=4;torch.cuda.synchronize(device)
                stats.update(step=step,split='train',variant=arm,lr=learning_rate(step,cfg),elapsed_sec=time.perf_counter()-t0,
                     geometry=met[:,:8].cpu().tolist(),keep=met[:,8:].cpu().tolist(),training_stream_sha256=stream.chain)
                if rank==0:
                    import json
                    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(stats,allow_nan=False)+'\n')
                    print(f'[{arm}] {step}/{cfg["updates"]} base={stats["loss"]:.6f} geom={met[:,0].mean():.4f} keep={met[:,8].mean():.4f}',flush=True)
            if step==1 or step%cfg['val_every']==0 or step==cfg['updates']:
                vb,vc=base.prepare_batch(api,dataset,original_oracle,args,device,'val')
                v=base.run_prepared_batch(ddp,vb,vc,weights,args);del vb
                gb=pool.draw('val',0,rank,device);_,gm=goal.geometry_objective(ddp.module,gb,oracle,masks,lo,hi,job['parent']['cfg'])
                rb=replay.draw('val',0,rank,cfg,device);_,km=keep_objective(ddp.module,rb,masks,lo,hi,cfg,False)
                met=torch.cat((gm,km),1).detach();dist.all_reduce(met);met/=4
                if rank==0:
                    import json
                    v.update(step=step,split='val',variant=arm,geometry=met[:,:8].cpu().tolist(),keep=met[:,8:].cpu().tolist())
                    with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
                del gb,rb,gm,km
            if step%cfg['save_every']==0 or step==cfg['updates']:
                save(out/'latest.pt',job,arm,ddp,opt,scaler,step,stream,base,replay_sha)
        # If resumed after the final latest.pt but before publication, finalize without retraining.
        require(stream.count==cfg['updates'] if rank==0 else True,'Training stream update count mismatch')
        save(out/'final.pt',job,arm,ddp,opt,scaler,cfg['updates'],stream,base,replay_sha)
        if rank==0:
            payload=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
            lite={k:v for k,v in payload.items() if k not in ('optimizer_state','scaler_state','rng_by_rank')}
            base.atomic_save(lite,out/'runtime.pt')
            write(out/'run.json',dict(status='COMPLETE',variant=arm,additional_updates=cfg['updates'],
                initial_model_sha256=TEACHER_SHA,final_sha256=sha(out/'final.pt'),runtime_sha256=sha(out/'runtime.pt'),
                base_stream=stream.chain,replay_sha256=replay_sha,optimizer='fresh Adam; matched LR schedule',
                final_selection='predeclared_20000',solver_evaluation='NOT_RUN'),replace=True)
            finish(out,parent_sha256=TEACHER_SHA,additional_updates=cfg['updates'],base_stream=stream.chain)
        dist.barrier()
    finally:
        if dist.is_initialized():dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args();train(read(a.job),a.arm)
if __name__=='__main__':main()
