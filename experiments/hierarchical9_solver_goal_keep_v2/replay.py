"""Fixed-attempt reference bank. Only original train x feed B's training loss."""
from __future__ import annotations
from pathlib import Path
import os,tempfile
import numpy as np
import torch
from k2_common import (require,read,write,write_lines,lines,sha,stage,finish,verify,
                       runtime,load_model,model_digest,TEACHER_SHA)
from keep import predict

def create_starts(job,rank):
    verify(job);out=Path(job['out']);cfg=job['cfg'];dest=out/'reference_inputs'/f'rank{rank}'
    if dest.exists():stage(dest);return
    parent=dest.parent;parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix=f'.rank{rank}.',dir=parent))
    core,goal,st,oracle,_,_,lo,hi,masks=runtime(job,torch.device('cpu'))
    with np.load(out/'splits.npz',allow_pickle=False) as z:
        x=z['x'];splits={sp:z[sp] for sp in ('train','val')}
    a={sp:{k:np.load(Path(job['pairs'])/sp/(k+'.npy'),mmap_mode='r')
           for k in ('x_index','sensor','q_plus','q_minus')} for sp in splits}
    records=[];coverage=[];case=0
    for split in ('train','val'):
        eligible=np.isin(a[split]['x_index'],splits[split])
        for sensor in range(8):
            pool=np.flatnonzero(eligible&(a[split]['sensor']==sensor))
            require(len(pool)>0,'No allowed pair for reference bank')
            for cohort in (0,1):
                for slot in range(cfg['reference_attempts_per_sensor_cohort'][split]):
                    ident=case;case+=1
                    if ident%cfg['world']!=rank:continue
                    rng=np.random.default_rng(np.random.SeedSequence([cfg['reference_seed'],ident]))
                    ix=int(pool[rng.integers(len(pool))]);xi=int(a[split]['x_index'][ix])
                    xx=torch.tensor(x[xi]);plus=np.asarray(a[split]['q_plus'][ix],np.float32)
                    q=np.asarray(a[split]['q_minus'][ix],np.float32)
                    require(oracle.value(xx,torch.tensor(plus),sensor)>0,'Stored witness is not visible')
                    if cohort==1:
                        choices=rng.uniform(job['lo'],job['hi'],(cfg['uniform_proposals'],7)).astype(np.float32)
                        gm=oracle.planes(xx,torch.tensor(choices),sensor).min(-1).values.detach().numpy()
                        found=np.flatnonzero(gm<0)
                        if not len(found):
                            coverage.append(dict(case_id=ident,status='NO_OUTSIDE_START',split=split,sensor=sensor,cohort=cohort));continue
                        q=choices[found[0]]
                    margin=oracle.value(xx,torch.tensor(q),sensor)
                    require(margin<0,'Reference initial configuration must be outside')
                    records.append(dict(case_id=ident,split=split,x_index=xi,x=x[xi].tolist(),sensor=sensor,
                        cohort=cohort,q_init=q.tolist(),witness_q=plus.tolist(),initial_g_m=margin))
                    coverage.append(dict(case_id=ident,status='QUALIFIED',split=split,sensor=sensor,cohort=cohort))
    write_lines(tmp/'starts.jsonl',records);write_lines(tmp/'coverage.jsonl',coverage)
    finish(tmp,planned=case,rank=rank,model_loaded=False);os.rename(tmp,dest)
    print(f'[reference inputs rank{rank}] {len(records)}; no teacher calls',flush=True)

def freeze_inputs(job):
    out=Path(job['out']);dest=out/'reference_inputs'
    if (dest/'complete.json').exists():stage(dest);return
    records=[];coverage=[]
    for rank in range(job['cfg']['world']):
        p=dest/f'rank{rank}';stage(p);records+=list(lines(p/'starts.jsonl'));coverage+=list(lines(p/'coverage.jsonl'))
    n=16*sum(job['cfg']['reference_attempts_per_sensor_cohort'].values())
    require(sorted(c['case_id'] for c in coverage)==list(range(n)),'Missing planned reference attempt')
    require(len({r['case_id'] for r in records})==len(records),'Duplicate reference start')
    with np.load(out/'splits.npz',allow_pickle=False) as z:
        for split in ('train','val'):
            require(np.isin([r['x_index'] for r in records if r['split']==split],z[split]).all(),'Train/val leakage')
    if not (dest/'starts.jsonl').exists():write_lines(dest/'starts.jsonl',sorted(records,key=lambda r:r['case_id']))
    else:require(list(lines(dest/'starts.jsonl'))==sorted(records,key=lambda r:r['case_id']),'Frozen reference differs')
    finish(dest,attempts=n,qualified=len(records),model_loaded=False)
    print('[REFERENCE FROZEN] fixed attempts; teacher success selection starts only now',flush=True)

def trace_points(result,q_init,limit):
    seq=[q_init,result['q_candidate'],result['q_zero']]
    seq += [r['q'] for key in ('projection_history','ascent_history') for r in result.get(key,[]) if 'q' in r]
    unique=[];seen=set()
    for q in seq:
        q=np.asarray(q,np.float32).reshape(7)
        require(np.isfinite(q).all(),'Nonfinite recorded configuration')
        key=q.tobytes()
        if key not in seen:unique.append(q);seen.add(key)
    return np.asarray(unique[:limit],np.float32)

def mine(job,rank):
    verify(job);out=Path(job['out']);stage(out/'reference_inputs')
    dest=out/'reference_mined'/f'rank{rank}'
    if dest.exists():stage(dest);return
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=f'.rank{rank}.',dir=dest.parent))
    device=torch.device('cuda:0');torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    core,goal,st,oracle,probe,make,lo,hi,masks=runtime(job,device)
    model,_=load_model(core,job['teacher'],TEACHER_SHA,device);model.eval().requires_grad_(False)
    before=model_digest(model);p=make(model);trajectory=[];outcomes=[]
    starts=[r for r in lines(out/'reference_inputs/starts.jsonl') if r['case_id']%job['cfg']['world']==rank]
    for i,r in enumerate(starts):
        x=torch.tensor(r['x'],device=device);q=torch.tensor(r['q_init'],device=device);s=r['sensor']
        result=probe.run_probe(p,oracle,x,q,s)
        outcomes.append(dict(start=r,result=result)) # Keep every failure; never replace it.
        if result['fov_pass']:
            qs=trace_points(result,r['q_init'],job['cfg']['max_trace_points'])
            yy_out=[];proj_out=[];asc_out=[]
            from keep import actions
            for qrow in qs:
                qq=torch.tensor(qrow,device=device).reshape(1,7)
                yy,gg=p.branch_value_and_grad(x.reshape(1,3),qq,s)
                qp,_=p._projection_step(qq,yy,gg);qa,_=p._ascent_step(qq,gg)
                dp,da=actions(qq,torch.tensor([yy],device=device),gg,masks[s],lo,hi)
                require(torch.allclose(qq+dp,qp,atol=2e-6,rtol=1e-5),'Projection formula mismatch')
                require(torch.allclose(qq+da,qa,atol=2e-6,rtol=1e-5),'Ascent formula mismatch')
                yy_out.append(yy);proj_out.append((qp-qq)[0].detach().cpu().numpy());asc_out.append((qa-qq)[0].detach().cpu().numpy())
            trajectory.append(dict(case_id=r['case_id'],split=r['split'],x=r['x'],x_index=r['x_index'],sensor=s,
                  cohort=r['cohort'],q=qs,score=np.asarray(yy_out,np.float32),project=np.asarray(proj_out),ascent=np.asarray(asc_out)))
        if (i+1)%100==0:print(f'[teacher rank{rank}] {i+1}/{len(starts)}; successful={len(trajectory)}',flush=True)
    require(model_digest(model)==before,'Teacher changed during reference mining')
    write_lines(tmp/'outcomes.jsonl',outcomes);write_lines(tmp/'successes.jsonl',trajectory)
    finish(tmp,teacher_sha256=TEACHER_SHA,attempted=len(starts),success=len(trajectory));os.rename(tmp,dest)

def merge(job):
    out=Path(job['out']);dest=out/'replay'
    if dest.exists():stage(dest);return
    rows=[];attempts=[]
    for rank in range(job['cfg']['world']):
        p=out/'reference_mined'/f'rank{rank}';m=stage(p)
        require(m['teacher_sha256']==TEACHER_SHA,'Reference teacher mismatch')
        rows+=list(lines(p/'successes.jsonl'));attempts+=list(lines(p/'outcomes.jsonl'))
    planned=list(lines(out/'reference_inputs/starts.jsonl'))
    require(sorted(r['start']['case_id'] for r in attempts)==sorted(r['case_id'] for r in planned),'Missing teacher outcomes')
    good={r['start']['case_id'] for r in attempts if r['result']['fov_pass']}
    require(good=={r['case_id'] for r in rows} and len(good)==len(rows),'Replay success selection mismatch')
    tmp=Path(tempfile.mkdtemp(prefix='.replay.',dir=out));counts={}
    for split in ('train','val'):
        data=sorted([r for r in rows if r['split']==split],key=lambda r:r['case_id'])
        count=np.bincount([r['sensor']*2+r['cohort'] for r in data],minlength=16)
        require(np.all(count>0),f'No teacher successes in stratum {split}: {count.tolist()}')
        counts[split]=count.tolist();ptr=np.r_[0,np.cumsum([len(r['q']) for r in data])]
        arrays={k:np.asarray([r[k] for r in data],np.int64 if k in ('case_id','x_index','sensor','cohort') else np.float32) for k in ('case_id','x_index','sensor','cohort','x')}
        arrays.update({k:np.concatenate([np.asarray(r[k],np.float32) for r in data]) for k in ('q','score','project','ascent')})
        arrays['ptr']=ptr.astype(np.int64);np.savez_compressed(tmp/(split+'.npz'),**arrays)
    finish(tmp,teacher_sha256=TEACHER_SHA,stratum_counts=counts,successful_trajectories=len(rows),attempts=len(planned),
           selection='fixed attempts; successful training and validation trajectories separated; trajectory-uniform sampler')
    os.rename(tmp,dest);print('[REPLAY COMPLETE]',counts,flush=True)
