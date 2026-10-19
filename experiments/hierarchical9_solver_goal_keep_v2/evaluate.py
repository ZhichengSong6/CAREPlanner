"""New predeclared non-grid confirmation; four fixed networks, unchanged solver."""
from __future__ import annotations
from pathlib import Path
import hashlib,os,tempfile
import numpy as np
import torch
from k2_common import (ARMS,R1_SHA,TEACHER_SHA,require,read,write,write_lines,lines,
  sha,stage,finish,verify,load,aliases,runtime,load_model,model_digest)
NAMES=('R1','PARENT','A_continue','B_keep')

def generation_modules(job):
    root=Path(job['deps'])/'experiments/hierarchical9_solver_goal_offgrid_v1'
    common=load('k2_frozen_offgrid_common',root/'common.py')
    with aliases(common=common): geo=load('k2_frozen_geometry',root/'geometry.py')
    return geo

def generate(job,rank):
    verify(job);out=Path(job['out']);dest=out/'confirmation/geometry'/f'rank{rank}'
    if dest.exists():stage(dest);return
    require(sha(out/'confirmation_points.json')==job['confirmation_points_sha256'],'Confirmation points changed')
    geo=generation_modules(job);cfg=job['confirmation_cfg']
    points=np.asarray(read(out/'confirmation_points.json')['points'],np.float32)
    gj=dict(repo=job['deps'],urdf=job['urdf'],cfg=cfg,lo=job['lo'],hi=job['hi'],masks=job['masks'])
    exact=geo.ExactGeometry(gj);bank=geo.SeedBank(job['cache'],cfg)
    lo,hi,masks=(np.asarray(job[k]) for k in ('lo','hi','masks'))
    records=[];coverage=[]
    for pid in range(rank,len(points),job['cfg']['world']):
        x=points[pid]
        for s in range(8):
            seeds,_=bank.seeds(x,s)
            exact.check_parity(x,(lo+hi)/2,s)
            cov,starts=geo.generate_pair(x,pid,s,seeds,exact,lo,hi,masks[s],cfg)
            coverage.append(cov);records+=starts
        if (pid//4+1)%8==0:print(f'[new geometry rank{rank}] points={pid//4+1}/128',flush=True)
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=f'.rank{rank}.',dir=dest.parent))
    write_lines(tmp/'starts.jsonl',records);write_lines(tmp/'coverage.jsonl',coverage)
    finish(tmp,models_used=False);os.rename(tmp,dest)

def freeze(job):
    out=Path(job['out'])/'confirmation';dest=out/'inputs'
    if dest.exists():stage(dest);return
    coverage=[];rows=[]
    for rank in range(4):
        p=out/'geometry'/f'rank{rank}';stage(p)
        coverage+=list(lines(p/'coverage.jsonl'));rows+=list(lines(p/'starts.jsonl'))
    points=read(Path(job['out'])/'confirmation_points.json')['points']
    keys=[(r['point_id'],r['sensor']) for r in coverage]
    require(len(keys)==len(set(keys))==len(points)*8,'Missing coverage entries')
    require(set(keys)=={(i,s) for i in range(len(points)) for s in range(8)},'Coverage IDs invalid')
    for r in rows:
        require(r['x']==points[r['point_id']] and r['initial_g_m']<0 and r['witness_g_m']>0,'New input geometry failed')
    rows.sort(key=lambda r:(r['point_id'],r['sensor'],r['cohort'],r.get('sample_slot',r.get('radius_rad',-1))))
    rows=[dict(r,case_id=i) for i,r in enumerate(rows)];require(len(rows)>0,'No new qualified starts')
    tmp=Path(tempfile.mkdtemp(prefix='.inputs.',dir=out))
    write_lines(tmp/'starts.jsonl',rows);write_lines(tmp/'coverage.jsonl',coverage)
    finish(tmp,count=len(rows),points=len(points),selection='predeclared geometry only; no candidate selection')
    os.rename(tmp,dest);print(f'[FROZEN] new confirmation inputs={len(rows)}',flush=True)

def evaluate(job,rank):
    verify(job);root=Path(job['out']);inputs=root/'confirmation/inputs';stage(inputs)
    require(stage(root/'A_continue')['base_stream']==stage(root/'B_keep')['base_stream'],'A/B base sampling differs')
    dest=root/'confirmation/evaluation'/f'rank{rank}'
    if dest.exists():stage(dest);return
    device=torch.device('cuda:0');torch.cuda.set_device(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    core,goal,st,oracle,probe,make,lo,hi,masks=runtime(job,device)
    paths={'R1':(job['r1'],R1_SHA),'PARENT':(job['teacher'],TEACHER_SHA)}
    for arm in ARMS:
        run=read(root/arm/'run.json');paths[arm]=(str(root/arm/'runtime.pt'),run['runtime_sha256'])
    models={};probes={};before={}
    for name,(path,digest) in paths.items():
        m,cp=load_model(core,path,digest,device);m.eval().requires_grad_(False)
        if name in ARMS:
            require(cp.get('variant')==name and cp.get('parent_sha256')==TEACHER_SHA and cp.get('stage_step')==20000,'Wrong continuation checkpoint')
        models[name]=m;probes[name]=make(m);before[name]=model_digest(m)
    starts=list(lines(inputs/'starts.jsonl'))[rank::4]
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix=f'.rank{rank}.',dir=dest.parent))
    if starts:
        r=starts[0];x=torch.tensor(r['x'],device=device);q=torch.tensor(r['q_init'],device=device)
        for _ in range(2):
            for name in NAMES:probe.run_probe(probes[name],oracle,x,q,r['sensor'])
    with (tmp/'solves.jsonl').open('x') as f:
        import json
        from k2_common import safe
        for i,r in enumerate(starts):
            x=torch.tensor(r['x'],device=device);q=torch.tensor(r['q_init'],device=device);s=r['sensor']
            require(oracle.value(x,torch.tensor(r['witness_q'],device=device),s)>0,'Runtime witness failed')
            row=dict(case_id=r['case_id'],start=r,models={})
            order=NAMES[(i+rank)%4:]+NAMES[:(i+rank)%4]
            for name in order:row['models'][name]=probe.run_probe(probes[name],oracle,x,q,s)
            f.write(json.dumps(safe(row),allow_nan=False)+'\n')
            if (i+1)%100==0:f.flush();print(f'[confirm rank{rank}] {i+1}/{len(starts)}',flush=True)
    require(all(model_digest(m)==before[n] for n,m in models.items()),'Evaluation modified model')
    finish(tmp,checkpoints={n:v[1] for n,v in paths.items()},count=len(starts),starts_sha256=sha(inputs/'starts.jsonl'))
    os.rename(tmp,dest)

def comparison(rows,left,right,npoints,seed):
    n=len(rows);gained=sum(r['models'][left]['fov_pass'] and not r['models'][right]['fov_pass'] for r in rows)
    lost=sum(r['models'][right]['fov_pass'] and not r['models'][left]['fov_pass'] for r in rows)
    counts=np.zeros(npoints);delta=np.zeros(npoints)
    for r in rows:
        pid=r['start']['point_id'];counts[pid]+=1
        delta[pid]+=int(r['models'][left]['fov_pass'])-int(r['models'][right]['fov_pass'])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(4000):
        ix=rng.integers(npoints,size=npoints);den=counts[ix].sum()
        if den:values.append(100*delta[ix].sum()/den)
    return dict(n=n,gained=gained,lost=lost,delta_pp=100*(gained-lost)/n if n else None,
       cluster_x_95_pp=np.quantile(values,[.025,.975]).tolist() if values else None)

def report(job):
    root=Path(job['out']);out=root/'confirmation';dest=out/'report'
    if dest.exists():stage(dest);return
    frozen=list(lines(out/'inputs/starts.jsonl'));rows=[]
    for rank in range(4):
        p=out/'evaluation'/f'rank{rank}';stage(p);rows+=list(lines(p/'solves.jsonl'))
    rows.sort(key=lambda r:r['case_id'])
    require([r['case_id'] for r in rows]==list(range(len(frozen))),'Incomplete evaluation rows')
    for r,start in zip(rows,frozen):
        require(r['start']==start,'Frozen start mismatch')
        for n in NAMES:
            m=r['models'][n];require(isinstance(m['fov_pass'],bool) and np.isfinite(m['candidate_g_m']),'Invalid result')
            require(m['fov_pass']==(m['candidate_in_limits'] and m['candidate_g_m']>=0),'FOV result mismatch')
    subsets={'local':[r for r in rows if r['start']['cohort']=='local'],
             'uniform':[r for r in rows if r['start']['cohort']=='uniform'],'all':rows}
    subsets.update({f'S{s}':[r for r in rows if r['start']['sensor']==s] for s in range(8)})
    result={};text=['# Solver-Goal v2：新空间点固定权重确认','',
        '四份固定网络、相同完整 solver；主要比较 B vs PARENT 与 B vs A。',
        '', '| Group | N | R1 | Parent | A | B | B−Parent pp [95%] | B−A pp [95%] |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for label,part in subsets.items():
        rates={n:sum(r['models'][n]['fov_pass'] for r in part)/len(part) if part else None for n in NAMES}
        bs=comparison(part,'B_keep','PARENT',job['cfg']['confirmation_points'],261020)
        ba=comparison(part,'B_keep','A_continue',job['cfg']['confirmation_points'],261020)
        result[label]=dict(rates=rates,B_vs_parent=bs,B_vs_A=ba)
        fmt=lambda v:'NA' if v is None else f'{100*v:.2f}%'
        ci=lambda v:'NA' if v['delta_pp'] is None else f"{v['delta_pp']:+.2f} [{v['cluster_x_95_pp'][0]:+.2f},{v['cluster_x_95_pp'][1]:+.2f}]"
        text.append(f"| {label} | {len(part)} | "+' | '.join(fmt(rates[n]) for n in NAMES)+f' | {ci(bs)} | {ci(ba)} |')
    from collections import Counter
    coverage=Counter(r['status'] for r in lines(out/'inputs/coverage.jsonl'))
    text+=['',f'覆盖：{dict(coverage)}。未确认组合不删除、不声称不可行。',
       '区间按 x 重采样，只反映两份固定权重的差，不证明跨训练种子稳定。',
       'FOV-only；无阈值扫描、无自动模型晋升、无 LOS/碰撞/轨迹认证。']
    support={n:{} for n in NAMES}
    for name in NAMES:
        for key in ('solver_ms','gradient_calls','value_calls'):
            v=[r['models'][name][key] for r in rows]
            support[name][key]={'mean':float(np.mean(v)),'p50':float(np.median(v)),'p95':float(np.quantile(v,.95))}
        support[name]['margin_rates']={str(g):sum(r['models'][name]['candidate_in_limits'] and r['models'][name]['candidate_g_m']>=g for r in rows)/len(rows) for g in (0,.005,.01,.02,.03)}
    tmp=Path(tempfile.mkdtemp(prefix='.report.',dir=out));write(tmp/'report.json',dict(groups=result,coverage=dict(coverage),support=support))
    (tmp/'summary.md').write_text('\n'.join(text)+'\n');finish(tmp,primary='B vs parent and B vs A; no automatic promotion');os.rename(tmp,dest)
    print('\n'.join(text),flush=True)
