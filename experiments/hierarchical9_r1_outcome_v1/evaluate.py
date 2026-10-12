#!/usr/bin/env python3
"""Read-only final-checkpoint regression. Original solver; no threshold search."""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import torch
import numpy as np
from support import load, read, write, sha, R1_SHA, STREAM, FORMAT, BASE

STARTS='0e4711fc21628fdbb341191ec634415781696800a2386a4ad48e592c34f3198c'
NAMES=('R1','OUTCOME')

def cohort(r):
    if '/local_' in r['group']:return 'local'
    if r['group'].endswith('uniform_outside'):return 'uniform'
    raise ValueError('Unknown frozen cohort')

def merge(out, starts):
    rows=[]
    for rank in range(4):
        complete=read(out/f'rank{rank}.done.json')
        if sha(out/f'solves.rank{rank}.jsonl')!=complete['sha256']:raise ValueError('Rank result hash changed')
        rows.extend(json.loads(s) for s in (out/f'solves.rank{rank}.jsonl').read_text().splitlines())
    rows.sort(key=lambda r:r['case_id'])
    if [r['case_id'] for r in rows]!=list(range(1963)):raise ValueError('Incomplete/duplicate cases')
    for r,s in zip(rows,starts):
        if any(r[k]!=s[k] for k in ('x','q_init','sensor','group')):raise ValueError('Starts changed')
    result={}
    for group in ('local','uniform','all'):
        subset=[r for r in rows if group=='all' or cohort(r)==group]
        a,b=[sum(r['models'][n]['fov_pass'] for r in subset) for n in NAMES]
        result[group]={'n':len(subset),'R1':a,'OUTCOME':b,'delta_pp':100*(b-a)/len(subset)}
    reproduced=[result[g]['R1'] for g in ('local','uniform','all')]==[618,673,1291]
    gained=sum(not r['models']['R1']['fov_pass'] and r['models']['OUTCOME']['fov_pass'] for r in rows)
    lost=sum(r['models']['R1']['fov_pass'] and not r['models']['OUTCOME']['fov_pass'] for r in rows)
    per={}
    for s in range(8):
        rr=[r for r in rows if r['sensor']==s]
        per[f'S{s}']={'n':len(rr),**{n:sum(r['models'][n]['fov_pass'] for r in rr) for n in NAMES}}
    report={'status':'COMPLETE' if reproduced else 'INVALID_BASELINE_NOT_REPRODUCED',
            'cohorts':result,'paired':{'gained':gained,'lost':lost,'net':gained-lost},'per_sensor':per,
            'failure_stages':{n:dict(Counter(r['models'][n]['failure_stage'] for r in rows)) for n in NAMES},
            'scope':'FOV-only reused regression; no execution/LOS/trajectory certificate; no promotion'}
    write(out/'report.json',report)
    lines=['# R1 vs outcome-trained R1 (unchanged runtime, final 50k)','',report['status'],'',
           '| cohort | N | R1 | OUTCOME | delta pp |','|---|---:|---:|---:|---:|']
    for g,r in result.items():lines.append(f"| {g} | {r['n']} | {r['R1']/r['n']:.5f} | {r['OUTCOME']/r['n']:.5f} | {r['delta_pp']:+.2f} |")
    lines+=['',f'Gained={gained}; lost={lost}; net={gained-lost:+d}.','',json.dumps(per,indent=2),'',report['scope']]
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print((out/'summary.md').read_text())
    if not reproduced:raise ValueError('Original R1 did not reproduce; new result cannot establish superiority')
    write(out/'complete.json',{'status':'COMPLETE','cases':1963})

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--train',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--merge',action='store_true');a=p.parse_args()
    spec=read(a.spec);paths=spec['paths'];r=read(a.train/'run.json')
    starts_path=Path(paths['r012'])/'evaluation_fresh_holdout_v1/starts.jsonl'
    if sha(starts_path)!=STARTS:raise ValueError('Regression starts SHA mismatch')
    starts=[json.loads(s) for s in starts_path.read_text().splitlines()]
    if len(starts)!=1963:raise ValueError('Unexpected starts count')
    if a.merge:merge(a.out,starts);return
    if not (r.get('baseline_stream_exact_match') and r.get('experiment')==FORMAT and r.get('training_stream_sha256')==STREAM):
        raise ValueError('Outcome run not verified')
    files={'R1':Path(paths['r012'])/'formal/R1/final.pt','OUTCOME':a.train/'runtime.pt'}
    expected={'R1':R1_SHA,'OUTCOME':r['runtime_sha256']}
    for n,path in files.items():
        if sha(path)!=expected[n]:raise ValueError('Checkpoint identity mismatch: '+n)
    sys.path.insert(0,str(a.repo/'src/care_visibility_cdf/scripts'))
    from hierarchical_visibility_cdf_model import HierarchicalSensorView
    probe=load('outcome_frozen_probe',a.repo/'experiments/hierarchical9_boundary_audit_v1/runtime_probe.py')
    oraclemod=load('outcome_frozen_oracle',a.repo/'experiments/hierarchical9_boundary_audit_v1/oracle.py')
    api=load('outcome_geometry_names',a.repo/'src/care_visibility_cdf/scripts/train_signed_visibility_cdf_pairwise_replace.py')
    arch=load('outcome_arch',a.repo/BASE/'model.py')
    torch.set_num_threads(2);device=torch.device('cuda',0);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(paths['data'],allow_pickle=True) as z:
        lo=torch.tensor(z['q_min'],device=device);hi=torch.tensor(z['q_max'],device=device)
        masks=torch.tensor(z['sensor_chain_masks'],device=device)
    oracle=oraclemod.SensorOracle(Path(paths['urdf']),device,api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES)
    probes={}
    for name,path in files.items():
        cp=torch.load(path,map_location='cpu',weights_only=False)
        m=arch.build_model('R1');m.load_state_dict(cp['model_state'],strict=True)
        probes[name]=probe.make_probe(HierarchicalSensorView(m.to(device).eval().requires_grad_(False)),masks,lo,hi)
    from rollout import SETTINGS
    from dataclasses import asdict
    for pr in probes.values():
        if {k:getattr(pr,k) for k in asdict(SETTINGS)}!=asdict(SETTINGS):raise ValueError('Runtime settings changed')
    file=a.out/f'solves.rank{a.rank}.jsonl'
    def safe(x):
        import math
        if isinstance(x,float):return x if math.isfinite(x) else None
        if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
        if isinstance(x,list):return [safe(v) for v in x]
        return x
    with file.open('x') as f:
        for count,i in enumerate(range(a.rank,1963,4),1):
            s=starts[i];x=torch.tensor(s['x'],device=device,dtype=torch.float32)
            q=torch.tensor(s['q_init'],device=device,dtype=torch.float32)
            row=dict(s,case_id=i,models={})
            for name in (NAMES if i%2==0 else NAMES[::-1]):
                row['models'][name]=probe.run_probe(probes[name],oracle,x,q,int(s['sensor']))
            f.write(json.dumps(safe(row),allow_nan=False)+'\n')
            if count%100==0:f.flush();print(f'[eval rank{a.rank}] {count} cases',flush=True)
    for n,path in files.items():
        if sha(path)!=expected[n]:raise ValueError('Input checkpoint changed during evaluation')
    write(a.out/f'rank{a.rank}.done.json',{'sha256':sha(file)})
if __name__=='__main__':main()
