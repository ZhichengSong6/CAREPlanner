#!/usr/bin/env python3
"""Final-only frozen 1963-case evaluator. No retuning and no surrogate scores."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch
from goal import load
from train import read, write, sha, verify_job, R1_SHA, oracle_for

STARTS_SHA='0e4711fc21628fdbb341191ec634415781696800a2386a4ad48e592c34f3198c'
NAMES=('R1','SOLVER_GOAL')


def summarize(rows):
    if [r['case_id'] for r in rows]!=list(range(1963)):
        raise ValueError('Missing/duplicate cases')
    groups={'local':[r for r in rows if '/local_' in r['group']],
            'uniform':[r for r in rows if r['group'].endswith('uniform_outside')], 'all':rows}
    report={}
    for name,subset in groups.items():
        report[name]={'n':len(subset),**{m:sum(r['models'][m]['fov_pass'] for r in subset) for m in NAMES}}
    if [(report[g]['n'],report[g]['R1']) for g in ('local','uniform','all')]!=[(683,618),(1280,673),(1963,1291)]:
        raise ValueError('Frozen R1 baseline NOT reproduced; no improvement claim')
    report['paired']={
        'gained':sum(r['models']['SOLVER_GOAL']['fov_pass'] and not r['models']['R1']['fov_pass'] for r in rows),
        'lost':sum(r['models']['R1']['fov_pass'] and not r['models']['SOLVER_GOAL']['fov_pass'] for r in rows)}
    report['per_sensor']={str(s):{'n':sum(r['sensor']==s for r in rows),
        **{m:sum(r['models'][m]['fov_pass'] for r in rows if r['sensor']==s) for m in NAMES}} for s in range(8)}
    report['failures']={m:dict(Counter(r['models'][m]['failure_stage'] for r in rows)) for m in NAMES}
    report['scope']='Same legacy solver; repeated regression set; FOV only; no production promotion'
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--merge',action='store_true')
    a=p.parse_args(); job=read(a.job); verify_job(job)
    source=Path(job['out']); complete=read(source/'solver_goal_complete.json')
    if complete.get('status')!='COMPLETE' or not complete.get('base_stream_match'):
        raise ValueError('Training incomplete/not comparable')
    if sha(source/'final.pt')!=complete['final_sha256']:raise ValueError('Candidate changed')
    starts_path=Path(job['r012'])/'evaluation_fresh_holdout_v1/starts.jsonl'
    if sha(starts_path)!=STARTS_SHA:raise ValueError('Frozen regression inputs changed')
    starts=[json.loads(line) for line in starts_path.read_text().splitlines() if line.strip()]
    if len(starts)!=1963:raise ValueError('Expected 1963 cases')
    if a.merge:
        rows=[]
        for r in range(4):rows += [json.loads(line) for line in (a.out/f'rank{r}.jsonl').read_text().splitlines() if line.strip()]
        rows.sort(key=lambda x:x['case_id'])
        for r in rows:
            for k in ('x','q_init','sensor','group'):
                if r[k]!=starts[r['case_id']][k]:raise ValueError('Recorded start mismatch')
        report=summarize(rows)
        report.update(status='COMPLETE',starts_sha256=STARTS_SHA,candidate_sha256=complete['final_sha256'])
        write(a.out/'report.json',report)
        lines=['# R1 vs solver-goal scratch: identical legacy solver', '',
               '| Group | N | R1 | New | Delta pp |','|---|---:|---:|---:|---:|']
        for g in ('local','uniform','all'):
            d=report[g];n=d['n'];lines.append(f"| {g} | {n} | {d['R1']/n:.5f} | {d['SOLVER_GOAL']/n:.5f} | {100*(d['SOLVER_GOAL']-d['R1'])/n:+.2f} |")
        lines += ['',f"Paired {report['paired']}",'','| Sensor | N | R1 | New |','|---|---:|---:|---:|']
        for s,d in report['per_sensor'].items():lines.append(f"| S{s} | {d['n']} | {d['R1']/d['n']:.5f} | {d['SOLVER_GOAL']/d['n']:.5f} |")
        lines += ['',report['scope'],'',json.dumps(report['failures'],indent=2)]
        (a.out/'summary.md').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines));return
    if not torch.cuda.is_available():raise RuntimeError('Evaluation GPU allocation required')
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    repo=Path(job['repo'])
    sys.path.insert(0,str(repo/'src/care_visibility_cdf/scripts'))
    arch=load('sg_eval_arch',repo/'experiments/hierarchical9_scratch50k_r012_v1/model.py')
    api=load('sg_eval_probe',repo/'experiments/hierarchical9_boundary_audit_v1/runtime_probe.py')
    core=load('sg_eval_json',repo/'experiments/hierarchical9_boundary_audit_v1/core.py')
    from hierarchical_visibility_cdf_model import HierarchicalSensorView
    with np.load(job['data'],allow_pickle=False) as bank:
        masks=torch.tensor(bank['sensor_chain_masks'],device=device)
        lo=torch.tensor(bank['q_min'],device=device);hi=torch.tensor(bank['q_max'],device=device)
    oracle=oracle_for(repo,Path(job['urdf']),device);probes={}
    for name,path,digest in (('R1',Path(job['r012'])/'formal/R1/final.pt',R1_SHA),
                              ('SOLVER_GOAL',source/'final.pt',complete['final_sha256'])):
        if sha(path)!=digest:raise ValueError('Checkpoint SHA mismatch')
        cp=torch.load(path,map_location='cpu',weights_only=False)
        if name=='SOLVER_GOAL' and cp.get('metadata',{}).get('solver_goal_v1')!=job:
            raise ValueError('Candidate lineage mismatch')
        m=arch.build_model('R1');m.load_state_dict(cp['model_state'],strict=True)
        m.to(device).eval().requires_grad_(False)
        probes[name]=api.make_probe(HierarchicalSensorView(m),masks,lo,hi)
    with (a.out/f'rank{a.rank}.jsonl').open('x') as f:
        for count,i in enumerate(range(a.rank,len(starts),4),1):
            spec=starts[i];x=torch.tensor(spec['x'],device=device);q=torch.tensor(spec['q_init'],device=device)
            row={**spec,'case_id':i,'models':{}}
            order=NAMES if count%2 else NAMES[::-1]
            for n in order:row['models'][n]=api.run_probe(probes[n],oracle,x,q,int(spec['sensor']))
            f.write(json.dumps(core.json_safe(row),allow_nan=False)+'\n')
            if count%50==0:f.flush();print(f'[eval rank{a.rank}] {count}',flush=True)

if __name__=='__main__':main()
