#!/usr/bin/env python3
"""Fresh promotion holdout for V1 vs E1 vs R1.

This evaluator uses only validation x-indices that were never used by the
completed 1388-case development benchmark. Start generation is deterministic
and written+hashed before any model is evaluated.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib, json, math
from pathlib import Path
import sys, time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
E012_DIR=REPO/'experiments/hierarchical9_e012_end2end_v1'
ABC_DIR=REPO/'experiments/hierarchical9_abc_capacity_routing_v1'
for p in (str(E012_DIR),str(ABC_DIR),str(HERE)):
    if p not in sys.path: sys.path.insert(0,p)
import e012_protocol as eproto
base=eproto._load('r012_fresh_base',HERE/'evaluate_unified.py')
old=eproto.old
NAMES=('V1','E1','R1')
SEED=20260921
POINTS_PER_SENSOR=48

def sha256_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def fraction(k,n):
    return {'passed':int(k),'count':int(n),'rate':float(k/n) if n else None}

def exact_two_sided_sign_p(a_only:int,b_only:int)->float:
    n=a_only+b_only
    if n==0: return 1.0
    k=min(a_only,b_only)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2.0*tail)

def aggregate(rows):
    out={}
    for cohort,chosen in (
        ('local',[r for r in rows if '/local_' in r['group']]),
        ('uniform',[r for r in rows if r['group'].endswith('uniform_outside')]),
        ('all',rows),
    ):
        out[cohort]={n:fraction(sum(r['models'][n]['fov_pass'] for r in chosen),len(chosen)) for n in NAMES}
    return out

def per_sensor(rows):
    return {f'S{s}':{n:fraction(sum(r['models'][n]['fov_pass'] for r in rows if r['sensor']==s),
                                    sum(r['sensor']==s for r in rows)) for n in NAMES} for s in range(8)}

def model_failure(rows,name):
    rr=[r['models'][name] for r in rows]
    return {
        'failure_stages':dict(Counter(r['failure_stage'] for r in rr)),
        'root_source_by_failure':dict(Counter(r['root_source']+' / '+r['failure_stage'] for r in rr)),
    }

def paired(rows,a,b):
    c=Counter()
    for r in rows:
        av=bool(r['models'][a]['fov_pass']); bv=bool(r['models'][b]['fov_pass'])
        c['both_pass' if av and bv else a+'_only' if av else b+'_only' if bv else 'both_fail'] += 1
    ao=c[a+'_only']; bo=c[b+'_only']
    return {**dict(c),'net_'+a+'_minus_'+b:ao-bo,'exact_sign_p':exact_two_sided_sign_p(ao,bo)}

def markdown(report):
    def f(v): return 'N/A' if v is None else f'{v:.5f}'
    lines=['# Fresh promotion holdout: V1 vs E1 vs R1','',
        f"Fresh seed: {report['fresh_seed']}. Development-used x_index values excluded globally: {report['excluded_development_x_count']}.",
        f"Starts SHA256: {report['starts_sha256']}. No model output was used to select starts.",'',
        '## Aggregate','','| Cohort | N | V1 | E1 | R1 |','|---|---:|---:|---:|---:|']
    for cohort in ('local','uniform','all'):
        row=report['aggregate'][cohort]; n=row['V1']['count']
        lines.append(f"| {cohort} | {n} | {f(row['V1']['rate'])} | {f(row['E1']['rate'])} | {f(row['R1']['rate'])} |")
    lines += ['','## Paired outcomes','','| Pair | A-only | B-only | Net A-B | exact sign p |','|---|---:|---:|---:|---:|']
    for key,a,b in (('R1_vs_V1','R1','V1'),('R1_vs_E1','R1','E1'),('E1_vs_V1','E1','V1')):
        r=report['paired'][key]
        lines.append(f"| {a} vs {b} | {r.get(a+'_only',0)} | {r.get(b+'_only',0)} | {r['net_'+a+'_minus_'+b]} | {r['exact_sign_p']:.6f} |")
    lines += ['','## Failure stages','','| Model | ROOT_NOT_FOUND | CANDIDATE_BUT_FOV_FAIL | SUCCESS |','|---|---:|---:|---:|']
    for n in NAMES:
        c=report['failures'][n]['failure_stages']
        lines.append(f"| {n} | {c.get('ROOT_NOT_FOUND',0)} | {c.get('LEARNED_CANDIDATE_BUT_FOV_FAIL',0)} | {c.get('SUCCESS',0)} |")
    lines += ['','## Per sensor','']
    for s in range(8):
        row=report['per_sensor'][f'S{s}']; n=row['V1']['count']
        lines += [f'### S{s} (N={n})','', '| Model | FOV pass |','|---|---:|']
        for name in NAMES: lines.append(f"| {name} | {f(row[name]['rate'])} |")
        lines.append('')
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--r012-root',type=Path,required=True)
    ap.add_argument('--reference-root',type=Path,required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device=torch.device('cuda',0); torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    rroot=args.r012_root.resolve(); refroot=args.reference_root.resolve()
    dev=rroot/'evaluation_unified_dev'; out=rroot/'evaluation_fresh_holdout_v1'
    if out.exists(): raise FileExistsError(f'No overwrite: {out}')
    dm=json.loads((dev/'manifest.json').read_text())
    if dm.get('status')!='COMPLETE': raise ValueError('Development evaluation must be COMPLETE')
    dev_rows=[json.loads(line) for line in (dev/'solves.jsonl').read_text().splitlines() if line.strip()]
    if len(dev_rows)!=1388: raise ValueError(f'Expected 1388 development solves, got {len(dev_rows)}')
    used_x={int(r['x_index']) for r in dev_rows}

    cache,p0,p0_sha=eproto.load_reference_root(refroot)
    artifact=Path(p0['args']['artifact_root'])
    models={}
    v1,_=old.load_v1(artifact/old.V1_REL,device); models['V1']=v1.eval().requires_grad_(False)
    ecp,edigest=eproto.load_checkpoint(eproto.output_dir(refroot,'E1','pilot')/'final.pt')
    eproto.assert_checkpoint(ecp,p0,p0_sha,cache.identity,require_pilot=True)
    models['E1']=eproto.load_model_from_checkpoint(ecp,p0,device).requires_grad_(False)
    rcp,rrun,rdigest=base.load_r_checkpoint(rroot,'R1')
    models['R1']=base.load_r_model(rcp,'R1',device)

    ev=old.module('evaluate_pair',eproto.abc.p2.LEGACY)
    stats=old.module('evaluate_p2',eproto.abc.P2_DIR)
    api=old.module('train_signed_visibility_cdf_pairwise_replace',old.SCRIPTS)
    dataset=api.VisibilityQ0Dataset(str(artifact/old.DATA_REL),1000,0); cache.verify_dataset(dataset,artifact/old.DATA_REL)
    oracle=old.module('oracle',old.AUDIT).SensorOracle(old.URDF,device,api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES)
    core=old.module('core',old.AUDIT); probeapi=old.module('runtime_probe',old.AUDIT)
    lo,hi=dataset.q_limits(device)
    probes={n:probeapi.make_probe(ev.SensorView(m),dataset.sensor_masks(device),lo,hi) for n,m in models.items()}
    old.module('audit',old.AUDIT).preflight(dataset,oracle,probes,device)

    d=cache.arrays['val']; rng=np.random.default_rng(SEED)
    starts=[]; selection={}; excluded=Counter()
    for s in range(8):
        pool=np.unique(d['x_index'][d['s']==s])
        pool=np.asarray([int(x) for x in pool if int(x) not in used_x],dtype=np.int64)
        if len(pool)<POINTS_PER_SENSOR:
            raise RuntimeError(f'S{s}: only {len(pool)} unused boundary-cache x values; need {POINTS_PER_SENSOR}. Refusing to reuse development x.')
        pts=np.sort(rng.choice(pool,POINTS_PER_SENSOR,replace=False))
        selection[f'S{s}']=[int(v) for v in pts]
        for xi in pts:
            for kind in (0,1):
                ii=np.flatnonzero((d['s']==s)&(d['x_index']==xi)&(d['kind']==kind))
                if not len(ii):
                    excluded[f'S{s}/missing_kind{kind}']+=1; continue
                i=int(ii[0]); x=torch.tensor(d['x'][i],device=device)
                for radius in (.02,.05):
                    qout=torch.tensor(d['q'][i]-radius*d['normal'][i],device=device)
                    qin=torch.tensor(d['q'][i]+radius*d['normal'][i],device=device)
                    group=f'S{s}/local_kind{kind}_r{radius}'
                    if (not core.within(qout,lo,hi) or not core.within(qin,lo,hi) or
                        oracle.value(x,qout,s)>=-1e-5 or oracle.value(x,qin,s)<=1e-5):
                        excluded[group]+=1; continue
                    starts.append({'group':group,'x_index':int(xi),'sensor':s,'x':x.tolist(),'q_init':qout.tolist()})
            x=dataset.x_cpu[int(xi)].to(device)
            for _ in range(2):
                found=None
                for _ in range(64):
                    q=torch.tensor(rng.uniform(cache.lo,cache.hi),device=device,dtype=torch.float32)
                    if oracle.value(x,q,s)<0:
                        found=q; break
                if found is None:
                    excluded[f'S{s}/uniform_no_outside_after_64']+=1; continue
                starts.append({'group':f'S{s}/uniform_outside','x_index':int(xi),'sensor':s,'x':x.tolist(),'q_init':found.tolist()})

    if any(int(r['x_index']) in used_x for r in starts): raise AssertionError('Fresh holdout leaked a development x_index')
    local=sum('/local_' in r['group'] for r in starts); uniform=sum(r['group'].endswith('uniform_outside') for r in starts)
    if local<1000 or uniform<700 or len(starts)<1800:
        raise RuntimeError(f'Fresh holdout too small after validity filtering: total={len(starts)} local={local} uniform={uniform}')

    out.mkdir(parents=True)
    start_bytes=(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in starts)).encode()
    starts_sha=sha256_bytes(start_bytes)
    (out/'starts.jsonl').write_bytes(start_bytes)
    manifest={'status':'RUNNING','evaluation':'fresh_promotion_holdout_v1','models':list(NAMES),'fresh_seed':SEED,
        'points_per_sensor':POINTS_PER_SENSOR,'development_solve_count':len(dev_rows),'excluded_development_x_count':len(used_x),
        'development_manifest_sha256':base.sha256(dev/'manifest.json'),'starts_sha256':starts_sha,'start_count':len(starts),
        'local_count':local,'uniform_count':uniform,'selected_x_by_sensor':selection,'selection_exclusions':dict(excluded),
        'checkpoint_sha256':{'E1':edigest,'R1':rdigest,'V1':old.V1_SHA},
        'limitations':'Fresh starts within the existing seed-0 validation dataset; no new physical/workspace data. FOV-only; LOS/collision/trajectory/actual-seen NOT_RUN.'}
    base.write_json(out/'manifest.json',manifest)
    print(f"[fresh] starts={len(starts)} local={local} uniform={uniform} excluded_dev_x={len(used_x)} sha={starts_sha}",flush=True)

    started=time.perf_counter(); rows=[]; groups=defaultdict(list)
    try:
        with (out/'solves.jsonl').open('w') as fs:
            for i,spec in enumerate(starts):
                x=torch.tensor(spec['x'],device=device,dtype=torch.float32); q=torch.tensor(spec['q_init'],device=device,dtype=torch.float32); s=int(spec['sensor'])
                row={**spec,'models':{}}
                order=list(NAMES[i%len(NAMES):])+list(NAMES[:i%len(NAMES)])
                for n in order: row['models'][n]=probeapi.run_probe(probes[n],oracle,x,q,s)
                fs.write(json.dumps(core.json_safe(row),allow_nan=False)+'\n'); fs.flush()
                rows.append(row); groups[row['group']].append(row)
                if (i+1)%100==0: print(f'[fresh] matched_solves={i+1}/{len(starts)}',flush=True)
        report={'status':'COMPLETE','fresh_seed':SEED,'starts_sha256':starts_sha,'excluded_development_x_count':len(used_x),
            'aggregate':aggregate(rows),'per_sensor':per_sensor(rows),
            'failures':{n:model_failure(rows,n) for n in NAMES},
            'paired':{'R1_vs_V1':paired(rows,'R1','V1'),'R1_vs_E1':paired(rows,'R1','E1'),'E1_vs_V1':paired(rows,'E1','V1')},
            'groups':{k:base.solve_summary(v,list(NAMES),stats) for k,v in groups.items()},
            'elapsed_seconds':time.perf_counter()-started}
        base.write_json(out/'report.json',report); (out/'summary.md').write_text(markdown(report),encoding='utf-8')
        manifest.update(status='COMPLETE',elapsed_seconds=report['elapsed_seconds']); base.write_json(out/'manifest.json',manifest)
    except Exception as exc:
        manifest.update(status='FAILED',error=repr(exc)); base.write_json(out/'manifest.json',manifest); raise
    print(f'[done] fresh_promotion_holdout_complete output={out}',flush=True)

if __name__=='__main__': main()
