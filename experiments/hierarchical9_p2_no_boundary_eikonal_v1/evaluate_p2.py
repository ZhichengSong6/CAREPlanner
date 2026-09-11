#!/usr/bin/env python3
"""Read-only P2 evaluation; reuse original samples, solver, global field/planning metrics."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

import protocol as p2
old = p2.old


def finite_dist(values):
    a = np.asarray(values, dtype=np.float64)
    total = a.size
    a = a[np.isfinite(a)]
    return dict(count=int(len(a)), missing=int(total-len(a)), mean=float(a.mean()) if len(a) else None,
                p05=float(np.quantile(a,.05)) if len(a) else None,
                p50=float(np.quantile(a,.5)) if len(a) else None,
                p95=float(np.quantile(a,.95)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def fraction(k, n):
    return {'passed':int(k), 'count':int(n), 'rate':float(k/n) if n else None}


def boundary_stats(a, ids):
    v, norm, cos, slope, shift = (a[k][ids] for k in
        ('value','norm','cosine','normal_slope','linearized_zero_shift_rad'))
    finite = np.isfinite(shift)
    conditioned = finite & (norm>1e-8) & (np.abs(cos)>=.1)
    return dict(abs_value=finite_dist(np.abs(v)), normal_cosine=finite_dist(cos),
        gradient_norm=finite_dist(norm), gradient_norm_error=finite_dist(np.abs(norm-1)),
        linearized_zero_shift_rad=finite_dist(shift), abs_linearized_zero_shift_rad=finite_dist(np.abs(shift)),
        linearization_defined=fraction(finite.sum(),len(v)),
        well_conditioned_linearization=fraction(conditioned.sum(),len(v)),
        abs_linearized_shift_rad_conditioned=finite_dist(np.abs(shift[conditioned])),
        nonpositive_normal_slope=fraction((slope<=0).sum(),len(v)),
        conditioning_note='abs(slope)>1e-6, norm>1e-8, abs(cos)>=.1; not a certified distance or valid linearization radius')


def confusion(value, g):
    predicted, actual = np.asarray(value)>=0, np.asarray(g)>=0
    tp = int((predicted & actual).sum()); tn = int((~predicted & ~actual).sum())
    fp = int((predicted & ~actual).sum()); fn = int((~predicted & actual).sum())
    return dict(sign_accuracy=fraction(tp+tn,len(actual)), true_positive=tp,true_negative=tn,
        false_positive=fp,false_negative=fn, actual_positive=tp+fn,actual_negative=tn+fp,
        positive_recall=fraction(tp,tp+fn),negative_recall=fraction(tn,tn+fp))


def solve_stats(rows, names):
    result = dict(count=len(rows), models={}, paired={})
    for name in names:
        rr = [r['models'][name] for r in rows]
        result['models'][name] = dict(
            fov_pass=fraction(sum(r['fov_pass'] for r in rr),len(rr)),
            root_002=fraction(sum(r['predicted_root_within_002'] for r in rr),len(rr)),
            solver_ms=finite_dist([r['solver_ms'] for r in rr]),
            failure_stages=dict(Counter(r['failure_stage'] for r in rr)),
            root_sources=dict(Counter(r['root_source'] for r in rr)),
            root_source_by_failure=dict(Counter(r['root_source']+' / '+r['failure_stage'] for r in rr)))
    candidate = 'P2' if 'P2' in names else 'P2_smoke'
    for ref in ('V1','P0','P1'):
        if ref not in names: continue
        result['paired'][ref+'_vs_'+candidate] = dict(Counter(
            'both_pass' if r['models'][ref]['fov_pass'] and r['models'][candidate]['fov_pass'] else
            candidate+'_only' if r['models'][candidate]['fov_pass'] else
            ref+'_only' if r['models'][ref]['fov_pass'] else 'both_fail' for r in rows))
    return result


def markdown(result, names):
    def fmt(v): return 'N/A' if v is None else f'{v:.5f}'
    candidate = names[-1]
    lines = ['# P2: remove only the additional boundary Eikonal term','',
             f"mode={result['mode']}; FOV-only; LOS/GCDF/VBC/trajectory/seen NOT_RUN.",
             'Smoke is a 2-update plumbing check, not a matched 2000-update comparison.','',
             '| Boundary | N | '+ ' | '.join(n+' abs(f)' for n in names)+f' | {candidate} norm | {candidate} cos |',
             '|---|---:|'+ '---:|'*(len(names)+2)]
    for k,r in result['boundary'].items():
        m=r['models']; vals=[m[n]['abs_value']['mean'] for n in names]
        vals += [m[candidate]['gradient_norm']['mean'],m[candidate]['normal_cosine']['mean']]
        lines.append(f"| {k} | {r['count']} | "+' | '.join(fmt(v) for v in vals)+' |')
    lines += ['','| Solve group | N | '+' | '.join(n+' FOV' for n in names)+' |',
              '|---|---:|'+'---:|'*len(names)]
    for k,r in sorted(result['solves'].items()):
        lines.append(f"| {k} | {r['count']} | "+' | '.join(fmt(r['models'][n]['fov_pass']['rate']) for n in names)+' |')
    lines += ['','| Global path | N | Proj abs(g)<.03 | Asc1 g>=.03 | Asc10 g>=.03 |','|---|---:|---:|---:|---:|']
    for k,r in result['planning_sentinel'].items():
        lines.append(f"| {k} | {r['count']} | "+' | '.join(fmt(r[t]) for t in
                     ('proj_oracle_boundary_030','asc1_g_ge_0p03','asc10_g_ge_0p03'))+' |')
    lines += ['','report.json includes actual sign confusion, root_source x failure, paired outcomes, and conditional absolute linearized shift.',
              'boundary_samples.npz preserves signed values/slopes and exact identities. Linearized shifts are not true nearest-root distances.',
              'No auto-promotion. Lower residuals/norms alone do not certify better solving.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--mode',choices=('smoke','pilot'),required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('No allocated CUDA device')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device); root=args.reference_root.resolve()
    out=root/('evaluation_p2_smoke' if args.mode=='smoke' else 'evaluation_p2')
    if out.exists(): raise FileExistsError(f'No overwrite: {out}')
    cache,c0,c1,references=p2.load_references(root)
    cp,h2=p2.load_saved(root/('P2_smoke' if args.mode=='smoke' else 'P2')/'final.pt')
    p2.assert_p2(cp,c0,cache.identity,references,require_pilot=args.mode=='pilot')
    if cp['mode']!=args.mode: raise ValueError('Checkpoint mode mismatch')
    artifact=Path(c0['args']['artifact_root'])
    v1,_=old.load_v1(artifact/old.V1_REL,device)
    models={'V1':v1}
    if args.mode=='pilot':
        models.update(P0=p2.model_from(c0,device),P1=p2.model_from(c1,device))
    models['P2' if args.mode=='pilot' else 'P2_smoke']=p2.model_from(cp,device)
    names=list(models)
    ev=old.module('evaluate_pair',p2.LEGACY)
    api=old.module('train_signed_visibility_cdf_pairwise_replace',old.SCRIPTS)
    dataset=api.VisibilityQ0Dataset(str(artifact/old.DATA_REL),1000,0)
    cache.verify_dataset(dataset,artifact/old.DATA_REL)
    if cache.manifest['urdf_sha256']!=old.sha256(old.URDF): raise ValueError('URDF changed')
    oracle=old.module('oracle',old.AUDIT).SensorOracle(old.URDF,device,api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES)
    core=old.module('core',old.AUDIT); probeapi=old.module('runtime_probe',old.AUDIT)
    lo,hi=dataset.q_limits(device); masks=dataset.sensor_masks(device)
    probes={k:probeapi.make_probe(ev.SensorView(v),masks,lo,hi) for k,v in models.items()}
    checks=old.module('audit',old.AUDIT).preflight(dataset,oracle,probes,device)
    out.mkdir()
    manifest=dict(status='RUNNING',mode=args.mode,pilot_updates=cp['pilot_updates'],
        pair_sample_streams=cp['pair_sample_streams'],cache_manifest_sha256=cache.identity,
        models={'V1':old.V1_SHA,**references,'P2' if args.mode=='pilot' else 'P2_smoke':h2},
        evaluated_models=names,reference_root=str(root),preflight=checks,
        source_sha256=old.source_fingerprints(),p2_source_sha256=p2.fingerprints(),
        boundary_weights=p2.WEIGHTS,field_batches=1 if args.mode=='smoke' else 10,
        planning_batches=1 if args.mode=='smoke' else 2,solve_points_per_sensor=2 if args.mode=='smoke' else 32,
        limitations='Development held-out validation, not an independent final test. FOV only. No runtime change.')
    old.write_json(out/'manifest.json',manifest)
    started=time.perf_counter()
    try:
        result=dict(status='RUNNING',mode=args.mode,boundary={},profiles={},solves={})
        d=cache.arrays['val']
        preds={k:ev.selected(v,d['x'],d['q'],d['s'],d['normal'],device) for k,v in models.items()}
        saved={k:d[k] for k in ('x','q','s','normal','x_index','kind')}
        saved.update({name+'_'+k:v for name,a in preds.items() for k,v in a.items()})
        np.savez_compressed(out/'boundary_samples.npz',**saved)
        for group,ids in enumerate(cache.groups['val']):
            key=f"S{group//2}/{'bank_refined' if group%2==0 else 'offbank_refined'}"
            result['boundary'][key]=dict(count=len(ids),models={k:boundary_stats(a,ids) for k,a in preds.items()})
        for offset in (-.05,-.02,-.01,-.005,.005,.01,.02,.05):
            qs=d['q']+offset*d['normal']
            ids=np.flatnonzero(((qs>=cache.lo)&(qs<=cache.hi)).all(1))
            gs=np.asarray([oracle.value(torch.tensor(d['x'][i],device=device),torch.tensor(qs[i],device=device),int(d['s'][i])) for i in ids])
            for name,model in models.items():
                yp=ev.selected(model,d['x'][ids],qs[ids],d['s'][ids],d['normal'][ids],device)['value']
                for group in range(16):
                    keep=(2*d['s'][ids]+d['kind'][ids])==group
                    key=f'S{group//2}/kind{group%2}/offset={offset:+.3f}'
                    entry=result['profiles'].setdefault(key,dict(in_limits=int(keep.sum()),models={}))
                    entry['models'][name]=confusion(yp[keep],gs[keep])
        print('[eval] boundary and two-sided confusion complete',flush=True)
        # Exact same start-generation RNG/order as the original evaluator.
        starts=[]; excluded=Counter(); rng=np.random.default_rng(91283)
        for s in range(8):
            pool=np.unique(d['x_index'][d['s']==s])
            pts=rng.choice(pool,min(manifest['solve_points_per_sensor'],len(pool)),replace=False)
            for xi in pts:
                for kind in (0,1):
                    inds=np.flatnonzero((d['s']==s)&(d['x_index']==xi)&(d['kind']==kind))
                    if not len(inds): excluded[f'S{s}/missing_kind{kind}']+=1; continue
                    i=inds[0]; x=torch.tensor(d['x'][i],device=device)
                    for radius in (.02,.05):
                        qout=torch.tensor(d['q'][i]-radius*d['normal'][i],device=device)
                        qin=torch.tensor(d['q'][i]+radius*d['normal'][i],device=device)
                        group=f'S{s}/local_kind{kind}_r{radius}'
                        if (not core.within(qout,lo,hi) or not core.within(qin,lo,hi) or
                            oracle.value(x,qout,s)>=-1e-5 or oracle.value(x,qin,s)<=1e-5):
                            excluded[group]+=1; continue
                        starts.append((group,int(xi),s,x,qout))
                x=dataset.x_cpu[int(xi)].to(device)
                for _ in range(2):
                    q=torch.tensor(rng.uniform(cache.lo,cache.hi),device=device,dtype=torch.float32)
                    if oracle.value(x,q,s)>=0: excluded[f'S{s}/uniform_initially_inside']+=1; continue
                    starts.append((f'S{s}/uniform_outside',int(xi),s,x,q))
        groups=defaultdict(list)
        with (out/'solves.jsonl').open('w') as f:
            for number,(group,xi,s,x,q) in enumerate(starts):
                row=dict(group=group,x_index=xi,sensor=s,x=x.tolist(),q_init=q.tolist(),models={})
                order=names[number%len(names):]+names[:number%len(names)]
                for name in order: row['models'][name]=probeapi.run_probe(probes[name],oracle,x,q,s)
                f.write(json.dumps(core.json_safe(row),allow_nan=False)+'\n'); f.flush()
                groups[group].append(row)
                if (number+1)%50==0: print(f'[eval] matched_solves={number+1}/{len(starts)}',flush=True)
        result['solves']={k:solve_stats(rows,names) for k,rows in groups.items()}
        result['excluded']=dict(excluded)
        sign_oracle=api.PinocchioFOVOracle(str(old.URDF),api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES,50.,66.,.2,.7,.01)
        result['field_sentinel']=ev.field_sentinel(models,dataset,sign_oracle,device,manifest['field_batches'])
        result['planning_sentinel']=ev.planning_sentinel(models,dataset,sign_oracle,device,out,manifest['planning_batches'])
        result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-started)
        old.write_json(out/'report.json',result)
        (out/'summary.md').write_text(markdown(result,names),encoding='utf-8')
        manifest.update(status='COMPLETE',elapsed_seconds=result['elapsed_seconds'])
        old.write_json(out/'manifest.json',manifest)
    except Exception as exc:
        manifest.update(status='FAILED',error=repr(exc));old.write_json(out/'manifest.json',manifest);raise
    print(f'[done] p2_evaluation_complete mode={args.mode} output={out}',flush=True)


if __name__=='__main__': main()
