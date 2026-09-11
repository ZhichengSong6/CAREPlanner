#!/usr/bin/env python3
"""V1/P0/P1/P2/P3 matched evaluation. No solver/threshold change, no runtime promotion."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

import p3_protocol as p3
import neighborhood as side
old, p2 = p3.old, p3.p2


def solve_summary(rows, names, stats):
    result = dict(count=len(rows), models={}, paired={})
    candidate = names[-1]
    for name in names:
        rr = [r['models'][name] for r in rows]
        result['models'][name] = dict(
            fov_pass=stats.fraction(sum(r['fov_pass'] for r in rr), len(rr)),
            root_002=stats.fraction(sum(r['predicted_root_within_002'] for r in rr), len(rr)),
            solver_ms=stats.finite_dist([r['solver_ms'] for r in rr]),
            failure_stages=dict(Counter(r['failure_stage'] for r in rr)),
            root_sources=dict(Counter(r['root_source'] for r in rr)),
            root_source_by_failure=dict(Counter(r['root_source']+' / '+r['failure_stage'] for r in rr)))
    for name in names[:-1]:
        result['paired'][name+'_vs_'+candidate] = dict(Counter(
            'both_pass' if r['models'][name]['fov_pass'] and r['models'][candidate]['fov_pass'] else
            candidate+'_only' if r['models'][candidate]['fov_pass'] else
            name+'_only' if r['models'][name]['fov_pass'] else 'both_fail' for r in rows))
    return result


def markdown(report, names):
    def fmt(v):
        return 'N/A' if v is None else f'{v:.5f}'
    candidate = names[-1]
    lines = ['# P3: actual-FOV neighborhood sign supervision', '',
        f"mode={report['mode']}; FOV-only. LOS/collision/trajectory/actual seen NOT_RUN.",
        'P3 adds labeled queries; it is NOT equal-supervision/equal-compute to P2.',
        'Smoke is a 2-update plumbing test, not a 2000-update effectiveness comparison.', '',
        '| Boundary | N | '+' | '.join(n+' abs(f)' for n in names)+f' | {candidate} cosine |',
        '|---|---:|'+'---:|'*(len(names)+1)]
    for key, row in report['boundary'].items():
        m = row['models']
        vals = [m[n]['abs_value']['mean'] for n in names]+[m[candidate]['normal_cosine']['mean']]
        lines.append(f"| {key} | {row['count']} | "+' | '.join(fmt(v) for v in vals)+' |')
    lines += ['', '| Solve group | N | '+' | '.join(n+' FOV' for n in names)+' |',
              '|---|---:|'+'---:|'*len(names)]
    for key, row in sorted(report['solves'].items()):
        lines.append(f"| {key} | {row['count']} | "+' | '.join(fmt(row['models'][n]['fov_pass']['rate']) for n in names)+' |')
    lines += ['', '| Global path | N | Proj abs(g)<.03 | Asc1 g>=.03 | Asc10 g>=.03 |',
              '|---|---:|---:|---:|---:|']
    for key, row in report['planning_sentinel'].items():
        vals = [row[t] for t in ('proj_oracle_boundary_030','asc1_g_ge_0p03','asc10_g_ge_0p03')]
        lines.append(f"| {key} | {row['count']} | "+' | '.join(fmt(v) for v in vals)+' |')
    lines += ['', 'report.json: actual TP/TN/FP/FN, root-source failures, paired outcomes, absolute/conditioned linearized shifts.',
              'neighborhood_sentinel is a fixed additional radial cohort; original benchmark starts are unchanged.',
              'Lower loss/sign error does not certify better acquisition; no automatic checkpoint promotion.']
    return '\n'.join(lines)+'\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-root', type=Path, required=True)
    ap.add_argument('--mode', choices=('smoke','pilot'), required=True)
    ap.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    args = ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No allocated CUDA device')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.reference_root.resolve()
    out = root/('evaluation_p3_smoke' if args.mode == 'smoke' else 'evaluation_p3')
    if out.exists():
        raise FileExistsError(f'No overwrite: {out}')
    cache, controls, refs = p3.load_references(root)
    cp, digest = p2.load_saved(root/('P3_smoke' if args.mode == 'smoke' else 'P3')/'final.pt')
    p3.assert_p3(cp, controls['P0'], cache.identity, refs, require_pilot=args.mode == 'pilot')
    if cp['mode'] != args.mode:
        raise ValueError('Checkpoint mode mismatch')
    artifact = Path(controls['P0']['args']['artifact_root'])
    v1, _ = old.load_v1(artifact/old.V1_REL, device)
    models = {'V1': v1}
    if args.mode == 'pilot':
        models.update({key: p2.model_from(c, device) for key, c in controls.items()})
    candidate = 'P3_smoke' if args.mode == 'smoke' else 'P3'
    models[candidate] = p2.model_from(cp, device)
    names = list(models)
    ev = old.module('evaluate_pair', p2.LEGACY)
    stats = old.module('evaluate_p2', p3.P2_DIR)
    api = old.module('train_signed_visibility_cdf_pairwise_replace', old.SCRIPTS)
    dataset = api.VisibilityQ0Dataset(str(artifact/old.DATA_REL),1000,0)
    cache.verify_dataset(dataset, artifact/old.DATA_REL)
    if cache.manifest['urdf_sha256'] != old.sha256(old.URDF):
        raise ValueError('URDF changed')
    oracle = old.module('oracle',old.AUDIT).SensorOracle(old.URDF,device,api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES)
    pair = side.PairwiseFOV(oracle)
    core, probeapi = old.module('core',old.AUDIT), old.module('runtime_probe',old.AUDIT)
    lo, hi = dataset.q_limits(device)
    probes = {k: probeapi.make_probe(ev.SensorView(v),dataset.sensor_masks(device),lo,hi) for k,v in models.items()}
    checks = old.module('audit',old.AUDIT).preflight(dataset,oracle,probes,device)
    checks['pairwise_neighbor_FOV'] = pair.verify(cache,device)
    out.mkdir()
    manifest = dict(status='RUNNING', mode=args.mode, pilot_updates=cp['pilot_updates'],
        pair_sample_streams=cp['pair_sample_streams'], cache_manifest_sha256=cache.identity,
        models={'V1':old.V1_SHA, **refs, candidate:digest}, evaluated_models=names,
        source_sha256=old.source_fingerprints(), p2_source_sha256=p2.fingerprints(),
        p3_source_sha256=p3.fingerprints(), boundary_weights=p2.WEIGHTS, neighborhood_config=p3.NEIGHBOR,
        reference_root=str(root), preflight=checks, field_batches=1 if args.mode=='smoke' else 10,
        planning_batches=1 if args.mode=='smoke' else 2, solve_points_per_sensor=2 if args.mode=='smoke' else 32,
        limitations='Development held-out validation; not an independent final test. FOV-only. No runtime changes.',
        new_supervision='8192 attempted side queries/update; old uniform+anchor streams match, not equal total supervision')
    old.write_json(out/'manifest.json',manifest)
    started = time.perf_counter()
    try:
        result = dict(status='RUNNING', mode=args.mode, boundary={},profiles={},solves={})
        d = cache.arrays['val']
        predictions = {n:ev.selected(m,d['x'],d['q'],d['s'],d['normal'],device) for n,m in models.items()}
        saved = {k:d[k] for k in ('x','q','s','normal','x_index','kind')}
        saved.update({n+'_'+k:v for n,a in predictions.items() for k,v in a.items()})
        np.savez_compressed(out/'boundary_samples.npz',**saved)
        for group, ids in enumerate(cache.groups['val']):
            key = f"S{group//2}/{'bank_refined' if group%2==0 else 'offbank_refined'}"
            result['boundary'][key] = dict(count=len(ids),models={n:stats.boundary_stats(a,ids) for n,a in predictions.items()})
        # Keep exactly the old fixed offsets and oracle.value path for comparability.
        for offset in (-.05,-.02,-.01,-.005,.005,.01,.02,.05):
            qs = d['q']+offset*d['normal']
            ids = np.flatnonzero(((qs>=cache.lo)&(qs<=cache.hi)).all(1))
            gs = np.asarray([oracle.value(torch.tensor(d['x'][i],device=device),torch.tensor(qs[i],device=device),int(d['s'][i])) for i in ids])
            for name, model in models.items():
                yp = ev.selected(model,d['x'][ids],qs[ids],d['s'][ids],d['normal'][ids],device)['value']
                for group in range(16):
                    keep = (2*d['s'][ids]+d['kind'][ids]) == group
                    key = f'S{group//2}/kind{group%2}/offset={offset:+.3f}'
                    row = result['profiles'].setdefault(key,dict(in_limits=int(keep.sum()),models={}))
                    row['models'][name] = stats.confusion(yp[keep],gs[keep])
        print('[eval] boundary and two-sided confusion complete',flush=True)
        # Additional fixed continuous-radius cohort. NOT substituted for old benchmark.
        radial = side.radii_for_update(controls['P0']['args']['seed'],0,len(d['s']))
        accum = {n:torch.zeros((32,len(side.COLUMNS)),device=device,dtype=torch.float64) for n in names}
        with torch.no_grad():
            for start in range(0,len(d['s']),128):
                ids = np.arange(start,min(start+128,len(d['s'])))
                nb = side.build_queries(cache.tensors('val',ids,device),radial[ids],pair,lo,hi)
                for name,m in models.items():
                    _, st = side.sign_loss(m,nb,torch.ones(32,device=device))
                    if not torch.isfinite(st).all():
                        raise RuntimeError('Nonfinite radial evaluation')
                    accum[name] += st
        result['neighborhood_sentinel'] = {n:side.summary(st) for n,st in accum.items()}
        starts, excluded = [], Counter()
        rng = np.random.default_rng(91283)
        for s in range(8):
            pool = np.unique(d['x_index'][d['s']==s])
            pts = rng.choice(pool,min(manifest['solve_points_per_sensor'],len(pool)),replace=False)
            for xi in pts:
                for kind in (0,1):
                    inds = np.flatnonzero((d['s']==s)&(d['x_index']==xi)&(d['kind']==kind))
                    if not len(inds):
                        excluded[f'S{s}/missing_kind{kind}']+=1
                        continue
                    i=inds[0]; x=torch.tensor(d['x'][i],device=device)
                    for radius in (.02,.05):
                        qout=torch.tensor(d['q'][i]-radius*d['normal'][i],device=device)
                        qin=torch.tensor(d['q'][i]+radius*d['normal'][i],device=device)
                        group=f'S{s}/local_kind{kind}_r{radius}'
                        if (not core.within(qout,lo,hi) or not core.within(qin,lo,hi) or
                            oracle.value(x,qout,s)>=-1e-5 or oracle.value(x,qin,s)<=1e-5):
                            excluded[group]+=1
                            continue
                        starts.append((group,int(xi),s,x,qout))
                x=dataset.x_cpu[int(xi)].to(device)
                for _ in range(2):
                    q=torch.tensor(rng.uniform(cache.lo,cache.hi),device=device,dtype=torch.float32)
                    if oracle.value(x,q,s)>=0:
                        excluded[f'S{s}/uniform_initially_inside']+=1
                        continue
                    starts.append((f'S{s}/uniform_outside',int(xi),s,x,q))
        groups=defaultdict(list)
        with (out/'solves.jsonl').open('w') as f:
            for number,(group,xi,s,x,q) in enumerate(starts):
                row=dict(group=group,x_index=xi,sensor=s,x=x.tolist(),q_init=q.tolist(),models={})
                order=names[number%len(names):]+names[:number%len(names)]
                for name in order:
                    row['models'][name]=probeapi.run_probe(probes[name],oracle,x,q,s)
                f.write(json.dumps(core.json_safe(row),allow_nan=False)+'\n'); f.flush()
                groups[group].append(row)
                if (number+1)%50==0:
                    print(f'[eval] matched_solves={number+1}/{len(starts)}',flush=True)
        result['solves']={k:solve_summary(rows,names,stats) for k,rows in groups.items()}
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
        manifest.update(status='FAILED',error=repr(exc));old.write_json(out/'manifest.json',manifest)
        raise
    print(f'[done] p3_evaluation_complete mode={args.mode} output={out}',flush=True)


if __name__=='__main__':
    main()
