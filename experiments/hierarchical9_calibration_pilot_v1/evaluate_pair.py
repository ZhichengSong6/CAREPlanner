#!/usr/bin/env python3
"""Read-only V1/P0/P1 evaluation: fixed held-out anchors, field sentinel, matched branch solves."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import (REPO, SCRATCH, AUDIT, SCRIPTS, URDF, V1_REL, V1_SHA, DATA_REL, PILOT_FORMAT,
                    BoundaryCache, load_v1, module, setup_paths, write_json, sha256, source_fingerprints)


def load_pilot(path, device, cache, expected_arm):
    run = json.loads((path.parent / 'run.json').read_text())
    if run.get('status') != 'COMPLETE' or run.get('final_sha256') != sha256(path):
        raise ValueError('Incomplete or changed pilot checkpoint')
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if (cp.get('format') != PILOT_FORMAT or cp.get('parent_sha256') != V1_SHA
            or cp.get('arm') != expected_arm or not cp.get('completed')
            or cp.get('cache_manifest_sha256') != cache.identity
            or cp.get('pilot_updates') != cp.get('args', {}).get('steps')):
        raise ValueError('Pilot lineage/arm/cache/update mismatch')
    model = module('model', SCRATCH).HierarchicalVisibilityCDF()
    model.load_state_dict(cp['model_state'], strict=True)
    return model.to(device).eval().requires_grad_(False), cp


def assert_matched(a, b):
    if a['sample_stream_sha256_by_rank'] != b['sample_stream_sha256_by_rank']:
        raise ValueError('P0/P1 did not see identical global and boundary sample streams')
    for key in ('parent_sha256', 'cache_manifest_sha256', 'pilot_updates', 'source_sha256', 'boundary_weights'):
        if a[key] != b[key]:
            raise ValueError('P0/P1 mismatch: ' + key)
    for key in a['args']:
        if key not in ('arm', 'output') and a['args'][key] != b['args'].get(key):
            raise ValueError('P0/P1 configuration mismatch: ' + key)


class SensorView(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x):
        # Preserve the existing audit's all-eight sensor path for matched latency.
        return self.model(x)[:, 1:9]


def selected(model, x, q, sensor, normals, device):
    outputs = {k: [] for k in ('value','norm','cosine','normal_slope','linearized_zero_shift_rad')}
    for start in range(0, len(q), 256):
        end = min(start+256, len(q))
        xx = torch.as_tensor(x[start:end], device=device, dtype=torch.float32)
        qq = torch.as_tensor(q[start:end], device=device, dtype=torch.float32).clone().requires_grad_(True)
        ss = torch.as_tensor(sensor[start:end], device=device, dtype=torch.long)
        nnorm = torch.as_tensor(normals[start:end], device=device, dtype=torch.float32)
        with torch.enable_grad(), torch.autocast(device.type, enabled=False):
            pred = model(torch.cat((xx, qq), 1))
            value = pred.gather(1, (ss+1)[:,None])[:,0]
            grad = torch.autograd.grad(value.sum(), qq)[0]
        if not torch.isfinite(value).all() or not torch.isfinite(grad).all():
            raise RuntimeError('Nonfinite model query in final evaluation')
        norm = grad.norm(dim=1)
        cosine = F.cosine_similarity(grad, nnorm, dim=1, eps=1e-8)
        slope = (grad * nnorm).sum(1)
        shift = torch.where(slope.abs() > 1e-6, -value / slope, torch.nan)
        for key, values in {'value': value, 'norm': norm, 'cosine': cosine,
                            'normal_slope': slope, 'linearized_zero_shift_rad': shift}.items():
            outputs[key].extend(values.detach().cpu().tolist())
    return {k: np.asarray(v) for k,v in outputs.items()}


def field_sentinel(models, dataset, oracle, device, batches=10):
    ev = module('evaluate', SCRATCH)
    api = module('train_signed_visibility_cdf_pairwise_replace', SCRIPTS)
    per = module('train_per_sensor_visibility_cdf', SCRIPTS)
    obj = module('objective', SCRATCH)
    stats = {name: {k: ev.FieldStats() for k in ('union', 'sensor_max', *(f's{s}' for s in range(8)))} for name in models}
    ranks = {name: ev.RankingStats() for name in models}
    rng = np.random.default_rng(123)
    lo, hi = dataset.q_limits(device)
    masks = dataset.sensor_masks(device)
    for _ in range(batches):
        ids = rng.choice(dataset.val_indices_np, 64, replace=True)
        x = dataset.x_cpu[ids].to(device)
        q = torch.as_tensor(rng.uniform(lo.cpu(), hi.cpu(), (100, 7)), device=device, dtype=torch.float32)
        with torch.no_grad():
            ds, dg, has = api.decode_per_sensor_distance_and_grad(dataset.qlib_cpu[ids].to(device),
                dataset.valid_cpu[ids].to(device), q, masks, x_chunk=16)
            _, signs = oracle.signed_fov_margins(x, q)
            target, grad, valid = per.per_sensor_signed_targets(ds, dg, signs, has)
            target, grad, valid = target.reshape(-1,8), grad.reshape(-1,8,7), valid.reshape(-1,8)
            yt, gt, mt, _ = obj.supervised_targets(target, grad, valid)
            inputs = api.make_input_pairs(x, q)
        for name, model in models.items():
            yp, gp = ev.output_gradients(model, inputs)
            row = mt[:,0]
            stats[name]['union'].add(yp[row,0], gp[row,0], yt[row,0], gt[row,0])
            _, vmax, gmax = ev.masked_union(yp[:,1:], gp[:,1:], valid)
            stats[name]['sensor_max'].add(vmax, gmax, yt[row,0], gt[row,0])
            for s in range(8):
                row = valid[:,s]
                stats[name][f's{s}'].add(yp[row,s+1], gp[row,s+1], target[row,s], grad[row,s])
            ranks[name].add(yp[:,1:], target, valid)
    return {name: {'fields': {k:v.result() for k,v in st.items()}, 'ranking': ranks[name].result()} for name,st in stats.items()}


def planning_sentinel(models, dataset, oracle, device, output, batches=2):
    """Reuse the original scalar-vs-max benchmark; no sign-crossing refinement here."""
    baseline = module('compare_scalar_vs_per_sensor_apples_to_apples', SCRIPTS)
    ev = module('evaluate', SCRATCH)
    lo, hi = dataset.q_limits(device)
    rng = np.random.default_rng(124)
    sums = {f'{name}/{kind}': defaultdict(float) for name in models for kind in ('union','sensor_max')}
    samples = {}
    for b in range(batches):
        ids = rng.choice(dataset.val_indices_np, 8, replace=True)
        x = dataset.x_cpu[ids].to(device)
        q = torch.tensor(rng.uniform(lo.cpu(), hi.cpu(), (64,7)), device=device, dtype=torch.float32)
        available = dataset.valid_cpu[ids].any(dim=1).to(device)
        q0 = q[None].expand(len(x),-1,-1).contiguous()
        g0 = baseline._oracle_pair_g(oracle,x,q0)
        samples[f'{b}_x_indices'] = ids
        samples[f'{b}_q_init'] = q.cpu().numpy()
        for name, model in models.items():
            for kind, mode, view in (('union','scalar',ev.HeadView(model,'union')),
                                     ('sensor_max','eight',ev.HeadView(model,'sensors'))):
                mask = available if mode == 'eight' else None
                qp = baseline._projection(view,mode,x,q0,lo,hi,10,.5,.25,sensor_available=mask)
                fp = baseline._pair_value(view,x,qp,mode,sensor_available=mask)
                gp = baseline._oracle_pair_g(oracle,x,qp)
                snaps = baseline._ascent_snapshots(view,mode,x,qp,lo,hi,.05,.25,(1,3,5,10),sensor_available=mask)
                ga = {k:baseline._oracle_pair_g(oracle,x,v) for k,v in snaps.items()}
                if not all(torch.isfinite(v).all() for v in (qp,fp,g0,gp,*ga.values())):
                    raise RuntimeError('Nonfinite planning sentinel result')
                baseline._accumulate_planning(sums[f'{name}/{kind}'],fp,g0,gp,ga)
                samples[f'{b}_{name}_{kind}_q_projection'] = qp.cpu().numpy()
        print(f'[eval] union/max planning sentinel {b+1}/{batches}',flush=True)
    np.savez_compressed(output/'planning_samples.npz',**samples)
    return {k:baseline._planning_summary(v) for k,v in sums.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-root', required=True, type=Path)
    p.add_argument('--run-root', required=True, type=Path)
    p.add_argument('--device', default='cuda', choices=('cuda','cpu'))
    p.add_argument('--field-batches', default=10, type=int)
    p.add_argument('--planning-batches', default=2, type=int)
    p.add_argument('--solve-points', default=32, type=int, help='Per sensor; same cap for all models')
    args = p.parse_args()
    if min(args.field_batches, args.solve_points, args.planning_batches) < 1: p.error('Counts must be positive')
    setup_paths()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    runroot = args.run_root.resolve()
    out = runroot / 'evaluation'
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    cache = BoundaryCache(runroot / 'cache')
    v1, _ = load_v1(args.artifact_root.resolve() / V1_REL, device)
    p0, c0 = load_pilot(runroot/'P0/final.pt', device, cache, 'P0')
    p1, c1 = load_pilot(runroot/'P1/final.pt', device, cache, 'P1')
    assert_matched(c0, c1)
    models = {'V1':v1, 'P0':p0, 'P1':p1}
    api = module('train_signed_visibility_cdf_pairwise_replace', SCRIPTS)
    dataset = api.VisibilityQ0Dataset(str(args.artifact_root.resolve()/DATA_REL), 1000, 0)
    cache.verify_dataset(dataset, args.artifact_root.resolve()/DATA_REL)
    if cache.manifest['urdf_sha256'] != sha256(URDF): raise ValueError('URDF changed')
    oracle = module('oracle', AUDIT).SensorOracle(URDF, device, api.DEFAULT_JOINT_NAMES, api.DEFAULT_SENSOR_FRAMES)
    probeapi = module('runtime_probe', AUDIT)
    core = module('core', AUDIT)
    lo, hi = dataset.q_limits(device)
    masks = dataset.sensor_masks(device)
    probes = {k:probeapi.make_probe(SensorView(v), masks, lo, hi) for k,v in models.items()}
    checks = module('audit', AUDIT).preflight(dataset, oracle, probes, device)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'status':'RUNNING','pair_sample_streams':'MATCH','pilot_updates':c0['pilot_updates'],
                'cache_manifest_sha256':cache.identity, 'preflight':checks,
                'models':{'V1':V1_SHA,'P0':sha256(runroot/'P0/final.pt'),'P1':sha256(runroot/'P1/final.pt')},
                'source_sha256':source_fingerprints(), 'field_batches':args.field_batches, 'planning_batches':args.planning_batches, 'solve_points_per_sensor':args.solve_points,
                'limitations':'Development validation, not independent final test. FOV-only. '
                    'Self-occlusion/GCDF/VBC/trajectory/actual_seen NOT_RUN. No runtime switch.',
                'metric_note':'Old discrete-label field errors and actual-boundary calibration are different metrics.'}
    write_json(out/'manifest.json', manifest)
    started = time.perf_counter()
    result = {'status':'RUNNING','boundary':{},'profiles':{},'solves':{},'excluded':{},'field_sentinel':{}}
    try:
        d = cache.arrays['val']
        preds = {k:selected(v, d['x'],d['q'],d['s'],d['normal'],device) for k,v in models.items()}
        for group, ids in enumerate(cache.groups['val']):
            key = f"S{group//2}/{'bank_refined' if group%2==0 else 'offbank_refined'}"
            result['boundary'][key] = {'count':len(ids), 'models':{}}
            for name in models:
                a = preds[name]
                result['boundary'][key]['models'][name] = {
                    'abs_value':core.distribution(np.abs(a['value'][ids])),
                    'normal_cosine':core.distribution(a['cosine'][ids]),
                    'gradient_norm':core.distribution(a['norm'][ids]),
                    'gradient_norm_error':core.distribution(np.abs(a['norm'][ids]-1)),
                    'linearized_zero_shift_rad':core.distribution(a['linearized_zero_shift_rad'][ids])}
        # All in-bounds side profiles; evaluate actual signs, do not force sign from t.
        for offset in (-.05,-.02,-.01,-.005,.005,.01,.02,.05):
            qs = d['q'] + offset*d['normal']
            ids = np.flatnonzero(((qs >= cache.lo)&(qs<=cache.hi)).all(1))
            gs = np.asarray([oracle.value(torch.tensor(d['x'][i],device=device),torch.tensor(qs[i],device=device),int(d['s'][i])) for i in ids])
            for name, model in models.items():
                yp = selected(model,d['x'][ids],qs[ids],d['s'][ids],d['normal'][ids],device)['value']
                for group in range(16):
                    keep = (2*d['s'][ids]+d['kind'][ids])==group
                    key = f"S{group//2}/kind{group%2}/offset={offset:+.3f}"
                    entry = result['profiles'].setdefault(key, {'in_limits':int(keep.sum()),'models':{}})
                    entry['models'][name] = {'sign_accuracy':core.rate(int(((yp[keep]>=0)==(gs[keep]>=0)).sum()),int(keep.sum()))}
        print('[eval] boundary and actual-sign side profiles complete', flush=True)
        # Same original solver for all models. No analytic control needed for this paired ablation.
        starts, excluded = [], Counter()
        rng = np.random.default_rng(91283)
        for s in range(8):
            pool = np.unique(d['x_index'][d['s']==s])
            pts = rng.choice(pool, min(args.solve_points, len(pool)), replace=False)
            for xi in pts:
                for kind in (0,1):
                    inds = np.flatnonzero((d['s']==s)&(d['x_index']==xi)&(d['kind']==kind))
                    if not len(inds): excluded[f'S{s}/missing_kind{kind}']+=1; continue
                    i = inds[0]
                    x = torch.tensor(d['x'][i],device=device)
                    for radius in (.02,.05):
                        qout = torch.tensor(d['q'][i]-radius*d['normal'][i],device=device)
                        qin = torch.tensor(d['q'][i]+radius*d['normal'][i],device=device)
                        group=f'S{s}/local_kind{kind}_r{radius}'
                        if (not core.within(qout,lo,hi) or not core.within(qin,lo,hi)
                                or oracle.value(x,qout,s)>=-1e-5 or oracle.value(x,qin,s)<=1e-5):
                            excluded[group]+=1; continue
                        starts.append((group,int(xi),s,x,qout))
                x = dataset.x_cpu[int(xi)].to(device)
                for _ in range(2):
                    q = torch.tensor(rng.uniform(cache.lo,cache.hi),device=device,dtype=torch.float32)
                    if oracle.value(x,q,s)>=0: excluded[f'S{s}/uniform_initially_inside']+=1; continue
                    starts.append((f'S{s}/uniform_outside',int(xi),s,x,q))
        groups = defaultdict(list)
        with (out/'solves.jsonl').open('w') as f:
            for number,(group,xi,s,x,q) in enumerate(starts):
                row={'group':group,'x_index':xi,'sensor':s,'x':x.tolist(),'q_init':q.tolist(),'models':{}}
                # Rotate order to avoid assigning every first/cold query to V1.
                names=list(models); names=names[number%3:]+names[:number%3]
                for name in names: row['models'][name]=probeapi.run_probe(probes[name],oracle,x,q,s)
                f.write(json.dumps(core.json_safe(row),allow_nan=False)+'\n'); f.flush()
                groups[group].append(row)
                if (number+1)%50==0: print(f'[eval] matched_solves={number+1}/{len(starts)}',flush=True)
        for group,rows in groups.items():
            entry={'count':len(rows),'models':{}}
            for name in models:
                rr=[r['models'][name] for r in rows]
                entry['models'][name]={'fov_pass':core.rate(sum(r['fov_pass'] for r in rr),len(rr)),
                    'root_002':core.rate(sum(r['predicted_root_within_002'] for r in rr),len(rr)),
                    'solver_ms':core.distribution(r['solver_ms'] for r in rr),
                    'failure_stages':dict(Counter(r['failure_stage'] for r in rr))}
            pair=Counter('both_pass' if r['models']['P0']['fov_pass'] and r['models']['P1']['fov_pass'] else
                         'P1_only' if r['models']['P1']['fov_pass'] else 'P0_only' if r['models']['P0']['fov_pass'] else
                         'both_fail' for r in rows)
            entry['P0_P1_paired']=dict(pair); result['solves'][group]=entry
        result['excluded']=dict(excluded)
        sign_oracle=api.PinocchioFOVOracle(str(URDF),api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES,50.,66.,.2,.7,.01)
        result['field_sentinel']=field_sentinel(models,dataset,sign_oracle,device,args.field_batches)
        result['planning_sentinel']=planning_sentinel(models,dataset,sign_oracle,device,out,args.planning_batches)
        result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-started)
        write_json(out/'report.json',result)
        lines=['# V1 / P0 / P1 boundary-calibration pilot','',
               'Same parent, matched training samples, held-out evaluation. FOV-only, no runtime switch.',
               'P1 is an added-boundary-supervision ablation, not an equal-supervision-budget claim.','',
               '| Boundary | N | V1 abs(f) | P0 abs(f) | P1 abs(f) | P1 norm | P1 normal cosine |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for key,row in result['boundary'].items():
            m=row['models']; values=[m[n]['abs_value']['mean'] for n in models]+[m['P1']['gradient_norm']['mean'],m['P1']['normal_cosine']['mean']]
            lines.append(f"| {key} | {row['count']} | "+' | '.join(f'{v:.5f}' for v in values)+' |')
        lines+=['','| Solve group | N | V1 FOV | P0 FOV | P1 FOV | P1-only | P0-only |','|---|---:|---:|---:|---:|---:|---:|']
        for key,row in sorted(result['solves'].items()):
            rates=[row['models'][n]['fov_pass']['rate'] for n in models]
            pair=row['P0_P1_paired']
            lines.append(f"| {key} | {row['count']} | "+' | '.join(f'{v:.4f}' for v in rates)+f" | {pair.get('P1_only',0)} | {pair.get('P0_only',0)} |")
        lines+=['','| Global path | N | Proj abs(g)<.03 | Asc1 g>=.03 | Asc10 g>=.03 |','|---|---:|---:|---:|---:|']
        for key,row in result['planning_sentinel'].items():
            vals=[row[k] for k in ('proj_oracle_boundary_030','asc1_g_ge_0p03','asc10_g_ge_0p03')]
            lines.append(f"| {key} | {row['count']} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
        lines+=['','See report.json for profiles, global field/ranking sentinel, denominator exclusions and P0/P1 full metrics.',
                'Lower boundary loss is not a guarantee of improved solving. Check far-field and union regressions.']
        (out/'summary.md').write_text('\n'.join(lines)+'\n')
        manifest.update(status='COMPLETE',elapsed_seconds=result['elapsed_seconds'])
        write_json(out/'manifest.json',manifest)
    except Exception as exc:
        manifest.update(status='FAILED',error=repr(exc));write_json(out/'manifest.json',manifest);raise
    print(f'[done] calibration_pair_evaluation_complete {out}',flush=True)


if __name__=='__main__': main()
