#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib, json
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
# e012_protocol loads the historical scratch trainer, which may register its
# own model.py as sys.modules["model"].  Load the R012 architecture explicitly
# by file path so evaluation cannot resolve the wrong module by name.
r012_model=eproto._load('r012_unified_local_model',HERE/'model.py')
build_r012_model=r012_model.build_model
compat=eproto._load('r012_unified_eval_compat',ABC_DIR/'eval_compat.py')
old=eproto.old
R_ARMS=('R0','R1','R2'); E_ARMS=('E0','E1')
MODEL_ORDER=('V1','P0','E0','E1','R0','R1','R2')
R_FORMAT='care_h9_scratch50k_r012_v1'

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def write_json(path:Path,value): old.write_json(path,value)
def fraction(k,n): return {'passed':int(k),'count':int(n),'rate':float(k/n) if n else None}

def load_r_checkpoint(root:Path,arm:str):
    base=root/'formal'/arm; final=base/'final.pt'; run_path=base/'run.json'
    if not final.is_file() or not run_path.is_file(): raise FileNotFoundError(base)
    run=json.loads(run_path.read_text()); digest=sha256(final)
    expected={'status':'COMPLETE','arm':arm,'successful_updates':50000,'training_stream_updates':50000,'final_sha256':digest}
    for k,v in expected.items():
        if run.get(k)!=v: raise ValueError(f'{arm} run mismatch {k}: {run.get(k)!r} != {v!r}')
    cp=torch.load(final,map_location='cpu',weights_only=False)
    checks={'format':R_FORMAT,'arm':arm,'completed':True,'step':50000,'initialization':'random_from_scratch','frozen_parameters':0}
    for k,v in checks.items():
        if cp.get(k)!=v: raise ValueError(f'{arm} checkpoint mismatch {k}: {cp.get(k)!r} != {v!r}')
    if cp.get('training_stream_sha256')!=run.get('training_stream_sha256'): raise ValueError(f'{arm} checkpoint/run stream hash mismatch')
    if cp.get('training_stream_updates')!=50000: raise ValueError(f'{arm} checkpoint stream update count mismatch')
    return cp,run,digest

def load_r_model(cp,arm,device):
    model=build_r012_model(arm); model.load_state_dict(cp['model_state'],strict=True)
    return model.to(device=device,dtype=torch.float32).eval().requires_grad_(False)

def solve_summary(rows,names,stats):
    out={'count':len(rows),'models':{},'paired_vs_V1':{}}
    for name in names:
        rr=[r['models'][name] for r in rows]
        out['models'][name]={'fov_pass':fraction(sum(r['fov_pass'] for r in rr),len(rr)),
            'root_002':fraction(sum(r['predicted_root_within_002'] for r in rr),len(rr)),
            'failure_stages':dict(Counter(r['failure_stage'] for r in rr)),
            'root_source_by_failure':dict(Counter(r['root_source']+' / '+r['failure_stage'] for r in rr)),
            'solver_ms':stats.finite_dist([r['solver_ms'] for r in rr])}
    for name in names:
        if name=='V1': continue
        out['paired_vs_V1'][name]=dict(Counter(
            'both_pass' if r['models']['V1']['fov_pass'] and r['models'][name]['fov_pass'] else
            name+'_only' if r['models'][name]['fov_pass'] else
            'V1_only' if r['models']['V1']['fov_pass'] else 'both_fail' for r in rows))
    return out

def aggregate(rows,names):
    result={}
    for key,chosen in (('local',[r for r in rows if '/local_' in r['group']]),
                       ('uniform',[r for r in rows if r['group'].endswith('uniform_outside')]),('all',rows)):
        result[key]={n:fraction(sum(r['models'][n]['fov_pass'] for r in chosen),len(chosen)) for n in names}
    return result

def per_sensor_aggregate(rows,names):
    out={}
    for s in range(8):
        chosen=[r for r in rows if r['sensor']==s]
        out[f'S{s}']={n:fraction(sum(r['models'][n]['fov_pass'] for r in chosen),len(chosen)) for n in names}
    return out

def markdown(report,names):
    def f(v): return 'N/A' if v is None else f'{v:.5f}'
    lines=['# Unified H9 development evaluation','',
        'Models: V1 / P0 / E0 / E1 / R0 / R1 / R2.',
        'R0/R1/R2 are scratch-50k runs with an exactly matched training stream.',
        'Development FOV-only evaluation; LOS/collision/trajectory/actual-seen NOT_RUN.','',
        '## Matched solver aggregate','',
        '| Cohort | N | '+' | '.join(n+' FOV' for n in names)+' |',
        '|---|---:|'+'---:|'*len(names)]
    for cohort in ('local','uniform','all'):
        row=report['solve_aggregate'][cohort]; n=next(iter(row.values()))['count']
        lines.append('| '+cohort+f' | {n} | '+' | '.join(f(row[nm]['rate']) for nm in names)+' |')
    lines += ['','## Offline sensor-max field','',
        '| Model | MAE | sign | grad cosine | rank top1 | rank top2 | fallback |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        fs=report['field_sentinel'][n]; sm=fs['fields']['sensor_max']; rk=fs['ranking']
        lines.append(f"| {n} | {f(sm['mae'])} | {f(sm['sign_accuracy'])} | {f(sm['gradient_cosine_mean'])} | {f(rk['winner_top1_accuracy_or_recall'])} | {f(rk['winner_top2_accuracy_or_recall'])} | {f(rk['fallback_accuracy_after_gt_winner_removed'])} |")
    lines += ['','## Union field','','| Model | MAE | sign | grad cosine |','|---|---:|---:|---:|']
    for n in names:
        u=report['field_sentinel'][n]['fields']['union']
        lines.append(f"| {n} | {f(u['mae'])} | {f(u['sign_accuracy'])} | {f(u['gradient_cosine_mean'])} |")
    lines += ['','## Planning sensor-max','','| Model | projection | ascent1 | ascent10 |','|---|---:|---:|---:|']
    for n in names:
        p=report['planning_sentinel'][f'{n}/sensor_max']
        lines.append(f"| {n} | {f(p['proj_oracle_boundary_030'])} | {f(p['asc1_g_ge_0p03'])} | {f(p['asc10_g_ge_0p03'])} |")
    lines += ['','## R0/R1/R2 scratch training','','| Arm | final val loss | best val | final SHA256 |','|---|---:|---:|---|']
    for arm in R_ARMS:
        t=report['training'][arm]
        lines.append(f"| {arm} | {f(t['final_val_loss'])} | {f(t['best_val'])} | {t['final_sha256']} |")
    lines += ['','## Per-sensor matched solve','']
    for s in range(8):
        row=report['solve_per_sensor'][f'S{s}']; n=next(iter(row.values()))['count']
        lines += [f'### S{s} (N={n})','', '| Model | FOV pass |','|---|---:|']
        for name in names: lines.append(f"| {name} | {f(row[name]['rate'])} |")
        lines.append('')
    lines += ['Full report.json contains boundary geometry, two-sided actual-FOV sign profiles, fixed radial neighborhood, failure/root-source tables, paired outcomes, field/ranking, and planning metrics.']
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--r012-root',type=Path,required=True); ap.add_argument('--reference-root',type=Path,required=True); args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device=torch.device('cuda',0); torch.cuda.set_device(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    rroot=args.r012_root.resolve(); refroot=args.reference_root.resolve(); out=rroot/'evaluation_unified_dev'
    if out.exists(): raise FileExistsError(f'No overwrite: {out}')
    cache,p0,p0_sha=eproto.load_reference_root(refroot)
    models={}; checkpoint_digests={}; training={}
    artifact=Path(p0['args']['artifact_root'])
    v1,_=old.load_v1(artifact/old.V1_REL,device); models['V1']=v1.eval().requires_grad_(False)
    models['P0']=eproto.p0_model(p0,device).eval().requires_grad_(False)
    for arm in E_ARMS:
        cp,digest=eproto.load_checkpoint(eproto.output_dir(refroot,arm,'pilot')/'final.pt')
        eproto.assert_checkpoint(cp,p0,p0_sha,cache.identity,require_pilot=True)
        models[arm]=eproto.load_model_from_checkpoint(cp,p0,device).requires_grad_(False); checkpoint_digests[arm]=digest
    r_streams=set()
    for arm in R_ARMS:
        cp,run,digest=load_r_checkpoint(rroot,arm); r_streams.add(run['training_stream_sha256'])
        models[arm]=load_r_model(cp,arm,device); checkpoint_digests[arm]=digest
        training[arm]={'final_val_loss':cp.get('stats',{}).get('val',{}).get('loss'),'best_val':cp.get('best_val'),'final_sha256':digest,
            'training_stream_sha256':run['training_stream_sha256'],'training_stream_updates':run['training_stream_updates'],'architecture':cp.get('architecture'),'routing':cp.get('routing')}
    if len(r_streams)!=1: raise ValueError(f'R0/R1/R2 stream mismatch: {r_streams}')
    names=list(MODEL_ORDER)
    if list(models)!=names: raise RuntimeError(f'Unexpected model order: {list(models)}')
    ev=old.module('r012_unified_evaluate_pair',eproto.abc.p2.LEGACY); stats=old.module('r012_unified_evaluate_p2',eproto.abc.P2_DIR)
    side=old.module('r012_unified_neighborhood',eproto.P3_DIR); api=old.module('r012_unified_train_api',old.SCRIPTS)
    dataset=api.VisibilityQ0Dataset(str(artifact/old.DATA_REL),1000,0); cache.verify_dataset(dataset,artifact/old.DATA_REL)
    oracle=old.module('r012_unified_oracle',old.AUDIT).SensorOracle(old.URDF,device,api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES)
    pair=side.PairwiseFOV(oracle); core=old.module('r012_unified_core',old.AUDIT); probeapi=old.module('r012_unified_runtime_probe',old.AUDIT)
    lo,hi=dataset.q_limits(device); probes={k:probeapi.make_probe(ev.SensorView(v),dataset.sensor_masks(device),lo,hi) for k,v in models.items()}
    checks=old.module('r012_unified_audit',old.AUDIT).preflight(dataset,oracle,probes,device); checks['pairwise_neighbor_FOV']=pair.verify(cache,device)
    out.mkdir(parents=True)
    manifest={'status':'RUNNING','evaluation':'unified_development','evaluated_models':names,'r012_training_streams':'MATCH','r012_training_stream_sha256':next(iter(r_streams)),
        'r012_training_stream_updates':50000,'checkpoint_sha256':checkpoint_digests,'p0_sha256':p0_sha,'cache_manifest_sha256':cache.identity,'preflight':checks,
        'field_batches':10,'planning_batches':2,'solve_points_per_sensor':32,'limitations':'Development held-out FOV-only evaluation; LOS/collision/trajectory/actual-seen NOT_RUN.'}
    write_json(out/'manifest.json',manifest); started=time.perf_counter()
    try:
        result={'status':'RUNNING','evaluation':'unified_development','training':training,'boundary':{},'profiles':{},'solves':{}}
        d=cache.arrays['val']; preds={n:ev.selected(m,d['x'],d['q'],d['s'],d['normal'],device) for n,m in models.items()}
        for group,ids in enumerate(cache.groups['val']):
            key=f"S{group//2}/{'bank_refined' if group%2==0 else 'offbank_refined'}"; result['boundary'][key]={'count':len(ids),'models':{n:stats.boundary_stats(a,ids) for n,a in preds.items()}}
        for offset in (-.05,-.02,-.01,-.005,.005,.01,.02,.05):
            qs=d['q']+offset*d['normal']; ids=np.flatnonzero(((qs>=cache.lo)&(qs<=cache.hi)).all(1))
            gs=np.asarray([oracle.value(torch.tensor(d['x'][i],device=device),torch.tensor(qs[i],device=device),int(d['s'][i])) for i in ids])
            for n,m in models.items():
                yp=ev.selected(m,d['x'][ids],qs[ids],d['s'][ids],d['normal'][ids],device)['value']
                for group in range(16):
                    keep=(2*d['s'][ids]+d['kind'][ids])==group; key=f'S{group//2}/kind{group%2}/offset={offset:+.3f}'
                    row=result['profiles'].setdefault(key,{'in_limits':int(keep.sum()),'models':{}}); row['models'][n]=stats.confusion(yp[keep],gs[keep])
        print('[eval] boundary/sign complete',flush=True)
        radial=side.radii_for_update(0,0,len(d['s'])); accum={n:torch.zeros((32,len(side.COLUMNS)),device=device,dtype=torch.float64) for n in names}
        with torch.no_grad():
            for start in range(0,len(d['s']),128):
                ids=np.arange(start,min(start+128,len(d['s']))); nb=side.build_queries(cache.tensors('val',ids,device),radial[ids],pair,lo,hi)
                for n,m in models.items(): _,st=side.sign_loss(m,nb,torch.ones(32,device=device)); accum[n]+=st
        result['neighborhood_sentinel']={n:side.summary(st) for n,st in accum.items()}
        starts=[];excluded=Counter();rng=np.random.default_rng(91283)
        for s in range(8):
            pool=np.unique(d['x_index'][d['s']==s]);pts=rng.choice(pool,min(manifest['solve_points_per_sensor'],len(pool)),replace=False)
            for xi in pts:
                for kind in (0,1):
                    ii=np.flatnonzero((d['s']==s)&(d['x_index']==xi)&(d['kind']==kind))
                    if not len(ii): excluded[f'S{s}/missing_kind{kind}']+=1;continue
                    i=ii[0];x=torch.tensor(d['x'][i],device=device)
                    for radius in (.02,.05):
                        qout=torch.tensor(d['q'][i]-radius*d['normal'][i],device=device);qin=torch.tensor(d['q'][i]+radius*d['normal'][i],device=device);group=f'S{s}/local_kind{kind}_r{radius}'
                        if (not core.within(qout,lo,hi) or not core.within(qin,lo,hi) or oracle.value(x,qout,s)>=-1e-5 or oracle.value(x,qin,s)<=1e-5): excluded[group]+=1;continue
                        starts.append((group,int(xi),s,x,qout))
                x=dataset.x_cpu[int(xi)].to(device)
                for _ in range(2):
                    q=torch.tensor(rng.uniform(cache.lo,cache.hi),device=device,dtype=torch.float32)
                    if oracle.value(x,q,s)>=0: excluded[f'S{s}/uniform_initially_inside']+=1;continue
                    starts.append((f'S{s}/uniform_outside',int(xi),s,x,q))
        groups=defaultdict(list);all_rows=[]
        with (out/'solves.jsonl').open('w') as fsolves:
            for number,(group,xi,s,x,q) in enumerate(starts):
                row={'group':group,'x_index':xi,'sensor':s,'x':x.tolist(),'q_init':q.tolist(),'models':{}};order=names[number%len(names):]+names[:number%len(names)]
                for n in order: row['models'][n]=probeapi.run_probe(probes[n],oracle,x,q,s)
                fsolves.write(json.dumps(core.json_safe(row),allow_nan=False)+'\n');fsolves.flush();groups[group].append(row);all_rows.append(row)
                if (number+1)%50==0: print(f'[eval] matched_solves={number+1}/{len(starts)}',flush=True)
        result['solves']={k:solve_summary(v,names,stats) for k,v in groups.items()};result['solve_aggregate']=aggregate(all_rows,names);result['solve_per_sensor']=per_sensor_aggregate(all_rows,names)
        result['solve_all_summary']=solve_summary(all_rows,names,stats);result['excluded']=dict(excluded)
        sign_oracle=api.PinocchioFOVOracle(str(old.URDF),api.DEFAULT_JOINT_NAMES,api.DEFAULT_SENSOR_FRAMES,50.,66.,.2,.7,.01)
        result['field_sentinel']=ev.field_sentinel(models,dataset,sign_oracle,device,manifest['field_batches'])
        planning_models={n:compat.legacy_compatible(m) for n,m in models.items()};result['planning_sentinel']=ev.planning_sentinel(planning_models,dataset,sign_oracle,device,out,manifest['planning_batches'])
        result.update(status='COMPLETE',elapsed_seconds=time.perf_counter()-started);write_json(out/'report.json',result);(out/'summary.md').write_text(markdown(result,names),encoding='utf-8')
        manifest.update(status='COMPLETE',elapsed_seconds=result['elapsed_seconds']);write_json(out/'manifest.json',manifest)
    except Exception as exc:
        manifest.update(status='FAILED',error=repr(exc));write_json(out/'manifest.json',manifest);raise
    print(f'[done] r012_unified_evaluation_complete output={out}',flush=True)
if __name__=='__main__': main()
