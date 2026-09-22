"""Resumable per-(query,sensor) labeling, a separate frozen audit, and paired export."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from common import (RepoOracle, core, old_io, open_run, record_path, lock, read_record,
                    write_record, geometry_identity, reference_gate, sha256_file,
                    distribution, load_npz, FORMAT)


def tasks_for(queries):
    return [[int(queries['query_id'][i]),int(s)] for i,s in np.argwhere(queries['support'])]


def recheck(old, q, x, s, bank, oracle, cfg, vc):
    result=core.screen_candidates(q,bank.masks[s],bank.lo,bank.hi,
                                 oracle.geometry(x,s),old['attempts'],cfg,vc)
    return reference_gate(oracle,x,s,q,result,cfg)


def worker(out,repo,urdf,device,rank=0,world=1,stage='base',max_tasks=None,oracle_factory=RepoOracle):
    if stage not in ('base','audit') or world<1 or not 0<=rank<world:
        raise ValueError('Invalid stage/rank/world')
    if max_tasks is not None and max_tasks<0:raise ValueError('max_tasks must be nonnegative')
    out,spec,bank,queries,digest=open_run(out)
    cfg=core.SolverConfig(**spec['solver']);vc=core.VerifyConfig(**spec['verify'])
    oracle=oracle_factory(repo,urdf,device,joint_names=bank.joints,sensor_frames=bank.sensors)
    if geometry_identity(oracle.identity)!=spec['geometry']:
        raise ValueError('Geometry changed since preparation')
    report=oracle.verify(bank,queries['x_index'],queries['q_query'])
    old_io.write_json(out/f'preflight.{stage}.rank{rank}.json',report)
    tasks=tasks_for(queries) if stage=='base' else old_io.read_json(out/'audit_plan.json')['tasks']
    locations={int(v):i for i,v in enumerate(queries['query_id'])}
    done=0
    for task_id in range(rank,len(tasks),world):
        if max_tasks is not None and done>=max_tasks:break
        qid,s=tasks[task_id];folder=record_path(out,stage,qid,s)
        with lock(out/'locks'/f'{stage}_{qid}_{s}.lock'):
            if folder.exists():
                existing,_=read_record(folder,digest,[qid,s])
                if stage=='audit':
                    _,meta=read_record(record_path(out,'base',qid,s),digest,[qid,s])
                    if existing['base_result_sha256']!=meta['result_sha256']:
                        raise ValueError('Base result changed after audit')
                print(f'[resume] {stage} {qid}:S{s} checksum verified',flush=True);done+=1;continue
            row=locations[qid];xi=int(queries['x_index'][row]);q=np.asarray(queries['q_query'][row],float)
            x=np.asarray(bank.x[xi],float);mask=bank.masks[s];points=bank.sensor_bank(xi,s)
            if not len(points):raise ValueError('Frozen support no longer has a bank')
            ref=float(oracle.reference_margins(x,q[None])[0,s])
            if abs(ref-float(queries['reference_g_m'][row,s]))>2e-6:
                raise ValueError('Query reference FOV changed')
            print(f'[task] {stage} rank={rank} task={task_id+1}/{len(tasks)} query={qid}:S{s} start',flush=True)
            start=time.perf_counter();geo=oracle.geometry(x,s)
            if stage=='base':
                old=core.legacy.old_bank_label(q,points,mask,1 if ref>=0 else -1)
                attempts=core.solve_attempts(q,points,mask,bank.lo,bank.hi,geo,cfg)
                new=core.screen_candidates(q,mask,bank.lo,bank.hi,geo,attempts,cfg,vc)
                new=reference_gate(oracle,x,s,q,new,cfg)
                new['verification']=dict(status='NOT_RUN',probes=[])
                result=dict(task=[qid,s],x_index=xi,split=int(queries['split'][row]),
                    query_group=int(queries['query_group'][row]),q_query=q,x=x,reference_g_m=ref,
                    old=old,new=new,attempts=attempts,elapsed_ms=1000*(time.perf_counter()-start),
                    stage='base',gradient_verified=False,training_ready=False,device=device)
                # Merely having a finite displacement is NEVER a verified gradient.
                if new['grad_valid']:raise AssertionError('Base generation unexpectedly verified gradient')
            else:
                base,meta=read_record(record_path(out,'base',qid,s),digest,[qid,s])
                if not np.array_equal(np.asarray(base['q_query'],np.float32),q.astype(np.float32)):
                    raise ValueError('Base/query mismatch')
                screened=recheck(base,q,x,s,bank,oracle,cfg,vc)
                warm=[c['q_star'] for c in screened['clusters'][:2] if c['qualified']]
                probe_cfg=replace(cfg,starts=vc.probe_starts,maxiter=vc.probe_maxiter)
                def probe(trial):
                    a=core.solve_attempts(trial,points,mask,bank.lo,bank.hi,geo,probe_cfg,warm)
                    rr=core.screen_candidates(trial,mask,bank.lo,bank.hi,geo,a,cfg,vc)
                    rr=reference_gate(oracle,x,s,trial,rr,cfg);rr['attempts']=a;return rr
                def progress(msg): print(f'[probe] query={qid}:S{s} {msg}',flush=True)
                new=core.verify_gradient(screened,q,mask,bank.lo,bank.hi,probe,cfg,vc,progress)
                result=dict(task=[qid,s],x_index=xi,split=int(queries['split'][row]),
                    q_query=q,new=new,base_result_sha256=meta['result_sha256'],
                    central_optimizations=0,central_attempts_reused=len(base['attempts']),
                    elapsed_ms=1000*(time.perf_counter()-start),stage='audit',device=device,
                    gradient_verified=bool(new['grad_valid']),training_ready=False)
            write_record(folder,result,digest)
            status=result['new']['verification']['status']
            print(f'[task] COMPLETE {stage} {qid}:S{s} value={new["value_valid"]} '
                  f'gradient_check={status} elapsed_s={result["elapsed_ms"]/1000:.3f} '
                  f'reasons={new["gradient_reasons"]}',flush=True)
        done+=1
    return done


def union_targets(a):
    """Keep the original sensor-support set; missing labels never erase a possible winner."""
    n=len(a['query_id']);a['paired_union_value_mask']=np.zeros(n,bool);a['paired_union_grad_mask']=np.zeros(n,bool)
    for mode in ('old','new'):
        a['union_'+mode+'_value']=np.full(n,np.nan,np.float32)
        a['union_'+mode+'_grad']=np.full((n,7),np.nan,np.float32)
    for i in range(n):
        support=a['support'][i]
        if not support.any() or not a['paired_value_mask'][i,support].all():continue
        a['paired_union_value_mask'][i]=True;good=True
        for mode in ('old','new'):
            values=np.where(support,a[mode+'_value'][i],-np.inf);s=int(np.argmax(values))
            ordered=np.sort(values[support]);unique=len(ordered)<2 or ordered[-1]-ordered[-2]>1e-5
            a['union_'+mode+'_value'][i]=values[s]
            if a['paired_grad_mask'][i,s]:a['union_'+mode+'_grad'][i]=a[mode+'_grad'][i,s]
            good=good and unique and bool(a['paired_grad_mask'][i,s])
        a['paired_union_grad_mask'][i]=good


def _attempt_stats(attempts):
    return dict(attempts=len(attempts),success=sum(bool(a.get('success')) for a in attempts),
                iterations=sum(int(a.get('iterations',0)) for a in attempts),
                geometry_calls=sum(int(a.get('geometry_calls',0)) for a in attempts),
                attempt_ms=sum(float(a.get('elapsed_ms',0)) for a in attempts),
                failed_attempt_ms=sum(float(a.get('elapsed_ms',0)) for a in attempts if not a.get('success')))


def merge(out,audited=False,shard_size=32):
    out,spec,bank,q,digest=open_run(out);plan=old_io.read_json(out/'audit_plan.json')
    required=set(tuple(v) for v in plan['tasks'])
    target=out/('paired_cache' if audited else 'base_cache')
    if shard_size<1:raise ValueError('shard_size must be positive')
    # All required records must exist before installing any complete cache.
    provenance={}
    for qid,s in tasks_for(q):
        _,meta=read_record(record_path(out,'base',qid,s),digest,[qid,s])
        provenance[f'base/{qid}:{s}']=meta['result_sha256']
    if audited:
        for qid,s in plan['tasks']:
            result,meta=read_record(record_path(out,'audit',qid,s),digest,[qid,s])
            if result['base_result_sha256']!=provenance[f'base/{qid}:{s}']:
                raise ValueError('Audit is attached to a different base result')
            provenance[f'audit/{qid}:{s}']=meta['result_sha256']
    if target.exists():
        idx=old_io.read_json(target/'dataset_index.json')
        if idx['spec_sha256']!=digest or idx['record_sha256']!=provenance or idx['audit_complete']!=audited:
            raise ValueError('Existing export identity differs; no overwrite')
        for e in idx['shards']:
            if sha256_file(target/e['path'])!=e['sha256']:raise ValueError('Corrupt export')
        if sha256_file(target/'summary.json')!=idx['summary_sha256']:raise ValueError('Corrupt summary')
        print('[resume] complete export verified',target,flush=True);return old_io.read_json(target/'summary.json')
    tmp=Path(tempfile.mkdtemp(prefix='.'+target.name+'.',dir=out))
    try:
        summary=dict(format=FORMAT,status='PAIRED_CACHE_COMPLETE_REVIEW_REQUIRED' if audited else 'BASE_LABELS_COMPLETE_AUDIT_NOT_RUN',
            scope=spec['scope'],training_ready=False,global_nearest_certified=False,
            query_count=len(q['query_id']),audit_selected=len(required),audit_complete=audited,
            audit_plan=plan,source_unchanged=True,by_sensor_split={},reasons={},strata={},
            limitations=['Validity is a local numerical gate, not global nearest-distance truth',
                'No R1 training/runtime or model improvement evaluated',
                'Unselected gradients remain NOT_RUN; audit failures are not replaced',
                'Near-bank normal pairs are NOT verified local-boundary witnesses',
                'Support-balanced finite point sample, not a full-dataset census',
                'Partial value/gradient/union masks need a cache-aware training objective'])
        stats=defaultdict(Counter);strata=defaultdict(Counter);reasons=Counter();cost=[];audit_cost=[]
        attempts_total=Counter();entries=[]
        for start in range(0,len(q['query_id']),shard_size):
            stop=min(start+shard_size,len(q['query_id']));n=stop-start
            a={k:np.asarray(v[start:stop]).copy() for k,v in q.items()}
            a['x']=np.asarray(bank.x[a['x_index']],np.float32)
            for key in ('old_value','new_value'):a[key]=np.full((n,8),np.nan,np.float32)
            for key in ('old_grad','new_grad','gradient_candidate','q_star'):a[key]=np.full((n,8,7),np.nan,np.float32)
            for key in ('old_value_valid','old_grad_regular','new_value_valid','gradient_candidate_valid',
                        'new_grad_valid','audit_selected','paired_value_mask','paired_grad_mask'):
                a[key]=np.zeros((n,8),bool)
            a['gradient_status']=np.full((n,8),'NO_BANK_SUPPORT',dtype='U32')
            for i in range(n):
                row=start+i;qid=int(a['query_id'][i]);split=int(a['split'][i])
                for s in np.flatnonzero(a['support'][i]):
                    s=int(s);task=[qid,s];base,_=read_record(record_path(out,'base',qid,s),digest,task)
                    if base['x_index']!=int(a['x_index'][i]) or base['split']!=split or not np.array_equal(np.asarray(base['q_query'],np.float32),a['q_query'][i]):
                        raise ValueError('Input/label identity mismatch')
                    new=base['new'];selected=(qid,s) in required
                    a['audit_selected'][i,s]=selected
                    if audited and selected:
                        review,_=read_record(record_path(out,'audit',qid,s),digest,task)
                        new=review['new'];audit_cost.append(review['elapsed_ms'])
                        for pr in new['verification']['probes']:attempts_total.update(_attempt_stats(pr.get('attempts',[])))
                    verification=new['verification']['status']
                    a['gradient_status'][i,s]=verification
                    for mode,values in (('old',base['old']),('new',new)):
                        a[mode+'_value'][i,s]=float(values['value']) if values['value'] is not None else np.nan
                        a[mode+'_grad'][i,s]=np.asarray(values['grad'],float)
                    a['old_value_valid'][i,s]=base['old']['valid'];a['old_grad_regular'][i,s]=base['old']['grad_regular']
                    a['new_value_valid'][i,s]=new['value_valid']
                    candidate=np.asarray(new['gradient_candidate'],float)
                    a['gradient_candidate'][i,s]=candidate;a['q_star'][i,s]=np.asarray(new['q_star'],float)
                    blockers=[r for r in new['gradient_reasons'] if r!='DISTANCE_DERIVATIVE_NOT_VERIFIED']
                    a['gradient_candidate_valid'][i,s]=bool(new['value_valid'] and np.isfinite(candidate).all() and not blockers)
                    a['new_grad_valid'][i,s]=bool(audited and selected and verification=='PASS' and new['grad_valid'] and new['value_valid'])
                    if new['grad_valid']!=a['new_grad_valid'][i,s]:raise ValueError('Verified gradient lacks an audited PASS')
                    k=f'{split}:S{s}';st=stats[k];st['supported']+=1;st['new_values']+=int(new['value_valid'])
                    st['candidate_gradients']+=int(a['gradient_candidate_valid'][i,s]);st['verified_gradients']+=int(a['new_grad_valid'][i,s])
                    st['audit_selected']+=int(selected);st['audit_'+verification]+=1
                    g=float(a['reference_g_m'][i,s]);sign='inside' if g>core.SolverConfig().sign_guard_m else 'outside' if g<-core.SolverConfig().sign_guard_m else 'near_zero'
                    stratum=strata[f'{k}/{int(a["query_group"][i])}/{sign}'];stratum['supported']+=1
                    stratum['value_valid']+=int(new['value_valid']);stratum['verified_gradients']+=int(a['new_grad_valid'][i,s])
                    stratum['audit_selected']+=int(selected)
                    reasons.update(blockers);cost.append(base['elapsed_ms']);attempts_total.update(_attempt_stats(base['attempts']))
            a['paired_value_mask']=a['old_value_valid']&a['new_value_valid']
            a['paired_grad_mask']=a['paired_value_mask']&a['old_grad_regular']&a['new_grad_valid']
            union_targets(a)
            for mode in ('old','new'):
                if not np.isfinite(a[mode+'_value'][a['paired_value_mask']]).all():raise ValueError('Nonfinite supervised value')
                if not np.isfinite(a[mode+'_grad'][a['paired_grad_mask']]).all():raise ValueError('Nonfinite supervised gradient')
            name=f'shard_{len(entries):05d}.npz';old_io.write_npz(tmp/name,a)
            entries.append(dict(path=name,sha256=sha256_file(tmp/name),rows=n))
            for split in (0,1):
                stats[f'{split}:union']['values']+=int(a['paired_union_value_mask'][a['split']==split].sum())
                stats[f'{split}:union']['gradients']+=int(a['paired_union_grad_mask'][a['split']==split].sum())
        summary.update(by_sensor_split=dict(stats),strata=dict(strata),reasons=dict(reasons),
                       base_elapsed_ms=distribution(cost),audit_elapsed_ms=distribution(audit_cost),optimization_work=dict(attempts_total))
        old_io.write_json(tmp/'summary.json',summary)
        lines=['# R1 paired offline labels v3','',f'Queries: {summary["query_count"]}; status: {summary["status"]}',
               'Approximate labels; NOT a full dataset or global-distance certificate. No training has run.','',
               '| Split:sensor | Supported | Valid values | Candidate gradients | FD-verified gradients |',
               '|---|---:|---:|---:|---:|']
        for k,st in sorted(stats.items()):
            if k.endswith('union'):continue
            lines.append(f'| {k} | {st["supported"]} | {st["new_values"]} | {st["candidate_gradients"]} | {st["verified_gradients"]} |')
        lines+=['',f'Audit complete: {audited}. NOT_RUN is not PASS.',
                'Inspect sampling_report.json and summary.json for actual inside/outside coverage, exclusions, failures and cost.',
                'training_ready=false: a cache-aware objective and a reviewed supervision protocol are still required.']
        (tmp/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        old_io.write_json(tmp/'dataset_index.json',dict(format=FORMAT,complete=True,audit_complete=audited,
            spec_sha256=digest,record_sha256=provenance,shards=entries,query_count=len(q['query_id']),
            training_ready=False,gradient_policy='FD_PASS_ONLY',label_semantics='per_sensor_approximate_signed_joint_distance',
            summary_sha256=sha256_file(tmp/'summary.json')))
        os.rename(tmp,target)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    print((target/'summary.md').read_text(),flush=True)
    print('[merge] COMPLETE',target,flush=True);return summary
