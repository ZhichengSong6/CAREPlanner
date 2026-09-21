"""Pair old/new labels on immutable queries; shard, resume, and report."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import sys
import scipy
import torch
import numpy as np
from cache_io import (FORMAT, clean_json, write_json, write_npz, read_json, open_job)
from label_core import SolverConfig, label_continuous, old_bank_label
from repo_oracle import RepoOracle, source_identity, sha256_file

def library_versions():
    return dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)


STATUS_CODES = {'NO_BANK_SUPPORT':0,'NO_FEASIBLE_BOUNDARY_FOUND':1,
                'FEASIBLE_LOW_CONFIDENCE':2,'APPROX_VALUE_ONLY':3,
                'APPROX_VALUE_AND_GRAD':4,'UPSTREAM_GEOMETRY_OR_SIGN_MISMATCH':5}


def add_union_labels(a):
    """Same historical max-envelope, but missing labels never erase a sensor.

    Shared pair masks prevent the old arm from having more supervision than the
    new arm. Both winners must have regular gradients for paired union-gradient
    supervision. This does NOT change the global runtime union semantics.
    """
    n=len(a['query_id'])
    a['paired_union_value_mask']=np.zeros(n,bool)
    a['paired_union_grad_mask']=np.zeros(n,bool)
    for mode in ('old','new'):
        a['union_'+mode+'_value']=np.full(n,np.nan,np.float32)
        a['union_'+mode+'_grad']=np.full((n,7),np.nan,np.float32)
    for i in range(n):
        support=a['support'][i]
        complete=bool(support.any() and np.all(a['paired_value_mask'][i][support]))
        if not complete: continue
        winners={};unique={}
        for mode in ('old','new'):
            y=np.where(support,a[mode+'_value'][i],-np.inf)
            winner=int(np.argmax(y));winners[mode]=winner
            a['union_'+mode+'_value'][i]=y[winner]
            vals=np.sort(y[support])
            unique[mode]=len(vals)==1 or vals[-1]-vals[-2]>1e-5
            if np.isfinite(a[mode+'_grad'][i,winner]).all():
                a['union_'+mode+'_grad'][i]=a[mode+'_grad'][i,winner]
        a['paired_union_value_mask'][i]=True
        a['paired_union_grad_mask'][i]=bool(all(unique.values()) and
            all(a['paired_grad_mask'][i,w] for w in winners.values()))


def label_rows(bank, queries, row_ids, oracle, cfg):
    n=len(row_ids)
    a={key:np.asarray(value[row_ids]).copy() for key,value in queries.items()}
    a['x']=np.asarray(bank.x[a['x_index']],np.float32)
    a['source_grid_index']=np.asarray(bank.arrays['k'][a['x_index']],np.int64)
    for name in ('old_value','new_value','old_distance_unfloored','g_query_reference_m',
                 'g_query_float64_m','boundary_residual_m','reference_boundary_g_m',
                 'stationarity_relative','normal_cosine','fd_relative_error','solver_ms'):
        a[name]=np.full((n,8),np.nan,np.float32)
    for name in ('old_grad','new_grad','q_star'):
        a[name]=np.full((n,8,7),np.nan,np.float32)
    for name in ('support','old_value_valid','old_grad_regular','new_value_valid','new_grad_valid','ambiguous'):
        a[name]=np.zeros((n,8),bool)
    a['status']=np.zeros((n,8),np.uint8)
    a['geometry_calls']=np.zeros((n,8),np.int32)
    audit=[]
    for i,row_id in enumerate(row_ids):
        xi=int(queries['x_index'][row_id]);q=np.asarray(queries['q_query'][row_id],float)
        x=np.asarray(bank.x[xi],float)
        ref=oracle.reference_margins(x,q[None])[0]
        record=dict(query_id=int(queries['query_id'][row_id]),x_index=xi,split=int(queries['split'][row_id]),
                    x=x.tolist(),q_query=q.tolist(),sensors={})
        for s in range(8):
            points=bank.sensor_bank(xi,s)
            a['support'][i,s]=len(points)>0
            a['g_query_reference_m'][i,s]=ref[s]
            if len(points)==0:
                record['sensors'][str(s)]=dict(status='NO_BANK_SUPPORT',global_nearest_certified=False)
                continue
            old=old_bank_label(q,points,bank.masks[s],1 if ref[s]>=0 else -1)
            a['old_value'][i,s]=old['value'];a['old_grad'][i,s]=old['grad']
            a['old_value_valid'][i,s]=old['valid'];a['old_grad_regular'][i,s]=old['grad_regular']
            a['old_distance_unfloored'][i,s]=old['distance']
            new=label_continuous(q,points,bank.masks[s],bank.lo,bank.hi,oracle.geometry(x,s),cfg)
            a['g_query_float64_m'][i,s]=new['query_g_m']
            reference_best=np.nan
            if np.isfinite(new['q_star']).all():
                reference_best=float(oracle.reference_margins(x,new['q_star'][None])[0,s])
                # Revalidate the SAVED FP32 candidate using the unmodified upstream oracle.
                geom_ok=abs(reference_best)<=cfg.boundary_tol_m
                sign_ok=(ref[s]>=0)==(new['query_g_m']>=0)
                near_boundary_ok=abs(ref[s])>cfg.sign_guard_m or (new['value']==0 and abs(ref[s])<=cfg.boundary_tol_m)
                if not geom_ok or not sign_ok or not near_boundary_ok:
                    new['value_valid']=False;new['grad_valid']=False
                    new['status']='UPSTREAM_GEOMETRY_OR_SIGN_MISMATCH'
            a['reference_boundary_g_m'][i,s]=reference_best
            a['new_value'][i,s]=new['value'];a['new_grad'][i,s]=new['grad'];a['q_star'][i,s]=new['q_star']
            a['new_value_valid'][i,s]=new['value_valid'];a['new_grad_valid'][i,s]=new['grad_valid']
            a['ambiguous'][i,s]=new['ambiguity'];a['status'][i,s]=STATUS_CODES[new['status']]
            for key in ('boundary_residual_m','stationarity_relative','normal_cosine','fd_relative_error'):
                a[key][i,s]=new[key]
            a['solver_ms'][i,s]=new['elapsed_ms'];a['geometry_calls'][i,s]=new['geometry_calls']
            record['sensors'][str(s)]=dict(old=old,new=new,reference_boundary_g_m=reference_best)
        audit.append(clean_json(record))
        print(f"[query] id={record['query_id']} supported={a['support'][i].sum()} "
              f"new_value_valid={a['new_value_valid'][i].sum()} new_grad_valid={a['new_grad_valid'][i].sum()}",flush=True)
    a['paired_value_mask']=a['old_value_valid']&a['new_value_valid']
    a['paired_grad_mask']=a['paired_value_mask']&a['old_grad_regular']&a['new_grad_valid']
    add_union_labels(a)
    return a,audit


def prepare_preflight(out,repo,urdf,device,cfg):
    out,m,bank,queries=open_job(out)
    oracle=RepoOracle(repo,urdf,device,joint_names=bank.joints,sensor_frames=bank.sensors)
    report=oracle.verify(bank,queries['x_index'],queries['q_query'])
    spec=dict(format=FORMAT,queries_sha256=m['queries_sha256'],solver=asdict(cfg),
              geometry_identity=oracle.identity,library_versions=library_versions(),status_codes=STATUS_CODES,
              approximation='best_found_stationary_boundary_candidate_not_global_certificate')
    target=out/'run_spec.json'
    if target.exists() and read_json(target)!=spec:
        raise ValueError('Run configuration changed; choose a NEW output directory')
    if not target.exists(): write_json(target,spec)
    write_json(out/'preflight.json',report)
    print('[preflight] PASS',out/'preflight.json',flush=True)
    return report


def _verify_shard(folder,spec_hash,query_ids):
    meta=read_json(folder/'complete.json')
    if meta['run_spec_sha256']!=spec_hash or meta['query_ids']!=[int(i) for i in query_ids]:
        raise ValueError(f'Existing shard identity mismatch: {folder}')
    for name,digest in meta['files_sha256'].items():
        if sha256_file(folder/name)!=digest: raise ValueError(f'Corrupt shard: {folder/name}')
    return meta


def worker(out,repo,urdf,device,rank=0,world=1,max_shards=None):
    out,m,bank,queries=open_job(out)
    if rank<0 or rank>=world or world<1: raise ValueError('Invalid rank/world')
    spec=read_json(out/'run_spec.json');spec_hash=sha256_file(out/'run_spec.json')
    if spec.get('library_versions')!=library_versions(): raise ValueError('Library versions changed; use a new output directory')
    if read_json(out/'preflight.json').get('status')!='PASS': raise ValueError('Preflight has not passed')
    oracle=RepoOracle(repo,urdf,device,joint_names=bank.joints,sensor_frames=bank.sensors)
    if oracle.identity!=spec['geometry_identity']:
        raise ValueError('Repository/URDF changed since preflight; do not mix geometry versions')
    cfg=SolverConfig(**spec['solver']);cfg.validate()
    # Verify on each selected device; CPU parity is not assumed to prove CUDA parity.
    report=oracle.verify(bank,queries['x_index'],queries['q_query'])
    write_json(out/f'preflight.rank{rank}.json',report)
    shards=out/'shards';shards.mkdir(exist_ok=True)
    completed=0
    for sid in range(rank,m['shard_count'],world):
        if max_shards is not None and completed>=max_shards: break
        row_ids=np.arange(sid*m['shard_size'],min((sid+1)*m['shard_size'],m['query_count']))
        qids=queries['query_id'][row_ids]
        final=shards/f'shard_{sid:06d}'
        with (shards/f'.lock_{sid:06d}').open('a+') as lock:
            try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError(f'Shard {sid} is already being written')
            if final.exists():
                _verify_shard(final,spec_hash,qids)
                print(f'[resume] verified shard={sid}',flush=True);completed+=1;continue
            temp=Path(tempfile.mkdtemp(prefix=f'.shard_{sid:06d}.',dir=shards))
            try:
                arrays,audit=label_rows(bank,queries,row_ids,oracle,cfg)
                write_npz(temp/'labels.npz',arrays)
                with gzip.open(temp/'audit.jsonl.gz','wt',encoding='utf-8') as f:
                    for row in audit: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                hashes={n:sha256_file(temp/n) for n in ('labels.npz','audit.jsonl.gz')}
                write_json(temp/'complete.json',dict(status='COMPLETE',run_spec_sha256=spec_hash,
                    query_ids=[int(x) for x in qids],files_sha256=hashes,rank=rank,device=str(device)))
                os.replace(temp,final)
            except BaseException:
                shutil.rmtree(temp,ignore_errors=True);raise
        completed+=1
        print(f'[shard] COMPLETE {sid} rows={len(qids)}',flush=True)
    return completed


def _distribution(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a): return dict(count=0,median=None,p95=None,max=None)
    return dict(count=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))


def merge(out):
    """Verify complete coverage, then index shards WITHOUT loading all labels in RAM."""
    out,m,bank,queries=open_job(out)
    spec_hash=sha256_file(out/'run_spec.json')
    stats={split:{'rows':0,'supported':np.zeros(8,np.int64),'value':np.zeros(8,np.int64),
                  'grad':np.zeros(8,np.int64),'paired_grad':np.zeros(8,np.int64),
                  'union_value_rows':0,'union_grad_rows':0,'statuses':Counter(),
                  'solver_ms':[],'distance_change':[],'boundary_residuals':[],'old_new_cosines':[]}
           for split in ('train','val')}
    entries=[]
    for sid in range(m['shard_count']):
        row_ids=np.arange(sid*m['shard_size'],min((sid+1)*m['shard_size'],m['query_count']))
        folder=out/'shards'/f'shard_{sid:06d}'
        if not folder.exists(): raise RuntimeError(f'Incomplete labeling: missing {folder.name}; resume workers first')
        meta=_verify_shard(folder,spec_hash,queries['query_id'][row_ids])
        with np.load(folder/'labels.npz',allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        if not np.array_equal(a['query_id'],queries['query_id'][row_ids]) or not np.array_equal(a['q_query'],queries['q_query'][row_ids]):
            raise ValueError('Label/query mismatch')
        for prefix,validkey,gradkey in [('old','old_value_valid','old_grad_regular'),('new','new_value_valid','new_grad_valid')]:
            if not np.isfinite(a[prefix+'_value'][a[validkey]]).all(): raise ValueError('Nonfinite valid value')
            if not np.isfinite(a[prefix+'_grad'][a[gradkey]]).all(): raise ValueError('Nonfinite valid gradient')
        for split_id,name in ((0,'train'),(1,'val')):
            keep=a['split']==split_id;st=stats[name]
            st['rows']+=int(keep.sum())
            for dst,key in [('supported','support'),('value','new_value_valid'),('grad','new_grad_valid'),('paired_grad','paired_grad_mask')]:
                st[dst]+=a[key][keep].sum(0)
            st['union_value_rows']+=int(a['paired_union_value_mask'][keep].sum())
            st['union_grad_rows']+=int(a['paired_union_grad_mask'][keep].sum())
            st['statuses'].update(a['status'][keep].reshape(-1).tolist())
            valid=a['new_value_valid'][keep]
            st['distance_change'].extend((np.abs(a['old_value'][keep])-np.abs(a['new_value'][keep]))[valid].tolist())
            st['boundary_residuals'].extend(a['boundary_residual_m'][keep][valid].tolist())
            grad_keep=a['paired_grad_mask'][keep]
            oldg=a['old_grad'][keep][grad_keep];newg=a['new_grad'][keep][grad_keep]
            if len(oldg):
                cos=np.sum(oldg*newg,axis=-1)/np.maximum(np.linalg.norm(oldg,axis=-1)*np.linalg.norm(newg,axis=-1),1e-8)
                st['old_new_cosines'].extend(cos.tolist())
            st['solver_ms'].extend(a['solver_ms'][keep][a['support'][keep]].tolist())
        entries.append(dict(path=str((folder/'labels.npz').relative_to(out)),sha256=meta['files_sha256']['labels.npz'],rows=len(row_ids)))
    summary=dict(format=FORMAT,status='LABELING_COMPLETE_NOT_A_GLOBAL_DISTANCE_CERTIFICATE',
                 query_count=m['query_count'],run_spec_sha256=spec_hash,
                 original_data_sha256=m['original_data_sha256'],splits={})
    reverse={v:k for k,v in STATUS_CODES.items()}
    for name,st in stats.items():
        support=st['supported']
        summary['splits'][name]=dict(rows=st['rows'],supported_by_sensor=support,
            new_value_valid_by_sensor=st['value'],new_grad_valid_by_sensor=st['grad'],
            paired_grad_by_sensor=st['paired_grad'],
            value_valid_rate_by_sensor=[float(k/n) if n else None for k,n in zip(st['value'],support)],
            gradient_valid_rate_by_sensor=[float(k/n) if n else None for k,n in zip(st['grad'],support)],
            paired_union_value_rows=st['union_value_rows'],paired_union_grad_rows=st['union_grad_rows'],
            status_counts={reverse[int(k)]:v for k,v in st['statuses'].items()},
            solver_ms=_distribution(st['solver_ms']),boundary_residual_m=_distribution(st['boundary_residuals']),
            old_minus_new_distance_rad=_distribution(st['distance_change']),
            old_new_gradient_cosine=_distribution(st['old_new_cosines']))
    summary['limitations']=['Validity means local numerical quality gates, not global closest-boundary proof',
                           'Positive old-minus-new does not prove a more accurate global distance',
                           'Per-sensor failures remain present and masked; missing sensor is not invisible',
                           'No R1 forward pass, training, ROS/Gazebo, LOS, collision or task evaluation has run']
    write_json(out/'dataset_index.json',dict(format=FORMAT,complete=True,query_count=m['query_count'],
                                           run_spec_sha256=spec_hash,shards=entries))
    write_json(out/'label_summary.json',summary)
    lines=['# R1 offline relabel summary','',f"Queries processed: {m['query_count']}",
           '', 'Status: labeling complete; approximate local distances, NOT globally certified labels.',
           '', '| Split | Queries | Supported heads | Valid new values | Valid new gradients | Paired union values |',
           '|---|---:|---:|---:|---:|---:|']
    for name,st in stats.items():
        lines.append(f"| {name} | {st['rows']} | {st['supported'].sum()} | {st['value'].sum()} | {st['grad'].sum()} | {st['union_value_rows']} |")
    lines += ['', 'Do not start training solely because labeling finished. Review coverage, masks,',
              'sensor-specific failures, label residuals, and high-budget spot checks first.',
              '', 'Detailed statistics: label_summary.json. Full attempts: shards/*/audit.jsonl.gz.']
    (out/'label_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[merge] COMPLETE',out/'label_summary.md',flush=True)
    return summary
