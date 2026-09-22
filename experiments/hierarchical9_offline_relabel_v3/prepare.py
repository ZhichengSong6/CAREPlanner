"""Finite, immutable queries; audit selection is fixed BEFORE solving any labels."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import shutil
import tempfile
import numpy as np
from common import (FORMAT, RepoOracle, core, old_io, source_bank, check_output_location,
                    geometry_identity, code_identity, versions, sha256_file, load_npz)


@dataclass(frozen=True)
class Sampling:
    train_x: int = 64
    val_x: int = 16
    uniform_per_x: int = 2
    normal_pairs_per_x: int = 1
    normal_step_rad: float = .02
    coverage_per_sensor: int = 2
    seed: int = 260923
    audit_per_stratum: int = 1
    audit_seed: int = 882616

    def validate(self):
        integer_fields=('train_x','val_x','uniform_per_x','normal_pairs_per_x','coverage_per_sensor','seed','audit_per_stratum','audit_seed')
        if any(type(getattr(self,k)) is not int for k in integer_fields):
            raise ValueError('Sampling counts and seeds must be integers')
        counts=(self.train_x,self.val_x,self.uniform_per_x,self.coverage_per_sensor)
        if min(counts)<1 or self.normal_pairs_per_x<0 or self.audit_per_stratum<0:
            raise ValueError('Positive finite sampling counts required; no implicit full-dataset mode')
        if not np.isfinite(self.normal_step_rad) or self.normal_step_rad<=0:
            raise ValueError('Bad normal displacement')
        if min(self.seed,self.audit_seed)<0:
            raise ValueError('Seeds must be nonnegative')
        if self.coverage_per_sensor>min(self.train_x,self.val_x):
            raise ValueError('Coverage quota exceeds selected point count')


def choose_points(pool, support, count, quota, rng):
    """A declared support-balanced design, NOT a uniform workspace benchmark."""
    pool = rng.permutation(np.asarray(pool, np.int64))
    if count>len(pool): raise ValueError('Requested more x than the available original split')
    available = support[pool].sum(0)
    if np.any(available<quota):
        raise ValueError(f'Insufficient original sensor support for quota: {available.tolist()}')
    selected=[]; covered=np.zeros(8, int); used=np.zeros(len(pool), bool)
    while np.any(covered<quota):
        scores=(support[pool]*np.maximum(quota-covered,0)).sum(1)
        scores[used]=-1; k=int(scores.argmax())
        if scores[k]<=0 or len(selected)>=count:
            raise ValueError('Requested x budget cannot meet support quotas; increase x budget')
        selected.append(int(pool[k])); used[k]=True; covered+=support[pool[k]]
    selected.extend(pool[~used][:count-len(selected)].tolist())
    return np.asarray(selected,np.int64)


def audit_plan(query, per_stratum, seed, guard):
    """Strata: original split x sensor x actual query FOV sign. Never use solver success."""
    rng=np.random.default_rng(seed); tasks=[]; strata=[]
    for split in (0,1):
        for s in range(8):
            for sign in (-1,1):
                g=query['reference_g_m'][:,s]
                eligible=np.flatnonzero((query['split']==split)&query['support'][:,s]&
                                         ((g < -guard) if sign<0 else (g > guard)))
                chosen=rng.permutation(eligible)[:per_stratum]
                for i in chosen: tasks.append([int(query['query_id'][i]),s])
                strata.append(dict(split=split,sensor_id=s,sign=sign,eligible=len(eligible),
                                   selected=len(chosen),query_ids=query['query_id'][chosen].tolist()))
    return dict(tasks=tasks,strata=strata,per_stratum=per_stratum,seed=seed,
                policy='selected_before_label_solving_no_success_conditioning',
                near_zero_not_audited=int(((np.abs(query['reference_g_m'])<=guard)&query['support']).sum()),
                no_replacement_for_failures=True)


def prepare(source,out,repo,urdf,sampling=None,resume=False,oracle_factory=RepoOracle):
    sampling=sampling or Sampling();sampling.validate()
    source,m,previous,bank,oldq=source_bank(source)
    out=Path(out).resolve();check_output_location(out,source,bank.root)
    requested=dict(sampling=asdict(sampling),source=str(source),repo=str(Path(repo).resolve()),
                   urdf=str(Path(urdf).resolve()))
    if out.exists():
        if not resume: raise FileExistsError(f'Use a new BATCH_OUT, or explicit RESUME=true: {out}')
        from common import open_run
        _,spec,_,_,_=open_run(out)
        if spec['request']!=requested: raise ValueError('Resume options differ')
        print('[resume] frozen query plan verified; no resampling',flush=True);return spec
    oracle=oracle_factory(repo,urdf,'cpu',joint_names=bank.joints,sensor_frames=bank.sensors)
    if geometry_identity(oracle.identity)!=geometry_identity(previous['geometry_identity']):
        raise ValueError('Geometry differs from the accepted source smoke')
    tr,va=bank.original_split(m['sampling']['val_count'],m['sampling']['split_seed'])
    # Preserve the complete original split; exclude diagnostic x only from this pilot selection.
    excluded=np.unique(oldq['x_index']);support=np.any(bank.valid,axis=1)
    rng=np.random.default_rng(sampling.seed); selected=[]
    for split,pool,n in ((0,tr,sampling.train_x),(1,va,sampling.val_x)):
        pool=pool[~np.isin(pool,excluded)]
        selected.append(choose_points(pool,support,n,sampling.coverage_per_sensor,rng))
    preflight=oracle.verify(bank,np.concatenate(selected),np.zeros((1,7)))
    rows=[];skips=[];source_sensor_counts=np.zeros(8,int)
    def add(xi,split,q,group,s=-1,k=-1,side=0):
        q=np.asarray(q,np.float32)
        if not np.isfinite(q).all() or np.any(q<bank.lo) or np.any(q>bank.hi):
            skips.append(dict(x_index=int(xi),split=split,group=group,sensor=s,side=side,reason='QUERY_OUT_OF_LIMITS_NO_CLAMP'));return
        # No success-conditioned filtering: keep actual inside/outside signs as observed.
        g=oracle.reference_margins(np.asarray(bank.x[xi]),q[None])[0]
        if not np.isfinite(g).all(): raise ValueError('Nonfinite query FOV margin')
        rows.append(dict(x_index=int(xi),split=split,q_query=q,query_group=group,
                         source_sensor=s,source_bank_slot=k,normal_side=side,
                         support=support[xi],reference_g_m=g))
    for split,ids in enumerate(selected):
        for xi in ids:
            for _ in range(sampling.uniform_per_x):
                add(xi,split,rng.uniform(bank.lo,bank.hi),0)
            for _ in range(sampling.normal_pairs_per_x):
                sensors=np.flatnonzero(support[xi]); minimum=source_sensor_counts[sensors].min()
                s=int(rng.choice(sensors[source_sensor_counts[sensors]==minimum]));source_sensor_counts[s]+=1
                slots=np.flatnonzero(bank.valid[xi,:,s]);k=int(rng.choice(slots))
                anchor=np.asarray(bank.q[xi,k,:,s],float)
                if not np.isfinite(anchor).all() or np.any(anchor<bank.lo) or np.any(anchor>bank.hi):
                    skips.append(dict(x_index=int(xi),split=split,sensor=s,reason='INVALID_BANK_ANCHOR'));continue
                h,j=oracle.geometry(np.asarray(bank.x[xi]),s)(anchor)
                n=np.asarray(j[int(np.argmin(h))])*bank.masks[s];norm=np.linalg.norm(n)
                if not np.isfinite(norm) or norm<=1e-8:
                    skips.append(dict(x_index=int(xi),split=split,sensor=s,reason='DEGENERATE_ANCHOR_NORMAL'));continue
                for side in (-1,1): add(xi,split,anchor+side*sampling.normal_step_rad*n/norm,1,s,k,side)
            print(f'[prepare] split={split} x_index={xi} queries={len(rows)}',flush=True)
    if not rows: raise ValueError('No queries generated')
    query={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    query['query_id']=np.arange(len(rows),dtype=np.int64)
    for k in ('x_index','query_id'):query[k]=query[k].astype(np.int64)
    for k in ('split','query_group'):query[k]=query[k].astype(np.uint8)
    query['q_query']=query['q_query'].astype(np.float32)
    query['support']=query['support'].astype(bool)
    query['reference_g_m']=query['reference_g_m'].astype(np.float64)
    cfg=core.SolverConfig(**previous['solver']);vc=core.VerifyConfig()
    plan=audit_plan(query,sampling.audit_per_stratum,sampling.audit_seed,cfg.sign_guard_m)
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(tempfile.mkdtemp(prefix='.'+out.name+'.prepare.',dir=out.parent))
    try:
        old_io.write_npz(tmp/'queries.npz',query)
        old_io.write_npz(tmp/'spatial_splits.npz',dict(train_x_indices=tr,val_x_indices=va))
        old_io.write_json(tmp/'audit_plan.json',plan)
        old_io.write_json(tmp/'sampling_report.json',dict(request=asdict(sampling),
            selected_train_x=selected[0],selected_val_x=selected[1],excluded_previous_diagnostic_x=excluded,
            query_count=len(rows),skips=skips,selected_support_by_split=[support[z].sum(0) for z in selected],
            query_support_by_split=[query['support'][query['split']==i].sum(0) for i in (0,1)],
            source_sensor_counts=source_sensor_counts,
            query_groups={'0':'uniform_joint_box','1':'bank_normal_pair_NOT_exact_boundary_witness'},
            actual_sign_counts={str(i):{'inside':((query['reference_g_m'][query['split']==i]>cfg.sign_guard_m)&query['support'][query['split']==i]).sum(0),
                'outside':((query['reference_g_m'][query['split']==i]<-cfg.sign_guard_m)&query['support'][query['split']==i]).sum(0)} for i in (0,1)},
            limitations=['Support-balanced point sampling, not population-uniform evaluation',
                         'No outcome-conditioned replacement; raw bank anchors are approximate']))
        old_io.write_json(tmp/'preflight.prepare.json',preflight)
        spec=dict(format=FORMAT,request=requested,source=str(source),
            source_manifest_sha256=sha256_file(source/'manifest.json'),
            source_run_spec_sha256=sha256_file(source/'run_spec.json'),
            bank_cache=str(bank.root),original_data_sha256=m['original_data_sha256'],
            geometry=geometry_identity(oracle.identity),versions=versions(),code_sha256=code_identity(),
            solver=asdict(cfg),verify=asdict(vc),query_count=len(rows),
            supported_tasks=int(query['support'].sum()),audit_tasks=len(plan['tasks']),
            frozen_files_sha256={n:sha256_file(tmp/n) for n in ('queries.npz','spatial_splits.npz','audit_plan.json','sampling_report.json')},
            scope='finite_pilot_not_all_x_not_all_q',training_ready=False,global_nearest_certified=False)
        old_io.write_json(tmp/'batch_spec.json',spec)
        os.rename(tmp,out)
    finally:
        if tmp.exists():shutil.rmtree(tmp)
    print(f'[prepared] queries={len(rows)} supported_tasks={spec["supported_tasks"]} audit_plan={spec["audit_tasks"]} out={out}',flush=True)
    return spec
