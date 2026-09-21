"""Optional higher-budget spot check. Does not modify frozen labels or masks."""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from cache_io import open_job,read_json,write_json
from label_core import SolverConfig,label_continuous
from repo_oracle import RepoOracle


def audit(out,repo,urdf,device='cpu',samples=8,starts=8,maxiter=200,seed=77491):
    out,m,bank,queries=open_job(out)
    if not (out/'dataset_index.json').is_file():
        raise RuntimeError('Merge all shards before the higher-budget audit')
    path=out/f'high_budget_audit_seed{seed}.json'
    if path.exists(): raise FileExistsError(path)
    spec=read_json(out/'run_spec.json');base=SolverConfig(**spec['solver'])
    if starts<base.starts or maxiter<base.maxiter or (starts==base.starts and maxiter==base.maxiter):
        raise ValueError('Use a genuinely larger budget; do not change quality thresholds')
    if samples<1: raise ValueError('samples>=1')
    cfg=replace(base,starts=starts,maxiter=maxiter)
    oracle=RepoOracle(repo,urdf,device,joint_names=bank.joints,sensor_frames=bank.sensors)
    if oracle.identity!=spec['geometry_identity']: raise ValueError('Geometry/source changed')
    oracle.verify(bank,queries['x_index'],queries['q_query'])
    pairs=[]
    # Sample supported cases, including failures; no cherry-picking only valid labels.
    for qi,xi in enumerate(queries['x_index']):
        pairs.extend((qi,s) for s in range(8) if np.any(bank.valid[xi,:,s]))
    rng=np.random.default_rng(seed);choice=rng.choice(len(pairs),min(samples,len(pairs)),replace=False)
    rows=[]
    for k in choice:
        qi,s=pairs[int(k)];xi=int(queries['x_index'][qi]);q=queries['q_query'][qi].astype(float);x=bank.x[xi].astype(float)
        sid,local=divmod(qi,m['shard_size'])
        with np.load(out/'shards'/f'shard_{sid:06d}'/'labels.npz',allow_pickle=False) as z:
            cached=dict(value=float(z['new_value'][local,s]),value_valid=bool(z['new_value_valid'][local,s]),
                        grad_valid=bool(z['new_grad_valid'][local,s]),grad=z['new_grad'][local,s].copy(),
                        status=int(z['status'][local,s]))
        higher=label_continuous(q,bank.sensor_bank(xi,s),bank.masks[s],bank.lo,bank.hi,oracle.geometry(x,s),cfg)
        ref=np.nan
        if np.isfinite(higher['q_star']).all():
            ref=float(oracle.reference_margins(x,higher['q_star'][None])[0,s])
        query_ref=float(oracle.reference_margins(x,q[None])[0,s])
        sign_ok=(query_ref>=0)==(higher['query_g_m']>=0)
        sign_guard_ok=abs(query_ref)>cfg.sign_guard_m or (higher['value']==0 and abs(query_ref)<=cfg.boundary_tol_m)
        higher_valid=bool(higher['value_valid'] and np.isfinite(ref) and abs(ref)<=cfg.boundary_tol_m and sign_ok and sign_guard_ok)
        delta=(abs(cached['value'])-abs(higher['value'])) if np.isfinite(cached['value']) and higher_valid else np.nan
        materially_closer=bool(cached['value_valid'] and higher_valid and delta>cfg.tie_abs_rad+cfg.tie_rel*abs(cached['value']))
        row=dict(query_id=int(queries['query_id'][qi]),sensor=s,cached=cached,higher=higher,
                 higher_reference_g_m=ref,higher_value_valid_after_reference_check=higher_valid,
                 cached_abs_minus_higher_abs_rad=delta,materially_closer_found=materially_closer)
        rows.append(row)
        print(f'[audit] query={qi} sensor={s} materially_closer={materially_closer}',flush=True)
    report=dict(status='HIGHER_BUDGET_AUDIT_COMPLETE_NOT_GLOBAL_CERTIFICATION',samples=len(rows),seed=seed,
                base_solver=asdict(base),higher_solver=asdict(cfg),
                materially_closer_count=sum(r['materially_closer_found'] for r in rows),rows=rows,
                note='Results never overwrite original labels; repeated convergence is not a global-optimum proof')
    write_json(path,report);print('[audit]',path,flush=True)
    return report
