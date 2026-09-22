#!/usr/bin/env python3
"""Targeted v2 verification of an existing immutable v1 smoke. No training cache."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from verified_core import (V1, SolverConfig, VerifyConfig, screen_candidates,
                           solve_attempts, verify_gradient)
from cache_io import BankCache, read_json, write_json, clean_json
from repo_oracle import RepoOracle, sha256_file

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = '4:0,1:7,5:6,3:2,2:4,2:3'


def source_path(root, name):
    p = (root / name).resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise ValueError(f'Unsafe/missing source member: {name}')
    return p


def read_source(root):
    """Read only. Validate saved query identities and every shard checksum."""
    root = Path(root).resolve()
    manifest = read_json(root/'manifest.json'); spec = read_json(root/'run_spec.json')
    index = read_json(root/'dataset_index.json')
    if not index.get('complete') or read_json(root/'preflight.json').get('status') != 'PASS':
        raise ValueError('Need a completed, preflight-passing v1 source')
    for name, expected in [('queries.npz', manifest['queries_sha256']),
                           ('spatial_splits.npz', manifest['spatial_splits_sha256']),
                           ('run_spec.json', index['run_spec_sha256'])]:
        if sha256_file(source_path(root, name)) != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
    if spec['queries_sha256'] != manifest['queries_sha256']:
        raise ValueError('Source run/query identity mismatch')
    for name, expected in manifest['code_sha256'].items():
        if sha256_file(V1/name) != expected:
            raise ValueError(f'v1 source code differs: {name}; do not overwrite old experiment')
    with np.load(root/'queries.npz', allow_pickle=False) as z:
        queries = {k:z[k].copy() for k in z.files}
    if len(queries['query_id']) != manifest['query_count'] or len(set(queries['query_id'].tolist())) != manifest['query_count']:
        raise ValueError('Source query IDs are not complete/unique')
    by_id = {int(k):i for i,k in enumerate(queries['query_id'])}
    records = {}; hashes = {n:sha256_file(root/n) for n in
        ('manifest.json','run_spec.json','dataset_index.json','queries.npz','spatial_splits.npz','preflight.json')}
    for entry in index['shards']:
        labels = source_path(root, entry['path']); folder = labels.parent
        meta = read_json(folder/'complete.json')
        hashes[str((folder/'complete.json').relative_to(root))] = sha256_file(folder/'complete.json')
        if meta['run_spec_sha256'] != index['run_spec_sha256']:
            raise ValueError('Mixed source run specs')
        for name, expected in meta['files_sha256'].items():
            p = source_path(root, str((folder/name).relative_to(root)))
            actual = sha256_file(p)
            if actual != expected: raise ValueError(f'Corrupt source shard: {p}')
            hashes[str(p.relative_to(root))] = actual
        if sha256_file(labels) != entry['sha256']: raise ValueError('Source index mismatch')
        with np.load(labels, allow_pickle=False) as z:
            ids = z['query_id'].copy(); q = z['q_query'].copy(); xi = z['x_index'].copy()
        if ids.tolist() != meta['query_ids']: raise ValueError('Shard IDs mismatch')
        with gzip.open(folder/'audit.jsonl.gz','rt',encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if [r['query_id'] for r in rows] != ids.tolist(): raise ValueError('Audit order mismatch')
        for j,r in enumerate(rows):
            k = int(r['query_id']); i = by_id[k]
            if k in records or not np.array_equal(q[j], queries['q_query'][i]) or int(xi[j]) != int(queries['x_index'][i]):
                raise ValueError('Duplicate or mismatched source query')
            if not np.array_equal(np.asarray(r['q_query'],np.float32), q[j]) or r['x_index'] != int(xi[j]):
                raise ValueError('Audit query mismatch')
            records[k] = r
    if set(records) != set(by_id): raise ValueError('Missing source queries')
    return manifest, spec, queries, records, hashes


def parse_cases(text):
    cases = [tuple(map(int,s.split(':'))) for s in text.split(',')]
    if not cases or len(set(cases)) != len(cases) or any(len(c)!=2 or c[0]<0 or not 0<=c[1]<8 for c in cases):
        raise ValueError('Use unique query_id:sensor_id pairs, e.g. 1:7,4:0')
    return cases


def prepare(source, out, cases=DEFAULT_CASES, resume=False):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out==source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Output must be separate from source, not its ancestor/descendant')
    m,s,qs,records,hashes = read_source(source)
    if out==Path(m['bank_cache']).resolve() or out.is_relative_to(Path(m['bank_cache']).resolve()):
        raise ValueError('Do not write into original bank cache')
    chosen = parse_cases(cases)
    for q,sensor in chosen:
        if q not in records or 'new' not in records[q]['sensors'].get(str(sensor),{}):
            raise ValueError(f'No supported saved case {q}:{sensor}')
    from pipeline import library_versions
    config = dict(library_versions=library_versions(), format='careplanner_offline_relabel_v2_targeted_verification',
        source=str(source), source_hashes=hashes, cases=chosen,
        source_queries_sha256=m['queries_sha256'], solver=s['solver'], verify=asdict(VerifyConfig()),
        v2_code_sha256={p.name:sha256_file(p) for p in HERE.glob('*.py')},
        training_ready=False, global_nearest_certified=False)
    config = clean_json(config)
    if out.exists():
        if not resume or read_json(out/'verification_spec.json') != config:
            raise ValueError('Output exists or identity changed; no overwrite/mixed resume')
    else:
        out.mkdir(parents=True)
        write_json(out/'verification_spec.json', config)
    print('[prepared] fixed cases='+cases+' source='+str(source),flush=True)
    return config


def reference_gate(oracle, x, sensor, q, result, cfg):
    """FP32 original-oracle check, distinct from double-precision optimization."""
    if result['selected'] is None: return result
    gm = float(oracle.reference_margins(x, np.asarray(result['q_star'])[None])[0,sensor])
    gq = float(oracle.reference_margins(x, np.asarray(q)[None])[0,sensor])
    ok = (abs(gm) <= cfg.boundary_tol_m and
          (gq>=0)==(result['query_g_m']>=0) and
          (abs(gq)>cfg.sign_guard_m or result['value']==0))
    result['reference_check'] = dict(boundary_g_m=gm, query_g_m=gq, passed=bool(ok))
    if not ok:
        result['value_valid']=False; result['grad_valid']=False
        result['gradient_reasons'].append('UPSTREAM_GEOMETRY_OR_SIGN_MISMATCH')
    return result


def worker(source, out, repo, urdf, device, rank, world):
    from dataclasses import replace
    import fcntl
    if world<1 or not 0<=rank<world: raise ValueError('Bad rank/world')
    out=Path(out).resolve(); config=read_json(out/'verification_spec.json')
    source=Path(source).resolve()
    if str(source)!=config['source']: raise ValueError('Wrong source')
    # prepare/resume is read-only on an existing matching output.
    prepare(source,out,','.join(f'{q}:{s}' for q,s in config['cases']),resume=True)
    m,old_spec,qs,records,hashes=read_source(source)
    bank_path=Path(m['bank_cache'])
    if sha256_file(bank_path/'bank_manifest.json') != m['bank_manifest_sha256']:
        raise ValueError('Original bank manifest mismatch')
    bank=BankCache(bank_path)
    if bank.manifest['source_sha256'] != m['original_data_sha256']:
        raise ValueError('Different original NPZ identity')
    oracle=RepoOracle(repo,urdf,device,joint_names=bank.joints,sensor_frames=bank.sensors)
    for key in ('urdf_sha256','source_sha256'):
        if oracle.identity[key] != old_spec['geometry_identity'][key]:
            raise ValueError(f'Original geometry changed: {key}')
    # Current repo HEAD may change; critical geometry and original data may NOT.
    preflight=oracle.verify(bank,qs['x_index'],qs['q_query'])
    write_json(out/f'preflight.rank{rank}.json',preflight)
    cfg=SolverConfig(**config['solver']); vc=VerifyConfig(**config['verify'])
    probe_cfg=replace(cfg,starts=vc.probe_starts,maxiter=vc.probe_maxiter)
    digest=sha256_file(out/'verification_spec.json')
    for index,(qid,sensor) in enumerate(config['cases']):
        if index%world != rank: continue
        folder=out/f'case_{qid:06d}_S{sensor}';folder.mkdir(exist_ok=True)
        with (folder/'.lock').open('a+') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (folder/'complete.json').exists():
                c=read_json(folder/'complete.json')
                if c['spec_sha256']!=digest or sha256_file(folder/'result.json')!=c['result_sha256']:
                    raise ValueError('Case resume mismatch')
                print(f'[resume] verified case={qid}:{sensor}',flush=True);continue
            t0=time.perf_counter(); r=records[qid]; old=r['sensors'][str(sensor)]['new']
            q=np.asarray(r['q_query'],float);xi=int(r['x_index']);x=np.asarray(bank.x[xi],float)
            if not np.array_equal(x.astype(np.float32),np.asarray(r['x'],np.float32)):
                raise ValueError('Source x differs from original bank')
            mask=bank.masks[sensor];geo=oracle.geometry(x,sensor)
            print(f'[case] start {qid}:{sensor} reuse {len(old["attempts"])} central attempts',flush=True)
            base=screen_candidates(q,mask,bank.lo,bank.hi,geo,old['attempts'],cfg,vc)
            base=reference_gate(oracle,x,sensor,q,base,cfg)
            warm=[c['q_star'] for c in base['clusters'][:2] if c['qualified']]
            points=bank.sensor_bank(xi,sensor)
            def probe(trial):
                a=solve_attempts(trial,points,mask,bank.lo,bank.hi,geo,probe_cfg,warm)
                rr=screen_candidates(trial,mask,bank.lo,bank.hi,geo,a,cfg,vc)
                rr=reference_gate(oracle,x,sensor,trial,rr,cfg);rr['attempts']=a
                return rr
            def progress(msg): print(f'[probe] case={qid}:{sensor} {msg}',flush=True)
            new=verify_gradient(base,q,mask,bank.lo,bank.hi,probe,cfg,vc,progress)
            result=dict(query_id=qid,sensor_id=sensor,x_index=xi,split=r['split'],q_query=q,
                old=dict(value=old['value'],value_valid=old['value_valid'],grad_valid=old['grad_valid']),
                new=new,elapsed_ms=1000*(time.perf_counter()-t0),
                central_attempts_reused=len(old['attempts']),new_central_optimizations=0,
                spec_sha256=digest,device=device,training_ready=False)
            write_json(folder/'result.json',result)
            write_json(folder/'complete.json',dict(spec_sha256=digest,
                result_sha256=sha256_file(folder/'result.json')))
            print(f'[case] COMPLETE {qid}:{sensor} value={new["value_valid"]} grad={new["grad_valid"]} reasons={new["gradient_reasons"]}',flush=True)
    # Ensure immutable source identity still matches after this worker.
    for name,sha in hashes.items():
        if sha256_file(source/name)!=sha: raise ValueError('Source was modified during verification')


def merge(out):
    out=Path(out).resolve();spec=read_json(out/'verification_spec.json');rows=[]
    digest=sha256_file(out/'verification_spec.json')
    for q,s in spec['cases']:
        folder=out/f'case_{q:06d}_S{s}'
        c=read_json(folder/'complete.json')
        if c['spec_sha256']!=digest or c['result_sha256']!=sha256_file(folder/'result.json'):
            raise ValueError('Case incomplete or corrupt')
        rows.append(read_json(folder/'result.json'))
    source=Path(spec['source'])
    for name,sha in spec['source_hashes'].items():
        if sha256_file(source/name)!=sha: raise ValueError('Source changed; no final report')
    short=[dict(query_id=r['query_id'],sensor_id=r['sensor_id'],old=r['old'],
                value_valid=r['new']['value_valid'],grad_valid=r['new']['grad_valid'],
                reasons=r['new']['gradient_reasons'],verification=r['new']['verification']['status'],
                selected_attempt_id=(r['new']['selected'] or {}).get('attempt_id'),elapsed_ms=r['elapsed_ms']) for r in rows]
    summary=dict(status='TARGETED_VERIFICATION_COMPLETE_REVIEW_REQUIRED',cases=short,
        source_unchanged=True,training_ready=False,global_nearest_certified=False,
        limitations=['Fixed diagnostic cases, not population accuracy or training benefit',
                     'Local multi-start numerical derivative checks do not prove global nearest',
                     'No neural inference, training, ROS, LOS or execution changed'])
    write_json(out/'verification_summary.json',summary)
    text=['# R1 relabel v2 targeted verification','',
          'Source unchanged. This is NOT a training dataset or global-distance certificate.','',
          '| Query:sensor | Old value/grad | New value/grad | Check | Reasons |',
          '|---|---|---|---|---|']
    for r in short:
        text.append(f"| {r['query_id']}:S{r['sensor_id']} | {r['old']['value_valid']}/{r['old']['grad_valid']} | {r['value_valid']}/{r['grad_valid']} | {r['verification']} | {', '.join(r['reasons']) or 'none'} |")
    (out/'verification_summary.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=('prepare','worker','merge'))
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=HERE.parents[1]);p.add_argument('--urdf',type=Path)
    p.add_argument('--cases',default=DEFAULT_CASES);p.add_argument('--resume',action='store_true')
    p.add_argument('--device',default='cpu');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    a=p.parse_args()
    if a.stage=='merge': merge(a.out)
    elif a.source is None: p.error('--source required')
    elif a.stage=='prepare': prepare(a.source,a.out,a.cases,a.resume)
    else: worker(a.source,a.out,a.repo,a.urdf or a.repo/'src/arm_description/urdf/Arm.urdf',a.device,a.rank,a.world_size)

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc();sys.exit(2)
