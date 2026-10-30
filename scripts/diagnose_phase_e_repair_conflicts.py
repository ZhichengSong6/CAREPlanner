#!/usr/bin/env python3
"""Offline-only forensics of stored hard REPAIR QPs; never publishes a plan.

Original PIQP replay is unchanged. LP subset checks are necessary-condition
diagnostics, not executable relaxations. FK uses the production shared library.
All learned evaluations use the frozen COLLISION CDF, not H9 visibility heads.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
import yaml

from analyze_phase_e_observation_events import string_rows
from evaluate_phase_d_run import joint_states, JOINTS
from summarize_phase_e_fix34_regression import bound_conflicts

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BATCH = REPO/'outputs/phase_e_target_lineage/20260911-081348-lineage'
COLLISION = REPO/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sparse(data):
    entries = np.asarray(data['coo'], dtype=float).reshape(-1, 3)
    return coo_matrix((entries[:, 2], (entries[:, 0].astype(int), entries[:, 1].astype(int))),
                      shape=tuple(data['shape'])).tocsr()


def vectors(data):
    return {k: np.asarray(data[k], dtype=float) for k in ('b', 'c', 'h_l', 'h_u', 'x_l', 'x_u')}


def lp_check(data, rows=None):
    """Same A/b/variable bounds, optional named subset of original G rows.

    No changed bound values. Returning feasible for a SUBSET is NOT a safety
    certificate. The full check keeps every original inequality.
    """
    v = vectors(data); A = sparse(data['A']); G = sparse(data['G'])
    ids = np.arange(G.shape[0]) if rows is None else np.asarray(rows, dtype=int)
    lower = v['h_l'][ids]; upper = v['h_u'][ids]
    # Match PIQP_INF exactly, not a new numerical/safety tolerance.
    lo = np.isfinite(lower) & (lower > -1e30)
    hi = np.isfinite(upper) & (upper < 1e30)
    ub = vstack((-G[ids[lo]], G[ids[hi]]), format='csr')
    rhs = np.concatenate((-lower[lo], upper[hi]))
    result = linprog(np.zeros(A.shape[1]), A_ub=ub if len(rhs) else None,
                     b_ub=rhs if len(rhs) else None, A_eq=A, b_eq=v['b'],
                     bounds=list(zip(v['x_l'], v['x_u'])), method='highs')
    residual = None
    if result.success:
        residual = max(float(np.max(np.abs(A @ result.x-v['b']))),
                       float(np.max(v['x_l']-result.x)), float(np.max(result.x-v['x_u'])),
                       float(np.max(ub @ result.x-rhs)) if len(rhs) else 0.)
    return dict(status=int(result.status), feasible=bool(result.success),
                max_constraint_violation=residual, message=result.message)


def reconstruct(data, cfg):
    """Recover q_bar only after proving no physical-limit clipping occurred."""
    nsteps = cfg['local_planner']['num_intervals']; dof = 7; nq = nsteps*dof
    v = vectors(data)
    if data['P']['shape'] != [2*nq, 2*nq] or len(v['b']) != nq:
        raise ValueError('unsupported dimensions or user slack variables')
    lo, hi = v['x_l'][:nq].reshape(nsteps,dof), v['x_u'][:nq].reshape(nsteps,dof)
    limits = cfg['mpc']['joint_position_limits']; margin = cfg['mpc']['joint_position_margin']
    jointlo = np.array([limits[n]['lower']+margin for n in JOINTS])
    jointhi = np.array([limits[n]['upper']-margin for n in JOINTS])
    if np.any(lo <= jointlo+1e-10) or np.any(hi >= jointhi-1e-10):
        raise ValueError('joint-clipped box: midpoint cannot identify q_bar')
    radii = (hi-lo)/2
    if np.ptp(radii) > 1e-10:
        raise ValueError('nonuniform box: cannot identify original trust radius')
    center = (hi+lo)/2; q0 = v['b'][:7]
    dt = cfg['local_planner']['horizon_duration']/nsteps
    expected = np.zeros((nq, 2*nq))
    for k in range(nsteps):
        for j in range(dof):
            r=k*dof+j; expected[r,r]=1; expected[r,nq+r]=-dt
            if k: expected[r,r-dof]=-1
    np.testing.assert_allclose(sparse(data['A']).toarray(), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(v['b'][7:], 0, rtol=0, atol=0)
    qlin = np.asarray(data['q_linearization_flat']).reshape(-1,7)
    k = np.asarray(data['original_timestep'],dtype=int)
    qfull = np.vstack((q0,center))
    match = float(np.max(np.abs(qfull[k]-qlin)))
    if match > cfg['local_planner']['cdf']['linearization_tolerance_inf']:
        raise ValueError('snapshot linearization/box mismatch')
    # For these hard-QP snapshots G contains CDF rows only, as configured.
    if cfg['local_planner']['enforce_acceleration_constraints'] or len(data['selected_pairs']) != data['G']['shape'][0]:
        raise ValueError('unexpected acceleration rows')
    grad = np.asarray(data['gradient_flat']).reshape(-1,7)
    for row,pair in data['selected_pairs']:
        assembled = sparse(data['G']).getrow(row).toarray().reshape(-1)
        expected_row = np.zeros(2*nq); expected_row[(k[pair]-1)*7:k[pair]*7]=grad[pair]
        np.testing.assert_allclose(assembled, expected_row, rtol=0, atol=1e-14)
        expected_lower = cfg['local_planner']['cdf']['safety_margin']-data['distance'][pair]+grad[pair]@qlin[pair]
        np.testing.assert_allclose(v['h_l'][row],expected_lower,rtol=0,atol=1e-12)
    return q0, center, float(radii[0,0]), match


def grid_point(point, config):
    lower = np.array([config[a+'_min'] for a in 'xyz'])
    step = config['resolution']; point=np.asarray(point,dtype=float)
    index=np.rint((point-lower)/step).astype(int)
    exact=lower+index*step
    if not np.array_equal(exact.astype(np.float32),point.astype(np.float32)):
        raise ValueError('point is not an exact FP32 grid voxel')
    return exact


def body_query(binary, queries, semantics, mapcfg):
    payload=dict(urdf=str(REPO/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
        body_samples=str(REPO/'src/care_confidence_map/config/body_samples.yaml'),
        # Proven from the frozen runtime config/log; not changed for this test.
        ignored_links=['base_link','link1'], body_inflation_m=float(semantics['gcdf_body_inflation_m']),
        map_lower=[mapcfg[a+'_min'] for a in 'xyz'],map_upper=[mapcfg[a+'_max'] for a in 'xyz'],queries=queries)
    result=subprocess.run([binary],input=json.dumps(payload),capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch',type=Path,default=DEFAULT_BATCH)
    ap.add_argument('--summary',type=Path,help='Explicit case summary for batches recovered by offline aggregation')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--body-binary',default='/tmp/care_repair_body_clearance')
    ap.add_argument('--replay-binary',default='/tmp/care_replay_qp')
    args=ap.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((args.batch/'frozen_sha256.json').read_text())
    # Reject changed runtime/model/geometry/compiled dependencies, but allow
    # new OFFLINE diagnosis scripts in this turn.
    for path,expected in frozen.items():
        if path.startswith(('src/','devel/')) and sha(REPO/path)!=expected:
            raise ValueError('runtime input changed: '+path)
    cfg=yaml.safe_load((REPO/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
    mapcfg=yaml.safe_load((REPO/'src/care_confidence_map/config/confidence_map_phase_e_ray.yaml').read_text())['confidence_map']
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(REPO/'src/care_collision_cdf/scripts'))
    from collision_cdf_model import CollisionCDF
    model=CollisionCDF(str(COLLISION),device='cpu',activation='gelu')
    def evaluate(p,q):
        d,g=model.pair_distance_and_gradient(torch.tensor(np.asarray(p)),torch.tensor(np.asarray(q)))
        return d.numpy().astype(float),g.numpy().astype(float)
    archived=json.loads((args.batch/'repair_audit/report.json').read_text())
    summary_path=args.summary or args.batch/'summary.json'
    runs=json.loads(summary_path.read_text())['cases']; results=[]
    evidence_paths={args.batch/'repair_audit/report.json',summary_path,args.batch/'frozen_sha256.json'}
    for case in archived['cases']:
        if not case['snapshots']: continue
        runinfo=next(c for c in runs if (c['case_id'],c['model'])==(case['case_id'],case['model']))
        run=Path(runinfo['run']); events=string_rows(run/'local_planner_summary.csv',runinfo['benchmark_window'])
        evidence_paths.update(run/name for name in ('local_planner_summary.csv','joint_states.csv','runtime_semantics.txt'))
        measured=joint_states(str(run/'joint_states.csv'),JOINTS,timestamps_ns=True)
        semantics=dict(line.split('=',1) for line in (run/'runtime_semantics.txt').read_text().splitlines() if '=' in line)
        for snap in case['snapshots']:
            path=Path(snap['path'])
            evidence_paths.add(path)
            if sha(path)!=snap['sha256']: raise ValueError('snapshot digest mismatch')
            data=json.loads(path.read_text()); q0,center,radius,qlin_error=reconstruct(data,cfg)
            p=np.asarray(data['point_flat']).reshape(-1,3); qlin=np.asarray(data['q_linearization_flat']).reshape(-1,7)
            d,g=evaluate(p,qlin)
            valerr=float(np.max(np.abs(d-data['distance'])))
            graderr=float(np.max(np.abs(g-np.asarray(data['gradient_flat']).reshape(-1,7))))
            if max(valerr,graderr)>1e-4: raise ValueError('frozen collision CDF reproduction failed')
            replay=subprocess.run([args.replay_binary,str(path)],capture_output=True,text=True,check=True)
            plan_events=[e for e in events if e.get('plan_seq')==str(data['plan_sequence']) and e['event'] in ('plan_started','scp_solved','scp_failed','repair_qp_exhausted','repair_qp_retry_scheduled')]
            failures=[e for e in plan_events if e['event']=='scp_failed']
            if len(failures)!=1 or failures[0]['observation_token']!=data['observation_token']:
                raise ValueError('cannot join unique plan failure')
            if abs(float(failures[0]['trust_q_inf'])-radius)>1e-10:
                raise ValueError('trace/box trust mismatch')
            conflicts=bound_conflicts(data); details=[]
            for conflict in conflicts:
                pair=conflict['pair']; k=data['original_timestep'][pair]; qbar=center[k-1]
                gp=grid_point(p[pair],mapcfg)
                geom=body_query(args.body_binary,[dict(point=gp.tolist(),q=q.tolist()) for q in (q0,qbar)],semantics,mapcfg)
                if abs(geom[1]['clearance_m']-data['approx_body_clearance_m'][pair])>1e-7:
                    raise ValueError('production geometric clearance reproduction failed')
                d0,g0=evaluate(p[pair:pair+1],q0.reshape(1,7))
                delta=qbar-q0; linear_from_q0=float(d0[0]+g0[0]@delta)
                # Exhaustive corner samples are observations, NOT a nonlinear
                # global maximization proof or an executable candidate search.
                corners=qbar+radius*np.asarray(list(itertools.product((-1.,1.),repeat=7)))
                corner_d,_=evaluate(np.tile(p[pair],(len(corners),1)),corners)
                saved_g=np.asarray(data['gradient_flat'][pair*7:(pair+1)*7])
                saved_d=data['distance'][pair]
                needed=(cfg['local_planner']['cdf']['safety_margin']-saved_d)/np.linalg.norm(saved_g,1)
                details.append(dict(conflict,point=p[pair].tolist(),grid_point=gp.tolist(),
                    initial_geometry=geom[0],failure_geometry=geom[1],
                    initial_excluded_by_geometric_cutoff=geom[0]['clearance_m']>float(semantics['local_scp_proximity_margin_m']),
                    failure_inside_geometric_cutoff=geom[1]['clearance_m']<=float(semantics['local_scp_proximity_margin_m']),
                    saved_failure_d=saved_d,initial_same_point_d=float(d0[0]),
                    linear_prediction_from_initial=linear_from_q0,
                    linear_prediction_error=float(d[pair]-linear_from_q0),
                    prefix_q_shift_inf=float(np.max(np.abs(delta))),
                    initial_q_inside_this_step_box=bool(np.max(np.abs(delta))<=radius+1e-12),
                    local_row_required_unclipped_radius=needed,
                    sampled_box_corner_max_d=float(corner_d.max()),
                    same_row_lp=lp_check(data,[conflict['row']]),
                    q0=q0.tolist(),failure_qbar=qbar.tolist()))
            full=lp_check(data); base=lp_check(data,[])
            if full['status']!=2 or not base['feasible'] or base['max_constraint_violation']>1e-7:
                raise ValueError('independent LP diagnosis unexpected')
            if not all(x['same_row_lp']['status']==2 for x in details):
                raise ValueError('single row certificate disagrees with LP')
            qmatch=[t for t,q in measured if np.array_equal(q0,np.asarray(q))]
            if not qmatch: raise ValueError('snapshot measured q0 not found exactly in raw joint trace')
            result=dict(model=case['model'],case_id=case['case_id'],snapshot=str(path),sha256=snap['sha256'],
                plan_sequence=data['plan_sequence'],query_stamp_ns=data['query_stamp_ns'],
                observation_token=data['observation_token'],trust_radius=radius,
                reconstructed_qbar_error_inf=qlin_error,physical_joint_limit_clipping=False,
                acceleration_rows_present=False,hard_cdf_slack_variables=0,
                value_reproduction_error_max=valerr,gradient_reproduction_error_max=graderr,
                unchanged_piqp_replay=json.loads(replay.stdout),full_constraints_lp=full,
                dynamics_and_same_box_only_lp=base,exact_measured_q0_stamp_matches=qmatch,
                source_counts=dict(Counter(data['source_type'])),selected_rows=len(data['selected_pairs']),
                plan_events=plan_events,conflicts=details)
            results.append(result)
            print(case['model'],data['plan_sequence'],'PASS replay/FK/CDF/LP',
                  'conflicts',len(details),'initial_cutoff_excluded',all(x['initial_excluded_by_geometric_cutoff'] for x in details),flush=True)
    if not results: raise ValueError('no snapshot was diagnosed; refusing vacuous PASS')
    report=dict(schema_version=1,diagnostic_only=True,device='cpu',torch=torch.__version__,
        collision_checkpoint=str(COLLISION),collision_sha256=sha(COLLISION),collision_architecture=model.architecture,
        tests_status='PASS',cases=results,source_batch=str(args.batch),
        evidence_sha256={str(path):sha(path) for path in sorted(evidence_paths)},
        runtime_inputs_unchanged=True,helper_binary_sha256=sha(args.body_binary),
        replay_binary_sha256=sha(args.replay_binary),
        limitations=['No earlier full QP/batch or confidence-map snapshot was recorded; no invented old active set.',
            'Exact body proxy reproduction is not a true-geometry collision or self-occlusion oracle.',
            'CPU/recorded GPU values agree within stated reproduction errors; they are not bitwise equal.',
            'LP subset feasibility and sampled corner values are NOT certified executable trajectories.',
            'No runtime change, new simulation, A3, full30, training, or real robot.'])
    with (out/'report.json').open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False)


if __name__=='__main__': main()
