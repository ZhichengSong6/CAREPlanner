#!/usr/bin/env python3
"""Offline exp71 QP/active-constraint diagnosis; never publishes commands."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import diagnose_phase_e_case001_exact_witness as d

RID = 'case_001_witness_margin_fix_20260921_exp71_retry2'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--solver', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    run = d.ROOT/'outputs/c5_5_vbc_gcdf_regime'/RID/'run'
    events = d.strings(run/'local_planner_summary.csv')
    pub = next(r for r in events if r.get('plan_seq')=='418' and r.get('event')=='candidate_published')
    start = next(r for r in events if r.get('plan_seq')=='418' and r.get('event')=='plan_started')
    query = next(r for r in d.numeric(run/'local_gcdf_query.csv') if r['field.header.stamp']==pub['query_stamp_ns'])
    qbar = np.array([d.values(query, f'field.points{k}.positions') for k in range(21)])
    ubar = np.array([d.values(query, f'field.points{k}.velocities') for k in range(20)])
    rows = sorted([r for r in d.strings(run/'local_witness_selector_diagnostics.csv')
                   if r.get('query_stamp_ns')==pub['query_stamp_ns']], key=lambda r:int(r['batch_pair']))
    assert len(rows)==int(pub['cdf_rows'])==300
    assert all(r['status']=='evaluated_unknown' for r in rows)
    assert pub['obligation_cdf_rows']=='300' and pub['screened_safe']=='0'
    # Replay the two exact queued-witness feedback events, independently of d(q).
    margins = {}
    feedback = []
    for event in d.strings(run/'verification_outcome.csv'):
        if event['ros_s']>start['ros_s'] or event.get('result')!='unsafe' or event.get('safety_gate')!='vbc':
            continue
        evidence = json.loads(event['vbc_evidence'])
        points = {d.solver_point_key(v['point']): v['point'] for v in evidence['violations']}
        for key in points: margins[key] = min(.50, margins.get(key,.02)+.10)
        feedback.append(dict(stamp=event['audited_trajectory_stamp_ns'], points=list(points.values())))
    assert len(feedback)==2 and len(margins)==10
    for row in rows:
        row['effective_margin'] = max(.02, margins.get(d.solver_point_key(json.loads(row['point'])), .02))
    # Same-stamp publications may arrive after the planner freezes its inputs.
    # Require the pre-start frontier and independently match its logged shift.
    frontier_rows = [r for r in d.numeric(run/'visibility_frontier_target.csv') if int(r['%time'])<round(start['ros_s']*1e9)]
    fr = frontier_rows[-1]
    frontier = d.values(fr, 'field.data',10)[3:]
    assert np.allclose(d.values(fr, 'field.data',3), [1,1,0])
    assert abs(np.max(np.abs(frontier-qbar[0]))-float(start['frontier_target_shift_inf']))<5e-8
    commands = [r for r in d.numeric(run/'actuator_command.csv') if int(r['%time'])<round(start['ros_s']*1e9)]
    previous = d.values(commands[-1], 'field.data')
    acq = [r for r in d.strings(run/'visibility_acquisition_summary.csv') if r['ros_s']<=start['ros_s']][-1]
    qvis = np.fromstring(acq['active_q_vis'], sep=',')
    geom = d.Geometry()
    limits = np.array([[geom.robot.joint_map[j].limit.lower,geom.robot.joint_map[j].limit.upper] for j in d.JOINTS])
    log = d.ROOT/'logs/c5_5_vbc_gcdf_regime'/RID/'run/controlled.log'
    config = log.read_text()
    for name,value in {'repair_task_tracking_scale':'0.0','u_reference_tracking_enabled':'False',
            'u_tracking_weight':'0.01','u_smooth_weight':'1.0','handoff_velocity_weight':'1.0',
            'visibility_waypoint_weight':'3000.0','enforce_acceleration_constraints':'False'}.items():
        assert f'/local_sparse_scp_planner_node/local_planner/{name}: {value}\n' in config, name
    problem = d.assemble(qbar, rows, frontier, previous, limits,float(pub['trust_q_inf']))
    problem['h_l'] = [h+r['effective_margin']-.02 for h,r in zip(problem['h_l'],rows)]
    problem['include_duals'] = True
    base = d.solve(args.solver,problem)
    assert base['status']=='solved',base['status']
    x=np.array(base['x']);q=np.vstack((qbar[0],x[:140].reshape(20,7)));u=x[140:].reshape(20,7)
    P=d.unpack_matrix(problem['P']);G=d.unpack_matrix(problem['G']);A=d.unpack_matrix(problem['A'])
    slack=G@x-np.array(problem['h_l'])
    objective=P@x+problem['c']
    reaction=G.T@np.array(base['z_l'])
    stationarity=objective+A.T@np.array(base['y'])-reaction+G.T@np.array(base['z_u'])-base['z_bl']+np.array(base['z_bu'])
    delta=frontier-qbar[0]
    active=[]
    for i,row in enumerate(rows):
        if base['z_l'][i]<.01: continue
        gradient=np.array(json.loads(row['gradient']))
        active.append(dict(pair=int(row['batch_pair']),point=json.loads(row['point']),k=int(row['timestep']),
            margin=row['effective_margin'],d=float(row['distance']),slack=float(slack[i]),dual=base['z_l'][i],
            gradient=gradient.tolist(),gradient_dot_frontier_step=float(gradient@delta),
            allowed_straight_step_fraction=float((float(row['distance'])-row['effective_margin'])/(-gradient@delta)) if gradient@delta<0 else None))
    # Check reconstructed trajectory shape against actually committed prefix.
    ver=next(r for r in d.strings(run/'verification_outcome.csv') if r.get('raw_candidate_stamp_ns')=='67159000000')
    assert ver['result']=='safe'
    identity=next(r for r in d.strings(run/'observation_candidate_identity.csv') if r.get('raw_candidate_stamp_ns')=='67159000000')
    assert identity['observation_token']==pub['observation_token']
    committed=next(r for r in d.numeric(run/'committed_trajectory.csv') if r['field.header.stamp']==ver['execution_stamp_ns'])
    classes=d.continuity_methods()
    raw=d.trajectory(q,u,classes)
    braked=classes[0]._repair_prefix_with_braking_tail(raw)
    recq=[];calq=[];recu=[];calu=[]
    k=0
    while f'field.points{k}.time_from_start' in committed and committed[f'field.points{k}.time_from_start']:
        t=float(committed[f'field.points{k}.time_from_start'])/1e9+float(ver['precommit_suffix_phase_s'])
        p=classes[0]._point_at_time(braked,t)
        recq.append(d.values(committed,f'field.points{k}.positions'));calq.append(p.positions)
        recu.append(d.values(committed,f'field.points{k}.velocities'));calu.append(p.velocities)
        k+=1
    recq=np.array(recq);calq=np.array(calq)
    shape_error=float(np.max(np.abs((recq-recq[0])-(calq-calq[0]))))
    velocity_error=float(np.max(np.abs(np.array(recu)-calu)))
    step=float(np.max(np.abs(q-qbar)))
    baseline_metrics=dict(solver_iterations=base['iterations'],logged_iterations=int(pub['piqp_iter']),
        step_inf=step,logged_step_inf=float(pub['step_inf']),commit_shape_error_inf=shape_error,
        commit_velocity_error_inf=velocity_error,commit_translation=(recq[0]-calq[0]).tolist(),
        q2_delta=(q[2]-qbar[0]).tolist(),q2_delta_inf=float(np.max(np.abs(q[2]-qbar[0]))),
        frontier_residual_inf=float(np.max(np.abs(q[2]-frontier))),
        kkt_stationarity_inf=float(np.max(np.abs(stationarity))),residuals=d.residuals(problem,x))
    d.save(out/'baseline_problem.json',problem);d.save(out/'baseline_solution.json',base)
    d.save(out/'selected_rows.json',rows)
    d.save(out/'baseline_metrics.json',baseline_metrics)
    print('baseline',json.dumps(baseline_metrics),flush=True)
    assert abs(step-float(pub['step_inf']))<5e-8 and shape_error<1e-7 and velocity_error<1e-6, 'Reconstruction not verified; counterfactuals blocked'
    variants={}
    points=np.array([json.loads(r['point']) for r in rows[::20]])
    # Diagnostic-only relaxations; never proposed for execution or certification.
    for name,cap in [('actual',None),('witness_cap_012',.12),('witness_cap_002',.02),('stronger_frontier',None)]:
        cp=copy.deepcopy(problem)
        if cap is not None:
            cp['h_l']=[h+min(r['effective_margin'],cap)-r['effective_margin'] for h,r in zip(problem['h_l'],rows)]
        if name=='stronger_frontier':
            pp=P.copy();cc=np.array(problem['c']);pp[7:14,7:14]+=np.eye(7)*54000;cc[7:14]-=54000*frontier
            cp['P']=d.qp_matrix(pp);cp['c']=cc.tolist()
        sol=base if name=='actual' else d.solve(args.solver,cp)
        d.save(out/f'{name}_solution.json',sol)
        d.save(out/f'{name}_problem.json',cp)
        if sol['status']!='solved': variants[name]={'status':sol['status']};continue
        xx=np.array(sol['x']);qq=np.vstack((qbar[0],xx[:140].reshape(20,7)))
        uu=xx[140:].reshape(20,7)
        msg=classes[0]._repair_prefix_with_braking_tail(d.trajectory(qq,uu,classes))
        sweep=geom.sweep(msg)
        unique=np.array(list({d.solver_point_key(json.loads(r['point'])):json.loads(r['point']) for r in rows}.values()))
        distances=np.array([geom.minimum(s['q'],unique)[0] for s in sweep])
        variants[name]=dict(status=sol['status'],q2_delta_inf=float(np.max(np.abs(qq[2]-qbar[0]))),
            frontier_residual_inf=float(np.max(np.abs(qq[2]-frontier))),
            frontier_direction_progress=float((qq[2]-qbar[0])@delta/(delta@delta)),
            original_constraints=d.residuals(problem,xx),
            local_exact_min_m=float(distances.min()),local_exact_intersecting_points=unique[(distances<0).any(axis=0)].tolist(),
            full_gate='NOT_RUN_missing_frozen_map')
    result=dict(run_id=RID,plan_seq=418,query_stamp_ns=pub['query_stamp_ns'],observation_token=pub['observation_token'],
        baseline=baseline_metrics,frontier=frontier.tolist(),frontier_receipt_ns=fr['%time'],q0=qbar[0].tolist(),
        qvis_rounded_log=qvis.tolist(),frontier_step=delta.tolist(),previous_command=previous.tolist(),
        margin_feedback=feedback,active_rows=active,variants=variants,
        evidence_scope='Reassembled QP, committed prefix shape validated; raw full candidate and matrix snapshot absent.',
        source_sha256={str(p):d.sha(p) for p in [Path(__file__),args.solver,d.ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',run/'local_witness_selector_diagnostics.csv',run/'local_gcdf_query.csv',run/'committed_trajectory.csv']})
    d.save(out/'report.json',result)
    print('active',json.dumps(active),flush=True)
    print('variants',json.dumps(variants),flush=True)


if __name__=='__main__':main()
