#!/usr/bin/env python3
"""Nonlinear/geometry checks and objective-only counterfactuals, never execution."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.optimize import minimize
import diagnose_phase_e_case001_exact_witness as d
from validate_visibility_oracle import DEFAULT_SENSOR_FRAMES, sensor_margin, find_chain_joints, fk_transform
from collision_cdf_model import CollisionCDF

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--solver',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.evidence_dir/'report.json').read_text());assert not report['unverified_plans']
    run=d.ROOT/'outputs/c5_5_vbc_gcdf_regime'/report['run_id']/'run'
    acq=d.strings(run/'visibility_acquisition_summary.csv')
    torch.set_num_threads(2)
    net=CollisionCDF(str(d.ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'),device='cpu',activation='gelu')
    geom=d.Geometry();sensor_robot=d.URDF.from_xml_file(str(d.ROOT/'src/arm_description/urdf/Arm.urdf'))
    chains={s:find_chain_joints(sensor_robot,'base_link',s) for s in DEFAULT_SENSOR_FRAMES}
    def model(points,qs):
        dv,gv=net.pair_distance_and_gradient(torch.tensor(np.array(points),dtype=torch.float32),torch.tensor(np.array(qs),dtype=torch.float32))
        return dv.detach().numpy(),gv.detach().numpy()
    def cdf_rows(rows,q):
        points=[json.loads(r['point']) for r in rows];margins=np.array([r['effective_margin'] for r in rows])
        dv,_=model(points,[q[int(r['timestep'])] for r in rows]);slack=dv-margins
        return dict(min_residual=float(slack.min()),violated_over_1e5=int(np.sum(slack < -1e-5)))
    def fov(points,q):
        transforms=[fk_transform(chains[s],dict(zip(d.JOINTS,q))) for s in DEFAULT_SENSOR_FRAMES]
        margins=np.array([[sensor_margin(np.array(p),tf,55.,72.,.15,.75)[0] for tf in transforms] for p in points])
        best=margins.max(axis=1)
        return dict(best_sensor_margin_m=best.tolist(),inside_any_fov_count=int(np.sum(best>=0)),point_count=len(points))
    rows0=json.loads((args.evidence_dir/'272/rows.json').read_text())
    unique={d.solver_point_key(json.loads(r['point'])):r for r in rows0}
    pointrows=list(unique.values());pts=np.array([json.loads(r['point']) for r in pointrows]);margins=np.array([r['effective_margin'] for r in pointrows])
    # Current O1 consists of the five +X=0.15 points, independently visible in
    # its acquisition geometry diagnostics; other rows include queued guards.
    activepts=pts[np.isclose(pts[:,0],.15)]
    targets={}
    for label,t in [('old_qvis',37.4),('new_qvis',54.2)]:
        a=[r for r in acq if r['ros_s']<=t][-1];q=np.fromstring(a['active_q_vis'],sep=',')
        dv,g=model(pts,np.repeat(q[None,:],len(pts),axis=0));slack=dv-margins
        target=dict(q_rounded_4decimals=q.tolist(),observation_token=a['active_q_vis_token'],
            cdf_min_margin_residual=float(slack.min()),violating_point_count=int(np.sum(slack < -1e-5)),
            cdf_points=[dict(point=p.tolist(),distance=float(v),margin=float(m),residual=float(v-m)) for p,v,m in zip(pts,dv,margins)],
            ideal_fov=fov(activepts,q),local_exact_min_m=float(geom.minimum(q,pts)[0].min()))
        targets[label]=target
    projected={}
    for label,t in [('old_qvis',37.4),('new_qvis',54.2)]:
        seed=np.array(targets[label]['q_rounded_4decimals']);cache={}
        def evaluate(q):
            key=tuple(q)
            if cache.get('key')!=key:
                dv,g=model(pts,np.repeat(q[None,:],len(pts),axis=0));cache.update(key=key,dv=dv,g=g)
            return cache['dv'],cache['g']
        lo=np.maximum(seed-.2,[geom.robot.joint_map[j].limit.lower+.01 for j in d.JOINTS])
        hi=np.minimum(seed+.2,[geom.robot.joint_map[j].limit.upper-.01 for j in d.JOINTS])
        # Endpoint-only feasibility diagnostic, NOT a target update or path.
        opt=minimize(lambda q:.5*np.sum((q-seed)**2),seed,jac=lambda q:q-seed,
            bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=lambda q:evaluate(q)[0]-margins-.0001,jac=lambda q:evaluate(q)[1])],
            method='SLSQP',options=dict(maxiter=40,ftol=1e-10))
        dv,_=evaluate(opt.x)
        a=[r for r in acq if r['ros_s']<=t][-1];qm=np.fromstring(a['measured_q'],sep=',')
        path=np.array([qm+s*(opt.x-qm) for s in np.linspace(0,1,101)])
        pp=np.tile(pts,(len(path),1));qq=np.repeat(path,len(pts),axis=0)
        ds,_=model(pp,qq);slack=ds.reshape(len(path),len(pts))-margins
        projected[label]=dict(optimizer_success=bool(opt.success),iterations=int(opt.nit),q=opt.x.tolist(),
            change_inf=float(np.max(np.abs(opt.x-seed))),min_endpoint_cdf_residual=float((dv-margins).min()),
            ideal_fov=fov(activepts,opt.x),local_exact_min_m=float(geom.minimum(opt.x,pts)[0].min()),
            sampled_straight_path_min_cdf_residual=float(slack.min()),sampled_straight_path_violating_poses=int(np.sum((slack < -1e-5).any(axis=1))),
            measured_path_start_rounded=qm.tolist(),full_gate='NOT_RUN_no_frozen_map')
    originals={};counter={};posechecks={}
    for n,info in report['plans'].items():
        sub=args.evidence_dir/n;rows=json.loads((sub/'rows.json').read_text());sol=json.loads((sub/'solution.json').read_text())
        q=np.vstack((info['q0'],np.array(sol['x'][:140]).reshape(20,7)))
        actual,grad=model([json.loads(r['point']) for r in rows],[json.loads(r['q_linearization']) for r in rows])
        err=float(np.max(np.abs(actual-[float(r['distance']) for r in rows])));assert err<1e-5,(n,err)
        originals[n]=dict(distance_requery_error=err,nonlinear=cdf_rows(rows,q))
    for ci,c in enumerate(report['cycles']):
        first=report['plans'][str(c['recovery_plans'][0])]
        normal=report['plans'][str(c['exit_plan'])];sub=args.evidence_dir/str(c['exit_plan'])
        rows=json.loads((sub/'rows.json').read_text());p=json.loads((sub/'problem.json').read_text())
        q0=np.array(normal['q0']);oldtarget=np.array(c['target']);delta=oldtarget-np.array(first['q0'])
        unit=delta/np.max(np.abs(delta));direction=delta/np.linalg.norm(delta)
        variants={}
        for label,target in [('actual_straight',np.array(normal['target'])),('keep_old_detour_target',oldtarget),('continue_same_direction',q0+.03*unit)]:
            cp=copy.deepcopy(p);cc=np.array(cp['c']);cc[7:14]+=6000*(np.array(normal['target'])-target);cp['c']=cc.tolist()
            for k in cp:
                if k!='c':assert cp[k]==p[k]
            sol=d.solve(args.solver,cp);assert sol['status']=='solved'
            x=np.array(sol['x']);q=np.vstack((q0,x[:140].reshape(20,7)))
            variants[label]=dict(q2=q[2].tolist(),q2_step_inf=float(np.max(np.abs(q[2]-q0))),
                tangent_signed_progress=float((q[2]-q0)@direction),
                old_detour_fraction_at_q2=float((q[2]-np.array(first['q0']))@delta/(delta@delta)),
                original_constraints=d.residuals(p,x),nonlinear=cdf_rows(rows,q),
                full_gate='NOT_RUN_no_frozen_map')
            d.save(out/f'cycle_{ci}_{label}_solution.json',sol)
        counter[str(c['trigger'])]=dict(exit_plan=c['exit_plan'],variants=variants)
        for label,q in [('start',c['q_trigger']),('exit',c['q_exit']),('end',c['q_end'])]:
            dv,_=model(pts,np.repeat(np.array(q)[None,:],len(pts),axis=0))
            posechecks[f'{ci}_{label}']=dict(ideal_fov=fov(activepts,q),min_cdf_residual=float((dv-margins).min()))
    # Isolate the k2-only objective's downstream motion, without touching any
    # safety, dynamics, trust-region or velocity constraint.
    n='274';info=report['plans'][n];sub=args.evidence_dir/n
    p=json.loads((sub/'problem.json').read_text());rows=json.loads((sub/'rows.json').read_text())
    cp=copy.deepcopy(p);P=d.unpack_matrix(cp['P']);cc=np.array(cp['c']);target=np.array(info['target'])
    for k in range(3,21):
        indices=slice((k-1)*7,k*7);P[indices,indices]+=6000*np.eye(7);cc[indices]-=6000*target
    cp['P']=d.qp_matrix(P);cp['c']=cc.tolist()
    for key in p:
        if key not in ('P','c'):assert cp[key]==p[key]
    sol=d.solve(args.solver,cp);assert sol['status']=='solved';xx=np.array(sol['x'])
    q=np.vstack((info['q0'],xx[:140].reshape(20,7)));u=xx[140:].reshape(20,7)
    classes=d.continuity_methods();msg=classes[0]._repair_prefix_with_braking_tail(d.trajectory(q,u,classes))
    tail=dict(q2_step_inf=float(np.max(np.abs(q[2]-q[0]))),raw_span=float(np.max(np.abs(q-q[0]))),
        prefix_brake_hold_span=float(np.max(np.abs(np.array([p.positions for p in msg.points])-q[0]))),
        original_constraints=d.residuals(p,xx),nonlinear=cdf_rows(rows,q),full_gate='NOT_RUN_no_frozen_map')
    d.save(out/'tail_hold_objective_problem.json',cp);d.save(out/'tail_hold_objective_solution.json',sol)
    d.save(out/'report.json',dict(originals=originals,qvis_targets=targets,projected_endpoint_diagnostics=projected,counterfactuals=counter,cycle_pose_checks=posechecks,tail_hold_objective_diagnostic=tail,
        active_points=activepts.tolist(),limits=['q_vis from four-decimal acquisition log; residuals are approximate.',
        'FOV uses configured URDF/range/angles, not frozen map confidence or primitive LOS.',
        'Counterfactuals are single-QP objective-only experiments, not trajectories approved for execution.']))
    print('QVIS',json.dumps(targets),flush=True)
    print('PROJECTED_ENDPOINTS',json.dumps(projected),flush=True)
    print('TAIL_HOLD',json.dumps(tail),flush=True)
    print('COUNTERFACTUALS',json.dumps(counter),flush=True)
    print('NONLINEAR',json.dumps(originals),flush=True)
    print('FOV',json.dumps(posechecks),flush=True)

if __name__=='__main__':main()
