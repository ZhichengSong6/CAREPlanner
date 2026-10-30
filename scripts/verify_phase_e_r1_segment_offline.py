#!/usr/bin/env python3
"""Exp74 R1/S7 local-segment counterfactual with reconstructed native guards.

No ROS imports or simulation. Reconstruct the observed QP before counterfactuals.
All safety results are scoped to the twelve recorded UNKNOWN point centers;
the complete map, exact VBC and execution remain unavailable.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

import diagnose_phase_e_case001_exact_witness as old
from diagnose_phase_e_r1_o1_o6_offline import save, sha, union_geometry, PerSensorVisibilityRuntime
from collision_cdf_model import CollisionCDF

ROOT = old.ROOT
RID = 'case_001_dependency_lock_fix_20260921_exp74'
RUN = ROOT/'outputs/c5_5_vbc_gcdf_regime'/RID/'run'
INPUT = ROOT/'outputs/phase_e_r1_o1_o6_offline/20260921_cpu'
LOG = ROOT/'logs/c5_5_vbc_gcdf_regime'/RID/'run/controlled.log'


class Duration:
    def __init__(self, value=0.): self.value = float(value)
    def to_sec(self): return self.value


class Point:
    def __init__(self):
        self.positions = []; self.velocities = []; self.accelerations = []; self.effort = []
        self.time_from_start = Duration()


class Trajectory:
    def __init__(self):
        self.header = SimpleNamespace(stamp=Duration())
        self.joint_names = []; self.points = []


def continuity():
    """Production interpolation/braking with inert message and clock containers."""
    wanted = {'_duration', '_lerp_array', '_interpolate_point', '_point_at_time',
              '_position_shift_inf', '_repair_prefix_with_braking_tail'}
    tree = ast.parse(old.CONTINUITY.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(methods) == len(wanted)
    ns = dict(copy=copy, math=math, JointTrajectory=Trajectory, JointTrajectoryPoint=Point,
        rospy=SimpleNamespace(Duration=Duration, Time=SimpleNamespace(now=lambda: Duration()),
                              logerr_throttle=lambda *a: None, logwarn_throttle=lambda *a: None))
    module = ast.Module(body=[ast.ClassDef(name='ContinuityMath', bases=[], keywords=[],
                        body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(old.CONTINUITY), 'exec'), ns)
    obj = ns['ContinuityMath']()
    obj.repair_execution_prefix_s = .25
    obj.repair_brake_dt_s = .05
    obj.repair_hold_s = .1
    obj.repair_acceleration_limits = [3., 3., 4., 4., 6., 6., 6.]
    obj.repair_brake_max_steps = 20
    return obj


def trajectory(q, u):
    result = Trajectory(); result.joint_names = old.JOINTS
    for k in range(21):
        p = Point(); p.positions = q[k].tolist()
        p.velocities = u[k].tolist() if k < 20 else [0.]*7
        p.time_from_start = Duration(.05*k); result.points.append(p)
    return result


def make_problem(qbar, rows, target, previous, limits, radius):
    result = old.assemble(qbar, rows, target, previous, limits, radius)
    result['h_l'] = [b+r['effective_margin']-.02 for b, r in zip(result['h_l'], rows)]
    # exp74 recovery owns the tail as well as k=2. The old exp70 helper lacks it.
    P = old.unpack_matrix(result['P']); c = np.array(result['c'])
    for k in range(3, 21):
        indices = np.arange((k-1)*7, k*7)
        P[indices, indices] += 6000.
        c[indices] -= 6000.*target
    result['P'] = old.qp_matrix(P); result['c'] = c.tolist()
    return result


def solve(binary, problem, q0):
    result = old.solve(binary, problem)
    if result['status'] != 'solved':
        return result, None, None
    x = np.array(result['x'])
    q = np.vstack((q0, x[:140].reshape(20, 7)))
    return result, q, x[140:].reshape(20, 7)


def field(net, points, qs):
    points, qs = np.asarray(points), np.asarray(qs)
    x = np.tile(points, (len(qs), 1))
    q = np.repeat(qs, len(points), axis=0)
    values, grads = [], []
    for i in range(0, len(x), 256):
        v, g = net.pair_distance_and_gradient(torch.tensor(x[i:i+256], dtype=torch.float32),
                                             torch.tensor(q[i:i+256], dtype=torch.float32))
        values.append(v.numpy()); grads.append(g.numpy())
    return np.concatenate(values).reshape(len(qs), len(points)), np.concatenate(grads).reshape(len(qs), len(points), 7)


def query_rows(net, points, margins, q):
    ds, gs = field(net, points, q[1:])
    rows = []
    for k in range(1, 21):
        for i, p in enumerate(points):
            rows.append(dict(timestep=str(k), point=json.dumps(p.tolist()),
                distance=str(float(ds[k-1, i])), gradient=json.dumps(gs[k-1, i].tolist()),
                q_linearization=json.dumps(np.asarray(q[k], dtype=np.float32).tolist()),
                effective_margin=float(margins[i])))
    return rows, float(ds[:5].min())


def replan(net, binary, points, margins, q0, previous, target, limits):
    qbar = np.tile(q0, (21, 1)); radius = .15; last_min = None
    history = []
    for iteration in range(6):
        rows, min_d = query_rows(net, points, margins, qbar)
        if last_min is not None:
            if min_d-last_min > .002: radius = min(.2, radius*1.5)
            elif min_d-last_min < -.002: radius = max(.02, radius*.5)
        problem = make_problem(qbar, rows, target, previous, limits, radius)
        result, q, u = solve(binary, problem, q0)
        if q is None:
            return dict(status=result['status'], iterations=history), None, None
        step = float(np.max(np.abs(q-qbar)))
        nonlinear, _ = field(net, points, q)
        history.append(dict(iteration=iteration+1, piqp_iterations=result['iterations'],
            trust_radius=radius, step_inf=step, nonlinear_min_slack=float((nonlinear-margins).min())))
        if step <= .01: break
        qbar = q; last_min = min_d
    return dict(status='solved', converged=step <= .01, iterations=history,
                qualified_as_production_scp=False,
                scope='hard-row subproblem with frozen point set and explicit replacement of frontier/tail objective'), q, u


def audit(net, geom, vis, cont, points, margins, target_points, msg, desired):
    # Five subdivisions per stored segment include knots, brake and hold.
    times = sorted({p.time_from_start.to_sec() for p in msg.points} |
                   {float(t) for a,b in zip(msg.points,msg.points[1:])
                    for t in np.linspace(a.time_from_start.to_sec(), b.time_from_start.to_sec(), 6)})
    qs = np.array([cont._point_at_time(msg, t).positions for t in times])
    ds, _ = field(net, points, qs); slack = ds-margins
    pi, wi = np.unravel_index(np.argmin(slack), slack.shape)
    geometric = np.array([geom.minimum(q, points)[0] for q in qs])
    begin_vis = union_geometry(vis, target_points, qs[0])
    end_vis = union_geometry(vis, target_points, qs[-1])
    return dict(sample_count=len(qs), times_s=times, q=qs,
        min_nonlinear_guard_slack=float(slack.min()),
        interior_min_nonlinear_guard_slack=float(slack[1:].min()),
        worst_time_s=times[pi], worst_point=points[wi], worst_margin=float(margins[wi]),
        sampled_native_guards_nonnegative=bool(slack.min() >= 0),
        min_point_center_distance_m=float(geometric.min()),
        sampled_point_centers_outside_body=bool(geometric.min() > 0),
        start_target_visible_fraction=begin_vis['visible_point_fraction'],
        end_target_visible_fraction=end_vis['visible_point_fraction'],
        start_best_fov_g=begin_vis['best_single_sensor_g'], end_best_fov_g=end_vis['best_single_sensor_g'],
        executed_span_inf=float(np.max(np.abs(qs-qs[0]))),
        endpoint_displacement_inf=float(np.max(np.abs(qs[-1]-qs[0]))),
        qvis_distance_inf_before=float(np.max(np.abs(desired-qs[0]))),
        qvis_distance_inf_after=float(np.max(np.abs(desired-qs[-1]))),
        qvis_l2_gain=float(np.linalg.norm(desired-qs[0])-np.linalg.norm(desired-qs[-1])),
        actual_seen='NOT_RUN', complete_map_gcdf_vbc='NOT_RUN', execution_authorized=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--solver', type=Path, required=True)
    args = ap.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    evidence = json.loads((INPUT/'snapshots.json').read_text())
    candidates = json.loads((INPUT/'diagnostic.json').read_text())
    snap = next(s for s in evidence['snapshots'] if s['query_stamp_ns']==22635000000)
    snapshot_rows = snap['witness_rows']; q0 = np.array(snap['measured_q'])
    planners = old.strings(RUN/'local_planner_summary.csv')
    outcomes = old.strings(RUN/'verification_outcome.csv')
    original = next(r for r in planners if r.get('query_stamp_ns')=='22635000000' and r['event']=='scp_solved')
    published = next(r for r in planners if r.get('plan_seq')=='140' and r['event']=='candidate_published')
    assert published['query_stamp_ns']=='22666000000'
    prefix_state = [v for v in outcomes if v['ros_s']<=22.635][-1]
    assert prefix_state['cycle_recovery_active']=='1.000000'
    effective_prefix = float(prefix_state['cycle_recovery_prefix_s'])
    assert effective_prefix==.15
    log = LOG.read_text()
    expected = {'repair_task_tracking_scale':'0.0', 'u_reference_tracking_enabled':'False',
        'u_tracking_weight':'0.01', 'u_smooth_weight':'1.0', 'handoff_velocity_weight':'1.0',
        'visibility_waypoint_weight':'3000.0', 'enforce_acceleration_constraints':'False'}
    for key, value in expected.items():
        assert f'/local_sparse_scp_planner_node/local_planner/{key}: {value}\n' in log
    for key, value in {'safety_margin':'.01','visibility_obligation_cdf_margin':'.02',
                       'vbc_feedback_cdf_margin_step':'.1','vbc_feedback_cdf_margin_max':'.5'}.items():
        assert f'/local_sparse_scp_planner_node/local_planner/cdf/{key}: {float(value)}\n' in log
    for key, value in {'repair_execution_prefix_s':.25,'repair_brake_dt_s':.05,'repair_hold_s':.1}.items():
        assert f'/optimized_trajectory_continuity/{key}: {value}\n' in log
    sources = [INPUT/'snapshots.json',INPUT/'diagnostic.json',LOG,args.solver,Path(__file__),
        ROOT/'scripts/solve_phase_e_offline_qp.cpp',ROOT/'scripts/diagnose_phase_e_case001_exact_witness.py',
        old.CONTINUITY,old.URDF_PATH,ROOT/'src/arm_description/urdf/Arm.urdf',
        ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',
        ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/safe_frontier_recovery.hpp']
    sources += list((ROOT/'src/care_collision_cdf/checkpoints').rglob('*.pt'))
    sources += list((ROOT/'src/care_visibility_cdf/checkpoints').rglob('*.pt'))
    sources += [RUN/(x+'.csv') for x in ['local_planner_summary','local_gcdf_query','verification_outcome',
                'actuator_command','committed_trajectory','observation_candidate_identity','local_witness_diagnostics',
                'local_witness_selector_diagnostics']]
    hashes = {str(p):sha(p) for p in sources}
    # Point-scoped feedback increments; retain these guards across same-mode target refreshes.
    margins_by_key = {}; events = []
    for v in outcomes:
        if v['ros_s']>=22.635 or v.get('result')!='unsafe' or v.get('safety_gate')!='vbc': continue
        e = json.loads(v['vbc_evidence'])
        assert e['trajectory_stamp_ns']==v['audited_trajectory_stamp_ns']
        points_by_key = {old.solver_point_key(z['point']):z['point'] for z in e['violations']}
        updated = []
        for key, point in points_by_key.items():
            margin = min(.5, margins_by_key.get(key,.02)+.1)
            margins_by_key[key] = margin
            updated.append(dict(point=point, margin=margin))
        events.append(dict(verification_seq=v['seq'], ros_s=v['ros_s'], raw=v['raw_candidate_stamp_ns'],
                           audited=v['audited_trajectory_stamp_ns'], updated=updated))
    assert len(events)==8
    feedback_log = [line for line in log.splitlines() if 'exact VBC queued-target feedback' in line]
    assert len(feedback_log)==8
    for event, line in zip(events,feedback_log):
        t=float(re.search(r', ([0-9.]+)\]',line)[1])
        assert abs(t-event['ros_s']) < .005
        assert int(re.search(r' points=(\d+)',line)[1])==len(event['updated'])
        assert abs(float(re.search(r'cdf_margin_max=([0-9.]+)',line)[1])-max(x['margin'] for x in event['updated']))<1e-12
    points = np.array([json.loads(r['point']) for r in snapshot_rows if r['timestep']=='1'])
    margins = np.array([margins_by_key[old.solver_point_key(p)] for p in points])
    assert len(points)==12 and len(snapshot_rows)==240
    for row in snapshot_rows:
        row['effective_margin']=margins_by_key[old.solver_point_key(json.loads(row['point']))]
    margin_report = dict(events=events,publisher_feedback_log=feedback_log,
        points=[dict(point=p,margin=m) for p,m in zip(points,margins)],
        qualification='history-derived, source-routing checked and independently validated by full next-query QP reproduction')
    geom = old.Geometry(); limits=np.array([[geom.robot.joint_map[j].limit.lower,geom.robot.joint_map[j].limit.upper] for j in old.JOINTS])
    queries = {r['field.header.stamp']:r for r in old.numeric(RUN/'local_gcdf_query.csv') if r.get('field.header.stamp') in ['22635000000','22666000000']}
    qbar = np.array([old.values(queries['22635000000'],f'field.points{k}.positions') for k in range(21)])
    recorded_next_q = np.array([old.values(queries['22666000000'],f'field.points{k}.positions') for k in range(21)])
    recorded_next_u = np.array([old.values(queries['22666000000'],f'field.points{k}.velocities') for k in range(20)])
    command = [r for r in old.numeric(RUN/'actuator_command.csv') if r['%time'].isdigit() and int(r['%time'])<=22635000000][-1]
    command_receipt_ns = int(command['%time'])
    previous = old.values(command,'field.data')
    original_target = np.array(json.loads(original['recovery_frontier_target']))
    problem = make_problem(qbar,snapshot_rows,original_target,previous,limits,.15)
    baseline,q,u = solve(args.solver,problem,q0)
    assert baseline['status']=='solved'
    errors = dict(q_max_abs=float(np.max(np.abs(q-recorded_next_q))),u_max_abs=float(np.max(np.abs(u-recorded_next_u))))
    assert errors['q_max_abs']<2e-6 and errors['u_max_abs']<2e-5
    assert baseline['iterations']==int(original['piqp_iter'])
    save(out/'baseline_problem.json',problem);save(out/'baseline_solution.json',baseline)
    save(out/'margins.json',margin_report)
    print('Baseline reconstruction PASS',errors,flush=True)
    net=CollisionCDF(str(ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'),device='cpu')
    for p in net.model.parameters():p.requires_grad_(False)
    # Compare actual collision model values AND gradients with recorded 240 rows.
    recorded_points=np.array([json.loads(r['point']) for r in snapshot_rows])
    recorded_q=np.array([json.loads(r['q_linearization']) for r in snapshot_rows])
    rd,rg=net.pair_distance_and_gradient(torch.tensor(recorded_points,dtype=torch.float32),torch.tensor(recorded_q,dtype=torch.float32))
    model_error=dict(value=float(np.max(np.abs(rd.numpy()-[float(r['distance']) for r in snapshot_rows]))),
        gradient=float(np.max(np.abs(rg.numpy()-np.array([json.loads(r['gradient']) for r in snapshot_rows])))))
    assert max(model_error.values())<2e-5,model_error
    vis=PerSensorVisibilityRuntime(checkpoint_path=str(ROOT/'src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt'),
        reference_urdf_path=str(ROOT/'src/arm_description/urdf/Arm.urdf'),self_filter_urdf_path=str(old.URDF_PATH),
        device=torch.device('cpu'),q_min=[-3.14,-2.30,-3.14,-2.65,-3.14,-3.14,-1.20],q_max=[3.14,2.30,3.14,2.65,3.14,3.14,1.20])
    cont=continuity();target_points=evidence['traces']['6']['active_set_points_xyz']
    # Validate the final QP and inert-message prefix/brake math against the
    # actually committed trajectory, allowing only its logged rebase/suffix.
    final_rows=[r for r in old.strings(RUN/'local_witness_selector_diagnostics.csv')
                if r.get('query_stamp_ns')=='22666000000' and r.get('plan_seq')=='140']
    assert len(final_rows)==240
    for row in final_rows:
        row['effective_margin']=margins_by_key[old.solver_point_key(json.loads(row['point']))]
    final_problem=make_problem(recorded_next_q,final_rows,original_target,previous,limits,float(published['trust_q_inf']))
    final_solution,final_q,final_u=solve(args.solver,final_problem,q0)
    assert final_q is not None
    identities=old.strings(RUN/'observation_candidate_identity.csv')
    raw_ids={r['raw_candidate_stamp_ns'] for r in identities if r.get('observation_token')==snap['dependency']['observation_token']}
    matches=[v for v in outcomes if v.get('raw_candidate_stamp_ns') in raw_ids and v.get('committed')=='1'
             and 22666000000<=int(v['raw_candidate_stamp_ns'])<=round(published['ros_s']*1e9)]
    assert len(matches)==1
    match=matches[0]
    assert float(match['cycle_recovery_prefix_s'])==effective_prefix
    stored=next(r for r in old.numeric(RUN/'committed_trajectory.csv') if r.get('field.header.stamp')==match['execution_stamp_ns'])
    braked=cont._repair_prefix_with_braking_tail(trajectory(final_q,final_u),effective_prefix)
    calcq=[];calcu=[];recq=[];recu=[];k=0
    while stored.get(f'field.points{k}.time_from_start'):
        t=float(stored[f'field.points{k}.time_from_start'])/1e9+float(match['precommit_suffix_phase_s'])
        point=cont._point_at_time(braked,t)
        calcq.append(point.positions);calcu.append(point.velocities)
        recq.append(old.values(stored,f'field.points{k}.positions'))
        recu.append(old.values(stored,f'field.points{k}.velocities'));k+=1
    calcq=np.array(calcq);recq=np.array(recq)
    commit_check=dict(raw=match['raw_candidate_stamp_ns'],audited=match['audited_trajectory_stamp_ns'],
        execution=match['execution_stamp_ns'],effective_prefix_s=effective_prefix,
        shape_error=float(np.max(np.abs((recq-recq[0])-(calcq-calcq[0])))),
        velocity_error=float(np.max(np.abs(np.array(recu)-calcu))),
        duration_error=abs(cont._duration(braked)-float(match['constructed_executable_duration_s'])),
        final_qp_iterations=final_solution['iterations'])
    assert commit_check['shape_error']<2e-6 and commit_check['velocity_error']<2e-5
    assert commit_check['duration_error']<1e-9
    print('Committed prefix/brake/hold replay PASS',commit_check,flush=True)
    desired={h:np.array(next(r['branch']['q_candidate'] for r in candidates['candidates'] if
        r['snapshot']==snap['label'] and r['obligation']==6 and r['model']=='R1' and r['sensor']==h)) for h in [4,7]}
    experiments=[]
    def one(label,head,start,command,step):
        delta=desired[head]-start
        target=start+delta*min(1.,step/max(float(np.max(np.abs(delta))),1e-12))
        scp,q,u=replan(net,args.solver,points,margins,start,command,target,limits)
        result=dict(label=label,head=head,start_q=start,frontier_target=target,requested_step_inf=step,scp=scp)
        if q is None:return result,None
        raw=trajectory(q,u);braked=cont._repair_prefix_with_braking_tail(raw,effective_prefix)
        assert braked is not None and max(abs(x) for x in braked.points[-1].velocities)==0
        result['raw_q']=q;result['raw_u']=u
        result['prefix_brake_hold'] = audit(net,geom,vis,cont,points,margins,target_points,braked,desired[head])
        result['prefix_brake_hold']['duration_s']=cont._duration(braked)
        result['prefix_brake_hold']['brake_displacement_inf']=cont._last_brake_displacement_inf
        result['raw_nonlinear_min_slack']=float((field(net,points,q)[0]-margins).min())
        experiments.append(result)
        a=result['prefix_brake_hold']
        print(label,'slack',a['min_nonlinear_guard_slack'],'span',a['executed_span_inf'],
              'gain',a['qvis_l2_gain'],'visible',a['end_target_visible_fraction'],flush=True)
        return result,np.array(braked.points[-1].positions)
    for head in [4,7]:
        for step in [.01,.03,.05]:one(f'S{head}_first_{step}',head,q0,previous,step)
    # Cap ideal stop-and-go trials at six; this is not the measured-motion
    # recovery state machine (which counts .01-rad transitions).
    rollouts=[]
    for head in [4,7]:
        current=q0.copy();command=previous.copy();labels=[];stop='six_segment_budget'
        for index in range(6):
            result,next_q=one(f'S{head}_rollout_{index+1}',head,current,command,.03)
            labels.append(result['label'])
            if next_q is None:stop='qp_failed';break
            a=result['prefix_brake_hold']
            if not a['sampled_native_guards_nonnegative'] or not a['sampled_point_centers_outside_body']:
                stop='recorded_subset_check_failed';break
            if not result['scp']['converged']:stop='scp_not_converged';break
            current=next_q;command=np.zeros(7)
        rollouts.append(dict(head=head,labels=labels,stop=stop,final_q=current,
            displacement_inf=float(np.max(np.abs(current-q0))),
            qvis_l2_gain=float(np.linalg.norm(desired[head]-q0)-np.linalg.norm(desired[head]-current)),
            qualification='hypothetical ideal segment completion, fixed twelve UNKNOWN guards, no actual execution'))
    # Re-query the long straight diagnostic path: do not impose it as an endpoint filter.
    line_checks=[]
    for head in [4,7]:
        qs=q0+np.linspace(0,1,129)[:,None]*(desired[head]-q0)
        ds,_=field(net,points,qs);slack=ds-margins
        line_checks.append(dict(head=head,min_slack=float(slack.min()),
            violating_samples=int(np.sum(slack.min(axis=1)<0)),sample_min_slack=slack.min(axis=1),
            qualification='lookahead diagnosis only; no requirement that whole UNKNOWN path pass in advance'))
    assert hashes=={str(p):sha(p) for p in sources}
    assert 'rospy' not in sys.modules
    report=dict(status='OFFLINE_RECORDED_GUARD_SEGMENT_DIAGNOSTIC',run_id=RID,query_stamp_ns=22635000000,
        baseline_errors=errors,baseline_iterations=baseline['iterations'],collision_model_replay_error=model_error,
        previous_command_receipt_ns=command_receipt_ns,committed_trajectory_replay=commit_check,
        effective_prefix_state=prefix_state,
        source_sha256=hashes,margins=margin_report,experiments=experiments,rollouts=rollouts,line_checks=line_checks,
        parameters=dict(qp_max_iterations=300,scp_max_iterations=6,scp_step_tolerance_inf=.01,
            diagnostic_segment_budget=6,frontier_weight=3000,frontier_step_rad=.03,
            configured_prefix_s=.25,effective_prefix_s=effective_prefix,brake_dt_s=.05,hold_s=.1,
            brake_acceleration_limits=[3,3,4,4,6,6,6]),
        scope='Explicit offline replacement of frontier/tail objective; not production policy integration.',
        not_run=['complete map and environment LOS','full final GCDF/exact VBC','tracker/robot dynamics',
                 'actual seen, guard release and parent resume','ROS/Gazebo'],
        protected_inputs_unchanged=True,rospy_imported=False)
    save(out/'report.json',report)
    print('DONE',out,flush=True)


if __name__=='__main__':main()
