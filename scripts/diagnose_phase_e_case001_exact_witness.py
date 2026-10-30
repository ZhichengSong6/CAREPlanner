#!/usr/bin/env python3
"""Offline exp70 lineage/reassembled-QP experiment. Never publishes ROS commands.

The saved final query is NOT relabelled as a recorded candidate. Reassembly must
reproduce its converged solution before counterfactuals are attempted. A local
witness clearance check is NOT a full map/VBC certificate.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'case_001_exec_vbc_urgent_recovery_20260919_exp70'
for path in ['src/care_visibility_cdf/scripts', 'src/care_collision_cdf/scripts']:
    sys.path.insert(0, str(ROOT/path))
from analyze_phase_e_observation_events import string_rows, solver_point_key
from validate_visibility_oracle import URDF, find_chain_joints, fk_transform
from check_visibility_self_occlusion import load_collision_primitives

JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']
URDF_PATH = ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
CONTINUITY = ROOT/'src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def strings(path):
    return string_rows(path, {'start_ros_s': 0., 'end_ros_s': math.inf})


def numeric(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def values(row, prefix, count=7):
    return np.array([float(row[prefix+str(j)]) for j in range(count)])


def qp_matrix(m):
    coo = sparse.coo_matrix(m)
    return dict(shape=list(coo.shape), coo=np.column_stack((coo.row, coo.col, coo.data)).tolist())


def solve(binary, problem):
    p = subprocess.run([str(binary)], input=json.dumps(problem, allow_nan=False),
                       text=True, capture_output=True, check=True)
    return json.loads(p.stdout)


def exact_sdf(kind, p, g):
    if kind == 'sphere':
        return np.linalg.norm(p, axis=-1)-g.radius
    if kind == 'box':
        a = np.abs(p)-np.array(g.size)/2
    else:
        a = np.stack((np.linalg.norm(p[..., :2], axis=-1)-g.radius,
                      np.abs(p[..., 2])-g.length/2), axis=-1)
    return np.linalg.norm(np.maximum(a, 0.), axis=-1)+np.minimum(a.max(axis=-1), 0.)


class Geometry:
    def __init__(self):
        self.robot = URDF.from_xml_file(str(URDF_PATH))
        self.primitives = [p for p in load_collision_primitives(self.robot)
                           if p['link'] not in ('base_link', 'link1')]
        if len(self.primitives) != 21:
            raise ValueError('Unexpected active primitive model')
        self.chains = {p['link']: find_chain_joints(self.robot, 'base_link', p['link'])
                       for p in self.primitives}
        self.radii = []
        for p in self.primitives:
            g = p['geometry']
            r = (np.linalg.norm(g.size)/2 if p['type']=='box' else
                 math.hypot(g.radius, g.length/2) if p['type']=='cylinder' else g.radius)
            r += np.linalg.norm(p['origin'][:3, 3])
            bounds = np.zeros(7)
            for joint in reversed(self.chains[p['link']]):
                if joint.type != 'fixed': bounds[JOINTS.index(joint.name)] = r
                r += np.linalg.norm(joint.origin.xyz if joint.origin else [0.,0.,0.])
            self.radii.append(bounds)
        self.radii = np.array(self.radii)

    def distances(self, q, points):
        points = np.asarray(points).reshape(-1, 3)
        qmap = dict(zip(JOINTS, q))
        frames = {link: fk_transform(chain, qmap) for link, chain in self.chains.items()}
        result = []
        for p in self.primitives:
            tf = frames[p['link']]@p['origin']
            local = (points-tf[:3, 3])@tf[:3, :3]
            result.append(exact_sdf(p['type'], local, p['geometry']))
        return np.asarray(result).T

    def minimum(self, q, points):
        d = self.distances(q, points)
        return d.min(axis=1), d.argmin(axis=1)

    def box_face(self, q, point, primitive_index, face=None):
        p=self.primitives[primitive_index]
        if p['type']!='box': raise ValueError('Entry-face diagnostic requires a box')
        tf=fk_transform(self.chains[p['link']],dict(zip(JOINTS,q)))@p['origin']
        local=tf[:3,:3].T@(point-tf[:3,3]); half=np.array(p['geometry'].size)/2
        if face is None:
            axis=int(np.argmax(np.abs(local)-half)); sign=1. if local[axis]>=0 else -1.
            face=(axis,sign)
        axis,sign=face
        return float(sign*local[axis]-half[axis]),face

    def sweep(self, msg):
        """Mirror computeContinuousTrajectory: sample midpoint, report entry time."""
        result = []
        for k, (a, b) in enumerate(zip(msg.points, msg.points[1:])):
            qa, qb = np.array(a.positions), np.array(b.positions)
            n = max(1, math.ceil(float(np.max(self.radii@np.abs(qb-qa)))/.01))
            ta, tb = a.time_from_start.to_sec(), b.time_from_start.to_sec()
            for i in range(n):
                result.append(dict(segment=k, subdivision=i, subdivisions=n,
                    entry_s=ta+i/n*(tb-ta), sample_s=ta+(i+.5)/n*(tb-ta),
                    q=(qa+(i+.5)/n*(qb-qa)).tolist()))
        return result


def continuity_methods():
    # Run the production interpolation/braking methods without __init__/ROS graph.
    sys.path.append('/usr/lib/python3/dist-packages')
    import rospy
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    names = {'_duration', '_lerp_array', '_interpolate_point', '_point_at_time',
             '_position_shift_inf', '_repair_prefix_with_braking_tail'}
    source = ast.parse(CONTINUITY.read_text())
    cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    module = ast.Module(body=[ast.ClassDef(name='ReplayContinuity', bases=[], keywords=[],
                                         body=methods, decorator_list=[])], type_ignores=[])
    # Clock-only stand-in; trajectory/message math remains production code.
    clock = SimpleNamespace(Duration=rospy.Duration,
        Time=SimpleNamespace(now=lambda: rospy.Time.from_sec(7.826)),
        logerr_throttle=lambda *a: None, logwarn_throttle=lambda *a: None)
    ns = dict(copy=copy, math=math, rospy=clock, JointTrajectory=JointTrajectory,
              JointTrajectoryPoint=JointTrajectoryPoint)
    exec(compile(ast.fix_missing_locations(module), str(CONTINUITY), 'exec'), ns)
    obj = ns['ReplayContinuity']()
    obj.repair_execution_prefix_s = .25
    obj.repair_acceleration_limits = [3., 3., 4., 4., 6., 6., 6.]
    obj.repair_brake_dt_s = .05
    obj.repair_brake_max_steps = 20
    obj.repair_hold_s = .1
    return obj, rospy, JointTrajectory, JointTrajectoryPoint


def trajectory(q, u, classes):
    _, rospy, JT, Point = classes
    msg = JT(); msg.joint_names = JOINTS
    for k in range(21):
        p = Point(); p.positions = q[k].tolist()
        p.velocities = u[k].tolist() if k < 20 else [0.]*7
        p.time_from_start = rospy.Duration(k*.05)
        msg.points.append(p)
    return msg


def assemble(qbar, rows, frontier, previous, limits, radius):
    # Exact production objective in this fixture: repair_task_scale=0,
    # frontier@k2 weight=3000, q_vis scale=0, u tracking=0.01, smooth=1,
    # boundary handoff=1, no acceleration rows/user slack, regularization=1e-8.
    n = 280; nq = 140
    P = np.eye(n)*1e-8; c = np.zeros(n)
    P[7:14, 7:14] += np.eye(7)*6000
    c[7:14] = -6000*frontier
    for k in range(20):
        for j in range(7):
            i = nq+k*7+j; P[i, i] += .02
            if k == 0:
                P[i, i] += 4.; c[i] -= 4.*previous[j]
            else:
                P[i, i] += 2.; P[i-7, i-7] += 2.
                P[i, i-7] -= 2.; P[i-7, i] -= 2.
    A = np.zeros((nq, n)); b = np.zeros(nq); b[:7] = qbar[0]
    for k in range(20):
        for j in range(7):
            i = k*7+j; A[i, i] = 1.; A[i, nq+i] = -.05
            if k: A[i, i-7] = -1.
    lo = np.tile(limits[:, 0]+.01, (20, 1))
    hi = np.tile(limits[:, 1]-.01, (20, 1))
    xl = np.r_[np.maximum(qbar[1:]-radius, lo).ravel(), -np.tile([2.,2.,2.,2.,2.5,2.5,2.5],20)]
    xu = np.r_[np.minimum(qbar[1:]+radius, hi).ravel(), np.tile([2.,2.,2.,2.,2.5,2.5,2.5],20)]
    G = np.zeros((len(rows), n)); hl = []
    for i, r in enumerate(rows):
        k = int(r['timestep']); g = np.array(json.loads(r['gradient']))
        qlin = np.array(json.loads(r['q_linearization']))
        G[i, (k-1)*7:k*7] = g
        hl.append(.02-float(r['distance'])+g@qlin)
    return dict(P=qp_matrix(P), c=c.tolist(), A=qp_matrix(A), b=b.tolist(), G=qp_matrix(G),
                h_l=hl, h_u=[1e30]*len(rows), x_l=xl.tolist(), x_u=xu.tolist(),
                max_iter=300, eps_abs=1e-5, eps_rel=1e-5)


def unpack_matrix(m):
    v = np.array(m['coo']).reshape(-1,3)
    return sparse.coo_matrix((v[:,2], (v[:,0].astype(int), v[:,1].astype(int))),
                            shape=m['shape']).toarray()


def residuals(problem, x):
    a = unpack_matrix(problem['A'])@x-np.array(problem['b'])
    gx = unpack_matrix(problem['G'])@x
    return dict(equality_inf=float(np.max(np.abs(a))),
        inequality_violation=max(0., float(np.max(np.array(problem['h_l'])-gx)),
                                 float(np.max(gx-np.array(problem['h_u'])))),
        box_violation=max(0.,float(np.max(np.array(problem['x_l'])-x)),
                          float(np.max(x-np.array(problem['x_u'])))))


def witness_ab(problem, baseline, q0, delta, geom, point, classes, binary, out):
    """Counterfactual keeps original P/c/A/b/G/boxes and PIQP settings.

    B adds exact point/body supporting planes of the constructed executable.
    Four offline refinement solves are predeclared, not a runtime policy.
    Map confidence and full VBC visibility histories are unavailable.
    """
    def executable(x):
        q = np.vstack((q0, np.array(x[:140]).reshape(20,7)))+delta
        u = np.array(x[140:]).reshape(20,7)
        return classes[0]._repair_prefix_with_braking_tail(trajectory(q,u,classes))

    def audit(x):
        msg = executable(x); sweep = geom.sweep(msg)
        samples = [dict(s, distance_m=float(geom.minimum(s['q'],[point])[0][0])) for s in sweep]
        raw_q = np.vstack((q0,np.array(x[:140]).reshape(20,7)))
        raw_d = [float(geom.minimum(q,[point])[0][0]) for q in raw_q]
        worst = min(samples,key=lambda r:r['distance_m'])
        return dict(witness_min_m=worst['distance_m'], witness_unsafe_samples=sum(s['distance_m']<0 for s in samples),
                    worst=worst, samples=samples, raw_knot_min_m=min(raw_d),
                    original_qp_residuals=residuals(problem,x),
                    executable_duration_s=classes[0]._duration(msg))

    def linearized_row(x, sample_s, face_spec=None):
        msg=executable(x); q=np.array(classes[0]._point_at_time(msg,sample_s).positions)
        def value(q):
            if face_spec is not None:
                return geom.box_face(q,point,face_spec[0],face_spec[1])[0]
            return float(geom.minimum(q,[point])[0][0])
        d=value(q); eps=1e-6
        grad=np.empty(7)
        for j in range(7):
            e=np.eye(7)[j]*eps
            grad[j]=(value(q+e)-value(q-e))/(2*eps)
        # Differentiate production prefix/braking construction, including u.
        jac=np.empty((7,280))
        for j in range(280):
            xp=x.copy(); xm=x.copy(); xp[j]+=eps; xm[j]-=eps
            qp=np.array(classes[0]._point_at_time(executable(xp),sample_s).positions)
            qm=np.array(classes[0]._point_at_time(executable(xm),sample_s).positions)
            jac[:,j]=(qp-qm)/(2*eps)
        g=grad@jac
        scale=max(float(np.linalg.norm(g)),1e-12)
        # Zero exact margin matches exp70. Normalize only for numerical scaling.
        return g/scale, (-d+g@x)/scale, dict(sample_s=sample_s,q=q.tolist(),distance_m=d,
            primitive=geom.primitives[face_spec[0] if face_spec is not None else int(geom.minimum(q,[point])[1][0])]['name'],
            fixed_entry_face=None if face_spec is None else list(face_spec[1]),
            exact_q_gradient=grad.tolist(),row_scale=scale)

    x=np.array(baseline['x']); a=audit(x); history=[]
    cuts=[]; lower=[]; metadata=[]
    # Begin with the most penetrating sampled pose and add newly exposed poses.
    for iteration in range(4):
        before=audit(x)
        g,h,meta=linearized_row(x,before['worst']['sample_s'])
        cuts.append(g); lower.append(h); metadata.append(meta)
        bp=copy.deepcopy(problem)
        bp['G']=qp_matrix(np.vstack((unpack_matrix(problem['G']),cuts)))
        bp['h_l']=problem['h_l']+lower; bp['h_u']=problem['h_u']+[1e30]*len(cuts)
        answer=solve(binary,bp)
        row=dict(iteration=iteration+1,solver={k:v for k,v in answer.items() if k!='x'},new_cut=meta)
        save(out/f'b{iteration+1}_problem.json',bp)
        save(out/f'b{iteration+1}_solution.json',answer)
        if answer['status']!='solved':
            history.append(row); break
        x=np.array(answer['x']); row['audit']=audit(x)
        row['augmented_qp_residuals']=residuals(bp,x)
        history.append(row)
        print('B',iteration+1,'witness min',row['audit']['witness_min_m'],flush=True)
        if row['audit']['witness_min_m']>=0: break
    # Follow-up after the nearest-surface variant: keep the separating box face
    # from the last safe sample before FIRST entry. This avoids selecting the
    # opposite exit face after the witness is already inside a thin plate.
    x0=np.array(baseline['x']); samples=geom.sweep(executable(x0))
    all_d=np.array([geom.distances(s['q'],[point])[0] for s in samples])
    bad_primitives=np.flatnonzero(all_d.min(axis=0)<0)
    face_specs=[]
    for pi in bad_primitives:
        first=int(np.flatnonzero(all_d[:,pi]<0)[0])
        prior_q=(samples[first-1]['q'] if first else executable(x0).points[0].positions)
        distance,face=geom.box_face(prior_q,point,int(pi))
        if distance<0: raise ValueError('No safe entry side for counterfactual')
        face_specs.append((int(pi),face))
    cuts=[];lower=[];face_meta=[]
    times=sorted(set([s['sample_s'] for s in samples]+[p.time_from_start.to_sec() for p in executable(x0).points]))
    # q(0) is fixed; check it directly instead of adding a zero Jacobian row.
    for spec in face_specs:
        for t in times:
            if t<1e-9: continue
            g,h,meta=linearized_row(x0,t,spec)
            cuts.append(g);lower.append(h);face_meta.append(meta)
    cp=copy.deepcopy(problem)
    cp['G']=qp_matrix(np.vstack((unpack_matrix(problem['G']),cuts)))
    cp['h_l']=problem['h_l']+lower;cp['h_u']=problem['h_u']+[1e30]*len(cuts)
    cr=solve(binary,cp)
    c_report=dict(solver={k:v for k,v in cr.items() if k!='x'},
                  constraint_count=len(cuts),faces=[dict(primitive=geom.primitives[i]['name'],face=list(f)) for i,f in face_specs],
                  variant='FOLLOWUP_FIXED_ENTRY_FACE_AFTER_B',
                  note='Conservative separating-face linearizations, not exact configuration-space convex constraints.')
    save(out/'c_entry_face_problem.json',cp);save(out/'c_entry_face_solution.json',cr)
    save(out/'c_entry_face_rows.json',face_meta)
    if cr['status']=='solved':
        cx=np.array(cr['x']);c_report['audit']=audit(cx)
        c_report['augmented_qp_residuals']=residuals(cp,cx)
        x=cx
        print('C entry face witness min',c_report['audit']['witness_min_m'],flush=True)
        c_report['refinements']=[]
        for iteration in range(2,5):
            if c_report['audit']['witness_min_m']>=0: break
            t=c_report['audit']['worst']['sample_s']
            for spec in face_specs:
                g,h,meta=linearized_row(cx,t,spec)
                cuts.append(g);lower.append(h);face_meta.append(meta)
            cp=copy.deepcopy(problem)
            cp['G']=qp_matrix(np.vstack((unpack_matrix(problem['G']),cuts)))
            cp['h_l']=problem['h_l']+lower;cp['h_u']=problem['h_u']+[1e30]*len(cuts)
            cr=solve(binary,cp)
            item=dict(iteration=iteration,solver={k:v for k,v in cr.items() if k!='x'})
            save(out/f'c{iteration}_entry_face_problem.json',cp)
            save(out/f'c{iteration}_entry_face_solution.json',cr)
            if cr['status']=='solved':
                cx=np.array(cr['x']);item['audit']=audit(cx)
                item['augmented_qp_residuals']=residuals(cp,cx)
                c_report['audit']=item['audit'];x=cx
                print('C entry face',iteration,'witness min',item['audit']['witness_min_m'],flush=True)
            c_report['refinements'].append(item)
            if cr['status']!='solved': break
    # Zero-margin C converges to contact. A distinct, stricter 1 mm local
    # optimization guard probes whether positive clearance is feasible; the
    # original gate and ALL original learned margins remain unchanged.
    gp=copy.deepcopy(cp)
    gp['h_l']=problem['h_l']+[h+.001/m['row_scale'] for h,m in zip(lower,face_meta)]
    gr=solve(binary,gp)
    guard=dict(solver={k:v for k,v in gr.items() if k!='x'},optimization_guard_m=.001,
               original_gate_margin_m=0.,variant='FOLLOWUP_POSITIVE_CLEARANCE')
    save(out/'c_guard1mm_problem.json',gp);save(out/'c_guard1mm_solution.json',gr)
    if gr['status']=='solved':
        gx=np.array(gr['x']);guard['audit']=audit(gx)
        guard['augmented_qp_residuals']=residuals(gp,gx)
        x=gx
        print('C guard1mm witness min',guard['audit']['witness_min_m'],flush=True)
    return dict(A=a,B=history,C=c_report,C_guard1mm=guard,exact_constraint_margin_m=0.,max_refinement_solves=4,
                same_original_objective_constraints_trust_and_piqp_settings=True,
                full_original_gate_status='NOT_RUN_MISSING_FROZEN_MAP_AND_FINAL_BATCH',
                note='A/B local witness feasibility only; no executable safety or task-success claim.'), x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--solver', type=Path, required=True)
    ap.add_argument('--ab', action='store_true')
    args = ap.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    run = ROOT/'outputs/c5_5_vbc_gcdf_regime'/RUN_ID/'run'
    log = ROOT/'logs/c5_5_vbc_gcdf_regime'/RUN_ID/'run/controlled.log'
    events = strings(run/'local_planner_summary.csv')
    published = next(r for r in events if r.get('event')=='candidate_published' and r['plan_seq']=='11')
    query_stamp = int(published['query_stamp_ns'])
    raw_stamp = '7822000000'; audited_stamp = '7826000000'
    outcome = next(r for r in strings(run/'verification_outcome.csv')
                   if r.get('audited_trajectory_stamp_ns') == audited_stamp)
    assert outcome['raw_candidate_stamp_ns'] == raw_stamp and outcome['safety_gate']=='vbc'
    evidence = json.loads(outcome['vbc_evidence'])
    assert evidence['trajectory_stamp_ns'] == audited_stamp
    identity = next(r for r in strings(run/'observation_candidate_identity.csv')
                    if r.get('raw_candidate_stamp_ns') == raw_stamp)
    assert identity['observation_token'] == published['observation_token']
    start = next(r for r in events if r.get('event')=='plan_started' and r['plan_seq']=='11')
    query = next(r for r in numeric(run/'local_gcdf_query.csv')
                 if int(r['field.header.stamp']) == query_stamp)
    qbar = np.array([values(query, f'field.points{k}.positions') for k in range(21)])
    ubar = np.array([values(query, f'field.points{k}.velocities') for k in range(20)])
    point = np.array(evidence['violations'][0]['point'])
    selector = [r for r in strings(run/'local_witness_selector_diagnostics.csv')
                if r.get('query_stamp_ns')==str(query_stamp) and r.get('status')=='evaluated_unknown']
    # In plan 11 all selected rows are the single active obligation, not the rejected point.
    rows = sorted(selector, key=lambda r: int(r['timestep']))
    assert len(rows)==20 and [int(r['timestep']) for r in rows]==list(range(1,21))
    assert published['cdf_rows']==published['obligation_cdf_rows']=='20'
    assert all(solver_point_key(json.loads(r['point'])) != solver_point_key(point) for r in rows)
    frontiers = [r for r in numeric(run/'visibility_frontier_target.csv')
                 if int(r['%time']) <= round(start['ros_s']*1e9)]
    frontier = values(frontiers[-1], 'field.data', 10)[3:]
    commands = [r for r in numeric(run/'actuator_command.csv')
                if int(r['%time']) <= round(start['ros_s']*1e9)]
    previous = values(commands[-1], 'field.data')
    geom = Geometry()
    limits = np.array([[geom.robot.joint_map[j].limit.lower, geom.robot.joint_map[j].limit.upper]
                       for j in JOINTS])
    config_text = log.read_text()
    expected = {'repair_task_tracking_scale':'0.0', 'u_reference_tracking_enabled':'False',
                'u_tracking_weight':'0.01', 'u_smooth_weight':'1.0', 'handoff_velocity_weight':'1.0',
                'visibility_waypoint_weight':'3000.0', 'enforce_acceleration_constraints':'False'}
    for name, v in expected.items():
        assert f'/local_sparse_scp_planner_node/local_planner/{name}: {v}\n' in config_text, name
    problem = assemble(qbar, rows, frontier, previous, limits, float(published['trust_q_inf']))
    baseline = solve(args.solver, problem)
    assert baseline['status']=='solved', baseline
    x = np.array(baseline['x']); q = np.vstack((qbar[0], x[:140].reshape(20,7)))
    u = x[140:].reshape(20,7)
    q_error = float(np.max(np.abs(q-qbar)))
    u_error = float(np.max(np.abs(u-ubar)))
    classes = continuity_methods()
    raw = trajectory(q, u, classes)
    # Approximate raw candidate via validated fixed-point reassembly, never a captured payload.
    joints = numeric(run/'joint_states.csv')
    measured_options = [r for r in joints if 7810000000 <= int(r['field.header.stamp']) <= 7826000000]
    check = []
    for r in measured_options:
        shifted = copy.deepcopy(raw); delta = values(r, 'field.position')-q[0]
        for p in shifted.points: p.positions = (np.array(p.positions)+delta).tolist()
        braked = classes[0]._repair_prefix_with_braking_tail(shifted)
        sweep = geom.sweep(braked)
        aligned = [s for s in sweep if s['segment']==evidence['violations'][0]['sweep_original_k']
                   and abs(s['entry_s']-evidence['violations'][0]['sweep_s'])<1e-8]
        if len(aligned)!=1: raise ValueError('Ambiguous/missing VBC interval sample')
        ds = geom.distances(aligned[0]['q'], [point])[0]
        pi = next(i for i,p in enumerate(geom.primitives)
                  if p['name']==evidence['violations'][0]['source_collision_name'])
        check.append(dict(joint_stamp_ns=int(r['field.header.stamp']),
                          shift_inf=float(np.max(np.abs(delta))),
                          sweep=aligned[0],
                          witness_sdf_m=float(ds[pi]),
                          error_m=float(ds[pi]-evidence['violations'][0]['primitive_signed_distance_m'])))
    sources = [run/n for n in ['local_planner_summary.csv','local_gcdf_query.csv',
        'verification_outcome.csv','observation_candidate_identity.csv','visibility_frontier_target.csv',
        'actuator_command.csv','joint_states.csv','local_witness_selector_diagnostics.csv']]
    sources += [log, URDF_PATH, CONTINUITY, Path(__file__), args.solver,
                ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp']
    report = dict(status='REASSEMBLY_DIAGNOSTIC', diagnostic_only=True,
                  plan_seq=11, raw_stamp_ns=raw_stamp, audited_stamp_ns=audited_stamp,
                  query_stamp_ns=query_stamp, observation_token=identity['observation_token'],
                  selected_point=json.loads(rows[0]['point']), rejected_point=point.tolist(),
                  candidate_published=published, outcome=outcome, vbc_evidence=evidence,
                  previous_command=previous.tolist(), previous_command_receipt_ns=commands[-1]['%time'],
                  frontier=frontier.tolist(), frontier_receipt_ns=frontiers[-1]['%time'],
                  original_candidate_payload_available=False, original_qp_snapshot_available=False,
                  reassembly_q_error_inf=q_error, reassembly_u_error_inf=u_error,
                  baseline_solver={k:v for k,v in baseline.items() if k!='x'},
                  measured_state_candidates=check,
                  source_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in sources},
                  limitations=['Reassembled final QP, not an original matrix snapshot.',
                    'Full confidence-map snapshot is absent; full original VBC gate replay unavailable.',
                    'Query point index is not executable trajectory point index.'])
    save(out/'baseline_problem.json', problem)
    save(out/'baseline_solution.json', baseline)
    save(out/'alignment.json', report)
    save(out/'selected_rows.json', rows)
    print(json.dumps(dict(q_error_inf=q_error, u_error_inf=u_error, measured=check), indent=2))
    if args.ab:
        if q_error>1e-8 or u_error>1e-8:
            raise ValueError('Baseline not reproduced: counterfactual blocked')
        # Identify measured state using both independently recorded rebase shift
        # and the exact midpoint geometry; do not select a state solely by time.
        matches=[r for r in check if abs(r['shift_inf']-float(outcome['handoff_rebase_shift_inf']))<5e-7
                 and abs(r['error_m'])<1e-8]
        if len(matches)!=1: raise ValueError('Cannot uniquely reproduce executed geometry')
        m=next(r for r in measured_options if int(r['field.header.stamp'])==matches[0]['joint_stamp_ns'])
        delta=values(m,'field.position')-q[0]
        ab, bx=witness_ab(problem,baseline,q[0],delta,geom,point,classes,args.solver,out)
        save(out/'ab_result.json',ab)
        # Compare learned values at the actual audit midpoint, not raw query k6.
        import torch
        from collision_cdf_model import CollisionCDF
        torch.set_num_threads(2)
        checkpoint=ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'
        net=CollisionCDF(str(checkpoint),device='cpu',activation='gelu')
        auditq=np.array(matches[0]['sweep']['q'])
        positions=np.array([point, json.loads(rows[0]['point'])])
        qt=np.repeat(auditq[None,:],2,axis=0)
        ds,gs=net.pair_distance_and_gradient(torch.tensor(positions,dtype=torch.float32),
                                           torch.tensor(qt,dtype=torch.float32))
        network=dict(checkpoint_sha256=sha(checkpoint),device='cpu',audit_midpoint_q=auditq.tolist(),
                     points=positions.tolist(),distance=ds.tolist(),gradient=gs.tolist(),
                     exact_minimum_m=geom.minimum(auditq,positions)[0].tolist())
        save(out/'aligned_network.json',network)
        # Locate the subsequent point-scoped feedback and audit margin routing.
        later=next(r for r in events if r.get('event')=='candidate_published' and r['plan_seq']=='13')
        later_rows=[r for r in strings(run/'local_witness_selector_diagnostics.csv')
                    if r.get('query_stamp_ns')==later['query_stamp_ns'] and r.get('status')=='evaluated_unknown'
                    and solver_point_key(json.loads(r['point']))==solver_point_key(point)]
        route=dict(status='SOURCE_AND_LOG_CONFIRMED',plan_seq=13,query_stamp_ns=later['query_stamp_ns'],
            total_selected_rows=int(later['cdf_rows']),classified_as_obligation_rows=int(later['obligation_cdf_rows']),
            repair_witness_count=int(later['repair_witness_count']),witness_rows=len(later_rows),
            logged_feedback_margin=.12,source_routed_margin=float(later['visibility_obligation_cdf_margin']),
            witness_distance_min=min(float(r['distance']) for r in later_rows),
            witness_distance_max=max(float(r['distance']) for r in later_rows),
            cause='repair_unknown_witnesses_ are appended to obligation_points; obligation_row ternary takes precedence over repair_witness_margin_for_row.',
            limitation='No original h_l snapshot; routed margin follows the current source, corroborated by the recorded obligation-row count. Both .02 and .12 are below the logged distances, so this defect alone does not explain all rejection.')
        save(out/'margin_routing.json',route)
        # Nonlinear re-query of every ORIGINAL selected learned row on the
        # modified raw trajectory. These are additional checks, not full GCDF.
        original_points=np.array([json.loads(r['point']) for r in rows])
        candidate_q=bx[:140].reshape(20,7)
        nd,ng=net.pair_distance_and_gradient(torch.tensor(original_points,dtype=torch.float32),
                                            torch.tensor(candidate_q,dtype=torch.float32))
        regress=dict(original_selected_rows=20,learned_requery_min=float(nd.min()),
                     original_learned_margin=.02,learned_requery_all_pass=bool(torch.all(nd>=.02)))
        # Check adaptive and 1 ms interpolation against all 21 primitives and
        # a fixed spatial neighborhood. Neighbor occupancy is UNKNOWN here.
        neighbors=np.array([point+[i*.025,j*.025,k*.025] for i in [-1,0,1]
                             for j in [-1,0,1] for k in [-1,0,1]])
        for name,sx in [('A',np.array(baseline['x'])),('C_guard1mm',bx)]:
            qx=np.vstack((q[0],sx[:140].reshape(20,7)))+delta
            ux=sx[140:].reshape(20,7)
            msg=classes[0]._repair_prefix_with_braking_tail(trajectory(qx,ux,classes))
            duration=classes[0]._duration(msg)
            times=np.linspace(0,duration,math.ceil(duration/.001)+1)
            minima=[]
            for t in times:
                qq=classes[0]._point_at_time(msg,t).positions
                minima.append(geom.minimum(qq,neighbors)[0])
            minima=np.array(minima)
            i,j=np.unravel_index(np.argmin(minima),minima.shape)
            P=unpack_matrix(problem['P']);c=np.array(problem['c'])
            regress[name]=dict(dense_samples=len(times),dense_witness_min_m=float(minima[:,13].min()),
                neighbor_grid_spacing_m=.025,neighbor_count=len(neighbors),
                geometrically_intersected_neighbor_count=int(np.sum(minima.min(axis=0)<0)),
                neighbor_min_m=float(minima[i,j]),neighbor_worst_point=neighbors[j].tolist(),
                raw_joint_step_inf=float(np.max(np.abs(sx[:140]-np.array(baseline['x'])[:140]))),
                frontier_q_error_inf=float(np.max(np.abs(sx[7:14]-frontier))),
                qp_objective_without_constant=float(.5*sx@P@sx+c@sx),
                note='Neighbor intersections do not establish new VBC violations without map confidence/visibility history.')
        save(out/'nonlinear_checks.json',regress)


if __name__=='__main__':
    main()
