#!/usr/bin/env python3
"""Offline evidence audit of CASE006 NORMAL stagnation and return-to-NORMAL gates."""
import argparse
import bisect
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from urdf_parser_py.urdf import URDF
from diagnose_phase_e_case006_o6_recovery import strings
from evaluate_phase_d_run import JOINTS, joint_states, load_fk_helper, fk_T, pose_err, quat_rot

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-id', default='r1_case006_sweep_fix_20261007')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    name = 'phase_e_goal_006_' + args.batch_id
    run = ROOT/'outputs/c5_5_vbc_gcdf_regime'/name/'run'
    batch = ROOT/'outputs/phase_e_r1_mainline_30'/args.batch_id
    inputs = set()

    def read_json(p):
        inputs.add(p)
        return json.loads(p.read_text())

    summary = read_json(batch/'case_summaries/phase_e_goal_006.json')
    window = summary['benchmark_window']
    begin, end = window['start_ros_ns'], window['end_ros_ns']

    def read_strings(name):
        p = run/(name+'.csv'); inputs.add(p)
        return [r for r in strings(p) if begin <= r['receipt_ns'] <= end]

    def numeric(name):
        p = run/(name+'.csv'); inputs.add(p)
        with p.open() as f:
            return list(csv.DictReader(f))

    def trajectory(row):
        q = []
        for k in range(1000):
            if not row.get(f'field.points{k}.positions0'):
                break
            q.append([float(row[f'field.points{k}.positions{j}']) for j in range(7)])
        return np.array(q)

    def previous(rr, t):
        selected = [r for r in rr if r['receipt_ns'] <= t]
        return selected[-1] if selected else None

    planner = read_strings('local_planner_summary')
    verification = read_strings('verification_outcome')
    regime = read_strings('regime_summary')
    acq = read_strings('visibility_acquisition_summary')
    progress = read_strings('nominal_progress_summary')
    witnesses = read_strings('local_witness_selector_diagnostics')
    queries = {r['field.header.stamp']:r for r in numeric('local_gcdf_query')}
    commits = {r['field.header.stamp']:r for r in numeric('committed_trajectory')}
    joint_path = run/'joint_states.csv'; inputs.add(joint_path)
    samples = [(t,q) for t,q in joint_states(str(joint_path), JOINTS, True) if begin <= t <= end]
    ts = np.array([s[0] for s in samples], dtype=np.int64)
    qs = np.array([s[1] for s in samples])
    urdf = ROOT/'src/arm_description/urdf/Arm.urdf'; inputs.add(urdf)
    helper = load_fk_helper(str(ROOT))
    robot = URDF.from_xml_file(str(urdf))
    chain = helper.find_chain_joints(robot, 'base_link', 'EE_link')
    goal = np.array(summary['goal_position'])
    rot = quat_rot(summary['goal_orientation_xyzw'])
    errors = np.array([pose_err(fk_T(helper,chain,dict(zip(JOINTS,q))),goal,rot) for q in qs])
    assert abs(errors[-1,0]-summary['final_position_error_m']) < 1e-10

    def measured(t):
        i = max(0,bisect.bisect_right(ts,t)-1)
        return dict(sample_ros_s=float(ts[i]/1e9), q=qs[i].tolist(),
                    position_error_m=float(errors[i,0]), orientation_error_rad=float(errors[i,1]))

    spans = []
    for lo,hi in [(14.521,19.393),(19.393,27.402),(27.412,29.171),(29.171,31.021),
                  (31.021,31.221),(31.221,38.571),(38.571,39.571),(39.571,39.821),(39.821,41.996)]:
        mask = (ts >= round(lo*1e9)) & (ts <= round(hi*1e9))
        q = qs[mask]
        vv = [r for r in verification if lo*1e9 <= r['receipt_ns'] < hi*1e9]
        pp = [r for r in planner if lo*1e9 <= r['receipt_ns'] < hi*1e9]
        spans.append(dict(start_ros_s=lo,end_ros_s=hi,start=measured(round(lo*1e9)),end=measured(round(hi*1e9)),
            measured_range_inf_rad=float(np.ptp(q,axis=0).max()),
            verification_counts=dict(Counter(r['result'] for r in vv)),
            solve_counts=dict(Counter(r.get('solved') for r in pp if r.get('event')=='scp_solved')),
            normal_reseed_publications=sum(r.get('event')=='candidate_published' and r.get('init')=='normal_conflict_measured_reseed' for r in pp)))

    # This query and raw/execution chain are stated in the measured-stall log.
    stalled = next(r for r in verification if r['raw_candidate_stamp_ns']=='27084000000')
    assert stalled['result']=='safe' and stalled['execution_stamp_ns']=='27166000000'
    solved = next(r for r in planner if r.get('plan_seq')=='80' and r.get('event')=='scp_solved' and r.get('solved')=='1')
    assert solved['query_stamp_ns']=='27071000000' and solved['cdf_rows']=='60'
    ww = [r for r in witnesses if r.get('query_stamp_ns')=='27071000000']
    assert len(ww)==60 and all(r['status']=='evaluated_unknown' for r in ww)
    seed = trajectory(queries['27049000000'])
    hold = trajectory(queries['27071000000'])
    executed = trajectory(commits[stalled['execution_stamp_ns']])
    desired = seed[-1]-hold[0]
    constraints = []
    for r in ww:
        grad = np.array(json.loads(r['gradient']))
        qlin = np.array(json.loads(r['q_linearization']))
        d = float(r['distance']); k = int(r['timestep'])
        dot = float(grad@desired)
        constraints.append(dict(point=json.loads(r['point']),timestep=k,distance=d,
            gradient_dot_desired_step=dot,
            linear_clearance_at_hold=float(d+grad@(hold[0]-qlin)),
            linear_clearance_at_committed_knot=float(d+grad@(executed[k]-qlin)),
            allowed_desired_fraction_at_hold=max(0,float((d+grad@(hold[0]-qlin)-.05)/(-dot))) if dot<0 else None))
    point_groups = {}
    for r in constraints:
        key = str([round(x,4) for x in r['point']])
        point_groups.setdefault(key,[]).append(r)
    point_stats = {k:dict(rows=len(v), min_d=min(r['distance'] for r in v),
        max_d=max(r['distance'] for r in v), min_gradient_dot_desired=min(r['gradient_dot_desired_step'] for r in v),
        min_linear_committed=min(r['linear_clearance_at_committed_knot'] for r in v),
        max_linear_committed=max(r['linear_clearance_at_committed_knot'] for r in v)) for k,v in point_groups.items()}
    # Report certified shape for each late NORMAL execution without assuming raw == audited.
    late_commits = []
    for r in verification:
        if not (19.393e9 <= r['receipt_ns'] <= 27.402e9 and r['result']=='safe'): continue
        q = trajectory(commits[r['execution_stamp_ns']])
        late_commits.append(dict(seq=int(r['seq']),ros_s=r['receipt_ns']/1e9,execution_stamp_ns=r['execution_stamp_ns'],
            committed_span_inf_rad=float(np.max(np.abs(q-q[0])))))
    selected = []
    for r in verification:
        if r['seq'] not in ('47','48','49','50','51','100','101','102','103','104'): continue
        e = json.loads(r['vbc_evidence'])
        assert e['trajectory_stamp_ns']==r['audited_trajectory_stamp_ns']
        item = {k:r[k] for k in ['seq','result','verification_view','raw_candidate_stamp_ns','audited_trajectory_stamp_ns',
            'execution_stamp_ns','constructed_executable_duration_s','probe_effective_prefix_s']}
        item.update(ros_s=r['receipt_ns']/1e9,evidence=e)
        item['acquisition_before']=previous(acq,r['receipt_ns'])
        selected.append(item)

    logpath = ROOT/'logs/c5_5_vbc_gcdf_regime'/name/'run/controlled.log'; inputs.add(logpath)
    log_events = []
    for i,line in enumerate(logpath.read_text(errors='replace').splitlines(),1):
        if any(tag in line for tag in ('classification=NORMAL_ACTIVE','qp_stall_existing_visibility_obligation','NORMAL_BOX_RESEED')):
            log_events.append(dict(line=i,text=line))
    meta = read_json(batch/'metadata.json')
    sources = [ROOT/p for p in ['src/arm_description/urdf/Arm.urdf',
        'src/care_confidence_map/include/care_confidence_map/vbc_primitive_model.hpp',
        'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',
        'src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/normal_active_no_progress.hpp',
        'src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py',
        'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml']]
    inputs.update(sources)
    source_matches={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()==meta['source_hashes'][str(p.relative_to(ROOT))] for p in sources}
    assert all(source_matches.values())
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    obligations = {}
    for i in range(3,8):
        p, = (run/'projector_traces').glob(f'c46_obligation_{i:03d}_*.json')
        data=read_json(p)
        observations=[r for r in acq if r.get('active_obligation_id')==str(i)]
        obligations[str(i)] = dict(points=data['active_set_points_xyz'],
            first_active_ros_s=observations[0]['receipt_ns']/1e9 if observations else None,
            last_active_ros_s=observations[-1]['receipt_ns']/1e9 if observations else None)
    # Include projection/coverage samples at the same boundaries as the FK trace.
    boundaries = [dict(ros_s=t,nominal=previous(progress,round(t*1e9)),
        acquisition=previous(acq,round(t*1e9))) for t in [14.521,19.393,27.402,29.171,31.021,38.571,39.571,41.996]]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    result=dict(batch_id=args.batch_id,spans=spans,late_normal_commits=late_commits,
        commit_views=dict(Counter(r['verification_view'] for r in verification if r['result']=='safe')),
        boundary_snapshots=boundaries,obligations=obligations,
        stalled_plan=dict(plan_seq=80,query_stamp_ns='27071000000',raw_stamp_ns='27084000000',execution_stamp_ns='27166000000',
            desired_gap_inf_rad=float(np.max(np.abs(desired))),
            seed_span_inf_rad=float(np.max(np.abs(seed-seed[0]))),hold_seed_span_inf_rad=float(np.max(np.abs(hold-hold[0]))),
            committed_span_inf_rad=float(np.max(np.abs(executed-executed[0]))),
            point_stats=point_stats,constraints=constraints,verification=stalled,planner=solved),
        selected_verifications=selected,log_events=log_events,source_matches=source_matches,input_sha256=hashes)
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with (out/'measured_progress.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['ros_s','position_error_m','orientation_error_rad']+JOINTS)
        writer.writerows([[t/1e9,*e,*q] for t,e,q in zip(ts,errors,qs)])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    colors={'NORMAL':'#cfe7ff','REPAIR':'#ffddae','PROBE_NORMAL':'#c9eedb'}
    changes=[]
    for r in regime:
        if not changes or r['state']!=changes[-1]['state']: changes.append(r)
    for a,b in zip(changes,changes[1:]+[dict(receipt_ns=end)]):
        for ax in axes:ax.axvspan(a['receipt_ns']/1e9,b['receipt_ns']/1e9,color=colors[a['state']],alpha=.55)
    axes[0].plot(ts/1e9,errors[:,0],color='#163c63');axes[0].set_ylabel('Position error (m)')
    axes[1].plot(ts/1e9,errors[:,1],color='#163c63');axes[1].set_ylabel('Orientation error (rad)')
    anchor=qs[bisect.bisect_left(ts,19393000000)]
    axes[2].plot(ts/1e9,np.max(np.abs(qs-anchor),axis=1),color='#163c63')
    axes[2].set_ylabel('Joint distance from\n19.393 s pose (rad, inf)')
    axes[2].set_xlabel('ROS time (s)')
    for ax in axes:
        ax.axvline(27.402,color='#9d2222',linestyle='--');ax.grid(alpha=.2);ax.set_xlim(begin/1e9,end/1e9)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=c,label=s) for s,c in colors.items()],loc='upper right')
    fig.suptitle('CASE006: certified execution stagnation and repeated recovery (2026-10-07 run)',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'progress.png',dpi=160);fig.savefig(out/'progress.svg');plt.close(fig)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps(dict(spans=spans,late_normal_commits=late_commits,point_stats=point_stats,
        stalled_commit_span=result['stalled_plan']['committed_span_inf_rad'],source_matches=source_matches),indent=2))


if __name__=='__main__':
    main()
