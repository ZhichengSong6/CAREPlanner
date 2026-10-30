#!/usr/bin/env python3
"""Read-only exp72 recovery diagnosis and validated offline QP reconstruction."""
import argparse
import copy
import json
from pathlib import Path
from collections import defaultdict
import bisect
import re
import numpy as np
import diagnose_phase_e_case001_exact_witness as d
import evaluate_phase_d_run as ev

RID='case_001_safe_frontier_20260921_exp72'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--solver',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    run=d.ROOT/'outputs/c5_5_vbc_gcdf_regime'/RID/'run'
    events=d.strings(run/'local_planner_summary.csv')
    pubs=[r for r in events if r.get('event')=='candidate_published' and 33.365<=r['ros_s']<=67.187]
    starts={r['plan_seq']:r for r in events if r.get('event')=='plan_started'}
    acq=d.strings(run/'visibility_acquisition_summary.csv')
    js=ev.joint_states(str(run/'joint_states.csv'),ev.JOINTS)
    jt=np.array([t for t,q in js]);jq=np.array([q for t,q in js])
    measured=lambda t:jq[max(0,bisect.bisect_right(jt,t)-1)]
    observations=lambda t:[r for r in acq if r['ros_s']<=t][-1]
    triggers=[r for r in events if r.get('event')=='repair_safe_stall_frontier_retry' and r['ros_s']>=33.365 and r['ros_s']<=67.187]
    cycles=[];chosen={}
    for i,tr in enumerate(triggers):
        t=tr['ros_s'];stop=triggers[i+1]['ros_s'] if i+1<len(triggers) else 67.187
        candidates=[r for r in pubs if t<=r['ros_s']<stop]
        rec=[r for r in candidates if r.get('recovery_frontier_active')=='1']
        if not rec:continue
        exitrow=next(r for r in events if t<r['ros_s']<stop and r.get('safe_stall_active')=='0')
        firstnormal=next(r for r in candidates if r['ros_s']>=exitrow['ros_s'] and r.get('recovery_frontier_active')=='0')
        prev=[r for r in pubs if r['ros_s']<t][-1]
        last=candidates[-1];target=np.array(json.loads(rec[0]['recovery_frontier_target']))
        q0=measured(t);qe=measured(exitrow['ros_s']);qend=measured(stop-.01)
        samples=jq[(jt>=t)&(jt<stop)]
        dv=target-q0;den=float(dv@dv)
        seen=[r for r in acq if t<=r['ros_s']<stop]
        cycle=dict(trigger=t,exit=exitrow['ros_s'],stop=stop,attempt=int(tr['safe_stall_attempts']),token=tr['observation_token'],
            before_plan=int(prev['plan_seq']),recovery_plans=[int(r['plan_seq']) for r in rec],exit_plan=int(firstnormal['plan_seq']),return_plan=int(last['plan_seq']),
            q_trigger=q0.tolist(),q_exit=qe.tolist(),q_end=qend.tolist(),target=target.tolist(),
            target_delta_inf=float(np.max(np.abs(dv))),exit_displacement_inf=float(np.max(np.abs(qe-q0))),
            exit_target_fraction=float((qe-q0)@dv/den),max_target_fraction=float(np.max((samples-q0)@dv/den)),
            end_target_fraction=float((qend-q0)@dv/den),net_displacement_inf=float(np.max(np.abs(qend-q0))),
            peak_displacement_inf=float(np.max(np.abs(samples-q0))),
            qvis_distance_start=float(observations(t)['active_q_distance_inf']),qvis_distance_end=float(observations(stop-.01)['active_q_distance_inf']),
            seen_fraction_max=max(float(r['active_seen_fraction']) for r in seen))
        cycles.append(cycle)
        for r in [prev,*rec,firstnormal,last]:chosen[int(r['plan_seq'])]=r
    # Include the first freshly reselected target, and a late settled plan.
    for n in (399,):
        if any(int(r['plan_seq'])==n for r in pubs):chosen[n]=next(r for r in pubs if int(r['plan_seq'])==n)
    queries={r['field.header.stamp']:r for r in d.numeric(run/'local_gcdf_query.csv')}
    wanted={r['query_stamp_ns'] for r in chosen.values()};rowsby=defaultdict(list)
    with (run/'local_witness_selector_diagnostics.csv').open() as f:
        for line in f:
            match=re.search(r'query_stamp_ns=(\d+)',line)
            if not match or match[1] not in wanted:continue
            row=ev.TOK.findall(line);row=dict(row)
            if row.get('status')=='evaluated_unknown':rowsby[match[1]].append(row)
    # High-rate ROS CSV subscribers can drop rows. The unthrottled original
    # publisher log carries the same immutable query+pair identity and values.
    indexed={stamp:{r['batch_pair']:r for r in rows} for stamp,rows in rowsby.items()}
    with (d.ROOT/'logs/c5_5_vbc_gcdf_regime'/RID/'run/controlled.log').open() as f:
        for line in f:
            if 'C5_WITNESS event=witness_reevaluation' not in line:continue
            if line.count('C5_WITNESS')!=1:continue
            match=re.search(r'query_stamp_ns=(\d+)',line)
            if not match or match[1] not in wanted:continue
            row=dict(ev.TOK.findall(line))
            if row.get('status')!='evaluated_unknown':continue
            if not all(k in row for k in ('point','gradient','distance','q_linearization','timestep','batch_pair')):continue
            try:
                for k,count in [('point',3),('gradient',7),('q_linearization',7)]:
                    assert len(json.loads(row[k]))==count
                float(row['distance']);int(row['batch_pair']);int(row['timestep'])
            except (ValueError,AssertionError):continue
            prior=indexed.setdefault(match[1],{}).get(row['batch_pair'])
            if prior:
                for k in ('point','gradient','distance','q_linearization','timestep'):
                    assert row[k]==prior[k],(match[1],row['batch_pair'],k,row[k],prior[k])
            indexed[match[1]][row['batch_pair']]=row
    rowsby={stamp:list(rows.values()) for stamp,rows in indexed.items()}
    frontiers=d.numeric(run/'visibility_frontier_target.csv');commands=d.numeric(run/'actuator_command.csv')
    vers=d.strings(run/'verification_outcome.csv');identities=d.strings(run/'observation_candidate_identity.csv')
    tokenbyraw={r['raw_candidate_stamp_ns']:r['observation_token'] for r in identities}
    commits={r['field.header.stamp']:r for r in d.numeric(run/'committed_trajectory.csv')}
    tracker=d.strings(run/'tracker_summary.csv');classes=d.continuity_methods();geom=d.Geometry()
    limits=np.array([[geom.robot.joint_map[j].limit.lower,geom.robot.joint_map[j].limit.upper] for j in d.JOINTS])
    reports={};failures=[]
    for n,pub in sorted(chosen.items()):
        start=starts[pub['plan_seq']];stamp=pub['query_stamp_ns'];query=queries[stamp]
        qbar=np.array([d.values(query,f'field.points{k}.positions') for k in range(21)])
        rows=sorted(rowsby[stamp],key=lambda r:int(r['batch_pair']))
        assert len(rows)==int(pub['cdf_rows'])==int(pub['obligation_cdf_rows']),(n,len(rows),pub['cdf_rows'])
        margins={}
        for v in vers:
            if v['ros_s']>=start['ros_s'] or v.get('result')!='unsafe' or v.get('safety_gate')!='vbc':continue
            pts={d.solver_point_key(z['point']) for z in json.loads(v['vbc_evidence'])['violations']}
            for key in pts:margins[key]=min(.5,margins.get(key,.02)+.1)
        for row in rows:row['effective_margin']=max(float(pub['visibility_obligation_cdf_margin']),margins.get(d.solver_point_key(json.loads(row['point'])),.02))
        fs=[r for r in frontiers if int(r['%time'])<round(start['ros_s']*1e9)][-5:]
        fs.sort(key=lambda r:abs(np.max(np.abs(d.values(r,'field.data',10)[3:]-qbar[0]))-float(start['frontier_target_shift_inf'])))
        fr=fs[0];straight=d.values(fr,'field.data',10)[3:]
        shift_error=abs(np.max(np.abs(straight-qbar[0]))-float(start['frontier_target_shift_inf']))
        target=np.array(json.loads(pub['recovery_frontier_target'])) if pub.get('recovery_frontier_active')=='1' else straight
        cmd=[r for r in commands if int(r['%time'])<round(start['ros_s']*1e9)][-1];previous=d.values(cmd,'field.data')
        problem=d.assemble(qbar,rows,target,previous,limits,float(pub['trust_q_inf']))
        problem['h_l']=[h+r['effective_margin']-.02 for h,r in zip(problem['h_l'],rows)];problem['include_duals']=True
        sol=d.solve(args.solver,problem);assert sol['status']=='solved',(n,sol['status'])
        x=np.array(sol['x']);q=np.vstack((qbar[0],x[:140].reshape(20,7)));u=x[140:].reshape(20,7)
        step=float(np.max(np.abs(q-qbar)));step_error=abs(step-float(pub['step_inf']))
        # Candidate association is bounded by the actual final query and same
        # observation token; unique committed trajectory SHAPE must validate it.
        options=[v for v in vers if int(v.get('raw_candidate_stamp_ns','0'))>=int(stamp) and int(v.get('raw_candidate_stamp_ns','0'))<=round((pub['ros_s']+.01)*1e9)
                 and tokenbyraw.get(v.get('raw_candidate_stamp_ns'))==pub['observation_token'] and v.get('committed')=='1']
        matches=[]
        for v in options:
            cr=commits.get(v['execution_stamp_ns'])
            if cr is None:continue
            msg=classes[0]._repair_prefix_with_braking_tail(d.trajectory(q,u,classes))
            recq=[];calcq=[];recu=[];calcu=[];k=0
            while cr.get(f'field.points{k}.time_from_start'):
                tt=float(cr[f'field.points{k}.time_from_start'])/1e9+float(v['precommit_suffix_phase_s'])
                p=classes[0]._point_at_time(msg,tt)
                recq.append(d.values(cr,f'field.points{k}.positions'));recu.append(d.values(cr,f'field.points{k}.velocities'))
                calcq.append(p.positions);calcu.append(p.velocities);k+=1
            recq=np.array(recq);calcq=np.array(calcq)
            shape=float(np.max(np.abs((recq-recq[0])-(calcq-calcq[0]))));vel=float(np.max(np.abs(np.array(recu)-calcu)))
            matches.append(dict(raw=v['raw_candidate_stamp_ns'],audited=v['audited_trajectory_stamp_ns'],execution=v['execution_stamp_ns'],shape_error=shape,velocity_error=vel,
                tracker_records=sum(r.get('execution_stamp_ns')==v['execution_stamp_ns'] for r in tracker),
                tracker_aborted=any(r.get('execution_stamp_ns')==v['execution_stamp_ns'] and r.get('execution_aborted')=='1' for r in tracker),
                committed_span=float(np.max(np.abs(recq-recq[0])))))
        valid=[m for m in matches if m['shape_error']<2e-6 and m['velocity_error']<2e-5]
        verified=step_error<2e-6 and len(valid)==1
        if not verified:failures.append(n)
        active=[];gx=d.unpack_matrix(problem['G'])@x-np.array(problem['h_l'])
        for i,row in enumerate(rows):
            if sol['z_l'][i]<.01:continue
            g=np.array(json.loads(row['gradient']));qlin=np.array(json.loads(row['q_linearization']));at0=float(row['distance'])+g@(qbar[0]-qlin)
            delta=straight-qbar[0];td=target-qbar[0];gap=at0-row['effective_margin']
            active.append(dict(point=json.loads(row['point']),k=int(row['timestep']),margin=row['effective_margin'],distance_at_start=at0,gap_at_start=gap,
                gradient=g.tolist(),dual=sol['z_l'][i],slack=float(gx[i]),
                straight_cos=float(delta@g/(np.linalg.norm(delta)*np.linalg.norm(g))),
                straight_allowed_fraction=float(max(0,gap)/(-g@delta)) if g@delta<0 else None,
                actual_target_dot_gradient=float(td@g)))
        info=dict(plan=n,time=pub['ros_s'],start=start['ros_s'],query=stamp,token=pub['observation_token'],recovery=pub.get('recovery_frontier_active')=='1',
            step=step,logged_step=float(pub['step_inf']),step_error=step_error,frontier_freeze_shift_error=shift_error,
            iterations=sol['iterations'],logged_iterations=int(pub['piqp_iter']),validated=verified,matches=matches,
            q0=qbar[0].tolist(),q2=q[2].tolist(),raw_span=float(np.max(np.abs(q-qbar[0]))),q2_delta_inf=float(np.max(np.abs(q[2]-qbar[0]))),
            target=target.tolist(),straight=straight.tolist(),active_rows=active,
            margins_histogram={str(v):sum(r['effective_margin']==v for r in rows) for v in sorted({r['effective_margin'] for r in rows})})
        reports[n]=info
        sub=out/str(n);sub.mkdir();d.save(sub/'problem.json',problem);d.save(sub/'solution.json',sol);d.save(sub/'rows.json',rows)
        print('PLAN',n,'verified',verified,'step_error',step_error,'matches',[(m['shape_error'],m['velocity_error']) for m in matches],'active',[(r['point'],r['margin'],r['straight_cos']) for r in active],flush=True)
    d.save(out/'report.json',dict(run_id=RID,cycles=cycles,plans=reports,unverified_plans=failures,
        limitations=['QP reassembly, not a recorded raw/matrix snapshot.','Full frozen map unavailable; no new execution certificates.','Unverified reassemblies must not support counterfactual conclusions.']))
    print('CYCLES',json.dumps(cycles),flush=True)
    print('UNVERIFIED',failures,flush=True)

if __name__=='__main__':main()
