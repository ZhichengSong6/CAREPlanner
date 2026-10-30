#!/usr/bin/env python3
"""Fixed CASE006 observation-chain replay; no ROS transport or execution.

Recomputes production scalar/per-sensor proposals, analytic endpoint geometry,
and replacement guards. Recorded outcomes remain the only VBC certificates.
"""
import argparse
import bisect
import copy
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_visibility_cdf/scripts'))
from diagnose_phase_e_case006_o6_recovery import strings, digest
from diagnose_phase_e_r1_o1_o6_offline import production_scalar, scalar_solve, clean, union_geometry
from audit_phase_e_replacement_score_gap import joint_limits
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime
from candidate_replacement_runtime import CandidateReplacement
from observation_identity import observation_region_token
from diagnose_phase_e_case001_exact_witness import Geometry
from evaluate_phase_d_run import joint_states, JOINTS

BATCH = 'phase_e_goal_006_r1_case006_early_observation_20261009'
RUN = ROOT/'outputs/c5_5_vbc_gcdf_regime'/BATCH/'run'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_paths = list((ROOT/'src/care_visibility_cdf/scripts').glob('*.py'))
    source_paths += [ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',Path(__file__).resolve()]
    inputs = []
    def read(path):
        inputs.append(path)
        return json.loads(path.read_text())
    def records(name):
        path = RUN/(name+'.csv'); inputs.append(path)
        return strings(path)
    traces = {i: read(next((RUN/'projector_traces').glob(f'c46_obligation_{i:03d}_*.json')))
              for i in [7,11]}
    tracepath = RUN/'projector_traces/candidate_replacement.jsonl'; inputs.append(tracepath)
    trace = [json.loads(s) for s in tracepath.read_text().splitlines()]
    replacement = next(r for r in trace if r['event']=='applied_requires_existing_gates' and r['request_id']==4)
    acquisitions = records('visibility_acquisition_summary')
    outcomes = records('verification_outcome')
    triggers = [r for r in records('candidate_replacement_trigger') if 31e9<r['receipt_ns']<36e9]
    identities = {r['raw_candidate_stamp_ns']:r['observation_token'] for r in records('observation_candidate_identity')}
    jointpath = RUN/'joint_states.csv'; inputs.append(jointpath)
    samples = joint_states(str(jointpath), JOINTS, True); times = [t for t,q in samples]
    def measured(t): return np.asarray(samples[bisect.bisect_right(times,int(t*1e9))-1][1])
    shared_row = next(r for r in acquisitions if r['active_q_vis_token']=='care_obs_v1_11_dcc0ace8d0c1528c253dfbb3')
    shared_logged = np.array([float(x) for x in shared_row['active_q_vis'].split(',')])
    checkpoint = ROOT/'src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt'
    scalar_checkpoint = ROOT/'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt'
    urdf = ROOT/'src/arm_description/urdf/Arm.urdf'
    collision = ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    source_paths += [checkpoint,scalar_checkpoint,urdf,collision]
    protected = {str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    assert digest(checkpoint)==traces[11]['per_sensor_hybrid']['config']['checkpoint_sha256']
    qmin,qmax = joint_limits()
    runtime = PerSensorVisibilityRuntime(str(checkpoint),str(urdf),str(collision),torch.device('cpu'),qmin,qmax)
    scalar = production_scalar(traces[11]['config'])
    geom = Geometry()
    points2 = np.array(traces[11]['active_set_points_xyz'])
    points5 = np.array(replacement['points'])
    parent = np.array(traces[7]['active_set_points_xyz'])
    qseed = np.array(replacement['measured_seed'])
    shared_seed = measured(32.658)

    # Exact recorded seeds for independent candidate generation; approximate
    # shared seed because that solver's input was not recorded separately.
    proposals = []
    for label,points,seed in [
        ('individual_o11_original_2',points2,np.array(traces[11]['c4_6_measured_seed_q'])),
        ('shared_o11_o7_3',np.vstack([points2,parent]),shared_seed),
        ('individual_o11_expanded_5',points5,qseed),
        ('shared_o11_o7_expanded_6',np.vstack([points5,parent]),qseed)]:
        result = scalar_solve(scalar,points,seed)
        hybrid = runtime.generate(points,result['q_zero'],branch_seed_row=seed)
        selected = hybrid['selected_q_vis'] if hybrid['accepted'] else result['q_vis']
        proposals.append(dict(label=label,points=points,seed=seed,scalar=result,hybrid=hybrid,
            selected_q=selected,endpoint_geometry=union_geometry(runtime,points,selected)))
        print(label,'accepted',hybrid['accepted'],'sensor',hybrid['selected_sensor_id'],
              'q',np.round(selected,5).tolist(),flush=True)
    original_error = float(np.max(np.abs(np.array(proposals[0]['selected_q'])-traces[11]['q_vis'])))
    assert original_error<1e-4, original_error
    shared_error = float(np.max(np.abs(np.array(proposals[1]['selected_q'])-shared_logged)))
    # Four-decimal recorder target; seed is at/before stack push, not an exact
    # internal callback snapshot. Keep measured error instead of assuming equality.
    shared_match = shared_error < 1e-3

    h = replacement['hybrid']
    replay_branch = runtime._optimize_branch(torch.tensor(points5,dtype=torch.float32),
        torch.tensor(qseed,dtype=torch.float32).reshape(1,7),3)
    replacement_error = float(np.max(np.abs(np.array(replay_branch['q_candidate'])-h['selected_q_vis'])))
    assert replacement_error<1e-4, replacement_error
    reranked = runtime.generate(points5,h['scalar_q_zero'],branch_seed_row=qseed,
        replacement=dict(excluded=[0],previous_q=shared_logged,score_gap_override=.75,minimum_conservative_g=0.))
    assert reranked['selected_sensor_id']==3 and reranked['accepted']
    # Compare geometrical directions against the same recorded unknown centers.
    # A sampled straight joint path is a diagnostic, never the recorded SCP path.
    directions=[]
    for label,q in [('logged_shared_rounded',shared_logged),
                    ('original_s0',traces[11]['q_vis']),('replacement_s3',h['selected_q_vis'])]:
        q=np.asarray(q); delta=q-qseed
        alphas=np.linspace(0,1,101)
        distances=np.array([geom.minimum(qseed+a*delta,points5)[0] for a in alphas])
        step=delta*min(1.,.01/max(np.max(np.abs(delta)),1e-12))
        directions.append(dict(label=label,q=q,delta=delta,
            endpoint_geometry=union_geometry(runtime,points5,q),
            alphas=alphas,point_distances_m=distances,
            local_step_min_m=float(geom.minimum(qseed+step,points5)[0].min()),
            min_sampled_m=float(distances.min()),start_min_m=float(distances[0].min()),
            mask=h['selected_joint_mask'] if label=='replacement_s3' else
                 traces[11]['q_vis_joint_mask'] if label=='original_s0' else [1.]*7))

    # Exercise actual production offer_final_vbc with isolated inert transport.
    # Contexts reproduce the recorded guard conditions, not unrecorded callback internals.
    guard_replays=[]
    def guard(trigger,now,obs,stack,published_token,expected):
        n=NS(_schedule_publish_lock=threading.RLock(),_obligation_lock=threading.RLock(),
            _obligations=copy.deepcopy(obs),_repair_stack=stack,
            _trace_published_target=dict(observation_token=published_token,
                region_token=observation_region_token(obs[-1]),q_vis=shared_logged.copy()))
        logs=[]; sent=[]
        r=CandidateReplacement(n,lambda:now,sent.append,lambda event,**kw:logs.append(dict(event=event,**kw)))
        r.offer_final_vbc(' '.join(f'{k}={v}' for k,v in trigger.items() if k!='receipt_ns'))
        outcome=logs[-1].get('reason',logs[-1]['event'])
        assert outcome==expected,(outcome,expected,logs)
        guard_replays.append(dict(raw=trigger['raw_candidate_stamp_ns'],now=now,expected=expected,
            logs=logs,sent=sent,scope='actual production guard with reconstructed context'))
    def ob(i,points,q):
        return dict(id=i,points=points,q_vis=q,generation_event_id=traces[i]['generation_event_id'],
                    per_sensor_selected_sensor_id=0 if i==11 else 4)
    o7=ob(7,parent,traces[7]['q_vis']);o11=ob(11,points2,shared_logged)
    guard(triggers[0],32.355,[o7],[7],triggers[0]['observation_token'],'replacement_points_not_live')
    # Freshness is evaluated before waiting for the schedule lock. The first
    # callback's logged rejection is 33.679, after the 1 s shared solve; using
    # that late time for its entry check would incorrectly replay it as stale.
    guard(triggers[1],32.918,[o7,o11],[7,11],shared_row['active_q_vis_token'],'final_vbc_owner_not_current')
    guard(triggers[2],33.634,[o7,o11],[7,11],shared_row['active_q_vis_token'],'final_vbc_owner_not_current')
    grown=ob(11,points5,shared_logged)
    newtoken='care_obs_v1_11_b237b1c0692c88669c07c8f2'
    guard(triggers[3],34.298,[o7,grown],[7,11],newtoken,'published_target_identity_changed')
    guard(triggers[4],35.458,[o7,grown],[7,11],newtoken,'final_vbc_repeat')

    selected_outcomes=[]
    for r in outcomes:
        if not 31.227e9<=r['receipt_ns']<=38.314e9:continue
        token=identities[r['raw_candidate_stamp_ns']]
        selected_outcomes.append(dict(receipt_ns=r['receipt_ns'],token=token,result=r['result'],
            raw=r['raw_candidate_stamp_ns'],audited=r['audited_trajectory_stamp_ns'],
            execution=r['execution_stamp_ns'],evidence=json.loads(r['vbc_evidence'])))
    qwindow=np.array([q for t,q in samples if 32.202e9<=t<=35.791e9])
    # Fixed-event scheduling counterfactual: wait after unowned repeated-point
    # rejection until the recorded target revision. No fabricated safe outcomes.
    held=[r for r in selected_outcomes if r['token']==triggers[0]['observation_token']
          and 32.355e9<r['receipt_ns']<33.685e9 and r['result']=='unsafe']
    assert len(held)==2
    # Candidate-policy experiment, not a production patch: carry failed
    # direction evidence over a strict point-set growth, then require a new
    # exact rejection under the current identity before requesting replacement.
    def carry_failure(old_points,new_points,old_q,new_q,motion,old_epoch,new_epoch):
        key=lambda ps:{tuple(np.asarray(p,dtype=np.float32)) for p in ps}
        return (key(old_points)<key(new_points) and old_epoch==new_epoch
                and np.max(np.abs(np.array(old_q)-new_q))<=1e-4 and motion<=.005)
    movement=float(np.ptp(qwindow,axis=0).max())
    assert carry_failure(points2,points5,shared_logged,shared_logged,movement,2,2)
    for a,b,q,m,e in [(points5,points2,shared_logged,movement,2),
                     (points2,points5,shared_logged+.01,movement,2),
                     (points2,points5,shared_logged,.02,2),
                     (points2,points5,shared_logged,movement,3)]:
        assert not carry_failure(a,b,shared_logged,q,m,2,e)
    growth_outcomes=[r for r in selected_outcomes if r['token']==newtoken and r['result']=='unsafe']
    assert len(growth_outcomes)==2
    old_outcomes=[r for r in selected_outcomes if r['token']==shared_row['active_q_vis_token'] and r['result']=='unsafe']
    common={tuple(v['point']) for v in old_outcomes[-1]['evidence']['violations']} & {
        tuple(v['point']) for v in growth_outcomes[0]['evidence']['violations']}
    assert common
    report=dict(status='PASS',scope='Fixed proposal/geometry/guard replay; no new closed-loop run or full VBC replay',
        calibration=dict(original_s0_q_error_inf=original_error,replacement_s3_q_error_inf=replacement_error,
            shared_seed_ros_ns=samples[bisect.bisect_right(times,32658000000)-1][0],
            shared_rounded_q_error_inf=shared_error,shared_reproduced_within_1mrad=shared_match),
        proposals=proposals,replacement_branch=replay_branch,replacement_reranking=reranked,
        directions=directions,guard_replays=guard_replays,recorded_outcomes=selected_outcomes,
        stationary_window=dict(ros_s=[32.202,35.791],joint_range_inf=float(np.ptp(qwindow,axis=0).max())),
        scheduling_counterfactual=dict(policy='Bounded pending-owner wait after first unmatched repeated-point trigger; reevaluate on owner/target events',
            omitted_recorded_rejections=[r['raw'] for r in held],count=len(held),
            scope='Fixed-event accounting only; no claim of time saved or task success; requires fresh identity/map checks and existing bounded fallback'),
        growth_memory_counterfactual=dict(qualification='Prototype only: rounded target identity plus production shared-cache persistence; not an execution certificate',
            common_rejected_point_count=len(common),negative_boundary_cases=4,
            fresh_identity_rejection_raw=growth_outcomes[0]['raw'],
            request_opportunity_ros_s=growth_outcomes[0]['receipt_ns']/1e9,
            actual_second_rejection_ros_s=growth_outcomes[1]['receipt_ns']/1e9,
            earlier_evidence_s=(growth_outcomes[1]['receipt_ns']-growth_outcomes[0]['receipt_ns'])/1e9,
            omitted_rejection_opportunity=1,
            note='Retain only failed-direction history; current token/raw, live ownership, freshness and original budget must still be checked. Counts are not additive closed-loop savings.'),
        limitations=['Rejected audited trajectories and complete map snapshots are absent (rejection_snapshot=disabled).',
            'Shared internal seed not separately recorded; previous measured sample and rounded published target used explicitly.',
            'Straight joint interpolation is a geometry diagnostic, not a feasible trajectory or continuous VBC certificate.',
            'Guard replay contexts reconstructed to match recorded conditions; private callback ordering not fully captured.',
            'Alternative candidates require hard QP, final GCDF, exact VBC and tracker checks before execution.'],
        protected_sha256=protected,input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs})
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    (args.output_dir/'result.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(clean(dict(status='PASS',calibration=report['calibration'],
        joint_range=report['stationary_window'],guard_cases=len(guard_replays),
        held_rejections=len(held),directions=[{k:d[k] for k in ['label','start_min_m','local_step_min_m','min_sampled_m']} for d in directions])),indent=2))

if __name__=='__main__':main()
