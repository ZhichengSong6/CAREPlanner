#!/usr/bin/env python3
"""Replay recorded O11 outcomes and measured motion through production C++ memory."""
import argparse
import json
from pathlib import Path
import subprocess
from diagnose_phase_e_case006_o6_recovery import strings,digest
from evaluate_phase_d_run import joint_states,JOINTS

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy-binary',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    run=root/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_006_r1_case006_early_observation_20261009/run'
    paths=[run/(n+'.csv') for n in ['verification_outcome','observation_candidate_identity','local_planner_summary',
                                   'joint_states','visibility_acquisition_summary']]
    tracepath=next((run/'projector_traces').glob('c46_obligation_011_*.json'))
    replacepath=run/'projector_traces/candidate_replacement.jsonl'
    paths.extend([tracepath,replacepath,a.policy_binary.resolve(),Path(__file__).resolve(),
        root/'src/egocentric_arm_planner/include/egocentric_arm_planner/vbc_failure_direction_memory.hpp'])
    hashes={str(p.relative_to(root)):digest(p) for p in paths}
    outcomes=strings(paths[0]);ids={r['raw_candidate_stamp_ns']:r['observation_token'] for r in strings(paths[1])}
    planner=strings(paths[2]);samples=joint_states(str(paths[3]),JOINTS,True);acq=strings(paths[4])
    old='care_obs_v1_11_dcc0ace8d0c1528c253dfbb3';new='care_obs_v1_11_b237b1c0692c88669c07c8f2'
    points2=json.loads(tracepath.read_text())['active_set_points_xyz']
    repl=next(r for line in replacepath.read_text().splitlines()
              if (r:=json.loads(line)).get('request_id')==4 and r['event']=='applied_requires_existing_gates')
    points5=repl['points']
    oldq={r['active_q_vis'] for r in acq if r['active_q_vis_token']==old}
    newq={r['active_q_vis'] for r in acq if r['active_q_vis_token']==new}
    assert len(oldq)==1 and oldq==newq
    q=[float(v) for v in next(iter(oldq)).split(',')]
    events=[]
    def context(t,token,points):
        fields=[token,len(points)]+[v for p in points for v in p]+q+[1]*7
        events.append((t,0,'C '+' '.join(map(str,fields))))
    # First plan rows explicitly record the planner's point seq/count/token.
    for token,points in [(old,points2),(new,points5)]:
        first=next(r for r in planner if r['event']=='plan_started' and r['observation_token']==token)
        assert int(first['vbc_obligation_point_count'])==len(points)
        context(first['receipt_ns'],token,points)
    matched=[]
    for r in outcomes:
        raw=int(r['raw_candidate_stamp_ns']);token=ids[str(raw)]
        if token not in [old,new] or r['result']!='unsafe':continue
        # Identity is exact in outcomes/identity streams. Query comes from the
        # unique candidate-publication row within 10 ms; receipt offset is saved.
        pubs=[p for p in planner if p['event']=='candidate_published' and p['observation_token']==token
              and 0<=p['receipt_ns']-raw<=10000000]
        assert len(pubs)==1,(raw,pubs)
        query=int(pubs[0]['query_stamp_ns']);audited=int(r['audited_trajectory_stamp_ns'])
        ev=json.loads(r['vbc_evidence']);assert int(ev['trajectory_stamp_ns'])==audited
        points=[v['point'] for v in ev['violations']]
        events.append((raw,2,f'R {raw} {query}'))
        events.append((r['receipt_ns'],3,'F '+' '.join(map(str,
            [r['receipt_ns']/1e9,raw,audited,query,len(points)]+[v for p in points for v in p]))))
        matched.append(dict(raw=raw,query=query,audited=audited,token=token,receipt_ns=r['receipt_ns'],
                            publication_receipt_offset_ns=pubs[0]['receipt_ns']-raw))
    for t,mq in samples:
        if 33.68e9<=t<=35.42e9:events.append((t,1,'M '+' '.join(map(str,[t/1e9]+list(mq)))))
    text='\n'.join(e[2] for e in sorted(events))+'\n';(a.output_dir/'events.txt').write_text(text)
    p=subprocess.run([str(a.policy_binary.resolve()),'--replay'],input=text,text=True,capture_output=True,check=True)
    fired=[list(map(int,line.split())) for line in p.stdout.splitlines()]
    assert fired==[[34975000000,34218000000,9]],fired
    assert all(digest(root/path)==h for path,h in hashes.items())
    result=dict(status='PASS',inherited=fired,first_current_rejection_ros_s=35.012,
        prior_second_rejection_ros_s=35.409,earlier_evidence_ros_s=.397,
        candidates=matched,input_sha256=hashes,
        scope='Production C++ memory with fixed recorded outcomes; not full planner/Gazebo counterfactual.',
        target_scope='Shared target is constant at recorded four-decimal precision; replay uses that same vector. Production compares exact q/mask.',
        publication_scope='Exact raw/audited joins; query joined to unique publication receipt within 10 ms, offsets retained.')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
