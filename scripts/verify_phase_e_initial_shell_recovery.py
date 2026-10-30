#!/usr/bin/env python3
"""Private production QP/selector/commit verification; no robot or actuator."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client

ROOT=Path(__file__).resolve().parents[1]
NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--state-file',type=Path,default=ROOT/'outputs/phase_e_initial_shell_recovery_fix_20261012/fixed_state.json')
    ap.add_argument('--qp-fixture',type=Path,default=ROOT/'outputs/phase_e_initial_shell_recovery_fix_20261012/qp_fixture.json')
    ap.add_argument('--normal-motion-fixture',type=Path)
    ap.add_argument('--coupled-qp-snapshot',type=Path)
    ap.add_argument('--recorded-reserve-fixture',type=Path)
    ap.add_argument('--recorded-path-fixture',type=Path)
    args=ap.parse_args()
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    state=json.loads(args.state_file.read_text())
    fixture=args.qp_fixture.resolve()
    fixture_data=json.loads(fixture.read_text())
    expected_cdf_rows=20*len(fixture_data['rows'])
    flags=(ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/flags.make').read_text()
    inc=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1])
    link=shlex.split((ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/link.txt').read_text());libs=link[link.index('-o')+2:]
    obj=ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src'
    header=(ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp').read_text()
    (out/'planner_diagnostic_access.hpp').write_text(header.replace('private:','public:'))
    with (out/'compile.log').open('x') as log:
        subprocess.run(['c++','-std=c++14','-O2',*inc,'-I'+str(out),str(ROOT/'scripts/test_phase_e_initial_shell_qp.cpp'),
            str(obj/'local_sparse_scp_planner.cpp.o'),str(obj/'primitive_observation_guard.cpp.o'),'-o',str(out/'qp_test'),*libs],stdout=log,stderr=subprocess.STDOUT,check=True)
    os.environ.update(ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[];streams=[];report=dict(status='FAIL',checks=[],qualification='Saved states/witnesses/learned rows; production QP and full selector on a controlled complete map. Not a reconstruction of historical full QP, complete map, SCP schedule or physical contact.',
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.state_file,fixture]})
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    def start(argv,name):
        stream=(out/(name+'.log')).open('x');streams.append(stream)
        p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p
    def wait(predicate,timeout=10):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if predicate():return
            time.sleep(.01)
        raise AssertionError('private ROS condition timed out')
    try:
        master=start(['roscore','-p',str(port)],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/shell_test')[0]==1
            except OSError:return False
        wait(ready)
        import rospy,yaml
        from std_msgs.msg import String,Bool
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from care_confidence_map.srv import QueryConfidence,QueryConfidenceResponse,QueryConfidenceSnapshot,QueryConfidenceSnapshotResponse
        sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
        from initial_shell_recovery import context,commit_check
        rospy.init_node('shell_test_driver',disable_signals=True)
        cfg=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        cfg['local_planner']['primitive_observation_guard']=dict(enabled=True,fk_urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
            primitive_urdf=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),base_frame='base_link',ignored_links=['base_link','link1'],margin_m=.02)
        cfg['local_planner']['repair_executable_view']=dict(enabled=True,prefix_s=.25,cycle_enabled=True,cycle_prefixes_s=[.15,.1,.05],
            brake_dt_s=.05,max_brake_steps=20,acceleration_limits=[3.,3.,4.,4.,6.,6.,6.])
        rospy.set_param('/initial_shell_qp',cfg)
        qp_args=[str(out/'qp_test'),str(fixture),str(out/'qp_results.json')]
        if args.coupled_qp_snapshot:qp_args.append(str(args.coupled_qp_snapshot.resolve()))
        qp=start(qp_args,'qp');assert qp.wait(timeout=30)==0
        results=json.loads((out/'qp_results.json').read_text());report['qp']=results
        assert results[0]['status']=='primitive_observation_start_blocked' and not results[0]['solved']
        assert results[1]['solved'] and results[1]['shell_points']==1 and results[1]['penalty_rows']>0
        assert results[0]['cdf_rows']==results[1]['cdf_rows']==expected_cdf_rows and results[1]['max_slack']==0.
        assert results[2]['classification']=='unknown_start' and results[3]['classification']=='occupied_start'
        assert results[4]['failure_dispatch']=='PASS'
        assert results[5]['owned_margin_hard_qp']=='PASS'
        if args.coupled_qp_snapshot:
            assert results[6]['coupled_unknown_restore_points']=='PASS'
            report['checks'].append('Recorded coupled hard-QP failure has no single-row start conflict but retains explicit UNKNOWN restoration points')
        report['checks'].append('Native first/later UNKNOWN/OCCUPIED conflict classification and bounded stable-state recovery dispatcher')
        report['checks'].append('Same native hard CDF rows: old start rejected, recovery QP solved without CDF slack')
        unknown=copy.deepcopy(state['points'])
        mode=['unknown']
        occupied=[]
        def confidence(req):
            if mode[0]=='invalid':return QueryConfidenceResponse([],[],[])
            low=[mode[0]!='known' and any(max(abs(p[j]-[x.x,x.y,x.z][j]) for j in range(3))<1e-5 for p in unknown) for x in req.points]
            return QueryConfidenceResponse([0. if x else 1. for x in low],[0.]*len(low),[1]*len(low))
        service=rospy.Service('/shell_test/query',QueryConfidence,confidence)
        def snapshot(req):
            response=QueryConfidenceSnapshotResponse()
            response.valid=mode[0]!='invalid';response.instance='fixed_map';response.epoch=1;response.revision=1
            response.snapshot_stamp=rospy.Time.now();response.map_update_stamp=response.snapshot_stamp
            response.frame='base_link';response.resolution=.05
            response.inside_map=[];response.confidence_now=[];response.occupancy=[]
            for x in req.points:
                p=[x.x,x.y,x.z]
                low=mode[0]!='known' and any(max(abs(a-b) for a,b in zip(p,q))<1e-5 for q in unknown)
                occ=any(max(abs(a-b) for a,b in zip(p,q))<1e-5 for q in occupied)
                response.inside_map.append(1);response.confidence_now.append(0. if low else 1.)
                response.occupancy.append(1. if occ else .5 if low else 0.)
            return response
        snapshot_service=rospy.Service('/shell_test/query_snapshot',QueryConfidenceSnapshot,snapshot)
        cfg=yaml.safe_load((ROOT/'src/care_confidence_map/config/trajectory_risk.yaml').read_text())['trajectory_vbc']
        cfg.update(geometry_backend='primitive',robot_urdf_file=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
            primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            base_frame='base_link',input_trajectory_topic='/shell_test/task',predicted_trajectory_topic='/shell_test/candidate',
            prefer_predicted_trajectory=True,predicted_trajectory_timeout=.5,event_driven_eval=True,eval_rate=20.,
            predicted_periodic_refresh_rate=20.,swept_volume_margin_m=.02,continuous_motion_bound_enabled=True,
            ignored_risk_links=['base_link','link1'],confidence_query_service='/shell_test/query',
            initial_shell_recovery_enabled=True,initial_shell_execution_margin_m=.01)
        rospy.set_param('/shell_selector',{'trajectory_vbc':cfg})
        summaries=[];protocols=[]
        sub=rospy.Subscriber('/shell_test/summary',String,lambda msg:summaries.append(msg.data),queue_size=20)
        protocol_sub=rospy.Subscriber('/shell_test/candidate/protocol_status',String,lambda msg:protocols.append(msg.data),queue_size=20)
        selector=start([str(ROOT/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node'),'__name:=shell_selector',
            '/care_planner/trajectory_risk/vbc_summary:=/shell_test/summary'],'selector')
        pub=rospy.Publisher('/shell_test/candidate',JointTrajectory,queue_size=1)
        ctx=rospy.Publisher('/shell_test/candidate/initial_shell_context',String,queue_size=4)
        joint=rospy.Publisher('/care_arm/joint_states',JointState,queue_size=1)
        task_pub=rospy.Publisher('/shell_test/task',JointTrajectory,queue_size=1)
        force_pub=rospy.Publisher('/care_planner/trajectory_risk/force_bootstrap',Bool,queue_size=1)
        wait(lambda:pub.get_num_connections() and ctx.get_num_connections() and joint.get_num_connections())
        measured=list(state['q0']);velocity=[0.]*7
        def joint_tick(_):joint.publish(JointState(header=__import__('std_msgs.msg',fromlist=['Header']).Header(stamp=rospy.Time.now()),name=NAMES,position=list(measured),velocity=list(velocity)))
        timer=rospy.Timer(rospy.Duration(.005),joint_tick);time.sleep(.05)
        def tokens(text):return dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',text))
        def evaluate(label,q,ts=None,eligible=True,context_mutation=None,with_context=True,velocities=None,
                     geometric=False,expect_certificate=True,unique_query_identity=False,margin_recovery=False):
            candidate=JointTrajectory();candidate.header.frame_id='base_link';candidate.header.stamp=rospy.Time.now();candidate.header.seq=len(report.get('selector',[]))+1
            candidate.joint_names=NAMES
            if ts is None:ts=[k*.05 for k in range(len(q))]
            for k,(pose,t) in enumerate(zip(q,ts)):candidate.points.append(JointTrajectoryPoint(positions=list(pose),velocities=list(velocities[k]) if velocities is not None else [0.]*7,time_from_start=rospy.Duration(t)))
            raw=context(candidate,candidate.header.stamp.to_nsec()+500000000,eligible,[3.,3.,4.,4.,6.,6.,6.],.05,20)
            if geometric:
                data=json.loads(raw);stamp=candidate.header.stamp.to_nsec()
                data['kind']='geometric_restore'
                data['planned_prefix']=dict(schema=1,session='a'*32,revision=1,prefix_s=min(.05,ts[-1]),
                    brake_dt_s=.05,brake_steps=20,acceleration=[3.,3.,4.,4.,6.,6.,6.])
                data['geometric_restore']=dict(schema=1,raw_candidate_stamp_ns=str(stamp),query_stamp_ns=str(stamp-1),
                    mode_epoch='2',observation_token='care_obs_v1_1_fixture',points=state['points'])
                if unique_query_identity:
                    data['geometric_restore'].update(schema=2,query_stamp_ns=str(stamp+1),query_ros_ns=str(stamp-1))
                if margin_recovery:
                    data['geometric_restore'].update(schema=2,query_ros_ns=str(stamp-1),margin_recovery=True)
                raw=json.dumps(data,separators=(',',':'))
            if context_mutation:
                data=json.loads(raw);context_mutation(data);raw=json.dumps(data,separators=(',',':'))
            if with_context:ctx.publish(String(data=raw))
            pub.publish(candidate)
            if context_mutation or not with_context or not expect_certificate:
                time.sleep(.12)
                assert not any(tokens(s).get('trajectory_stamp_ns')==str(candidate.header.stamp.to_nsec()) for s in summaries)
                row=dict(label=label,decision='no_certificate')
                if geometric:
                    wait(lambda:any(tokens(s).get('trajectory_stamp_ns')==str(candidate.header.stamp.to_nsec()) and
                        tokens(s).get('reason') in ('geometric_restore_certificate_unavailable','geometric_restore_path_counterexample') for s in protocols),.45)
                    matching=[tokens(s) for s in protocols if tokens(s).get('trajectory_stamp_ns')==str(candidate.header.stamp.to_nsec())]
                    reasons={p.get('reason') for p in matching}
                    assert reasons & {'geometric_restore_certificate_unavailable','geometric_restore_path_counterexample'},matching
                    row['protocol_reasons']=sorted(reasons)
                    row['path_counterexample_evidence']=[json.loads(value) for value in sorted({
                        p['path_counterexample_evidence'] for p in matching
                        if p.get('reason')=='geometric_restore_path_counterexample' and
                        p.get('path_counterexample_evidence','none')!='none'})]
                    if label in ('geometric_restore_execution_floor_dip','outer_start_reserve_legacy_rejected'):
                        assert 'geometric_restore_path_counterexample' in reasons,matching
                    if label=='geometric_restore_occupied_volume_rejected':
                        assert 'geometric_restore_path_counterexample' not in reasons,matching
                    assert not any(tokens(s).get('trajectory_stamp_ns')==str(candidate.header.stamp.to_nsec()) for s in summaries)
            else:
                wait(lambda:any(tokens(s).get('trajectory_stamp_ns')==str(candidate.header.stamp.to_nsec()) for s in summaries),.45)
                row=tokens(next(s for s in summaries if tokens(s).get('trajectory_stamp_ns')==str(candidate.header.stamp.to_nsec())))
                row['label']=label
            report.setdefault('selector',[]).append(row);return row,candidate
        proposals=state['proposals']
        away=proposals[0]
        old,_=evaluate('ordinary_same_away',away['q'],away['times_s'],eligible=False)
        assert old['has_violation']=='1' and old['shell_recovery_count']=='0'
        new,candidate=evaluate('recovery_away',away['q'],away['times_s'])
        assert new['has_violation']=='0' and new['shell_recovery_count']=='1',new
        assert new['reason']=='initial_shell_recovery_certified'
        evidence=json.loads(new['shell_recovery_evidence']);time_ns=int(evidence['measured_stamp_ns'])
        check_now=max(time_ns,int(evidence['map_snapshot_stamp_ns']))
        assert commit_check(new['shell_recovery_evidence'],candidate.header.stamp.to_nsec(),check_now,time_ns,NAMES,
            dict(zip(NAMES,evidence['measured_q'])),dict(zip(NAMES,[0.]*7)),[3.,3.,4.,4.,6.,6.,6.],.05,20,.01,actuator_velocity=dict(zip(NAMES,[0.]*7)))
        assert not commit_check(new['shell_recovery_evidence'],candidate.header.stamp.to_nsec()+1,check_now,time_ns,NAMES,
            dict(zip(NAMES,evidence['measured_q'])),dict(zip(NAMES,[0.]*7)),[3.,3.,4.,4.,6.,6.,6.],.05,20,.01,actuator_velocity=dict(zip(NAMES,[0.]*7)))
        report['checks'].append('Same executable: ordinary VBC rejects; scoped continuous recovery passes; onset evidence binds exact stamp')
        assert int(new['sweep_refinement_fk'])<=256
        alternate,_=evaluate('explicit_geometric_restore_shell',away['q'],away['times_s'],geometric=True,unique_query_identity=True)
        assert alternate['has_violation']=='0' and json.loads(alternate['shell_recovery_evidence'])['contract']=='geometric_restore'
        report['checks'].append('Uniquified query identity can lead ROS time by1ns while separately bound physical clock stays fresh')
        measured[:]=away['q'][-1];time.sleep(.03)
        shifted=[[a+(b-c) for a,b,c in zip(q,measured,state['q0'])] for q in away['q']]
        normal_restore,_=evaluate('explicit_geometric_restore_outside_shell',shifted,away['times_s'],geometric=True)
        assert normal_restore['has_violation']=='0'
        assert all(p['floor_m']==.02 for p in json.loads(normal_restore['shell_recovery_evidence'])['points'])
        # Initial state outside20mm, a safe temporary entry into its reserve,
        # and an outward endpoint. Only explicit owned margin recovery may
        # certify this path at10mm; ordinary and legacy intent keep20mm.
        boundary=[.5*(a+b) for a,b in zip(state['q0'],away['q'][-1])]
        measured[:]=boundary;time.sleep(.03)
        crossing=[boundary,state['q0'],away['q'][-1]]
        ordinary,_=evaluate('outer_start_reserve_ordinary_rejected',crossing,[0.,.05,.25],eligible=False)
        assert ordinary['has_violation']=='1'
        evaluate('outer_start_reserve_legacy_rejected',crossing,[0.,.05,.25],geometric=True,expect_certificate=False)
        reserve,_=evaluate('outer_start_owned_margin_recovery',crossing,[0.,.05,.25],geometric=True,margin_recovery=True)
        reserve_evidence=json.loads(reserve['shell_recovery_evidence'])
        assert reserve['has_violation']=='0' and all(p['start_m']>.02 and p['floor_m']==.01 for p in reserve_evidence['points'])
        report['checks'].append('Outer start with safe reserve entry: ordinary/legacy reject; explicit owned margin recovery passes native continuous10mm proof')
        approach,_=evaluate('owned_margin_safe_short_approach',[boundary,state['q0']],[0.,.05],geometric=True,margin_recovery=True)
        assert approach['has_violation']=='0'
        report['checks'].append('Owned safe short approach may consume reserve before visibility; finite owner ledger governs observation progress')
        measured[:]=away['q'][-1];time.sleep(.03)
        # A bootstrap evaluation after a restoration must not inherit its
        # declaration. The complete current map here is known free.
        mode[0]='known'
        wait(lambda:task_pub.get_num_connections() and force_pub.get_num_connections())
        bootstrap=JointTrajectory(joint_names=NAMES)
        bootstrap.header.frame_id='base_link';bootstrap.header.stamp=rospy.Time.now()
        for t in [0.,.05]:bootstrap.points.append(JointTrajectoryPoint(positions=list(measured),velocities=[0.]*7,time_from_start=rospy.Duration(t)))
        task_pub.publish(bootstrap);time.sleep(.02);force_pub.publish(Bool(data=True))
        wait(lambda:any(tokens(s).get('trajectory_source')=='bootstrap' and
            tokens(s).get('trajectory_stamp_ns')==str(bootstrap.header.stamp.to_nsec()) for s in summaries))
        bootstrap_row=tokens(next(s for s in summaries if tokens(s).get('trajectory_source')=='bootstrap' and
            tokens(s).get('trajectory_stamp_ns')==str(bootstrap.header.stamp.to_nsec())))
        assert bootstrap_row['has_violation']=='0' and bootstrap_row['shell_recovery_count']=='0',bootstrap_row
        report['checks'].append('Known-free bootstrap after geometric restoration inherits no recovery points or certificate')
        force_pub.publish(Bool(data=False));mode[0]='unknown';time.sleep(.03)
        measured[:]=state['q0'];time.sleep(.03)
        deep_control=[a+10*(b-a) for a,b in zip(state['q0'],proposals[2]['q'][-1])]
        evaluate('geometric_restore_execution_floor_dip',[state['q0'],deep_control,away['q'][-1]],
                 [0.,.05,.1],geometric=True,expect_certificate=False)
        report['checks'].append('Explicit geometric restoration certifies shell and ordinary-floor paths; execution-floor dip has no certificate')
        # Exact native-QP -> production prefix/brake constructor -> selector.
        import optimized_trajectory_continuity_node as continuity
        constructor=continuity.OptimizedTrajectoryContinuityNode.__new__(continuity.OptimizedTrajectoryContinuityNode)
        constructor.repair_execution_prefix_s=.25;constructor.repair_acceleration_limits=[3.,3.,4.,4.,6.,6.,6.]
        constructor.repair_brake_dt_s=.05;constructor.repair_brake_max_steps=20;constructor.repair_hold_s=.1
        constructor.repair_motion_consistency_enabled=True
        constructor._latest_measured_stamp_ns=rospy.Time.now().to_nsec()
        constructor._latest_actuator_stamp_ns=constructor._latest_measured_stamp_ns
        constructor._latest_measured_q_by_name=dict(zip(NAMES,state['q0']))
        constructor._latest_measured_velocity_by_name=dict(zip(NAMES,[0.]*7))
        constructor._latest_actuator_velocity=dict(zip(NAMES,[0.]*7))
        raw=JointTrajectory(joint_names=NAMES)
        for k,pose in enumerate(results[1]['q']):
            raw.points.append(JointTrajectoryPoint(positions=pose,velocities=results[1]['u'][k] if k<20 else [0.]*7,time_from_start=rospy.Duration(k*.05)))
        executable=constructor._repair_prefix_with_braking_tail(raw,.25);assert executable is not None
        native,_=evaluate('native_qp_exact_prefix_brake_hold',[p.positions for p in executable.points],[p.time_from_start.to_sec() for p in executable.points],velocities=[p.velocities for p in executable.points])
        assert native['has_violation']=='0' and native['shell_recovery_count']=='1',native
        report['native_executable_q']=[p.positions for p in executable.points]
        report['native_executable_velocity']=[p.velocities for p in executable.points]
        report['checks'].append('Native solved QP exact production prefix/brake/hold passes continuous recovery VBC on controlled complete map')
        if args.normal_motion_fixture:
            normal=json.loads(args.normal_motion_fixture.read_text())
            report['inputs'][str(args.normal_motion_fixture.resolve())]=hashlib.sha256(args.normal_motion_fixture.read_bytes()).hexdigest()
            saved_unknown=copy.deepcopy(unknown);saved_q=list(state['q0'])
            unknown[:]=[[.05,.2,.55]];state['q0']=normal['measured_q']
            old_normal,_=evaluate('recorded_normal_fake_terminal_stop',
                [p['q'] for p in normal['raw']['knots']],[p['t'] for p in normal['raw']['knots']],
                eligible=False,velocities=[p['v'] for p in normal['raw']['knots']])
            rebuilt_normal,_=evaluate('normal_actual_ramp_and_brake',
                [p['q'] for p in normal['curve']],[p['t'] for p in normal['curve']],
                eligible=False,velocities=[p['v'] for p in normal['curve']])
            assert old_normal['has_violation']=='0' and rebuilt_normal['has_violation']=='1',(old_normal,rebuilt_normal)
            unknown[:]=saved_unknown;state['q0']=saved_q
            report['checks'].append('Recorded NORMAL false terminal stop accepted by old geometry; actual feasible full ramp/brake is rejected by unchanged native VBC')
        # Re-audits share the first request's FK allowance and deadline.
        _,repeated=evaluate('shared_budget_seed',away['q'],away['times_s'])
        budget_rows=[]
        for attempt in range(4):
            count=len(summaries);pub.publish(repeated)
            wait(lambda:len(summaries)>count,.2)
            matching=[tokens(x) for x in summaries[count:] if tokens(x).get('trajectory_stamp_ns')==str(repeated.header.stamp.to_nsec())]
            for row in matching:
                if row.get('shell_recovery_evidence','none')!='none':
                    budget_rows.append(json.loads(row['shell_recovery_evidence'])['remaining_fk'])
        assert budget_rows==sorted(budget_rows,reverse=True) and all(0<=x<=256 for x in budget_rows)
        report['shared_budget_remaining']=budget_rows
        report['checks'].append('Same immutable audit replays share the remaining FK budget without resetting the request deadline')
        toward,_=evaluate('approach_rejected',proposals[2]['q'],proposals[2]['times_s']);assert toward['has_violation']=='1'
        deep=[a+10*(b-a) for a,b in zip(state['q0'],proposals[2]['q'][-1])]
        dip=[away['q'][0],deep,away['q'][-1]]
        bad,_=evaluate('braking_or_between_knots_dip_rejected',dip,[0.,.05,.1]);assert bad['has_violation']=='1'
        unknown.append(state['new_encounter_point'])
        encountered,_=evaluate('new_unknown_keeps_ordinary_vbc',state['new_encounter_q'],away['times_s'])
        assert encountered['has_violation']=='1'
        violations=json.loads(encountered['vbc_evidence'])['violations']
        assert any(max(abs(a-b) for a,b in zip(v['point'],state['new_encounter_point']))<1e-5 for v in violations)
        unknown.pop()
        for label,change in [('foreign_payload',lambda x:x['q'][1].__setitem__(0,x['q'][1][0]+.001)),
                             ('foreign_stamp',lambda x:x.__setitem__('trajectory_stamp_ns',str(int(x['trajectory_stamp_ns'])-1))),
                             ('foreign_velocity',lambda x:x['v'][0].append(.1)),
                             ('expired_context',lambda x:x.__setitem__('deadline_ns',str(int(x['trajectory_stamp_ns'])-1)))]:
            evaluate(label,away['q'],away['times_s'],context_mutation=change)
        evaluate('missing_context',away['q'],away['times_s'],with_context=False)
        # Historical baseline survives, but a safe outward escape from a lower start is allowed.
        measured[:]=proposals[2]['q'][-1];time.sleep(.03)
        shifted=[[a+(b-c) for a,b,c in zip(q,measured,state['q0'])] for q in away['q']]
        escape,_=evaluate('lower_start_safe_escape',shifted,away['times_s']);assert escape['has_violation']=='0',escape
        measured[:]=state['q0'];velocity[:]=[-2.]*7;time.sleep(.03)
        moving,_=evaluate('incoming_braking_unsafe',away['q'],away['times_s']);assert moving['has_violation']=='1'
        evaluate('owned_margin_incoming_braking_unsafe',away['q'],away['times_s'],geometric=True,
                 margin_recovery=True,expect_certificate=False)
        velocity[:]=[0.]*7;mode[0]='known';occupied.append(state['points'][0]);time.sleep(.03)
        # Other witnesses remain unknown so there is still a recovery request;
        # the measured stop must reject occupied voxel volume independently.
        mode[0]='unknown'
        occ,_=evaluate('occupied_stop_volume_rejected',away['q'],away['times_s']);assert occ['has_violation']=='1'
        evaluate('geometric_restore_occupied_volume_rejected',away['q'],away['times_s'],geometric=True,expect_certificate=False)
        occupied.clear();mode[0]='known';time.sleep(.03)
        known,_=evaluate('fresh_observed_returns_ordinary',away['q'],away['times_s'],eligible=False);assert known['has_violation']=='0' and known['shell_recovery_count']=='0'
        if args.recorded_reserve_fixture:
            records=json.loads(args.recorded_reserve_fixture.read_text())
            report['inputs'][str(args.recorded_reserve_fixture.resolve())]=hashlib.sha256(args.recorded_reserve_fixture.read_bytes()).hexdigest()
            report['recorded_reserve']=[]
            mode[0]='unknown'
            for k,record in enumerate(records):
                e=record['evidence'];motion=record['context']
                state['points']=copy.deepcopy(e['restoration_points'])
                unknown[:]=state['points']+[[-.25,.1,.9]]
                measured[:]=e['measured_q'];velocity[:]=e['measured_velocity'];time.sleep(.03)
                row,candidate=evaluate('recorded_B_reserve_'+str(k),motion['q'],motion['times_s'],
                    velocities=motion['v'],geometric=True,unique_query_identity=True)
                assert row['has_violation']=='0',row
                improved=json.loads(row['shell_recovery_evidence'])
                assert improved['certified_residual_m']>.002,improved
                assert 0<=improved['remaining_fk']<=256
                # Substitute only the certificate for the saved commit inputs.
                # This is a controlled complete-map replay, not the unsaved
                # historical confidence snapshot. Recorded state/command bounds
                # remain unchanged, including the full feedback gain of 2.5.
                bound=record['diagnostics']['motion_bound_m']
                assert e['certified_residual_m']<bound<improved['certified_residual_m']
                replay=copy.deepcopy(e);replay['certified_residual_m']=improved['certified_residual_m']
                live=record['live'];names=live['names'];actuator=dict(zip(names,record['actuator_record_v']))
                def check(proof,position=None,speed=None,when=None):
                    return commit_check(json.dumps(proof,separators=(',',':')),int(e['trajectory_stamp_ns']),
                        when or record['now_ns'],record['live_stamp_ns'],names,
                        dict(zip(names,live['q'] if position is None else position)),
                        dict(zip(names,live['v'] if speed is None else speed)),motion['acceleration'],.05,20,.01,
                        feedback_gain=2.5,actuator_velocity=actuator)
                assert not check(e) and check(replay)
                assert not check(replay,position=[q+.02 for q in live['q']])
                assert not check(replay,speed=[2.]*7)
                assert not check(replay,when=record['now_ns']+51000000)
                report['recorded_reserve'].append(dict(index=k,old_reserve_m=e['certified_residual_m'],
                    new_reserve_m=improved['certified_residual_m'],recorded_bound_m=bound,
                    remaining_fk=improved['remaining_fk'],original_commit=False,refined_commit=True))
            report['checks'].append('Four saved B motions: controlled complete-map certificate refinement makes original commit bounds pass; large drift, unsafe velocity and stale evidence still reject')
        if args.recorded_path_fixture:
            records=json.loads(args.recorded_path_fixture.read_text())
            record=records[1];motion=record['context'];point=record['point']
            report['inputs'][str(args.recorded_path_fixture.resolve())]=hashlib.sha256(args.recorded_path_fixture.read_bytes()).hexdigest()
            state['points']=copy.deepcopy(motion['geometric_restore']['points'])
            unknown[:]=state['points']+[point];mode[0]='unknown'
            measured[:]=record['actual'];velocity[:]=record['velocity'];time.sleep(.03)
            row,_=evaluate('recorded_002_ordinary_neighbor_path',motion['q'],motion['times_s'],
                velocities=motion['v'],geometric=True,margin_recovery=True,expect_certificate=False)
            evidence=row['path_counterexample_evidence']
            assert len(evidence)==1 and evidence[0]['kind']=='unknown_ordinary_path_floor',row
            assert evidence[0]['floor_m']==.02 and max(abs(a-b) for a,b in zip(evidence[0]['point'],point))<1e-5,row
            report['checks'].append('Saved002 exact ordinary neighbor path rejected and exported as audit-bound20mm point feedback without a commit or fabricated collision witness')
        timer.shutdown()
        report['checks'].append('Approach, intermediate/braking dip, foreign/stale/missing context, unsafe incoming speed all rejected; regressed safe escape accepted; observed path ordinary')
        report['status']='PASS'
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        for stream in streams:stream.close()
        save('report.json',report)
    print(json.dumps(dict(status=report['status'],checks=report['checks'])))


if __name__=='__main__':main()
