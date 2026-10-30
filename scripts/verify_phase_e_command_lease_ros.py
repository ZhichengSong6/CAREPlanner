#!/usr/bin/env python3
"""Private ROS protocol test with the actual plugin and prescribed ideal HW.

Synthetic leases are TEST FIXTURES. No geometry/mechanical qualification and no
formal task window. The isolated plugin path is mandatory; canonical ELF is not
overwritten or loaded as the controller under test.
"""
import argparse,copy,hashlib,json,os,signal,socket,subprocess,threading,time,xmlrpc.client
from pathlib import Path
from phase_e_owned_processes import identity,remember_descendants,reap_owned

ROOT=Path(__file__).resolve().parents[1]
NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
BASE='/lease_test/controller'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--native-dir',type=Path,required=True)
    ap.add_argument('--port',type=int,default=12072)
    ap.add_argument('--bounded-targets',action='store_true')
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True)
    assert not list(out.iterdir()),'output must be empty'
    native=args.native_dir.resolve()
    expected=json.loads((ROOT/'outputs/phase_e_soft_mount_targeted_20261019/final_host_check_after_restore.json').read_text())['protected_processes']
    def protected():
        result={}
        for pid,wanted in expected.items():
            p=Path('/proc')/pid
            result[pid]=dict(starttime=(p/'stat').read_text().rsplit(')',1)[1].split()[19],
                command_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest())
            assert result[pid]['starttime']==wanted['starttime'] and result[pid]['command_sha256']==wanted['command_sha256']
        return result
    before=protected()
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',args.port))!=0,'private port busy'
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[];handles=[];owned={};records=[];lock=threading.Lock()
    result=dict(status='FAIL',scope='Real ROS/plugin on prescribed ideal hardware; synthetic leases, no stopping safety certificate',
                runtime_admission=False,new_formal_windows=0,checks=[])
    result['bounded_targets']=args.bounded_targets
    dependencies=[native/name for name in ('libcare_velocity_controller.so','leased_controller_host','trajectory_execution_manager_node')]
    if (native/'test_tracker_stop_component').exists():dependencies.append(native/'test_tracker_stop_component')
    dependencies += [ROOT/'src/egocentric_arm_planner/scripts'/name for name in ('joint_velocity_rate_limiter_node.py','execution_command_lease.py')]
    dependencies += [Path(__file__).resolve(),ROOT/'scripts/phase_e_leased_stop_evidence.py',ROOT/'scripts/phase_e_stopping_contract.py']
    dependencies += list((ROOT/'src/care_velocity_controller/msg').glob('*.msg'))
    dependencies += list((ROOT/'src/care_velocity_controller/include/care_velocity_controller').glob('*.hpp'))
    dependencies += [ROOT/'src/care_velocity_controller/src/expiring_velocity_controller.cpp',
        ROOT/'src/egocentric_arm_planner/src/trajectory_execution_manager.cpp',
        ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/trajectory_execution_manager.hpp']
    result['inputs_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    def start(argv,name):
        h=(out/(name+'.log')).open('x');handles.append(h)
        p=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);owned[p.pid]=identity(p.pid);return p
    def wait(predicate,label,timeout=10.):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if predicate():return
            if any(p.poll() is not None for p in processes):raise AssertionError('child exited: '+label)
            time.sleep(.003)
        raise AssertionError('timeout: '+label)
    state=[None];joint=[None];summaries=[];desired=[];limited=[]
    def record(kind,row):
        with lock:records.append(dict(kind=kind,**row))
    try:
        start(['roscore','-p',str(args.port)],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/lease_probe')[0]==1
            except OSError:return False
        wait(ready,'private master')
        import rospy,yaml
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from care_velocity_controller.msg import CommandStatus,ExecutionLease,LeasedVelocityCommand
        from phase_e_leased_stop_evidence import status_record,command_record,evaluate
        rospy.set_param('/use_sim_time',True);rospy.init_node('command_lease_probe',disable_signals=True)
        if (native/'test_tracker_stop_component').exists():
            regression=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
            regression['execution'].update(require_command_lease=False,primitive_tracking_guard=True,
                primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                limiter_output_command='/legacy_component/limiter',output_velocity_command='/legacy_component/desired')
            rospy.set_param('/tracker_stop_component',regression)
            with (out/'legacy_tracker_component.log').open('x') as f:
                subprocess.run([str(native/'test_tracker_stop_component')],stdout=f,stderr=subprocess.STDOUT,
                    timeout=30,check=True,start_new_session=True)
            result['checks'].append('legacy_tracker_actual_callbacks_regression')
        cfg=yaml.safe_load((ROOT/'src/arm_description/config/controllers_watchdog.yaml').read_text())['care_arm']['arm_group_velocity_controller']
        cfg.update(require_execution_lease=True)
        rospy.set_param(BASE,cfg)
        def on_state(m):
            state[0]=m;record('state',status_record(m))
        def on_desired(m):
            desired.append(copy.deepcopy(m));record('desired',command_record(m))
        def on_limited(m):
            limited.append(copy.deepcopy(m));record('limited',command_record(m))
        rospy.Subscriber(BASE+'/state',CommandStatus,on_state,queue_size=100,tcp_nodelay=True)
        rospy.Subscriber('/lease_test/joints',JointState,lambda m:joint.__setitem__(0,m),queue_size=1)
        rospy.Subscriber('/lease_test/desired',LeasedVelocityCommand,on_desired,queue_size=100,tcp_nodelay=True)
        rospy.Subscriber(BASE+'/command_leased',LeasedVelocityCommand,on_limited,queue_size=100,tcp_nodelay=True)
        direct=rospy.Publisher(BASE+'/command_leased',LeasedVelocityCommand,queue_size=10,tcp_nodelay=True)
        def on_summary(m):
            summaries.append(m.data)
            record('tracker_summary',dict(data=m.data))
        rospy.Subscriber('/lease_test/tracker_summary',String,on_summary,queue_size=100)
        grants=rospy.Publisher(BASE+'/execution_lease',ExecutionLease,queue_size=10)
        upstream=rospy.Publisher('/lease_test/desired',LeasedVelocityCommand,queue_size=10)
        trajectories=rospy.Publisher('/lease_test/trajectory',JointTrajectory,queue_size=1)
        host=start([native/'leased_controller_host',native/'libcare_velocity_controller.so'],'controller_host')
        wait(lambda:state[0] is not None and joint[0] is not None and rospy.Time.now().to_sec()>10.,'controller receipt')
        tracker=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        tracker['execution'].update(require_command_lease=True,primitive_tracking_guard=False,
            joint_states='/lease_test/joints',input_trajectory='/lease_test/trajectory',
            summary_topic='/lease_test/tracker_summary',controller_state_topic=BASE+'/state',
            lease_topic=BASE+'/execution_lease',leased_command_topic='/lease_test/desired')
        rospy.set_param('/lease_tracker',tracker)
        tracker_process=start([native/'trajectory_execution_manager_node','__name:=lease_tracker'],'tracker')
        limiter=dict(joint_names=NAMES,require_execution_lease=True,leased_input_topic='/lease_test/desired',
            leased_output_topic=BASE+'/command_leased',controller_state_topic=BASE+'/state',
            input_topic='/lease_test/unused',output_topic='/lease_test/unused_output',
            vbc_safety_hold_topic='/lease_test/vbc',external_safety_hold_topic='/lease_test/gcdf',
            mpc=dict(joint_velocity_limits=[2.]*7,joint_acceleration_limits=[3.,3.,4.,4.,6.,6.,6.]))
        rospy.set_param('/lease_limiter',limiter)
        limiter_process=start(['/usr/bin/python3',ROOT/'src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py','__name:=lease_limiter'],'limiter')
        wait(lambda:grants.get_num_connections()>=2 and trajectories.get_num_connections()>0 and upstream.get_num_connections()>=2,'complete chain connections')
        def output_ready():
            code,_,node_uri=xmlrpc.client.ServerProxy(uri).lookupNode('/lease_probe','/lease_limiter')
            if code!=1:return False
            code,_,bus=xmlrpc.client.ServerProxy(node_uri).getBusInfo('/lease_probe')
            active=[row for row in bus if row[2]=='o' and row[4]==BASE+'/command_leased' and row[5]]
            return code==1 and len(active)>=2
        wait(output_ready,'limiter output connected to BOTH recorder and controller')
        time.sleep(.04)
        lease_sequence=0;source_sequence=0
        def grant(token=None):
            nonlocal lease_sequence
            lease_sequence+=1;m=ExecutionLease();m.controller_session=state[0].controller_session
            m.execution_stamp_ns=token or rospy.Time.now().to_nsec();m.lease_sequence=lease_sequence
            m.issued_at=rospy.Time.now();m.valid_until=m.issued_at+rospy.Duration(.1);m.certificate_sha256='a'*64
            m.target_range_enforced=args.bounded_targets
            m.target_abs_velocity=[0.,.4,0.,0.,0.,0.,0.] if args.bounded_targets else [0.]*7
            grants.publish(m);record('synthetic_lease',dict(session=m.controller_session,token=m.execution_stamp_ns,sequence=m.lease_sequence,until_ns=m.valid_until.to_nsec()))
            return m
        def send(g,velocity=.3,stop=False):
            nonlocal source_sequence
            source_sequence+=1;m=LeasedVelocityCommand();m.header.stamp=rospy.Time.now();m.source_sequence=source_sequence
            m.controller_session=g.controller_session;m.execution_stamp_ns=g.execution_stamp_ns;m.lease_sequence=g.lease_sequence
            m.valid_until=g.valid_until;m.certificate_sha256=g.certificate_sha256;m.joint_names=NAMES
            m.target_range_enforced=g.target_range_enforced;m.target_abs_velocity=g.target_abs_velocity
            m.velocity=[0.,velocity,0.,0.,0.,0.,0.];m.stop=stop;upstream.publish(m);return m
        g=grant();time.sleep(.012)
        curve=JointTrajectory();curve.header.stamp=rospy.Time(0,g.execution_stamp_ns);curve.joint_names=NAMES
        for t,delta,velocity in [(0.,0.,0.),(.1,.002,.04),(.2,.004,0.)]:
            p=JointTrajectoryPoint();p.positions=list(joint[0].position);p.positions[1]+=delta
            p.velocities=[0.,velocity,0.,0.,0.,0.,0.];p.time_from_start=rospy.Duration(t);curve.points.append(p)
        trajectories.publish(curve)
        wait(lambda:any(m.execution_stamp_ns==g.execution_stamp_ns for m in desired),'tracker produced original commands')
        wait(lambda:state[0].execution_stamp_ns==g.execution_stamp_ns and state[0].mode==1,'full chain command applied')
        active_summary=next(dict(x.split('=',1) for x in s.split() if '=' in x) for s in summaries
            if 'source=active_trajectory' in s and ('execution_stamp_ns='+str(g.execution_stamp_ns)+' ') in s)
        origin=int(active_summary['phase_origin_ros_ns'])
        assert g.issued_at.to_nsec()<=origin<=int(active_summary['sample_ros_ns'])
        assert abs((int(active_summary['sample_ros_ns'])-origin)*1e-9-float(active_summary['phase_s']))<1e-5
        result['checks'].append('tracker_phase_origin_is_acceptance_time_independent_of_business_stamp')
        wait(lambda:state[0].mode==3,'fixed lease stopping')
        wait(lambda:any('execution_state=STOPPED_HOLD' in s for s in summaries),'tracker observes controller stop identity')
        result['checks'].append('tracker_limiter_actual_plugin_fixed_deadline_and_stopped_identity')
        session=g.controller_session
        states=[r for r in records if r['kind']=='state' and r['controller_session']==session and r['execution_stamp_ns']==g.execution_stamp_ns]
        assert all(r['mode']!=1 for r in states if r['t_ns']>=g.valid_until.to_nsec())
        assert any(r['mode']==1 and r['source_sequence']>=2 for r in states),'no fresh updates before deadline'
        table={(m.controller_session,m.source_sequence):command_record(m) for m in desired}
        for m in limited:
            if m.controller_session!=session:continue
            original=table[(m.controller_session,m.source_sequence)];limited_row=command_record(m)
            for key in ('generated_ns','valid_until_ns','execution_stamp_ns','lease_sequence','certificate_sha256',
                        'target_range_enforced','target_abs_velocity'):
                assert original[key]==limited_row[key],key
        if args.bounded_targets:
            assert all(r['target_range_enforced'] and r['target_abs_velocity']==[0.,.4,0.,0.,0.,0.,0.]
                       for r in states if r['mode']==1)
            result['checks'].append('tracker_limiter_and_applied_receipt_preserve_authorized_target_range')
        result['checks'].append('original_generation_sequence_token_certificate_and_deadline_preserved')
        sample=next(r for r in reversed(states) if r['mode']==1)
        exact=table[(sample['controller_session'],sample['source_sequence'])]
        bound=evaluate(now_ns=sample['t_ns'],state=sample,command=exact,velocity_limits=[2.]*7,
            acceleration_limits=[3.,3.,4.,4.,6.,6.,6.])
        assert bound['binding_consistent'] and not bound['qualified'] and not bound['runtime_admission']
        result['stopping_qualification']=bound
        restart=rospy.ServiceProxy('/lease_test/restart',Trigger)
        def reset():
            old=state[0].controller_session;assert restart().success
            wait(lambda:state[0].controller_session!=old and state[0].mode==0,'fresh controller session')
            time.sleep(.025)
        reset();g=grant();time.sleep(.012)
        # Keep sending fresh tracker-generation timestamps through the limiter.
        # The absolute grant still expires; neither intermediary can renew it.
        until=time.monotonic()+.18
        while time.monotonic()<until:send(g);time.sleep(.006)
        wait(lambda:state[0].mode==3,'fresh producer cannot extend lease')
        assert state[0].execution_stamp_ns==g.execution_stamp_ns
        result['checks'].append('continuous_fresh_source_cannot_extend_fixed_authorization')
        if args.bounded_targets:
            reset();g=grant();time.sleep(.012);original=send(g)
            wait(lambda:state[0].mode==1 and state[0].source_sequence==original.source_sequence,'bounded direct rejection setup')
            illegal_sequences=[]
            for forge in [False,True]:
                bad=copy.deepcopy(original);bad.header.stamp=rospy.Time.now()
                bad.source_sequence+=1000+(1 if forge else 0);illegal_sequences.append(bad.source_sequence)
                bad.velocity[0]=.2
                if forge:bad.target_abs_velocity=[.4]*7
                direct.publish(bad);time.sleep(.006)
            assert state[0].source_sequence not in illegal_sequences
            assert abs(state[0].applied_velocity[0])<1e-12
            wait(lambda:state[0].mode==3,'bounded rejection still reaches fixed stop')
            assert all(r['source_sequence'] not in illegal_sequences for r in records
                       if r['kind']=='state' and r['controller_session']==g.controller_session)
            result['checks'].append('actual_plugin_rejects_out_of_range_axis_and_forged_range_identity')
        reset();g=grant();time.sleep(.012)
        for _ in range(4):send(g);time.sleep(.006)
        wait(lambda:state[0].mode==1 and abs(state[0].applied_velocity[1])>0.,'moving before limiter suspension')
        os.kill(limiter_process.pid,signal.SIGSTOP)
        wait(lambda:state[0].mode==3,'controller stops without limiter callbacks')
        os.kill(limiter_process.pid,signal.SIGCONT)
        for _ in range(8):send(g);time.sleep(.006)
        assert state[0].mode==3
        result['checks'].append('limiter_sigstop_independent_braking_and_no_resume_on_sigcont')
        # A different post-stop lease plus zero and explicit rearm is required.
        g2=grant();time.sleep(.012);send(g2,0.);time.sleep(.012)
        assert state[0].mode==3
        rearm=rospy.ServiceProxy(BASE+'/rearm',Trigger);assert rearm().success
        wait(lambda:state[0].mode==1 and state[0].execution_stamp_ns==g2.execution_stamp_ns,'explicit new-token zero rearm')
        send(g2,.3)
        wait(lambda:abs(state[0].applied_velocity[1])>0.,'post-rearm fresh motion')
        send(g2,0.,True);time.sleep(.015)
        for _ in range(5):send(g2,.3);time.sleep(.005)
        wait(lambda:state[0].mode==3,'terminal stop remains latched')
        result['checks'].append('new_token_fresh_zero_explicit_rearm_and_terminal_stop')
        result['status']='PASS'
    finally:
        remember_descendants({p.pid for p in processes if p.poll() is None},owned)
        for p in processes:
            if p.poll() is None:
                try:os.kill(p.pid,signal.SIGCONT)
                except ProcessLookupError:pass
        for p in reversed(processes[1:]):
            if p.poll() is None:
                os.kill(p.pid,signal.SIGINT)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:pass
        result['cleanup']=reap_owned(owned)
        for p in processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        for h in handles:h.close()
        result['protected_identity_equal']=protected()==before
        with socket.socket() as sock:result['port_released']=sock.connect_ex(('127.0.0.1',args.port))!=0
        result['inputs_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in result['inputs_sha256'].items())
        (out/'trace.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in records))
        result['trace_rows']=len(records)
        result['trace_sha256']=hashlib.sha256((out/'trace.jsonl').read_bytes()).hexdigest()
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        assert not result['cleanup']['remaining'] and result['protected_identity_equal'] and result['port_released'] and result['inputs_unchanged']
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
