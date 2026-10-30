#!/usr/bin/env python3
"""Actual tracker/limiter/plugin ZERO-only handshake, with synthetic leases.

Nonconstant curves are protocol fixtures, not geometric certificates. No
nonzero command, qualified issuer, Gazebo physics, or formal task score.
"""
import argparse,copy,hashlib,json,os,signal,socket,subprocess,sys,tarfile,threading,time,xmlrpc.client
from pathlib import Path
from phase_e_owned_processes import identity,remember_descendants,reap_owned
from verify_phase_e_leased_response_gazebo import protected
ROOT=Path(__file__).resolve().parents[1]
BASE='/lease_test/controller'
TOPIC='/care_handoff_test'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--planner-prefix',type=Path,required=True)
    ap.add_argument('--port',type=int,default=12190)
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    before=protected()
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',args.port))!=0
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    prefix=args.planner_prefix.resolve();native=ROOT/'outputs/phase_e_trajectory_observation_20261020/native_final'
    binaries=[prefix/'lib/egocentric_arm_planner/trajectory_execution_manager_node',
              native/'leased_controller_host',native/'libcare_velocity_controller.so']
    inputs=set(json.loads((out.parent/'before_source_sha256.json').read_text()))
    inputs.update(str(p.relative_to(ROOT)) for p in (ROOT/'src/egocentric_arm_planner/scripts').glob('*.py'))
    inputs.update(str(p.relative_to(ROOT)) for p in (ROOT/'src/egocentric_arm_planner/include').rglob('*.hpp'))
    inputs.update(str(p.relative_to(ROOT)) for p in (prefix/'lib/python3/dist-packages').rglob('*.py'))
    inputs.update(str(p.relative_to(ROOT)) for p in (prefix/'lib/egocentric_arm_planner').glob('*.py'))
    inputs.update(['scripts/verify_phase_e_prepared_execution_ros.py','scripts/test_phase_e_execution_handoff.py',
                   'src/care_visibility_cdf/scripts/recovery_policy.py',
                   'scripts/test_phase_e_task_context.py','scripts/phase_e_owned_processes.py',
                   'scripts/verify_phase_e_leased_response_gazebo.py',
                   'src/arm_description/config/controllers_watchdog.yaml',
                   'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml'])
    inputs.update(str(p.relative_to(ROOT)) for p in binaries)
    for binary in binaries:
        for token in subprocess.check_output(['ldd',str(binary)],text=True).split():
            if token.startswith(str(ROOT)) and Path(token).is_file():inputs.add(str(Path(token).resolve().relative_to(ROOT)))
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(inputs)}
    (out/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with tarfile.open(out/'input_sources.tar.gz','w:gz',dereference=True) as archive:
        for p in hashes:archive.add(ROOT/p,arcname=p,recursive=False)
    (out/'runtime_environment.json').write_text(json.dumps({k:os.environ.get(k,'') for k in
        ['ROS_MASTER_URI','ROS_PACKAGE_PATH','CMAKE_PREFIX_PATH','PYTHONPATH','LD_LIBRARY_PATH']},indent=2)+'\n')
    children=[];handles=[];owned={};done=threading.Event();heartbeat=None;lock=threading.RLock()
    events=[];states=[];desired=[];limited=[];tracker_status=[];shadow_status=[];staged=[];leases=[]
    joint=[None];state=[None]
    result=dict(status='FAIL',scope=__doc__,runtime_admission=False,production_issuer_enabled=False,
                new_formal_windows=0,actuator_motion=False,checks=[],qualified_issuer=False)
    def start(argv,name):
        h=(out/(name+'.log')).open('x');handles.append(h)
        p=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);owned[p.pid]=identity(p.pid);return p
    def wait(predicate,label,timeout=10.):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            if predicate():return
            if any(p.poll() is not None for p in children):raise AssertionError('child exited: '+label)
            time.sleep(.003)
        raise AssertionError('timeout: '+label)
    def record(kind,value):events.append(dict(kind=kind,value=value))
    try:
        start(['roscore','-p',args.port],'master')
        def master():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/handoff_test')[0]==1
            except OSError:return False
        wait(master,'private master')
        import rospy,yaml
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
        from egocentric_arm_planner.msg import CandidateTransfer
        from care_velocity_controller.msg import CommandStatus,ExecutionLease,LeasedVelocityCommand
        from test_phase_e_execution_handoff import fixture,NAMES,VELOCITY,ACCELERATION
        from test_phase_e_task_context import command
        from task_observation_binding import ObservationTaskOwner,canonical
        from task_recovery_events import TaskEventSource,regime_identity
        from execution_preparation import ExecutionPreparationLedger
        from candidate_admission import payload_sha256
        from phase_e_leased_stop_evidence import status_record,command_record
        rospy.set_param('/use_sim_time',True);rospy.init_node('prepared_execution_test',disable_signals=True)
        cfg=yaml.safe_load((ROOT/'src/arm_description/config/controllers_watchdog.yaml').read_text())['care_arm']['arm_group_velocity_controller']
        cfg.update(require_execution_lease=True);rospy.set_param(BASE,cfg)
        def on_state(m):state[0]=m;states.append(status_record(m))
        rospy.Subscriber(BASE+'/state',CommandStatus,on_state,queue_size=100)
        rospy.Subscriber('/lease_test/joints',JointState,lambda m:joint.__setitem__(0,m))
        rospy.Subscriber(TOPIC+'/desired',LeasedVelocityCommand,lambda m:desired.append(command_record(m)),queue_size=100)
        rospy.Subscriber(BASE+'/command_leased',LeasedVelocityCommand,lambda m:limited.append(command_record(m)),queue_size=100)
        rospy.Subscriber(TOPIC+'/tracker_status',String,lambda m:tracker_status.append(json.loads(m.data)),queue_size=100)
        rospy.Subscriber(TOPIC+'/shadow_status',String,lambda m:shadow_status.append(json.loads(m.data)),queue_size=100)
        rospy.Subscriber(TOPIC+'/staged',CandidateTransfer,lambda m:staged.append(dict(session=m.session,sequence=m.sequence,
            digest=m.payload_sha256,remaining_budget_s=m.remaining_budget_s,
            expires_ns=json.loads(m.planned_prefix_context)['expires_ns'])),queue_size=100)
        contexts=rospy.Publisher(TOPIC+'/context',String,queue_size=10)
        controls=rospy.Publisher(TOPIC+'/control',String,queue_size=10)
        preparations=rospy.Publisher(TOPIC+'/prepared',CandidateTransfer,queue_size=10)
        direct_stages=rospy.Publisher(TOPIC+'/staged',CandidateTransfer,queue_size=10)
        raw=rospy.Publisher(TOPIC+'/raw',JointTrajectory,queue_size=1)
        grants=rospy.Publisher(BASE+'/execution_lease',ExecutionLease,queue_size=10)
        start([native/'leased_controller_host',native/'libcare_velocity_controller.so'],'controller')
        wait(lambda:state[0] is not None and joint[0] is not None and rospy.Time.now().to_sec()>10.,'controller receipt')
        cfg=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        cfg['execution'].update(require_command_lease=True,require_prepared_execution=True,primitive_tracking_guard=False,
            joint_states='/lease_test/joints',input_trajectory=TOPIC+'/raw',prepared_trajectory_topic=TOPIC+'/staged',
            prepared_status_topic=TOPIC+'/tracker_status',task_context_topic=TOPIC+'/context',task_control_topic=TOPIC+'/control',
            controller_state_topic=BASE+'/state',lease_topic=BASE+'/execution_lease',leased_command_topic=TOPIC+'/desired')
        rospy.set_param('/handoff_tracker',cfg)
        tracker_process=start([binaries[0],'__name:=handoff_tracker'],'tracker')
        cfg.update(velocity_limits=VELOCITY,acceleration_limits=ACCELERATION,
            prepared_topic=TOPIC+'/prepared',staged_topic=TOPIC+'/staged',status_topic=TOPIC+'/shadow_status',
            tracker_status_topic=TOPIC+'/tracker_status',
            task_context_topic=TOPIC+'/context',task_control_topic=TOPIC+'/control',controller_state_topic=BASE+'/state')
        rospy.set_param('/handoff_shadow',cfg)
        start(['/usr/bin/python3',prefix/'lib/egocentric_arm_planner/execution_preparation_handoff_node.py','__name:=handoff_shadow'],'shadow')
        rospy.set_param('/handoff_limiter',dict(joint_names=NAMES,require_execution_lease=True,
            leased_input_topic=TOPIC+'/desired',leased_output_topic=BASE+'/command_leased',controller_state_topic=BASE+'/state',
            vbc_safety_hold_topic=TOPIC+'/vbc',external_safety_hold_topic=TOPIC+'/gcdf',
            mpc=dict(joint_velocity_limits=VELOCITY,joint_acceleration_limits=ACCELERATION)))
        start(['/usr/bin/python3',prefix/'lib/egocentric_arm_planner/joint_velocity_rate_limiter_node.py','__name:=handoff_limiter'],'limiter')
        wait(lambda:contexts.get_num_connections()==2 and controls.get_num_connections()==2 and
            preparations.get_num_connections()==1 and direct_stages.get_num_connections()>=2 and
            grants.get_num_connections()==2 and tracker_status and shadow_status,'all connections')
        def output_ready(node,topic):
            code,_,address=xmlrpc.client.ServerProxy(uri).lookupNode('/handoff_test',node)
            if code!=1:return False
            code,_,bus=xmlrpc.client.ServerProxy(address).getBusInfo('/handoff_test')
            return code==1 and sum(1 for row in bus if row[2]=='o' and row[4]==topic and row[5])>=2
        wait(lambda:output_ready('/handoff_shadow',TOPIC+'/staged') and
             output_ready('/handoff_limiter',BASE+'/command_leased'),'both output connections')
        owner=ObservationTaskOwner('ros-handoff-owner');source=TaskEventSource('handoff-regime')
        current=dict(context=None,control=None,revision=0)
        def publish_context():
            with lock:
                now=rospy.Time.now().to_nsec()
                rows=owner.tick(now)
                if not rows:return
                context=rows[-1];contexts.publish(String(data=canonical(context)))
                control=source.event('regime_control',context,dict(mode='NORMAL',mode_revision=current['revision'],ready=True,
                    clear=False,replan_seq=0,reference_refresh_seq=0,force_bootstrap=False,qp_retry=''),now)
                controls.publish(String(data=canonical(control)));current.update(context=context,control=control)
                record('context',context);record('control',control)
        def beat():
            while not done.wait(.015):publish_context()
        heartbeat=threading.Thread(target=beat);heartbeat.start()
        restart=rospy.ServiceProxy('/lease_test/restart',Trigger)
        trial=0;sequence=0
        def begin(label,reset=True,pause_tracker=False):
            nonlocal trial
            if reset:
                old=state[0].controller_session;assert restart().success
                wait(lambda:state[0].controller_session!=old and state[0].mode==0,'fresh session')
            with lock:
                trial+=1;current['revision']+=1
                owner.command(command('task'+str(trial)),rospy.Time.now().to_nsec())
                publish_context();identity_value=regime_identity(current['control'])
            wait(lambda:tracker_status[-1]['controller_session']==state[0].controller_session and
                tracker_status[-1]['task_regime']==identity_value and shadow_status[-1]['task_regime']==identity_value,'context accepted')
            with lock:
                now=rospy.Time.now().to_nsec();context=current['context'];control=current['control']
                curve=fixture(now)[2].trajectory
                ledger=ExecutionPreparationLedger()
                msg=ledger.prepare(curve,context['candidate_binding'],identity_value,('fixture'+str(trial),1),
                    payload_sha256(curve,'',context['candidate_binding'],identity_value),1,now,
                    min(now+200000000,control['valid_until_ns'],context['valid_until_ns']))
                if pause_tracker:os.kill(tracker_process.pid,signal.SIGSTOP)
                preparations.publish(msg);record('prepare',dict(label=label,digest=msg.payload_sha256,
                    metadata=json.loads(msg.planned_prefix_context),token=msg.trajectory.header.stamp.to_nsec()))
            if pause_tracker:
                try:time.sleep(.045)
                finally:os.kill(tracker_process.pid,signal.SIGCONT)
            wait(lambda:tracker_status[-1]['pending'] and tracker_status[-1]['preparation_sha256']==msg.payload_sha256,'native preparation staged',1.)
            return msg
        message=begin('zero_handshake',pause_tracker=True)
        attempts=[s for s in staged if s['session']==message.session and s['sequence']==message.sequence]
        assert len(attempts)>=2 and all(s['digest']==message.payload_sha256 for s in attempts)
        assert attempts[-1]['remaining_budget_s']<attempts[0]['remaining_budget_s']
        assert all(s['expires_ns']==json.loads(message.planned_prefix_context)['expires_ns'] for s in attempts)
        result['checks'].append('paused_native_receiver_gets_same_preparation_identity_within_first_deadline')
        count=len(desired);raw.publish(message.trajectory);time.sleep(.025)
        assert len(desired)==count and state[0].mode==0
        assert tracker_status[-1]['phase_origin_ns']==0 and tracker_status[-1]['phase_s']==0
        result['checks'].append('staging_and_bare_commit_never_advance_phase_or_generate_motion')
        bad=copy.deepcopy(message)
        for point in bad.trajectory.points:point.positions[0]+=.01
        direct_stages.publish(bad);time.sleep(.01)
        assert tracker_status[-1]['preparation_sha256']==message.payload_sha256
        result['checks'].append('native_tracker_independently_rejects_mutated_payload_digest')
        sequence+=1;g=ExecutionLease();g.controller_session=state[0].controller_session
        g.execution_stamp_ns=message.trajectory.header.stamp.to_nsec();g.lease_sequence=sequence
        g.issued_at=rospy.Time.now();g.valid_until=rospy.Time(0,min(g.issued_at.to_nsec()+80000000,
            json.loads(message.planned_prefix_context)['expires_ns']))
        g.certificate_sha256='a'*64;g.target_range_enforced=True;g.target_abs_velocity=[0.]*7
        grants.publish(g);leases.append(dict(session=g.controller_session,token=g.execution_stamp_ns,sequence=g.lease_sequence,
            issued_ns=g.issued_at.to_nsec(),until_ns=g.valid_until.to_nsec(),synthetic_fixture=True,target_abs_velocity=[0.]*7))
        wait(lambda:any(s['zero_applied'] and s['execution_stamp_ns']==g.execution_stamp_ns for s in tracker_status),'actual zero application',1.)
        application=next(s for s in tracker_status if s['zero_applied'] and s['execution_stamp_ns']==g.execution_stamp_ns)
        assert application['zero_application_source_sequence']>0 and not application['execution_authorized']
        assert application['phase_s']==0 and application['phase_origin_ns']==0
        result['checks'].append('actual_tracker_limiter_plugin_receipt_confirms_same_token_zero_without_activation')
        preparations.publish(message);direct_stages.publish(message)
        wait(lambda:state[0].mode==3,'fixed zero lease expires')
        assert state[0].execution_stamp_ns==g.execution_stamp_ns
        result['checks'].append('zero_authorization_keeps_its_fixed_deadline_and_retires_token')
        direct_stages.publish(message);preparations.publish(message);time.sleep(.025)
        wait(lambda:not tracker_status[-1]['pending'] and state[0].mode==3,'tracker observes retired controller token',1.)
        assert not tracker_status[-1]['pending'] and state[0].mode==3
        result['checks'].append('late_preparation_cannot_revive_retired_token')
        expired=begin('no_authorization_expiry')
        wait(lambda:not tracker_status[-1]['pending'],'preparation expires without lease',1.)
        direct_stages.publish(expired);preparations.publish(expired);time.sleep(.025)
        assert not tracker_status[-1]['pending'] and state[0].mode==0
        result['checks'].append('first_preparation_deadline_expires_without_implicit_renewal')
        old=begin('task_replacement')
        with lock:
            current['revision']+=1;trial+=1;owner.command(command('task'+str(trial)),rospy.Time.now().to_nsec());publish_context()
        wait(lambda:not tracker_status[-1]['pending'],'new task invalidates preparation')
        direct_stages.publish(old);preparations.publish(old);time.sleep(.025)
        assert not tracker_status[-1]['pending']
        result['checks'].append('real_task_replacement_rejects_late_old_preparation')
        begin('controller_restart')
        old_session=state[0].controller_session;assert restart().success
        wait(lambda:state[0].controller_session!=old_session and not tracker_status[-1]['pending'],'controller restart invalidation')
        result['checks'].append('controller_restart_invalidates_pending_and_requires_new_handshake')
        assert desired and limited and all(not any(row['velocity']) for row in desired+limited)
        assert all(not any(row['applied']) and not any(row['measured']) for row in states)
        assert all(not row['execution_authorized'] and not row['activation_available'] and row['phase_origin_ns']==0 for row in tracker_status)
        assert all(not row['runtime_admission'] and len(row['missing_qualification'])==4 for row in shadow_status)
        desired_by={(r['controller_session'],r['source_sequence']):r for r in desired}
        for row in limited:
            origin=desired_by[row['controller_session'],row['source_sequence']]
            for key in ['generated_ns','valid_until_ns','execution_stamp_ns','lease_sequence','certificate_sha256','target_abs_velocity']:
                assert row[key]==origin[key],key
        result['checks'].append('all_outputs_zero_all_four_qualifications_unmet_limiter_preserves_source_identity')
        loaded={}
        for child in children:
            paths={line.split()[-1] for line in Path('/proc',str(child.pid),'maps').read_text().splitlines()
                   if line.split() and line.split()[-1].startswith(str(ROOT)) and Path(line.split()[-1]).is_file()}
            loaded[str(child.pid)]={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(paths)}
        (out/'loaded_project_objects.json').write_text(json.dumps(loaded,indent=2)+'\n')
        result.update(status='PASS_ZERO_HANDSHAKE_ONLY',zero_application=application,zero_application_receipts=1)
    finally:
        done.set()
        if heartbeat:heartbeat.join(2.)
        remember_descendants({p.pid for p in children if p.poll() is None},owned)
        result['cleanup']=reap_owned(owned)
        for p in children:
            try:p.wait(timeout=3.)
            except subprocess.TimeoutExpired:pass
        for h in handles:h.close()
        result['protected_identity_equal']=protected()==before
        with socket.socket() as s:result['port_released']=s.connect_ex(('127.0.0.1',args.port))!=0
        result['inputs_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        for name,value in dict(events=events,states=states,desired=desired,limited=limited,tracker_status=tracker_status,
            shadow_status=shadow_status,staged=staged,synthetic_zero_leases=leases).items():
            (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        assert not result['cleanup']['remaining'] and result['protected_identity_equal'] and result['port_released'] and result['inputs_unchanged']
    assert result['status']=='PASS_ZERO_HANDSHAKE_ONLY'
    print('PREPARED_ZERO_HANDSHAKE_PASS',len(result['checks']),'checks; actual motion0')

if __name__=='__main__':main()
