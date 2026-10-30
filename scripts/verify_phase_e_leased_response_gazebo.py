#!/usr/bin/env python3
"""Private Gazebo response study with synthetic leases and passive map checks.

This is a diagnostic, with no production authorizer or formal task score.
Finite physics-step observations do not certify an actuator response bound.
"""
import argparse,copy,csv,hashlib,io,json,os,shlex,signal,socket,subprocess
import tarfile,threading,time,traceback,xml.etree.ElementTree as ET,xmlrpc.client,zipfile
from pathlib import Path
import yaml
from phase_e_owned_processes import identity,remember_descendants,reap_owned

ROOT=Path(__file__).resolve().parents[1]
NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
BASE='/care_arm/arm_group_velocity_controller'

def protected():
    expected=json.loads((ROOT/'outputs/phase_e_soft_mount_targeted_20261019/final_host_check_after_restore.json').read_text())['protected_processes']
    rows={}
    for pid,wanted in expected.items():
        p=Path('/proc')/pid
        rows[pid]=dict(starttime=(p/'stat').read_text().rsplit(')',1)[1].split()[19],
                       command_sha256=hashlib.sha256((p/'cmdline').read_bytes()).hexdigest())
        assert all(rows[pid][k]==wanted[k] for k in rows[pid]),pid
    return rows

def historical_profile():
    case=next(x for x in json.loads((ROOT/'outputs/phase_e_soft_mount_targeted_20261019/results_partial.json').read_text()) if x['tag']=='r2_010')
    archive=ROOT/case['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==case['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        raw=z.read('run/joint_states.csv')
        rows=list(csv.DictReader(io.StringIO(raw.decode())))
        chosen=min(rows,key=lambda r:abs(float(r['field.header.stamp'])*1e-9-25.12))
        q=[float(chosen['field.position'+str(j)]) for j in range(7)]
        v=[float(chosen['field.velocity'+str(j)]) for j in range(7)]
    bounded=[max(-2.,min(2.,value)) for value in v]
    return q,bounded,dict(archive=case['archive'],member='run/joint_states.csv',
        member_sha256=hashlib.sha256(raw).hexdigest(),sample_stamp_ns=int(chosen['field.header.stamp']),
        historical_measured_velocity=v,diagnostic_target_velocity=bounded,
        note='historical measured velocity clipped to current command limits; no historical trajectory replay')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--controller-prefix',type=Path,required=True)
    parser.add_argument('--port',type=int,default=12090)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--bounded-targets',action='store_true')
    parser.add_argument('--local-geometry',action='store_true')
    parser.add_argument('--adaptive-geometry',action='store_true')
    parser.add_argument('--map-binary',type=Path,default=ROOT/'devel/lib/care_confidence_map/confidence_map_node')
    parser.add_argument('--observation-trace-selection',type=Path)
    parser.add_argument('--observer-library',type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    prefix=args.controller_prefix.resolve();map_binary=args.map_binary.resolve();before=protected()
    for port in [args.port,args.port+1]:
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,'private port busy'
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',GAZEBO_MASTER_URI='http://127.0.0.1:'+str(args.port+1),
        GAZEBO_MODEL_DATABASE_URI='',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    paths=[ROOT/'scripts'/name for name in ['phase_e_stopping_contract.py','phase_e_leased_stop_evidence.py',
        'phase_e_leased_shadow_node.py','phase_e_physics_response_observer.cpp','verify_phase_e_leased_response_gazebo.py']]
    paths += [ROOT/'scripts/phase_e_local_stop_geometry.py',ROOT/'scripts/phase_e_stop_shadow.py']
    paths += [ROOT/'scripts'/name for name in ['phase_e_stop_shadow_node.py','phase_e_observation_evidence.py',
        'phase_e_stop_envelope.py','phase_e_owned_processes.py','audit_case001_motion_prior.py']]
    paths += [ROOT/'src/care_velocity_controller/src/expiring_velocity_controller.cpp',
        ROOT/'src/care_velocity_controller/include/care_velocity_controller/leased_velocity_core.hpp',
        ROOT/'src/care_velocity_controller/include/care_velocity_controller/expiring_velocity_core.hpp',
        ROOT/'src/arm_description/urdf/Arm.urdf',ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
        ROOT/'src/arm_description/worlds/maixsense_empty.world',ROOT/'src/arm_description/config/controllers_watchdog.yaml',
        ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml',
        ROOT/'src/care_confidence_map/src/confidence_map_node.cpp',map_binary,
        prefix/'lib/libcare_velocity_controller.so']
    paths += list((prefix/'lib/python3/dist-packages/care_velocity_controller/msg').glob('*.py'))
    if args.observation_trace_selection:paths.append(args.observation_trace_selection.resolve())
    if args.observer_library:paths.append(args.observer_library.resolve())
    result=dict(status='FAIL',scope='synthetic fixed leases in private Gazebo; observations, not a safety certificate',
        runtime_admission=False,physical_bound_certified=False,new_formal_windows=0,trials=[],
        bounded_targets=args.bounded_targets,local_geometry=args.local_geometry,adaptive_geometry=args.adaptive_geometry,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    with tarfile.open(out/'input_sources.tar.gz','w:gz') as tar:
        for p in paths:
            data=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(ROOT)));info.size=len(data)
            tar.addfile(info,io.BytesIO(data))
    processes=[];handles=[];owned={};stop_sender=threading.Event();sender=None
    lock=threading.Lock();records=[];commands=[];leases=[];state=[None]
    trace=(out/'controller_trace.jsonl').open('x')
    def start(argv,label):
        handle=(out/(label+'.log')).open('x');handles.append(handle)
        process=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);owned[process.pid]=identity(process.pid);return process
    last_scan=[0.]
    def wait(predicate,label,timeout=30.):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if time.monotonic()-last_scan[0]>.5:
                remember_descendants({p.pid for p in processes if p.poll() is None},owned);last_scan[0]=time.monotonic()
            if predicate():return
            if any(p.poll() is not None for p in processes):raise AssertionError('child exited: '+label)
            time.sleep(.005)
        raise AssertionError('timeout: '+label)
    def last():
        with lock:return copy.deepcopy(state[0])
    def event(kind,row):
        with lock:
            trace.write(json.dumps(dict(kind=kind,**row))+'\n')
    try:
        result['stage']='observer build'
        observer_library=out/'libphysics_response_observer.so'
        if args.observer_library:
            observer_library.write_bytes(args.observer_library.read_bytes())
            result['observer_reused_from']=str(args.observer_library.resolve())
        else:
            flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','gazebo'],text=True))
            command=['g++','-shared','-fPIC',str(ROOT/'scripts/phase_e_physics_response_observer.cpp'),
                     '-o',str(observer_library)]+flags
            result['observer_build_command']=command
            subprocess.run(command,stdout=(out/'observer_build.log').open('x'),stderr=subprocess.STDOUT,check=True)
        robot=ET.parse(ROOT/'src/arm_description/urdf/Arm.urdf')
        plugin=ET.SubElement(ET.SubElement(robot.getroot(),'gazebo'),'plugin',
            name='care_physics_response_observer',filename=str(out/'libphysics_response_observer.so'))
        ET.SubElement(plugin,'output').text=str(out/'physics_steps.jsonl')
        robot.write(out/'observed_robot.urdf',encoding='unicode')
        result['stage']='private master'
        start(['roscore','-p',str(args.port)],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/response_study')[0]==1
            except OSError:return False
        wait(ready,'master',15.)
        import rospy
        from care_velocity_controller.msg import CommandStatus,ExecutionLease,LeasedVelocityCommand
        from phase_e_leased_stop_evidence import status_record,command_record
        from std_srvs.srv import Trigger
        from controller_manager_msgs.srv import ListControllers
        def ros_time(ns):return rospy.Time(ns//1000000000,ns%1000000000)
        rospy.set_param('/use_sim_time',True);rospy.init_node('leased_response_study',disable_signals=True)
        def on_state(msg):
            row=status_record(msg)
            row['integral']=list(msg.applied_position_integral)
            with lock:state[0]=row;records.append(row)
            event('state',row)
        rospy.Subscriber(BASE+'/state',CommandStatus,on_state,queue_size=1000,tcp_nodelay=True)
        authority=rospy.Publisher(BASE+'/execution_lease',ExecutionLease,queue_size=10,tcp_nodelay=True)
        direct=rospy.Publisher(BASE+'/command_leased',LeasedVelocityCommand,queue_size=10,tcp_nodelay=True)
        result['stage']='Gazebo startup'
        start(['roslaunch','arm_description','gazebo_velocity_control.launch','use_command_watchdog:=true',
            'require_execution_lease:=true','gazebo_gui:=false','use_rviz:=false',
            'apply_initial_joint_overrides:=false','robot_urdf:='+str(out/'observed_robot.urdf')],'gazebo')
        wait(lambda:last() is not None and authority.get_num_connections()>=1 and direct.get_num_connections()>=1,
             'controller subscriptions and state',90.)
        selected=next(c for c in rospy.ServiceProxy('/care_arm/controller_manager/list_controllers',ListControllers)().controller
                      if c.name=='arm_group_velocity_controller')
        assert selected.state=='running' and selected.type=='care_velocity_controller/ExpiringJointVelocityController'
        assert last()['lease_required'] and last()['mode']==0
        remember_descendants({p.pid for p in processes if p.poll() is None},owned)
        loaded=[]
        for pid in owned:
            p=Path('/proc')/str(pid)
            try:
                if b'gzserver' in (p/'cmdline').read_bytes():
                    loaded += [line.split()[-1] for line in (p/'maps').read_text().splitlines() if 'libcare_velocity_controller.so' in line]
            except FileNotFoundError:pass
        assert str(prefix/'lib/libcare_velocity_controller.so') in loaded,loaded
        assert str(ROOT/'devel/lib/libcare_velocity_controller.so') not in loaded
        result['loaded_controller_libraries']=sorted(set(loaded))
        result['stage']='map and passive observer'
        config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml').read_text())
        config['confidence_map'].update(query_service='/shadow_map/query')
        if args.observation_trace_selection:
            selection=json.loads(args.observation_trace_selection.read_text())
            config['confidence_map'].update(observation_trace_file=str(out/'observations.jsonl'),
                observation_trace_watch_grid_indices=selection['watch_grid_indices'])
        config['current_body_prior'].update(geometry_backend='primitive',
            primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            refresh_service='/shadow_map/refresh',deactivate_service='/shadow_map/deactivate')
        rospy.set_param('/shadow_map',config)
        result['map_observation_mode']=config['confidence_map']['observation_mode']
        start([map_binary,'__name:=shadow_map'],'map')
        rospy.wait_for_service('/shadow_map/refresh',timeout=20.)
        refresh=rospy.ServiceProxy('/shadow_map/refresh',Trigger)()
        assert refresh.success and 'skipped_samples=0' in refresh.message,refresh.message
        shadow=start(['/home/zhicheng/miniconda3/envs/viscdf/bin/python',ROOT/'scripts/phase_e_leased_shadow_node.py',
            '--output-dir',out/'shadow']+(['--local-geometry'] if args.local_geometry else [])+
            (['--adaptive-geometry'] if args.adaptive_geometry else []),'shadow')
        control={'kind':'velocity','value':[0.]*7,'renew':True,'token':rospy.Time.now().to_nsec()}
        send_lock=threading.Lock();current=[None];failure=[]
        def set_control(**values):
            with send_lock:control.update(values)
        def send():
            seq=0;lease_seq=0;last_generation=0
            try:
                while not stop_sender.is_set():
                    s=last();now=rospy.Time.now().to_nsec()
                    with send_lock:c=copy.deepcopy(control)
                    if s and now>last_generation:
                        grant=current[0]
                        cap=([.5]*7 if c['kind']=='pose' else [abs(v) for v in c['value']]) if args.bounded_targets else [0.]*7
                        if c['renew'] and (not grant or grant.execution_stamp_ns!=c['token'] or
                            list(grant.target_abs_velocity)!=cap or now-grant.issued_at.to_nsec()>=15000000):
                            grant=ExecutionLease();grant.controller_session=s['controller_session'];grant.execution_stamp_ns=c['token']
                            lease_seq+=1;grant.lease_sequence=lease_seq;grant.issued_at=ros_time(now)
                            grant.valid_until=ros_time(now+100000000);grant.certificate_sha256='a'*64
                            grant.target_range_enforced=args.bounded_targets;grant.target_abs_velocity=cap
                            current[0]=grant;authority.publish(grant)
                            row=dict(session=grant.controller_session,token=c['token'],sequence=lease_seq,
                                     issued_ns=now,until_ns=now+100000000,
                                     target_range_enforced=grant.target_range_enforced,target_abs_velocity=list(grant.target_abs_velocity))
                            with lock:leases.append(row)
                            event('synthetic_lease',row)
                        if grant:
                            velocity=c['value'] if c['kind']=='velocity' else [max(-.5,min(.5,3.*(t-q))) for t,q in zip(c['value'],s['q'])]
                            m=LeasedVelocityCommand();m.header.stamp=ros_time(now)
                            seq+=1;m.source_sequence=seq;m.controller_session=grant.controller_session
                            m.execution_stamp_ns=grant.execution_stamp_ns;m.lease_sequence=grant.lease_sequence
                            m.valid_until=grant.valid_until;m.certificate_sha256=grant.certificate_sha256
                            m.target_range_enforced=grant.target_range_enforced;m.target_abs_velocity=grant.target_abs_velocity
                            m.joint_names=NAMES;m.velocity=velocity;direct.publish(m)
                            row=command_record(m)
                            with lock:commands.append(row)
                            event('command',row);last_generation=now
                    time.sleep(.001)
            except BaseException:failure.append(traceback.format_exc())
        sender=threading.Thread(target=send,daemon=True);sender.start()
        rearm=rospy.ServiceProxy(BASE+'/rearm',Trigger)
        def move(target):
            set_control(kind='pose',value=target,renew=True)
            wait(lambda:max(abs(a-b) for a,b in zip(last()['q'],target))<.003 and
                 max(map(abs,last()['measured']))<.01,'pose reached',120.)
            set_control(kind='velocity',value=[0.]*7)
            tick=last()['t_ns'];wait(lambda:last()['t_ns']>=tick+100000000 and max(map(abs,last()['measured']))<.001,'pose rest')
            assert last()['mode']==1
        def arm_next():
            count=last()['rearm_successes']
            set_control(kind='velocity',value=[0.]*7,renew=True,token=rospy.Time.now().to_nsec())
            tick=last()['t_ns'];wait(lambda:last()['t_ns']>=tick+50000000,'post-stop zero')
            rearm();wait(lambda:last()['rearm_successes']>count,'new-token explicit rearm')
        wait(lambda:last()['mode']==1,'initial synthetic zero')
        trials=[]
        for j in range(7):
            for sign in [-1,1]:
                v=[0.]*7;v[j]=sign*2.;q=[0.]*7;q[j]=-sign*.65
                trials.append(dict(label=NAMES[j]+'_2_'+str(sign),q=q,v=v))
        for speed in [.3,1.]:
            for j in [0,3,5]:
                v=[0.]*7;v[j]=speed*(-1 if j%2 else 1)
                trials.append(dict(label=NAMES[j]+'_'+str(speed),q=[0.]*7,v=v))
        for sign in [-1,1]:
            v=[sign*2.*(-1 if j%2 else 1) for j in range(7)]
            trials.append(dict(label='coupled_2_'+str(sign),q=[-.65*(1 if x>0 else -1) for x in v],v=v))
        q010,v010,source010=historical_profile();result['historical_profile_source']=source010
        trials += [dict(label='010_velocity_direction',q=q010,v=v010),
                   dict(label='010_reverse_direction',q=q010,v=[-v for v in v010])]
        if args.smoke:trials=trials[:1]
        result['planned_trials']=len(trials)
        for index,trial in enumerate(trials):
            result['stage']=trial['label'];move(trial['q'])
            set_control(kind='velocity',value=trial['v'])
            wait(lambda:max(abs(a-b) for a,b in zip(last()['applied'],trial['v']))<1e-5,'target applied velocity',45.)
            tick=last()['t_ns'];wait(lambda:last()['t_ns']>=tick+30000000,'short steady interval')
            baseline=last();set_control(renew=False)
            wait(lambda:last()['mode']==3,'fixed lease expires and brakes')
            stopped=last();tick=stopped['t_ns']
            wait(lambda:last()['t_ns']>=tick+150000000 and max(map(abs,last()['measured']))<=.001,'measured rest',15.)
            end=last()
            with lock:segment=[r for r in records if baseline['t_ns']<=r['t_ns']<=end['t_ns']]
            assert segment and all(r['integral_valid'] and not r['clock_fault'] for r in segment)
            assert not failure,failure
            trial.update(index=index,baseline_ns=baseline['t_ns'],end_ns=end['t_ns'],
                fixed_until_ns=stopped['valid_until_ns'],command_stopped_sample_ns=stopped['t_ns'],
                baseline_q=baseline['q'],baseline_applied=baseline['applied'],
                sampled_error_rad=[max(abs((r['q'][j]-baseline['q'][j])-(r['integral'][j]-baseline['integral'][j])) for r in segment) for j in range(7)])
            result['trials'].append(trial)
            (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(dict(index=index,label=trial['label'],sampled_error=trial['sampled_error_rad'])),flush=True)
            arm_next()
        move([0.]*7)
        result['stage']='complete';result['status']='PASS'
    except BaseException:
        result['error']=traceback.format_exc();raise
    finally:
        stop_sender.set()
        if sender:sender.join(3)
        remember_descendants({p.pid for p in processes if p.poll() is None},owned)
        result['cleanup']=reap_owned(owned)
        for process in processes:
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:pass
        for handle in handles:handle.close()
        trace.close()
        result['protected_identity_equal']=protected()==before
        result['input_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in result['input_sha256'].items())
        result['port_released']={}
        for port in [args.port,args.port+1]:
            with socket.socket() as sock:result['port_released'][str(port)]=sock.connect_ex(('127.0.0.1',port))!=0
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        assert not result['cleanup']['remaining'] and result['protected_identity_equal'] and result['input_unchanged']
        assert all(result['port_released'].values())

if __name__=='__main__':main()
