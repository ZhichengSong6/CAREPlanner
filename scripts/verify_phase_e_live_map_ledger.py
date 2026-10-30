#!/usr/bin/env python3
"""Explicit private-Gazebo integration of real map, serial ledger and controller.

Diagnostic command envelopes only. The experiment deliberately exercises failed
renewals; neither this driver nor its records enable production admission.
"""
import argparse,collections,copy,gc,gzip,hashlib,io,json,os,socket,subprocess,sys,tarfile,threading,time,traceback,xmlrpc.client
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.append('/usr/lib/python3/dist-packages')
import yaml
from phase_e_owned_processes import identity,remember_descendants,reap_owned
from phase_e_serial_lease_ledger import SerialLeaseLedger,LedgerReject
from phase_e_local_stop_geometry import LocalPrimitiveEnclosure
from verify_phase_e_leased_response_gazebo import protected,NAMES,BASE

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--map-binary',type=Path,required=True)
    ap.add_argument('--controller-prefix',type=Path,required=True)
    ap.add_argument('--diagnostic-private-master',action='store_true',required=True)
    ap.add_argument('--port',type=int,default=12132)
    ap.add_argument('--observation-prefix-scenario',type=Path)
    args=ap.parse_args()
    if args.port in (11311,11548,11549) or not 12000<=args.port<=12998:
        ap.error('Use an isolated diagnostic master in the reserved test range')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    prefix=args.controller_prefix.resolve();binary=args.map_binary.resolve();before=protected()
    for port in [args.port,args.port+1]:
        with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',GAZEBO_MASTER_URI='http://127.0.0.1:'+str(args.port+1),
        GAZEBO_MODEL_DATABASE_URI='',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    observer=ROOT/'outputs/phase_e_adaptive_ledger_20261019/gazebo_adaptive/libphysics_response_observer.so'
    paths=[Path(__file__).resolve(),binary,observer,prefix/'lib/libcare_velocity_controller.so']
    paths += [ROOT/'scripts'/name for name in ['phase_e_serial_lease_ledger.py','phase_e_leased_stop_evidence.py',
        'phase_e_local_stop_geometry.py','phase_e_observation_evidence.py','phase_e_stop_shadow.py',
        'phase_e_stop_shadow_node.py','phase_e_stopping_contract.py','phase_e_stop_envelope.py',
        'verify_phase_e_leased_response_gazebo.py','phase_e_owned_processes.py','audit_case001_motion_prior.py']]
    paths += [ROOT/'src'/name for name in ['arm_description/urdf/Arm.urdf','arm_description/urdf/Arm_with_self_filter_collision.urdf',
        'arm_description/worlds/maixsense_empty.world','arm_description/config/controllers_watchdog.yaml',
        'care_confidence_map/src/confidence_map_node.cpp','care_confidence_map/config/confidence_map_case001_motion_prior.yaml']]
    paths+=list((prefix/'lib/python3/dist-packages/care_velocity_controller/msg').glob('*.py'))
    paths.append(ROOT/'outputs/phase_e_observation_ledger_20261020/memory_ros/selection.json')
    if args.observation_prefix_scenario:
        paths += [args.observation_prefix_scenario.resolve(),ROOT/'scripts/phase_e_observation_prefix_gazebo.py',
            ROOT/'src/care_visibility_cdf/scripts/stopping_observation_demands.py',
            ROOT/'src/care_visibility_cdf/scripts/vbc_multi_deadline_obligation_impl.py',
            ROOT/'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py',
            ROOT/'src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py',
            ROOT/'src/care_visibility_cdf/scripts/pending_vbc_blockers.py',
            ROOT/'src/care_visibility_cdf/CMakeLists.txt']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with tarfile.open(out/'input_sources.tar.gz','w:gz') as tar:
        for name in hashes:
            data=(ROOT/name).read_bytes();info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    (out/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    result=dict(status='RUNNING',runtime_admission=False,production_issuer_enabled=False,new_formal_windows=0,
        scope='Real ROS map + serial command-domain ledger + actual isolated controller/Gazebo; physical qualification remains unmet',
        runtime_fingerprint_sha256=fingerprint,checks=[],phases=[],rejections=[])
    children=[];handles=[];owned={};done=threading.Event();sender=None
    state_lock=threading.Lock();send_lock=threading.Lock();trace_lock=threading.Lock()
    latest=[None];records=collections.deque(maxlen=4000);errors=[];current=[None];target=[0.]*7;drop_ack_until=[0];pose_target=[None]
    ledger=SerialLeaseLedger(fingerprint)
    protocol_stats=dict(commits=0,sensor_commits=0,peak_buffered_events=0,sensor_samples=[])
    checked_summary={};gc_starts={}
    gc_stats=dict(count=0,maximum_pause_ms=0.,slow_examples=[])
    def gc_callback(phase,info):
        if phase=='start':gc_starts[info['generation']]=time.monotonic()
        elif info['generation'] in gc_starts:
            elapsed=(time.monotonic()-gc_starts.pop(info['generation']))*1000
            gc_stats['count']+=1;gc_stats['maximum_pause_ms']=max(gc_stats['maximum_pause_ms'],elapsed)
            if elapsed>5 and len(gc_stats['slow_examples'])<64:
                gc_stats['slow_examples'].append(dict(generation=info['generation'],pause_ms=elapsed))
    gc.callbacks.append(gc_callback)
    trace=(out/'controller_trace.jsonl').open('x');events=gzip.open(out/'ledger_events.jsonl.gz','xt',compresslevel=1)
    points=gzip.open(out/'point_evidence.jsonl.gz','xt',compresslevel=1)
    def event(kind,**fields):
        with trace_lock:trace.write(json.dumps(dict(kind=kind,**fields))+'\n')
    def drain_events():
        values=ledger.drain_events()
        protocol_stats['peak_buffered_events']=max(protocol_stats['peak_buffered_events'],len(values))
        for v in values:
            if v['kind']=='check':
                checked_summary[v['proposal_sha256']]=v['result'].get('observation_window',{}).get('sensor_only_points',0)
            elif v['kind']=='diagnostic_commit':
                protocol_stats['commits']+=1;count=checked_summary.pop(v['grant']['proposal_sha256'],0)
                if count:
                    protocol_stats['sensor_commits']+=1
                    if len(protocol_stats['sensor_samples'])<32:
                        protocol_stats['sensor_samples'].append(dict(commit_ns=v['commit_ns'],sequence=v['grant']['lease_sequence'],sensor_only_points=count))
            elif v['kind']=='abort':checked_summary.pop(v.get('proposal_sha256'),None)
            elif v['kind']=='invalidate':checked_summary.clear()
            events.write(json.dumps(v)+'\n')
        events.flush()
    def start(cmd,label):
        f=(out/(label+'.log')).open('x');handles.append(f)
        p=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);owned[p.pid]=identity(p.pid);return p
    last_scan=[0.]
    def wait(fn,label,timeout=20.):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if time.monotonic()-last_scan[0]>.5:
                remember_descendants({p.pid for p in children if p.poll() is None},owned);last_scan[0]=time.monotonic()
            if errors:raise AssertionError(errors)
            if fn():return
            if any(p.poll() is not None for p in children):raise AssertionError('child exited: '+label)
            time.sleep(.002)
        raise AssertionError('timeout: '+label)
    def check(label,value):
        assert value,label
        result['checks'].append(label)
    def last():
        with state_lock:return copy.deepcopy(latest[0])
    try:
        robot=ET.parse(ROOT/'src/arm_description/urdf/Arm.urdf')
        plugin=ET.SubElement(ET.SubElement(robot.getroot(),'gazebo'),'plugin',name='care_physics_response_observer',filename=str(observer))
        ET.SubElement(plugin,'output').text=str(out/'physics_steps.jsonl')
        robot.write(out/'observed_robot.urdf',encoding='unicode')
        start(['roscore','-p',args.port],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/live_map_ledger')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy
        from care_velocity_controller.msg import CommandStatus,ExecutionLease,LeasedVelocityCommand
        from care_confidence_map.srv import QueryConfidenceSnapshot,QueryConfidenceSnapshotRequest
        from geometry_msgs.msg import Point
        from std_srvs.srv import Trigger
        from controller_manager_msgs.srv import ListControllers
        from phase_e_leased_stop_evidence import status_record,command_record
        from phase_e_stop_shadow_node import snapshot_dict,BoundedQuery,PersistentServiceTransport
        rospy.set_param('/use_sim_time',True);rospy.init_node('live_map_ledger',disable_signals=True)
        def ns():return rospy.Time.now().to_nsec()
        def rostime(value):return rospy.Time(value//10**9,value%10**9)
        def on_state(msg):
            s=status_record(msg);s['integral']=list(msg.applied_position_integral)
            with state_lock:latest[0]=s;records.append(s)
            event('state',**s)
            try:
                if s['t_ns']>=drop_ack_until[0]:ledger.observe(s)
                else:event('diagnostic_dropped_receipt',t_ns=s['t_ns'],lease_sequence=s['lease_sequence'])
            except Exception:errors.append(traceback.format_exc())
        rospy.Subscriber(BASE+'/state',CommandStatus,on_state,queue_size=1000,tcp_nodelay=True)
        authority=rospy.Publisher(BASE+'/execution_lease',ExecutionLease,queue_size=10,tcp_nodelay=True)
        direct=rospy.Publisher(BASE+'/command_leased',LeasedVelocityCommand,queue_size=10,tcp_nodelay=True)
        start(['roslaunch','arm_description','gazebo_velocity_control.launch','use_command_watchdog:=true',
            'require_execution_lease:=true','gazebo_gui:=false','use_rviz:=false',
            'apply_initial_joint_overrides:=false','robot_urdf:='+str(out/'observed_robot.urdf')],'gazebo')
        wait(lambda:last() is not None and authority.get_num_connections() and direct.get_num_connections(),'controller',90.)
        selected=next(c for c in rospy.ServiceProxy('/care_arm/controller_manager/list_controllers',ListControllers)().controller if c.name=='arm_group_velocity_controller')
        check('actual lease-required controller starts waiting',selected.state=='running' and
            selected.type=='care_velocity_controller/ExpiringJointVelocityController' and last()['mode']==0 and last()['lease_required'])
        remember_descendants({p.pid for p in children if p.poll() is None},owned)
        loaded=[]
        for pid in owned:
            try:
                p=Path('/proc')/str(pid)
                if b'gzserver' in (p/'cmdline').read_bytes():loaded += [s.split()[-1] for s in (p/'maps').read_text().splitlines() if 'libcare_velocity_controller.so' in s]
            except FileNotFoundError:pass
        result['loaded_controller_libraries']=sorted(set(loaded))
        check('isolated controller binary loaded',str(prefix/'lib/libcare_velocity_controller.so') in loaded and str(ROOT/'devel/lib/libcare_velocity_controller.so') not in loaded)
        config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml').read_text())
        selection=json.loads((ROOT/'outputs/phase_e_observation_ledger_20261020/memory_ros/selection.json').read_text())
        if args.observation_prefix_scenario:
            scenario=json.loads(args.observation_prefix_scenario.read_text())
            selection['watch_grid_indices']=sorted(set(selection['watch_grid_indices'])|set(scenario['watch_grid_indices']))
        config['confidence_map'].update(query_service='/ledger_map/query',observation_trace_file=str(out/'observations.jsonl'),
            observation_trace_watch_grid_indices=selection['watch_grid_indices'])
        config['current_body_prior'].update(geometry_backend='primitive',
            primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            refresh_service='/ledger_map/refresh',deactivate_service='/ledger_map/deactivate')
        rospy.set_param('/ledger_map',config);start([binary,'__name:=ledger_map'],'map')
        rospy.wait_for_service('/ledger_map/refresh',timeout=20.)
        refresh=rospy.ServiceProxy('/ledger_map/refresh',Trigger)()
        check('initial bootstrap frozen once',refresh.success and 'skipped_samples=0' in refresh.message)
        geometry=LocalPrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
        delayed=[False]
        def encode(p,horizon,expected):
            req=QueryConfidenceSnapshotRequest(points=[Point(*x) for x in p],horizon_s=horizon,minimum_confidence=.8)
            if expected:req.expected_instance,req.expected_epoch,req.expected_revision=expected
            return req
        def make_query(timeout):
            transport=PersistentServiceTransport(
                lambda:rospy.ServiceProxy('/ledger_map/query_snapshot',QueryConfidenceSnapshot,persistent=True),encode,snapshot_dict)
            def call(p,horizon,expected):
                if p and delayed[0]:delayed[0]=False;time.sleep(.060)
                return transport(p,horizon,expected)
            return BoundedQuery(call,timeout)
        query=make_query(.3)
        def send():
            seq=0;generation=0
            try:
                while not done.is_set():
                    now=ns()
                    with send_lock:g=copy.deepcopy(current[0]);v=list(target);pose=copy.deepcopy(pose_target[0])
                    if pose is not None:v=[2.5*(x-q) for x,q in zip(pose,last()['q'])]
                    if g and now>generation:
                        seq+=1;m=LeasedVelocityCommand();m.header.stamp=rostime(now);m.source_sequence=seq
                        m.controller_session=g['controller_session'];m.execution_stamp_ns=g['execution_stamp_ns'];m.lease_sequence=g['lease_sequence']
                        m.valid_until=rostime(g['valid_until_ns']);m.certificate_sha256=g['certificate_sha256']
                        m.target_range_enforced=True;m.target_abs_velocity=g['target_abs_velocity'];m.joint_names=NAMES
                        m.velocity=[max(-c,min(c,x)) for c,x in zip(g['target_abs_velocity'],v)]
                        direct.publish(m);event('command',**command_record(m));generation=now
                    time.sleep(.005 if args.observation_prefix_scenario else .001)
            except BaseException:errors.append(traceback.format_exc())
        sender=threading.Thread(target=send,daemon=True);sender.start()
        token=[ns()]
        def make_proof(cap):
            p=ledger.begin(now_ns=ns(),token=token[0],target_abs_velocity=cap)
            try:return p,ledger.check(p,geometry=geometry,query=query,now_ns=ns,evidence_sink=lambda r:points.write(json.dumps(r)+'\n'))
            except Exception:ledger.abort(p,'check_exception');raise
        def issue(cap,required=False):
            if required:
                initial_mode=last()['mode'];deadline=time.monotonic()+3.
                while time.monotonic()<deadline:
                    value=issue(cap)
                    if value:return value
                    if initial_mode==1 and last()['mode']!=1:
                        raise AssertionError('required renewal reached stopping mode')
                    time.sleep(.003)
                raise AssertionError('no admissible diagnostic grant before acquisition timeout')
            p=None
            try:
                p,proof=make_proof(cap)
                commit=ledger.commit(proof,query=query,now_ns=ns);g=commit['grant']
                assert not commit['runtime_admission'] and not commit['production_qualified']
                m=ExecutionLease();m.controller_session=g['controller_session'];m.execution_stamp_ns=g['execution_stamp_ns']
                m.lease_sequence=g['lease_sequence'];m.issued_at=rostime(g['issued_at_ns']);m.valid_until=rostime(g['valid_until_ns'])
                m.certificate_sha256=g['certificate_sha256'];m.target_range_enforced=True;m.target_abs_velocity=g['target_abs_velocity']
                authority.publish(m)
                with send_lock:current[0]=g
                event('checked_diagnostic_lease',**g)
                return g
            except (ValueError,RuntimeError) as exc:
                if p:ledger.abort(p,'issue_rejected')
                row=dict(t_ns=ns(),reason=str(exc),active_until_ns=last()['valid_until_ns'])
                result['rejections'].append(row);event('diagnostic_issue_rejected',**row)
                return None
            finally:drain_events();points.flush()
        if args.observation_prefix_scenario:
            planning_geometry=LocalPrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
            initial_meta=query([],0.,None)
            initial_box=planning_geometry.raster(last()['q'],[0.]*7,initial_meta,subdivisions=1)
            from phase_e_observation_prefix_gazebo import benchmark_transports
            old_query=BoundedQuery(lambda p,h,e:snapshot_dict(rospy.ServiceProxy(
                '/ledger_map/query_snapshot',QueryConfidenceSnapshot)(encode(p,h,e))),.3)
            result['rpc_pairing']=benchmark_transports(old_query,query,initial_box['points'].tolist())
            print(json.dumps(dict(stage='paired_service_transport',summary=result['rpc_pairing']['summary'])),flush=True)
        zero=issue([0.]*7,True)
        wait(lambda:last()['mode']==1 and last()['lease_sequence']==zero['lease_sequence'],'first zero applied')
        if args.observation_prefix_scenario:
            from phase_e_observation_prefix_gazebo import run_sensor_prefix_scenario
            def set_pose(pose):
                with send_lock:
                    pose_target[0]=copy.deepcopy(pose)
                    target[:]=[0.]*7
            run_sensor_prefix_scenario(plan_path=args.observation_prefix_scenario,ns=ns,last=last,issue=issue,
                current=current,ledger=ledger,result=result,check=check,event=event,set_pose=set_pose,
                query_factory=lambda:make_query(.5),
                planning_geometry=planning_geometry,
                owned=owned,remember=lambda:remember_descendants({p.pid for p in children if p.poll() is None},owned),
                protocol_stats=protocol_stats)
        else:
            cap=[.03]+[0.]*6
            with send_lock:target[:]=cap
            issue(cap,True)
            def sustain(seconds,label):
                start_ns=ns();q0=last()['q'];commits0=ledger.sequence
                while ns()<start_ns+int(seconds*1e9):
                    s=last();assert s['mode']==1,(label,s)
                    if current[0]['valid_until_ns']-ns()<=70000000:issue(cap)
                    if errors:raise AssertionError(errors)
                    time.sleep(.002)
                result['phases'].append(dict(label=label,duration_s=(ns()-start_ns)*1e-9,
                    commits=ledger.sequence-commits0,measured_joint1_progress_rad=last()['q'][0]-q0[0]))
            sustain(1.6,'continuous_real_map_progress')
            check('continuous progress with real-map checked renewals',result['phases'][-1]['measured_joint1_progress_rad']>.03 and result['phases'][-1]['commits']>10)
            # The ledger intentionally does not receive a newly applied ACK.
            issue(cap,True);drop_ack_until[0]=ns()+45000000
            wide=issue([.04]+[0.]*6,True)
            p=ledger.begin(now_ns=ns(),token=token[0],target_abs_velocity=cap)
            record=p.record();ledger.abort(p,'pending_union_inspection_only')
            check('unacknowledged wider emitted grant remains enclosed',any(g['lease_sequence']==wide['lease_sequence'] for g in record['outstanding']) and record['enclosed_target_abs_velocity'][0]==.04)
            wait(lambda:ns()>=drop_ack_until[0],'receipt delay elapsed');drop_ack_until[0]=0
            sustain(.3,'after_delayed_ack')
            issue(cap,True);old=current[0]['valid_until_ns'];p,proof=make_proof(cap)
            version=proof.record().get('map_identity')
            if proof.record().get('diagnostic_accepted'):
                wait(lambda:query([],0.,None)['revision']!=version[2],'map version advances')
                try:ledger.commit(proof,query=query,now_ns=ns)
                except (ValueError,RuntimeError) as exc:event('injected_version_rejection',reason=str(exc))
                else:raise AssertionError('changed version committed')
            else:ledger.abort(p,'already_rejected_before_version_control')
            check('changed map version cannot extend the old deadline',current[0]['valid_until_ns']==old)
            drain_events();sustain(.3,'after_map_version_change')
            def fixed_stop(label):
                fixed=current[0]['valid_until_ns'];start_ns=ns()
                wait(lambda:last()['mode']==3,label+' latch',5.)
                latched=last();wait(lambda:ns()>=latched['t_ns']+150000000 and max(map(abs,last()['measured']))<=.001,label+' physical rest',5.)
                subset=[s for s in records if fixed<=s['t_ns']<=last()['t_ns']]
                check(label+' expiry precedes latch and fresh old commands cannot revive',subset and all(s['mode'] in (2,3) for s in subset))
                result['phases'].append(dict(label=label,pause_ns=start_ns,fixed_until_ns=fixed,latched_ns=latched['t_ns'],rest_ns=last()['t_ns']))
            fixed_stop('issuer_pause')
            rearm=rospy.ServiceProxy(BASE+'/rearm',Trigger)
            def recover():
                count=last()['rearm_successes'];token[0]=ns()
                with send_lock:target[:]=[0.]*7
                issue([0.]*7,True);ready_ns=ns()+20000000
                wait(lambda:ns()>=ready_ns,'new-token zero received')
                response=rearm();assert response.success,response.message
                wait(lambda:last()['mode']==1 and last()['rearm_successes']>count,'explicit new-token rearm')
                with send_lock:target[:]=cap
                issue(cap,True)
            recover();sustain(.3,'after_explicit_rearm')
            issue(cap,True);old=current[0]['valid_until_ns'];delayed[0]=True
            rejected=issue(cap)
            check('late complete-query response cannot create a lease',rejected is None and current[0]['valid_until_ns']==old)
            fixed_stop('rpc_timeout')
            recover();sustain(.3,'after_rpc_timeout_rearm')
            issue(cap,True);old=current[0]['valid_until_ns']
            deactivated=rospy.ServiceProxy('/ledger_map/deactivate',Trigger)()
            check('real map bootstrap deactivation succeeds',deactivated.success)
            rejected=issue(cap)
            check('real unknown points reject renewal without sliding deadline',rejected is None and current[0]['valid_until_ns']==old and
                result['rejections'][-1]['reason']=='uncovered_reference_pose_margin')
            fixed_stop('real_map_missing_evidence')
        result['status']='PASS'
    except BaseException:
        result['status']='FAIL';result['error']=traceback.format_exc()
    finally:
        done.set()
        if sender:sender.join(3.)
        remember_descendants({p.pid for p in children if p.poll() is None},owned)
        result['cleanup']=reap_owned(owned)
        for p in children:
            try:p.wait(timeout=3.)
            except subprocess.TimeoutExpired:pass
        drain_events();trace.close();events.close();points.close()
        gc.callbacks.remove(gc_callback)
        result['gc_diagnostics']=gc_stats
        result['protocol_diagnostics']=protocol_stats
        for f in handles:f.close()
        result['diagnostic_commits']=ledger.sequence
        result['protected_identity_equal']=protected()==before
        result['input_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
        result['port_released']={}
        for port in [args.port,args.port+1]:
            with socket.socket() as sock:result['port_released'][str(port)]=sock.connect_ex(('127.0.0.1',port))!=0
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='PASS' and result['protected_identity_equal'] and result['input_unchanged'] and not result['cleanup']['remaining'] and all(result['port_released'].values()) else 1

if __name__=='__main__':raise SystemExit(main())
