#!/usr/bin/env python3
"""Actual map + prefix observer + installed learned scheduler, private ROS.

The candidate is a declared triangular-velocity fixture, not an online MPC
result. Controller receipts are real plugin outputs on ideal prescribed HW;
no execution leases or velocity commands are published by this experiment.
"""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import xmlrpc.client
sys.path.append('/usr/lib/python3/dist-packages')
import yaml
from phase_e_owned_processes import identity, remember_descendants, reap_owned
from verify_phase_e_leased_response_gazebo import protected
from verify_phase_e_observation_memory_ros import RobotFrames, JOINTS

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--native-dir',type=Path,required=True)
    ap.add_argument('--visibility-prefix',type=Path,required=True)
    ap.add_argument('--map-binary',type=Path,required=True)
    ap.add_argument('--port',type=int,default=12152)
    ap.add_argument('--diagnostic-private-master',action='store_true',required=True)
    args=ap.parse_args()
    if not 12000<=args.port<=12998:ap.error('private test port required')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    native=args.native_dir.resolve();prefix=args.visibility_prefix.resolve();binary=args.map_binary.resolve()
    before=protected()
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',args.port))!=0
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml').read_text())
    model=ROOT/'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt'
    dependencies=[Path(__file__).resolve(),ROOT/'scripts/phase_e_trajectory_observer_node.py',binary,
        ROOT/'src/care_visibility_cdf/scripts/stopping_trajectory_prefix.py',
        native/'leased_controller_host',native/'libcare_velocity_controller.so',model,
        ROOT/'scripts/verify_phase_e_observation_memory_ros.py',ROOT/'scripts/phase_e_owned_processes.py']
    dependencies+=list((prefix/'share/care_visibility_cdf/scripts').glob('*.py'))
    dependencies+=list((prefix/'lib/care_visibility_cdf').glob('*.py'))
    dependencies+=[ROOT/name for name in json.loads((out.parent/'before_source_sha256.json').read_text())]
    dependencies=list(dict.fromkeys(dependencies))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    (out/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with tarfile.open(out/'input_sources.tar.gz','w:gz') as tar:
        for p in dependencies:
            data=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(ROOT)));info.size=len(data);tar.addfile(info,io.BytesIO(data))
    result=dict(status='FAIL',runtime_admission=False,production_issuer_enabled=False,new_formal_windows=0,
        scope=__doc__,checks=[],candidate_source='explicit triangular velocity fixture; no online MPC')
    processes=[];handles=[];owned={};done=threading.Event();tf_thread=None
    def start(argv,label,env=None):
        h=(out/(label+'.log')).open('x');handles.append(h)
        p=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        processes.append(p);owned[p.pid]=identity(p.pid);return p
    def wait(predicate,label,timeout=15.):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if predicate():return
            if any(p.poll() is not None for p in processes):raise AssertionError('child exited: '+label)
            time.sleep(.01)
        raise AssertionError('timeout: '+label)
    reports=[];schedules=[];states=[];candidates=[];tf_rows=[]
    try:
        start(['roscore','-p',args.port],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/prefix_probe')[0]==1
            except OSError:return False
        wait(ready,'private master')
        import rospy,tf2_ros
        from std_msgs.msg import String,Float64MultiArray
        from geometry_msgs.msg import TransformStamped
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from care_velocity_controller.msg import CommandStatus
        from care_confidence_map.srv import QueryConfidenceSnapshot
        from std_srvs.srv import Trigger
        from phase_e_leased_stop_evidence import status_record
        sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
        from stopping_trajectory_prefix import trajectory_sha256,validate_trajectory_report,trajectory_record
        rospy.set_param('/use_sim_time',True);rospy.init_node('prefix_probe',disable_signals=True)
        cc=yaml.safe_load((ROOT/'src/arm_description/config/controllers_watchdog.yaml').read_text())['care_arm']['arm_group_velocity_controller']
        cc['require_execution_lease']=True;rospy.set_param('/lease_test/controller',cc)
        rospy.Subscriber('/lease_test/controller/state',CommandStatus,lambda m:states.append(status_record(m)),queue_size=100)
        start([native/'leased_controller_host',native/'libcare_velocity_controller.so'],'controller')
        wait(lambda:states and rospy.Time.now().to_nsec()>10_000_000_000,'actual waiting receipt')
        assert states[-1]['mode']==0 and states[-1]['lease_required']
        robot=RobotFrames(config);broadcaster=tf2_ros.TransformBroadcaster()
        def tf_loop():
            poses=robot.transforms([0.]*7)
            while not done.is_set():
                stamp=rospy.Time.now();messages=[]
                for row in poses:
                    m=TransformStamped();m.header.frame_id='base_link';m.child_frame_id=row['frame'];m.header.stamp=stamp
                    m.transform.translation.x,m.transform.translation.y,m.transform.translation.z=row['translation']
                    m.transform.rotation.x,m.transform.rotation.y,m.transform.rotation.z,m.transform.rotation.w=row['quaternion_xyzw']
                    messages.append(m)
                broadcaster.sendTransform(messages);tf_rows.append(dict(stamp_ns=stamp.to_nsec(),q=[0.]*7));done.wait(.01)
        tf_thread=threading.Thread(target=tf_loop,daemon=True);tf_thread.start()
        config['confidence_map'].update(query_service='/prefix_map/query',observation_trace_file=str(out/'observations.jsonl'))
        config['current_body_prior'].update(geometry_backend='primitive',primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            refresh_service='/prefix_map/refresh',deactivate_service='/prefix_map/deactivate')
        rospy.set_param('/prefix_map',config);start([binary,'__name:=prefix_map'],'map')
        rospy.wait_for_service('/prefix_map/refresh',timeout=15.)
        assert rospy.ServiceProxy('/prefix_map/refresh',Trigger)().success
        result['checks'].append('actual_map_frozen_prior_and_real_waiting_controller_receipts')
        installed_env=dict(os.environ)
        installed_env['ROS_PACKAGE_PATH']=str(prefix/'share')+':'+str(ROOT/'src')+':/opt/ros/noetic/share'
        installed_env['PYTHONPATH']=str(prefix/'share/care_visibility_cdf/scripts')+':'+str(prefix/'lib/care_visibility_cdf')+':'+os.environ.get('PYTHONPATH','')
        params=dict(checkpoint=str(model),urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),device='cpu',enable_oracle_diagnostics=False,
            output_root=str(out/'scheduler_data'),region_schedule_mode='accumulated_multi_deadline',enable_stopping_observation_demands=True,
            predicted_trajectory_topic='/prefix/candidate',predicted_trajectory_timeout=1.,trajectory_topic='/prefix/bootstrap',
            confidence_query_service='/prefix_map/query',joint_state_topic='/lease_test/joints',candidate_vbc_summary_topic='/prefix/verdict',
            schedule_topic='/prefix/schedule',schedule_summary_topic='/prefix/schedule_summary',rate=20.,max_obligations=12)
        rospy.set_param('/prefix_scheduler',params)
        start([sys.executable,prefix/'lib/care_visibility_cdf/vbc_deadline_waypoint_online_node.py','__name:=prefix_scheduler'],'scheduler',installed_env)
        rospy.Subscriber('/prefix/schedule',Float64MultiArray,lambda m:schedules.append(dict(ns=rospy.Time.now().to_nsec(),data=list(m.data))),queue_size=100)
        wait(lambda:(out/'scheduler.log').exists() and 'ONLINE WARMUP READY' in (out/'scheduler.log').read_text(),'learned scheduler warmup',60.)
        oc=dict(trajectory_topic='/prefix/candidate',output_topic='/prefix_scheduler/stopping_observation_demands',
            state_topic='/lease_test/controller/state',lease_topic='/lease_test/controller/execution_lease',query_service='/prefix_map/query_snapshot',
            horizon_s=20.,admission_wait_s=.8,input_max_age_s=.8,wall_budget_s=.5,lease_s=.1,update_gap_s=.001,max_braking_step_s=.02,
            target_abs_velocity=[.06,.03,.15,.08,.4,.4,.4],
            observation_timing=dict(motion_s=2.,acquisition_s=.1,verification_s=.05,stop_reserve_s=.2))
        (out/'observer_config.json').write_text(json.dumps(oc,indent=2)+'\n')
        observer=start([sys.executable,ROOT/'scripts/phase_e_trajectory_observer_node.py','--diagnostic-private-master',
            '--config',out/'observer_config.json','--output-dir',out/'observer'],'observer')
        rospy.Subscriber(oc['output_topic'],String,lambda m:reports.append(json.loads(m.data)),queue_size=100)
        pub=rospy.Publisher('/prefix/candidate',JointTrajectory,queue_size=1)
        verdict=rospy.Publisher('/prefix/verdict',String,queue_size=1)
        delayed=rospy.Publisher(oc['output_topic'],String,queue_size=1)
        wait(lambda:pub.get_num_connections()>=2 and delayed.get_num_connections()>0,'observer and scheduler input connected')
        curve=JointTrajectory();curve.header.stamp=rospy.Time.now();curve.header.frame_id='base_link';curve.joint_names=JOINTS
        for t in range(0,21,2):
            q=-.0014*t*t if t<=10 else -.28+.0014*(20-t)**2
            v=-.0028*t if t<=10 else -.0028*(20-t)
            p=JointTrajectoryPoint();p.positions=[0.,q,0.,0.,0.,0.,0.];p.velocities=[0.,v,0.,0.,0.,0.,0.]
            p.time_from_start=rospy.Duration(t);curve.points.append(p)
        timer=rospy.Timer(rospy.Duration(.03),lambda _:pub.publish(curve))
        candidates.append(trajectory_record(curve))
        wait(lambda:reports,'whole-prefix report',15.)
        validate_trajectory_report(reports[-1],curve,rospy.Time.now().to_sec())
        assert len(reports[-1]['trajectory_construction']['start_window_ns'])==2
        wait(lambda:any(s['data'] for s in schedules),'learned scheduler observation obligation',5.)
        result['checks'].append('actual_map_report_reaches_installed_learned_scheduler')
        verdict.publish(String(data='trajectory_source=predicted has_violation=0'));time.sleep(.1)
        assert schedules[-1]['data']
        result['checks'].append('safe_verdict_preserves_unobserved_future_requests')
        assert identity(observer.pid)==owned[observer.pid];os.kill(observer.pid,signal.SIGSTOP)
        old=copy.deepcopy(reports[-1]);timer.shutdown()
        replacement=copy.deepcopy(curve)
        for p in replacement.points:p.positions[1]*=.5;p.velocities[1]*=.5
        assert trajectory_sha256(replacement)!=trajectory_sha256(curve)
        curve=replacement;candidates.append(trajectory_record(curve))
        timer=rospy.Timer(rospy.Duration(.03),lambda _:pub.publish(curve))
        wait(lambda:schedules and not schedules[-1]['data'],'same-stamp replacement retirement')
        delayed.publish(String(data=json.dumps(old)));time.sleep(.1)
        assert not schedules[-1]['data']
        result['checks'].append('same_stamp_replacement_and_delayed_report_cannot_transfer_requests')
        timer.shutdown();os.kill(observer.pid,signal.SIGCONT)
        # Restore the original route that contains missing cells. The shorter
        # replacement above is covered by the initial prior and cannot measure
        # useful observation scheduling under replacement pressure.
        for p in curve.points:p.positions[1]*=2.;p.velocities[1]*=2.
        # Diagnostic replacement pressure: keep strict identity rejection and
        # measure useful report throughput instead of assuming planner cadence.
        streams=[]
        for hz in [10,20]:
            begin=time.monotonic();count=0;initial=len(reports)
            while time.monotonic()-begin<2.:
                curve.header.stamp=rospy.Time.now();candidates.append(trajectory_record(curve));pub.publish(curve);count+=1
                time.sleep(1./hz)
            streams.append(dict(hz=hz,candidates=count,published_reports=len(reports)-initial,
                wall_begin_s=begin,wall_end_s=time.monotonic()))
        result.update(status='PASS',reports=len(reports),nonempty_schedules=sum(bool(s['data']) for s in schedules),
            missing_cells=max(sum(c['action']!='covered' for c in r['cells']) for r in reports),
            reference_pieces=10,replacement_streams=streams,
            controller_remained_waiting=all(s['mode']==0 and s['applied']==[0.]*7 for s in states))
        assert result['controller_remained_waiting']
    except Exception as exc:
        result.update(error=str(exc),traceback=traceback.format_exc())
    finally:
        done.set()
        if tf_thread:tf_thread.join(timeout=2.)
        remember_descendants({p.pid for p in processes if p.poll() is None},owned)
        for p in processes:
            if p.poll() is None:
                try:os.kill(p.pid,signal.SIGCONT)
                except ProcessLookupError:pass
        result['cleanup']=reap_owned(owned)
        for p in processes:
            try:p.wait(timeout=5.)
            except subprocess.TimeoutExpired:pass
        for h in handles:h.close()
        result['protected_identity_equal']=protected()==before
        with socket.socket() as sock:result['port_released']=sock.connect_ex(('127.0.0.1',args.port))!=0
        result['inputs_unchanged']=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
        for name,data in [('reports',reports),('schedules',schedules),('states',states),('candidates',candidates),('tf_trace',tf_rows)]:
            (out/(name+'.json')).write_text(json.dumps(data)+'\n')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['status']=='PASS' and not result['cleanup']['remaining'] and result['protected_identity_equal'] and result['port_released'] and result['inputs_unchanged']


if __name__=='__main__':main()
