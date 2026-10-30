#!/usr/bin/env python3
"""Real Gazebo + confidence map + shadow observer; deterministic transport fault injection."""
import argparse
import copy
import math
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import yaml
import xmlrpc.client

from phase_e_owned_processes import identity, remember_descendants, reap_owned

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
BASE = '/care_arm/arm_group_velocity_controller'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    for port in (11514,11515):
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1',port)) != 0, 'private port busy'
    os.environ.update(ROS_MASTER_URI='http://127.0.0.1:11514', ROS_HOSTNAME='localhost',
                      GAZEBO_MASTER_URI='http://127.0.0.1:11515', GAZEBO_MODEL_DATABASE_URI='',
                      ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[]; streams=[]; owned={}; stop_sender=threading.Event(); sender=None
    records=[]; joints=[]; stamped=[]; reports=[]; lock=threading.Lock()
    relay={'state':True,'command':True}; fault={'mode':'normal'}
    result=dict(status='RUNNING',scope='Real Gazebo/controller plugin; diagnostic, not CASE006 benchmark or physical worst-case proof')
    dependencies=[ROOT/'src/care_velocity_controller/src/expiring_velocity_controller.cpp',
                  ROOT/'src/care_velocity_controller/include/care_velocity_controller/expiring_velocity_core.hpp',
                  ROOT/'src/care_velocity_controller/msg/CommandStatus.msg',
                  ROOT/'devel/lib/libcare_velocity_controller.so',
                  ROOT/'src/arm_description/config/controllers_watchdog.yaml',
                  ROOT/'src/arm_description/urdf/Arm.urdf',
                  ROOT/'src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py',
                  ROOT/'scripts/phase_e_stop_shadow.py',ROOT/'scripts/phase_e_stop_shadow_node.py',
                  ROOT/'scripts/phase_e_stop_envelope.py',ROOT/'scripts/audit_case001_motion_prior.py',
                  ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
                  ROOT/'src/care_confidence_map/srv/QueryConfidenceSnapshot.srv',
                  ROOT/'devel/lib/care_confidence_map/confidence_map_node',
                  ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml',
                  Path(__file__).resolve()]
    result['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    def start(argv,label):
        stream=(out/(label+'.log')).open('w'); streams.append(stream)
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc); owned[proc.pid]=identity(proc.pid)
        return proc
    last_scan=0.
    def wait(predicate,label,timeout=45.):
        nonlocal last_scan
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if time.monotonic()-last_scan>.5:
                remember_descendants({p.pid for p in processes if p.poll() is None},owned)
                last_scan=time.monotonic()
            if predicate(): return
            if any(p.poll() is not None for p in processes):
                raise AssertionError('child exited during '+label)
            time.sleep(.005)
        raise AssertionError('timeout: '+label)
    def last():
        with lock:
            return records[-1] if records else None
    try:
        result['stage']='master'
        start(['roscore','-p','11514'],'master')
        def ready():
            try: return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/watchdog_test')[0]==1
            except (OSError,xmlrpc.client.Error): return False
        wait(ready,'master',15.)
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool,Float64MultiArray,String
        from care_confidence_map.srv import QueryConfidenceSnapshot
        from std_srvs.srv import Trigger
        from controller_manager_msgs.srv import ListControllers
        from care_velocity_controller.msg import CommandStatus
        rospy.set_param('/use_sim_time',True)
        rospy.init_node('actuator_watchdog_test',anonymous=True,disable_signals=True)
        state_relay=rospy.Publisher('/shadow_test/state',CommandStatus,queue_size=100)
        command_relay=rospy.Publisher('/shadow_test/command',JointState,queue_size=100)
        def relay_command(msg):
            # rospy serializes Header.seq from its own publisher counter. A
            # diagnostic relay must preserve the controller's original identity.
            command_relay.impl.acquire()
            try:
                original_sequence=msg.header.seq
                command_relay.impl.seq=original_sequence-1  # rospy increments before serializing
                command_relay.publish(msg)
                assert msg.header.seq==original_sequence
            finally: command_relay.impl.release()
        def state(msg):
            if relay['state']: state_relay.publish(msg)
            row=dict(t=msg.header.stamp.to_sec(),mode=msg.mode,clock_fault=msg.clock_fault,
                     stamp=msg.command_stamp.to_sec(),sequence=msg.command_sequence,
                     fault=msg.fault_stamp.to_sec(),stopped=msg.stopped_stamp.to_sec(),
                     rejected=msg.rejected_commands,rearms=msg.rearm_successes,rearm_rejections=msg.rearm_rejections,
                     applied=list(msg.applied_velocity),measured=list(msg.measured_velocity),
                     q=list(msg.measured_position),integral=list(msg.applied_position_integral),
                     integral_valid=msg.integral_valid,max_update_gap=msg.max_update_gap)
            with lock: records.append(row)
        def measured(msg):
            with lock: joints.append(dict(t=msg.header.stamp.to_sec(),name=list(msg.name),q=list(msg.position),v=list(msg.velocity)))
        def request(msg):
            if relay['command']: relay_command(msg)
            with lock: stamped.append(copy.deepcopy(msg))
        rospy.Subscriber(BASE+'/state',CommandStatus,state,queue_size=200,tcp_nodelay=True)
        rospy.Subscriber('/care_arm/joint_states',JointState,measured,queue_size=200,tcp_nodelay=True)
        rospy.Subscriber(BASE+'/command_stamped',JointState,request,queue_size=10,tcp_nodelay=True)
        direct=rospy.Publisher(BASE+'/command_stamped',JointState,queue_size=1)
        result['stage']='Gazebo plugin startup'
        start(['roslaunch','arm_description','gazebo_velocity_control.launch','use_command_watchdog:=true',
               'gazebo_gui:=false','use_rviz:=false','apply_initial_joint_overrides:=false'],'gazebo')
        # Our capture subscriber is one connection; the controller must be the
        # second before sending a non-latched fault packet.
        wait(lambda:last() is not None and direct.get_num_connections()>=2 and rospy.Time.now().to_sec()>.2,'controller state',60.)
        controllers=rospy.ServiceProxy('/care_arm/controller_manager/list_controllers',ListControllers)().controller
        selected=next(c for c in controllers if c.name=='arm_group_velocity_controller')
        assert selected.state=='running' and selected.type=='care_velocity_controller/ExpiringJointVelocityController'
        result['controller_type']=selected.type
        assert last()['mode']==0
        result['stage']='map and observer startup'
        config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml').read_text())
        # Deterministic no-observation diagnostic: frozen prior + UNKNOWN, 30 Hz.
        config['confidence_map'].update(observation_mode='tof_ray',query_service='/shadow_map/query')
        config['current_body_prior'].update(geometry_backend='primitive',
            primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            refresh_service='/shadow_map/refresh',deactivate_service='/shadow_map/deactivate')
        rospy.set_param('/shadow_map',config)
        start([str(ROOT/'devel/lib/care_confidence_map/confidence_map_node'),'__name:=shadow_map'],'map')
        rospy.wait_for_service('/shadow_map/query_snapshot',timeout=15)
        backend=rospy.ServiceProxy('/shadow_map/query_snapshot',QueryConfidenceSnapshot)
        deactivate=rospy.ServiceProxy('/shadow_map/deactivate',Trigger)
        def proxy(req):
            if fault['mode']=='timeout':
                fault['mode']='normal'; time.sleep(.7)
            response=backend(req)
            if req.points and response.valid:
                if fault['mode']=='mutate_after_read':
                    fault['mode']='normal'; deactivate()
                elif fault['mode']=='short_array':
                    fault['mode']='normal'; response.confidence_now=list(response.confidence_now)[:-1]
            return response
        service=rospy.Service('/shadow_test/snapshot',QueryConfidenceSnapshot,proxy)
        def report(m):
            with lock: reports.append(json.loads(m.data))
        rospy.Subscriber('/care_planner/stop_shadow/report',String,report,queue_size=100)
        observer=start(['/home/zhicheng/miniconda3/envs/viscdf/bin/python',str(ROOT/'scripts/phase_e_stop_shadow_node.py'),
            '--output-dir',str(out/'shadow'),'--state-topic','/shadow_test/state','--command-topic','/shadow_test/command',
            '--snapshot-service','/shadow_test/snapshot','--rate','10','--query-timeout','.15'],'shadow')
        wait(lambda:state_relay.get_num_connections()>0 and command_relay.get_num_connections()>0,'observer subscriptions')
        serial=[0]
        def publish(velocity):
            m=JointState(); m.name=NAMES; m.velocity=velocity; serial[0]+=1
            m.header.stamp=rospy.Time.now(); m.header.seq=serial[0]; direct.publish(m); return m
        def since(index):
            with lock: return list(reports[index:])
        def expect(reason,index):
            wait(lambda:any(reason in r['reasons'] for r in since(index)),reason,20.)
        result['checks']=[]
        def check(label): result['checks'].append(label); print(label,flush=True)
        publish([0.]*7)
        wait(lambda:last()['mode']==3,'initial zero lease expires')
        expect('unknown_occupied_or_decayed_points',0); check('real UNKNOWN map rejected')
        refresh=rospy.ServiceProxy('/shadow_map/refresh',Trigger)()
        assert refresh.success and 'skipped_samples=0' in refresh.message,refresh.message
        begin=len(reports)
        wait(lambda:any(r.get('evidence_counts') and r['evidence_counts']['bootstrap']>0 for r in since(begin)),
             'bootstrap provenance in real body enclosure')
        check('frozen primitive prior queried with full enclosure')
        # A fresh stamped zero and explicit rearm enable the controlled motion.
        publish([0.]*7); tick=last()['t']; wait(lambda:last()['t']>=tick+.03,'fresh zero')
        count=last()['rearms']; rospy.ServiceProxy(BASE+'/rearm',Trigger)()
        wait(lambda:last()['rearms']>count,'rearm')
        motion_start=rospy.Time.now().to_sec()
        def stream_commands():
            while not stop_sender.is_set() and rospy.Time.now().to_sec()-motion_start<.8: publish([0.,.15,0.,0.,0.,0.,0.]); time.sleep(.01)
        begin=len(reports); sender=threading.Thread(target=stream_commands,daemon=True); sender.start()
        wait(lambda:last()['applied'][1]>.14,'motion')
        expect('controller_command_changed_during_check',begin); check('accepted command changes invalidate in-flight audit')
        expect('newer_request_observed',begin); check('new requests during audit are recorded')
        stop_sender.set(); sender.join(2)
        wait(lambda:last()['mode']==3 and max(map(abs,last()['measured']))<.001,'producer stopped, independent braking complete')
        check('observer did not prevent independent expiry and stop')
        begin=len(reports); relay['state']=False
        expect('stale_or_future_measured_state',begin); check('dropped state stream rejected as stale')
        relay['state']=True
        begin=len(reports); relay['command']=False
        publish([0.]*7); tick=last()['t']; wait(lambda:last()['t']>=tick+.03,'hidden fresh zero')
        count=last()['rearms']; rospy.ServiceProxy(BASE+'/rearm',Trigger)()
        wait(lambda:last()['rearms']>count,'hidden rearm command')
        expect('missing_exact_command_identity',begin); check('missing command packet cannot be guessed from applied output')
        relay['command']=True
        # Refresh a zero after the hidden lease has expired, then rearm so the
        # controller identity again has a packet observed by the shadow.
        wait(lambda:last()['mode']==3,'hidden zero expires')
        zero=publish([0.]*7); tick=last()['t']; wait(lambda:last()['t']>=tick+.03,'visible fresh zero')
        count=last()['rearms']; rospy.ServiceProxy(BASE+'/rearm',Trigger)()
        wait(lambda:last()['rearms']>count,'visible rearm')
        wait(lambda:last()['mode']==3,'visible zero expires')
        begin=len(reports); fault['mode']='mutate_after_read'
        expect('map_version_changed_during_check',begin); check('map mutation after full read invalidates recheck')
        begin=len(reports); fault['mode']='short_array'
        expect('incomplete_snapshot_arrays',begin); check('truncated snapshot rejected')
        begin=len(reports); fault['mode']='timeout'
        expect('snapshot_service_timeout',begin)
        expect('snapshot_service_busy_after_timeout',begin); check('RPC timeout bounded, no unbounded request queue')
        tick=time.monotonic(); wait(lambda:time.monotonic()-tick>.8,'late RPC drain')
        begin=len(reports)
        conflict=copy.deepcopy(zero); conflict.velocity=[.1]*7; relay_command(conflict)
        expect('ambiguous_command_identity',begin); check('same stamp and sequence with different payload rejected')
        assert reports and all(not r['runtime_admission'] for r in reports)
        check('all reports deny runtime admission')
        pubs=xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getSystemState('/shadow_test')[2][0]
        topics=[topic for topic,nodes in pubs if '/phase_e_stop_shadow' in nodes]
        assert sorted(topics)==['/care_planner/stop_shadow/report','/rosout'],topics
        result['observer_published_topics']=topics; check('observer has no control publishers')
        result['status']='PASS'; result['stage']='complete'
    except BaseException:
        result['status']='FAIL'; result['error']=traceback.format_exc()
        raise
    finally:
        stop_sender.set()
        if sender: sender.join(2)
        remember_descendants({p.pid for p in processes if p.poll() is None},owned)
        for p in processes:
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGCONT)
                except ProcessLookupError: pass
        result['cleanup']=reap_owned(owned)
        for p in processes:
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
        for stream in streams: stream.close()
        (out/'reports.json').write_text(json.dumps(reports)+'\n')
        (out/'states.json').write_text(json.dumps(records)+'\n')
        (out/'joint_states.json').write_text(json.dumps(joints)+'\n')
        result['input_hashes_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==result['sha256'][str(p.relative_to(ROOT))] for p in dependencies)
        if result['cleanup']['remaining'] or not result['input_hashes_unchanged']: result['status']='FAIL'
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    assert result['status']=='PASS'


if __name__=='__main__':
    main()
