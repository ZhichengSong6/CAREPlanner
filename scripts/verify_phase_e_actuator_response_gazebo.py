#!/usr/bin/env python3
"""Multi-pose actuator response measurements; observations are not certified bounds."""
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
    records=[]; joints=[]; stamped=[]; lock=threading.Lock()
    result=dict(status='RUNNING',scope='Real Gazebo/controller plugin; diagnostic, not CASE006 benchmark or physical worst-case proof')
    dependencies=[ROOT/'src/care_velocity_controller/src/expiring_velocity_controller.cpp',
                  ROOT/'src/care_velocity_controller/include/care_velocity_controller/expiring_velocity_core.hpp',
                  ROOT/'src/care_velocity_controller/msg/CommandStatus.msg',
                  ROOT/'devel/lib/libcare_velocity_controller.so',
                  ROOT/'src/arm_description/config/controllers_watchdog.yaml',
                  ROOT/'src/arm_description/urdf/Arm.urdf',
                  ROOT/'src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py',
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
        from std_msgs.msg import Bool,Float64MultiArray
        from std_srvs.srv import Trigger
        from controller_manager_msgs.srv import ListControllers
        from care_velocity_controller.msg import CommandStatus
        rospy.set_param('/use_sim_time',True)
        rospy.init_node('actuator_watchdog_test',anonymous=True,disable_signals=True)
        def state(msg):
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
        control={'kind':'velocity','value':[0.]*7}
        send_lock=threading.Lock()
        def set_control(kind,value=None):
            with send_lock: control.update(kind=kind,value=value)
        def send():
            seq=0
            while not stop_sender.is_set():
                with send_lock:
                    kind,value=control['kind'],control['value']
                    if kind!='pause':
                        if kind=='pose':
                            q=last()['q']
                            velocity=[max(-.25,min(.25,3.*(target-current))) for target,current in zip(value,q)]
                        else: velocity=value
                        m=JointState(); m.name=NAMES; m.velocity=velocity
                        m.header.stamp=rospy.Time.now(); m.header.seq=seq; seq+=1
                        direct.publish(m)
                time.sleep(.003)
        sender=threading.Thread(target=send,daemon=True); sender.start()
        rearm=rospy.ServiceProxy(BASE+'/rearm',Trigger)
        def move(target):
            set_control('pose',target)
            wait(lambda:max(abs(a-b) for a,b in zip(last()['q'],target))<.001 and
                 max(map(abs,last()['measured']))<.002,'pose reached',90.)
            set_control('velocity',[0.]*7)
            tick=last()['t']
            wait(lambda:last()['t']>=tick+.15 and max(map(abs,last()['measured']))<.001,'pose rest')
        poses=[[0.]*7,[.2,.2,-.25,.2,.2,-.2,.2],[-.2,-.2,.25,-.2,-.2,.2,-.2]]
        result['poses']=poses; result['trials']=[]
        result['sampling_note']='Positions and command integral share controller update time; status sampled at 100 Hz, not continuous physical bound.'
        for pose_index,pose in enumerate(poses):
            speed=.15 if pose_index==0 else .30
            vectors=[]
            for j in range(7):
                for sign in (-1,1):
                    v=[0.]*7; v[j]=sign*speed; vectors.append(('joint%d_%+d'%(j+1,sign),v))
            for sign in (-1,1):
                vectors.append(('coupled_%+d'%sign,[sign*speed*(1 if j%2==0 else -1) for j in range(7)]))
            for label,velocity in vectors:
                result['stage']='pose%d_%s'%(pose_index,label)
                move(pose)
                set_control('velocity',velocity)
                wait(lambda:max(abs(a-b) for a,b in zip(last()['applied'],velocity))<1e-6,'target velocity')
                tick=last()['t']; wait(lambda:last()['t']>=tick+.20,'steady velocity')
                set_control('pause')
                wait(lambda:last()['mode']==3,'latched stop')
                stopped=last()
                def rest_window():
                    with lock: rows=[r for r in records if r['t']>=stopped['stopped']]
                    for i,r in enumerate(rows):
                        window=[v for v in rows[i:] if v['t']<=r['t']+.111]
                        if (window and window[-1]['t']-r['t']>=.1 and
                            all(max(map(abs,v['measured']))<=.001 for v in window) and
                            all(b['t']-a['t']<=.03 for a,b in zip(window,window[1:]))): return window
                    return None
                wait(lambda:rest_window() is not None,'sustained measured rest')
                window=rest_window()
                with lock:
                    baseline=next(r for r in reversed(records) if r['t']<=stopped['stamp'])
                    segment=[r for r in records if baseline['t']<=r['t']<=window[-1]['t']]
                assert all(r['integral_valid'] and not r['clock_fault'] for r in segment)
                assert max(b['t']-a['t'] for a,b in zip(segment,segment[1:]))<=.03
                error=[max(abs((r['q'][j]-baseline['q'][j])-(r['integral'][j]-baseline['integral'][j])) for r in segment) for j in range(7)]
                travel=[max(abs(r['q'][j]-baseline['q'][j]) for r in segment) for j in range(7)]
                trial=dict(pose_index=pose_index,label=label,velocity=velocity,
                           baseline_s=baseline['t'],baseline_age_at_generation_s=stopped['stamp']-baseline['t'],
                           generation_s=stopped['stamp'],expiry_s=stopped['fault'],command_zero_s=stopped['stopped'],
                           measured_rest_s=window[0]['t'],confirmation_end_s=window[-1]['t'],
                           sampled_trajectory_error_rad=error,sampled_joint_travel_rad=travel,
                           measured_extra_settling_s=window[0]['t']-stopped['stopped'],
                           max_update_gap_s=max(r['max_update_gap'] for r in segment))
                assert .099<=stopped['fault']-stopped['stamp']<=.105
                result['trials'].append(trial)
                print(json.dumps(trial),flush=True)
                set_control('velocity',[0.]*7)
                tick=last()['t']; wait(lambda:last()['t']>=tick+.04,'fresh zero delivered')
                count=last()['rearms']; rearm()
                wait(lambda:last()['rearms']>count,'explicit zero rearm')
        move([0.]*7)
        result['max_sampled_trajectory_error_rad']=[max(t['sampled_trajectory_error_rad'][j] for t in result['trials']) for j in range(7)]
        result['max_observed_extra_settling_s']=max(t['measured_extra_settling_s'] for t in result['trials'])
        result['physical_bound_certified']=False
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
        (out/'states.json').write_text(json.dumps(records)+'\n')
        (out/'joint_states.json').write_text(json.dumps(joints)+'\n')
        result['input_hashes_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==result['sha256'][str(p.relative_to(ROOT))] for p in dependencies)
        if result['cleanup']['remaining'] or not result['input_hashes_unchanged']: result['status']='FAIL'
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    assert result['status']=='PASS'


if __name__=='__main__':
    main()
