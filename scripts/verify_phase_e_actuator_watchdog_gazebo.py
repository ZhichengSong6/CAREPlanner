#!/usr/bin/env python3
"""Private Gazebo fault test of the real controller plugin and production limiter."""
import argparse
import copy
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
                     applied=list(msg.applied_velocity),measured=list(msg.measured_velocity))
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
        result['stage']='invalid stamped input'
        for defect in ('old','future','nan','wrong_joint_names'):
            before=last()['rejected']
            m=JointState(); m.name=NAMES; m.velocity=[0.,.3,0.,0.,0.,0.,0.]
            m.header.stamp=rospy.Time.now()
            if defect=='old': m.header.stamp-=rospy.Duration(.2)
            if defect=='future': m.header.stamp+=rospy.Duration(1.)
            if defect=='nan': m.velocity[1]=float('nan')
            if defect=='wrong_joint_names': m.name=list(reversed(NAMES))
            direct.publish(m)
            wait(lambda:last()['rejected']>before,defect+' rejected')
            assert last()['mode']==0 and max(map(abs,last()['applied']))==0.
        result['invalid_inputs_rejected']=['old','future','nan','wrong_joint_names']
        desired=rospy.Publisher('/watchdog_test/desired',Float64MultiArray,queue_size=1)
        vbc=rospy.Publisher('/watchdog_test/vbc',Bool,queue_size=1,latch=True)
        gcdf=rospy.Publisher('/watchdog_test/gcdf',Bool,queue_size=1,latch=True)
        vbc.publish(Bool(data=False)); gcdf.publish(Bool(data=False))
        params=dict(joint_names=NAMES,input_topic='/watchdog_test/desired',output_topic=BASE+'/requested_command',
                    stamped_output_topic=BASE+'/command_stamped',summary_topic='/watchdog_test/limiter_summary',
                    vbc_safety_hold_topic='/watchdog_test/vbc',external_safety_hold_topic='/watchdog_test/gcdf',
                    rate=100.,input_timeout=.1)
        for key,value in params.items(): rospy.set_param('/watchdog_limiter/'+key,value)
        rospy.set_param('/watchdog_limiter/mpc/joint_velocity_limits',[2.,2.,2.,2.,2.5,2.5,2.5])
        rospy.set_param('/watchdog_limiter/mpc/joint_acceleration_limits',[3.,3.,4.,4.,6.,6.,6.])
        limiter=start(['/usr/bin/python3',str(ROOT/'src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py'),
                       '__name:=watchdog_limiter'],'limiter')
        wait(lambda:desired.get_num_connections()>0,'limiter subscription')
        speed=[0.]
        def send():
            while not stop_sender.is_set():
                desired.publish(Float64MultiArray(data=[0.,speed[0],0.,0.,0.,0.,0.])); time.sleep(.005)
        sender=threading.Thread(target=send,daemon=True); sender.start()
        rearm=rospy.ServiceProxy(BASE+'/rearm',Trigger)
        result['trials']=[]
        for index,velocity in enumerate((.3,.15)):
            result['stage']='fault trial '+str(index+1)
            speed[0]=velocity
            wait(lambda:last()['mode']==1 and last()['applied'][1]>velocity-.002,'moving before fault')
            with lock: old=copy.deepcopy(stamped[-1])
            os.kill(limiter.pid,signal.SIGSTOP)
            wait(lambda:'\nState:\tT' in Path('/proc/{}/status'.format(limiter.pid)).read_text(),'limiter suspended')
            rejected_before=last()['rejected']
            # Replay a genuine already generated command while the producer is
            # stopped. It must not extend its original generation-time lease.
            replay_stop=threading.Event()
            def replay():
                while not replay_stop.is_set(): direct.publish(old); time.sleep(.01)
            replay_thread=threading.Thread(target=replay,daemon=True) if index else None
            if replay_thread: replay_thread.start()
            try:
                wait(lambda:last()['mode']==3,'independent latched stop')
                stopped=last()
                assert .099 <= stopped['fault']-stopped['stamp'] <= .105
                assert stopped['stopped']-stopped['fault'] <= velocity/3.+.005
                assert limiter.poll() is None and '\nState:\tT' in Path('/proc/{}/status'.format(limiter.pid)).read_text()
                def measured_stop():
                    with lock: rows=[r for r in records if r['t']>=stopped['fault']]
                    for i,r in enumerate(rows):
                        window=[v for v in rows[i:] if v['t']<=r['t']+.111]
                        if (window and window[-1]['t']-r['t']>=.1 and
                            all(max(map(abs,v['measured']))<=1e-3 for v in window) and
                            all(b['t']-a['t']<=.03 for a,b in zip(window,window[1:]))):
                            return r['t']
                    return None
                wait(lambda:measured_stop() is not None,'measured sustained stop')
                physical_stop=measured_stop()
                if index: assert last()['rejected']>rejected_before
                trial=dict(initial_joint2_command_rad_s=velocity,final_valid_generation_s=stopped['stamp'],
                           expiry_s=stopped['fault'],command_zero_s=stopped['stopped'],measured_zero_s=physical_stop,
                           generation_to_command_zero_s=stopped['stopped']-stopped['stamp'],
                           generation_to_measured_zero_s=physical_stop-stopped['stamp'],
                           measured_after_command_zero_s=physical_stop-stopped['stopped'],replay_injected=bool(index))
            finally:
                replay_stop.set()
                if replay_thread: replay_thread.join(2)
            os.kill(limiter.pid,signal.SIGCONT)
            at_resume=last()['t']
            wait(lambda:last()['t']>=at_resume+.2,'nonzero resumed producer remains locked')
            assert last()['mode']==3 and max(map(abs,last()['applied']))==0.
            before=last()['rearm_rejections']; rearm()
            wait(lambda:last()['rearm_rejections']>before,'nonzero rearm rejected')
            speed[0]=0.
            def fresh_zero():
                with lock: return stamped and max(map(abs,stamped[-1].velocity))<1e-12 and stamped[-1].header.stamp.to_sec()>stopped['stopped']
            wait(fresh_zero,'fresh post-stop zero')
            # Allow that packet to reach the control update before the service.
            tick=last()['t']; wait(lambda:last()['t']>=tick+.03,'zero receipt')
            before=last()['rearms']; rearm()
            wait(lambda:last()['rearms']>before,'explicit zero rearm')
            assert last()['mode']==1 and max(map(abs,last()['applied']))==0.
            trial['resumed_nonzero_blocked']=True; trial['explicit_zero_rearm_passed']=True
            result['trials'].append(trial)
            print(json.dumps(trial),flush=True)
        # Verify actual motion after the final rearm, then return to rest.
        speed[0]=.1; wait(lambda:last()['applied'][1]>.09,'fresh motion after rearm')
        speed[0]=0.; wait(lambda:max(map(abs,last()['applied']))<1e-9,'final commanded rest')
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
