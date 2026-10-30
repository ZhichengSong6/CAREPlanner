#!/usr/bin/env python3
"""Study-only physical contact latch; pause Gazebo on first report or sensor loss.

Wall time drives the watchdog so pausing /clock cannot freeze this process.
The owner also watches this state file and reaps only its own simulator on failure.
"""
from pathlib import Path
import json,math,os,threading,time

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    import rospy
    from gazebo_msgs.msg import ContactsState
    from gazebo_msgs.srv import GetPhysicsProperties
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from std_srvs.srv import Empty
    prefix=Path(os.environ['CARE_SCENE_GUARD_PREFIX']);prefix.parent.mkdir(parents=True,exist_ok=True)
    trace=Path(str(prefix)+'_trace.jsonl').open('x',buffering=1)
    state_path=Path(str(prefix)+'_state.json');trip_path=Path(str(prefix)+'_trip.json')
    assert not state_path.exists() and not trip_path.exists()
    needs_box=os.environ.get('CARE_STATIC_SCENE_ID')!='baseline'
    fixture=os.environ.get('CARE_GUARD_FIXTURE')=='true'
    gate={} if fixture else json.loads(Path(os.environ['CARE_BOOTSTRAP_GATE']).read_text())
    lock=threading.RLock();clock_stamp=0;clock_wall=None;contact_wall=None;joint_wall=None
    state=dict(pid=os.getpid(),ready=False,connected=False,armed=False,tripped=False,
        contact_messages=0,nonempty_messages=0,pause_requested=False,pause_ack=False,pause_confirmed=False,
        maximum_prearm_joint_deviation_rad=0.,started_monotonic_s=time.monotonic())
    def emit(event,**kw):
        row=dict(event=event,monotonic_s=time.monotonic(),wall_s=time.time());row.update(kw)
        trace.write(json.dumps(row)+'\n')
    def pause():
        try:
            rospy.wait_for_service('/gazebo/pause_physics',timeout=.8)
            rospy.ServiceProxy('/gazebo/pause_physics',Empty)()
            ack=time.monotonic()
            with lock:
                state.update(pause_ack=True,pause_ack_monotonic_s=ack,
                    report_to_pause_ack_ms=(ack-state['trip_monotonic_s'])*1000)
                emit('pause_ack',latency_ms=state['report_to_pause_ack_ms'])
            p=rospy.ServiceProxy('/gazebo/get_physics_properties',GetPhysicsProperties)()
            with lock:
                state['pause_confirmed']=bool(p.success and p.pause)
                emit('physics_pause_verified',paused=state['pause_confirmed'])
        except Exception as exc:
            with lock:state['pause_error']=repr(exc);emit('pause_failed',error=repr(exc))
    def trip(reason,**evidence):
        with lock:
            if state['tripped']:return
            now=time.monotonic();state.update(tripped=True,ready=False,trip_reason=reason,trip_monotonic_s=now,pause_requested=True)
            payload=dict(reason=reason,monotonic_s=now,wall_s=time.time(),ros_ns=clock_stamp,**evidence)
            atomic(trip_path,payload);emit('trip',**payload)
        threading.Thread(target=pause,daemon=True).start()
    def contact(msg):
        nonlocal contact_wall
        with lock:
            contact_wall=time.monotonic();state['contact_messages']+=1
            if msg.states:
                state['nonempty_messages']+=1
                trip('physical_box_contact',contact_ros_ns=msg.header.stamp.to_nsec(),contacts=[dict(collision1=x.collision1_name,collision2=x.collision2_name,depths_m=list(x.depths)) for x in msg.states])
    def on_clock(msg):
        nonlocal clock_stamp,clock_wall
        with lock:
            n=msg.clock.to_nsec()
            if n>clock_stamp:clock_stamp=n;clock_wall=time.monotonic()
    def joint(msg):
        nonlocal joint_wall
        with lock:
            values=dict(zip(msg.name,msg.position))
            if state['armed'] or fixture:return
            if any(n not in values or not math.isfinite(values[n]) for n in gate['joint_names']):
                trip('invalid_prearm_joint_sample');return
            delta=max(abs(values[n]-q) for n,q in zip(gate['joint_names'],gate['initial_q']))
            state['maximum_prearm_joint_deviation_rad']=max(delta,state['maximum_prearm_joint_deviation_rad'])
            joint_wall=time.monotonic()
            if delta>gate['maximum_prearm_joint_deviation_rad']:trip('initial_bootstrap_joint_bound_exceeded',deviation_rad=delta)
    def armed(msg):
        with lock:
            if msg.data==os.environ.get('CARE_BENCHMARK_ARM_TOKEN') and msg.data:
                state['armed']=True;emit('armed')
    emit('started')
    rospy.init_node('phase_e_static_contact_guard',disable_signals=False)
    subs=[rospy.Subscriber('/care_planner/static_box/contacts',ContactsState,contact,queue_size=100),
          rospy.Subscriber('/clock',Clock,on_clock,queue_size=10),
          rospy.Subscriber('/care_arm/joint_states',JointState,joint,queue_size=100),
          rospy.Subscriber('/care_planner/benchmark/arm',String,armed,queue_size=1)]
    try:
        rospy.wait_for_service('/gazebo/pause_physics',timeout=40)
        rospy.wait_for_service('/gazebo/get_physics_properties',timeout=5)
        deadline=time.monotonic()+5
        while needs_box and subs[0].get_num_connections()==0:
            if time.monotonic()>deadline:raise RuntimeError('No Gazebo bumper publisher connection')
            time.sleep(.05)
        with lock:state['connected']=True;emit('connected')
        while not rospy.is_shutdown():
            with lock:
                now=time.monotonic()
                fresh_clock=clock_wall is not None and now-clock_wall<2
                fresh_contact=not needs_box or (contact_wall is not None and now-contact_wall<2)
                fresh_joint=fixture or state['armed'] or (joint_wall is not None and now-joint_wall<2)
                ready=fresh_clock and fresh_contact and fresh_joint and not state['tripped']
                if state['armed'] and not state['tripped'] and not ready:
                    trip('guard_input_stale',fresh_clock=fresh_clock,fresh_contact=fresh_contact,fresh_joint=fresh_joint)
                if ready and not state['ready']:emit('ready')
                state.update(ready=ready,heartbeat_monotonic_s=now,clock_ros_ns=clock_stamp,
                    contact_age_s=now-contact_wall if contact_wall is not None else None)
                atomic(state_path,state)
            time.sleep(.05)
    finally:
        with lock:
            state.update(finished=True,ready=False,heartbeat_monotonic_s=time.monotonic())
            atomic(state_path,state);emit('finished',state=state);trace.close()

if __name__=='__main__':main()
