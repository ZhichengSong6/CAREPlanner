#!/usr/bin/env python3
"""Private ROS transport check for the actuator limiter; no robot or Gazebo."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xmlrpc.client

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase_e_rate_limiter_hold_20260928'


def wait(predicate, label, timeout=8):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('timeout: '+label)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',11512))!=0, 'private port busy'
    os.environ['ROS_MASTER_URI']='http://127.0.0.1:11512'
    os.environ['ROS_LOG_DIR']=str(OUT/'ros_logs')
    children=[]; streams=[]
    def start(argv,name):
        stream=(OUT/name).open('w'); streams.append(stream)
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc)
        return proc
    try:
        start(['roscore','-p','11512'],'master.log')
        def ready():
            try:
                return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/hold_test')[0]==1
            except (OSError,xmlrpc.client.Error):
                return False
        wait(ready,'master')
        import rospy
        from std_msgs.msg import Bool,Float64MultiArray,String
        rospy.init_node('limiter_hold_test',anonymous=True,disable_signals=True)
        rospy.set_param('/limiter_under_test/joint_names',['j'+str(i) for i in range(7)])
        for key,value in dict(input_topic='/hold_test/desired',output_topic='/hold_test/output',
            summary_topic='/hold_test/summary',vbc_safety_hold_topic='/hold_test/vbc',
            external_safety_hold_topic='/hold_test/gcdf',rate=100.,input_timeout=.1).items():
            rospy.set_param('/limiter_under_test/'+key,value)
        rospy.set_param('/limiter_under_test/mpc/joint_acceleration_limits',3.)
        rospy.set_param('/limiter_under_test/mpc/joint_velocity_limits',2.)
        output=[]; summaries=[]
        rospy.Subscriber('/hold_test/output',Float64MultiArray,lambda m:output.append(list(m.data)))
        rospy.Subscriber('/hold_test/summary',String,lambda m:summaries.append(m.data))
        desired=rospy.Publisher('/hold_test/desired',Float64MultiArray,queue_size=1)
        vbc=rospy.Publisher('/hold_test/vbc',Bool,queue_size=1,latch=True)
        gcdf=rospy.Publisher('/hold_test/gcdf',Bool,queue_size=1,latch=True)
        start(['/usr/bin/python3',str(ROOT/'src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py'),
               '__name:=limiter_under_test'],'limiter.log')
        wait(lambda:desired.get_num_connections() and vbc.get_num_connections() and gcdf.get_num_connections(),'subscriptions')
        def send(value,duration):
            end=time.monotonic()+duration
            while time.monotonic()<end:
                desired.publish(Float64MultiArray(data=[value]*7)); time.sleep(.01)
        send(.6,.4)
        assert max(output[-1])>.5
        vbc.publish(Bool(data=True)); gcdf.publish(Bool(data=True))
        wait(lambda:any('vbc_hold=1 gcdf_hold=1' in s for s in summaries),'both hold callbacks')
        begin=len(output)
        send(1.5,.4)
        held=output[begin:]
        assert held and max(abs(x) for x in held[-1])<1e-9
        assert all(max(b)<=max(a)+1e-8 for a,b in zip(held,held[1:])), 'held output accelerated'
        vbc.publish(Bool(data=False)); send(1.5,.15)
        assert max(abs(x) for x in output[-1])<1e-9, 'GCDF OR gate released early'
        gcdf.publish(Bool(data=False))
        wait(lambda:any('vbc_hold=0 gcdf_hold=0' in s for s in summaries[-5:]),'both releases')
        time.sleep(.15)
        assert max(abs(x) for x in output[-1])<1e-9, 'old desired reused after release'
        send(.3,.15)
        assert max(output[-1])>.1, 'fresh desired did not resume'
        result=dict(status='PASS',held_output_samples=len(held),
                    checks=['direct VBC/GCDF input wiring','brake despite nonzero desired stream',
                            'both holds must release','release waits for new desired','fresh desired resumes'],
                    scope='Real ROS transport and production limiter, no actuator dynamics')
        (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
        rospy.signal_shutdown('complete')
    finally:
        for proc in reversed(children):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        for stream in streams: stream.close()


if __name__=='__main__':
    main()
