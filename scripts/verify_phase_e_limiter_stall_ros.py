#!/usr/bin/env python3
"""Suspend only a private production limiter, observe ROS outputs, no robot."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import xmlrpc.client

from verify_phase_e_rate_limiter_hold_ros import wait

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase_e_stopping_bounds_20260930/ros_stall'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT/'result.json'
    # A failed rerun must never leave an old PASS as the current result.
    result_path.write_text(json.dumps(dict(status='RUNNING'))+'\n')
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1',11513)) != 0, 'private port busy'
    os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:11513'
    os.environ['ROS_LOG_DIR'] = str(OUT/'ros_logs')
    children = []; streams = []
    def start(argv,name):
        stream = (OUT/name).open('w'); streams.append(stream)
        proc = subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc)
        return proc
    try:
        start(['roscore','-p','11513'],'master.log')
        def ready():
            try:
                return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/stall_test')[0] == 1
            except (OSError,xmlrpc.client.Error):
                return False
        wait(ready,'master')
        import rospy
        from std_msgs.msg import Bool, Float64MultiArray, String
        rospy.init_node('limiter_stall_test',anonymous=True,disable_signals=True)
        params = dict(joint_names=['j'+str(i) for i in range(7)],
                      input_topic='/stall_test/desired', output_topic='/stall_test/output',
                      summary_topic='/stall_test/summary', vbc_safety_hold_topic='/stall_test/vbc',
                      external_safety_hold_topic='/stall_test/gcdf', rate=100., input_timeout=.1)
        for k,v in params.items():
            rospy.set_param('/limiter_under_test/'+k,v)
        rospy.set_param('/limiter_under_test/mpc/joint_acceleration_limits',3.)
        rospy.set_param('/limiter_under_test/mpc/joint_velocity_limits',2.)
        output = []; summaries = []; lock = threading.Lock()
        def receive(msg):
            with lock:
                output.append((time.monotonic(),list(msg.data)))
        def snapshot():
            with lock:
                return list(output)
        rospy.Subscriber('/stall_test/output',Float64MultiArray,receive,tcp_nodelay=True)
        rospy.Subscriber('/stall_test/summary',String,lambda m:summaries.append(m.data))
        desired = rospy.Publisher('/stall_test/desired',Float64MultiArray,queue_size=1)
        hold = rospy.Publisher('/stall_test/vbc',Bool,queue_size=1,latch=True)
        source = ROOT/'src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py'
        limiter = start(['/usr/bin/python3',str(source),'__name:=limiter_under_test'],'limiter.log')
        wait(lambda: desired.get_num_connections() and hold.get_num_connections(),'subscriptions')
        def send(value,duration):
            deadline = time.monotonic()+duration
            while time.monotonic() < deadline:
                desired.publish(Float64MultiArray(data=[value]*7)); time.sleep(.005)
        trials = []
        for duration in (.1,.3):
            hold.publish(Bool(data=False))
            wait(lambda: summaries and 'vbc_hold=0' in summaries[-1],'hold released')
            send(.6,.5)
            assert min(snapshot()[-1][1]) > .59
            os.kill(limiter.pid,signal.SIGSTOP)
            def stopped():
                return '\nState:\tT' in Path('/proc/{}/status'.format(limiter.pid)).read_text()
            wait(stopped,'limiter suspended')
            time.sleep(.03)  # let any already published packets drain
            before = snapshot()
            published = time.monotonic()
            hold.publish(Bool(data=True))
            time.sleep(duration)
            after = snapshot()
            resume = time.monotonic()
            assert len(after) == len(before), 'unexpected output while process suspended'
            assert min(after[-1][1]) > .59
            os.kill(limiter.pid,signal.SIGCONT)
            wait(lambda: any('vbc_hold=1' in s for s in summaries[-10:]),'hold accepted after resume')
            wait(lambda: max(abs(x) for x in snapshot()[-1][1]) < 1e-9,'command zero after resume')
            time.sleep(.12)
            observed = snapshot()
            first_zero = next(t for t,v in observed if t >= published and max(map(abs,v)) < 1e-9)
            trials.append(dict(injected_pause_after_hold_s=duration,
                               observed_no_output_after_hold_s=resume-published,
                               last_command_rad_s=after[-1][1],
                               hold_publish_to_zero_s=first_zero-published,
                               resume_to_zero_s=first_zero-resume))
        result = dict(status='PASS', conclusion='Input timeout cannot act while limiter process is suspended',
                      scope='Private ROS topics; command observations only, no actuator or Gazebo',
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), trials=trials)
        result_path.write_text(json.dumps(result,indent=2)+'\n')
        (OUT/'trace.json').write_text(json.dumps(snapshot())+'\n')
        print(json.dumps(result,indent=2))
        rospy.signal_shutdown('complete')
    except BaseException as exc:
        result_path.write_text(json.dumps(dict(status='FAIL',error=repr(exc)),indent=2)+'\n')
        raise
    finally:
        for proc in reversed(children):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGCONT)
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
