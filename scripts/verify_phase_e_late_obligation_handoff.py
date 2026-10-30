#!/usr/bin/env python3
"""Actual ROS manager: recorded SAFE-before-waypoint ordering, no actuator."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import xmlrpc.client


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--manager',type=Path,required=True)
    ap.add_argument('--expect-repair',action='store_true')
    ap.add_argument('--port',type=int,default=11960)
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',args.port))!=0,'port in use'
    uri=f'http://127.0.0.1:{args.port}'
    os.environ.update(ROS_MASTER_URI=uri,ROS_IP='127.0.0.1',ROS_HOSTNAME='127.0.0.1',
        ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    procs=[];streams=[];report=dict(status='FAIL',expected_repair=args.expect_repair)
    def start(argv,name):
        f=(out/name).open('x');streams.append(f)
        p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p)
    def wait(fn,label,seconds=10):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if fn():return
            time.sleep(.02)
        raise AssertionError('timeout: '+label)
    timer=None
    try:
        start(['roscore','-p',str(args.port)],'master.log')
        def master_ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/late_obligation_test')[0]==1
            except OSError:return False
        wait(master_ready,'master')
        import rospy
        from std_msgs.msg import Bool,String
        rospy.init_node('late_obligation_test',disable_signals=True)
        ready=rospy.Publisher('/care_planner/execution/ready',Bool,queue_size=1,latch=True)
        active=rospy.Publisher('/care_planner/active_sensing/visibility_waypoint_active',Bool,queue_size=1,latch=True)
        stall=rospy.Publisher('/care_planner/local_planner/task_stall',String,queue_size=10)
        verdict=rospy.Publisher('/care_planner/candidate_vbc/summary',String,queue_size=10)
        summaries=[];triggers=[]
        sub=rospy.Subscriber('/care_planner/c4_4/regime_summary',String,
            lambda m:summaries.append(dict(re.findall(r'(\w+)=([^\s]+)',m.data))),queue_size=100)
        trigger_sub=rospy.Subscriber('/care_planner/execution/predicted_vbc_recovery_triggered',Bool,
            lambda m:triggers.append(bool(m.data)),queue_size=100)
        start(['/usr/bin/python3',str(args.manager.resolve()),'__name:=late_obligation_manager'],'manager.log')
        wait(lambda:all(p.get_num_connections()>0 for p in [ready,active,stall,verdict]),'subscribers')
        active.publish(Bool(data=False));ready.publish(Bool(data=True))
        timer=rospy.Timer(rospy.Duration(.05),lambda _:ready.publish(Bool(data=True)))
        wait(lambda:summaries and summaries[-1].get('execution_ready')=='1','ready')
        stamp=25611000000
        stall.publish(String(data=f'status=stalled query_stamp_ns={stamp} failures=5'))
        wait(lambda:summaries[-1].get('blocker_rediscovery_pending')=='1','rediscovery')
        verdict.publish(String(data=f'trajectory_source=scp_rediscovery trajectory_stamp_ns={stamp-1} success=1 has_violation=0'))
        time.sleep(.15)
        assert summaries[-1]['blocker_rediscovery_pending']=='1'
        assert not any(triggers)
        verdict.publish(String(data=f'trajectory_source=scp_rediscovery trajectory_stamp_ns={stamp} success=1 has_violation=0'))
        wait(lambda:summaries[-1].get('task_stall_outcome')=='QP_NO_PROGRESS_NO_VBC_BLOCKER','same query SAFE')
        assert summaries[-1]['state']=='NORMAL' and not any(triggers)
        time.sleep(1.)
        active.publish(Bool(data=True))
        if args.expect_repair:
            wait(lambda:summaries[-1]['state']=='REPAIR' and any(triggers),'late target enters REPAIR')
            assert summaries[-1]['reason']=='qp_stall_late_visibility_obligation'
            entries=summaries[-1]['repair_entry_count']
            for _ in range(3):active.publish(Bool(data=True));time.sleep(.06)
            assert summaries[-1]['repair_entry_count']==entries=='1'
        else:
            time.sleep(.3)
            assert summaries[-1]['visibility_waypoint_active']=='1'
            assert summaries[-1]['state']=='NORMAL' and not any(triggers)
        assert not any('arm_group_velocity_controller/command' in name for name,_ in rospy.get_published_topics())
        report.update(status='PASS',final_summary=summaries[-1],summaries=summaries,
            checks=['wrong query ignored','SAFE without target remains NORMAL',
                'late actual target handled according to expected version','no actuator topics'],
            qualification='Actual production manager and ROS transport; synthetic replay of recorded event ordering, not execution safety certification.')
    finally:
        if timer:timer.shutdown()
        for p in reversed(procs):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for f in streams:f.close()
        report['remaining_owned_pids']=[p.pid for p in procs if p.poll() is None]
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(status=report['status'],remaining_owned_pids=report['remaining_owned_pids'])))


if __name__=='__main__':main()
