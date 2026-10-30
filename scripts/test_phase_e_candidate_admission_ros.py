#!/usr/bin/env python3
"""Private ROS transport fault injection, using real production node processes.

No Gazebo/tracker/actuator. Synthetic verdicts check transport ownership only.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--port',type=int,default=11620);args=ap.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(args.port),
        ROS_HOSTNAME='127.0.0.1',ROS_IP='127.0.0.1',ROS_LOG_DIR=str(out/'ros_logs'),ROS_HOME=str(out/'ros_home'))
    os.environ.update({k:v for k,v in env.items() if k.startswith('ROS_')})
    processes=[];logs=[];result=dict(status='FAIL',scenarios=[],qualification=
        'Private real ROS processes with CASE007 raw/q fixture and synthetic GCDF/VBC responses. No physical collision check or original TCP trigger reproduction.')
    stopped=None
    try:
        log=(out/'roscore.log').open('w');logs.append(log)
        master=subprocess.Popen(['roscore','-p',str(args.port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(master)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if master.poll() is not None:raise RuntimeError('roscore exited; see roscore.log')
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1',args.port))==0:break
            time.sleep(.05)
        else:raise RuntimeError('ROS master startup timeout')
        import rospy
        from std_msgs.msg import String
        from sensor_msgs.msg import JointState
        from egocentric_arm_planner.msg import CandidateTransfer
        from trajectory_msgs.msg import JointTrajectory
        from care_collision_cdf.msg import CollisionCDFConstraintBatch
        sys.path.insert(0,str(ROOT/'scripts'))
        from test_phase_e_candidate_admission import load_fixture
        rospy.init_node('candidate_admission_fault_test',disable_signals=True)
        data,raw=load_fixture()
        lock=threading.RLock();scenario=dict(name='startup',drops=0,duplicates=0)
        observed=dict(status=[],commits=[],queries=[],replans=[],ready=[])
        transfer_out=rospy.Publisher('/admission_test/receiver/transfer',CandidateTransfer,queue_size=20)
        ack_out=rospy.Publisher('/admission_test/gate/admission_status',String,queue_size=20)
        raw_pub=rospy.Publisher('/admission_test/raw',JointTrajectory,queue_size=1)
        joint_pub=rospy.Publisher('/admission_test/q',JointState,queue_size=1)
        gcdf_pub=rospy.Publisher('/admission_test/gcdf_result',CollisionCDFConstraintBatch,queue_size=4)
        vbc_pub=rospy.Publisher('/admission_test/vbc_result',String,queue_size=4)
        ping_pub=rospy.Publisher('/admission_test/gate/admission_ping',String,queue_size=4)
        def transfer(msg):
            with lock:
                if scenario['name']=='drop_receipt_ack' and msg.action=='deliver':
                    scenario['deliver_count']=scenario.get('deliver_count',0)+1
                    scenario['duplicate_seen']=scenario['deliver_count']>1
                if scenario['name']=='drop_first_deliver' and msg.action=='deliver' and scenario['drops']==0:
                    scenario['drops']+=1;return
                count=3 if scenario['name']=='duplicate_deliver' and msg.action=='deliver' else 1
                scenario['duplicates']+=count-1
            for _ in range(count):transfer_out.publish(copy.deepcopy(msg))
        def status(msg):
            with lock:
                observed['status'].append(msg.data)
                if scenario['name']=='drop_receipt_ack' and any('admission_stage='+s in msg.data for s in ['received','gcdf','vbc']):
                    # Suppress all initial receipts, then allow the resend reply.
                    if not scenario.get('duplicate_seen'):
                        scenario['drops']+=1;return
            ack_out.publish(msg)
        def gcdf(msg):
            with lock:
                observed['queries'].append(msg.header.stamp.to_nsec())
            verdict=CollisionCDFConstraintBatch();verdict.header.stamp=msg.header.stamp
            if scenario['name']=='drop_receipt_ack':
                threading.Timer(.32,lambda:gcdf_pub.publish(verdict)).start()
            else:
                gcdf_pub.publish(verdict)
        def vbc(msg):
            vbc_pub.publish(String(data='trajectory_source=predicted trajectory_stamp_ns={} has_violation=0'.format(msg.header.stamp.to_nsec())))
        def commit(msg):
            with lock:observed['commits'].append(msg.header.stamp.to_nsec())
        def replan(msg):
            with lock:observed['replans'].append(bool(msg.data))
        from std_msgs.msg import Bool
        rospy.Subscriber('/admission_test/gate/transfer',CandidateTransfer,transfer,queue_size=20)
        rospy.Subscriber('/admission_test/receiver/admission_status',String,status,queue_size=20)
        rospy.Subscriber('/admission_test/gcdf_query',JointTrajectory,gcdf,queue_size=4)
        rospy.Subscriber('/admission_test/vbc_query',JointTrajectory,vbc,queue_size=4)
        rospy.Subscriber('/admission_test/committed',JointTrajectory,commit,queue_size=4)
        rospy.Subscriber('/admission_test/replan',Bool,replan,queue_size=4)
        rospy.Subscriber('/admission_test/gate/admission_ping/ready',String,
            lambda msg:observed['ready'].append(msg.data),queue_size=4)
        params={
            '/admission_receiver/candidate_admission_enabled':True,
            '/admission_receiver/input_topic':'/admission_test/receiver',
            '/admission_receiver/output_topic':'/admission_test/vbc_query',
            '/admission_receiver/committed_output_topic':'/admission_test/committed',
            '/admission_receiver/joint_state_topic':'/admission_test/q',
            '/admission_receiver/global_summary_topic':'/admission_test/vbc_result',
            '/admission_receiver/final_gcdf_query_topic':'/admission_test/gcdf_query',
            '/admission_receiver/final_gcdf_batch_topic':'/admission_test/gcdf_result',
            '/admission_receiver/final_gcdf_enabled':True,
            '/admission_receiver/smooth_handoff_rebase_enabled':True,
            '/admission_receiver/summary_topic':'/admission_test/receiver_summary',
            '/admission_sender/candidate_admission_enabled':True,
            '/admission_sender/input_topic':'/admission_test/raw',
            '/admission_sender/output_topic':'/admission_test/gate',
            '/admission_sender/replan_request_topic':'/admission_test/replan',
            '/admission_sender/summary_topic':'/admission_test/gate_summary'}
        for k,v in params.items():rospy.set_param(k,v)
        for name,file in [('admission_receiver','optimized_trajectory_continuity_node.py'),('admission_sender','probe_single_flight_gate_node.py')]:
            log=(out/(name+'.log')).open('w');logs.append(log)
            p=subprocess.Popen(['/usr/bin/python3',str(ROOT/'src/egocentric_arm_planner/scripts'/file),'__name:='+name],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(p)
        receiver_proc=processes[-2]
        joint_timer=rospy.Timer(rospy.Duration(.01),lambda _:
            joint_pub.publish(JointState(name=raw.joint_names,position=data['measured_q'])))
        def wait_for(predicate,timeout=3):
            until=time.monotonic()+timeout
            while time.monotonic()<until:
                for p in processes:
                    if p.poll() is not None:raise RuntimeError('Owned node exited; see logs')
                if predicate():return
                time.sleep(.01)
            raise TimeoutError('scenario timed out: '+scenario['name'])
        wait_for(lambda:raw_pub.get_num_connections()>0 and transfer_out.get_num_connections()>0 and ack_out.get_num_connections()>0,8)
        nonce='real_roundtrip'
        until=time.monotonic()+5
        while time.monotonic()<until:
            ping_pub.publish(String(data=nonce))
            time.sleep(.05)
            if any('ready_nonce='+nonce in v for v in observed['ready']):break
        else:raise TimeoutError('admission roundtrip readiness missing')
        result['readiness_roundtrip']=True
        from phase_e_benchmark_arm import prepare_arm, ARM_TOPIC
        arm_sub=rospy.Subscriber(ARM_TOPIC,String,lambda msg:None,queue_size=1)
        arm_pub=prepare_arm([],timeout_s=5,admission_ping_topic='/admission_test/gate/admission_ping')
        result['benchmark_arm_roundtrip']=True
        for name in ['normal','drop_first_deliver','duplicate_deliver','drop_receipt_ack']:
            with lock:
                scenario.update(name=name,drops=0,duplicates=0)
                before_commits=len(observed['commits']);before_queries=len(observed['queries'])
            fresh=copy.deepcopy(raw);fresh.header.stamp=rospy.Time.now()
            raw_pub.publish(fresh)
            wait_for(lambda:len(observed['commits'])>before_commits)
            time.sleep(.30)  # expose duplicate commits/late retries
            with lock:
                row=dict(name=name,commits=len(observed['commits'])-before_commits,
                         queries=len(observed['queries'])-before_queries,drops=scenario['drops'],duplicates=scenario['duplicates'])
            assert row['commits']==row['queries']==1,row
            if name=='drop_first_deliver':assert row['drops']==1,row
            if name=='duplicate_deliver':assert row['duplicates']==2,row
            if name=='drop_receipt_ack':assert row['drops']>=2 and scenario.get('duplicate_seen'),row
            result['scenarios'].append(row)
        # Stop the actual receiver before the first offer is processed. Upon
        # resume, its stale offer cannot trigger certification: sender already
        # cancelled and therefore never sends deliver.
        scenario.update(name='receiver_stopped',drops=0,duplicates=0)
        before_commits=len(observed['commits']);before_queries=len(observed['queries'])
        os.kill(receiver_proc.pid,signal.SIGSTOP);stopped=receiver_proc
        fresh=copy.deepcopy(raw);fresh.header.stamp=rospy.Time.now();raw_pub.publish(fresh)
        time.sleep(.80)
        os.kill(receiver_proc.pid,signal.SIGCONT);stopped=None
        wait_for(lambda:any('admission_stage=cancelled' in s for s in observed['status']))
        time.sleep(.35)
        row=dict(name='receiver_stopped',commits=len(observed['commits'])-before_commits,
                 queries=len(observed['queries'])-before_queries,stopped_wall_s=.80)
        assert row['commits']==row['queries']==0,row
        result['scenarios'].append(row)
        result['status']='PASS'
        (out/'status_messages.json').write_text(json.dumps(observed,indent=2)+'\n')
        joint_timer.shutdown()
        rospy.signal_shutdown('finished')
    except BaseException as exc:
        result['error']=repr(exc)
        raise
    finally:
        if stopped is not None:os.kill(stopped.pid,signal.SIGCONT)
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
        for p in reversed(processes):
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=2)
        for log in logs:log.close()
        result['owned_remaining']=[p.pid for p in processes if p.poll() is None]
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
