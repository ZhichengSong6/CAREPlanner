#!/usr/bin/env python3
"""Fault injection on real regime/planner callbacks; no robot or execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--port', type=int, default=11646)
    args = ap.parse_args(); out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', args.port)) != 0, 'port occupied'
    os.environ.update(ROS_MASTER_URI='http://127.0.0.1:'+str(args.port),
                      ROS_IP='127.0.0.1', ROS_HOSTNAME='127.0.0.1',
                      ROS_LOG_DIR=str(out/'ros_logs'), ROS_HOME=str(out/'ros_home'), ROOT_OUT=str(out))
    children = []; streams = []; report = dict(status='FAIL')
    def start(argv, name):
        stream = (out/(name+'.log')).open('x'); streams.append(stream)
        child = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child); return child
    def wait(check, name, timeout=8.):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            if check(): return
            time.sleep(.02)
        raise AssertionError('timeout '+name)
    try:
        start(['roscore','-p',str(args.port)], 'master')
        def master_ready():
            try: return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/retry_test')[0] == 1
            except OSError: return False
        wait(master_ready, 'master')
        import rospy
        import yaml
        from std_msgs.msg import String, Bool, Header
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from phase_e_benchmark_arm import prepare_arm, ARM_TOPIC
        rospy.init_node('qp_retry_driver', disable_signals=True)
        config = yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        lp = config['local_planner']; rospy.set_param('/retry_test', config)
        rospy.set_param('/care_planner/benchmark/qp_retry_readiness_required', True)
        base = lp['replan_request_topic']
        events = []; stalls = []; requests = []; acks = []; summaries = []; candidates = []; arm = []
        control = dict(deliver=False, drop_first_ack=True, pause_queries=False, held_queries=[], accepted=0)
        wire = rospy.Publisher('/retry_test/wire_request', String, queue_size=10)
        ack_pub = rospy.Publisher(base+'/qp_retry_ack', String, queue_size=10)
        gate = rospy.Publisher('/care_planner/execution_ready', Bool, queue_size=1, latch=True)
        batch_pub = rospy.Publisher(lp['cdf_batch_topic'], Batch, queue_size=10)
        joint = rospy.Publisher(lp['joint_states'], JointState, queue_size=1)
        reference = rospy.Publisher(lp['reference_trajectory'], JointTrajectory, queue_size=1)
        def tokens(text): return dict(w.split('=',1) for w in text.split() if '=' in w)
        def request(msg):
            requests.append(msg.data)
            if control['deliver']: wire.publish(msg)
        def ack(msg):
            acks.append(msg.data); fields = tokens(msg.data)
            if fields.get('status') == 'accepted':
                control['accepted'] += 1
                if control['drop_first_ack']:
                    control['drop_first_ack'] = False
                    control['pause_queries'] = True
                    return
            ack_pub.publish(msg)
        def reply(msg):
            # Deliberately impossible fixed linear CDF field. This tests retry
            # accounting, not the physical model or any specific case geometry.
            ks = list(range(1, len(msg.points)))
            b = Batch(header=msg.header, dof=7, num_pairs=len(ks), original_timestep=ks,
                source_type=[0]*len(ks), point_flat=[.15,.1,.45]*len(ks),
                q_linearization_flat=[v for k in ks for v in msg.points[k].positions],
                distance=[-2.]*len(ks), gradient_flat=[1.,0.,0.,0.,0.,0.,0.]*len(ks),
                approx_body_clearance_m=[.03]*len(ks))
            batch_pub.publish(b)
        def query(msg):
            if control['pause_queries']: control['held_queries'].append(msg)
            else: reply(msg)
        subs = [rospy.Subscriber(base+'/qp_retry',String,request,queue_size=20),
                rospy.Subscriber('/retry_test/wire_ack',String,ack,queue_size=20),
                rospy.Subscriber(lp['summary_topic'],String,lambda m:events.append(m.data),queue_size=100),
                rospy.Subscriber('/care_planner/local_planner/task_stall',String,lambda m:stalls.append(m.data),queue_size=20),
                rospy.Subscriber('/care_planner/c4_4/regime_summary',String,lambda m:summaries.append(m.data),queue_size=20),
                rospy.Subscriber(lp['candidate_trajectory_topic'],JointTrajectory,lambda m:candidates.append(m),queue_size=10),
                rospy.Subscriber(lp['query_trajectory_topic'],JointTrajectory,query,queue_size=20),
                rospy.Subscriber(ARM_TOPIC,String,lambda m:arm.append(m.data))]
        start([str(ROOT/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'), '__name:=retry_test',
               base+'/qp_retry:=/retry_test/wire_request', base+'/qp_retry_ack:=/retry_test/wire_ack'], 'planner')
        start(['/usr/bin/python3',str(ROOT/'src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py'),
               '_execution_ready_topic:=/care_planner/execution_ready'], 'regime')
        wait(lambda:wire.get_num_connections() and joint.get_num_connections() and reference.get_num_connections(), 'planner inputs')
        begin = time.monotonic()
        try: prepare_arm([],timeout_s=.7,admission_ping_topic='')
        except RuntimeError as e: report['disconnected_startup_rejected'] = str(e)
        else: raise AssertionError('startup passed disconnected control channel')
        assert time.monotonic()-begin < 1.2 and not arm and not candidates
        control['deliver'] = True
        pub = prepare_arm([],timeout_s=4.,admission_ping_topic='')
        report['control_readiness'] = pub.care_qp_retry_readiness
        pub.unregister(); assert not arm and not candidates
        # Gate closed at first failure; then drop requests after opening it.
        control['deliver'] = False
        gate.publish(Bool(data=False))
        q = [0.]*7; names = config['joint_names']
        def measured(_=None): joint.publish(JointState(header=Header(stamp=rospy.Time.now()),name=names,position=q,velocity=[0.]*7))
        timer = rospy.Timer(rospy.Duration(.02),measured)
        time.sleep(.1)
        ref = JointTrajectory(header=Header(stamp=rospy.Time.now()),joint_names=names,
            points=[JointTrajectoryPoint(positions=q,time_from_start=rospy.Duration(0)),
                    JointTrajectoryPoint(positions=[.3]+[0.]*6,time_from_start=rospy.Duration(2))])
        reference.publish(ref)
        wait(lambda:any('status=retry ' in s for s in stalls),'first failed QP')
        first = next(tokens(s)['token'] for s in stalls if 'status=retry ' in s)
        time.sleep(.2); assert first not in requests, 'sent through closed execution gate'
        gate.publish(Bool(data=True))
        wait(lambda:requests.count(first)>=3,'retransmissions while receiver missing')
        def starts(): return sum('event=plan_started ' in s for s in events)
        assert starts() == 1 and not candidates
        control['pause_queries'] = True  # Prevent next result before duplicate ACK test.
        control['deliver'] = True
        wait(lambda:any('token='+first+' status=duplicate' == s for s in acks),'duplicate after lost acknowledgement')
        wait(lambda:starts()==2,'exactly one new plan')
        # Release before the original 0.5 s CDF wait timeout. A timeout would
        # exercise the separate numerical-uncertified retry path.
        assert starts() == 2
        report['lost_ack_duplicate_did_not_restart_plan'] = True
        report['delayed_receiver_resumed_after_sends'] = requests.count(first)
        assert control['held_queries']
        control['pause_queries'] = False
        reply(control['held_queries'][-1]); control['held_queries'].clear()
        wait(lambda:any('status=stalled ' in s for s in stalls),'original outer QP budget',timeout=15.)
        stalled = next(tokens(s) for s in stalls if 'status=stalled ' in s)
        assert stalled['failures'] == '5' and starts() == 5 and control['accepted'] == 4
        before = starts(); wire.publish(String(data=first)); wire.publish(String(data='previous_process_123'))
        time.sleep(.4); assert starts() == before and not candidates
        report.update(original_failure_budget=5,outer_plan_starts=starts(),accepted_tickets=control['accepted'],
                      stale_and_exhausted_requests_do_not_restart=True)
        # A fresh reference supersedes an old ticket, and readiness pings never
        # cause a solve. Budget resets only through the existing reference path.
        control['pause_queries'] = True
        ref.header.stamp = rospy.Time.now(); reference.publish(ref)
        wait(lambda:starts()==6,'fresh reference plan')
        wire.publish(String(data=first)); wire.publish(String(data='ready_'+('a'*32)))
        time.sleep(.25); assert starts() == 6
        timer.shutdown()
        report.update(status='PASS',candidates=len(candidates),arm_messages=len(arm),
            qualification='Real production C++ planner and Python regime, controlled transport loss and synthetic infeasible CDF. No robot, execution, or CASE018 task-success claim.',
            binary_sha256=hashlib.sha256((ROOT/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node').read_bytes()).hexdigest())
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        for stream in streams: stream.close()
        report['owned_remaining'] = [p.pid for p in children if p.poll() is None]
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if 'events' in locals():
            (out/'events.json').write_text(json.dumps(dict(events=events,stalls=stalls,requests=requests,acks=acks,regime=summaries),indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
