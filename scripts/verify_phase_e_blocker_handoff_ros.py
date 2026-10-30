#!/usr/bin/env python3
"""Private TCPROS old/current handoff contrast; controlled map, no robot/arm."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from phase_e_owned_processes import identity, remember_descendants, reap_owned

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    os.environ.update(ROS_MASTER_URI=f'http://127.0.0.1:{port}', ROS_HOSTNAME='127.0.0.1',
        ROS_IP='127.0.0.1', ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    processes = []; streams = []; owned = {}
    report = dict(status='FAIL', checks=[], cases=[])
    def start(command, name):
        stream = (out/(name+'.log')).open('x'); streams.append(stream)
        p = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(p); owned[p.pid] = identity(p.pid)
        return p
    def wait(check, label, timeout=10.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            remember_descendants([p.pid for p in processes], owned)
            try:
                if check(): return
            except (OSError, xmlrpc.client.Fault): pass
            time.sleep(.025)
        raise AssertionError('timeout '+label)
    try:
        start(['roscore', '-p', str(port)], 'master')
        master = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
        wait(lambda: master.getPid('/driver')[0] == 1, 'master')
        import rospy
        from std_msgs.msg import Bool, String, Float64MultiArray, MultiArrayDimension
        rospy.init_node('blocker_handoff_driver', disable_signals=True)
        states = []; completions = []; arms = []
        subscriptions = [rospy.Subscriber('/blocker_test/state', String, lambda msg: states.append(json.loads(msg.data))),
                         rospy.Subscriber('/blocker_test/complete', Bool, lambda msg: completions.append(msg.data)),
                         rospy.Subscriber('/care_planner/benchmark/arm', String, lambda msg: arms.append(msg.data))]
        candidate = rospy.Publisher('/blocker_test/candidate', Float64MultiArray, queue_size=8)
        execution = rospy.Publisher('/blocker_test/execution', Float64MultiArray, queue_size=8)
        command = rospy.Publisher('/blocker_test/command', String, queue_size=8)
        def bundle(seq, stamp, points):
            msg = Float64MultiArray()
            msg.data = [float(seq), 0. if points else float('nan'), float(len(points))]+[v for p in points for v in p]
            msg.layout.dim = [MultiArrayDimension(label=f'care_vbc_v1_{stamp}_{seq}', size=len(msg.data), stride=len(msg.data))]
            return msg
        for version in ['before', 'current']:
            states.clear(); completions.clear()
            worker = start(['/usr/bin/python3', '-u', 'scripts/verify_phase_e_blocker_handoff.py', '--worker', version], version)
            wait(lambda: states and states[-1]['received'] == 0 and
                 min(candidate.get_num_connections(), execution.get_num_connections(), command.get_num_connections()) >= 1, 'consumer '+version)
            candidate.publish(bundle(85, 20470000000, [[0.,-.25,.85]]))
            wait(lambda: states[-1]['received'] == 1, 'candidate actual receipt')
            execution.publish(bundle(518, 20471000000, []))
            wait(lambda: states[-1]['received'] == 2, 'execution actual receipt')
            after_receipt = states[-1]
            count = len(states); command.publish(String(data='process'))
            wait(lambda: len(states) > count, 'process actual receipt')
            after_process = states[-1]
            assert len(after_process['obligations']) == (0 if version == 'before' else 1)
            count = len(states); command.publish(String(data='unseen'))
            wait(lambda: len(states) > count, 'unseen actual receipt')
            after_unseen = states[-1]
            assert after_unseen['complete'] == (version == 'before')
            if version == 'current':
                assert True not in completions
                count = len(states); command.publish(String(data='seen'))
                wait(lambda: len(states) > count, 'actual known response')
                assert states[-1]['complete'] and states[-1]['seen'] == 1
                assert after_process['events'][-1]['vbc_identity']['trajectory_stamp_ns'] == '20470000000'
            report['cases'].append(dict(version=version, after_receipt=after_receipt,
                after_process=after_process, after_unseen=after_unseen, final=states[-1], completions=list(completions)))
            worker.send_signal(signal.SIGINT); worker.wait(timeout=5.)
            wait(lambda: candidate.get_num_connections() == 0 and execution.get_num_connections() == 0, 'consumer shutdown')
        assert not arms
        report['arm_message_count'] = len(arms)
        report['checks'] = ['independent candidate/execution TCPROS receipt then processing',
            'same old messages: old loses responsibility and reports completion, current materializes exact point/audit',
            'current never publishes complete before controlled actual map evidence', 'no benchmark arm messages']
        report['status'] = 'PASS'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        remember_descendants([p.pid for p in processes], owned)
        report['cleanup'] = reap_owned(owned)
        for p in processes:
            try: p.wait(timeout=1.)
            except subprocess.TimeoutExpired: pass
        for stream in streams: stream.close()
        report['qualification'] = 'Actual production Python methods over real TCPROS with controlled generation/confidence results; no robot, trained model, or geometric safety certificate.'
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], checks=report['checks'], cleanup=report['cleanup']), indent=2))

if __name__ == '__main__': main()
