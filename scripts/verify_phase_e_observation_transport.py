#!/usr/bin/env python3
"""Private ROS fault injection; actual planner/gate/regime, no robot or goals."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = r'''
import os,sys,time
sys.path.insert(0, os.path.join(os.environ['CARE_TEST_REPO'], 'src/care_visibility_cdf/scripts'))
import rospy
from std_msgs.msg import Bool,String,Float64MultiArray
mode=sys.argv[1]
if mode=='fixed':
    from bounded_tcpros import install_observation_tcp_admission
    install_observation_tcp_admission()
rospy.init_node('vbc_deadline_waypoint_rolling' if mode=='fixed' else 'old_observation_source', disable_signals=True)
if mode=='old':
    pub=rospy.Publisher('/old_observation/data',String,queue_size=1,latch=True)
    while not rospy.is_shutdown():
        pub.publish(String(data='alive'));time.sleep(.025)
else:
    from std_srvs.srv import SetBool,SetBoolResponse
    service=rospy.Service('/observation_transport/echo',SetBool,
                          lambda req:SetBoolResponse(success=req.data,message='echo'))
    prefix='/care_planner/active_sensing/'
    values={
      'visibility_waypoint_active':Bool(data=False),
      'visibility_waypoint_q':Float64MultiArray(),
      'current_visibility_obligation_points':Float64MultiArray(data=[1.,-1.,0.]),
      'visibility_frontier_target':Float64MultiArray(data=[0.,0.,1.]+[0.]*7+[1.]*7),
      'visibility_waypoint_summary':String(data='active=0 ready=0 reason=startup'),
      'visibility_acquisition_complete':Bool(data=False),
      'visibility_acquisition_summary':String(data='active_id=-1 seen_count=0 remaining_count=0'),
      'blocker_stack_summary':String(data='stack=none'),
      'visibility_frontier_summary':String(data='active=0')}
    pubs={key:rospy.Publisher(prefix+key,type(value),queue_size=1,latch=True) for key,value in values.items()}
    rospy.set_param('/fixture_ready', True)
    q_sent=False
    while not rospy.is_shutdown():
        for key,value in values.items():
            if key=='visibility_waypoint_q':
                if q_sent or not rospy.get_param('/fixture_publish_q',False):continue
                q_sent=True
            pubs[key].publish(value)
        time.sleep(.025)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True, type=Path)
    args = ap.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    os.environ.update(ROS_MASTER_URI='http://127.0.0.1:'+str(port), ROS_HOSTNAME='127.0.0.1',
                      ROS_IP='127.0.0.1', ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'),
                      CARE_TEST_REPO=str(ROOT))
    children, streams, sockets = [], [], []
    report = dict(status='FAIL', checks=[])

    def start(argv, name):
        stream = (out/(name+'.log')).open('x'); streams.append(stream)
        p = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p); return p

    def wait(check, label, timeout=8.):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            try:
                if check(): return
            except (OSError, xmlrpc.client.Fault): pass
            time.sleep(.02)
        raise AssertionError('timeout '+label)

    def passed(label): report['checks'].append(label)

    def raw_peer(uri, topic, payload):
        result = xmlrpc.client.ServerProxy(uri).requestTopic('/observation_transport_test', topic, [['TCPROS']])
        assert result[0] == 1, result
        sock = socket.create_connection((result[2][1], result[2][2]), timeout=1.)
        sock.sendall(payload); sockets.append(sock)
        return sock

    try:
        start(['roscore', '-p', str(port)], 'master')
        master = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
        wait(lambda: master.getPid('/driver')[0] == 1, 'master')
        import rospy
        import yaml
        from std_msgs.msg import String, Bool
        from trajectory_msgs.msg import JointTrajectory
        from phase_e_benchmark_arm import prepare_arm, ARM_TOPIC
        import phase_e_observation_readiness as obs
        original_check = obs.ObservationReadiness.check
        def diagnostic_check(self, *args):
            result = original_check(self, *args)
            with (out/'readiness_checks.jsonl').open('a') as log:
                log.write(json.dumps(self.last)+'\n')
            return result
        obs.ObservationReadiness.check = diagnostic_check
        rospy.init_node('observation_transport_test', disable_signals=True)
        (out/'fixture.py').write_text(FIXTURE)
        start(['/usr/bin/python3', '-u', str(out/'fixture.py'), 'old'], 'old_source')
        wait(lambda: master.lookupNode('/driver', '/old_observation_source')[0] == 1, 'old source')
        old_uri = master.lookupNode('/driver', '/old_observation_source')[2]
        wait(lambda: xmlrpc.client.ServerProxy(old_uri).getPublications('/driver')[2], 'old publisher')
        bad = raw_peer(old_uri, '/old_observation/data', struct.pack('<I', 100)+b'abcdef')
        old_msgs = []
        time.sleep(.08)  # Accepted partial header remains deliberately open.
        old_sub = rospy.Subscriber('/old_observation/data', String, lambda m: old_msgs.append(m.data))
        time.sleep(.7)
        assert not old_msgs, 'unmodified ROS should block later subscriber admission'
        bad.close()
        wait(lambda: old_msgs, 'old source resumes after EOF')
        old_sub.unregister()
        passed('old_partial_handshake_blocks_valid_subscriber_until_eof')

        start(['/usr/bin/python3', '-u', str(out/'fixture.py'), 'fixed'], 'fixed_source')
        wait(lambda: rospy.get_param('/fixture_ready', False), 'fixed publishers')
        uri = master.lookupNode('/driver', obs.GENERATOR)[2]
        bad = raw_peer(uri, obs.PREFIX+'visibility_waypoint_active', struct.pack('<I', 100)+b'abcdef')
        seen = []
        sub = rospy.Subscriber(obs.PREFIX+'visibility_waypoint_active', Bool, lambda m: seen.append(m.data))
        wait(lambda: seen, 'valid subscriber despite incomplete peer', timeout=.45)
        assert seen == [False]*len(seen)
        transport_log = out/'ros_logs/vbc_deadline_waypoint_rolling.log'
        wait(lambda: 'received_bytes=10' in transport_log.read_text(), 'bounded handshake rejection', timeout=1.)
        passed('fixed_valid_subscriber_served_while_incomplete_peer_open')
        passed('fixed_deadline_logs_peer_and_received_bytes')
        # Size abuse, invalid fields and EOF are rejected without killing the loop.
        for payload in [struct.pack('<I', 65537), struct.pack('<I', 4)+b'bad!', b'\x01']:
            s = raw_peer(uri, obs.PREFIX+'visibility_waypoint_active', payload)
            s.shutdown(socket.SHUT_WR)
            time.sleep(.02)
        before = len(seen)
        wait(lambda: len(seen) > before+2, 'stream survives malformed peers')
        passed('malformed_oversized_and_eof_peers_do_not_interrupt_established_stream')

        # Deliberate saturation cannot create an unbounded worker population.
        held = [raw_peer(uri, obs.PREFIX+'visibility_waypoint_active', struct.pack('<I', 100)+b'abcdef')
                for _ in range(12)]
        wait(lambda: 'reason=admission_capacity_or_shutdown' in transport_log.read_text(), 'capacity rejection')
        time.sleep(.6)
        for s in held: s.close()
        passed('bounded_worker_capacity_and_timeout_recovery')
        from std_srvs.srv import SetBool
        echo = rospy.ServiceProxy('/observation_transport/echo', SetBool, persistent=True)
        assert echo(True).success is True and echo(False).success is False
        echo.close()
        passed('normal_and_persistent_ros_service_requests_preserve_payload')

        config = yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        rospy.set_param(obs.PLANNER, config)
        rospy.set_param('/c4_4_verified_regime_manager/repair_completion_gate_enabled', True)
        start([str(ROOT/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node')], 'planner')
        for filename, name in [('vbc_execution_reference_gate_node.py', 'gate'),
                               ('c4_4_verified_regime_manager_node.py', 'regime')]:
            start(['/usr/bin/python3', str(ROOT/'src/egocentric_arm_planner/scripts'/filename)], name)
        # Auditor transport is represented by a named subscriber; no geometric
        # auditor, map or robot is needed for this receipt-only check.
        auditor = out/'auditor.py'
        auditor.write_text("import rospy\nfrom std_msgs.msg import Bool\nrospy.init_node('predicted_vbc_auditor_node')\nsub=rospy.Subscriber('/care_planner/active_sensing/visibility_waypoint_active',Bool,lambda m:None)\nrospy.spin()\n")
        start(['/usr/bin/python3', str(auditor)], 'auditor')
        recorder_names = []
        for i, topic in enumerate(obs.RECORDED):
            name = '/transport_recorder_'+str(i); recorder_names.append(name)
            start(['rostopic', 'echo', '-p', obs.PREFIX+topic, '__name:='+name], 'recorder_'+str(i))
        arm_msgs, candidates = [], []
        arm_sub = rospy.Subscriber(ARM_TOPIC, String, lambda m: arm_msgs.append(m.data))
        candidate_sub = rospy.Subscriber(config['local_planner']['candidate_trajectory_topic'], JointTrajectory,
                                         lambda m: candidates.append(m))
        rospy.set_param('/care_planner/benchmark/observation_readiness_required', True)
        expected = {obs.PLANNER, '/vbc_execution_reference_gate', '/c4_4_verified_regime_manager',
                    '/predicted_vbc_auditor_node', *recorder_names}
        def registered():
            graph = master.getSystemState('/driver')[2]
            return expected <= {node for _, nodes in graph[1] for node in nodes}
        wait(registered, 'all actual observation consumers registered', timeout=15.)
        started = time.monotonic()
        try: prepare_arm(recorder_names, timeout_s=.8, admission_ping_topic='')
        except RuntimeError as exc:
            assert 'visibility_waypoint_q' in str(exc), str(exc)
            report['missing_q_error'] = str(exc)
        else: raise AssertionError('registered q connection without a received sample passed startup')
        assert time.monotonic()-started < 1.2 and not arm_msgs and not candidates
        passed('startup_rejects_registered_but_silent_q_with_original_deadline')
        rospy.set_param('/fixture_publish_q', True)
        pub = prepare_arm(recorder_names, timeout_s=5., admission_ping_topic='')
        report['readiness'] = pub.care_observation_readiness
        assert report['readiness']['ready'] and len(report['readiness']['receiver_counts']) == 15
        assert not arm_msgs and not candidates
        pub.unregister()
        passed('actual_cpp_planner_python_gate_regime_and_six_recorders_receive')
        passed('empty_q_probe_does_not_arm_or_publish_candidate')
        report.update(status='PASS', arm_messages=len(arm_msgs), candidates=len(candidates),
            qualification='Real TCPROS partial headers and bounded admission; actual built C++ planner, Python gate/regime, six rostopic recorders. Synthetic inactive observation source and named auditor subscriber. No task goal or robot; not closed-loop task-success attribution.')
    finally:
        for s in sockets: s.close()
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try: p.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=3)
        for stream in streams: stream.close()
        report['owned_remaining'] = [p.pid for p in children if p.poll() is None]
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], checks=len(report['checks']), output=str(out)), indent=2))


if __name__ == '__main__': main()
