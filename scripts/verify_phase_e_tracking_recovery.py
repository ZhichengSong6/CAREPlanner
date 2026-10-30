#!/usr/bin/env python3
"""Verify tracking-protection recovery, without Gazebo or an actuator.

Python checks run production callbacks with fake transport/time.
--tracker-component runs the built C++ binary on a private ROS master with
manually stepped time and prescribed joint states (no dynamics simulation).
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
from unittest.mock import patch
import xmlrpc.client

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / 'outputs/phase_e_safety_contract_fix/20260915-160904-safety-contract-case011-016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text):
    return dict(s.split('=', 1) for s in text.split() if '=' in s)


class Recorder:
    def __init__(self, *args, **kwargs):
        self.messages = []

    def publish(self, msg):
        self.messages.append(copy.deepcopy(msg))

    def shutdown(self):
        pass


def load(name):
    path = REPO / 'src/egocentric_arm_planner/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def window_rows(case, name):
    w = case['window']
    with (Path(case['run']) / (name + '.csv')).open() as f:
        return [r for r in csv.DictReader(f)
                if w['start_ros_ns'] <= int(r['%time']) <= w['end_ros_ns']]


def replay_python(report):
    import rospy
    from std_msgs.msg import Bool, String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    gate_module = load('probe_single_flight_gate_node')
    manager_module = load('c4_4_verified_regime_manager_node')
    case = json.loads((BATCH / 'formal_2/evidence_audit.json').read_text())['cases'][1]
    rows = window_rows(case, 'tracker_summary')
    first = next(i for i, r in enumerate(rows)
                 if tokens(r['field.data']).get('source') == 'primitive_tracking_envelope_hold')
    outcome = next(r['field.data'] for r in window_rows(case, 'verification_outcome')
                   if tokens(r['field.data']).get('execution_stamp_ns') == '13240000000')
    token = int(tokens(outcome)['execution_stamp_ns'])
    prefix = float(tokens(outcome)['probe_effective_prefix_s'])
    candidate = JointTrajectory(points=[JointTrajectoryPoint(positions=[0.] * 7)])
    clock = [13.66]
    with patch.object(rospy, 'Publisher', Recorder), \
            patch.object(rospy, 'Subscriber', Recorder), \
            patch.object(rospy, 'Timer', Recorder), \
            patch.object(rospy, 'get_param', lambda key, default=None: default), \
            patch.object(rospy.Time, 'now', lambda: rospy.Time.from_sec(clock[0])), \
            patch.object(rospy, 'logwarn', lambda *a, **kw: None):
        def fixture():
            gate = gate_module.ProbeSingleFlightGate()
            manager = manager_module.C44VerifiedRegimeManager()
            manager.execution_ready = True
            manager.state = manager.PROBE_NORMAL
            gate._probe_active_cb(Bool(data=True))
            gate._candidate_cb(candidate)
            gate._verification_outcome_cb(String(data=outcome))
            manager._candidate_outcome_cb(String(data=outcome))
            assert gate.phase == gate.EXECUTING
            assert manager.pending_probe_execution_stamp_ns == token
            return gate, manager

        gate, manager = fixture()
        for row in rows[first:]:
            clock[0] = int(row['%time']) / 1e9
            fields = row['field.data']
            if tokens(fields).get('source') == 'primitive_tracking_envelope_hold':
                fields += ' execution_aborted=1'
            msg = String(data=fields)
            gate._tracker_summary_cb(msg)
            manager._tracker_summary_cb(msg)
            manager._timer_cb(None)
        for _ in range(66):
            gate._candidate_cb(candidate)
        # The first post-abort candidate is admitted for fresh verification;
        # later candidates remain serialized behind that new flight.
        assert gate.phase == gate.VERIFYING and gate.execution_stamp_ns == 0
        assert manager.pending_probe_execution_stamp_ns == 0
        assert manager.probe_completed_prefix_streak == 0
        assert gate.abort_release_count == 1 and gate.forward_count == 2
        assert len(manager.replan_request_pub.messages) == 1
        report['checks']['016_recorded_abort_releases_both_waiters'] = {
            'passed': True, 'tracker_rows_replayed': len(rows) - first,
            'token': token, 'prefix_s': prefix, 'abort_phase_s': .419,
            'additional_candidates_forwarded': gate.forward_count - 1,
            'manager_replan_pulses': len(manager.replan_request_pub.messages),
            'note': '66 synthetic candidate callbacks after recorded tracker replay; not full scheduler replay'}

        # Normal prefix completion is a positive control: the fixture can release.
        gate, manager = fixture()
        complete = String(data=f'active=1 complete=0 execution_stamp_ns={token} phase_s={prefix} source=active_trajectory')
        gate._tracker_summary_cb(complete)
        manager._tracker_summary_cb(complete)
        assert gate.phase == gate.IDLE and manager.pending_probe_execution_stamp_ns == 0
        assert manager.probe_completed_prefix_streak == 1
        report['checks']['valid_prefix_completion_releases_both_waiters'] = {'passed': True}

        # Wrong token does not count; an abort at the threshold does not count.
        gate, manager = fixture()
        wrong = String(data=f'active=0 complete=1 execution_stamp_ns={token-1} phase_s=2 source=trajectory_complete_hold')
        gate._tracker_summary_cb(wrong)
        manager._tracker_summary_cb(wrong)
        assert gate.phase == gate.EXECUTING and manager.probe_completed_prefix_streak == 0
        report['checks']['wrong_token_completion_ignored'] = {'passed': True}
        aborted = String(data=f'active=0 complete=0 execution_stamp_ns={token} phase_s={prefix} source=primitive_tracking_envelope_hold execution_aborted=1')
        gate._tracker_summary_cb(aborted)
        manager._tracker_summary_cb(aborted)
        assert gate.phase == gate.IDLE and manager.probe_completed_prefix_streak == 0
        report['checks']['abort_at_prefix_boundary_does_not_count_as_success'] = {
            'passed': True, 'scope': 'synthetic edge case'}

        gate, manager = fixture()
        abort = String(data=rows[first]['field.data'] + ' execution_aborted=1')
        for _ in range(3):
            gate._tracker_summary_cb(abort)
            manager._tracker_summary_cb(abort)
        # A matching completion arriving after abort is rejected as stale.
        delayed = String(data=f'active=0 complete=1 execution_stamp_ns={token} phase_s=2 source=trajectory_complete_hold')
        gate._tracker_summary_cb(delayed)
        manager._tracker_summary_cb(delayed)
        assert gate.phase == gate.IDLE and manager.probe_completed_prefix_streak == 0
        report['checks']['delayed_completion_after_abort_is_ignored'] = {
            'passed': True, 'scope': 'synthetic ordering injection'}


def tracker_component(out, report):
    children, streams = [], []
    subscribers = []
    summaries, commands, requests = [], [], []

    def start(argv, name):
        f = (out / name).open('x')
        streams.append(f)
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=REPO)
        children.append(p)
        return p

    def wait(pred, label, timeout=10):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred():
                return
            time.sleep(.01)
        raise AssertionError('timeout: ' + label)

    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        uri = f'http://127.0.0.1:{port}'
        os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost',
                          ROS_HOME=str(out / 'ros_home'), ROS_LOG_DIR=str(out / 'ros_logs'))
        start(['roscore', '-p', str(port)], 'roscore.log')

        def ready():
            try:
                return xmlrpc.client.ServerProxy(uri).getPid('/recovery_check')[0] == 1
            except OSError:
                return False
        wait(ready, 'private master')
        import rospy
        import yaml
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, String, Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('tracking_recovery_fixture', disable_signals=True)
        names = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']
        clock_pub = rospy.Publisher('/clock', Clock, queue_size=10, latch=True)
        js_pub = rospy.Publisher('/recovery_test/joints', JointState, queue_size=10)
        trajectory_pub = rospy.Publisher('/recovery_test/trajectory', JointTrajectory, queue_size=10)
        vbc_hold_pub = rospy.Publisher('/recovery_test/vbc_hold', Bool, queue_size=10, latch=True)
        gcdf_hold_pub = rospy.Publisher('/recovery_test/external_hold', Bool, queue_size=10, latch=True)
        subscribers.extend([
            rospy.Subscriber('/recovery_test/summary', String,
                             lambda m: summaries.append(dict(time=rospy.Time.now().to_sec(), data=m.data))),
            rospy.Subscriber('/recovery_test/command', Float64MultiArray,
                             lambda m: commands.append(list(m.data))),
            rospy.Subscriber('/recovery_test/replan', Bool,
                             lambda m: requests.append(dict(time=rospy.Time.now().to_sec(), data=m.data)))])
        config = yaml.safe_load((REPO / 'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        config['execution'].update(dict(
            primitive_tracking_guard=True, certified_tracking_margin_m=.010,
            primitive_urdf_file=str(REPO / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            joint_states='/recovery_test/joints', input_trajectory='/recovery_test/trajectory',
            output_velocity_command='/recovery_test/command', summary_topic='/recovery_test/summary',
            reference_state='/recovery_test/reference', replan_request_topic='/recovery_test/replan',
            smooth_replan_request_topic='/recovery_test/smooth_replan', smooth_handoff_enabled=False,
            external_safety_hold_topic='/recovery_test/external_hold',
            vbc_safety_hold_topic='/recovery_test/vbc_hold'))
        assert config['execution']['replan_request_min_interval'] == .5
        assert config['execution']['replan_tracking_error_inf'] == .25
        rospy.set_param('/recovery_tracker', config)
        (out / 'tracker_config.json').write_text(json.dumps(config, indent=2))
        clock_pub.publish(Clock(clock=rospy.Time.from_sec(100.)))
        start([str(REPO / 'devel/lib/egocentric_arm_planner/trajectory_execution_manager_node'),
               '__name:=recovery_tracker'], 'tracker.log')
        wait(lambda: js_pub.get_num_connections() and trajectory_pub.get_num_connections()
             and vbc_hold_pub.get_num_connections() and gcdf_hold_pub.get_num_connections()
             and all(s.get_num_connections() for s in subscribers), 'tracker transports')
        t = [100.]

        def joint(q):
            js_pub.publish(JointState(header=__import__('std_msgs.msg', fromlist=['Header']).Header(
                stamp=rospy.Time.from_sec(t[0])), name=names,
                position=[q] + [0.] * 6, velocity=[0.] * 7))
            time.sleep(.015)

        def step(dt, q):
            joint(q)
            n = len(summaries)
            t[0] = round(t[0] + dt, 9)
            clock_pub.publish(Clock(clock=rospy.Time.from_sec(t[0])))
            wait(lambda: len(summaries) > n, 'tracker tick')
            time.sleep(.015)
            return tokens(summaries[-1]['data'])

        def trajectory(q):
            msg = JointTrajectory(joint_names=names)
            msg.header.stamp = rospy.Time.from_sec(t[0])
            msg.points = [JointTrajectoryPoint(positions=[q] + [0.] * 6,
                         velocities=[0.] * 7, time_from_start=rospy.Duration(x)) for x in (0., 2.)]
            trajectory_pub.publish(msg)
            time.sleep(.04)
            return msg

        joint(0.)
        step(.02, 0.)
        # With the configuration-aware max primitive bound enabled, a 0.02 rad
        # joint-1 offset is only about 2.4 mm and should pass.  Use a larger
        # prescribed offset so this recovery fixture still exercises a real
        # >10 mm envelope abort rather than the legacy false positive.
        first = trajectory(.10)
        before_first = len(summaries)
        step(.02, 0.)
        first_records = [tokens(r['data']) for r in summaries[before_first:]]
        a = next(r for r in first_records if r.get('execution_aborted') == '1')
        assert a['source'] == 'primitive_tracking_envelope_hold' and a['execution_aborted'] == '1', a
        wait(lambda: len(requests) == 1, 'first recovery request')
        assert max(abs(x) for x in commands[-1]) == 0
        first_abort_time = t[0]
        # Prescribe a fresh measured state; there is no motion integration.
        b = step(.02, .02)
        assert b['source'] == 'zero_velocity_hold' and b['execution_stamp_ns'] == '0', b
        assert max(abs(x) for x in commands[-1]) == 0, commands[-1]
        second = trajectory(.20)
        before_second = len(summaries)
        step(.074, .10)
        # A stepped clock can dispatch multiple timer callbacks. The abort
        # token exists only in the first one; later records already contain 0.
        second_records = [tokens(r['data']) for r in summaries[before_second:]]
        c = next(r for r in second_records
                 if int(r['execution_stamp_ns']) == second.header.stamp.to_nsec())
        assert c['source'] == 'primitive_tracking_envelope_hold' and c['execution_aborted'] == '1', c
        assert int(c['execution_stamp_ns']) == second.header.stamp.to_nsec()
        assert len(requests) == 1, requests
        second_abort_time = t[0]
        d = step(.02, .19)
        assert d['source'] == 'zero_velocity_hold', d
        assert max(abs(x) for x in commands[-1]) == 0, commands[-1]
        old_hold_command = list(commands[-1])
        for _ in range(65):
            d = step(.02, .039)
        assert len(requests) >= 2 and d['active'] == '0' and d['complete'] == '0'
        report['checks']['011_second_abort_request_is_deferred'] = {
            'passed': True, 'first_abort_s': first_abort_time,
            'second_abort_s': second_abort_time, 'abort_gap_s': round(second_abort_time-first_abort_time, 6),
            'replan_requests': copy.deepcopy(requests), 'end_s': t[0],
            'note': 'actual C++ binary; request is deferred until the 0.50s interval elapses'}
        assert max(abs(x) for x in old_hold_command) == 0
        report['checks']['aborted_token_clears_old_reference_and_holds_zero'] = {
            'passed': True, 'post_abort_source': b['source'],
            'post_abort_token': b['execution_stamp_ns'], 'post_abort_command': old_hold_command}

        # Advisory request retry exists only while envelope violation persists.
        third = trajectory(.30)
        before_third = len(summaries)
        step(.02, .20)
        assert any(tokens(r['data']).get('execution_aborted') == '1'
                   for r in summaries[before_third:])
        wait(lambda: len(requests) >= 3, 'third token recovery request after interval')
        for _ in range(27):
            d = step(.02, .20)
        assert len(requests) >= 3
        report['checks']['persistent_violation_retries_after_interval'] = {'passed': True}

        # A delayed duplicate of an aborted token must be rejected.
        joint(.30)
        trajectory_pub.publish(third)
        time.sleep(.04)
        d = step(.02, .30)
        assert d['active'] == '0' and d['source'] == 'zero_velocity_hold', d
        report['checks']['aborted_token_duplicate_is_rejected'] = {
            'passed': True, 'scope': 'synthetic delayed duplicate on committed-input topic'}
        t[0] = round(t[0] + .01, 9)
        clock_pub.publish(Clock(clock=rospy.Time.from_sec(t[0])))
        fresh = trajectory(.30)
        d = step(.02, .30)
        assert d['active'] == '1' and int(d['execution_stamp_ns']) == fresh.header.stamp.to_nsec()
        report['checks']['fresh_matching_input_can_activate_tracker'] = {
            'passed': True, 'scope': 'tracker input only; not fresh GCDF/VBC certification'}
        vbc_hold_pub.publish(Bool(data=True))
        time.sleep(.03)
        d = step(.02, .30)
        assert d['source'] == 'external_vbc_hold' and max(abs(x) for x in commands[-1]) == 0, d
        gcdf_hold_pub.publish(Bool(data=True))
        time.sleep(.03)
        vbc_hold_pub.publish(Bool(data=False))
        time.sleep(.03)
        d = step(.02, .30)
        assert d['source'] == 'external_gcdf_hard_hold' and max(abs(x) for x in commands[-1]) == 0, d
        gcdf_hold_pub.publish(Bool(data=False))
        time.sleep(.03)
        d = step(.02, .30)
        assert d['active'] == '1' and d['source'] == 'active_trajectory', d
        report['checks']['vbc_hold_reaches_tracker_and_combines_with_gcdf_hold'] = {'passed': True}
        (out / 'tracker_trace.json').write_text(json.dumps(dict(
            summaries=summaries, commands=commands, replan_requests=requests), indent=2))
    finally:
        (out / 'tracker_trace.json').write_text(json.dumps(dict(
            summaries=summaries, commands=commands, replan_requests=requests), indent=2))
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait(timeout=5)
        for stream in streams:
            stream.close()
        report['owned_processes_stopped'] = all(p.poll() is not None for p in children)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tracker-component', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((BATCH / 'frozen_sha256.json').read_text())
    relevant = [p for p in frozen if any(s in p for s in (
        'trajectory_execution_manager', 'probe_single_flight_gate',
        'c4_4_verified_regime_manager', 'planner_c5_5_vbc_gcdf_regime.yaml',
        'Arm_with_self_filter_collision.urdf'))]
    report = dict(status='ERROR', meaning='isolated recovery component verification; no task simulation',
                  checks={}, frozen_input_matches={p: sha(REPO / p) == frozen[p] for p in relevant})
    try:
        expected_changed = {
            'src/egocentric_arm_planner/scripts/probe_single_flight_gate_node.py',
            'src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py',
            'src/egocentric_arm_planner/src/trajectory_execution_manager.cpp',
            'src/egocentric_arm_planner/include/egocentric_arm_planner/trajectory_execution_manager.hpp',
            'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml',
            'devel/lib/egocentric_arm_planner/probe_single_flight_gate_node.py',
            'devel/lib/egocentric_arm_planner/c4_4_verified_regime_manager_node.py',
            'devel/lib/egocentric_arm_planner/trajectory_execution_manager_node',
        }
        unexpected = [p for p, same in report['frozen_input_matches'].items()
                      if not same and p not in expected_changed]
        assert not unexpected, unexpected
        report['expected_changed_inputs'] = sorted(expected_changed)
        replay_python(report)
        if args.tracker_component:
            tracker_component(out, report)
        report['status'] = 'RECOVERY_COMPONENT_PASS'
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['frozen_unrelated_inputs_unchanged_after'] = all(
            sha(REPO / p) == frozen[p]
            for p in relevant
            if p not in {
                'src/egocentric_arm_planner/scripts/probe_single_flight_gate_node.py',
                'src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py',
                'src/egocentric_arm_planner/src/trajectory_execution_manager.cpp',
                'src/egocentric_arm_planner/include/egocentric_arm_planner/trajectory_execution_manager.hpp',
                'devel/lib/egocentric_arm_planner/probe_single_flight_gate_node.py',
                'devel/lib/egocentric_arm_planner/c4_4_verified_regime_manager_node.py',
                'devel/lib/egocentric_arm_planner/trajectory_execution_manager_node',
            })
        report['script_sha256'] = sha(Path(__file__))
        (out / 'result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
