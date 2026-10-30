#!/usr/bin/env python3
"""CASE012 replay against real VBC nodes with measured-body rejection controls.

Private ROS only: synthetic confidence and joint states, no controller or Gazebo.
The legacy switch supports a before/after replay of the identical fixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import xmlrpc.client

from run_phase_e_repeatability import fingerprint

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['joint1', 'joint2', 'joint3', 'joint4',
         'wrist_joint1', 'wrist_joint2', 'wrist_joint3']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--fixture', type=Path,
                    default=ROOT/'scripts/fixtures/phase_e_case012_sweep_refinement.json')
    ap.add_argument('--selector-binary', type=Path,
                    default=ROOT/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node')
    ap.add_argument('--expect-legacy', action='store_true')
    ap.add_argument('--timing-repetitions', type=int, default=20)
    args = ap.parse_args()
    if not 1 <= args.timing_repetitions <= 50:
        ap.error('timing repetitions must be 1..50')
    fixture = json.loads(args.fixture.read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen = fingerprint()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    uri = 'http://127.0.0.1:'+str(port)
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='127.0.0.1',
                      ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    children, streams = [], []
    timer = None
    report = {'status': 'FAIL', 'expect_legacy': args.expect_legacy,
              'qualification': 'real production selectors, frozen recorded trajectory, synthetic confidence and measured-state controls; private ROS without robot',
              'scenarios': {}, 'timing': {}}

    def start(argv, label):
        stream = (out/(label+'.log')).open('x')
        streams.append(stream)
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        children.append((label, process))
        return process

    def wait(test, timeout=15):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            if test():
                return
            time.sleep(.025)
        raise AssertionError('condition timed out')

    try:
        start(['roscore', '-p', str(port)], 'master')

        def ready():
            try:
                return xmlrpc.client.ServerProxy(uri).getPid('/execution_refinement_test')[0] == 1
            except OSError:
                return False

        wait(ready)
        import rospy
        import yaml
        from care_confidence_map.srv import QueryConfidence, QueryConfidenceResponse
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        rospy.init_node('execution_refinement_test', disable_signals=True)
        state = {'known': False, 'point': fixture['unknown_point'],
                 'q': fixture['measured_q_far'], 'publish': False, 'missing_joint': False}

        def confidence(req):
            point, known = list(state['point']), state['known']
            values = [1. if known or max(abs(p.x-point[0]), abs(p.y-point[1]),
                                        abs(p.z-point[2])) > 1e-8 else 0.
                      for p in req.points]
            return QueryConfidenceResponse(values, [0.]*len(values), [1]*len(values))

        service = rospy.Service('/execution_refinement_test/confidence', QueryConfidence, confidence)
        measured_pub = rospy.Publisher('/care_arm/joint_states', JointState, queue_size=1)

        def measured_tick(_):
            if not state['publish']:
                return
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name, msg.position = list(NAMES), list(state['q'])
            if state['missing_joint']:
                msg.name, msg.position = msg.name[:-1], msg.position[:-1]
            measured_pub.publish(msg)

        timer = rospy.Timer(rospy.Duration(.01), measured_tick)
        base = yaml.safe_load((ROOT/'src/care_confidence_map/config/trajectory_risk.yaml').read_text())['trajectory_vbc']
        summaries, publishers, subscribers = {}, {}, []
        for label, include in [('candidate', False), ('execution', True)]:
            name, topic = 'refinement_'+label, '/refinement_'+label
            config = dict(base, geometry_backend='primitive',
                robot_urdf_file=str(ROOT/fixture['urdf']),
                primitive_urdf_file=str(ROOT/fixture['primitive_urdf']),
                swept_volume_margin_m=.02, continuous_motion_bound_enabled=True,
                include_measured_body=include,
                input_trajectory_topic=topic+'/task', predicted_trajectory_topic=topic+'/prediction',
                output_target_topic=topic+'/target', candidate_active_topic=topic+'/active',
                active_set_points_topic=topic+'/points', active_set_bundle_topic=topic+'/bundle',
                force_bootstrap_topic=topic+'/bootstrap',
                confidence_query_service='/execution_refinement_test/confidence',
                prefer_predicted_trajectory=True, predicted_trajectory_timeout=3.,
                event_driven_eval=True, preserve_periodic_eval=include,
                eval_rate=20. if include else 5., min_visibility_before_sweep_margin_s=.3,
                map_resolution=.05, map_x_min=-.95, map_x_max=.95,
                map_y_min=-.95, map_y_max=.95, map_z_min=0., map_z_max=1.15)
            rospy.set_param('/'+name, {'trajectory_vbc': config})
            summaries[label] = []
            subscribers.append(rospy.Subscriber(topic+'/summary', String,
                lambda msg, key=label: summaries[key].append(msg.data), queue_size=100))
            process = start([str(args.selector_binary.resolve()), '__name:='+name,
                '/care_planner/trajectory_risk/vbc_summary:='+topic+'/summary'], label)
            pub = rospy.Publisher(topic+'/prediction', JointTrajectory, queue_size=1)
            wait(lambda: pub.get_num_connections() > 0 or process.poll() is not None)
            assert process.poll() is None
            publishers[label] = pub
        wait(lambda: measured_pub.get_num_connections() > 0)

        def evaluate(label, verdict=True):
            msg = JointTrajectory()
            msg.header.stamp, msg.joint_names = rospy.Time.now(), NAMES
            stamp = str(msg.header.stamp.to_nsec())
            for row in fixture['trajectory']:
                point = JointTrajectoryPoint(positions=row['q'])
                point.time_from_start = rospy.Duration(row['time_s'])
                msg.points.append(point)
            publishers[label].publish(msg)

            def matches():
                return [text for text in summaries[label]
                        if ' trajectory_stamp_ns='+stamp+' ' in text]

            if not verdict:
                time.sleep(.35)
                assert not matches(), 'invalid measured state produced a verdict'
                return {'no_verdict': True, 'trajectory_stamp_ns': stamp}
            wait(lambda: bool(matches()))
            fields = dict(re.findall(r'(\w+)=([^\s]+)', matches()[0]))
            if 'vbc_evidence' in fields:
                fields['evidence'] = json.loads(fields.pop('vbc_evidence'))
            assert 0 <= int(fields['sweep_refinement_fk']) <= 256
            return fields

        report['scenarios']['missing_initial_measurement'] = evaluate('execution', False)
        state['publish'] = True
        time.sleep(.1)
        for scenario in ['recorded_unknown', 'recorded_known', 'measured_inside', 'nominal_inside']:
            state['known'] = scenario == 'recorded_known'
            state['point'] = fixture['genuine_nominal_point'] if scenario == 'nominal_inside' else fixture['unknown_point']
            state['q'] = fixture['measured_q_inside'] if scenario == 'measured_inside' else fixture['measured_q_far']
            time.sleep(.1)
            candidate, execution = evaluate('candidate'), evaluate('execution')
            report['scenarios'][scenario] = {'candidate': candidate, 'execution': execution}
            if scenario in ('recorded_unknown', 'recorded_known'):
                assert candidate['has_violation'] == '0'
                assert execution['has_violation'] == ('1' if args.expect_legacy and scenario == 'recorded_unknown' else '0')
                if scenario == 'recorded_unknown':
                    assert candidate['sweep_refinement_excluded'] == '1'
                    assert execution['sweep_refinement_excluded'] == ('0' if args.expect_legacy else '1')
            elif scenario == 'measured_inside':
                assert candidate['has_violation'] == '0' and execution['has_violation'] == '1'
                assert execution['sweep_time_s'] == '0' and execution['sweep_refinement_excluded'] == '0'
                assert float(execution['primitive_signed_distance_m']) < .02
            else:
                assert candidate['has_violation'] == execution['has_violation'] == '1'
            print(scenario, 'candidate', candidate['has_violation'], 'execution', execution['has_violation'], flush=True)

        state.update(known=False, point=fixture['unknown_point'], q=fixture['measured_q_far'])
        time.sleep(.1)
        for label in ['candidate', 'execution']:
            report['timing'][label] = [evaluate(label) for _ in range(args.timing_repetitions)]
            assert all(r['has_violation'] == ('1' if label == 'execution' and args.expect_legacy else '0')
                       for r in report['timing'][label])
        count = len(summaries['execution'])
        time.sleep(.3)
        assert len(summaries['execution']) > count, 'execution periodic audit stopped'
        report['periodic_execution_audit_preserved'] = True
        state['publish'] = False
        time.sleep(.25)
        report['scenarios']['stale_measurement'] = evaluate('execution', False)
        state.update(publish=True, missing_joint=True)
        time.sleep(.1)
        report['scenarios']['missing_measured_joint'] = evaluate('execution', False)
        state['missing_joint'] = False
        time.sleep(.1)
        recovered = evaluate('execution')
        assert recovered['has_violation'] == ('1' if args.expect_legacy else '0')
        report['scenarios']['fresh_measurement_recovery'] = recovered
        assert fingerprint() == frozen, 'source/binary changed during fixed test'
        report.update(status='PASS', source_fingerprint_unchanged=True,
            input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [args.fixture.resolve(), args.selector_binary.resolve(),
                                    ROOT/'devel/lib/libtrajectory_risk_evaluator.so', Path(__file__).resolve()]})
    finally:
        if timer is not None:
            timer.shutdown()
        for label, process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
        for stream in streams:
            stream.close()
        report['cleanup'] = {'remaining': [p.pid for _, p in children if p.poll() is None]}
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS', 'legacy' if args.expect_legacy else 'aligned', flush=True)


if __name__ == '__main__':
    main()
