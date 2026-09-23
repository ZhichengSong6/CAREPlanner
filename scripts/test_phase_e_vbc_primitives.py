#!/usr/bin/env python3
"""Stage-one C++ primitive geometry/benchmark and isolated ROS selector tests.

Starts only a private roscore, selectors and a synthetic confidence service.
No simulator, robot, controller, model inference, or production map is started.
"""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--oracle-python', default='/home/zhicheng/miniconda3/envs/viscdf/bin/python')
    ap.add_argument('--execution-event-policy', action='store_true', help='Also verify immediate evaluation retains full periodic map checks')
    ap.add_argument('--persistent-confidence', action='store_true', help='Also kill/restart a separate confidence server to test fail-closed reconnect')
    ap.add_argument('--verification-tcp-nodelay', action='store_true')
    ap.add_argument('--shadow-policy', action='store_true', help='Same input/predicted topic as production shadow, event + original 20Hz audit')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Binding also detects sandbox socket restrictions before starting children.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    uri = f'http://127.0.0.1:{port}'
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost',
                      ROS_HOME=str(out / 'ros_home'), ROS_LOG_DIR=str(out / 'ros_logs'))
    children, handles, checks = [], [], []
    report = dict(status='FAIL', checks=checks, cpp=[], scope='isolated_no_robot')

    def start(argv, name):
        stream = (out / name).open('x')
        handles.append(stream)
        p = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p)
        return p

    def wait_for(predicate, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.025)
        raise AssertionError('timed out waiting for test condition')

    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=5)

    try:
        start(['roscore', '-p', str(port)], 'roscore.log')

        def ready():
            try:
                return xmlrpc.client.ServerProxy(uri).getPid('/primitive_test')[0] == 1
            except OSError:
                return False
        wait_for(ready)
        cpp = start([str(repo / 'devel/lib/care_confidence_map/test_vbc_primitives'),
                     str(repo), str(out)], 'cpp.log')
        cpp.wait(timeout=300)
        report['cpp'] = [json.loads(line) for line in (out / 'cpp.log').read_text().splitlines()
                         if line.startswith('{"test":')]
        assert cpp.returncode == 0, 'C++ checks failed; see cpp.log'
        assert len(report['cpp']) == 5, 'missing C++ test/benchmark results'
        checks.append('C++ geometry, relative-FK max bound, loader faults, voxel AABB, witness identity and knot timing PASS')
        oracle = start([args.oracle_python, str(repo / 'scripts/verify_phase_e_vbc_primitive_fk.py'),
                        str(out / 'primitive_fk.json')], 'independent_oracle.log')
        assert oracle.wait(timeout=30) == 0, 'independent FK/Coal oracle failed'
        report['independent_oracle'] = json.loads((out / 'independent_oracle.json').read_text())
        checks.append('independent Python URDF FK and Coal point-to-solid distances PASS')

        import rospy
        import yaml
        import roslaunch
        from care_confidence_map.srv import QueryConfidence, QueryConfidenceResponse
        from std_msgs.msg import String
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        rospy.init_node('primitive_integration_test', disable_signals=True)
        class Mode(list):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if args.persistent_confidence:
                    rospy.set_param('/primitive_test/mode', value)
        mode = Mode(['unknown']); mode[0] = 'unknown'

        def confidence(req):
            if mode[0] == 'invalid':
                return QueryConfidenceResponse([], [], [])
            n = len(req.points)
            return QueryConfidenceResponse([1. if mode[0] == 'known' else 0.] * n, [0.] * n, [1] * n)

        service = None
        def start_server(label):
            return start(['/usr/bin/python3', str(repo/'scripts/phase_e_test_confidence_server.py')], label+'.log')
        if args.persistent_confidence:
            server = start_server('confidence_server')
        else:
            service = rospy.Service('/primitive_test/query', QueryConfidence, confidence)
        base_config = yaml.safe_load((repo / 'src/care_confidence_map/config/trajectory_risk.yaml').read_text())
        binary = str(repo / 'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node')
        urdf = str(repo / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
        fk_urdf = str(repo / 'src/arm_description/urdf/Arm.urdf')
        yaml_path = str(repo / 'src/care_confidence_map/config/body_samples.yaml')
        summaries = {}
        subscribers = []
        report['audits'] = {}
        for backend in ('samples', 'primitive'):
            name = f'primitive_test_{backend}'
            topic = '/' + name
            config = dict(base_config['trajectory_vbc'], geometry_backend=backend,
                          robot_urdf_file=fk_urdf, primitive_urdf_file=urdf,
                          body_samples_file=yaml_path if backend == 'samples' else '/missing/samples.yaml',
                          base_frame='base_link', input_trajectory_topic=topic + '/task',
                          predicted_trajectory_topic=topic + '/prediction', prefer_predicted_trajectory=True,
                          predicted_trajectory_timeout=.5, event_driven_eval=True, eval_rate=5.,
                          output_target_topic=topic + '/target', candidate_active_topic=topic + '/active',
                          active_set_points_topic=topic + '/points', active_set_bundle_topic=topic + '/bundle',
                          force_bootstrap_topic=topic + '/bootstrap', confidence_query_service='/primitive_test/query')
            if args.execution_event_policy:
                config.update(preserve_periodic_eval=True, eval_rate=20., predicted_trajectory_timeout=2.)
            if args.shadow_policy:
                config.update(input_trajectory_topic=topic+'/prediction', preserve_periodic_eval=True,
                              event_driven_eval=True, eval_rate=20., predicted_trajectory_timeout=.5)
            config['confidence_query_persistent'] = args.persistent_confidence
            config['tcp_nodelay'] = args.verification_tcp_nodelay
            rospy.set_param('/' + name, {'trajectory_vbc': config})
            summaries[backend] = []
            subscribers.append(rospy.Subscriber(topic + '/summary', String,
                               lambda m, key=backend: summaries[key].append(m.data)))
            p = start([binary, '__name:=' + name,
                       '/care_planner/trajectory_risk/vbc_summary:=' + topic + '/summary'], name + '.log')
            pub = rospy.Publisher(topic + '/prediction', JointTrajectory, queue_size=1)
            wait_for(lambda: pub.get_num_connections() > 0 or p.poll() is not None)
            assert p.poll() is None, f'{backend} initialization failed'
            msg = JointTrajectory()
            msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
            # Read names from the URDF, not an assumed arm naming convention.
            import xml.etree.ElementTree as ET
            msg.joint_names = [j.attrib['name'] for j in ET.parse(urdf).getroot().findall('joint')
                               if j.attrib['type'] in ('revolute', 'prismatic', 'continuous')]
            assert len(msg.joint_names) == 7
            for k in range(20):
                point = JointTrajectoryPoint(positions=[0.] * 7)
                point.time_from_start = rospy.Duration(k * .05)
                msg.points.append(point)

            def publish():
                msg.header.stamp = rospy.Time.now()
                stamp = str(msg.header.stamp.to_nsec())
                pub.publish(msg)
                return stamp

            def matching(stamp):
                return [s for s in summaries[backend] if f' trajectory_stamp_ns={stamp} ' in s]

            mode[0] = 'unknown'
            stamp = publish()
            wait_for(lambda: bool(matching(stamp)))
            raw = matching(stamp)[-1]
            assert 'has_violation=1 ' in raw, raw
            tokens = dict(x.split('=', 1) for x in raw.split() if '=' in x)
            evidence = json.loads(tokens['vbc_evidence'])
            assert evidence['geometry_backend'] == backend
            assert evidence['trajectory_stamp_ns'] == stamp and evidence['violations']
            if backend == 'primitive':
                for v in evidence['violations']:
                    assert v['sample_index'] == -1 and v['raw_radius_m'] is None and v['swept_radius_m'] is None
                    assert v['source_collision_name'] and v['body_source'].startswith('primitive_')
                    assert v['primitive_signed_distance_m'] <= 1e-12
            assert any(v['sweep_original_k'] == 0 and v['latest_see_s'] < 0 for v in evidence['violations'])
            report['audits'][backend] = dict(unknown_summary=raw, violations=len(evidence['violations']))
            # Warm production-process total_eval_ms includes the real query
            # service, sensor FK, visibility scan and clustering, but ends before
            # output serialization/publication (the existing timing contract).
            # The separate C++ benchmark above interleaves varied trajectories.
            samples_ms = []
            for trial in range(32):
                stamp = publish()
                wait_for(lambda: bool(matching(stamp)))
                fields = dict(x.split('=', 1) for x in matching(stamp)[0].split() if '=' in x)
                assert fields['has_violation'] == '1'
                if trial >= 8:
                    samples_ms.append({key: float(fields[key]) for key in
                                       ('body_fk_ms', 'swept_voxel_build_ms', 'total_eval_ms')})
            def quantile(values, fraction):
                return sorted(values)[int(fraction * (len(values) - 1))]
            report['audits'][backend]['warm_ros_timing'] = {
                'repetitions': len(samples_ms),
                **{key: {'p50': quantile([row[key] for row in samples_ms], .5),
                         'p95': quantile([row[key] for row in samples_ms], .95)}
                   for key in samples_ms[0]}}
            mode[0] = 'known'
            stamp = publish()
            wait_for(lambda: bool(matching(stamp)))
            assert 'has_violation=0 ' in matching(stamp)[-1]
            report['audits'][backend]['known_summary'] = matching(stamp)[-1]
            if args.execution_event_policy or args.shadow_policy:
                before = len(matching(stamp))
                wait_for(lambda: len([s for s in matching(stamp)[before:] if 'evaluation_trigger=timer ' in s]) >= 2)
                mode[0] = 'unknown'
                # No new trajectory: a map change must still be re-audited.
                wait_for(lambda: any('has_violation=1 ' in s and 'evaluation_trigger=timer ' in s
                                     for s in matching(stamp)[before:]))
                checks.append(f'{backend}: unchanged trajectory is periodically re-audited after map changes with event mode PASS')
                if args.shadow_policy:
                    callbacks = [s for s in matching(stamp) if 'evaluation_trigger=predicted_callback ' in s]
                    assert len(callbacks) == 1, 'same-topic shadow should evaluate once on arrival'
                    assert 'predicted_periodic_skip_count=0 ' in matching(stamp)[-1]+' '
                    checks.append(f'{backend}: shadow shared input topic has one arrival evaluation, periodic skip remains zero PASS')
            if args.persistent_confidence:
                # Destroy the actual host process, not just unregister a service
                # while a persistent connection might still keep its handler alive.
                mode[0] = 'known'; stop(server)
                stamp = publish(); time.sleep(.65)
                assert not matching(stamp), 'dead server produced a certificate'
                mode[0] = 'unknown'
                server = start_server('confidence_restart_'+backend)
                rospy.wait_for_service('/primitive_test/query', timeout=5.)
                stamp = publish(); wait_for(lambda: bool(matching(stamp)))
                assert 'has_violation=1 ' in matching(stamp)[-1], 'reconnect reused old clear response'
                mode[0] = 'error'
                stamp = publish(); time.sleep(.65)
                assert not matching(stamp), 'service exception produced a certificate'
                mode[0] = 'unknown'
                stamp = publish(); wait_for(lambda: bool(matching(stamp)))
                assert 'has_violation=1 ' in matching(stamp)[-1]
                checks.append(f'{backend}: server death and exception fail closed; restart/reconnect queries fresh UNKNOWN PASS')
            mode[0] = 'invalid'
            stamp = publish()
            time.sleep(.65)
            assert not matching(stamp), 'invalid confidence response certified safe'
            mode[0] = 'unknown'
            msg.points[0].positions[0] = float('nan')
            stamp = publish()
            time.sleep(.65)
            assert not matching(stamp), 'nonfinite q certified safe'
            assert p.poll() is None, 'fault caused selector crash'
            stop(p)
            checks.append(f'{backend}: unknown unsafe, known clear, identity retained, malformed response/NaN q fail closed PASS')

        # Invalid configuration is rejected before subscriptions/timers start.
        for label, backend, path in [('bad_backend', 'primtive', urdf),
                                     ('mesh_urdf', 'primitive', str(repo / 'src/arm_description/urdf/Arm.urdf'))]:
            name = 'primitive_test_' + label
            rospy.set_param('/' + name, {'trajectory_vbc': dict(base_config['trajectory_vbc'],
                geometry_backend=backend, robot_urdf_file=fk_urdf, primitive_urdf_file=path,
                base_frame='base_link', body_samples_file=yaml_path)})
            p = start([binary, '__name:=' + name], label + '.log')
            wait_for(lambda: p.poll() is not None)
            assert p.returncode != 0, 'bad configuration accepted'
        checks.append('unknown backend and mesh URDF reject startup without fallback PASS')

        launch = str(repo / 'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch')
        for backend in ('samples', 'primitive'):
            config = roslaunch.config.load_config_default([(launch, ['vbc_geometry_backend:=' + backend])], None)
            routed = {k: v.value for k, v in config.params.items()
                      if k.endswith('/trajectory_vbc/geometry_backend')}
            assert len(routed) == 2 and set(routed.values()) == {backend}, routed
            for key in routed:
                prefix = key.rsplit('/', 1)[0]
                path = config.params[prefix + '/robot_urdf_file'].value
                assert Path(path).name == 'Arm.urdf'
                assert Path(config.params[prefix + '/primitive_urdf_file'].value).name == 'Arm_with_self_filter_collision.urdf'
                assert config.params[prefix + '/swept_volume_margin_m'].value == 0
        checks.append('expanded parent launch binds both candidate/execution selectors to same backend/URDF/margin PASS')
        assert not any('controller/command' in topic for topic, _ in rospy.get_published_topics())
        checks.append('no controller command publisher PASS')
        if service is not None:
            service.shutdown()
        ros_samples = report['audits']['samples']['warm_ros_timing']['total_eval_ms']
        ros_primitive = report['audits']['primitive']['warm_ros_timing']['total_eval_ms']
        report['warm_ros_performance'] = dict(samples=ros_samples, primitive=ros_primitive,
            status='PASS' if all(ros_primitive[k] <= ros_samples[k] for k in ('p50', 'p95')) else 'FAIL')
        report['sha256'] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(urdf), Path(fk_urdf), Path(yaml_path), Path(binary), repo / 'devel/lib/libtrajectory_risk_evaluator.so']}
        report['status'] = 'PASS' if (all(row['status'] == 'PASS' for row in report['cpp']) and
                                      report['warm_ros_performance']['status'] == 'PASS') else 'FAIL_PERFORMANCE'
    except Exception as e:
        report['error'] = repr(e)
        raise
    finally:
        (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        for p in reversed(children):
            stop(p)
        for stream in handles:
            stream.close()
        print(json.dumps({k: v for k, v in report.items() if k != 'audits'}, indent=2), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
