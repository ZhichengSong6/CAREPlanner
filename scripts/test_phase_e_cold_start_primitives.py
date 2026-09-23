#!/usr/bin/env python3
"""Cold-start migration acceptance: private ROS only, no robot/Gazebo/cases/CUDA."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import threading
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    uri = 'http://127.0.0.1:{}'.format(port)
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost',
                      ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    children, handles = [], []
    report = dict(status='FAIL', scope='private ROS + offline fixtures; no cases or actuators')
    files = ['src/care_confidence_map/src/confidence_map_node.cpp',
             'src/care_confidence_map/config/body_samples.yaml',
             'src/care_confidence_map/config/confidence_map_phase_e_ray.yaml',
             'src/care_confidence_map/config/confidence_map.yaml',
             'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
             'src/care_confidence_map/include/care_confidence_map/primitive_geometry.hpp',
             'src/care_confidence_map/include/care_confidence_map/gcdf_primitive_anchors.hpp',
             'devel/lib/care_confidence_map/confidence_map_node',
             'devel/lib/care_confidence_map/test_cold_start_primitives',
             'devel/lib/libtrajectory_risk_evaluator.so', 'devel/lib/libbody_sample_model.so',
             'scripts/test_phase_e_cold_start_primitives.cpp',
             'scripts/test_phase_e_cold_start_primitives.py',
             'src/care_confidence_map/scripts/initialize_body_prior_once.py']
    def hashes():
        return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    report['sha256'] = hashes()

    def start(argv, name):
        log = (out/name).open('x'); handles.append(log)
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p)
        return p

    def wait(predicate, label, seconds=15):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(.02)
        raise AssertionError('Timeout: '+label)

    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=5)

    try:
        start(['roscore', '-p', str(port)], 'master.log')
        def ready():
            try:
                return xmlrpc.client.ServerProxy(uri).getPid('/prior_test')[0] == 1
            except OSError:
                return False
        wait(ready, 'private ROS master')
        import rospy
        import yaml
        import roslaunch.config
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs import point_cloud2
        from std_msgs.msg import Bool, String
        from std_srvs.srv import Trigger
        from tf2_ros import StaticTransformBroadcaster
        rospy.init_node('coldstart_acceptance', disable_signals=True)
        config = yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_phase_e_ray.yaml').read_text())
        config['current_body_prior']['body_samples_file'] = str(ROOT/'src/care_confidence_map/config/body_samples.yaml')
        config['current_body_prior']['primitive_urdf_file'] = str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
        rospy.set_param('/coldstart_test', config)
        p = start([str(ROOT/'devel/lib/care_confidence_map/test_cold_start_primitives'), str(ROOT), str(out)], 'cpp.log')
        wait(lambda: p.poll() is not None, 'C++ geometry/lifecycle/paired tests', 90)
        assert p.returncode == 0, 'C++ test failed: see cpp.log'
        report['cpp'] = 'PASS'
        report['coverage'] = [json.loads(s) for s in (out/'coverage.jsonl').read_text().splitlines()]
        rows = [json.loads(s) for s in (out/'timings.jsonl').read_text().splitlines()]
        assert len(rows) == 240
        stats = []
        for pose in range(3):
            pair = {}
            for backend in [0, 1]:
                v = sorted(r['ms'] for r in rows if r['pose']==pose and r['primitive']==backend)
                pair['primitive' if backend else 'samples'] = dict(p50=v[19], p95=v[37])
            pair['pose'] = pose
            pair['status'] = 'PASS' if all(pair['primitive'][k] <= pair['samples'][k] for k in ['p50','p95']) else 'FAIL'
            stats.append(pair)
        report['timing'] = stats
        report['performance'] = 'PASS' if all(s['status']=='PASS' for s in stats) else 'FAIL'
        launch = str(ROOT/'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch')
        for backend in ['samples','primitive']:
            c = roslaunch.config.load_config_default([(launch,['body_prior_geometry_backend:='+backend])],None,verbose=False)
            assert c.params['/confidence_map_node/current_body_prior/geometry_backend'].value == backend
            assert c.params['/confidence_map_node/current_body_prior/inflation_radius'].value == .1
        report['launch'] = 'PASS'
        # Real service and unmodified ready coordinator with synthetic static link TFs.
        # Deliberately not a physical pose or case; C++ above covers actual URDF FK.
        cfg = copy.deepcopy(config)
        cfg['current_body_prior']['geometry_backend'] = 'primitive'
        cfg['current_body_prior']['body_samples_file'] = str(out/'MISSING.yaml')
        rospy.set_param('/prior_runtime', cfg)
        node = start([str(ROOT/'devel/lib/care_confidence_map/confidence_map_node'), '__name:=prior_runtime'], 'node.log')
        service = cfg['current_body_prior']['refresh_service']
        rospy.wait_for_service(service, timeout=15)
        refresh = rospy.ServiceProxy(service, Trigger)
        deact = rospy.ServiceProxy(cfg['current_body_prior']['deactivate_service'], Trigger)
        ready_values, clouds, summaries = [], [], []
        subs = [rospy.Subscriber('/prior_test/ready', Bool, lambda m: ready_values.append(m.data)),
                rospy.Subscriber(cfg['confidence_map']['pointcloud_topic'], PointCloud2, lambda m: clouds.append(m)),
                rospy.Subscriber('/prior_initializer/summary', String, lambda m: summaries.append(m.data))]
        initializer = start(['/usr/bin/python3', str(ROOT/'src/care_confidence_map/scripts/initialize_body_prior_once.py'),
                             '__name:=prior_initializer', '_ready_topic:=/prior_test/ready'], 'initializer.log')
        wait(lambda: ready_values and clouds, 'initial pending ready/map')
        assert not any(ready_values)
        response = refresh()
        assert 'skipped_samples=0' not in response.message and 'transformed_samples=' in response.message
        def support(cloud):
            return {(x,y,z) for x,y,z,c in point_cloud2.read_points(cloud, field_names=('x','y','z','confidence')) if c>.5}
        assert not support(clouds[-1]), 'incomplete TF released FREE prior'
        broadcaster = StaticTransformBroadcaster()
        transforms = []
        for link in ['link1','link2','link3','link4','wrist_link1','wrist_link2','wrist_link3']:
            t = TransformStamped(); t.header.frame_id='base_link';t.child_frame_id=link;t.header.stamp=rospy.Time.now()
            t.transform.rotation.w=1.;transforms.append(t)
        broadcaster.sendTransform(transforms)
        wait(lambda: any(ready_values), 'complete TF ready')
        wait(lambda: summaries and 'transformed_samples=26' in summaries[-1], '26 primitives ready protocol')
        wait(lambda: support(clouds[-1]), 'effective prior cloud')
        initial = support(clouds[-1])
        assert 'locked' in refresh().message
        for t in transforms:
            t.transform.translation.x = .5;t.header.stamp=rospy.Time.now()
        broadcaster.sendTransform(transforms)
        count=len(clouds);wait(lambda:len(clouds)>count+2, 'post-motion map')
        assert support(clouds[-1]) == initial
        # Concurrent query/deactivation must expose one coherent map state,
        # never a partially cleared bootstrap or a cached SAFE after return.
        from care_confidence_map.srv import QueryConfidence
        from geometry_msgs.msg import Point
        query = rospy.ServiceProxy(cfg['confidence_map']['query_service'], QueryConfidence, persistent=True)
        points = [Point(*p) for p in sorted(initial)] * 2
        query_ready = threading.Event(); deactivated = threading.Event(); query_stop = threading.Event()
        query_errors, query_rows = [], []
        def query_during_deactivation():
            try:
                while not query_stop.is_set():
                    after = deactivated.is_set()
                    response = query(points)
                    assert len(response.confidence) == len(points) and all(response.inside_map)
                    values = set(response.confidence)
                    assert values in ({0.}, {1.}), values
                    if after: assert values == {0.}, 'stale SAFE after deactivation'
                    query_rows.append(dict(after=after, confidence=next(iter(values))))
                    query_ready.set()
                    time.sleep(.001)
            except Exception as exc:
                query_errors.append(repr(exc)); query_ready.set()
        reader = threading.Thread(target=query_during_deactivation, daemon=True); reader.start()
        wait(lambda: query_ready.is_set(), 'concurrent query ready')
        assert not query_errors
        assert deact().success
        deactivated.set()
        try:
            wait(lambda: query_errors or sum(r['after'] for r in query_rows) >= 20, 'post-deactivation live queries')
        finally:
            query_stop.set(); reader.join(timeout=5)
        assert not reader.is_alive() and not query_errors, query_errors
        report['concurrent_queries'] = dict(status='PASS', rows=query_rows)
        wait(lambda: not support(clouds[-1]), 'deactivation cloud')
        assert 'locked' in refresh().message
        count=len(clouds);wait(lambda:len(clouds)>count+2, 'no reactivation')
        assert not support(clouds[-1]) and len(summaries)==1
        assert node.poll() is None and initializer.poll() is None
        report['ros'] = ['partial TF blocks ready and FREE', '26 primitives + missing YAML accepted',
                         'original coordinator completes once', 'moving TF does not move prior',
                         'deactivation removes bootstrap; refresh cannot resurrect']
        pubs = rospy.get_published_topics()
        assert not any('command' in name or 'committed_trajectory' in name for name,_ in pubs)
        report['no_actuator'] = 'PASS'
        assert report['sha256'] == hashes(), 'sources or binaries changed during tests'
        report['status'] = 'PASS'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        for p in reversed(children):
            stop(p)
        for h in handles:
            h.close()
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
