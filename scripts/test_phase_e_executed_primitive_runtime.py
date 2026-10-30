#!/usr/bin/env python3
"""Isolated real ROS logger lifecycle + paired geometry timing; no execution."""
import argparse
import csv
import hashlib
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


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    uri = 'http://127.0.0.1:'+str(port)
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost', ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    children, handles = [], []; report = dict(status='FAIL', checks=[])
    sys.path.insert(0, str(ROOT/'src/care_confidence_map/scripts'))
    files = [ROOT/'src/care_confidence_map/scripts'/x for x in ['executed_vbc_logger_node.py', 'primitive_point_geometry.py']]
    files += [ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf', ROOT/'src/care_confidence_map/config/body_samples.yaml']
    digest = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report['sha256'] = digest()
    def start(cmd, name, trace=False):
        if trace:
            cmd = ['strace', '-f', '-qq', '-e', 'trace=%file', '-o', str(out/(name+'.files'))]+cmd
        h = (out/(name+'.log')).open('x'); handles.append(h)
        p = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT, start_new_session=True); children.append(p); return p
    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=5)
    def wait(predicate, label, seconds=15):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            if predicate(): return
            time.sleep(.02)
        raise AssertionError('Timeout '+label)
    try:
        start(['roscore', '-p', str(port)], 'master')
        def ready():
            try: return xmlrpc.client.ServerProxy(uri).getPid('/executed_test')[0] == 1
            except OSError: return False
        wait(ready, 'master')
        import rospy
        import roslaunch.config
        from geometry_msgs.msg import PointStamped, TransformStamped
        from std_msgs.msg import String
        from tf2_ros import StaticTransformBroadcaster
        from care_confidence_map.srv import QueryConfidence, QueryConfidenceResponse
        from primitive_point_geometry import load_primitives
        from executed_vbc_logger_node import ExecutedVBCLogger
        rospy.init_node('executed_primitive_test', disable_signals=True)
        urdf = str(files[-2]); samples = str(files[-1]); missing = str(out/'MISSING_body_samples.yaml')
        for parent in ['phaseB2_controlled_ab.launch', 'phaseB2_vbc_waypoint_controlled.launch']:
            for backend in ['samples', 'primitive']:
                cfg = roslaunch.config.load_config_default([(str(ROOT/'src/egocentric_arm_planner/launch'/parent),
                     ['geometry_backend:='+backend, 'body_samples_file:='+missing])], None, verbose=False)
                relevant = [p.value for k, p in cfg.params.items() if k.endswith('/geometry_backend')]
                assert len(relevant) >= 4 and all(v == backend for v in relevant)
                assert all(p.value == missing for k, p in cfg.params.items() if k.endswith('/body_samples_file'))
        report['checks'].append('both B2 launch trees propagate backend and missing YAML, old defaults supported')
        mode = [0.]; calls = []
        def query(req):
            calls.append(len(req.points)); return QueryConfidenceResponse([mode[0]]*len(req.points), [mode[0]]*len(req.points), [1]*len(req.points))
        service = rospy.Service('/executed_test/query', QueryConfidence, query)
        broadcaster = StaticTransformBroadcaster(); transforms = []
        for link in ['link1', 'link2', 'link3', 'link4', 'wrist_link1', 'wrist_link2', 'wrist_link3']:
            t = TransformStamped(); t.header.frame_id = 'base_link'; t.child_frame_id = link; t.transform.rotation.w = 1.
            transforms.append(t)
        broadcaster.sendTransform(transforms)
        model = load_primitives(urdf, ['base_link', 'link1'])
        target = PointStamped(); target.header.frame_id = 'base_link'
        target.point.x, target.point.y, target.point.z = model['link2'][0].center
        publishers, subscribers, observed, nodes = {}, [], {}, []
        for backend in ['samples', 'primitive']:
            name = 'executed_'+backend; observed[backend] = []
            params = dict(geometry_backend=backend, primitive_urdf_file=urdf,
                          body_samples_file=missing if backend == 'primitive' else samples,
                          target_topic='/executed_test/'+backend+'/target', confidence_query_service='/executed_test/query',
                          output_root=str(out/backend), ignored_links=['base_link', 'link1'], rate=50.)
            rospy.set_param('/'+name, params)
            topic = '/executed_test/'+backend+'/summary'
            subscribers.append(rospy.Subscriber(topic, String, lambda m, b=backend: observed[b].append(m.data)))
            publishers[backend] = rospy.Publisher(params['target_topic'], PointStamped, queue_size=1, latch=True)
            nodes.append(start([str(ROOT/'devel/lib/care_confidence_map/executed_vbc_logger_node.py'), '__name:='+name,
                               '/care_planner/trajectory_risk/executed_vbc_summary:='+topic], name, trace=True))
        wait(lambda: all(p.get_num_connections() for p in publishers.values()), 'both real logger nodes')
        for p in publishers.values(): p.publish(target)
        wait(lambda: all(observed.values()), 'CSV and summaries')
        assert 'sweep_latched=1' in observed['primitive'][-1] and 'seen_latched=0' in observed['primitive'][-1]
        assert 'closest_sample=-1' in observed['primitive'][-1] and 'closest_primitive_index=' in observed['primitive'][-1]
        mode[0] = 1.
        wait(lambda: all('seen_latched=1' in values[-1] for values in observed.values()), 'sensor-seen latch')
        report['checks'].append('real ROS both backends: no-YAML primitive, sweep/seen latch, compatible summary/CSV')
        mode[0] = float('nan'); time.sleep(.1)
        before = {k: len(v) for k, v in observed.items()}; time.sleep(.15)
        assert before == {k: len(v) for k, v in observed.items()}, 'invalid query produced valid diagnostic'
        report['checks'].append('NaN confidence blocks new rows/events on both backends')
        for node in nodes: stop(node)
        for backend in ['samples', 'primitive']:
            rows = list(csv.DictReader(next((out/backend).glob('*.csv')).open()))
            summary = json.loads(next((out/backend).glob('*.json')).read_text())
            assert rows and summary['geometry_backend'] == backend and summary['diagnostic_only']
            assert int(rows[-1]['seen_latched']) == 1
        assert 'body_samples.yaml' not in (out/'executed_primitive.files').read_text()
        assert 'body_samples.yaml' in (out/'executed_samples.files').read_text(), 'file tracer not effective'
        report['checks'].append('strace positive-control samples read; primitive zero body_samples YAML access; final JSON')
        # Pair the actual old/new _compute_min_clearance functions, with identical cached TF fixture.
        # No ROS service/disk/strace in timing; do not present this as full logger wall latency.
        import numpy as np
        cached = {t.child_frame_id: t for t in transforms}
        class Buffer:
            def lookup_transform(self, base, frame, stamp, timeout): return cached[frame]
        loggers = {}
        for backend in ['samples', 'primitive']:
            logger = ExecutedVBCLogger.__new__(ExecutedVBCLogger)
            logger.geometry_backend = backend; logger.ignored_links = {'base_link', 'link1'}
            logger.base_frame = 'base_link'; logger.tf_timeout = .01; logger.tf_buffer = Buffer()
            logger.primitives_by_frame = model; logger.samples_by_frame = logger._load_samples(Path(samples))
            loggers[backend] = logger
        timings = {'samples': [], 'primitive': []}; rng = np.random.RandomState(4)
        for i in range(210):
            point = PointStamped(); point.point.x, point.point.y, point.point.z = rng.uniform(-.3, .3, 3)
            for backend in (['samples', 'primitive'] if i % 2 else ['primitive', 'samples']):
                start_time = time.perf_counter(); result = loggers[backend]._compute_min_clearance(point)
                ms = (time.perf_counter()-start_time)*1000
                assert result is not None
                if i >= 10: timings[backend].append(ms)
        report['geometry_ms'] = {k: dict(p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95))) for k, v in timings.items()}
        report['performance'] = 'PASS' if all(report['geometry_ms']['primitive'][q] <= report['geometry_ms']['samples'][q] for q in ['p50', 'p95']) else 'FAIL'
        report['timing_scope'] = 'actual per-target geometry methods, cached synthetic link TF; excludes service and CSV/JSON'
        (out/'timings.json').write_text(json.dumps(timings, indent=2)+'\n')
        report['checks'].append('200 paired target geometry timings, same inputs, alternating order')
        # Startup and missing-TF rejection with the actual executable; no fallback.
        for label, update in [('bad_backend', {'geometry_backend': 'typo'}), ('missing_urdf', {'primitive_urdf_file': str(out/'MISSING.urdf')})]:
            params = dict(geometry_backend='primitive', primitive_urdf_file=urdf, body_samples_file=missing, output_root=str(out/label))
            params.update(update); rospy.set_param('/'+label, params)
            node = start([str(ROOT/'devel/lib/care_confidence_map/executed_vbc_logger_node.py'), '__name:='+label], label)
            wait(lambda: node.poll() is not None, label+' rejection'); assert node.returncode != 0
            assert not (out/label).exists()
        report['checks'].append('unknown backend/missing URDF exit nonzero before log outputs, no samples fallback')
        logger = loggers['primitive']
        class MissingBuffer:
            def lookup_transform(self, *args): raise RuntimeError('missing TF')
        logger.tf_buffer = MissingBuffer(); assert logger._compute_min_clearance(target) is None
        report['checks'].append('missing TF rejects complete geometry, no partial minimum')
        assert digest() == report['sha256'] and not Path(missing).exists()
        assert not any('command' in name or 'committed_trajectory' in name for name, _ in rospy.get_published_topics())
        report['checks'].append('immutable inputs; no actuator or committed publisher')
        report['status'] = 'PASS'
    except Exception as exc:
        report['error'] = repr(exc); raise
    finally:
        for p in reversed(children): stop(p)
        for h in handles: h.close()
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
