#!/usr/bin/env python3
"""Private ROS queue isolation probe: missing marker TF, real UNKNOWN map.

No production thresholds/rates are changed; no robot, GPU or controller.
Compares the two query scheduling policies in the SAME installed executable.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import xmlrpc.client


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install-prefix', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args(); prefix = args.install_prefix.resolve(); out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    uri = f'http://127.0.0.1:{port}'
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost', ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    children, logs = [], []
    result = dict(status='FAIL', scope=__doc__, runs=[])
    def start(cmd, label):
        handle = (out/(label+'.log')).open('x'); logs.append(handle)
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(proc); return proc
    def stop(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try: proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=5)
    try:
        start(['roscore', '-p', str(port)], 'master')
        deadline = time.monotonic()+15
        while True:
            try:
                if xmlrpc.client.ServerProxy(uri).getPid('/query_queue_probe')[0] == 1: break
            except OSError: pass
            assert time.monotonic() < deadline, 'master startup timeout'
            time.sleep(.05)
        import rospy
        import yaml
        from geometry_msgs.msg import Point
        from care_confidence_map.srv import QueryConfidence
        rospy.init_node('query_queue_probe', disable_signals=True)
        base = yaml.safe_load((prefix/'share/care_confidence_map/config/confidence_map_phase_e_ray.yaml').read_text())
        base['current_body_prior']['enabled'] = False
        # No TF broadcaster: real marker lookups exercise their unchanged wait.
        for index, enabled in enumerate([False, True, True, False]):
            name = 'queue_probe_'+str(index); config = copy.deepcopy(base)
            service = '/'+name+'/query'
            config['confidence_map'].update(query_dedicated_queue=enabled, query_service=service)
            rospy.set_param('/'+name, config)
            proc = start([str(prefix/'lib/care_confidence_map/confidence_map_node'), '__name:='+name], name)
            rospy.wait_for_service(service, timeout=10)
            query = rospy.ServiceProxy(service, QueryConfidence, persistent=True)
            rows = []
            for i in range(202):
                begin = time.monotonic()
                response = query([Point(.1, .1, .3)]*50)
                elapsed = (time.monotonic()-begin)*1000
                assert len(response.confidence) == 50 and all(v == 0 for v in response.confidence)
                assert all(response.inside_map) and all(v == 0 for v in response.current_visibility)
                if i >= 2: rows.append(elapsed)
                time.sleep(.003)
            query.close(); stop(proc)
            assert proc.returncode == 0, ('shutdown', proc.returncode)
            ordered = sorted(rows)
            result['runs'].append(dict(dedicated_queue=enabled, latency_ms=rows,
                p50_ms=ordered[99], p95_ms=ordered[189], max_ms=max(rows), semantics='UNKNOWN'))
        # Diagnostic timing, not a replacement for the fixed mainline gate.
        assert not any('command' in t or 'committed_trajectory' in t for t, _ in rospy.get_published_topics())
        result['status'] = 'PASS'
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        for proc in reversed(children): stop(proc)
        for handle in logs: handle.close()
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'runs'}, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
