#!/usr/bin/env python3
"""Same-input production before/after tests on private ROS; no robot/world."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    artifact = ROOT/'outputs/phase_e_cdf_conflict_handling_fix_20261010'
    flags = (ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/flags.make').read_text()
    inc = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M)[1])
    defines = shlex.split(re.search(r'^CXX_DEFINES = (.*)$', flags, re.M)[1])
    link = shlex.split((ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/link.txt').read_text())
    libs = link[link.index('-o')+2:]
    obj = ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src'
    header_path = Path('src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp')
    cpp_path = Path('src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp')
    with (out/'compile.log').open('x') as log:
        for variant in ['before', 'after']:
            folder = out/variant; folder.mkdir()
            source_root = artifact/'before' if variant == 'before' else ROOT
            header = (source_root/header_path).read_text()
            (folder/'planner_diagnostic_access.hpp').write_text(header.replace('private:', 'public:'))
            if variant == 'before':
                shim = folder/'include/egocentric_arm_planner'; shim.mkdir(parents=True)
                (shim/'local_sparse_scp_planner.hpp').write_text(header)
                object_file = folder/'planner.o'
                cmd = ['c++', '-std=c++14', '-O3', *defines, '-I'+str(folder/'include'), *inc,
                       '-c', str(source_root/cpp_path), '-o', str(object_file)]
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            else:
                object_file = obj/'local_sparse_scp_planner.cpp.o'
            cmd = ['c++', '-std=c++14', '-O2', '-DCARE_NEW='+str(int(variant == 'after')),
                   *inc, '-I'+str(folder), str(ROOT/'scripts/test_phase_e_cdf_conflict_handling.cpp'),
                   str(object_file), str(obj/'primitive_observation_guard.cpp.o'),
                   '-o', str(folder/'test'), *libs]
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

    env = dict(os.environ, ROS_MASTER_URI='http://127.0.0.1:'+str(port),
               ROS_HOSTNAME='127.0.0.1', ROS_IP='127.0.0.1',
               ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    children = []; streams = []
    def start(argv, path, extra=None):
        stream = path.open('x'); streams.append(stream)
        child = subprocess.Popen(argv, env=dict(env, **(extra or {})),
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child); return child
    report = dict(status='FAIL', scope='Frozen full CDF rows plus controlled linear field; real production PIQP and worker. No closed loop, original nonlinear map reconstruction, physical collision or model retraining claim.')
    try:
        master = start(['roscore', '-p', str(port)], out/'master.log')
        end = time.monotonic()+10
        proxy = xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        while True:
            try:
                if proxy.getPid('/cdf_test')[0] == 1: break
            except OSError: pass
            assert time.monotonic() < end and master.poll() is None, 'private master unavailable'
            time.sleep(.05)
        os.environ.update({k: env[k] for k in ['ROS_MASTER_URI', 'ROS_HOSTNAME', 'ROS_IP', 'ROS_HOME', 'ROS_LOG_DIR']})
        import rospy
        import yaml
        rospy.init_node('cdf_conflict_test_driver', disable_signals=True)
        config = yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        # No VBC feedback primitive is present in these inputs. Disable that
        # separate proposal layer to match the original zero geometry rows.
        config['local_planner']['primitive_observation_guard'] = dict(enabled=False)
        # R1 supplies this runtime switch through the launch file, outside
        # the YAML. Exercise the same production branch on both variants.
        config['local_planner']['candidate_replacement_enabled'] = True
        rospy.set_param('/cdf_conflict_validation', config)
        fixture = ROOT/'outputs/phase_e_cdf_unreachable_8_20261010/report.json'
        variants = {}
        for variant in ['before', 'after']:
            folder = out/variant
            child = start([str(folder/'test'), str(fixture), str(folder/'results.json')],
                          folder/'run.log', dict(ROOT_OUT=str(folder)))
            assert child.wait(timeout=45) == 0, variant+' production test failed'
            variants[variant] = json.loads((folder/'results.json').read_text())
        assert variants['before']['frozen_cases'] == variants['after']['frozen_cases']
        assert not variants['before']['worker_cases'][0]['requery']
        assert variants['after']['worker_cases'][0]['requery']
        assert all(not r['requery'] for r in variants['after']['worker_cases'][1:])
        snapshots = sorted((out/'after').glob('qp_repair_conflict_*.json'))
        assert len(snapshots) == 6
        for p in snapshots:
            snap = json.loads(p.read_text()); sub = snap['subproblem']
            assert not snap['executable'] and sub['selected_cdf_rows'] == len(sub['cdf_rows']) == 20
            assert sub['G']['shape'][0] >= len(sub['cdf_rows'])
            assert all(row['margin'] == .05 and row['repair_witness'] for row in sub['cdf_rows'])
            assert snap['mode_epoch'] == 1
        assert all(not json.loads(p.read_text())['margin_rollback_proven'] for p in snapshots)
        assert not list((out/'before').glob('qp_repair_conflict_*.json'))
        report.update(status='PASS', variants=variants, exact_snapshot_count=len(snapshots),
                      snapshot_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshots})
    finally:
        for child in reversed(children):
            if child.poll() is None: os.killpg(child.pid, signal.SIGINT)
        for child in reversed(children):
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
        report['owned_remaining'] = [p.pid for p in children if p.poll() is None]
        for stream in streams: stream.close()
        (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['variants', 'snapshot_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
