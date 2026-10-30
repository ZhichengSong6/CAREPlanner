#!/usr/bin/env python3
"""Private ROS context for an offline C++ geometry benchmark; no simulation or robot."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xmlrpc.client

from run_phase_e_repeatability import REPO, save
from run_phase_e_vbc_geometry_pair import distribution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--repetitions', type=int, default=6)
    args = ap.parse_args()
    assert args.repetitions >= 6 and args.repetitions % 6 == 0
    # Detect sandbox restrictions before allocating an output identity.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    uri = 'http://127.0.0.1:'+str(port)
    env = dict(os.environ, ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost',
               ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    binary = REPO/'devel/lib/care_confidence_map/benchmark_vbc_real_trajectories'
    paths = [binary, args.input.resolve(), REPO/'scripts/fixtures/vbc_primitive_sweep_reference.hpp',
        REPO/'scripts/benchmark_phase_e_vbc_real_trajectories.cpp',
        REPO/'src/care_confidence_map/include/care_confidence_map/vbc_primitive_model.hpp',
        REPO/'src/care_confidence_map/src/trajectory_vbc_selector_temporal_cluster_node.cpp',
        REPO/'src/care_confidence_map/config/body_samples.yaml',
        REPO/'src/arm_description/urdf/Arm.urdf',
        REPO/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
        REPO/'devel/lib/libtrajectory_risk_evaluator.so', REPO/'devel/lib/libbody_sample_model.so',
        REPO/'build/care_confidence_map/CMakeFiles/benchmark_vbc_real_trajectories.dir/flags.make']
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(out/'preflight.json', dict(sha256=frozen, cpu_affinity=sorted(os.sched_getaffinity(0)),
                                  cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0]))
    children = []
    report = dict(status='FAIL', scope='offline_geometry_only_no_robot')
    try:
        with (out/'roscore.log').open('x') as log:
            master = subprocess.Popen(['roscore', '-p', str(port)], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(master)
        ready = False
        for _ in range(100):
            try:
                ready = xmlrpc.client.ServerProxy(uri).getPid('/vbc_benchmark')[0] == 1
            except OSError:
                pass
            if ready: break
            assert master.poll() is None, 'private master failed'
            time.sleep(.1)
        assert ready, 'private master timeout'
        with (out/'cpp.log').open('x') as log:
            proc = subprocess.Popen([str(binary), str(REPO), str(args.input.resolve()),
                str(out/'timings.jsonl'), str(args.repetitions)], env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(proc)
            for _ in range(120):
                try:
                    proc.wait(timeout=5); break
                except subprocess.TimeoutExpired:
                    if _ % 6 == 5: print('C++ real-trajectory replay running', flush=True)
            assert proc.poll() == 0, 'C++ replay failed or timed out; see cpp.log'
        inputs = json.loads(args.input.read_text())
        lookup = {t['id']: t for t in inputs['trajectories']}
        rows = [json.loads(line) for line in (out/'timings.jsonl').read_text().splitlines()]
        assert len(rows) == len(lookup)*len(inputs['margins'])*args.repetitions*3
        groups = defaultdict(list)
        for row in rows:
            t = lookup[row['id']]
            for group in ('ALL', t['case_id'], t['case_id']+'/'+t['recorded_backend']+'/'+t['role']):
                groups[(group, row['margin'], row['backend'])].append(row)
        stats = []
        for (group, margin, backend), selected in sorted(groups.items()):
            stats.append(dict(group=group, margin=margin, backend=('samples','primitive_reference','primitive')[backend],
                timings={k:distribution([r[k] for r in selected]) for k in ('fk_ms','sweep_ms','total_ms')}))
        # A repeat paired by id/rep is retained in raw output, not replaced by
        # a ratio of medians. Acceptance below additionally checks each stratum.
        acceptance = []
        for s in [s for s in stats if s['backend'] == 'primitive']:
            for comparator in ('samples','primitive_reference'):
                a = next(a for a in stats if a['group']==s['group'] and a['margin']==s['margin'] and a['backend']==comparator)
                acceptance.append(dict(group=s['group'], margin=s['margin'], against=comparator,
                    total_pass=all(s['timings']['total_ms'][q] <= a['timings']['total_ms'][q] for q in ('p50','p95')),
                    sweep_pass=all(s['timings']['sweep_ms'][q] <= a['timings']['sweep_ms'][q] for q in ('p50','p95'))))
        assert frozen == {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'input/build changed during benchmark'
        report.update(status='PASS', trajectories=len(lookup), timing_rows=len(rows), repetitions=args.repetitions,
            stats=stats, acceptance=acceptance,
            sha256=frozen, frozen_inputs_equal=True,
            correctness='Exact ordered voxel/evidence comparison against frozen original; every measured primitive run',
            caveat='PASS is functional replay; performance pass/fail is separate in acceptance. '
                   'Only FK+sweep, not full VBC/service or rejected candidate replay. Same current build flags for all backends.')
        for s in stats:
            if s['group']=='ALL': print(s['margin'],s['backend'],s['timings'],flush=True)
    except Exception as e:
        report['error'] = repr(e)
        raise
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=5)
        save(out/'result.json', report)


if __name__ == '__main__':
    main()
