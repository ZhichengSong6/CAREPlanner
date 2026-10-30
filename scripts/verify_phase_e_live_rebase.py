#!/usr/bin/env python3
"""Real native planner before/after controls on private ROS; no robot or task."""
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
EVIDENCE = ROOT / 'outputs/phase_e_five_point_repair_20261013'
CPP = Path('src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp')
HPP = Path('src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp')
OBJ = Path('build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--compile-only', action='store_true')
    ap.add_argument('--reuse-compiled', action='store_true')
    args = ap.parse_args()
    out = args.output_dir.resolve()
    if not args.reuse_compiled:
        out.mkdir(parents=True, exist_ok=False)
        flags = (ROOT / OBJ.parent / 'flags.make').read_text()
        inc = shlex.split(re.search(r'^CXX_INCLUDES = (.*)$', flags, re.M)[1])
        link = shlex.split((ROOT / OBJ.parent / 'link.txt').read_text())
        libraries = link[link.index('-o') + 2:]
        with (out / 'compile.log').open('x') as log:
            for variant in ('before', 'after'):
                folder = out / variant
                folder.mkdir()
                source = EVIDENCE / 'before' if variant == 'before' else ROOT
                (folder / 'planner_diagnostic_access.hpp').write_text(
                    (source / HPP).read_text().replace('private:', 'public:'))
                object_file = source / OBJ / 'local_sparse_scp_planner.cpp.o'
                if variant == 'before':
                    saved = json.loads((EVIDENCE / 'before_sha256.json').read_text())
                    assert sha(object_file) == saved[str(OBJ / 'local_sparse_scp_planner.cpp.o')]
                subprocess.run(['c++', '-std=c++14', '-O2', '-DCARE_FIXED=' + str(int(variant == 'after')),
                                *inc, '-I' + str(folder), str(ROOT / 'scripts/test_phase_e_live_rebase.cpp'),
                                str(object_file), str(source / OBJ / 'primitive_observation_guard.cpp.o'),
                                '-o', str(folder / 'test'), *libraries],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        inputs = [ROOT / CPP, ROOT / HPP, ROOT / OBJ / 'local_sparse_scp_planner.cpp.o',
                  ROOT / OBJ / 'primitive_observation_guard.cpp.o',
                  ROOT / 'scripts/test_phase_e_live_rebase.cpp', Path(__file__).resolve(),
                  out / 'before/test', out / 'after/test']
        (out / 'compiled_sha256.json').write_text(json.dumps({str(p): sha(p) for p in inputs}, indent=2) + '\n')
    if args.compile_only:
        print('COMPILE_PASS')
        return
    saved = json.loads((out / 'compiled_sha256.json').read_text())
    assert all(sha(Path(p)) == digest for p, digest in saved.items()), 'compiled inputs changed'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, ROS_MASTER_URI=f'http://127.0.0.1:{port}', ROS_HOSTNAME='127.0.0.1',
               ROS_HOME=str(out / 'ros_home'), ROS_LOG_DIR=str(out / 'ros_logs'))
    children, streams = [], []
    report = dict(status='FAIL', qualification='Controlled lock-handoff using actual jointStateCallback, '
                  'publishQueryTrajectory, requestPlanLocked, startPlan and CDF reception. '
                  'Frozen complete before/current planner objects. No robot, trained CDF, actuator or task.')

    def start(command, log_path):
        stream = log_path.open('x')
        streams.append(stream)
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(proc)
        return proc

    try:
        master = start(['roscore', '-p', str(port)], out / 'master.log')
        deadline = time.monotonic() + 10
        while True:
            try:
                if xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/live_rebase_driver')[0] == 1:
                    break
            except OSError:
                pass
            assert master.poll() is None and time.monotonic() < deadline
            time.sleep(.05)
        os.environ.update(env)
        import rospy
        import yaml
        from std_msgs.msg import Bool
        from trajectory_msgs.msg import JointTrajectory
        from egocentric_arm_planner.msg import CandidateTransfer
        rospy.init_node('live_rebase_driver', disable_signals=True)
        cfg = yaml.safe_load((ROOT / 'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        cfg['local_planner']['primitive_observation_guard'] = dict(enabled=True,
            fk_urdf=str(ROOT / 'src/arm_description/urdf/Arm.urdf'),
            primitive_urdf=str(ROOT / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            base_frame='base_link', ignored_links=['base_link', 'link1'], margin_m=.02)
        cfg['local_planner']['repair_executable_view'] = dict(enabled=True, prefix_s=.25, cycle_enabled=True,
            cycle_prefixes_s=[.25, .15, .10, .05], brake_dt_s=.05, max_brake_steps=20,
            acceleration_limits=[3., 3., 4., 4., 6., 6., 6.])
        cfg['local_planner']['repair_prefix_binding'] = dict(enabled=False)
        cfg['local_planner']['candidate_replacement_enabled'] = True
        rospy.set_param('/live_rebase_test', cfg)
        (out / 'test_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        arm, candidates = [], []
        subscribers = [rospy.Subscriber('/care_planner/task_start', Bool, lambda m: arm.append(m.data)),
                       rospy.Subscriber(cfg['local_planner']['candidate_trajectory_topic'], JointTrajectory,
                                        lambda m: candidates.append(m)),
                       rospy.Subscriber(cfg['local_planner']['candidate_trajectory_topic'] + '/planned',
                                        CandidateTransfer, lambda m: candidates.append(m))]
        for variant in ('before', 'after'):
            folder = out / variant
            env['ROOT_OUT'] = str(folder)
            proc = start([str(folder / 'test'), str(EVIDENCE / 'live_rebase_fixtures.json'), str(folder / 'results.json')], folder / 'run.log')
            assert proc.wait(timeout=30) == 0, variant
            report[variant] = json.loads((folder / 'results.json').read_text())
        assert len(report['after']['scenarios']) == 8
        latest = out / 'after/qp_repair_latest_failure.json'
        captured = json.loads(latest.read_text())
        assert captured['subproblem']['first_step_witness_conflicts'] == 0
        assert captured['subproblem']['later_witness_conflicts'] == 0
        assert not (out / 'before/qp_repair_latest_failure.json').exists()
        report['latest_coupled_capture_after_first_six'] = str(latest)
        assert not arm
        report.update(status='PASS', arm_messages=len(arm), candidate_messages=len(candidates),
                      controlled_scenarios=8,
                      qualification='Recorded CASE001/026 q/u/rebase/point inputs with controlled fresh positive/negative CDF; real PIQP and worker. Tests candidate admission only, no safety audit or actuator. Old candidates emitted; new branch requeries within six SCP steps or holds at exhaustion.')
    finally:
        remaining = []
        for proc in reversed(children):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
            if proc.poll() is None:
                remaining.append(proc.pid)
        for stream in streams:
            stream.close()
        report['cleanup_remaining'] = remaining
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
