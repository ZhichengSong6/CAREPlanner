#!/usr/bin/env python3
"""One original case through the existing full Gazebo runner; never hardware.

Fresh outputs, private ROS/Gazebo masters, owned-process cleanup, no ZIP/batch.
This is a bounded deployment smoke, not a case-success or performance gate.
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
import traceback
import xmlrpc.client
from phase_e_owned_processes import remember_descendants, reap_owned

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); return sock.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--case-id', default='phase_e_goal_000')
    ap.add_argument('--run-seconds', type=float, default=30.)
    args = ap.parse_args()
    if not 1 <= args.run_seconds <= 60: ap.error('bounded smoke requires 1..60 seconds')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    cases = ROOT/'outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json'
    case = next(c for c in json.loads(cases.read_text())['cases'] if c['case_id'] == args.case_id)
    assert len(case['initial_q']) == 7 and max(abs(q) for q in case['initial_q']) < 1e-4
    ros_port, gazebo_port = free_port(), free_port()
    while gazebo_port == ros_port: gazebo_port = free_port()
    env = dict(os.environ)
    profile = dict(
        ROS_MASTER_URI=f'http://127.0.0.1:{ros_port}', ROS_HOSTNAME='localhost',
        GAZEBO_MASTER_URI=f'http://127.0.0.1:{gazebo_port}', GAZEBO_MODEL_DATABASE_URI='',
        ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'),
        REPO=str(ROOT), CASE_FILE=str(cases), CASE_ID=args.case_id, RUN_SECONDS=str(args.run_seconds),
        OUT=str(out/'run'), LOG=str(out/'logs'),
        CONFIG_FILE=str(ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml'),
        WORLD_FILE=str(ROOT/'src/arm_description/worlds/maixsense_empty.world'),
        CONFIDENCE_MAP_CONFIG_FILE=str(ROOT/'src/care_confidence_map/config/confidence_map_phase_e_ray.yaml'),
        GAZEBO_GUI='false', USE_RVIZ='false', TOF_FUSION_ENABLED='true',
        EXECUTION_GCDF_AUDIT_ENABLED='true', GCDF_BODY_INFLATION_M='0.015',
        GEOMETRY_BACKEND='primitive', GCDF_GEOMETRY_BACKEND='primitive',
        BODY_PRIOR_GEOMETRY_BACKEND='primitive', DIAGNOSTIC_GEOMETRY_BACKEND='primitive', VBC_GEOMETRY_BACKEND='primitive',
        BODY_SAMPLES_FILE=str(out/'MISSING_body_samples.yaml'),
        CONFIDENCE_QUERY_DEDICATED_QUEUE='true', VERIFICATION_TCP_NODELAY='true',
        VBC_CONFIDENCE_QUERY_PERSISTENT='true', EXECUTION_VBC_EVENT_DRIVEN_EVAL='true', CDF_SHADOW_VBC_EVENT_DRIVEN_EVAL='true',
        NCDF_ENV='viscdf', NCDF_DEVICE='cpu', PER_SENSOR_HYBRID_ENABLED='false',
        USE_LOCAL_SPARSE_SCP='true', FINAL_EXECUTABLE_GCDF_ENABLED='true', COMMITTED_CONTINUATION_ENABLED='false',
        EXECUTION_AUDIT_STREAM_ENABLED='true', EXECUTION_VBC_TRAJECTORY_TOPIC='/care_planner/execution/audit_trajectory',
        PROBE_SINGLE_FLIGHT_ENABLED='true', LOCAL_SCP_GPU_SOCKET=str(out/'gpu.sock'),
        LOCAL_SCP_SELECTOR_JSONL=str(out/'selector.jsonl'), LOCAL_SCP_PROXIMITY_MARGIN='0.025', VBC_SWEPT_VOLUME_MARGIN_M='0.0',
        REPAIR_PREFIX_VERIFY='1', REPAIR_PREFIX_S='0.25', PROBE_PREFIX_S='0.15',
        SMOOTH_HANDOFF_ENABLED='true', REPAIR_BRAKE_DT_S='0.05', REPAIR_HOLD_S='0.10', CYCLE_RECOVERY_ENABLED='true',
        TRAJECTORY_RISK_REFRESH_BODY_PRIOR_BEFORE_QUERY='false', FRONTIER_STEERING_ENABLED='false',
        VBC_GATED_FRONTIER_STEP_ENABLED='false', ADAPTIVE_REFINEMENT_ENABLED='false',
        REGION_SCHEDULE_MODE='blocker_aware_acquisition', CARE_WEIGHT='3000.0', SAFETY_MARGIN='0.30', PREDICTION_TIMEOUT='0.50',
        FORCE_ZERO_INITIAL_Q='true', EARLY_STOP_ON_GOAL='true', RECORDING_PROFILE='full',
        INITIAL_GATE_MAX_TRIES='100', INITIAL_GATE_ECHO_TIMEOUT='0.20')
    env.update(profile)
    # ROS/Gazebo run under sourced system ROS; only the model worker uses conda.
    assert subprocess.check_output(['which', 'python3'], env=env, text=True).strip() == '/usr/bin/python3', 'source system ROS first'
    frozen = [cases, Path(profile['CONFIG_FILE']), Path(profile['WORLD_FILE']), Path(profile['CONFIDENCE_MAP_CONFIG_FILE']),
        ROOT/'src/arm_description/urdf/Arm.urdf', ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
        ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt',
        ROOT/'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt',
        ROOT/'scripts/run_phase_c4_4_verified_regime_smoke.sh', Path(__file__).resolve(), ROOT/'scripts/phase_e_owned_processes.py',
        ROOT/'devel/lib/care_confidence_map/confidence_map_node', ROOT/'devel/lib/libtrajectory_risk_evaluator.so',
        ROOT/'devel/lib/care_confidence_map/gcdf_primitive_anchor_node',
        ROOT/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node',
        ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',
        ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp']
    def hashes(): return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    result = dict(status='FAIL', scope=__doc__, case=case, profile=profile, sha256=hashes(), failure_stage='startup', graph_snapshots=[])
    (out/'plan.json').write_text(json.dumps(result, indent=2)+'\n')
    processes, logs = [], []
    owned = {}
    def start(cmd, label):
        log = (out/(label+'.log')).open('x'); logs.append(log)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc); return proc
    try:
        result['failure_stage'] = 'viscdf ROS/CUDA dependency preflight'
        preflight = start(['/home/zhicheng/miniconda3/envs/viscdf/bin/python', '-c',
            "import sys,json; sys.path.append('/usr/lib/python3/dist-packages'); "
            "import torch,rospy,rosgraph.network,netifaces; "
            "assert torch.cuda.is_available(); "
            "print(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,"
            "netifaces=netifaces.__file__,local_addresses=rosgraph.network.get_local_addresses())))"], 'environment')
        assert preflight.wait(timeout=30) == 0, 'missing viscdf ROS/CUDA runtime dependency'
        result['failure_stage'] = 'GPU worker startup'
        worker = start(['/home/zhicheng/miniconda3/envs/viscdf/bin/python', '-u',
            str(ROOT/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py'), '--checkpoint',
            str(ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'),
            '--activation', 'gelu', '--device', 'cuda', '--socket', str(out/'gpu.sock'),
            '--max-pairs', '8000', '--warmup-pairs', '2048'], 'worker')
        deadline = time.monotonic()+45
        while not (out/'gpu.sock').exists():
            assert worker.poll() is None and time.monotonic() < deadline, 'GPU worker startup'
            time.sleep(.1)
        result['failure_stage'] = 'Gazebo/common runner'
        runner = start(['bash', 'scripts/run_phase_c4_4_verified_regime_smoke.sh'], 'runner')
        socket.setdefaulttimeout(1.)
        master = xmlrpc.client.ServerProxy(profile['ROS_MASTER_URI'])
        deadline = time.monotonic()+240
        while runner.poll() is None:
            remember_descendants({p.pid for p in processes if p.poll() is None}, owned)
            assert time.monotonic() < deadline, 'overall 240s wall watchdog'
            try:
                code, _, state = master.getSystemState('/gazebo_smoke_observer')
                if code == 1:
                    result['graph_snapshots'].append(dict(wall_time=time.time(), state=state))
                    if any(t == '/care_planner/candidate_vbc/summary' for t, _ in state[0]) and 'resolved_params' not in result:
                        code, _, params = master.getParam('/gazebo_smoke_observer', '/')
                        if code == 1:
                            result['resolved_params'] = params
                            (out/'resolved_params.json').write_text(json.dumps(params, indent=2)+'\n')
            except (OSError, xmlrpc.client.Error): pass
            time.sleep(2.)
        result['runner_exit_code'] = runner.returncode
        assert runner.returncode == 0, 'common runner failed; see runner.log'
        result['failure_stage'] = 'offline evidence'
        evaluator = start(['/usr/bin/python3', 'scripts/evaluate_phase_d_run.py', '--run-dir', str(out/'run'),
            '--case-id', args.case_id, '--repo', str(ROOT), '--cases-json', str(cases),
            '--require-benchmark-window', '--output-json', str(out/'case_evaluation.json')], 'evaluation')
        assert evaluator.wait(timeout=30) == 0, 'offline evaluator failed'
        assert result['sha256'] == hashes(), 'frozen inputs changed'
        result['status'] = 'RUN_COMPLETED_REVIEW_REQUIRED'; result['failure_stage'] = None
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        remember_descendants({p.pid for p in processes if p.poll() is None}, owned)
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
        for log in logs: log.close()
        result['owned_process_cleanup'] = reap_owned(owned)
        if result['owned_process_cleanup']['remaining']:
            result['status'] = 'FAIL'; result['failure_stage'] = 'owned process cleanup'
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['graph_snapshots','resolved_params','sha256','profile','case']}, indent=2))
    return 0 if result['status'] == 'RUN_COMPLETED_REVIEW_REQUIRED' else 1


if __name__ == '__main__': raise SystemExit(main())
