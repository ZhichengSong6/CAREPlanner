#!/usr/bin/env python3
"""Three frozen-code repeats of QP 004/015/012 and scalar/H9 007/026.

Only orchestrates the existing simulation runners and read-only analyzers.
Never edits runtime, increases budgets, retries failed tasks, or runs 30 cases.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from summarize_phase_e_qp_events import collect

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/home/zhicheng/miniconda3/envs/viscdf/bin/python'
CHECKPOINTS = {
    'src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt':
        '979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199',
    'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt':
        'fea15cb71b278b9d003337d200f3796ccbbe8a59106287a8ea85abb2c89cb0df',
    'src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt':
        '43f962729adcd17aa114edb9fc410facbbb97ebe7343f0ad3309fe50d273acdb',
    'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt':
        '41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70',
}


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)


def capture(command):
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def fingerprint():
    suffixes = {'.cpp', '.hpp', '.h', '.py', '.yaml', '.yml', '.launch',
                '.sh', '.urdf', '.xacro', '.world', '.xml', '.json', '.msg', '.srv'}
    paths = {p for root in ('src', 'scripts') for p in (REPO / root).rglob('*')
             if p.is_file() and (p.suffix in suffixes or p.name == 'CMakeLists.txt')
             and not any(x in p.parts for x in ('third_party', '__pycache__'))}
    paths.update(REPO / p for p in CHECKPOINTS)
    paths.add(REPO / 'outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json')
    # Include the actual compiled executables/shared libraries, not just HEAD.
    paths.update(p for p in (REPO / 'devel/lib').rglob('*') if p.is_file()
                 and p.suffix != '.pyc' and '__pycache__' not in p.parts)
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def assert_frozen(expected):
    actual = fingerprint()
    changed = [p for p in expected.keys() | actual.keys()
               if expected.get(p) != actual.get(p)]
    if changed:
        raise RuntimeError('Frozen inputs changed; stop without restoring files: ' + repr(changed))


def assert_no_ros():
    check = subprocess.run(['ps', '-C', 'roscore,rosmaster,roslaunch,gzserver,gzclient',
                            '-o', 'pid=,comm='], text=True, capture_output=True)
    if check.returncode not in (0, 1) or check.stdout.strip():
        raise RuntimeError('Refuse runner cleanup while ROS/Gazebo is already running: '
                           + check.stdout + check.stderr)


def summarize(rows):
    groups = []
    for model, case_id in sorted({(r['model'], r['case_id']) for r in rows}):
        selected = [r for r in rows if (r['model'], r['case_id']) == (model, case_id)]
        groups.append(dict(model=model, case_id=case_id, repeats=len(selected),
            task_pass=sum(r['task_success'] for r in selected),
            safety_pass=sum(r['overall_safe'] for r in selected),
            startup_pass=sum(r['startup_order_pass'] for r in selected),
            final_position_errors_m=[r['final_position_error_m'] for r in selected],
            final_orientation_errors_rad=[r['final_orientation_error_rad'] for r in selected],
            normal_reseeds=[r['normal_box_reseed_count'] for r in selected],
            observation_stages=dict(sum((Counter(r['observation']['stage_counts'])
                                        for r in selected), Counter())),
            unmatched_verification=sum(r['observation']['unmatched_verification_count'] for r in selected),
            generation_errors=sum(len(r['observation']['failed_generations']) for r in selected),
            unfinished_generations=sum(len(r['observation']['unfinished_generations']) for r in selected)))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp', default=datetime.now().strftime('%Y%m%d-%H%M%S') + '-repeat3')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.stamp):
        parser.error('stamp must contain only letters, digits, underscore, hyphen')
    os.chdir(REPO)
    out = REPO / 'outputs/phase_e_repeatability' / args.stamp
    out.mkdir(parents=True, exist_ok=False)
    assert_no_ros()
    frozen = fingerprint()
    for path, expected in CHECKPOINTS.items():
        if frozen[path] != expected:
            raise RuntimeError('Checkpoint mismatch: ' + path)
    save(out / 'frozen_sha256.json', frozen)
    cuda = capture([PYTHON, '-c', 'import json,torch; '
        'assert torch.cuda.is_available(); '
        'q=torch.ones(7,device="cuda",requires_grad=True); '
        'g=torch.autograd.grad((q*q).sum(),q)[0]; torch.cuda.synchronize(); '
        'print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,'
        'gpu=torch.cuda.get_device_name(0),raw_q_gradient=g.tolist())))'])
    save(out / 'preflight.json', dict(git_head=capture(['git', 'rev-parse', 'HEAD']),
        dirty_status=capture(['git', 'status', '--short']), cuda=json.loads(cuda),
        gpu=capture(['nvidia-smi']), repeats=3, run_count=21,
        order='Each repeat: QP 004/015/012, scalar 007/026, H9 007/026',
        wall_watchdog_s=45, protocol='pre_goal_arm_v1',
        note='Serial simulations; no runtime edits; no failed-task replacement runs.'))
    print('OUTPUT', out, '\nCUDA', cuda, flush=True)
    rows = []
    batch_root = REPO / 'outputs/phase_e_empty_world_qualification'
    for repeat in range(1, 4):
        stem = args.stamp + '-r' + str(repeat)
        qp_stamp, ab_stamp = stem + '-qp', stem + '-ab'
        for kind, script, overrides in (
            ('qp', 'run_phase_e_qp_events_regression.sh',
             dict(BATCH_STAMP=qp_stamp, BATCH_ID='phase_e_qp_events_' + qp_stamp,
                  DIAGNOSTIC_CASES='004,015,012', PER_SENSOR_HYBRID_ENABLED='true')),
            ('ab', 'run_phase_e_observation_ab.sh', dict(AB_STAMP=ab_stamp))):
            assert_frozen(frozen)
            assert_no_ros()
            save(out / (stem + '-' + kind + '-gpu.json'), dict(
                wall_unix_s=time.time(), gpu=capture(['nvidia-smi'])))
            env = dict(os.environ, **overrides)
            # Retain per-case archives too; do not delete any output in this wrapper.
            env['KEEP_CASE_ZIPS'] = '1'
            print('START', repeat, kind, flush=True)
            with (out / (stem + '-' + kind + '.log')).open('x') as stream:
                result = subprocess.run(['bash', 'scripts/' + script], cwd=REPO,
                    env=env, stdout=stream, stderr=subprocess.STDOUT)
            save(out / (stem + '-' + kind + '-exit.json'), dict(return_code=result.returncode))
            assert_frozen(frozen)
            if result.returncode:
                raise RuntimeError('Runner failed; preserve outputs, no automatic retry: ' + kind)
            print('END', repeat, kind, flush=True)
        round_rows = collect(REPO, batch_root / ('phase_e_qp_events_' + qp_stamp), qp_stamp, 'qp_h9')
        for model in ('scalar', 'h9'):
            stamp = ab_stamp + '-' + model
            round_rows += collect(REPO, batch_root / ('phase_e_' + stamp), stamp, model)
        for row in round_rows:
            row['repeat'] = repeat
        save(out / ('round_' + str(repeat) + '.json'), dict(cases=round_rows))
        rows += round_rows
        print('ROUND', repeat, json.dumps(summarize(round_rows)), flush=True)
        if not all(r['startup_order_pass'] and r['overall_safe'] for r in round_rows):
            raise RuntimeError('Startup or safety failed; stop before next repeat, preserve evidence')
    assert_frozen(frozen)
    assert_no_ros()
    report = dict(schema_version=1, frozen_inputs_equal=True, planned_runs=21,
        completed_runs=len(rows), groups=summarize(rows), cases=rows,
        caveat='Three repeats are descriptive, not a precise success-rate estimate; '
               'observation-stage counts are revisions, not failed-case counts.')
    save(out / 'summary.json', report)
    print('COMPLETE', json.dumps(report['groups']), flush=True)


if __name__ == '__main__':
    main()
