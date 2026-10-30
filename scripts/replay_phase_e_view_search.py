#!/usr/bin/env python3
"""Freeze and replay recorded replacement branches without ROS or execution."""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/phase_e_rebase_task_ab_20261022'
REL_VIS = Path('src/care_visibility_cdf/scripts')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    # Retain NaN diagnostics from the original projection histories verbatim.
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(study):
    study.mkdir(parents=True, exist_ok=False)
    runtime = PARENT / 'runtime'
    root_before = {}
    for folder in ('src', 'scripts'):
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.cpp', '.hpp', '.h', '.launch', '.yaml', '.urdf', '.json', '.sh'):
                root_before[str(p.relative_to(ROOT))] = sha(p)
    save(study / 'root_source_before.json', root_before)
    parent_inputs = {}
    measurements_path = PARENT / 'measurements.json'
    parent_inputs[str(measurements_path.relative_to(ROOT))] = sha(measurements_path)
    measurements = json.loads(measurements_path.read_text())
    rows, outside = [], []
    for tag, run in sorted(measurements.items()):
        path = runtime / run['metrics']['run_directory'] / 'projector_traces/candidate_replacement.jsonl'
        if not path.exists():
            continue
        parent_inputs[str(path.relative_to(ROOT))] = sha(path)
        window = run['rejoin_timing']
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            event = json.loads(line)
            if 'hybrid' not in event:
                continue
            row = dict(id=f"{tag}_request{event['request_id']}", tag=tag,
                       case=run['metrics']['case'], line=line_number,
                       source=str(path.relative_to(ROOT)), event=event)
            if window['start_ros_ns'] <= round(event['ros_s'] * 1e9) <= window['end_ros_ns']:
                rows.append(row)
            else:
                outside.append(row)
    assert len(rows) == 11 and sum(r['event']['event'] == 'no_geometric_candidate' for r in rows) == 6
    save(study / 'inputs.json', rows)
    save(study / 'outside_original_window.json', outside)
    baseline = study / 'baseline'
    baseline.mkdir()
    for p in (runtime / REL_VIS).glob('*.py'):
        shutil.copy2(p, baseline / p.name)
        parent_inputs[str(p.relative_to(ROOT))] = sha(p)
    for filename in ('Arm.urdf', 'Arm_with_self_filter_collision.urdf'):
        p = runtime / 'src/arm_description/urdf' / filename
        shutil.copy2(p, baseline / filename)
        parent_inputs[str(p.relative_to(ROOT))] = sha(p)
    limits = {}
    tree = ast.parse((baseline / 'vbc_deadline_waypoint_node.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('DEFAULT_Q_MIN', 'DEFAULT_Q_MAX'):
                    limits[target.id] = ast.literal_eval(node.value)
    save(study / 'joint_limits.json', limits)
    checkpoint = runtime / 'src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt'
    parent_inputs[str(checkpoint.relative_to(ROOT))] = sha(checkpoint)
    save(study / 'parent_inputs_sha256.json', parent_inputs)
    save(study / 'protocol.json', dict(
        status='FROZEN_BEFORE_CANDIDATE', original_windows=12,
        retained_inputs=11, original_failures=6, original_success_controls=5,
        excluded_outside_window=len(outside), repeats_per_input=3,
        device='cpu', torch_threads=1,
        factor='replacement geometry completion using unused projection steps',
        gates=['baseline candidate q and geometric decisions reproduce the original records',
               'preserve all five originally accepted controls',
               'improve at least one of six originally rejected inputs',
               'projection plus completion steps never exceed original projection_iters',
               'ordinary generation and default-off branch results unchanged',
               'one reservation still optimizes one sensor branch',
               'unchanged FOV thresholds, primitive LOS, joint limits and downstream execution gates'],
        qualification='Endpoint proposal experiment only; no path, map visibility, collision CDF, execution or task-success claim',
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint)))
    print('PREPARED', len(rows), 'in-window inputs;', len(outside), 'excluded outside window', flush=True)


def replay(study, arm):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for p, expected in json.loads((study / 'parent_inputs_sha256.json').read_text()).items():
        assert sha(ROOT / p) == expected, p
    source = study / arm / 'per_sensor_visibility_runtime.py'
    sys.path.insert(0, str(study / 'baseline'))
    spec = importlib.util.spec_from_file_location('recorded_view_runtime', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = json.loads((study / 'inputs.json').read_text())
    protocol = json.loads((study / 'protocol.json').read_text())
    limits = json.loads((study / 'joint_limits.json').read_text())
    cfg = inputs[0]['event']['hybrid']['config']
    signature = inspect.signature(module.PerSensorVisibilityRuntime)
    options = {k: v for k, v in cfg.items() if k in signature.parameters}
    options.update(checkpoint_path=str(ROOT / protocol['checkpoint']),
                   reference_urdf_path=str(study / 'baseline/Arm.urdf'),
                   self_filter_urdf_path=str(study / 'baseline/Arm_with_self_filter_collision.urdf'),
                   device=torch.device('cpu'), q_min=limits['DEFAULT_Q_MIN'], q_max=limits['DEFAULT_Q_MAX'])
    if arm == 'candidate':
        options['replacement_geometry_completion_enabled'] = True
    runtime = module.PerSensorVisibilityRuntime(**options)
    runtime.warmup()
    results = []
    for row in inputs:
        event = row['event']
        attempt, = event['hybrid']['attempts']
        threshold = attempt['geometry']['minimum_conservative_g_required']
        sid = attempt['sensor_id']
        points = torch.tensor(event['points'], dtype=torch.float32)
        seed = torch.tensor(event['hybrid']['branch_seed_q'], dtype=torch.float32).reshape(1, 7)
        repeats = []
        for repeat in range(1, protocol['repeats_per_input'] + 1):
            tic = time.perf_counter()
            branch = runtime._optimize_branch(points, seed, sid, geometry_guided=True, geometry_threshold=threshold)
            geometry = runtime._candidate_geometry(event['points'], branch['q_candidate'], sid,
                                                   minimum_conservative_g=threshold)
            elapsed_ms = (time.perf_counter() - tic) * 1000
            q_difference = float(np.max(np.abs(np.array(branch['q_candidate']) - attempt['q_candidate'])))
            if arm == 'baseline':
                assert q_difference < 1e-4, (row['id'], q_difference)
                assert geometry['accepted'] == attempt['geometry']['accepted'], row['id']
                assert abs(geometry['min_nominal_margin'] - attempt['geometry']['min_nominal_margin']) < 1e-5, row['id']
            repeats.append(dict(repeat=repeat, elapsed_ms=elapsed_ms, original_q_difference_inf=q_difference,
                                branch=branch, geometry=geometry))
        # A deterministic solver must not rely on favorable repeat selection.
        assert all(json.dumps(r['branch'], sort_keys=True) == json.dumps(repeats[0]['branch'], sort_keys=True)
                   and r['geometry'] == repeats[0]['geometry'] for r in repeats)
        results.append(dict(id=row['id'], tag=row['tag'], case=row['case'], sensor_id=sid,
                            original_accepted=attempt['geometry']['accepted'], repeats=repeats))
        print(arm, row['id'], 'accepted', repeats[0]['geometry']['accepted'],
              'margin', round(repeats[0]['geometry']['min_nominal_margin'], 6), flush=True)
    # Existing branch histories use NaN for unavailable gradient diagnostics.
    output = study / f'{arm}_replay.json'
    assert not output.exists(), output
    output.write_text(json.dumps(dict(arm=arm, source_sha256=sha(source), torch_version=torch.__version__,
                                     rows=results), indent=2, ensure_ascii=False) + '\n')
    print('REPLAY_COMPLETE', arm, len(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--arm', choices=('baseline', 'candidate'))
    args = parser.parse_args()
    study = args.study_dir.resolve()
    if args.prepare:
        assert args.arm is None
        prepare(study)
    else:
        assert args.arm is not None
        replay(study, args.arm)


if __name__ == '__main__':
    main()
