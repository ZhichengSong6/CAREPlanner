#!/usr/bin/env python3
"""Audit retained neural queries and freeze a future model A/B, without ROS."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
LOW = [-.8, -.8, 0.]
HIGH = [.8, .8, 1.]
# Stored selector points are float32, including nominal map faces +/-0.8.
# This tolerance is for historical classification only, not a runtime guard.
TOL = 1e-7
FIELD = re.compile(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\[[^\]]*\]|\S+)')
STUDIES = ['phase_e_current_30x3_20261022', 'phase_e_rebase_task_ab_20261022']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def seal_hashes(value):
    entries = value.get('artifacts', value)
    return {name: item['sha256'] if isinstance(item, dict) else item
            for name, item in entries.items()}


def qp_payload(value):
    payload = value.get('subproblem', value)
    if not all(k in payload for k in ('query_stamp_ns', 'cdf_rows', 'G')):
        raise ValueError('Unrecognized saved QP format')
    return payload


def in_domain(point):
    if len(point) != 3 or not all(math.isfinite(x) for x in point):
        raise ValueError('Invalid/nonfinite point')
    return all(a-TOL <= x <= b+TOL for x, a, b in zip(point, LOW, HIGH))


def parse_line(line):
    if line.strip() == '%time,field.data':
        return None, {'recorder_header': True}
    if line.strip() in ('WARNING: no messages received and simulated time is active.',
                        'Is /clock being published?'):
        return None, {'recorder_diagnostic': line.strip()}
    stamp, _, payload = line.rstrip('\n').partition(',')
    if not stamp.isdigit():
        raise ValueError('Malformed timestamp in string recorder')
    fields = dict(FIELD.findall(payload))
    for key, value in list(fields.items()):
        if value.startswith('['):
            fields[key] = json.loads(value)
    return int(stamp), fields


class Points:
    def __init__(self):
        self.n = self.out = self.near = 0
        self.low = [math.inf]*3
        self.high = [-math.inf]*3
        self.unique = set()
        self.examples = []
        self.face_counts = Counter()
        self.min_face_distance = math.inf

    def add(self, point, evidence):
        inside = in_domain(point)
        self.n += 1
        self.out += not inside
        # Round only the descriptive unique-point count, never classification.
        self.unique.add(tuple(round(x, 7) for x in point))
        face_distance = min(min(x-a, b-x) for x, a, b in zip(point, LOW, HIGH))
        self.min_face_distance = min(self.min_face_distance, face_distance)
        self.near += inside and face_distance <= .025+TOL
        for j, x in enumerate(point):
            self.low[j] = min(self.low[j], x)
            self.high[j] = max(self.high[j], x)
            if x < LOW[j]-TOL:
                self.face_counts['xyz'[j]+'_lower'] += 1
            if x > HIGH[j]+TOL:
                self.face_counts['xyz'[j]+'_upper'] += 1
        if not inside and len(self.examples) < 20:
            self.examples.append(dict(point_m=point, evidence=evidence))

    def report(self):
        return dict(occurrences=self.n, unique_xyz_rounded_1e7=len(self.unique),
                    outside=self.out, inside_within_25mm_of_face=self.near,
                    bounds_min_m=self.low if self.n else None,
                    bounds_max_m=self.high if self.n else None,
                    minimum_signed_face_distance_m=self.min_face_distance if self.n else None,
                    outside_faces=dict(self.face_counts), outside_examples=self.examples)


def choose_controls(cases, summary):
    eligible = [c for c in cases if c['case_id'][-3:] not in ('008', '029')
                and all(summary['cases'][c['case_id'][-3:]][key] == 3
                        for key in ('completed', 'valid_windows', 'task_success', 'strict_hold', 'overall_safe'))
                and summary['cases'][c['case_id'][-3:]]['execution_unsafe'] == 0]
    pairs = sorted(itertools.combinations(eligible, 2), key=lambda p:
                   (-math.dist(p[0]['goal_position'], p[1]['goal_position']),
                    p[0]['case_id'], p[1]['case_id']))
    if not pairs:
        raise ValueError('No eligible normal controls')
    return eligible, pairs[0]


def make_schedule(cases):
    runs = []
    for repeat in range(1, 4):
        for index, case in enumerate(cases):
            for model in (['old', 'new'] if (repeat-1+index) % 2 == 0 else ['new', 'old']):
                runs.append(dict(order=len(runs)+1, repeat=repeat, case=case, model=model,
                                 tag=f'{model}_r{repeat}_{case}'))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    for p in (out, *out.parents):
        if (p/'artifact_sha256.json').exists():
            raise ValueError('Output is under a sealed study')
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    seals = {}

    def track(path):
        path = Path(path).resolve()
        key = str(path.relative_to(ROOT))
        if key not in inputs:
            digest = sha(path)
            inputs[key] = digest
            for parent in path.parents:
                seal = parent/'artifact_sha256.json'
                if seal.exists():
                    if parent not in seals:
                        seals[parent] = seal_hashes(json.loads(seal.read_text()))
                    relative = str(path.relative_to(parent))
                    if seals[parent].get(relative) != digest:
                        raise ValueError('Source absent from or differs from its seal: '+key)
                    break
        return path

    def read(path):
        return json.loads(track(path).read_text())

    # Record initial root runtime inputs, including existing ELFs, for closeout.
    before_sources = {str(p.relative_to(ROOT)): sha(p) for folder in ('src', 'scripts')
                      for p in (ROOT/folder).rglob('*') if p.is_file() and not p.is_symlink()}
    elf_inputs = read(ROOT/'outputs/phase_e_rebase_task_ab_20261022/root_before.json')
    before_elf = {name: sha(ROOT/name) for name in elf_inputs}
    save(out/'root_before.json', dict(sources=before_sources, elf=before_elf))
    for name in ('NEXT_SESSION.md', 'outputs/README.md'):
        shutil.copy2(ROOT/name, out/(Path(name).stem+'_before.md'))

    all_runs, totals = [], {}
    global_sets = {key: Points() for key in ('batch_minimum', 'evaluated_witness',
                                             'requested_witness', 'saved_qp_cdf_row')}
    status_counts = Counter()
    exact_payload_files = []
    for study_name in STUDIES:
        study = ROOT/'outputs'/study_name
        results = read(study/'results.json')
        for run in results:
            assert run['valid_window']
            directory = study/'runtime'/run['run_directory']
            window = run['benchmark_window']
            start, end = window['start_ros_ns'], window['end_ros_ns']
            groups = {key: Points() for key in global_sets}
            detail = dict(study=study_name, tag=run['tag'], case=run['case'],
                          start_ros_ns=start, end_ros_ns=end,
                          query_batches=0, retained_inference_pairs=0, qp_snapshots=0,
                          witness_status_counts={}, full_batch_recorders=[], recorder_diagnostic_lines=[],
                          witness_recorder_header_seen=False)

            def add(key, point, path, line):
                evidence = dict(source=str(path.relative_to(ROOT)), line=line, tag=run['tag'])
                groups[key].add(point, evidence)
                global_sets[key].add(point, evidence)

            for path in sorted(directory.parent.glob('*selector*.jsonl')):
                for line_number, line in enumerate(track(path).open(), 1):
                    row = json.loads(line)
                    stamp = round(row['anchor_cloud_stamp']*1e9)
                    if not start <= stamp <= end:
                        continue
                    assert row['retained_pair_count'] > 0
                    detail['query_batches'] += 1
                    detail['retained_inference_pairs'] += row['retained_pair_count']
                    add('batch_minimum', row['global_min_pair']['point'], path, line_number)
            path = directory/'local_witness_selector_diagnostics.csv'
            statuses = Counter()
            with track(path).open() as stream:
                for line_number, line in enumerate(stream, 1):
                    receipt, row = parse_line(line)
                    if receipt is None:
                        if row.get('recorder_header'):
                            assert not detail['witness_recorder_header_seen'], 'Duplicate recorder header'
                            detail['witness_recorder_header_seen'] = True
                        else:
                            detail['recorder_diagnostic_lines'].append(dict(line=line_number, **row))
                        continue
                    assert detail['witness_recorder_header_seen'], 'Data without CSV schema'
                    query = int(row.get('query_stamp_ns', receipt))
                    if not start <= query <= end:
                        continue
                    status = row['status']
                    statuses[status] += 1
                    status_counts[status] += 1
                    add('requested_witness', row['point'], path, line_number)
                    if status in ('evaluated_unknown', 'evaluated_occupied'):
                        if not all(key in row for key in ('distance', 'q_linearization', 'gradient')):
                            raise ValueError('Evaluated witness lacks returned model evidence')
                        assert len(row['q_linearization']) == len(row['gradient']) == 7
                        assert all(math.isfinite(float(x)) for x in
                                   [row['distance'], *row['q_linearization'], *row['gradient']])
                        add('evaluated_witness', row['point'], path, line_number)
            detail['witness_status_counts'] = dict(statuses)
            for path in sorted(directory.parent.glob('qp*.json')):
                sub = qp_payload(read(path))
                if not start <= sub['query_stamp_ns'] <= end:
                    continue
                detail['qp_snapshots'] += 1
                for index, row in enumerate(sub.get('cdf_rows', [])):
                    add('saved_qp_cdf_row', row['point'], path, index)
            # Do not infer full-pair coverage from selected rows or summary counts.
            for path in sorted(directory.glob('*gcdf_constraint_batch.csv')):
                track(path)
                exact_payload_files.append(str(path.relative_to(ROOT)))
                detail['full_batch_recorders'].append(str(path.relative_to(ROOT)))
            detail['groups'] = {k: v.report() for k, v in groups.items()}
            all_runs.append(detail)
        print('AUDITED', study_name, len(results), flush=True)
    assert len(all_runs) == 102
    for name in STUDIES:
        rows = [r for r in all_runs if r['study'] == name]
        totals[name] = dict(windows=len(rows), query_batches=sum(r['query_batches'] for r in rows),
                           inference_pairs=sum(r['retained_inference_pairs'] for r in rows),
                           saved_qp_snapshots=sum(r['qp_snapshots'] for r in rows))
    save(out/'domain_by_run.json', all_runs)

    workspace = read(ROOT/'outputs/phase_e_safety_workspace_audit_20261022/workspace_summary.json')
    rt = ROOT/'outputs/phase_e_state_timestamp_ab_20261022/runtime'
    common_path = track(ROOT/'outputs/phase_e_case029_task_rejoin_20261022/common.py')
    spec = importlib.util.spec_from_file_location('domain_audit_frozen_geometry', common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    geometry = common.Geometry()
    for path in (rt/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
                 rt/'src/care_visibility_cdf/scripts/validate_visibility_oracle.py',
                 rt/'src/care_visibility_cdf/scripts/check_visibility_self_occlusion.py'):
        track(path)
    import numpy as np
    # Exactly the map/selector floor formula, including floating-point endpoints.
    lo, hi, resolution = [-.95, -.95, 0.], [.95, .95, 1.15], .05
    shape = [math.floor((b-a)/resolution)+1 for a, b in zip(lo, hi)]
    grid = np.array(list(itertools.product(*[[a+i*resolution for i in range(n)] for a, n in zip(lo, shape)])))
    outside = np.array([not in_domain(p) for p in grid])
    distance, nearest = geometry.minimum([0.]*7, grid[outside])
    # Baseline body inflation=0.0; local proximity=.025, final proximity=.075.
    proximity = {str(band): int(np.sum(distance <= band)) for band in (.025, .075)}
    nearest_order = np.argsort(distance, kind='stable')[:5]
    probe = dict(q_rad=[0.]*7, map_grid_shape=shape, map_grid_points=len(grid),
                 map_grid_outside_new_domain=int(outside.sum()),
                 body_inflation_m=0., eligible_outside_geometric_candidates=proximity,
                 examples=[dict(point_m=grid[outside][i].tolist(), primitive_distance_m=float(distance[i]),
                                primitive=geometry.primitives[int(nearest[i])]['name']) for i in nearest_order],
                 scope='Geometric candidates before map UNKNOWN/OCCUPIED filtering and pair caps; not actual historical model queries.')
    assert len(grid) == 33212
    save(out/'geometric_domain_probe.json', probe)

    fault = read(ROOT/'outputs/phase_e_cdf_qualification_20261023/bundle/queries.json')
    fault_groups = {}
    for group in sorted({q['group'] for q in fault}):
        points = Points()
        for q in fault:
            if q['group'] == group:
                points.add(q['point_m'], q['id'])
        fault_groups[group] = points.report()
    assert fault_groups['retained_fault']['outside'] == 0
    report = dict(status='RETAINED_QUERY_SUBSETS_IN_DOMAIN_FULL_QUERY_COVERAGE_UNPROVEN',
                  target_min_m=LOW, target_max_m=HIGH, historical_classification_tolerance_m=TOL,
                  totals=totals, groups={k: v.report() for k, v in global_sets.items()},
                  witness_status_counts=dict(status_counts), exact_batch_payload_files=exact_payload_files,
                  full_query_coverage_proven=False, frozen_qualification_groups=fault_groups,
                  measured_workspace=workspace['measured'], geometric_probe=probe,
                  limitations=['Only minimum pair, explicit witness responses and retained selected QP rows expose coordinates.',
                               'These sources overlap and have selection bias; their occurrence counts are not independent query counts.',
                               'Unrecorded pairs cannot be classified; a zero subset-outside count is not a full-domain proof.',
                               'Robot-body extent and point-query domain describe different quantities.',
                               'Offline domain tolerance is not a new production admission rule.'])
    save(out/'domain_summary.json', report)

    full = ROOT/'outputs/phase_e_current_30x3_20261022'
    cases_path = full/'runtime/outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json'
    cases = read(cases_path)['cases']
    summary = read(full/'summary.json')
    eligible, chosen = choose_controls(cases, summary)
    controls = [c['case_id'][-3:] for c in chosen]
    save(out/'control_selection.json', dict(rule='Exclude008/029; require valid/task/strict/safe3of3 and zero execution VBC violations; choose maximum Euclidean goal separation, lexical tie-break.',
                                          eligible=[c['case_id'][-3:] for c in eligible],
                                          selected=controls, goal_separation_m=math.dist(*[c['goal_position'] for c in chosen]),
                                          cases=[dict(case=c['case_id'][-3:], goal_position=c['goal_position'],
                                                      historical=summary['cases'][c['case_id'][-3:]]) for c in chosen],
                                          scope='Development controls selected before any new-model outputs; not independent generalization cases.'))
    # Use coherent timestamp handling, with unpromoted recovery experiments OFF.
    runtime = ROOT/'outputs/phase_e_rebase_task_ab_20261022/runtime'
    seal = seal_hashes(json.loads((runtime.parent/'artifact_sha256.json').read_text()))
    frozen = {name[len('runtime/'):]: digest for name, digest in seal.items()
              if name.startswith(('runtime/src/', 'runtime/scripts/', 'runtime/devel/'))}
    for name, digest in frozen.items():
        assert sha(track(runtime/name)) == digest
    save(out/'runtime_lock.json', dict(source=str(runtime.relative_to(ROOT)), files=frozen,
                                     purpose='Future isolated copies must match this common runtime; paths may be relocated with an explicit relocation record.'))
    track(runtime/'scripts/run_phase_e_case001_r1_mainline.sh')
    track(runtime/'scripts/run_phase_c4_4_verified_regime_smoke.sh')
    track(runtime/'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch')
    track(runtime/'src/care_collision_cdf/src/cpp_forbidden_voxel_gpu_shadow_node.cpp')
    track(runtime/'src/care_confidence_map/src/confidence_map_node.cpp')
    task_cases = ['008', '029', *controls]
    schedule = make_schedule(task_cases)
    plan = dict(status='PREREGISTERED_PENDING_MODEL_AND_COMPLETE_QUERY_DOMAIN_EVIDENCE',
                fault_cases=['008', '029'], control_cases=controls, runs=schedule,
                repeats_per_case_per_model=3, task_windows=24,
                order='Adjacent model pairs; six old-first and six new-first pairs; all attempts retained.',
                factor='Only collision checkpoint and its true declared activation; identical common software and all other settings.',
                common_runtime_lock='runtime_lock.json',
                common_settings=dict(summary_state_clock_coherent=True,
                                     fixed_start_recovery_enabled=False, fixed_start_rebase_reaudit_enabled=False,
                                     candidate_vbc_margin_m=.020, execution_vbc_margin_m=.010,
                                     tracking_guard_enforced=True, tracking_limit_m=.020,
                                     state_freshness_limit_ms=50, task_wall_budget_s=60, startup_budget_s=20,
                                     ranking=False, pruning=False, body_inflation_m=0., local_query_proximity_m=.025,
                                     failed_task_retries=0, failed_window_replacement=False),
                domain_evidence_requirement='All actual neural point inputs, all active query channels, must be counted and their bounds/outside counts retained. Historical partial samples cannot satisfy this.',
                preparation_before_tasks=['Review true training artifacts and any geometric mismatch.',
                                          'Run frozen1878-query/28-QP qualification against original old baseline.',
                                          'Validate identical passive full-input domain recording in both arms and prove recorder readiness before original task deadline.',
                                          'If a real out-of-domain input appears, retain it and resolve coverage/adaptation before claiming in-domain model A/B. Do not silently drop, clamp, or extrapolate it.',
                                          'Resolve hardcoded old checkpoint hashes and shell-forced activation in a new isolated harness; bind actual artifact hashes and attest the active loaded model. Keep both arms on the same harness.'],
                metrics=['task/strict success per case and model', 'observation completion and remaining NORMAL time',
                         'CDF/QP conflicts and terminal holds', 'actual commits/execution and tracking stops',
                         'VBC margin violations; physical contacts only if independently recorded',
                         'all neural input domain counts/bounds and inference/loop timing'],
                expansion_gate='Require mechanism improvement, no normal-control task/strict regression and no new execution-audit violations; review all failures before candidate30x3. Three trials per condition do not establish statistical superiority.',
                no_automatic_launch=True, new_checkpoint=None)
    save(out/'model_ab_plan.json', plan)
    manifest = read(ROOT/'outputs/phase_e_cdf_qualification_20261023/bundle/training_manifest_template.json')
    manifest['activation'] = None
    manifest['instruction'] = 'Fill actual training evidence; every unknown remains null. Current runtime URDF is only a reference, not evidence of what was trained.'
    save(out/'training_manifest_pending.json', manifest)
    save(out/'training_evidence_status.json', dict(status='USER_CONFIRMS_NEW_MODEL_NOT_READY',
         checkpoint=None, activation=None, actual_training_urdf=None, actual_label_generator=None,
         actual_dataset_manifest=None, actual_training_config=None, heldout_evaluation=None,
         target=dict(min_m=LOW, max_m=HIGH, spacing_m=.025, shape=[65, 65, 41], spatial_points=173225),
         available_reference_only=dict(runtime_urdf='src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
                                       runtime_urdf_sha256=sha(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'))))
    after_sources = {name: sha(ROOT/name) for name in before_sources}
    after_elf = {name: sha(ROOT/name) for name in before_elf}
    assert after_sources == before_sources and after_elf == before_elf
    save(out/'validation.json', dict(status='PASS_OFFLINE_PREPARATION', windows=102,
         source_seals_checked_on_consumption=True, full_query_coverage_proven=False,
         root_source_and_data_files_unchanged=len(before_sources), root_elf_unchanged=len(before_elf),
         new_model_ready=False, new_model_tested=False, ros_gazebo_started=False,
         model_ab_predeclared_windows=24, control_cases=controls))
    save(out/'source_inputs_sha256.json', inputs)
    shutil.copy2(Path(__file__), out/Path(__file__).name)
    print('PREPARATION_COMPLETE', json.dumps(dict(controls=controls, totals=totals)), flush=True)


if __name__ == '__main__':
    main()
