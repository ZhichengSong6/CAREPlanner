#!/usr/bin/env python3
"""Independently audit paired Gazebo traces; observed envelopes are not bounds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from phase_e_response_contract import audit_steps, conditional_error_budget, persistent_rest_index
from phase_e_stopping_contract import qualify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(source):
    run = json.loads((source / 'result.json').read_text())
    assert run['status'] == 'PASS_DIAGNOSTIC_NOT_QUALIFICATION'
    assert run['inputs_unchanged'] and run['protected_identity_equal']
    assert not run['cleanup']['remaining'] and all(run['private_ports_released'].values())
    assert not run['runtime_admission'] and not run['production_issuer_enabled'] and run['new_formal_windows'] == 0
    rows = [json.loads(line) for line in (source / 'physics_steps.jsonl').open()]
    a = audit_steps(rows, allow_discontinuities=(run['scenario_set'] == 'lifecycle'))
    t, h, values = a['t'], a['h'], a['values']
    controller = [r for r in map(json.loads, (source / 'controller_trace.jsonl').open()) if r['kind'] == 'state']
    # Never match a pre-reset receipt to a post-reset state with the same stamp.
    if run['scenario_set'] == 'lifecycle':
        stop = int(a['discontinuity_indices'][0])+1 if len(a['discontinuity_indices']) else len(t)
        t_for_matching = t[:stop]
        first_regression = next((i for i in range(1, len(controller)) if controller[i]['t_ns'] < controller[i-1]['t_ns']), len(controller))
        controller = controller[:first_regression]
    else:
        t_for_matching = t
    stamp_index = {int(stamp): i for i, stamp in enumerate(t_for_matching)}
    matches = [(row, stamp_index[row['t_ns']]) for row in controller if row['t_ns'] in stamp_index]
    assert matches
    alignment = dict(
        motor=max(float(np.max(np.abs(values['motor'][i] - row['applied']))) for row, i in matches),
        pose=max(float(np.max(np.abs(values['q_before'][i] - row['q']))) for row, i in matches),
        velocity=max(float(np.max(np.abs(values['v_before'][i] - row['measured']))) for row, i in matches))
    assert max(alignment.values()) < 1e-9, alignment
    nominal = run['scenario_set'] != 'lifecycle'
    integral_alignment = None
    if nominal:
        prefix = np.vstack((np.zeros(7), np.cumsum(h[:, None] * values['motor'], axis=0)))
        anchor, ai = matches[0]
        integral_alignment = max(float(np.max(np.abs(
            np.asarray(row['integral']) - anchor['integral'] - (prefix[i] - prefix[ai])))) for row, i in matches)
        assert integral_alignment < 1e-8, integral_alignment
        assert all(row['integral_valid'] and not row['clock_fault'] and row['max_update_gap'] <= .001+1e-12 for row, _ in matches)
    trials = []
    for trial in run['trials']:
        ids = np.flatnonzero((t >= trial['baseline_ns']) & (t <= trial['end_ns']))
        stop_ids = np.flatnonzero((t >= trial['decision_ns']) & (t <= trial['end_ns']))
        assert len(ids) > 100 and len(stop_ids) > 100
        q0 = values['q_before'][ids[0]]
        command_integral = np.cumsum(h[ids, None] * values['motor'][ids], axis=0)
        error = values['q_after'][ids] - q0 - command_integral
        eps = np.max(np.abs(a['velocity_defect'][ids]), axis=0)
        eta = np.max(np.abs(a['increment_defect'][ids]), axis=0)
        conditional = conditional_error_budget(steps=len(ids), step_s=.001,
            velocity_residual_bound=eps, pose_increment_residual_bound=eta)
        observed = np.max(np.abs(error), axis=0)
        assert np.all(observed <= np.asarray(conditional['error_rad']) + 1e-10)
        # For stopping, q and applied have exactly the same measurement origin.
        sq0 = values['q_before'][stop_ids[0]]
        physical_travel = np.sum(np.abs(values['q_after'][stop_ids] - values['q_before'][stop_ids]), axis=0)
        physical_displacement = np.max(np.abs(values['q_after'][stop_ids] - sq0), axis=0)
        command_travel = np.sum(np.abs(values['motor'][stop_ids]) * h[stop_ids, None], axis=0)
        candidate_zero = np.flatnonzero((t[stop_ids] >= trial['withdrawal_ns']) &
                                       (np.max(np.abs(values['motor'][stop_ids]), axis=1) <= 1e-12))
        assert len(candidate_zero), 'no full zero command'
        zero = int(candidate_zero[0])
        assert np.max(np.abs(values['motor'][stop_ids[zero:]])) <= 1e-12
        rest = persistent_rest_index(values['velocity'][stop_ids], zero, hold_steps=101)
        assert rest is not None, 'no 100ms persistent measured rest'
        zidx, ridx = stop_ids[zero], stop_ids[rest]
        # End states follow an integration step h even though World::UpdateEnd
        # retains the begin stamp. Avoid reporting rest one physics step early.
        zero_ns = int(t[zidx])
        rest_ns = int(t[ridx] + round(h[ridx]*1e9))
        tail_travel = np.sum(np.abs(values['q_after'][stop_ids[zero:]] -
                                    values['q_before'][stop_ids[zero:]]), axis=0)
        post_rest_drift = np.max(np.abs(values['q_after'][stop_ids[rest:]] - values['q_after'][ridx]), axis=0)
        post_rest_velocity = float(np.max(np.abs(values['velocity'][stop_ids[rest:]])))
        qstate = trial['decision_state']
        assert qstate['t_ns'] == trial['decision_ns']
        rights = trial['outstanding_rights']
        assert rights and all(r['session'] == trial['controller_session'] and r['token'] == trial['token'] for r in rights)
        deadline = max(r['until_ns'] for r in rights)
        cap = np.max(np.asarray([r['target_abs_velocity'] for r in rights]), axis=0)
        budget_checks = []
        for age in (0., .01, .04, .049):
            now = qstate['t_ns']*1e-9 + age
            if now >= deadline*1e-9:
                continue
            result = qualify(now_s=now, measured_s=qstate['t_ns']*1e-9, applied=qstate['applied'],
                acceleration=qstate['acceleration_limits'], velocity_limits=qstate['velocity_limits'],
                next_decision_s=.01, transport_bound_s=0., update_gap_s=.001,
                lease_end_s=deadline*1e-9, lease_enforced=True, physical_error_rad=None,
                physical_settle_s=None, snapshot=None, max_braking_step_s=qstate['max_braking_step_s'],
                authorized_target_abs=cap.tolist())
            assert not result['qualified'] and not result['runtime_admission']
            for mode, b in result['budgets'].items():
                bound = np.asarray(b['command_travel_rad'])
                assert np.all(command_travel <= bound + 1e-8), (trial['label'], mode, command_travel, bound)
                assert zero_ns*1e-9 <= b['required_until_s'] + 1e-8
                budget_checks.append(dict(age_s=age, mode=mode, command_covered=True,
                    physical_displacement_excess_rad=np.maximum(0., physical_displacement-bound).tolist(),
                    scope='actual trace with observed 1ms cadence; future timing and response unqualified'))
        inactive = [j for j in range(7) if np.max(np.abs(values['motor'][ids, j])) <= 1e-12]
        drift = {str(j): float(np.max(np.abs(values['q_after'][ids, j] - q0[j]))) for j in inactive}
        def optional_max(key):
            values = [rows[i].get(key) for i in ids]
            # The four entries are bilateral, contact normal, friction, total.
            # These are global RMS summaries, never per-motor error bounds.
            scalars = [v[3] if isinstance(v, list) and len(v) == 4 else v for v in values]
            finite = [v for v in scalars if isinstance(v, (int, float)) and np.isfinite(v)]
            return max(finite) if finite else None
        trials.append(dict(label=trial['label'], profile=trial['profile'], stop=trial['stop'],
            physics_steps=len(ids), max_response_error_rad=observed.tolist(),
            max_velocity_defect_rad_s=eps.tolist(), max_pose_increment_defect_rad=eta.tolist(),
            observational_triangle_enclosure_rad=conditional['error_rad'], future_bound=False,
            inactive_axis_drift_rad=drift, command_zero_ns=zero_ns,
            measured_persistent_rest_ns=rest_ns, extra_settling_s=(rest_ns-zero_ns)*1e-9,
            residual_tail_travel_rad=tail_travel.tolist(), actual_absolute_command_travel_rad=command_travel.tolist(),
            post_rest_drift_rad=post_rest_drift.tolist(), post_rest_max_velocity=post_rest_velocity,
            actual_absolute_physical_travel_rad=physical_travel.tolist(),
            max_solver_constraint_residual=optional_max('constraint_residual'),
            max_solver_delta_lambda_rms=optional_max('rms_error'), max_solver_contacts=optional_max('num_contacts'),
            budget_checks=budget_checks))
    runtime = json.loads((source / 'physics_runtime.json').read_text())
    providers = runtime['global_symbol_providers']
    assert all('libgazebo_ode.so' in p for p in providers.values()), providers
    return dict(status='PASS_INDEPENDENT_OBSERVATIONAL_AUDIT', run=str(source),
        scenario_set=run['scenario_set'], runtime_admission=False, production_issuer_enabled=False,
        physical_bound_certified=False, new_formal_windows=0, physics_steps=len(rows),
        controller_receipt_matches=len(matches), maximum_alignment_error=alignment,
        maximum_integral_alignment_error=integral_alignment, runtime=runtime, trials=trials,
        nonnominal_transitions=len(a['nonnominal_indices']), discontinuities=len(a['discontinuity_indices']),
        lifecycle=run['lifecycle'],
        input_sha256={str(source/name): sha(source/name) for name in (
            'result.json', 'physics_steps.jsonl', 'controller_trace.jsonl', 'physics_runtime.json',
            'expanded_model.sdf', 'expanded_physics.sdf', 'input_sha256.json', 'input_sources.tar.gz', 'loaded_objects.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', action='append', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runs = [audit(p.resolve()) for p in args.input_dir]
    for i, run in enumerate(runs):
        (out / f'run_{i+1}.json').write_text(json.dumps(run, indent=2) + '\n')
    trials = [dict(run=Path(run['run']).name, **trial) for run in runs for trial in run['trials']]
    matrix = [r for r in runs if r['scenario_set'] == 'matrix']
    if matrix:
        assert all(len(r['trials']) == 32 for r in matrix)
        assert all([t['label'] for t in r['trials']] == [t['label'] for t in matrix[0]['trials']] for r in matrix)
    summary = dict(status='PASS_OBSERVATIONS_ONLY', runtime_admission=False,
        physical_bound_certified=False, production_issuer_enabled=False, new_formal_windows=0,
        runs=len(runs), matrix_runs=len(matrix), trials=len(trials),
        physics_steps=sum(r['physics_steps'] for r in runs),
        controller_receipt_matches=sum(r['controller_receipt_matches'] for r in runs),
        command_budget_checks=sum(len(t['budget_checks']) for t in trials),
        maximum_response_error_rad=np.max([t['max_response_error_rad'] for t in trials], axis=0).tolist() if trials else None,
        maximum_extra_settling_s=max((t['extra_settling_s'] for t in trials), default=None),
        maximum_inactive_axis_drift_rad=max((v for t in trials for v in t['inactive_axis_drift_rad'].values()), default=None),
        primary_runs=[r['run'] for r in runs],
        interpretation='Repeated fixed-fixture observations and counterexample search, not future bounds or formal case results',
        analysis_input_sha256={str(Path(__file__).resolve()): sha(__file__),
            str(Path(__file__).with_name('phase_e_response_contract.py')): sha(Path(__file__).with_name('phase_e_response_contract.py')),
            str(Path(__file__).with_name('phase_e_stopping_contract.py')): sha(Path(__file__).with_name('phase_e_stopping_contract.py'))})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'trial_summary.csv').open('x', newline='') as f:
        fields = ['run', 'label', 'profile', 'stop', 'physics_steps', 'extra_settling_s',
                  'max_solver_constraint_residual', 'max_solver_delta_lambda_rms', 'max_solver_contacts']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: trial[k] for k in fields} for trial in trials)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
