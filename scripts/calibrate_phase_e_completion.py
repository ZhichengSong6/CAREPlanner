#!/usr/bin/env python3
"""Calibrate a ranking proxy from preserved development runs, never safety bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np


def fit(rows):
    x = np.asarray([r['proxy_s'] for r in rows])
    y = np.asarray([r['actual_wall_s']-r['offset_s'] for r in rows])
    # Nonnegative intercept and scale >= 1. These are descriptive coefficients.
    candidates = []
    for scale in np.linspace(1., 15., 281):
        intercept = max(0., float(np.median(y-scale*x)))
        error = float(np.mean(np.abs(y-scale*x-intercept)))
        candidates.append((error, scale, intercept))
    _, scale, intercept = min(candidates)
    return float(scale), float(intercept)


def metrics(rows, scale, intercept):
    baseline = [abs(r['actual_wall_s']-r['proxy_s']-r['offset_s']) for r in rows]
    calibrated = [abs(r['actual_wall_s']-scale*r['proxy_s']-intercept-r['offset_s']) for r in rows]
    return dict(episodes=len(rows), cases=sorted({r['case'] for r in rows}),
                baseline_mae_s=float(np.mean(baseline)), calibrated_mae_s=float(np.mean(calibrated)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    root = args.repo
    runs = json.loads((root/'outputs/phase_e_case003_handoff_30x3_20261018/results.json').read_text())
    approach, returns, provenance, excluded = [], [], {}, []
    clock = re.compile(r'\[(\d+\.\d+),\s*(\d+\.\d+)\]')
    for run in runs:
        path = root/run['archive']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == run['archive_sha256'], run['tag']
        provenance[str(path.relative_to(root))] = digest
        with zipfile.ZipFile(path) as archive:
            log = archive.read('logs/waypoint_generator.log').decode(errors='replace')
            added, acquired = {}, {}
            for line in log.splitlines():
                stamp = clock.search(line)
                if not stamp:
                    continue
                wall, ros = map(float, stamp.groups())
                add = re.search(r'\[vbc_multi_deadline\] ADD obligation=(\d+)', line)
                if add:
                    added[int(add[1])] = dict(wall=wall, ros=ros)
                seen = re.search(r'ACTUAL VISIBILITY acquired obligations=([0-9:]+)', line)
                if seen:
                    for oid in re.findall(r'\d+', seen[1]):
                        acquired.setdefault(int(oid), dict(wall=wall, ros=ros))
            stop = json.loads(archive.read('run/goal_stop_status.json'))
            window = stop['benchmark_window']
            duration = stop.get('elapsed_wall_s', stop.get('wall_elapsed_s'))
            if duration is None:
                duration = window.get('end_monotonic_s', 0.)-window['watchdog_start_monotonic_s']
            end_wall = window['watchdog_start_wall_unix_s']+duration
            rate = ((window['end_ros_ns']-window['start_ros_ns'])/1e9 /
                    max(1e-9, window['end_monotonic_s']-window['start_monotonic_s']))
            completed = []
            for member in archive.namelist():
                if not ('/c46_obligation_' in member and member.endswith('.json')):
                    continue
                value = json.loads(archive.read(member))
                oid = value.get('c4_6_obligation_id')
                h = value.get('per_sensor_hybrid', {})
                candidates = h.get('task_rejoin', {}).get('candidates', [])
                cost = next((x for x in candidates if x['sensor_id'] == h.get('selected_sensor_id')), None)
                if (oid not in added or oid not in acquired or cost is None or
                        not value.get('per_sensor_hybrid_used') or
                        not np.allclose(value['q_vis'], h.get('selected_q_vis', []), atol=1e-6, rtol=0.)):
                    excluded.append(dict(run=run['tag'], member=member, reason='no_chosen_generation_cost_and_acquired_obligation'))
                    continue
                actual = acquired[oid]['wall']-added[oid]['wall']
                if actual < 0. or acquired[oid]['ros']*1e9 > window['end_ros_ns']:
                    continue
                preview = cost.get('corridor_preview', {})
                risks = preview if preview.get('status') == 'valid' else {}
                a = dict(episode_id=run['tag']+':'+str(oid), case=run['case'],
                    source_member=member, proxy_s=cost['approach_s'], actual_wall_s=actual,
                    offset_s=cost['approach_s']*risks.get('approach', {}).get('risk_fraction', 0.),
                    meaning='generation complete to real region acquisition; may include scheduling and intervening observations')
                approach.append(a)
                completed.append((acquired[oid]['wall'], oid, cost, risks))
            if run.get('strict_hold') and completed:
                seen_wall, oid, cost, risks = max(completed)
                # A later generated/acquired region would invalidate this as the final return.
                if any(x['wall'] > seen_wall+.02 for x in acquired.values()):
                    continue
                actual = end_wall-seen_wall
                if actual > 0. and rate > 0.:
                    returns.append(dict(episode_id=run['tag']+':return:'+str(oid), case=run['case'],
                        proxy_s=cost['rejoin_s'], actual_wall_s=actual,
                        offset_s=cost['rejoin_s']*risks.get('rejoin', {}).get('risk_fraction', 0.)+1.2/rate,
                        meaning='last real region acquisition to strict watcher exit; completed returns only'))
    (out/'episodes.json').write_text(json.dumps(dict(approach=approach, returns=returns,
        excluded=excluded, archives=provenance), indent=2)+'\n')
    report = dict(status='INSUFFICIENT_DATA', learned_model_changed=False,
        scope='development calibration; conditional completed episodes; no counterfactual ranking labels or safety bounds',
        limitations=['Legacy joins use per-run obligation id; intervening region/target changes and scheduling may confound duration.',
                     'Return labels include completed returns only; failed returns are censored and not imputed.',
                     'Historical planner/actuator mismatch differs from the corrected research runtime.',
                     'Prediction improvement is not causal candidate-ranking or task-success improvement.'],
        split='case number divisible by 3 is held-out development validation; others train', metrics={})
    parameters = {}
    for name, rows in [('approach', approach), ('return', returns)]:
        train = [r for r in rows if int(r['case']) % 3 != 0]
        validation = [r for r in rows if int(r['case']) % 3 == 0]
        if len(train) < 8 or len(validation) < 4 or len({r['case'] for r in validation}) < 3:
            report['metrics'][name] = dict(training_episodes=len(train), validation_episodes=len(validation))
            continue
        scale, intercept = fit(train)
        parameters[name] = (scale, intercept)
        report['metrics'][name] = dict(scale=scale, intercept_wall_s=intercept,
            training=metrics(train, scale, intercept), validation=metrics(validation, scale, intercept))
    if len(parameters) == 2:
        passes = all(x['validation']['calibrated_mae_s'] < .95*x['validation']['baseline_mae_s']
                     for x in report['metrics'].values())
        report['status'] = 'PASS_ESTIMATION_ONLY' if passes else 'REJECTED_NO_VALIDATION_GAIN'
        if passes:
            profile = dict(schema=1, profile_id='historical_development_case_split_v1',
                training_episode_ids=[r['episode_id'] for r in approach+returns if int(r['case']) % 3 != 0],
                approach_scale=parameters['approach'][0], acquisition_wall_s=parameters['approach'][1],
                rejoin_scale=parameters['return'][0], return_overhead_wall_s=parameters['return'][1],
                tail_ros_s=1.1, hold_ros_s=.1,
                scope='proposal ranking only; historical completed-episode selection bias; no task or safety guarantee')
            (out/'profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
