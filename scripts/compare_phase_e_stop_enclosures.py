#!/usr/bin/env python3
"""Four-way offline comparison; recorded-prefix ranges are counterfactual.

The saved map rows lack per-point evidence, so this script cannot infer new
free-space coverage. No future recorded target is treated as an online proof.
"""
import argparse
import bisect
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from audit_case001_motion_prior import frame_placements
from phase_e_local_stop_geometry import LocalPrimitiveEnclosure
from phase_e_stop_envelope import vector
from phase_e_stopping_contract import bounded_update_travel

ROOT = Path(__file__).resolve().parents[1]
LOW = np.array([-.95, -.95, 0.])
HIGH = np.array([.95, .95, 1.15])


def target_range_budget(applied, acceleration, velocity_limits, target_abs,
                        hold_s, gap_s, max_step_s):
    """Conditional absolute travel if ALL future targets obey target_abs.

    Existing nonzero output is preserved even when a target bound is zero.
    Its decay while tracking is conservatively ignored. This does not amend or
    revoke any already-issued rights; enforcement is a separate prerequisite.
    """
    return bounded_update_travel(applied,acceleration,velocity_limits,hold_s,gap_s,max_step_s,target_abs)


def global_boxes(geometry, q, travel, margin=.020):
    full = np.zeros(geometry.model.nq); full[geometry.qindices] = q
    placements = frame_placements(geometry.model, geometry.data, geometry.shapes, full)
    inflation = geometry.radii @ travel
    result = {}
    for i in geometry.active:
        shape, placement = geometry.shapes[i], placements[i]
        rotation = np.asarray(placement.rotation)
        if shape['kind'] == 'box':
            extent = np.abs(rotation) @ shape['half_size']
        elif shape['kind'] == 'cylinder':
            extent = shape['radius']*np.sqrt(rotation[:, 0]**2+rotation[:, 1]**2)+shape['half_length']*np.abs(rotation[:, 2])
        else:
            extent = np.full(3, shape['radius'])
        result[i] = (placement.translation-extent-margin-inflation[i],
                     placement.translation+extent+margin+inflation[i])
    return result


def extent(geometry, q, travel, local=False):
    global_ = global_boxes(geometry, q, travel)
    boxes = geometry.boxes(q, travel) if local else None
    lows = []; highs = []; outside = []
    for i in geometry.active:
        pieces = boxes[i] if local else [global_[i]]
        for low, high in pieces:
            low, high = np.maximum(low, global_[i][0]), np.minimum(high, global_[i][1])
            if np.any(low > high):
                continue
            lows.append(low); highs.append(high)
            if np.any(low < LOW) or np.any(high > HIGH):
                outside.append(geometry.shapes[i]['name'])
    minimum, maximum = np.min(lows, axis=0), np.max(highs, axis=0)
    return dict(inside=not outside, minimum=minimum.tolist(), maximum=maximum.tolist(),
                outside_primitives=sorted(set(outside)),
                maximum_global_inflation_m=float(max((geometry.radii@travel)[geometry.active])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--subdivisions', type=int, default=8)
    args = parser.parse_args(); source = args.input_dir.resolve(); out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    geometry = LocalPrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
                                       subdivisions=args.subdivisions)
    trace = [json.loads(line) for line in (source/'controller_trace.jsonl').open()]
    commands = [row for row in trace if row['kind'] == 'command']
    commands.sort(key=lambda row: row['generated_ns'])
    times = [row['generated_ns'] for row in commands]
    identity = lambda r: tuple(r[k] for k in ['controller_session', 'execution_stamp_ns', 'lease_sequence', 'source_sequence'])
    command_index = {identity(row): row for row in commands}
    modes = ['baseline', 'target_range_only', 'local_geometry_only', 'combined']
    counts = {mode: collections.Counter() for mode in modes}; latencies = {mode: [] for mode in modes}
    run = json.loads((source/'result.json').read_text())
    states = {row['t_ns']: row for row in trace if row['kind'] == 'state'}
    records = []
    for row in map(json.loads, (source/'shadow/rows.jsonl').open()):
        if 'map' in row:
            records.append(dict(group='shadow', state=row['state'], now_ns=row['attempt_ns'],
                until_ns=row['enclosed_authorization_until_ns'], budgets=row['budgets']))
    for trial in run['trials']:
        state = states[trial['baseline_ns']]
        records.append(dict(group='trial', label=trial['label'], state=state, now_ns=state['t_ns'],
            until_ns=trial['fixed_until_ns']))
    with (out/'rows.jsonl').open('x') as stream:
        for index, record in enumerate(records):
            state = record['state']; extra = .01 if record['group'] == 'shadow' else 0.
            until_ns = record['until_ns']+round(extra*1e9)
            relevant = commands[bisect.bisect_left(times, state['t_ns']):bisect.bisect_right(times, until_ns)]
            exact_command = command_index[identity(state)]
            cap = np.max(np.abs([c['velocity'] for c in relevant+[exact_command]]), axis=0)
            hold = (record['until_ns']-state['t_ns'])*1e-9+extra
            args_budget = [state['applied'], state['acceleration_limits'], state['velocity_limits']]
            base, _ = target_range_budget(*args_budget, state['velocity_limits'], hold, .001, state['max_braking_step_s'])
            restricted, _ = target_range_budget(*args_budget, cap, hold, .001, state['max_braking_step_s'])
            # Verify the reconstructed baseline agrees with the archived budget.
            if record.get('budgets'):
                np.testing.assert_allclose(base, record['budgets']['continue_then_stop']['command_travel_rad'], atol=1e-12)
            result = dict(group=record['group'], label=record.get('label'), state_ns=state['t_ns'],
                source_identity=list(identity(state)), target_abs=cap.tolist(),
                baseline_travel_rad=base.tolist(), restricted_travel_rad=restricted.tolist(), modes={})
            for mode in modes:
                start = time.monotonic()
                box = extent(geometry, state['q'], restricted if mode in ('target_range_only', 'combined') else base,
                             local=mode in ('local_geometry_only', 'combined'))
                elapsed = time.monotonic()-start
                result['modes'][mode] = dict(**box, wall_s=elapsed)
                counts[mode][record['group']+'_count'] += 1
                counts[mode][record['group']+'_inside'] += int(box['inside'])
                latencies[mode].append(elapsed)
            stream.write(json.dumps(result)+'\n'); stream.flush()
            if index % 100 == 0:
                print(json.dumps(dict(completed=index+1, total=len(records))), flush=True)
    summary = dict(status='PASS_OFFLINE_COMPARISON', runtime_admission=False,
        scope='command-domain geometry; restricted ranges use recorded future targets and require new controller enforcement',
        map_confidence_recomputed=False, physical_response_qualified=False, subdivisions=args.subdivisions,
        counts={k: dict(v) for k, v in counts.items()},
        wall_s={k: dict(p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)), maximum=max(v)) for k,v in latencies.items()},
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/'controller_trace.jsonl', source/'shadow/rows.jsonl', source/'result.json',
             Path(__file__).resolve(), ROOT/'scripts/phase_e_local_stop_geometry.py',
             ROOT/'scripts/phase_e_stopping_contract.py',ROOT/'scripts/phase_e_stop_envelope.py',
             ROOT/'scripts/phase_e_stop_shadow.py',ROOT/'scripts/audit_case001_motion_prior.py',
             ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf']})
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
