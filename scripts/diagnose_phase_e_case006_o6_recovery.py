#!/usr/bin/env python3
"""Read-only reconstruction of CASE006 O6 recovery; no ROS/model execution.

Raw candidates, VBC evidence and executions are joined by IDs. Joint samples
use the last recorded measurement at/before an event and are descriptive only:
the recorder does not expose callback ordering or the ledger's private anchor.
"""
import argparse
import bisect
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = 'r1_case006_stop_shadow_20261005'


def strings(path):
    """rostopic String payloads are not CSV quoted; preserve embedded JSON."""
    result = []
    with path.open() as stream:
        assert stream.readline().strip() == '%time,field.data', path
        for line in stream:
            stamp, sep, payload = line.rstrip('\r\n').partition(',')
            if not sep or not stamp.isdigit():
                continue
            if payload.startswith('"'):
                decoded = next(csv.reader([payload]))
                assert len(decoded) == 1
                payload = decoded[0]
            fields = (json.loads(payload) if payload.startswith('{') else
                      dict(re.findall(r'(\w+)=([^\s]+)', payload)))
            result.append(dict(fields, receipt_ns=int(stamp)))
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def norm_inf(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'outputs/phase_e_case006_o6_recovery_20261005')
    args = parser.parse_args()
    run = ROOT/'outputs/c5_5_vbc_gcdf_regime'/('phase_e_goal_006_'+BATCH)/'run'
    names = ['candidate_replacement_request', 'candidate_replacement_grant',
             'candidate_replacement_trigger', 'local_planner_summary',
             'verification_outcome', 'observation_candidate_identity',
             'observation_dependency', 'blocker_stack_summary',
             'local_witness_selector_diagnostics', 'tracker_summary']
    paths = [run/(name+'.csv') for name in names]
    trace_path = run/'projector_traces/candidate_replacement.jsonl'
    o6_path, = (run/'projector_traces').glob('c46_obligation_006_*.json')
    o9_path, = (run/'projector_traces').glob('c46_obligation_009_*.json')
    sources = [ROOT/p for p in [
        'src/egocentric_arm_planner/include/egocentric_arm_planner/safe_frontier_recovery.hpp',
        'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',
        'src/care_visibility_cdf/scripts/candidate_replacement_runtime.py',
        'src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py']]
    metadata_path = ROOT/'outputs/phase_e_r1_mainline_30'/BATCH/'metadata.json'
    inputs = paths+[trace_path, o6_path, o9_path, run/'joint_states.csv', metadata_path]+sources
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    frozen = json.loads(metadata_path.read_text())['source_hashes']
    source_matches = {str(p.relative_to(ROOT)):
                      hashes[str(p.relative_to(ROOT))] == frozen[str(p.relative_to(ROOT))]
                      for p in sources}
    assert all(source_matches.values()), 'Production source differs from recorded run'
    data = {name: strings(run/(name+'.csv')) for name in names}
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    o6, o9 = json.loads(o6_path.read_text()), json.loads(o9_path.read_text())
    requests = {int(r['request_id']): r for r in data['candidate_replacement_request']
                if r['action'] != 'finish'}
    grants = {int(r['request_id']): r for r in data['candidate_replacement_grant']}
    assert len(requests) == len(grants) == 9
    applied = [t for t in trace if t['event'] == 'applied_requires_existing_gates']
    assert [t['request_id'] for t in applied] == [2, 3, 7]
    candidates = []
    initial_h = o6['per_sensor_hybrid']
    for request_id, token, hybrid, seed in [
        (0, o6['observation_token'], initial_h, o6['c4_6_measured_seed_q'])
    ]+[(t['request_id'], t['new_token'], t['hybrid'], t['measured_seed']) for t in applied]:
        selected = next(a for a in hybrid['attempts']
                        if a['sensor_id'] == hybrid['selected_sensor_id'])
        q = hybrid['selected_q_vis']
        candidates.append(dict(request_id=request_id, token=token,
            sensor_id=hybrid['selected_sensor_id'], frame=hybrid['selected_sensor_frame'],
            rank=hybrid['selected_rank'], q_vis=q, joint_mask=hybrid['selected_joint_mask'],
            measured_seed=seed, delta_q=[x-y for x, y in zip(q, seed)],
            target_gap_inf_rad=norm_inf(q, seed), final_score=selected['final_score'],
            geometry=selected['geometry'],
            score_override=hybrid.get('replacement_score_override'),
            group_priority=hybrid.get('replacement_group_priority')))
    tokens = {c['token']: c for c in candidates}
    identities = {}
    for r in data['observation_candidate_identity']:
        raw = r['raw_candidate_stamp_ns']
        assert raw not in identities or identities[raw] == r['observation_token']
        identities[raw] = r['observation_token']
    outcomes = []
    for r in data['verification_outcome']:
        token = identities.get(r.get('raw_candidate_stamp_ns'))
        if token not in tokens:
            continue
        assert r['safety_gate'] == 'vbc'
        evidence = json.loads(r['vbc_evidence'])
        assert str(evidence['trajectory_stamp_ns']) == r['audited_trajectory_stamp_ns']
        outcomes.append(dict(receipt_ns=r['receipt_ns'], token=token,
            sensor_id=tokens[token]['sensor_id'], result=r['result'],
            committed=int(r['committed']), raw_ns=int(r['raw_candidate_stamp_ns']),
            audited_ns=int(r['audited_trajectory_stamp_ns']),
            execution_ns=int(r['execution_stamp_ns']), evidence=evidence))
    for c in candidates:
        c['exact_vbc_outcomes'] = dict(Counter(o['result'] for o in outcomes if o['token']==c['token']))
    assert [c['exact_vbc_outcomes'] for c in candidates] == [
        {'unsafe': 2}, {'unsafe': 2}, {'unsafe': 3}, {'safe': 7}]
    queries = [r for r in data['local_planner_summary'] if r['event']=='cdf_query_armed']
    transactions = []
    for request_id, request in requests.items():
        grant = grants[request_id]
        events = [t for t in trace if str(t.get('request_id')) == str(request_id)]
        terminal = events[-1]
        prior_query = max((q for q in queries if q['receipt_ns']<=grant['receipt_ns']),
                          key=lambda q:q['receipt_ns'])
        transactions.append(dict(request_id=request_id,
            request_receipt_ns=request['receipt_ns'], reason=request['reason'],
            grant_receipt_ns=grant['receipt_ns'], granted=int(grant['granted']),
            attempts=int(grant['attempts']), segments=int(grant['segments']),
            query_ns=int(request['query_stamp_ns']),
            latest_recorded_query_ns=int(prior_query['query_stamp_ns']),
            owner_promotion=int(grant['owner_promotion']), outcome=terminal['event'],
            finish_trace_ros_s=terminal['ros_s'],
            sensor_id=terminal.get('hybrid', {}).get('selected_sensor_id')))
    assert transactions[5]['query_ns'] != transactions[5]['latest_recorded_query_ns']
    holds = [r for r in data['local_planner_summary']
             if r['event']=='repair_candidate_replacement_exhausted_hold']
    assert len(holds) == 1
    hold = holds[0]
    assert hold['safe_stall_attempts']=='6' and hold['safe_stall_trigger_commits']=='0'
    last_token = candidates[-1]['token']
    recovery_events = [r for r in data['local_planner_summary']
        if r.get('observation_token')==last_token and r['event'] in
        ['repair_safe_stall_frontier_retry', 'repair_candidate_replacement_exhausted_hold']]
    assert len(recovery_events)==6
    assert [int(r['safe_stall_attempts']) for r in recovery_events]==[5]*5+[6]
    assert all(float(r['safe_stall_trigger_duration_s'])==0 for r in recovery_events)
    assert not [o for o in outcomes if o['token']==last_token and o['result']!='safe']
    with (run/'joint_states.csv').open() as stream:
        joints = [dict(receipt_ns=int(r['%time']), stamp_ns=int(r['field.header.stamp']),
                       q=[float(r['field.position'+str(i)]) for i in range(7)])
                  for r in csv.DictReader(stream) if r['%time'].isdigit()]
    joints.sort(key=lambda r:r['receipt_ns'])
    receipts = [j['receipt_ns'] for j in joints]
    def measured(ns):
        j = joints[bisect.bisect_right(receipts, ns)-1]
        return dict(j, event_ns=ns, age_ms=(ns-j['stamp_ns'])/1e6)
    movement = [dict(event=r['event'], attempts=int(r['safe_stall_attempts']),
                    latest_execution_ns=int(r['latest_execution_stamp_ns']),
                    **measured(r['receipt_ns'])) for r in recovery_events]
    start = measured(round(applied[-1]['ros_s']*1e9))
    finish = measured(hold['receipt_ns'])
    delta = [x-y for x,y in zip(finish['q'], start['q'])]
    deps = [r for r in data['observation_dependency']
            if r.get('observation_token') in tokens]
    final_witnesses = [r for r in data['local_witness_selector_diagnostics']
                       if r['query_stamp_ns']==hold['query_stamp_ns']]
    point_states = {}
    for r in final_witnesses:
        p = json.loads(r['point'])
        key = ','.join(f'{v:.5f}' for v in p)
        point_states.setdefault(key, Counter())[r['status']] += 1
    assert len(final_witnesses)==280
    assert all(r['status']=='evaluated_unknown' for r in final_witnesses)
    stack_changes, prior = [], None
    for r in data['blocker_stack_summary']:
        if not 20_800_000_000<=r['receipt_ns']<=22_910_000_000:
            continue
        item = {k:r[k] for k in ['stack','current_target_id','dependency_reason','switch_reason']}
        if item != prior:
            stack_changes.append(dict(item, receipt_ns=r['receipt_ns']))
            prior = item
    result = dict(batch=BATCH, scope='offline diagnosis of recorded run; no counterfactual rollout',
        counts=dict(reservations=9, granted=6, denied=3, sensor_replacements=3,
                    owner_promotions=2, granted_context_changed=1),
        obligation_points=o6['active_set_points_xyz'], o9_points=o9['active_set_points_xyz'],
        candidates=candidates, transactions=transactions, outcomes=outcomes,
        dependencies=deps, stack_changes=stack_changes, hold=hold,
        ledger_explanation=[
            'request 2: inactive -> 1 for rejected original -> 2 for replacement',
            'second S3 VBC rejection: 2 -> 3; request 3 reuses that charge',
            'second S4 VBC rejection: 3 -> 4; request 4 reuses charge, then context_changed',
            'O9 owner promotion does not spend O6 budget; restoring O6 retains stage 4',
            'request 6 stale query, denied; request 7: 4 -> 5, final replacement used',
            'requests 8/9 denied; stage 5 replacement already used (8 also stale query)',
            'six measured detour-segment transitions at stage 5 -> stage 6 -> hold'],
        terminal_transition_basis=dict(
            classification='source-and-event inference: max_segments reached, not tiny-motion stall',
            private_segments_not_recorded_at_hold=True,
            events=movement, stall_trigger_commits=int(hold['safe_stall_trigger_commits']),
            stall_trigger_duration_s=float(hold['safe_stall_trigger_duration_s']),
            source_rule='observe: >=.01 rad from anchor, !following_frontier, ++segments>=6 -> advance; blocked when attempts>5'),
        movement=dict(start=start, hold=finish, delta_q=delta,
            displacement_inf_rad=max(map(abs,delta)),
            target_gap_start_inf_rad=norm_inf(start['q'],candidates[-1]['q_vis']),
            target_gap_hold_inf_rad=norm_inf(finish['q'],candidates[-1]['q_vis'])),
        last_query=dict(stamp_ns=int(hold['query_stamp_ns']),
            witness_rows=len(final_witnesses), points={p:dict(c) for p,c in point_states.items()}),
        limitations=[
            'Grant messages lack explicit denial reason; stale query and spent final replacement are independently sufficient failed gates.',
            'Receipt-aligned joint samples do not identify the exact in-process callback sample or anchor.',
            'Positive primitive distance inside 40 mm candidate shell is not body penetration.',
            'Terminal FOV acceptance is not path feasibility or completed observation.',
            'This run cannot establish whether a larger budget, another sensor or a different detour would succeed.'])
    after = {str(p.relative_to(ROOT)):digest(p) for p in inputs}
    assert hashes == after, 'An input changed during analysis'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir/'summary.json', result)
    write_json(args.output_dir/'inputs_sha256.json', dict(unchanged=True, files=hashes,
        sources_match_formal_run=source_matches, analysis_script_sha256=digest(Path(__file__))))
    with (args.output_dir/'transactions.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(transactions[0]))
        writer.writeheader(); writer.writerows(transactions)
    with (args.output_dir/'vbc_points.csv').open('w') as stream:
        keys=['sensor_id','raw_ns','audited_ns','point','source_collision_name',
              'primitive_signed_distance_m','confidence','sweep_s','latest_see_s','nominally_visible']
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
        for outcome in outcomes:
            for v in outcome['evidence']['violations']:
                writer.writerow({k:(outcome[k] if k in outcome else
                    json.dumps(v[k]) if k=='point' else v[k]) for k in keys})
    print(json.dumps(dict(output=str(args.output_dir), counts=result['counts'],
        outcomes={str(c['sensor_id']):c['exact_vbc_outcomes'] for c in candidates},
        hold_s=hold['receipt_ns']/1e9, measured_displacement_inf_rad=max(map(abs,delta)),
        input_hashes_unchanged=True),indent=2))


if __name__ == '__main__':
    main()
