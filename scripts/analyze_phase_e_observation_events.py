#!/usr/bin/env python3
"""Join observation -> raw -> verification -> execution by exact IDs, not time proximity."""
import argparse
import csv
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path

# Per-audit voxel evidence can exceed csv's conservative 128 KiB default.
csv.field_size_limit(16 * 1024 * 1024)


def solver_point_key(point):
    # Solver inputs are FP32; GCDF-origin points may already be FP32 while
    # analytic VBC centers remain FP64. This is NOT nearest-distance matching.
    return struct.pack('!3f', *point)


def tokens(text):
    return dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)', text))


def string_rows(path, window):
    if not path.exists(): return []
    out=[]
    with path.open() as stream:
        # rostopic writes its header only after the first message. A recorder
        # may emit simulated-clock warnings while waiting for that message.
        preamble = {'WARNING: no messages received and simulated time is active.',
                    'Is /clock being published?'}
        for first_line in stream:
            header = first_line.strip()
            if header in preamble: continue
            if header != '%time,field.data':
                raise ValueError('Expected a two-column ROS String trace: ' + str(path))
            break
        else:
            return []
        for line in stream:
            stamp_text, separator, payload = line.rstrip('\r\n').partition(',')
            if not separator: continue
            # rostopic echo -p does NOT CSV-quote String payloads containing
            # commas. Preserve the entire remainder, including JSON/xyz lists.
            # Also accept conventional CSV produced by offline fixtures/tools.
            if payload.startswith('"'):
                parsed = next(csv.reader([payload]))
                if len(parsed) != 1: raise ValueError('Malformed quoted String trace')
                payload = parsed[0]
            try: stamp=float(stamp_text)*1e-9
            except ValueError: continue
            if window['start_ros_s'] <= stamp <= window['end_ros_s']:
                out.append(dict(tokens(payload), ros_s=stamp))
    return out


def exact_raw_map(log):
    result={}
    for line in log.splitlines():
        if '[OBS_CANDIDATE]' not in line: continue
        t=tokens(line); key=t.get('raw_candidate_stamp_ns'); token=t.get('observation_token')
        if key and token:
            result.setdefault(key,set()).add(token)
    return result


def attach_vbc_evidence(outcomes, events):
    counts = Counter()
    routes = {}
    for event in events:
        identity = event.get('vbc_identity')
        if event.get('stage') == 'blocker_route' and identity:
            key = (str(identity['trajectory_stamp_ns']), identity['bundle_seq'])
            routes.setdefault(key, []).append(event)
    for outcome in outcomes:
        if outcome.get('safety_gate') != 'vbc':
            continue
        encoded = outcome.get('vbc_evidence', 'none')
        if encoded == 'none':
            counts['missing'] += 1
            continue
        try:
            evidence = json.loads(encoded)
            stamp = str(evidence['trajectory_stamp_ns'])
            if (evidence['schema'] != 1 or int(stamp) <= 0 or
                    stamp != outcome.get('audited_trajectory_stamp_ns')):
                raise ValueError('audit identity mismatch')
            if not isinstance(evidence['violations'], list):
                raise ValueError('invalid violations')
        except (ValueError, KeyError, TypeError):
            counts['invalid_or_mismatched'] += 1
            continue
        outcome['vbc_detail'] = evidence
        outcome['vbc_route_events'] = routes.get((stamp, evidence['bundle_seq']), [])
        counts['exact_audits'] += 1
        if evidence['violations']:
            counts['rejected_audits'] += 1
            counts['with_exact_route'] += bool(outcome['vbc_route_events'])
            counts['without_exact_route'] += not bool(outcome['vbc_route_events'])
        for voxel in evidence['violations']:
            counts['violation_records'] += 1
            counts['requires_seen_before_trajectory_start'] += voxel['latest_see_s'] < 0
    return dict(counts)


def analyze(run, log):
    window=json.loads((run/'goal_stop_status.json').read_text())['benchmark_window']
    path=run/'projector_traces/observation_events.jsonl'
    events=[]; malformed=0
    if path.exists():
        for line in path.read_text().splitlines():
            try: event=json.loads(line)
            except ValueError: malformed+=1; continue
            if window['start_ros_s'] <= event['ros_s'] <= window['end_ros_s']: events.append(event)
    rawmap=exact_raw_map(log)
    identity_rows=string_rows(run/'observation_candidate_identity.csv',window)
    if identity_rows:
        rawmap={}
        for row in identity_rows:
            rawmap.setdefault(row['raw_candidate_stamp_ns'],set()).add(row['observation_token'])
    outcomes=string_rows(run/'verification_outcome.csv',window)
    vbc_audit = attach_vbc_evidence(outcomes, events)
    tracker=string_rows(run/'tracker_summary.csv',window)
    local_plans=string_rows(run/'local_planner_summary.csv',window)
    observations={}
    for event in events:
        token=event.get('observation_token')
        if token: observations.setdefault(token,[]).append(event)
    rows=[]
    for token, history in observations.items():
        rawids={raw for raw,ids in rawmap.items() if ids=={token}}
        verified=[v for v in outcomes if v.get('raw_candidate_stamp_ns') in rawids]
        commits=[v for v in verified if v.get('committed')=='1']
        # rospy rewrites header.seq at publication. The immutable execution
        # timestamp is the existing cross-component execution identity.
        execution={v.get('execution_stamp_ns') for v in commits if v.get('execution_stamp_ns') not in (None,'0')}
        ambiguous_execution={stamp for stamp in execution if
            len({v.get('seq') for v in outcomes if v.get('committed')=='1' and v.get('execution_stamp_ns')==stamp})>1}
        executed=[t for t in tracker if t.get('execution_stamp_ns') in execution-ambiguous_execution]
        seen=[e for e in history if e.get('actual_seen')]
        seen_queries=[e for e in history if e['stage']=='seen_query']
        scheduled=[e for e in history if e['stage']=='scheduled']
        target = scheduled[-1] if scheduled else {}
        target_points = {tuple(p) for p in target.get('target_solver_points', [])}
        if not target_points:
            target_points = {tuple(p) for p in target.get('points', [])}
        solver_points = {solver_point_key(p) for p in target_points}
        for outcome in verified:
            for voxel in outcome.get('vbc_detail', {}).get('violations', []):
                voxel['in_selected_target_point_set'] = (
                    tuple(voxel['point']) in target_points if target_points else None)
                voxel['in_selected_solver_fp32_inputs'] = (
                    solver_point_key(voxel['point']) in solver_points if solver_points else None)
        distances=[e['q_distance_inf'] for e in history if math.isfinite(e.get('q_distance_inf',math.nan))]
        rejected=[v for v in verified if v.get('committed')!='1']
        planning=[p for p in local_plans if p.get('observation_token') == token]
        stage=('ACTUAL_SEEN' if seen else
               ('COMMITTED_NOT_SEEN' if seen_queries else 'COMMITTED_SEEN_UNATTRIBUTED') if commits else
               'CERTIFICATION_REJECTED' if rejected else 'NO_VERIFICATION_OUTCOME' if rawids else
               'NO_LOCAL_CANDIDATE' if scheduled else 'GENERATED_NOT_SCHEDULED')
        rows.append(dict(observation_token=token,history=history,raw_candidate_stamps=sorted(rawids),
            verification=verified,tracker_records=len(executed),
            scheduled_count=len(scheduled),seen_query_count=len(seen_queries),
            tracker_seq_values=sorted({t.get('seq') for t in executed}),
            ambiguous_execution_stamps=sorted(ambiguous_execution),
            tracker_complete=any(t.get('complete')=='1' for t in executed),
            measured_q_distance_min_inf=min(distances) if distances else None,
            first_actual_seen_ros_s=seen[0]['ros_s'] if seen else None,last_stage=stage))
        rows[-1]['local_planning_events'] = planning
        rows[-1]['last_local_planning_event'] = planning[-1] if planning else None
    generations=[e for e in events if e['stage']=='generated']
    calls=[e['hybrid'] for e in events if e['stage']=='hybrid_call']
    hybrid=calls if calls else [e['per_sensor_hybrid'] for e in generations if e.get('per_sensor_hybrid',{}).get('enabled')]
    latency={}
    for name,group in [('all',hybrid),('accepted',[h for h in hybrid if h.get('accepted')]),
                       ('all_rejected',[h for h in hybrid if not h.get('accepted') and not h.get('runtime_error')]),
                       ('runtime_error',[h for h in hybrid if h.get('runtime_error')])]:
        values=[h['compute_ms'] for h in group if math.isfinite(h.get('compute_ms',math.nan))]
        latency[name]=dict(count=len(group),recorded_count=len(values),
            mean_ms=sum(values)/len(values) if values else None,max_ms=max(values) if values else None)
    return dict(schema_version=1,run=str(run),window=window,trace_available=path.exists(),
        candidate_identity_source='dedicated_topic' if identity_rows else 'legacy_mixed_log_partial',
        malformed_lines=malformed,events=len(events),generation_count=len(generations),
        failed_generations=[e for e in events if e['stage']=='generation_failed'],
        unfinished_generations=[e for e in events if e['stage']=='generation_started' and
            not any(x.get('generation_event_id')==e['generation_event_id'] and
                    x['stage'] in ('generated','generation_failed') for x in events)],
        ambiguous_raw_stamps=[k for k,v in rawmap.items() if len(v)>1],
        unmatched_verification_count=sum(v.get('raw_candidate_stamp_ns') not in rawmap for v in outcomes),
        hybrid_latency_ms=latency,stage_counts=dict(Counter(r['last_stage'] for r in rows)),observations=rows,
        hybrid_latency_source='all_calls' if calls else 'materialized_generations_only',
        vbc_evidence_audit=vbc_audit,
        caveat='tracker complete is not measured q_vis arrival; seen is actual map evidence, not causal head attribution; missing joins remain unknown')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True); ap.add_argument('--log',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--geometry-check', action='store_true', help='Offline only: same frozen primitive FOV/LOS for all sensors at scheduled q_vis')
    a=ap.parse_args(); report=analyze(Path(a.run),Path(a.log).read_text(errors='replace'))
    if a.geometry_check:
        from test_per_sensor_visibility_runtime_hierarchical9 import make_runtime, H9
        runtime=make_runtime(H9,'cpu')
        report['geometry_audit_role']='offline diagnostic; no optimization, no change to online acceptance or budget'
        report['geometry_audit_config']=dict(min_conservative_g=runtime.min_conservative_g,
            require_primitive_los=runtime.require_primitive_los, checkpoint=str(H9))
        for row in report['observations']:
            target=next((e for e in row['history'] if 'q_vis' in e and 'points' in e),None)
            if target:
                row['scheduled_qvis_geometry']=[dict(sensor_id=sid,
                    **runtime._candidate_geometry(target['points'],target['q_vis'],sid)) for sid in range(8)]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f: json.dump(report,f,indent=2)
    digest={k:v for k,v in report.items() if k not in ('observations','window','failed_generations','unfinished_generations')}
    digest['failed_generation_count']=len(report['failed_generations'])
    digest['unfinished_generation_count']=len(report['unfinished_generations'])
    print(json.dumps(digest,indent=2))


if __name__=='__main__': main()
