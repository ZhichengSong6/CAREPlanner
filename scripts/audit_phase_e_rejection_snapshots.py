#!/usr/bin/env python3
"""Offline integrity checks; queued diagnostics are not proof of saved snapshots."""
import hashlib
import json
import math
from pathlib import Path

from analyze_phase_e_observation_events import string_rows


def validate_record(record):
    raw, audited = record['raw_candidate_stamp_ns'], record['audited_trajectory_stamp_ns']
    gate = record['safety_gate']
    assert record['result']=='unsafe' and (gate=='vbc' or gate.startswith('gcdf')) and record['committed'] is False
    assert record['context_status']=='exact_raw_stamp'
    context=record['local_context']
    assert context['raw_candidate_stamp_ns']==raw
    assert record['raw_trajectory']['stamp_ns']==raw
    assert record['audited_trajectory']['stamp_ns']==audited
    evidence=record['vbc_evidence'] if gate=='vbc' else record['gcdf_evidence']
    assert str(evidence['trajectory_stamp_ns'])==audited
    assert evidence['violations'], 'Empty rejection evidence'
    assert record['final_gcdf_batch']['header']['stamp']==record['final_gcdf_trajectory']['stamp_ns']
    assert context['query_trajectory']['stamp_ns']==context['local_gcdf_batch']['stamp_ns']
    assert record['final_gcdf_trajectory']['points']==record['audited_trajectory']['points'], 'GCDF/VBC geometry changed'
    if gate != 'vbc':
        batch=record['final_gcdf_batch']
        assert evidence['batch_stamp_ns']==audited==batch['header']['stamp']
        for row in evidence['violations']:
            i,k=row['pair'],row['timestep']
            assert row['point']==batch['point_flat'][3*i:3*i+3] and len(row['point'])==3
            assert row['distance']==batch['distance'][i] < evidence['safety_margin']
            assert row['source_type']==batch['source_type'][i]
            assert row['gradient']==batch['gradient_flat'][7*i:7*i+7] and len(row['gradient'])==7
            assert row['q_linearization']==batch['q_linearization_flat'][7*i:7*i+7]
            assert row['candidate_q']==record['audited_trajectory']['points'][k]['positions']
            assert row['time_ns']==record['audited_trajectory']['points'][k]['time_ns']
    names=record['raw_trajectory']['joint_names']
    assert len(names)==7 and len(set(names))==7
    for trajectory in [record[k] for k in ['raw_trajectory','handoff_trajectory',
            'final_gcdf_trajectory','audited_trajectory']]+[context['query_trajectory']]:
        assert trajectory['joint_names']==names
        assert trajectory['points']
        times=[int(p['time_ns']) for p in trajectory['points']]
        assert times[0]>=0 and times==sorted(times)
        for point in trajectory['points']:
            assert len(point['positions'])==7 and all(math.isfinite(v) for v in point['positions'])
    return raw,audited


def audit_snapshots(run, window):
    run=Path(run)
    outcomes=string_rows(run/'verification_outcome.csv',window)
    rejected={(r['raw_candidate_stamp_ns'],r['audited_trajectory_stamp_ns']):r for r in outcomes
              if r.get('result')=='unsafe' and
              (r.get('safety_gate')=='vbc' or r.get('safety_gate','').startswith('gcdf'))}
    files=sorted((run/'rejection_snapshots').glob('*.json'))
    total_bytes=sum(p.stat().st_size for p in files)
    errors=[]; saved=[]
    if len(files)>16 or total_bytes>32*1024*1024:
        errors.append('Snapshot count/bytes cap exceeded')
    for path in files:
        try:
            raw=path.read_bytes(); record=json.loads(raw)
            key=validate_record(record)
            inside=key in rejected
            if inside and record['safety_gate']=='vbc':
                original=json.loads(rejected[key]['vbc_evidence'])
                assert record['vbc_evidence']==original, 'Saved evidence differs from exact outcome'
            saved.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                raw_candidate_stamp_ns=key[0],audited_trajectory_stamp_ns=key[1],
                safety_gate=record['safety_gate'],within_window=inside))
        except Exception as exc:
            errors.append(str(path)+': '+type(exc).__name__+': '+str(exc))
    for gate in {r['safety_gate'] for r in rejected.values()}:
        if not any(r['within_window'] and r['safety_gate']==gate for r in saved):
            errors.append(gate+' rejection occurred but no valid in-window snapshot was saved')
    return dict(status='FAIL' if errors else 'PASS' if rejected else 'NOT_RUN',
        errors=errors,final_vbc_rejections=sum(r['safety_gate']=='vbc' for r in rejected.values()),
        final_gcdf_rejections=sum(r['safety_gate'].startswith('gcdf') for r in rejected.values()),saved=saved,
        snapshot_bytes=total_bytes,note='Deduplicated/bounded capture, not every transaction; NOT_RUN means no GCDF/VBC rejection in window.')
