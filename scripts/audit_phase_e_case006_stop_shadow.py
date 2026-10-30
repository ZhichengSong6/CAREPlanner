#!/usr/bin/env python3
"""Bind formal benchmark window, controller/command CSVs and shadow point evidence."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch-id',required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    name='phase_e_goal_006_'+args.batch_id
    batch=ROOT/'outputs/phase_e_r1_mainline_30'/args.batch_id
    run=ROOT/'outputs/c5_5_vbc_gcdf_regime'/name/'run'
    summary=json.loads((batch/'case_summaries/phase_e_goal_006.json').read_text())
    status=json.loads((batch/'logs/phase_e_goal_006_status.json').read_text())
    cleanup=json.loads((batch/'logs/phase_e_goal_006_cleanup.json').read_text())
    manifest=json.loads((run/'stop_shadow/result.json').read_text())
    assert status==dict(runner_return_code=0,evaluator_return_code=0)
    assert not cleanup['remaining'] and manifest['input_hashes_unchanged']
    assert (batch/'summary.json').is_file(), 'runner fingerprint gate not completed'
    rows=[json.loads(s) for s in (run/'stop_shadow/rows.jsonl').read_text().splitlines()]
    assert len(rows)==manifest['rows']>0
    window=summary['benchmark_window']; begin,end=window['start_ros_ns'],window['end_ros_ns']
    selected=[r for r in rows if begin<=r['attempt_start_ros_ns']<=end]
    def csv_rows(file):
        with (run/file).open() as f: return list(csv.DictReader(f))
    states={int(r['field.header.stamp']):r for r in csv_rows('actuator_watchdog_state.csv') if r.get('field.header.stamp','').isdigit()}
    commands={(int(r['field.header.stamp']),int(r['field.header.seq'])):r for r in csv_rows('actuator_stamped_request.csv') if r.get('field.header.stamp','').isdigit()}
    hist=Counter(); counts=Counter(); point_counts=[]; unknown=Counter(); examples=[]; latencies=[]
    for row in selected:
        assert row['protocol']=='content_snapshot_then_recheck_v2' and row['runtime_admission'] is False
        assert not any(s.startswith('observer_exception') for s in row['reasons'])
        hist.update(row['reasons']); latencies.append(row['wall_elapsed_s'])
        counts['binding_consistent']+=bool(row['binding_consistent'])
        counts['check_straddles_window_end']+=row['attempt_end_ros_ns']>end
        if 'state' in row and row['state']:
            s=row['state']; original=states.get(s['t_ns'])
            if original:
                assert int(original['field.command_stamp'])==s['stamp_ns'] and int(original['field.command_sequence'])==s['sequence']
                for key,csv_key in [('q','measured_position'),('applied','applied_velocity')]:
                    assert np.max(np.abs(np.array(s[key])-[float(original['field.'+csv_key+str(j)]) for j in range(7)]))<1e-9
                counts['state_csv_matched']+=1
            else: counts['state_csv_missing']+=1
        if 'exact_command' in row:
            cmd=row['exact_command']; original=commands.get((cmd['stamp_ns'],cmd['sequence']))
            if original:
                assert np.max(np.abs(np.array(cmd['velocity'])-[float(original['field.velocity'+str(j)]) for j in range(7)]))<1e-9
                counts['command_csv_matched']+=1
            else: counts['command_csv_missing']+=1
        if 'enclosure' not in row: continue
        meta=row['metadata']; ids=np.array(row['enclosure']['grid_indices'],dtype=np.int64)
        n=np.array(meta['grid_size']); assert len(ids) and np.all(np.diff(ids)>0)
        xyz=np.column_stack([ids//(n[1]*n[2]),ids//n[2]%n[1],ids%n[2]])
        points=np.array(meta['map_min'])+meta['resolution']*xyz
        assert hashlib.sha256(np.asarray(points,dtype='<f8').tobytes()).hexdigest()==row['enclosure']['point_set_sha256']
        counts['reconstructed_point_sets']+=1; counts['reconstructed_points']+=len(ids); point_counts.append(len(ids))
        counts['enclosure_may_leave_map']+=not row['enclosure']['inside']
        evidence=row.get('evidence')
        if not evidence or not evidence['valid']: continue
        assert all(len(evidence[k])==len(ids) for k in ('confidence_at_horizon','occupancy','inside_map','model_free_at_horizon','provenance'))
        counts['valid_full_point_queries']+=1
        advanced=evidence['revision']>meta['revision']; counts['content_revision_advanced_before_read']+=advanced
        assert (meta['instance'],meta['epoch'])==(evidence['instance'],evidence['epoch'])
        bad=(np.array(evidence['confidence_at_horizon'])<.8)|(np.array(evidence['occupancy'])>.5)|(np.array(evidence['inside_map'])!=1)
        counts['point_coverage_complete']+=not bad.any()
        counts['point_coverage_and_inside_map']+=(not bad.any()) and row['enclosure']['inside']
        counts['queries_with_occupied_points']+=bool(np.any(np.array(evidence['occupancy'])>.5))
        counts['queries_using_bootstrap']+=bool(np.any(np.array(evidence['provenance'])&2))
        counts['queries_using_test_oracle']+=bool(np.any(np.array(evidence['provenance'])&4))
        counts['ideal_fov_queries']+=evidence['observation_mode']=='ideal_fov'
        if row.get('version_recheck',{}).get('valid'):
            counts['version_recheck_passed']+=1
            counts['advanced_revision_recheck_passed']+=advanced
        for p in points[bad]: unknown[tuple(np.round(p,6))]+=1
        if bad.any() and len(examples)<5:
            examples.append(dict(row_id=row['row_id'],state_t_ns=row['state']['t_ns'],unknown_count=int(bad.sum()),
                sample_points=points[bad][:8].tolist(),reasons=row['reasons']))
    inside_states=[r for t,r in states.items() if begin<=t<=end]
    mode_hist=Counter(r['field.mode'] for r in inside_states)
    archive=ROOT/('CAREPlanner_C5_RESULT_'+name+'.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        zip_shadow=[p for p in z.namelist() if 'stop_shadow/' in p]
        assert any(p.endswith('rows.jsonl') for p in zip_shadow)
    result=dict(status='PASS',batch_id=args.batch_id,benchmark_window_ros_s=[begin/1e9,end/1e9],
        task_success=summary['task_success'],overall_safe=summary['overall_safe'],commit_count=summary['commit_count'],
        execution_vbc_unsafe=summary['execution_vbc_unsafe_records'],execution_vbc_records=summary['execution_vbc_records'],
        final_position_error_m=summary['final_position_error_m'],final_orientation_error_rad=summary['final_orientation_error_rad'],
        all_shadow_rows=len(rows),window_shadow_rows=len(selected),counts=dict(counts),reason_counts=dict(hist),
        point_count_range=[min(point_counts),max(point_counts)] if point_counts else None,
        wall_check_ms=dict(median=float(np.median(latencies))*1e3,p95=float(np.percentile(latencies,95))*1e3,max=max(latencies)*1e3),
        most_frequent_uncovered_centers=[dict(point=list(p),queries=k) for p,k in unknown.most_common(12)],examples=examples,
        controller_mode_counts=dict(mode_hist),controller_clock_fault_rows=sum(r['field.clock_fault']=='1' for r in inside_states),
        zip_shadow_entries=zip_shadow,cleanup_remaining=cleanup['remaining'],runtime_admission=False,
        caution='Point coverage failures are conditional stopping-envelope diagnostics, not executed VBC violations or proof of minimum required margin.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('examples','most_frequent_uncovered_centers','zip_shadow_entries')},indent=2))

if __name__=='__main__': main()
