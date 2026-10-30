#!/usr/bin/env python3
"""Replay each shadow point set and verify that no incomplete evidence was admitted."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from phase_e_stop_shadow import BLOCKERS,identity


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    run=json.loads((args.run_dir/'result.json').read_text())
    manifest=json.loads((args.run_dir/'shadow/result.json').read_text())
    rows=[json.loads(line) for line in (args.run_dir/'shadow/rows.jsonl').read_text().splitlines()]
    assert run['status']=='PASS' and run['input_hashes_unchanged'] and not run['cleanup']['remaining']
    assert manifest['input_hashes_unchanged'] and manifest['rows']==len(rows)>0
    histogram=Counter(); points_total=complete_queries=bound_checks=coherent=0; point_counts=[]
    for r in rows:
        assert r['runtime_admission'] is False
        assert all(b in r['reasons'] for b in BLOCKERS),r
        assert not any(s.startswith('observer_exception') for s in r['reasons'])
        histogram.update(r['reasons'])
        if 'exact_command' in r:
            assert identity(r['exact_command'])==identity(r['state']); bound_checks+=1
        if 'enclosure' not in r: continue
        meta=r['metadata']; ids=np.asarray(r['enclosure']['grid_indices'],dtype=np.int64)
        assert len(ids)>0 and np.all(np.diff(ids)>0)
        size=np.array(meta['grid_size']); assert ids.min()>=0 and ids.max()<int(np.prod(size))
        ix=ids//(size[1]*size[2]); iy=(ids//size[2])%size[1]; iz=ids%size[2]
        points=np.array(meta['map_min'])+meta['resolution']*np.column_stack([ix,iy,iz])
        assert hashlib.sha256(np.asarray(points,dtype='<f8').tobytes()).hexdigest()==r['enclosure']['point_set_sha256']
        point_counts.append(len(ids)); points_total+=len(ids)
        evidence=r.get('evidence')
        if not evidence or not evidence['valid']: continue
        arrays=['inside_map','confidence_now','confidence_at_horizon','occupancy','last_seen_s','provenance','model_free_at_horizon']
        if any(len(evidence[k])!=len(ids) for k in arrays):
            assert 'incomplete_snapshot_arrays' in r['reasons'] and not r['binding_consistent']; continue
        complete_queries+=1
        if r['binding_consistent']:
            coherent+=1
            assert r['conditional_map_coverage'] and r['enclosure']['inside']
            assert identity(r['end_state'])==identity(r['state'])
            assert not r['pending_newer_request_ids'] and not r['initial_pending_newer_request_ids']
            assert r['end_state']['rearms']==r['state']['rearms']
            assert all(v==1 for v in evidence['inside_map'])
            assert all(v>=.8 for v in evidence['confidence_at_horizon'])
            assert all(v<=.5 for v in evidence['occupancy'])
            assert all(evidence[k]==r['version_recheck'][k] for k in ('instance','epoch','revision'))
    assert complete_queries>0 and bound_checks>0 and points_total>0
    required=['unknown_occupied_or_decayed_points','controller_command_changed_during_check','newer_request_observed',
              'stale_or_future_measured_state','missing_exact_command_identity','map_version_changed_during_check',
              'incomplete_snapshot_arrays','snapshot_service_timeout','snapshot_service_busy_after_timeout','ambiguous_command_identity']
    assert all(histogram[x]>0 for x in required)
    summary=dict(status='PASS',rows=len(rows),exact_command_bindings=bound_checks,complete_queries=complete_queries,
        reconstructed_point_sets=len(point_counts),reconstructed_points=points_total,point_count_min=min(point_counts),
        point_count_max=max(point_counts),coherent_covered_rows=coherent,
        max_observed_check_wall_s=max(r['wall_elapsed_s'] for r in rows),reason_counts=dict(histogram),
        runtime_admission=False,physical_bound_certified=False,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [args.run_dir/'result.json',args.run_dir/'shadow/rows.jsonl',args.run_dir/'shadow/result.json',Path(__file__).resolve()]})
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},indent=2))

if __name__=='__main__': main()
