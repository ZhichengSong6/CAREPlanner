#!/usr/bin/env python3
"""Join point evidence to actual map-update TF and audit recorded raster hits."""
import argparse,collections,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
from phase_e_observation_evidence import (local_samples,raster_indices,point_coverage,
    matched_update,sensor_expiry_s)


def analyze(traces,point_path,output):
    output.mkdir(parents=True,exist_ok=False)
    models={};updates={};counts=collections.Counter();tf_ages=[];mismatches=[]
    raster_cache={}
    for path in traces:
        model=None
        for line in path.open():
            row=json.loads(line)
            if row['event']=='model':
                model=row;models[row['instance']]=row
                updates[row['instance']]={};continue
            if row['event']=='map_write':
                # A frozen-layer change also advances the content version.
                # Bind it to the actual most recent sensor update, which it
                # does not modify, instead of silently dropping this record.
                previous=list(updates[row['instance']].values())
                latest=previous[-1] if previous else dict(sensors=[],observation_ns=0)
                key=(row['epoch'],row['revision'])
                updates[row['instance']][key]=dict(latest,**row)
                raster_cache[(row['instance'],)+key]=(
                    raster_cache[(row['instance'],latest['epoch'],latest['revision'])] if previous else [])
                continue
            if row['event']!='ideal_fov_update':continue
            assert model is not None and row['instance']==model['instance']
            counts['map_updates']+=1
            counts['updates_with_missing_sensor_tf']+=int(len(row['sensors'])!=model['expected_sensor_count'])
            watched={w['grid_index'] for w in model['watched']}
            samples=local_samples(model);rasters=[]
            for sensor in row['sensors']:
                usable=sensor.get('applied_to_map',True)
                counts['rejected_'+sensor.get('tf_time_reason','legacy_time_policy')]+=int(not usable)
                ids=raster_indices(model,sensor,samples) if usable else np.array([],dtype=np.int64)
                rasters.append(ids)
                expected=watched.intersection(map(int,ids));actual=set(sensor['hit_watch_grid_indices'])
                counts['sensor_raster_watch_comparisons']+=1
                if expected!=actual:
                    mismatches.append(dict(instance=row['instance'],revision=row['revision'],frame=sensor['frame'],
                        predicted_only=sorted(expected-actual),recorded_only=sorted(actual-expected)))
                if sensor['tf_stamp_ns'] and usable:
                    tf_ages.append((row['observation_ns']-sensor['tf_stamp_ns'])*1e-9)
            key=(row['epoch'],row['revision'])
            updates[row['instance']][key]=row;raster_cache[(row['instance'],)+key]=rasters
    histogram=collections.Counter();links=collections.Counter();unique=set();sensor_ages=[];error=[]
    from phase_e_local_stop_geometry import LocalPrimitiveEnclosure
    from audit_case001_motion_prior import frame_placements,collision_sdf
    geometry=LocalPrimitiveEnclosure(Path(__file__).resolve().parents[1]/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
    with gzip.open(output/'classified_points.jsonl.gz','xt',compresslevel=1) as stream:
        if point_path is not None:
            for index,line in enumerate(gzip.open(point_path,'rt')):
                raw=json.loads(line);e=raw['evidence'];counts['point_query_records']+=1
                bad=[j for j,c in enumerate(e['confidence_at_horizon']) if c<.8]
                try:
                    model=models[e['instance']];update=matched_update(e,model,updates[e['instance']])
                except (KeyError,ValueError):
                    counts['records_without_matching_observation_update']+=1
                    continue
                counts['records_matched_to_actual_update_tf']+=1
                if not bad:continue
                counts['matched_records_with_unknown']+=1
                refs=set(raw['reference_pose_grid_indices'] or [])
                points=np.array(raw['points'])[bad];grid=np.array(raw['grid_indices'])[bad]
                visible=np.zeros(len(bad),bool);raster=np.zeros(len(bad),bool);near=np.zeros(len(bad),bool)
                for sensor,ids in zip(update['sensors'],raster_cache[(e['instance'],e['epoch'],e['revision'])]):
                    if not sensor.get('applied_to_map',True):continue
                    c=point_coverage(points,model,sensor);visible|=c['visible'];near|=c['too_near'];raster|=np.isin(grid,ids)
                full=np.zeros(geometry.model.nq);full[geometry.qindices]=raw['q']
                poses=frame_placements(geometry.model,geometry.data,geometry.shapes,full)
                distance=np.array([collision_sdf(points,geometry.shapes[k],poses[k]) for k in geometry.active])
                nearest=np.argmin(distance,axis=0);minimum=distance.min(axis=0)
                details=[]
                for k,j in enumerate(bad):
                    reference=int(grid[k]) in refs
                    kind=('visible_raster_hit_but_unknown' if raster[k] else
                          'continuous_fov_but_raster_miss' if visible[k] else
                          'angle_eligible_but_below_min_range' if near[k] else 'outside_current_sensor_fov')
                    counts['matched_unknown_point_instances']+=1
                    if reference:
                        counts['reference_unknown_point_instances']+=1
                        counts['reference_already_unknown_now']+=int(e['confidence_now'][j]<.8)
                        histogram[kind]+=1;unique.add(int(grid[k]))
                        links[geometry.shapes[geometry.active[nearest[k]]]['link']]+=1
                        counts['reference_inside_collision_model' if minimum[k]<=0 else 'reference_outside_model_within20mm']+=1
                        counts['reference_provenance_'+str(e['provenance'][j])]+=1
                        if e['provenance'][j]==1:
                            age=e['snapshot_ns']*1e-9-e['last_seen_s'][j];sensor_ages.append(age)
                            error.append(abs(math.exp(-age/e['temporal_decay_time'])-e['confidence_now'][j]))
                    details.append(dict(grid_index=int(grid[k]),point=raw['points'][j],reference_margin=reference,
                        classification=kind,confidence_now=e['confidence_now'][j],confidence_at_horizon=e['confidence_at_horizon'][j],
                        provenance=e['provenance'][j],last_seen_s=e['last_seen_s'][j],
                        sensor_expiry_s=sensor_expiry_s(e['last_seen_s'][j],e['temporal_decay_time']) if e['provenance'][j]==1 else None,
                        current_pose_sdf_m=float(minimum[k]),nearest_link=geometry.shapes[geometry.active[nearest[k]]]['link']))
                stream.write(json.dumps(dict(record=index,q=raw['q'],instance=e['instance'],epoch=e['epoch'],revision=e['revision'],
                    snapshot_ns=e['snapshot_ns'],map_update_ns=e['map_update_ns'],actual_update_observation_ns=update['observation_ns'],
                    source_tf_stamps_ns={s['frame']:s['tf_stamp_ns'] for s in update['sensors']},details=details))+'\n')
    def stats(values):
        return dict(zip(['min','median','p95','max'],map(float,np.percentile(values,[0,50,95,100])))) if values else None
    summary=dict(status='PASS_EXACT_TRACE_RASTER_AUDIT' if not mismatches else 'FAIL_TRACE_RASTER_MISMATCH',
        runtime_admission=False,scope='Actual recorded map-update TF; ideal_fov has no occlusion model; point instances include repeats',
        counts=dict(counts),reference_unknown_classification=dict(histogram),nearest_link_point_instances=dict(links),
        unique_reference_unknown_grid_points=len(unique),sensor_tf_age_s=stats(tf_ages),reference_sensor_observation_age_s=stats(sensor_ages),
        confidence_formula_max_error=max(error) if error else None,raster_mismatches=mismatches,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in traces+([point_path] if point_path else [])})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--point-evidence',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    traces=sorted(args.input_dir.glob('*observations.jsonl'))
    assert traces,'No observation traces'
    result=analyze(traces,args.point_evidence,args.output_dir)
    raise SystemExit(0 if result['status'].startswith('PASS') else 1)
