#!/usr/bin/env python3
"""Offline URDF point/voxel geometry and QP support diagnostics; no plan output."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
import yaml

from diagnose_phase_e_repair_conflicts import REPO, JOINTS, sha, sparse, reconstruct

sys.path.insert(0,str(REPO/'src/care_visibility_cdf/scripts'))
from check_visibility_self_occlusion import load_collision_primitives
from validate_visibility_oracle import URDF, find_chain_joints, fk_transform


def primitive_sdf(kind, point, size):
    """Exact signed point distance to one box, finite cylinder, or sphere."""
    p=np.asarray(point,dtype=float)
    if kind=='box': a=np.abs(p)-np.asarray(size)/2
    elif kind=='cylinder': a=np.array([np.linalg.norm(p[:2])-size[0],abs(p[2])-size[1]/2])
    elif kind=='sphere': return float(np.linalg.norm(p)-size[0])
    else: raise ValueError('unsupported primitive '+kind)
    return float(np.linalg.norm(np.maximum(a,0))+min(float(a.max()),0.))


def row_support(data,row,bounds,dynamics):
    g=sparse(data['G']).getrow(row).toarray().ravel()
    if not dynamics:
        lo,hi=np.asarray(bounds).T
        value=float(np.where(g>=0,g*hi,g*lo).sum())
        return dict(status=0,max_lhs=value,deficit=float(data['h_l'][row]-value))
    result=linprog(-g,A_eq=sparse(data['A']),b_eq=data['b'],bounds=bounds,method='highs')
    if not result.success: raise ValueError('base dynamics infeasible: '+result.message)
    value=float(-result.fun)
    return dict(status=int(result.status),max_lhs=value,deficit=float(data['h_l'][row]-value))


def coverage_audit(robot,samples):
    by_link={s['link_name']:s for s in samples}
    missing=[]; type_mismatches=[]
    for link in robot.links:
        stored=by_link.get(link.name,{}).get('samples',[])
        for i,collision in enumerate(link.collisions):
            group=[s for s in stored if s['source_collision_index']==i]
            kind=collision.geometry.__class__.__name__.lower()
            entry=dict(link=link.name,collision_index=i,name=collision.name,type=kind)
            if not group: missing.append(entry)
            elif any(s['source_type']!=kind for s in group): type_mismatches.append(entry)
    return dict(missing_primitive_samples=missing,source_type_mismatches=type_mismatches,
        caveat='Matching an index/type is not a proof of geometric enclosure or matching origin/dimensions.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnosis',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    import coal
    import pinocchio as pin
    source=json.loads(args.diagnosis.read_text())
    assert source['tests_status']=='PASS'
    batch=Path(source['source_batch'])
    frozen=json.loads((batch/'frozen_sha256.json').read_text())
    for path,digest in frozen.items():
        if path.startswith(('src/','devel/')): assert sha(REPO/path)==digest,path
    urdf=REPO/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    robot=URDF.from_xml_file(str(urdf)); primitives=load_collision_primitives(robot)
    assert len(primitives)==sum(len(l.collisions) for l in robot.links),'unsupported geometry omitted'
    pm=pin.buildModelFromUrdf(str(urdf)); pd=pm.createData()
    assert list(pm.names)[1:]==JOINTS,'Pinocchio joint order mismatch'
    cfg=yaml.safe_load((REPO/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
    mapcfg=yaml.safe_load((REPO/'src/care_confidence_map/config/confidence_map_phase_e_ray.yaml').read_text())['confidence_map']
    runs=json.loads((batch/'witness_audit/summary.json').read_text())['cases']
    samples=yaml.safe_load((REPO/'src/care_confidence_map/config/body_samples.yaml').read_text())['body_sampling']['links']
    req=coal.DistanceRequest(); req.enable_signed_distance=True
    tiny_radius=1e-8
    max_fk_error=max_coal_error=max_proxy_error=0.
    results=[]
    for case in source['cases']:
        data=json.loads(Path(case['snapshot']).read_text()); assert sha(case['snapshot'])==case['sha256']
        q0,center,radius,_=reconstruct(data,cfg)
        run=Path(next(c['run'] for c in runs if (c['model'],c['case_id'])==(case['model'],case['case_id'])))
        sem=dict(l.split('=',1) for l in (run/'runtime_semantics.txt').read_text().splitlines() if '=' in l)
        inflation=float(sem['gcdf_body_inflation_m']); assert inflation==.015
        for conflict in case['conflicts']:
            pair=conflict['pair']; p=np.asarray(conflict['grid_point']); k=data['original_timestep'][pair]
            configurations=[]
            for label,q,ref in [('measured',q0,conflict['initial_geometry']),('linearization',center[k-1],conflict['failure_geometry'])]:
                qmap=dict(zip(JOINTS,q)); pin.framesForwardKinematics(pm,pd,q)
                rows=[]; proxy=[]; transforms={}
                for primitive in primitives:
                    link=primitive['link']
                    tf=fk_transform(find_chain_joints(robot,'base_link',link),qmap)
                    transforms[link]=tf
                    independent=np.asarray(pd.oMf[pm.getFrameId(link)].homogeneous)
                    max_fk_error=max(max_fk_error,float(np.max(np.abs(tf-independent))))
                    tf=tf@primitive['origin']; g=primitive['geometry']; kind=primitive['type']
                    if kind=='box': size=list(g.size); shape=coal.Box(*size)
                    elif kind=='cylinder': size=[g.radius,g.length]; shape=coal.Cylinder(*size)
                    else: size=[g.radius]; shape=coal.Sphere(*size)
                    local=tf[:3,:3].T@(p-tf[:3,3]); d=primitive_sdf(kind,local,size)
                    pose=coal.Transform3s(tf[:3,:3],tf[:3,3]); pointpose=coal.Transform3s(np.eye(3),p)
                    check=coal.distance(shape,pose,coal.Sphere(tiny_radius),pointpose,req,coal.DistanceResult())+tiny_radius
                    max_coal_error=max(max_coal_error,abs(d-check))
                    voxel_d=coal.distance(shape,pose,coal.Box(*([mapcfg['resolution']]*3)),pointpose,req,coal.DistanceResult())
                    rows.append(dict(link=link,name=primitive['name'],type=kind,point_local=local.tolist(),
                        point_sdf_m=d,point_inflated_sdf_m=d-inflation,voxel_signed_distance_m=float(voxel_d),
                        ignored_by_selector=link in ('base_link','link1')))
                for link in samples:
                    if not link['include_for_risk'] or link['link_name'] in ('base_link','link1'): continue
                    tf=transforms[link['link_name']]
                    for i,s in enumerate(link['samples']):
                        v=tf[:3,:3]@np.asarray(s['center'])+tf[:3,3]
                        if any(v[j]<mapcfg[a+'_min'] or v[j]>=mapcfg[a+'_max'] for j,a in enumerate('xyz')): continue
                        v=v.astype(np.float32).astype(float)
                        d=float(np.float32(np.linalg.norm(v-p)-np.float32(s['radius']+inflation)))
                        proxy.append(dict(link=link['link_name'],sample_index=i,clearance_m=d,
                            source_collision_index=s['source_collision_index']))
                closest_proxy=min(proxy,key=lambda x:x['clearance_m'])
                max_proxy_error=max(max_proxy_error,abs(closest_proxy['clearance_m']-ref['clearance_m']))
                assert closest_proxy['link']==ref['link'] and closest_proxy['sample_index']==ref['sample_index']
                eligible=[r for r in rows if not r['ignored_by_selector']]
                configurations.append(dict(label=label,q=q.tolist(),nearest_proxy=closest_proxy,
                    nearest_primitive_all=min(rows,key=lambda r:r['point_sdf_m']),
                    nearest_primitive_eligible=min(eligible,key=lambda r:r['point_sdf_m']),
                    nearest_voxel_eligible=min(eligible,key=lambda r:r['voxel_signed_distance_m']),
                    primitive_rows=rows))
            bounds=list(zip(data['x_l'],data['x_u']))
            physical=list(bounds); lim=cfg['mpc']['joint_position_limits']; margin=cfg['mpc']['joint_position_margin']
            for i in range(len(center)*7):
                v=lim[JOINTS[i%7]]; physical[i]=(v['lower']+margin,v['upper']-margin)
            row=conflict['row']
            results.append(dict(model=case['model'],case_id=case['case_id'],plan_sequence=case['plan_sequence'],
                snapshot=case['snapshot'],point=p.tolist(),timestep=k,trust_radius=radius,
                learned_d=data['distance'][pair],configurations=configurations,
                original_box_support=row_support(data,row,bounds,False),
                original_box_and_dynamics_support=row_support(data,row,bounds,True),
                diagnostic_physical_limits_only_support=row_support(data,row,physical,False),
                diagnostic_no_trust_physical_limits_and_dynamics_support=row_support(data,row,physical,True)))
            print(case['model'],case['plan_sequence'],'eligible primitive',configurations[-1]['nearest_primitive_eligible'],flush=True)
    assert results
    assert max_fk_error<1e-10 and max_proxy_error<1e-7 and max_coal_error<1e-6,(max_fk_error,max_proxy_error,max_coal_error)
    out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
    report=dict(status='PASS',diagnostic_only=True,source=str(args.diagnosis),source_sha256=sha(args.diagnosis),
        urdf_sha256=sha(urdf),pinocchio_version=pin.__version__,coal_version=coal.__version__,
        body_sample_coverage=coverage_audit(robot,samples),
        crosschecks=dict(max_independent_fk_error=max_fk_error,max_proxy_reproduction_error_m=max_proxy_error,
                        max_analytic_vs_coal_point_error_m=max_coal_error),cases=results,
        limitations=['URDF collision primitives are not verified physical/training ground truth.',
            'GCDF checkpoint has weight snapshots only; its training geometry/normalization are not established.',
            'Voxel overlap means geometric intersection with UNKNOWN cell extent, not detected occupied matter.',
            'Inflation subtraction is an offline Euclidean offset diagnostic, not a runtime configuration change.',
            'No-trust LP removes only q trust bounds for attribution; no resulting trajectory is published or certified.'])
    with (out/'report.json').open('x') as f: json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__': main()
