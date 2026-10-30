#!/usr/bin/env python3
"""Exact saved final-GCDF rows and same-q URDF/sphere diagnostics, no ROS writes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src/care_collision_cdf/scripts'))
sys.path.insert(0,str(REPO/'src/care_visibility_cdf/scripts'))
from audit_phase_e_rejection_snapshots import validate_record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    import torch
    import pinocchio as pin
    import yaml
    from collision_cdf_model import CollisionCDF
    from check_visibility_self_occlusion import load_collision_primitives
    from validate_visibility_oracle import URDF
    from diagnose_phase_e_point_geometry import primitive_sdf
    checkpoint=REPO/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(checkpoint)=='41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70'
    if a.device=='cuda':assert torch.cuda.is_available(), 'CUDA unavailable; no silent CPU substitution'
    torch.set_num_threads(1)
    net=CollisionCDF(str(checkpoint),device=a.device)
    urdf=REPO/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    robot=URDF.from_xml_file(str(urdf)); primitives=load_collision_primitives(robot)
    model=pin.buildModelFromUrdf(str(urdf));data=model.createData()
    joints=list(model.names)[1:];assert len(joints)==7
    samplepath=REPO/'src/care_confidence_map/config/body_samples.yaml'
    samples=yaml.safe_load(samplepath.read_text())['body_sampling']['links']
    def geometry(p,q,collision=None):
        pin.framesForwardKinematics(model,data,np.asarray(q,dtype=float))
        rows=[];spheres=[]
        for solid in primitives:
            if solid['link'] in ('base_link','link1'):continue
            pose=np.asarray(data.oMf[model.getFrameId(solid['link'])].homogeneous)@solid['origin']
            local=pose[:3,:3].T@(p-pose[:3,3]);g=solid['geometry'];kind=solid['type']
            size=list(g.size) if kind=='box' else [g.radius,g.length] if kind=='cylinder' else [g.radius]
            rows.append(dict(link=solid['link'],collision=solid['name'],sdf_m=primitive_sdf(kind,local,size)))
        for link in samples:
            if not link['include_for_risk'] or link['link_name'] in ('base_link','link1'):continue
            pose=np.asarray(data.oMf[model.getFrameId(link['link_name'])].homogeneous)
            for i,s in enumerate(link['samples']):
                center=pose[:3,:3]@np.asarray(s['center'])+pose[:3,3]
                spheres.append(dict(link=link['link_name'],sample_index=i,
                    sdf_m=float(np.linalg.norm(p-center)-s['radius'])))
        return dict(primitive=min(rows,key=lambda r:r['sdf_m']),samples=min(spheres,key=lambda r:r['sdf_m']),
            recorded_collision=next((r for r in rows if r['collision']==collision),None))
    def value(p,q):
        pt=torch.tensor(p,device=a.device,dtype=torch.float32).reshape(1,3)
        qt=torch.tensor(q,device=a.device,dtype=torch.float32).reshape(1,7)
        with torch.no_grad():return float(net.model(torch.cat((pt,qt),dim=1)).item())
    def stages(record,p,time_ns):
        views=[('query',record['local_context']['query_trajectory'])]+[
            (name,record[name]) for name in ['raw_trajectory','handoff_trajectory','audited_trajectory']]
        rows=[]
        for name,traj in views:
            points=[pt for pt in traj['points'] if pt['time_ns']==time_ns]
            if len(points)!=1:
                rows.append(dict(stage=name,status='NO_UNIQUE_EXACT_TIME'));continue
            q=points[0]['positions'];geom=geometry(p,q)
            rows.append(dict(stage=name,status='exact_time',time_ns=time_ns,q=q,
                learned_distance=value(p,q),geometry=geom))
        return rows
    results=[];inputs=[];sources={};maxd=maxg=0.
    for path in sorted((a.run/'rejection_snapshots').glob('*.json')):
        record=json.loads(path.read_text());validate_record(record);sources[str(path)]=digest(path)
        trajectory=record['audited_trajectory'];assert trajectory['joint_names']==joints
        batch=record['final_gcdf_batch'];n=int(batch['num_pairs']);details=[]
        if n:
            p=np.asarray(batch['point_flat']).reshape(n,3);q=np.asarray(batch['q_linearization_flat']).reshape(n,7)
            pt=torch.tensor(p,device=a.device,dtype=torch.float32)
            qt=torch.tensor(q,device=a.device,dtype=torch.float32).requires_grad_(True)
            f=net.model(torch.cat((pt,qt),dim=1)).reshape(-1)
            grad=torch.autograd.grad(f.sum(),qt)[0]
            d=f.detach().cpu().numpy();g=grad.detach().cpu().numpy()
            de=float(np.max(np.abs(d-np.asarray(batch['distance']))))
            ge=float(np.max(np.abs(g-np.asarray(batch['gradient_flat']).reshape(n,7))))
            maxd=max(maxd,de);maxg=max(maxg,ge)
            assert de<1e-5 and ge<2e-5,(path,de,ge)
            for i in range(n):
                k=batch['original_timestep'][i];cq=np.asarray(trajectory['points'][k]['positions'])
                assert np.array_equal(cq.astype(np.float32),q[i].astype(np.float32)), 'batch not evaluated at audited q'
                if batch['distance'][i]>=0:continue
                measured=[record['measured_at_dispatch'][j] for j in joints]
                local=record['local_context']['local_gcdf_batch']
                matches=[j for j in range(local['num_pairs']) if
                    np.array_equal(np.asarray(local['point_flat'][3*j:3*j+3],dtype=np.float32),p[i].astype(np.float32))]
                details.append(dict(pair=i,point=p[i].tolist(),timestep=k,
                    time_ns=trajectory['points'][k]['time_ns'],candidate_q=cq.tolist(),
                    learned_distance=float(d[i]),gradient=g[i].tolist(),source_type=batch['source_type'][i],
                    measured_geometry=geometry(p[i],measured),candidate_geometry=geometry(p[i],cq),
                    exact_time_stages=stages(record,p[i],trajectory['points'][k]['time_ns']),
                    local_same_point_rows=[dict(pair=j,timestep=local['original_timestep'][j],
                        distance=local['distance'][j],q=local['q_linearization_flat'][7*j:7*j+7]) for j in matches]))
        vbc_details=[]
        for v in record.get('vbc_evidence',{}).get('violations',[]):
            k=v['sweep_original_k'];p=np.asarray(v['point']);q=trajectory['points'][k]['positions']
            geom=geometry(p,q,v.get('source_collision_name'))
            if v.get('primitive_signed_distance_m') is not None:
                assert abs(geom['recorded_collision']['sdf_m']-v['primitive_signed_distance_m'])<1e-9
            matches=[i for i in range(n) if batch['original_timestep'][i]==k and
                np.array_equal(np.asarray(batch['point_flat'][3*i:3*i+3],dtype=np.float32),p.astype(np.float32))]
            vbc_details.append(dict(evidence=v,candidate_q=q,geometry=geom,
                independently_evaluated_gcdf=value(p,q),same_point_step_final_batch_rows=matches,
                note='Missing batch row is not a GCDF clearance certificate.'))
        results.append(dict(snapshot=str(path),safety_gate=record['safety_gate'],pairs=n,
            violations=details,vbc_violations=vbc_details))
        inputs.append(dict(id=len(inputs),case_id=a.run.parent.name,recorded_backend='primitive',role='rejected',
            stamp_ns=trajectory['stamp_ns'],joint_names=joints,
            points=[dict(q=p['positions'],time_ns=int(p['time_ns'])) for p in trajectory['points']]))
    assert inputs, 'No validated actual snapshots; replay NOT_RUN'
    report=dict(status='PASS',snapshots=results,max_distance_error=maxd,max_gradient_error=maxg,
        source_sha256=sources,checkpoint_sha256=digest(checkpoint),urdf_sha256=digest(urdf),
        samples_sha256=digest(samplepath),device=a.device,
        limitations=['No full map snapshot: full VBC replay NOT_RUN.',
            'Primitive/sphere SDF compares bare geometry; no inflation or safety threshold changed.',
            'Same-point local rows may have different q/t; not a replay of the original QP matrix.',
            'Reproducing rejection is PASS for diagnosis, not task success.'])
    for name,payload in [('result.json',report),('inputs.json',dict(trajectories=inputs,
        grid=dict(min=[-.95,-.95,0.],max=[.95,.95,1.15],resolution=.05),
        max_eval_timesteps=50,fallback_dt=.05,margins=[0.,.01,.02]))]:
        with (a.output_dir/name).open('x') as f:json.dump(payload,f,indent=2,allow_nan=False)
    print('PASS',len(inputs),'snapshots; GCDF errors',maxd,maxg)


if __name__=='__main__':main()
