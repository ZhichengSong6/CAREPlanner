#!/usr/bin/env python3
"""Offline current-URDF body-sample validation and before/after impact audit.

Uses the real production C++ loader/FK helper, without ROS or map writes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src/care_visibility_cdf/scripts'))
from validate_visibility_oracle import URDF, find_chain_joints, fk_transform, DEFAULT_JOINT_NAMES

PARAMS=('cylinder_spacing','cylinder_safety_margin','cylinder_radius_scale',
        'cylinder_center_span_scale','box_cell_size','box_safety_margin','box_radius_scale')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(doc,urdf):
    body=doc['body_sampling']; root=ET.parse(urdf).getroot()
    assert body['source_urdf_sha256']==sha(urdf),'stale URDF digest'
    links={l.attrib['name']:l for l in root.findall('link')}
    total=risk=0; covered=set(); seen=set()
    for item in body['links']:
        name=item['link_name']; assert name not in seen; seen.add(name)
        assert item['frame']==name and name in links
        assert item['include_for_risk']==(name!='base_link')
        collisions=links[name].findall('collision')
        for s in item['samples']:
            assert len(s['center'])==3 and np.isfinite(s['center']).all()
            assert np.isfinite(s['radius']) and s['radius']>0
            index=s['source_collision_index']; assert 0<=index<len(collisions)
            c=collisions[index]; assert s['source_collision_name']==c.attrib.get('name','')
            assert s['source_type']==list(c.find('geometry'))[0].tag
            covered.add((name,index)); total+=1; risk+=int(item['include_for_risk'])
    expected={(n,i) for n,l in links.items() for i,_ in enumerate(l.findall('collision'))}
    assert covered==expected,('missing collisions',expected-covered)
    assert total==body['total_samples'] and risk==body['total_risk_samples']
    return dict(total=total,risk=risk,collision_count=len(covered),
        per_link={l['link_name']:len(l['samples']) for l in body['links']})


def static_mask(doc,robot,grid,q,inflations,ignored=()):
    mask=np.zeros(len(grid),dtype=bool); qmap=dict(zip(DEFAULT_JOINT_NAMES,q))
    for link in doc['body_sampling']['links']:
        if link['link_name'] in ignored: continue
        tf=fk_transform(find_chain_joints(robot,'base_link',link['frame']),qmap)
        for s in link['samples']:
            center=tf[:3,:3]@s['center']+tf[:3,3]
            radius=s['radius']+inflations.get(link['link_name'],0.)
            mask|=np.sum((grid-center)**2,axis=1)<=radius**2
    return mask


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--body-binary',default='/tmp/care_repair_body_clearance')
    args=ap.parse_args()
    urdf=REPO/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    current=REPO/'src/care_confidence_map/config/body_samples.yaml'
    generator=REPO/'src/care_confidence_map/scripts/generate_body_samples.py'
    before=yaml.safe_load(args.before.read_text()); after=yaml.safe_load(current.read_text())
    checks=[]; counts=validate(after,urdf); checks.append('all current URDF primitives represented with valid frame/type/index/name/hash')
    for key in PARAMS+('cylinder_endpoint_sampling',):
        assert before['body_sampling'][key]==after['body_sampling'][key],key
    assert {l['link_name']:l['include_for_risk'] for l in before['body_sampling']['links']}=={
        l['link_name']:l['include_for_risk'] for l in after['body_sampling']['links']}
    checks.append('all seven generation parameters, endpoint policy and per-link risk flags preserved')
    with tempfile.TemporaryDirectory(prefix='care_body_generation_') as tmp:
        generated=Path(tmp)/'samples.yaml'
        command=[sys.executable,str(generator),'--urdf',str(urdf),'--output',str(generated)]
        for key in PARAMS: command+=['--'+key.replace('_','-'),str(after['body_sampling'][key])]
        subprocess.run(command,check=True,capture_output=True,text=True)
        assert generated.read_bytes()==current.read_bytes(),'generation is not byte-reproducible'
    checks.append('byte-identical regeneration with original sampling parameters')
    for fault in ('hash','missing','type','name','radius'):
        bad=copy.deepcopy(after); s=bad['body_sampling']['links'][2]['samples'][0]
        if fault=='hash': bad['body_sampling']['source_urdf_sha256']='invalid'
        elif fault=='missing': bad['body_sampling']['links'][2]['samples']=[]
        elif fault=='type': s['source_type']='invalid'
        elif fault=='name': s['source_collision_name']='invalid'
        else: s['radius']=float('nan')
        try: validate(bad,urdf)
        except AssertionError: pass
        else: raise AssertionError('fault was not rejected: '+fault)
    checks.append('stale hash/missing primitive/wrong type/wrong name/nonfinite radius rejected')
    diag=REPO/'outputs/phase_e_repair_conflict_diagnosis/20260911-witness-points/report.json'
    source=json.loads(diag.read_text()); queries=[]; labels=[]
    for case in source['cases']:
        for c in case['conflicts']:
            for label,key in [('measured','q0'),('linearization','failure_qbar')]:
                queries.append(dict(point=c['grid_point'],q=c[key])); labels.append(dict(model=case['model'],plan=case['plan_sequence'],configuration=label))
    config=yaml.safe_load((REPO/'src/care_confidence_map/config/confidence_map_phase_e_ray.yaml').read_text())
    m=config['confidence_map']; prior=config['current_body_prior']
    assert m['frame']=='base_link'
    assert prior['lock_after_complete_refresh'] and not prior['refresh_on_startup'] and not prior['risk_samples_only']
    results=[]
    for path in (args.before,current):
        payload=dict(urdf=str(urdf),body_samples=str(path.resolve()),ignored_links=['base_link','link1'],
            body_inflation_m=.015,map_lower=[m[a+'_min'] for a in 'xyz'],map_upper=[m[a+'_max'] for a in 'xyz'],queries=queries)
        proc=subprocess.run([args.body_binary],input=json.dumps(payload),check=True,capture_output=True,text=True)
        values=json.loads(proc.stdout); assert len(values)==len(queries)
        assert all(np.isfinite(x['clearance_m']) for x in values); results.append(values)
    checks.append('unchanged production C++ YAML loader/FK accepts old and new data for 10 historical queries')
    grid=np.array(np.meshgrid(*[m[a+'_min']+np.arange(int(np.floor((m[a+'_max']-m[a+'_min'])/m['resolution']))+1)*m['resolution'] for a in 'xyz'],indexing='ij')).reshape(3,-1).T
    robot=URDF.from_xml_file(str(urdf)); masks=[]; vbc=[]
    for doc in (before,after):
        masks.append(static_mask(doc,robot,grid,np.zeros(7),prior['link_inflation_radius']))
        vbc.append(static_mask(doc,robot,grid,np.zeros(7),{},ignored=('base_link','link1')))
    def comparison(pair):
        old,new=pair
        return dict(old_count=int(old.sum()),new_count=int(new.sum()),added=int((new&~old).sum()),removed=int((old&~new).sum()))
    checks.append('offline q=0 startup-prior and uninflated static VBC footprint impact quantified; no map writes')
    out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
    report=dict(status='PASS',checks=checks,new_counts=counts,
        old_counts={l['link_name']:len(l['samples']) for l in before['body_sampling']['links']},
        body_sha256_before=sha(args.before),body_sha256_after=sha(current),urdf_sha256=sha(urdf),
        generator_sha256=sha(generator),body_helper_sha256=sha(args.body_binary),
        diagnostic_queries_sha256=sha(diag),q0_startup_prior_footprint=comparison(masks),
        q0_static_vbc_footprint=comparison(vbc),
        point_distances=[dict(label,**query,before=a,after=b) for label,query,a,b in zip(labels,queries,*results)],
        limitations=['Primitive representation does not certify complete sphere-union enclosure.',
            'Footprint counts are offline q=0 geometry, not a new simulator confidence map or task result.',
            'Unchanged startup inflation constants were not globally re-certified for the new samples.',
            'No ROS/Gazebo/robot or full closed-loop VBC/bootstrap test.'])
    with (out/'result.json').open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('point_distances','limitations')},indent=2))


if __name__=='__main__': main()
