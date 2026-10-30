#!/usr/bin/env python3
"""Compare offline radii with the built production VbcPrimitiveModel, all solids."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from audit_case001_motion_prior import ROOT,URDF,load_shapes
from phase_e_stop_envelope import primitive_radii


def main():
    out=ROOT/'outputs/phase_e_stop_envelope_20261001/geometry_parity'
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'scripts/dump_phase_e_primitive_stop_radii.cpp'
    library=ROOT/'devel/lib/libtrajectory_risk_evaluator.so'
    binary=out/'dump_radii'
    env=dict(os.environ)
    # Keep the probe on the ROS build's ABI, not the Python/conda Pinocchio ABI.
    env['LD_LIBRARY_PATH']=str(ROOT/'devel/lib')+':/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu'
    command=['g++','-std=c++14','-O0','-DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION','-DPINOCCHIO_WITH_URDFDOM',
        '-DPINOCCHIO_WITH_HPP_FCL','-DBOOST_MPL_LIMIT_LIST_SIZE=30','-DBOOST_MPL_LIMIT_VECTOR_SIZE=30',
        '-I'+str(ROOT/'src/care_confidence_map/include'),'-I'+str(ROOT/'devel/include'),
        '-I/opt/ros/noetic/include','-I/usr/include/eigen3',str(source),'-o',str(binary),
        '-L'+str(ROOT/'devel/lib'),'-L/opt/ros/noetic/lib','-ltrajectory_risk_evaluator','-lpinocchio_default','-lroscpp','-lrosconsole','-lrostime']
    subprocess.run(command,env=env,check=True)
    raw=subprocess.check_output([str(binary),str(URDF)],env=env,text=True)
    (out/'production_radii.jsonl').write_text(raw)
    actual={(r['link'],r['name']):r['radii'] for r in map(json.loads,raw.splitlines())}
    shapes=load_shapes(ET.parse(URDF).getroot()); expected=primitive_radii(URDF,shapes)
    assert len(actual)==len(shapes)==26
    differences=[]
    for i,s in enumerate(shapes):
        differences.extend(np.abs(np.array(actual[s['link'],s['name']])-expected[i]))
    assert max(differences)<1e-12
    result=dict(status='PASS',primitives=26,joint_coefficients=182,max_error_m=float(max(differences)),
                build_argv=command,input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [source,library,URDF,ROOT/'scripts/phase_e_stop_envelope.py']})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('build_argv','input_sha256')},indent=2))


if __name__=='__main__': main()
