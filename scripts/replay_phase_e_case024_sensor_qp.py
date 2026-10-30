#!/usr/bin/env python3
"""Fixed final subproblem through production C++/PIQP, not full SCP/map replay."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import xmlrpc.client
import numpy as np
from diagnose_phase_e_case006_o6_recovery import strings

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase_e_case024_owner_fallback_20261009'
RUN=ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_024_r1_case024_frontier_direction_20261009/run'
def main():
    # Fail before allocating a run if the sandbox denies a private ROS master.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    probe=json.loads((ROOT/'outputs/phase_e_case024_o6_sensor_probe_20261009/report.json').read_text())
    def rows(name):
        with (RUN/(name+'.csv')).open() as f:return list(csv.DictReader(f))
    def matrix(r,field,n):
        return [[float(r[f'field.points{k}.{field}{j}']) for j in range(7)] for k in range(n)]
    query=next(r for r in rows('local_gcdf_query') if r['field.header.stamp']=='20411000000')
    raw=next(r for r in rows('raw_candidate_trajectory') if r['field.header.stamp']=='20444000000')
    ws=sorted([r for r in strings(RUN/'local_witness_selector_diagnostics.csv') if r['query_stamp_ns']=='20411000000'],key=lambda r:int(r['batch_pair']))
    assert [int(r['batch_pair']) for r in ws]==list(range(200))
    margins={}
    log=(ROOT/'logs/c5_5_vbc_gcdf_regime/phase_e_goal_024_r1_case024_frontier_direction_20261009/run/controlled.log').read_text()
    for line in log.splitlines():
        if 'exact VBC point feedback point=[' not in line and 'point_margin_rollback point=[' not in line:continue
        # Anchor to the feedback message, even when two console records interleave.
        timestamp=float(re.findall(r', ([\d.]+)\]',line)[-1])
        if timestamp>20.411:continue
        pt=tuple(round(float(v),5) for v in re.search(r'point=\[([^]]+)\]',line)[1].split())
        margins[pt]=float(re.search(r'(?:to=|cdf_margin=)([\d.e+-]+)',line)[1])
    points=sorted({tuple(json.loads(w['point'])) for w in ws})
    witnesses=[dict(point=p,margin=max(.05,margins.get(tuple(round(v,5) for v in p),.05))) for p in points]
    fs=[r for r in rows('visibility_frontier_target') if int(r['%time'])<=20354000000]
    f=fs[-1]
    previous=next(r for r in reversed(rows('actuator_command')) if int(r['%time'])<=20354000000)
    base=dict(id='original_S7',qbar=matrix(query,'positions',21),ubar=matrix(query,'velocities',20),
        previous=[float(previous['field.data'+str(j)]) for j in range(7)],radius=.2,attempt=0,
        frontier=[float(f['field.data'+str(j)]) for j in range(3,10)],mask=[1.]*7,
        witnesses=witnesses,rows=[dict(k=int(w['timestep']),point=json.loads(w['point']),d=float(w['distance']),
            g=json.loads(w['gradient']),qlin=json.loads(w['q_linearization'])) for w in ws])
    cases=[base]
    for sid in (0,4):
        b=probe['branches'][sid];c=copy.deepcopy(base);c.update(id='S'+str(sid),attempt=2,mask=b['mask'],
            frontier=(np.array(probe['seed'])+np.array(b['direction']['step'])).tolist());cases.append(c)
    (OUT/'qp_inputs.json').write_text(json.dumps(cases,indent=2)+'\n')
    uri='http://127.0.0.1:'+str(port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='localhost',ROS_HOME=str(OUT/'qp_ros_home'),ROS_LOG_DIR=str(OUT/'qp_ros_logs'))
    children=[]
    try:
        with (OUT/'qp_master.log').open('w') as log:
            master=subprocess.Popen(['roscore','-p',str(port)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(master)
        for _ in range(100):
            try:
                if xmlrpc.client.ServerProxy(uri).getPid('/sensor_qp')[0]==1:break
            except OSError:pass
            assert master.poll() is None;time.sleep(.05)
        else:raise RuntimeError('master timeout')
        import rospy,yaml
        rospy.init_node('sensor_qp_setup',disable_signals=True)
        cfg=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        cfg['urdf_path']=str(ROOT/'src/arm_description/urdf/Arm.urdf')
        rospy.set_param('/case024_sensor_qp',cfg)
        rospy.set_param('/robot_description',(ROOT/'src/arm_description/urdf/Arm.urdf').read_text())
        with (OUT/'qp_cpp.log').open('w') as log:
            p=subprocess.Popen([str(OUT/'sensor_qp'),str(OUT/'qp_inputs.json'),str(OUT/'qp_results.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(p)
            assert p.wait(timeout=30)==0
        result=json.loads((OUT/'qp_results.json').read_text())
        err=float(np.max(abs(np.array(result[0]['q'])-matrix(raw,'positions',21))))
        assert err<1e-6,('original candidate reproduction',err)
        assert all(r['status']=='solved' and r['rows']==200 and r['max_slack']==0 for r in result)
        summary=dict(status='PASS',original_candidate_q_error_inf=err,results=[{k:v for k,v in r.items() if k not in ('q','u')} for r in result],
            scope='Production hard-QP final subproblem with all 200 saved rows, point margins, dynamics and bounds. Alternative recovery attempt=2. Not an SCP relinearization loop or full map/VBC replay.',
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),Path(__file__).resolve().with_suffix('.cpp'),OUT/'sensor_qp',OUT/'qp_inputs.json',ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src/local_sparse_scp_planner.cpp.o']})
        (OUT/'qp_validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
if __name__=='__main__':main()
