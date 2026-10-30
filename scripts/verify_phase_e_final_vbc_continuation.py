#!/usr/bin/env python3
"""Same-input production callback checks on old/fixed built planner objects.

Uses a private ROS master, manual callback delivery, and no robot/actuator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
import xmlrpc.client

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'outputs/phase_e_final_vbc_continuation_fix_20261012'
    flags=(ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/flags.make').read_text()
    includes=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1])
    link=shlex.split((ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/link.txt').read_text())
    libs=link[link.index('-o')+2:]
    header=ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp'
    (out/'planner_diagnostic_access.hpp').write_text(header.read_text().replace('private:','public:'))
    test=ROOT/'scripts/test_phase_e_final_vbc_continuation.cpp'
    obj=ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src'
    object_paths=[]
    with (out/'compile.log').open('x') as log:
        for variant,directory in [('before',evidence/'before_objects'),('fixed',obj)]:
            objects=[directory/'local_sparse_scp_planner.cpp.o',directory/'primitive_observation_guard.cpp.o']
            object_paths.extend(objects)
            subprocess.run(['c++','-std=c++14','-O2',*includes,'-I'+str(out),str(test),
                            *map(str,objects),'-o',str(out/variant),*libs],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    env=os.environ.copy()
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_HOSTNAME='127.0.0.1',
               ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    report=dict(status='FAIL',scope='Full production finalVerificationCallback, jointStateCallback, requestPlanLocked, selectRepairTargetLocked and external Bool callback, linked old/fixed objects. Controlled callback order; no live scheduling, planner worker, CDF/geometry certification or robot execution.',checks=[])
    with (out/'master.log').open('x') as log:
        master=subprocess.Popen(['roscore','-p',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+15
            while True:
                try:
                    if xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/continuation_driver')[0]==1:break
                except OSError:pass
                if time.monotonic()>deadline:raise TimeoutError('Private ROS master failed to start')
                time.sleep(.05)
            for variant in ['before','fixed']:
                with (out/(variant+'.log')).open('x') as stream:
                    subprocess.run([str(out/variant),str(out/(variant+'.json')),variant],env=env,
                                   stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=30)
                report[variant]=json.loads((out/(variant+'.json')).read_text())
            report['checks']=[r['name'] for r in report['fixed']['checks']]
            report['status']='PASS'
        finally:
            if master.poll() is None:
                os.killpg(master.pid,signal.SIGTERM)
                try:master.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(master.pid,signal.SIGKILL);master.wait(timeout=5)
            report['owned_master_return_code']=master.returncode
            inputs=[test,header,Path(__file__).resolve(),*object_paths]
            report['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(report['checks']),output=str(out)),indent=2))


if __name__=='__main__':
    main()
