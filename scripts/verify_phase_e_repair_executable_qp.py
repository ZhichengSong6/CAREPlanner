#!/usr/bin/env python3
"""Same-input before/after real production PIQP; private ROS without actuators."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import shutil
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
    artifact=ROOT/'outputs/phase_e_executable_view_fix_20261010'
    flags=(ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/flags.make').read_text()
    inc=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1])
    defines=shlex.split(re.search(r'^CXX_DEFINES = (.*)$',flags,re.M)[1])
    link=shlex.split((ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/link.txt').read_text())
    libs=link[link.index('-o')+2:]
    obj=ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src'
    header_path=Path('src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp')
    cpp_path=Path('src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp')
    with (out/'compile.log').open('x') as log:
        for variant in ['before','after']:
            folder=out/variant;folder.mkdir()
            source=artifact/'before' if variant=='before' else ROOT
            header=(source/header_path).read_text();(folder/'planner_diagnostic_access.hpp').write_text(header.replace('private:','public:'))
            if variant=='before':
                shim=folder/'include/egocentric_arm_planner';shim.mkdir(parents=True)
                (shim/'local_sparse_scp_planner.hpp').write_text(header)
                object_file=folder/'planner.o'
                cached=artifact/'fixed_ros/before/planner.o'
                if cached.exists():
                    import hashlib
                    expected=json.loads((artifact/'before_sha256.json').read_text())[str(cpp_path)]
                    assert hashlib.sha256((source/cpp_path).read_bytes()).hexdigest()==expected
                    shutil.copy2(cached,object_file)
                else:
                    subprocess.run(['c++','-std=c++14','-O3',*defines,'-I'+str(folder/'include'),*inc,
                        '-c',str(source/cpp_path),'-o',str(object_file)],stdout=log,stderr=subprocess.STDOUT,check=True)
            else:object_file=obj/'local_sparse_scp_planner.cpp.o'
            subprocess.run(['c++','-std=c++14','-O2','-DCARE_NEW='+str(int(variant=='after')),*inc,'-I'+str(folder),
                str(ROOT/'scripts/test_phase_e_repair_executable_qp.cpp'),str(object_file),str(obj/'primitive_observation_guard.cpp.o'),
                '-o',str(folder/'test'),*libs],stdout=log,stderr=subprocess.STDOUT,check=True)
    env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_HOSTNAME='127.0.0.1',
             ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[];streams=[];report=dict(status='FAIL',qualification='Real production PIQP with frozen CASE028 q/u and controlled safe CDF rows/objective; no map or original full successful QP reconstruction.')
    def start(argv,path):
        stream=path.open('x');streams.append(stream)
        proc=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);processes.append(proc);return proc
    try:
        master=start(['roscore','-p',str(port)],out/'master.log');end=time.monotonic()+10
        while True:
            try:
                if xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/executable_test')[0]==1:break
            except OSError:pass
            assert master.poll() is None and time.monotonic()<end
            time.sleep(.05)
        os.environ.update(env)
        import rospy,yaml
        rospy.init_node('executable_validation_driver',disable_signals=True)
        config=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        config['local_planner']['primitive_observation_guard']=dict(enabled=True,
            fk_urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),primitive_urdf=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            base_frame='base_link',ignored_links=['base_link','link1'],margin_m=.02)
        rospy.set_param('/repair_executable_test',config)
        for variant in ['before','after']:
            folder=out/variant
            fixture=artifact/'boundary_qp_fixture.json'
            if not fixture.exists():fixture=artifact/'math/qp_fixture.json'
            proc=start([str(folder/'test'),str(fixture),str(folder/'results.json')],folder/'run.log')
            assert proc.wait(timeout=30)==0
            results=json.loads((folder/'results.json').read_text());report[variant]=results
            assert all(r['cdf_rows']==20 and r['max_slack']==0. for r in results)
        assert report['before'][0]['solved']
        assert report['after'][-1]['solved'] and report['after'][-1]['executable_valid']
        assert report['after'][0]['executable_rows']>0
        fixture_data=json.loads(fixture.read_text())
        previous=fixture_data['u'];max_ratio=0.
        for result in report['after']:
            for k in [1,2,3,5]:
                for j,a in enumerate(fixture_data['acceleration']):
                    max_ratio=max(max_ratio,abs(result['u'][k][j]-previous[k][j])/(a*.05))
            previous=result['u']
        assert max_ratio<=1.001,max_ratio
        report['max_endpoint_velocity_update_in_brake_quanta']=max_ratio
        worker=json.loads((out/'after/results.json.worker.json').read_text())
        assert all(r['initial_solved'] and not r['initial_executable_valid'] and r['candidates']==0 for r in worker)
        assert worker[0]['iteration_after']==6 and worker[1]['iteration_after']==5
        assert worker[2]['iteration_after']==6 and worker[2]['replacement_pending']
        assert not worker[0]['replacement_pending'] and not worker[1]['replacement_pending']
        report['worker']=worker
        boundary=out/'after/results.json.boundary.json'
        if boundary.exists():report['fixed_start_boundary']=json.loads(boundary.read_text())
        report['status']='PASS'
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
        for stream in streams:stream.close()
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:([{a:b for a,b in r.items() if a not in ('q','u')} for r in v] if isinstance(v,list) else v) for k,v in report.items()}))


if __name__=='__main__':main()
