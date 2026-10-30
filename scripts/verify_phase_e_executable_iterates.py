#!/usr/bin/env python3
"""Actual before/after worker and PIQP; private ROS, no robot or actuator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import time
import xmlrpc.client

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'outputs/phase_e_case001_feasible_iterate_fix_20261012'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--reuse-compiled',action='store_true')
    args=ap.parse_args();out=args.output_dir.resolve()
    if args.reuse_compiled:
        assert out.exists() and all((out/v/'test').is_file() for v in ['before','after'])
    else:out.mkdir(parents=True,exist_ok=False)
    cpp=Path('src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp')
    hpp=Path('src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp')
    flags=(ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/flags.make').read_text()
    inc=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1])
    defines=shlex.split(re.search(r'^CXX_DEFINES = (.*)$',flags,re.M)[1])
    link=shlex.split((ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/link.txt').read_text())
    libraries=link[link.index('-o')+2:]
    objects=ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src'
    if not args.reuse_compiled:
        with (out/'compile.log').open('x') as log:
            for variant in ['before','after']:
                folder=out/variant;folder.mkdir()
                source=EVIDENCE/'before' if variant=='before' else ROOT
                header=(source/hpp).read_text()
                (folder/'planner_diagnostic_access.hpp').write_text(header.replace('private:','public:'))
                if variant=='before':
                    shim=folder/'include/egocentric_arm_planner';shim.mkdir(parents=True)
                    (shim/'local_sparse_scp_planner.hpp').write_text(header)
                    object_file=folder/'planner.o'
                    assert sha(source/cpp)==json.load(open(EVIDENCE/'before_sha256.json'))[str(cpp)]
                    subprocess.run(['c++','-std=c++14','-O3',*defines,'-I'+str(folder/'include'),*inc,
                        '-c',str(source/cpp),'-o',str(object_file)],stdout=log,stderr=subprocess.STDOUT,check=True)
                else:object_file=objects/'local_sparse_scp_planner.cpp.o'
                subprocess.run(['c++','-std=c++14','-O2','-DCARE_NEW='+str(int(variant=='after')),*inc,'-I'+str(folder),
                    str(ROOT/'scripts/test_phase_e_executable_iterates.cpp'),str(object_file),
                    str(objects/'primitive_observation_guard.cpp.o'),'-o',str(folder/'test'),*libraries],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/cpp,ROOT/hpp,ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/repair_executable_iterates.hpp',
            ROOT/'scripts/test_phase_e_executable_iterates.cpp',Path(__file__).resolve(),
            out/'before/test',out/'after/test',objects/'local_sparse_scp_planner.cpp.o',objects/'primitive_observation_guard.cpp.o']}
        (out/'compiled_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.compile_only:
        print('COMPILE_PASS');return
    manifest=json.load(open(out/'compiled_sha256.json'))
    assert all(sha(ROOT/p)==digest for p,digest in manifest.items()),'compiled inputs changed'
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=dict(os.environ,ROS_MASTER_URI=f'http://127.0.0.1:{port}',ROS_HOSTNAME='127.0.0.1',
        ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[];streams=[]
    report=dict(status='FAIL',qualification='Actual before/after production worker and PIQP using recorded CASE001 q/u, controlled safe/hostile CDF rows and tracking objective. The retained earlier hard output is explicitly seeded. No original full-QP/map replay or execution certificate.')
    def start(command,path):
        stream=path.open('x');streams.append(stream)
        proc=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(proc);return proc
    try:
        master=start(['roscore','-p',str(port)],out/'master.log');end=time.monotonic()+10
        while True:
            try:
                if xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/iterate_driver')[0]==1:break
            except OSError:pass
            assert master.poll() is None and time.monotonic()<end
            time.sleep(.05)
        os.environ.update(env)
        import rospy,yaml
        rospy.init_node('iterate_validation_driver',disable_signals=True)
        config=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        config['local_planner']['primitive_observation_guard']=dict(enabled=True,
            fk_urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
            primitive_urdf=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            base_frame='base_link',ignored_links=['base_link','link1'],margin_m=.02)
        config['local_planner']['repair_executable_view']=dict(enabled=True,prefix_s=.25,cycle_enabled=True,
            cycle_prefixes_s=[.25,.15,.10,.05],brake_dt_s=.05,max_brake_steps=20,acceleration_limits=[3.,3.,4.,4.,6.,6.,6.])
        rospy.set_param('/feasible_iterate_test',config)
        from std_msgs.msg import Bool
        arm=[];sub=rospy.Subscriber('/care_planner/task_start',Bool,lambda msg:arm.append(msg.data))
        fixture=ROOT/'outputs/phase_e_case001_geometry_analysis_20261012/fixtures.json'
        for variant in ['before','after']:
            folder=out/variant
            proc=start([str(folder/'test'),str(fixture),str(folder/'results.json')],folder/'run.log')
            assert proc.wait(timeout=30)==0,variant
            report[variant]=json.load(open(folder/'results.json'))
        before=report['before']['scenarios'];after=report['after']['scenarios']
        assert before[0]['restore_inf_rad']>.001 and after[0]['restore_inf_rad']<1e-12
        assert before[0]['candidates']==after[0]['candidates']==0
        assert after[0]['step_scale']==.5 and before[0]['step_scale']==1.
        assert not before[1]['reserved'] and after[1]['candidates']==1
        assert after[1]['reserved'] and after[1]['consumed']==6 and not after[1]['second_query']
        assert after[2]['candidates']==after[3]['candidates']==0
        assert not after[3]['first_query'] and not arm
        report.update(status='PASS',arm_messages=len(arm),checks=[
            'Same failed trial: old center overwritten; new candidate restored and step halved without publication',
            'Original final iteration: new reserves a fresh fixed full hard-QP recheck; old continues optimizing the overwritten center',
            'Fresh hostile80CDF hard rows still reject the saved candidate; no slack',
            'No saved iterate at final iteration: hold without seventh query',
            'Changed start/identity and exhausted reservation cannot reuse the ledger',
            'Zero task-arm messages; no robot or actuator process started'])
        report['qualifications']=[
            'Both versions emit a candidate in the controlled final-positive scenario. Its different CDF/objective does not reproduce the original six-step terminal failure.',
            'The fixed before/after evidence isolates center preservation/step reduction/reservation and fresh rejection, not whole CASE001 success.']
    finally:
        cleanup=[]
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
            if proc.poll() is None:cleanup.append(proc.pid)
        for stream in streams:stream.close()
        report['cleanup_remaining']=cleanup
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=report.get('checks'),arm_messages=report.get('arm_messages'),cleanup_remaining=report['cleanup_remaining']),indent=2))

if __name__=='__main__':main()
