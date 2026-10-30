#!/usr/bin/env python3
"""One bounded stationary Gazebo sensor check; no planner, goal or case score."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xmlrpc.client
from phase_e_owned_processes import remember_descendants, reap_owned
from phase_e_sim_environment import simulation_environment

ROOT=Path(__file__).resolve().parents[1]


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    env=simulation_environment({})
    ros_port,gazebo_port=port(),port()
    while gazebo_port==ros_port: gazebo_port=port()
    env.update(ROS_MASTER_URI=f'http://127.0.0.1:{ros_port}',ROS_HOSTNAME='localhost',
               GAZEBO_MASTER_URI=f'http://127.0.0.1:{gazebo_port}',GAZEBO_MODEL_DATABASE_URI='',
               ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(dict(environment=env,task_goals=0,case_runs=0,
        scope=__doc__,max_wall_s=120),indent=2))
    launch=out/'perception.launch'
    launch.write_text('''<launch>
  <include file="$(find arm_description)/launch/gazebo_velocity_control.launch">
    <arg name="world_file" value="$(find arm_description)/worlds/maixsense_empty.world"/>
    <arg name="gazebo_gui" value="false"/><arg name="use_rviz" value="false"/>
    <arg name="apply_initial_joint_overrides" value="false"/>
  </include>
  <include file="$(find care_confidence_map)/launch/tof_fusion_self_filter.launch"/>
  <include file="$(find care_confidence_map)/launch/confidence_map.launch">
    <arg name="config_file" value="$(find care_confidence_map)/config/confidence_map_phase_e_ray.yaml"/>
    <arg name="body_prior_geometry_backend" value="primitive"/>
  </include>
</launch>\n''')
    owned={};processes=[];files=[];result=dict(status='FAIL',task_goals=0,case_runs=0)
    deadline=time.monotonic()+120

    def start(command,label):
        log=(out/(label+'.log')).open('x');files.append(log)
        proc=subprocess.Popen(['bash','--noprofile','--norc','-c',
            'source /opt/ros/noetic/setup.bash && source devel/setup.bash && exec "$@"',
            'perception_smoke',*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
        processes.append(proc)
        return proc

    def observe():
        remember_descendants([p.pid for p in processes if p.poll() is None],owned)
        if time.monotonic()>deadline: raise TimeoutError('120s stationary smoke deadline')

    try:
        sim=start(['roslaunch',str(launch)],'gazebo')
        socket.setdefaulttimeout(1.)
        master=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        while True:
            observe()
            if sim.poll() is not None: raise RuntimeError('Gazebo launch exited')
            try:
                code,_,state=master.getSystemState('/perception_smoke_observer')
                if code==1 and any(t=='/care_arm/joint_states' for t,_ in state[0]): break
            except (OSError,xmlrpc.client.Error): pass
            time.sleep(.2)
        print('Gazebo joint state publisher ready; checking eight cameras and map rays.',flush=True)
        raw_checker=start(['/usr/bin/python3',str(ROOT/'scripts/wait_for_phase_e_perception.py'),
            '--config',str(ROOT/'src/care_confidence_map/config/tof_fusion_self_filter.yaml'),
            '--raw-only','--timeout','30','--output',str(out/'perception_raw_ready.json')],'raw_readiness')
        while raw_checker.poll() is None:
            observe();time.sleep(.2)
        if raw_checker.returncode: raise RuntimeError('Eight-camera raw readiness failed')
        checker=start(['/usr/bin/python3',str(ROOT/'scripts/wait_for_phase_e_perception.py'),
            '--config',str(ROOT/'src/care_confidence_map/config/tof_fusion_self_filter.yaml'),
            '--timeout','30','--output',str(out/'perception_ready.json')],'readiness')
        while checker.poll() is None:
            observe();time.sleep(.2)
        if checker.returncode: raise RuntimeError('Real ToF readiness failed')
        code,_,state=master.getSystemState('/perception_smoke_observer')
        if code!=1: raise RuntimeError('Unable to inspect graph')
        result.update(status='PASS',perception=json.loads((out/'perception_ready.json').read_text()),
                      graph=state)
        if result['perception']['summaries']['map']['latest'].get('body_prior_geometry_backend')!='primitive':
            raise RuntimeError('Expected current primitive body prior')
        print('PASS: real camera callbacks, fusion and nonempty map rays.',flush=True)
    except Exception as exc:
        result['status']='FAIL'
        result['error']=repr(exc)
        raise
    finally:
        remember_descendants([p.pid for p in processes if p.poll() is None],owned)
        for p in reversed(processes):
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGINT)
                except ProcessLookupError: pass
        result['cleanup']=reap_owned(owned)
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        for f in files:f.close()
        if result['cleanup']['remaining']:result['status']='FAIL'
        (out/'result.json').write_text(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
