#!/usr/bin/env python3
"""Build the installed-controller test and retain its dependency fingerprints."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase_e_stopping_bounds_20260930'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT/'controller_command_age.json'
    result_path.write_text(json.dumps(dict(status='RUNNING'))+'\n')
    env = dict(os.environ)
    for key, prefix in [('PKG_CONFIG_PATH','/opt/ros/noetic/lib/pkgconfig'),
                        ('LD_LIBRARY_PATH','/opt/ros/noetic/lib')]:
        env[key] = prefix + (':' + env[key] if env.get(key) else '')
    source = ROOT/'scripts/verify_phase_e_controller_command_age.cpp'
    flags = shlex.split(subprocess.check_output(
        ['pkg-config','--cflags','--libs','roscpp'],env=env,text=True))
    command = ['g++','-std=c++14','-O2',str(source),'-o',str(OUT/'controller_command_age'),
               '-L/opt/ros/noetic/lib','-lvelocity_controllers'] + flags
    try:
        subprocess.run(command,env=env,check=True)
        result = json.loads(subprocess.check_output([str(OUT/'controller_command_age')],env=env,text=True))
        dependencies = [source,
                        Path('/opt/ros/noetic/include/forward_command_controller/forward_joint_group_command_controller.h'),
                        Path('/opt/ros/noetic/include/velocity_controllers/joint_group_velocity_controller.h'),
                        Path('/opt/ros/noetic/lib/libvelocity_controllers.so')]
        result['input_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
        result['build_argv'] = command
        result_path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    except BaseException as exc:
        result_path.write_text(json.dumps(dict(status='FAIL',error=repr(exc)),indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
