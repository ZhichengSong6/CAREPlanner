#!/usr/bin/env python3
"""Build and run the standalone simulation from this checkout."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/phase_e_static_baseline_20261030'
PROFILE = json.loads((PACKAGE / 'demo/runtime.json').read_text())


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, cwd=ROOT, **kwargs)


def conda_setup():
    candidates = [os.environ.get('CONDA_SH')]
    if os.environ.get('CONDA_EXE'):
        candidates.append(str(Path(os.environ['CONDA_EXE']).resolve().parents[1] / 'etc/profile.d/conda.sh'))
    candidates += [str(Path.home() / p / 'etc/profile.d/conda.sh')
                   for p in ('miniconda3', 'anaconda3', 'miniforge3', 'mambaforge')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise RuntimeError('Conda setup not found. Run scripts/setup_runtime.sh or set CONDA_SH.')


def check_environment(env):
    missing = [name for name in ('catkin', 'cmake', 'g++', 'roslaunch', 'gzserver', 'git', 'zip')
               if not shutil.which(name)]
    if missing:
        raise RuntimeError('Missing commands: ' + ', '.join(missing) + '; see scripts/setup_runtime.sh')
    code = ('import json,numpy,yaml,torch; '
            'assert torch.cuda.is_available(), "CUDA PyTorch cannot access an NVIDIA GPU"; '
            'x=torch.ones(7,device="cuda",requires_grad=True); '
            'torch.autograd.grad((x*x).sum(),x); torch.cuda.synchronize(); '
            'print(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0))))')
    visibility_code = ('import json,numpy,yaml,torch,rospkg,rospy; '
                       'from urdf_parser_py.urdf import URDF; '
                       'blas=numpy.__config__.CONFIG["Build Dependencies"]["blas"]["name"]; '
                       'assert "mkl" in blas.lower(), "R1 requires the frozen MKL NumPy environment; run scripts/setup_runtime.sh"; '
                       'assert numpy.__version__ == "2.2.5" and torch.__version__ == "2.11.0+cpu", "Use the pinned CPU visibility environment"; '
                       'assert torch.get_num_threads() == 2; '
                       'print(json.dumps(dict(visibility_env="OK",numpy=numpy.__version__,torch=torch.__version__,blas=blas,threads=torch.get_num_threads())))')
    for name, snippet in ((env['GPU_ENV'], code), (env['NCDF_ENV'], visibility_code)):
        run(['bash', '--noprofile', '--norc', '-c',
             'set -e; source "$1"; conda activate "$2"; exec python -c "$3"',
             'care-demo-check', env['CONDA_SH'], name, snippet], env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('baseline', 'nonblocking_box', 'detour_box'), default='detour_box')
    parser.add_argument('--unknown-vbc', choices=('enforce', 'shadow'), default='enforce',
                        help='enforce (default): require original VBC gates; shadow: constraint-removal ablation only, permits unobserved-space motion')
    parser.add_argument('--output', type=Path, help='New output directory; default outputs/demo_TIMESTAMP')
    parser.add_argument('--env', default='careplanner', help='Conda environment for inference')
    parser.add_argument('--visibility-env', default='careplanner_visibility', help='CPU/MKL R1 environment')
    parser.add_argument('--seed', type=int, default=20263007, help='Gazebo noise seed; asynchronous timing can still vary')
    parser.add_argument('--build-only', action='store_true', help='Build without model assets or a GPU check')
    parser.add_argument('--preflight-only', action='store_true', help='Prepare/build/check without starting Gazebo')
    args = parser.parse_args()
    if not args.build_only and not os.environ.get('DISPLAY'):
        xvfb = shutil.which('xvfb-run')
        if not xvfb:
            parser.error('Headless camera simulation needs xvfb-run; run scripts/setup_runtime.sh')
        os.execv(xvfb, [xvfb, '-a', '-s', '-screen 0 1280x720x24',
                       '/usr/bin/python3', str(Path(__file__).resolve()), *sys.argv[1:]])
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.env) or (args.visibility_env and not re.fullmatch(r'[A-Za-z0-9_.-]+', args.visibility_env)):
        parser.error('Use conda environment names, not shell fragments')
    # Legacy ROS launch wrappers use quoted bash -c commands internally.
    out = (args.output or ROOT / 'outputs' / datetime.now().strftime('demo_%Y%m%d_%H%M%S')).resolve()
    if any(c.isspace() or c in "'\"" for path in (ROOT, out) for c in str(path)):
        parser.error('Use checkout/output paths without whitespace or quote characters')
    if out.exists():
        parser.error('Choose a new output directory; existing runs are retained')
    env = dict(os.environ, **PROFILE['environment'])
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CARE_SIM_SEED=str(args.seed))
    if not args.build_only:
        env.update(CONDA_SH=conda_setup(), GPU_ENV=args.env,
                   NCDF_ENV=args.visibility_env, VISIBILITY_ENV=args.visibility_env)
        if any(c in env['CONDA_SH'] for c in "'\n\r"):
            parser.error('The Conda setup path must not contain single quotes or newlines')
        check_environment(env)
        assets = json.loads((PACKAGE / 'model_assets.json').read_text())['models']
        for asset in assets:
            if not asset['required']:
                continue
            path = ROOT / asset['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset['sha256']:
                raise RuntimeError('Model missing or mismatched: ' + asset['path'] +
                                   '. Obtain the exact model assets listed in docs/model_assets.txt.')
    if not (ROOT / 'src/third_party/piqp/include/piqp/piqp.hpp').is_file():
        raise RuntimeError('Initialize the dependency: git submodule update --init --recursive')
    command = ['/usr/bin/python3', '-B', PACKAGE / 'prepare.py', '--demo', '--output', out, '--assets-root', ROOT]
    if args.build_only:
        command.append('--without-models')
    run(command, env=env)
    run(['bash', PACKAGE / 'build.sh', out], env=env)
    if args.build_only:
        print('BUILD COMPLETE: ' + str(out), flush=True)
        return
    command = ['/usr/bin/python3', '-B', out / 'run.py', '--case', '007',
               '--configuration', args.scene, '--tag', 'demo', '--unknown-vbc', args.unknown_vbc]
    for option, value in PROFILE['arguments'].items():
        command.extend(['--' + option, str(value)])
    # Prepared outputs may live outside the checkout; Git is used only for
    # provenance in the legacy launch wrappers, never as a source dependency.
    env['GIT_DIR'] = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', '--absolute-git-dir'], text=True).strip()
    env['GIT_WORK_TREE'] = str(ROOT)
    run(command + ['--preflight'], env=env)
    if not args.preflight_only:
        run(command, env=env)
    print('OUTPUT: ' + str(out), flush=True)


if __name__ == '__main__':
    scratch = ROOT / '.runtime_tmp'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        os.environ.update(TMPDIR=temporary, TMP=temporary, TEMP=temporary,
                          GPU_SOCKET=str(Path(temporary) / 'cdf.sock'))
        if len(os.fsencode(os.environ['GPU_SOCKET'])) >= 108:
            raise RuntimeError('Checkout path is too long for the local CDF Unix socket; use a shorter path.')
        main()
