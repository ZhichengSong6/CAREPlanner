#!/usr/bin/env python3
"""Installed optional-backend ABI tests. No ROS graph, CUDA or robot."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install-prefix', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    prefix = args.install_prefix.resolve(); out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status='FAIL', checks={})
    root = Path(__file__).resolve().parents[1]
    def run(cmd, name, env=None):
        with (out/(name+'.log')).open('x') as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=90)
        assert proc.returncode == 0, (name, proc.returncode)
    try:
        common = ['g++', '-std=c++14', '-I'+str(prefix/'include'), '-I/opt/ros/noetic/include',
                  str(root/'scripts/test_phase_e_legacy_backend.cpp'), '-L'+str(prefix/'lib'),
                  '-Wl,-rpath,'+str(prefix/'lib')]
        run(common+['-ltrajectory_risk_evaluator', '-o', str(out/'lazy')], 'compile_lazy')
        run(common+['-DLEGACY_DIRECT', '-lbody_sample_model', '-o', str(out/'direct')], 'compile_direct')
        sample = str(prefix/'share/care_confidence_map/config/body_samples.yaml')
        run([str(out/'lazy'), sample, 'available'], 'lazy')
        result['checks']['lazy_load_copy_and_bad_yaml'] = 'PASS'
        run([str(out/'direct'), sample, 'available'], 'direct')
        result['checks']['original_Cpp_API'] = 'PASS'
        # Deliberately isolated COPY of the core, without its optional peer.
        # Never rename/delete a deployed library, and never search another prefix.
        isolated = out/'isolated'; isolated.mkdir()
        shutil.copy2(prefix/'lib/libtrajectory_risk_evaluator.so', isolated)
        env = dict(os.environ, LD_LIBRARY_PATH=str(isolated)+':'+str(prefix/'lib')+':'+os.environ.get('LD_LIBRARY_PATH', ''))
        run([str(out/'lazy'), sample, 'missing'], 'missing', env)
        result['checks']['missing_peer_fails_closed_without_search_fallback'] = 'PASS'
        result['status'] = 'PASS'
    except Exception:
        result['error'] = traceback.format_exc()
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
