#!/usr/bin/env python3
"""Mocked sbatch workflow test; NEVER contacts a Slurm controller."""
from pathlib import Path
import tempfile
import shutil
import subprocess
import os
import json


def main():
    src = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = root / 'repo'
        target = repo / 'experiments/hierarchical9_calibration_pilot_v1'
        shutil.copytree(src, target, ignore=shutil.ignore_patterns('__pycache__'))
        subprocess.run(['git', 'init', '-q', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c',
                        'user.email=test@example.invalid', 'commit', '-qm', 'synthetic submission test'], check=True)
        fake = root / 'bin'
        fake.mkdir()
        log = root / 'calls.jsonl'
        fakebatch = fake / 'sbatch'
        fakebatch.write_text(
            '#!/usr/bin/env python3\n'
            'import json,os,sys,pathlib\n'
            "p=pathlib.Path(os.environ['MOCK_LOG'])\n"
            'n=len(p.read_text().splitlines()) if p.exists() else 0\n'
            "with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "print(str(91000+n)+';syntheticcluster')\n")
        fakebatch.chmod(0o755)
        env = dict(os.environ, PATH=str(fake)+':'+os.environ['PATH'], MOCK_LOG=str(log),
                   CAL_ARTIFACT_ROOT=str(root/'artifacts'))
        res = subprocess.run(['bash', str(target/'submit.sh'), 'smoke'], cwd=repo,
                             env=env, text=True, capture_output=True)
        print(res.stdout)
        print(res.stderr)
        res.check_returncode()
        calls = [json.loads(x) for x in log.read_text().splitlines()]
        assert len(calls) == 3
        assert '--gres=gpu:3090:1' in calls[0]
        assert '--gres=gpu:3090:4' in calls[1]
        assert '--gres=gpu:3090:1' in calls[2]
        assert '--array=0-1%1' in calls[1]
        assert '--dependency=afterok:91000' in calls[1]
        assert '--dependency=afterok:91001' in calls[2]
        assert not any(arg.startswith('--mem') for call in calls for arg in call)
        for call in calls:
            for arg in call:
                if arg.startswith('--output=') or arg.startswith('--error='):
                    path = Path(arg.split('=', 1)[1])
                    assert path.is_absolute() and path.parent.is_dir()
        marker = root/'artifacts/outputs/mainline_b/last_h9_cal_smoke.path'
        run = Path(marker.read_text().strip())
        jobs = (run/'jobs.env').read_text()
        assert 'PREP_JOB=91000' in jobs and 'TRAIN_JOB=91001' in jobs and 'EVAL_JOB=91002' in jobs
        print('PASS: mocked submission, absolute logs, GPU counts, dependencies, pointer/jobs.env')


if __name__ == '__main__':
    main()
