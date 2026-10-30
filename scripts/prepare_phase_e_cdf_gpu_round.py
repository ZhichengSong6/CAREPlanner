#!/usr/bin/env python3
"""Copy the sealed CDF harness for the old-model GPU/12-window diagnostic."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'outputs/phase_e_cdf_harness_20261023'
OUT = ROOT/'outputs/phase_e_cdf_gpu_closedloop_20261023'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def save(p, v):
    with p.open('x') as f: json.dump(v, f, indent=2, ensure_ascii=False); f.write('\n')

def main():
    inventory = json.loads((PARENT/'artifact_sha256.json').read_text())
    for n, h in inventory.items(): assert sha(PARENT/n) == h, n
    OUT.mkdir(exist_ok=False)
    rt = OUT/'runtime'
    subprocess.run(['cp', '-a', '--reflink=auto', str(PARENT/'runtime'), str(rt)], check=True)
    shutil.copytree(PARENT/'calibration_final', OUT/'calibration_final')
    relocations = []
    for base, dirs, files in os.walk(rt, followlinks=False):
        for name in dirs+files:
            p = Path(base)/name
            if p.is_symlink():
                a = os.readlink(p); b = a.replace(str(PARENT), str(OUT))
                if a != b: p.unlink(); p.symlink_to(b); relocations.append(str(p.relative_to(rt)))
            elif p.is_file() and p.stat().st_size < 16*1024**2:
                a = p.read_bytes()
                if b'\0' not in a:
                    b = a.replace(str(PARENT).encode(), str(OUT).encode())
                    if a != b: p.write_bytes(b); relocations.append(str(p.relative_to(rt)))
    save(OUT/'parent_manifest.json', dict(path=str(PARENT), manifest_sha256=sha(PARENT/'artifact_sha256.json'), files=inventory))
    save(OUT/'relocated_paths.json', relocations)
    binding = json.loads((PARENT/'old_model_binding.json').read_text())
    binding['checkpoint'] = binding['checkpoint'].replace(str(PARENT), str(OUT))
    save(OUT/'old_model_binding.json', binding)
    for name in ('NEXT_SESSION.md', 'outputs/README.md'):
        shutil.copy2(ROOT/name, OUT/(Path(name).stem+'_before.md'))
    save(OUT/'root_before.json', {str(p.relative_to(ROOT)):sha(p)
         for base in ('src','scripts','devel/lib') for p in (ROOT/base).rglob('*')
         if p.is_file() and not p.is_symlink() and p.suffix != '.pyc'})
    tasks = rt/'outputs/cdf_model_ab'
    prior = ROOT/'outputs/phase_e_rebase_task_ab_20261022/runtime/outputs/rebase_task_ab'
    for name in ('run.py','review.py','analyze.py'): shutil.copy2(prior/name, tasks/name)
    plan = json.loads((prior/'plan.json').read_text())
    plan.update(status='PREDECLARED_OLD_MODEL_HARNESS_DIAGNOSTIC', arms=['old'], cases=['008','029','025','026'],
        variable='No model/algorithm intervention; full worker input recording and GPU runtime characterization',
        intervention='Passive input statistics and attested loading', shared_code='Sealed CDF harness copy',
        fixed_start_recovery_enabled=False, source_candidate=str(PARENT/'runtime'),
        authorization='User approved GPU verification followed by four cases each three times.',
        primary_mechanism='Actual GPU model binding, complete worker input domain, runtime timing.',
        completion_profile=str(OUT/'calibration_final/profile.json'),
        limitation='12 old-model diagnostic windows, not model A/B or causal logger overhead evidence.',
        order='Each round: 008,029,025,026. Three rounds. Fixed before results.', runs=[])
    for repeat in (1,2,3):
        for case in plan['cases']:
            tag=f'old_r{repeat}_{case}'; batch=f'cdfgpu_r{repeat}_{case}_20261023'
            plan['runs'].append(dict(arm='old',recovery=False,reaudit=False,coherent=True,backend='native',
                repeat=repeat,case=case,case_id='phase_e_goal_'+case,tag=tag,batch=batch,
                run_id='phase_e_goal_'+case+'_'+batch,ranking=False,pruning=False))
    save(OUT/'plan.json', plan); save(tasks/'plan.json', plan)
    save(OUT/'copied_runtime_sha256.json', {str(p.relative_to(rt)):sha(p) for p in rt.rglob('*')
         if p.is_file() and not p.is_symlink()})
    shutil.copy2(__file__, OUT/Path(__file__).name)
    print('PREPARED', OUT, flush=True)

if __name__ == '__main__': main()
