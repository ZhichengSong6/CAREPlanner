#!/usr/bin/env python3
"""Export study source; retain recorded inputs and results outside Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', 'checkpoints', 'data', 'generated', 'CMakeFiles',
        'outputs', 'build', 'devel', 'install', 'logs', 'tmp', 'slurm_logs', 'wandb'}
PAYLOAD_SUFFIXES = {'.pt', '.pth', '.npz', '.npy', '.bag', '.so', '.o', '.a', '.pyc'}
RECORDED_CASES = {
    'src/egocentric_arm_planner/config/phase_c2_vbc_cases.json',
    'src/egocentric_arm_planner/config/phase_e_obstacle_core12_v1.json',
}


def external_runtime_input(rel):
    return str(rel).startswith('scripts/fixtures/') or str(rel) in RECORDED_CASES


def portable_case_runner(text):
    """Fingerprint the study's actual models, including its scalar projector."""
    text = text.replace('from run_phase_e_repeatability import fingerprint as base_fingerprint\n', '')
    old = '    result=base_fingerprint()'
    replacement = '''    suffixes = {'.cpp', '.hpp', '.h', '.py', '.yaml', '.yml', '.launch',
                '.sh', '.urdf', '.xacro', '.world', '.xml', '.json', '.msg', '.srv'}
    paths = {p for root in ('src', 'scripts') for p in (REPO / root).rglob('*')
             if p.is_file() and (p.suffix in suffixes or p.name == 'CMakeLists.txt')
             and not any(x in p.parts for x in ('third_party', '__pycache__'))}
    paths.update((R1, SCALAR, GCDF, CASES, MODEL_BINDING_PATH))
    paths.update(p for p in (REPO / 'devel/lib').rglob('*') if p.is_file()
                 and p.suffix != '.pyc' and '__pycache__' not in p.parts)
    result = {os.path.relpath(p, REPO): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}'''
    if old not in text:
        raise ValueError('Unrecognized case runner fingerprint implementation')
    text = text.replace(old, replacement, 1)
    text = text.replace('MODEL_BINDING_PATH = ',
        'SCALAR = REPO / "src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt"\nMODEL_BINDING_PATH = ', 1)
    text = text.replace('    assert PRIOR.is_file() and MAINLINE.is_file()',
        '    assert sha(SCALAR) == "fea15cb71b278b9d003337d200f3796ccbbe8a59106287a8ea85abb2c89cb0df", "scalar projector SHA changed"\n    assert PRIOR.is_file() and MAINLINE.is_file()', 1)
    text = text.replace('GCDF_SHA = MODEL_BINDING["checkpoint_sha256"]',
        'GCDF_SHA = MODEL_BINDING["checkpoint_sha256"]\n'
        "DEMO_PATH = REPO.parent / 'demo_configuration.json'\n"
        'DEMO = json.loads(DEMO_PATH.read_text()) if DEMO_PATH.is_file() else None\n'
        "if DEMO:\n    CASE_SHA = DEMO['cases_sha256']", 1)
    text = text.replace('    assert ids == [f"phase_e_goal_{i:03d}" for i in range(30)], "not original ordered 30"',
        '    expected = DEMO[\'case_ids\'] if DEMO else [f"phase_e_goal_{i:03d}" for i in range(30)]\n'
        '    assert ids == expected, "unexpected task configuration"', 1)
    return text


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def export(study, destination):
    study, destination = study.resolve(), destination.resolve()
    state = json.loads((study / 'execution_handoff.json').read_text())
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Export requires a completed study')
    if (destination / 'source_manifest.json').exists():
        raise FileExistsError('A baseline export already exists at this destination')
    runtime = study / 'fixed/runtime'
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    inputs = []

    def external_input(original, relative, relocate=False):
        inputs.append(dict(path=str(relative), sha256=sha(original.read_bytes()),
                           relocate_paths=relocate))

    for top in ('src', 'scripts'):
        for original in sorted((runtime / top).rglob('*')):
            rel = original.relative_to(runtime)
            if any(part in SKIP for part in rel.parts) or original.suffix in PAYLOAD_SUFFIXES:
                continue
            if rel.name.endswith('training_contract.json'):
                continue  # Local model audit/analysis reports, not runtime config.
            if external_runtime_input(rel):
                if original.is_file():
                    external_input(original, Path('fixed/runtime')/rel)
                continue
            if original.is_symlink():
                if original.is_dir():
                    continue
                entries.append(dict(path=str(rel),kind='symlink',target=os.readlink(original)))
                continue
            if not original.is_file():
                continue
            canonical = ROOT / rel
            # Documentation may remove local result references without changing
            # the frozen executable sources.
            data = canonical.read_bytes() if rel.suffix == '.md' and canonical.is_file() else original.read_bytes()
            digest = sha(data)
            same = canonical.is_file() and canonical.read_bytes() == data
            if str(rel) == 'src/third_party/piqp/include/piqp/kkt_system.hpp':
                same = False  # The fixed header is not in the upstream gitlink.
            entry = dict(path=str(rel),kind='file',sha256=digest,
                         git_blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),
                         mode=original.stat().st_mode & 0o777,source='repository' if same else 'overlay')
            if not same:
                target = destination / 'overlay' / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(entry['mode'])
            entries.append(entry)
    save(destination/'source_manifest.json',dict(schema=1,study=study.name,files=entries,
        scope='Source and build/runtime configuration only; recorded fixtures, qualifications, calibration data and results are external.'))

    def template(source, relative):
        text = source.read_text()
        text = text.replace(str(study), '@BASELINE_RUN@').replace(str(ROOT), '@REPO_ROOT@')
        if str(relative) == 'fixed/runtime_inputs/cdf_model_ab/run_case.py':
            text = portable_case_runner(text)
        elif str(relative) == 'fixed/run_study.py':
            text = text.replace('    return {str(p.relative_to(O)):sha(p)',
                "    files += [RT/'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt']\n    return {str(p.relative_to(O)):sha(p)", 1)
            text = text.replace("    files += list(T.glob('*.py'))+list(O.glob('*.py'))+[O/n for n in ('plan.json','scenes.json','old_model_binding.json','mapping_gate.json','bootstrap_gate.json','guard_qualification.json','detour_route.json')]",
                "    names=['plan.json','scenes.json','old_model_binding.json','bootstrap_gate.json']\n"
                "    names += ['demo_configuration.json'] if (O/'demo_configuration.json').is_file() else ['mapping_gate.json','guard_qualification.json','detour_route.json']\n"
                "    files += list(T.glob('*.py'))+list(O.glob('*.py'))+[O/n for n in names]", 1)
        target = destination / 'template' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        target.chmod(source.stat().st_mode & 0o777)

    for name in ('run.py','review_stationary_failure.py'):
        template(study/name,Path(name))
    for name in ('run_study.py','old_model_binding.json','new_model_binding.json'):
        template(study/'fixed'/name,Path('fixed')/name)
    for name in ('scenes.json','mapping_gate.json','bootstrap_gate.json',
                 'guard_qualification.json','detour_route.json','calibration_final/profile.json'):
        external_input(study/'fixed'/name, Path('fixed')/name, relocate=True)
    for original in (study/'fixed/scenes').iterdir():
        if original.is_file():template(original,Path('fixed/scenes')/original.name)
    plan=json.loads((study/'fixed/plan.json').read_text())
    plan.update(status='READY_FOR_NEW_RUNS',study_status='REPRODUCTION',runs=[],
                source_plan=study.name,formal_runs_in_this_study=0)
    save(destination/'template/fixed/plan.json',plan)
    for original in (runtime/'outputs').rglob('*.py'):
        if '__pycache__' not in original.parts:
            template(original,Path('fixed/runtime_inputs')/original.relative_to(runtime/'outputs'))
    for rel in ['outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json']:
        external_input(runtime/rel, Path('fixed/runtime')/rel)
    save(destination/'input_assets.json',dict(schema=1,files=inputs,
        path_replacements={str(study):'@BASELINE_RUN@',str(ROOT):'@REPO_ROOT@'},
        note='Paths and checksums only. Recorded inputs stay local and are supplied with prepare.py --inputs-root.'))
    save(destination/'model_assets.json',dict(models=[
        dict(path='src/care_collision_cdf/checkpoints/cdf_25mm_v2/model.pt',sha256='d153c5b08af0d2b8ce12da500fe13d5d0b04cfc0d9ce07a2e6ad4d2dfd93a4e6',activation='relu',checkpoint_key='5000'),
        dict(path='src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt',sha256='41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70',activation='gelu',checkpoint_key='latest'),
        dict(path='src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt',sha256='4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002'),
        dict(path='src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt',sha256='fea15cb71b278b9d003337d200f3796ccbbe8a59106287a8ea85abb2c89cb0df',role='scalar VisCDF coarse projector used by the hybrid runtime')]))
    print(json.dumps(dict(files=len(entries),overlays=sum(e.get('source')=='overlay' for e in entries),destination=str(destination))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    export(args.study,args.destination)
