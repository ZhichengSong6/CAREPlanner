#!/usr/bin/env python3
"""Reconstruct the completed study in a NEW directory; never starts a run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def source_bytes(entry):
    path=(ROOT if entry['source']=='repository' else PACKAGE/'overlay')/entry['path']
    data=path.read_bytes() if path.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest()!=entry['sha256']:
        result=subprocess.run(['git','-C',str(ROOT),'cat-file','blob',entry['git_blob']],capture_output=True)
        data=result.stdout if result.returncode==0 else None
    if data is None or hashlib.sha256(data).hexdigest()!=entry['sha256']:
        raise RuntimeError('Missing baseline source: '+entry['path']+'; use the matching publication commit')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets-root',type=Path,default=ROOT,
                        help='Directory containing the src/... checkpoint paths in model_assets.json')
    parser.add_argument('--without-models',action='store_true',help='Prepare a source/build check only')
    parser.add_argument('--inputs-root',type=Path,
                        help='Local study directory containing the external inputs in input_assets.json')
    args=parser.parse_args();out=args.output.resolve();assets=args.assets_root.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory: '+str(out))
    models=json.loads((PACKAGE/'model_assets.json').read_text())['models']
    if not args.without_models:
        for model in models:
            p=assets/model['path']
            if not p.is_file() or digest(p)!=model['sha256']:
                raise RuntimeError('Missing or different model asset: '+str(p))
    input_manifest=json.loads((PACKAGE/'input_assets.json').read_text())
    input_root=args.inputs_root.resolve() if args.inputs_root else None
    if input_root is None and not args.without_models:
        raise ValueError('Simulation inputs are external; supply --inputs-root, or use --without-models for a source-only build')
    if input_root is not None:
        for entry in input_manifest['files']:
            p=input_root/entry['path']
            if not p.is_file() or digest(p)!=entry['sha256']:
                raise RuntimeError('Missing or different local input: '+str(p))
    manifest=json.loads((PACKAGE/'source_manifest.json').read_text())
    # Resolve all source inputs before creating the destination.
    sources={e['path']:source_bytes(e) for e in manifest['files'] if e['kind']=='file'}
    runtime=out/'fixed/runtime';runtime.mkdir(parents=True)
    for entry in manifest['files']:
        target=runtime/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if entry['kind']=='symlink':target.symlink_to(entry['target'])
        else:target.write_bytes(sources[entry['path']]);target.chmod(entry['mode'])
    for source in sorted((PACKAGE/'template').rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(PACKAGE/'template')
        # Old local template copies may remain ignored after source-only cleanup.
        # They must never silently supply excluded recordings.
        if source.suffix=='.json' and relative.as_posix() not in {
                'fixed/plan.json','fixed/old_model_binding.json','fixed/new_model_binding.json'}:
            continue
        if relative.parts[:2]==('fixed','runtime_inputs'):
            relative=Path('fixed/runtime/outputs',*relative.parts[2:])
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes()
        if b'@BASELINE_RUN@' in data or b'@REPO_ROOT@' in data:
            data=data.replace(b'@BASELINE_RUN@',os.fsencode(out)).replace(b'@REPO_ROOT@',os.fsencode(ROOT))
        target.write_bytes(data);target.chmod(source.stat().st_mode & 0o777)
    if input_root is not None:
        for entry in input_manifest['files']:
            data=(input_root/entry['path']).read_bytes()
            if entry['relocate_paths']:
                for original,replacement in input_manifest['path_replacements'].items():
                    destination=str(out) if replacement=='@BASELINE_RUN@' else str(ROOT)
                    data=data.replace(os.fsencode(original),os.fsencode(destination))
            target=out/entry['path'];target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
    build_edits=[]
    for package in ('egocentric_arm_planner','care_confidence_map'):
        cmake=runtime/'src'/package/'CMakeLists.txt';original=digest(cmake)
        cmake.write_text(cmake.read_text()+'\n'+(PACKAGE/'build_support'/(package+'.cmake')).read_text())
        build_edits.append(dict(path=str(cmake.relative_to(runtime)),original_sha256=original,prepared_sha256=digest(cmake)))
    for model in models:
        if args.without_models:continue
        target=runtime/model['path'];target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(assets/model['path'])
    if input_root is not None:
        gate=out/'fixed/bootstrap_gate.json';data=json.loads(gate.read_text())
        data['source_study_scenes_sha256']=data['scenes_sha256']
        data['scenes_sha256']=digest(out/'fixed/scenes.json')
        gate.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for name in ('runs','versions','evidence'):(out/name).mkdir()
    (out/'preparation.json').write_text(json.dumps(dict(status='PREPARED',source_manifest_sha256=digest(PACKAGE/'source_manifest.json'),
        files=len(manifest['files']),models_attached=not args.without_models,inputs_attached=input_root is not None,build_only_cmake_additions=build_edits,
        selected_policy='current',new_windows=0),indent=2)+'\n')
    print(json.dumps(dict(status='PREPARED',runtime=str(runtime),new_windows=0)))


if __name__=='__main__':main()
