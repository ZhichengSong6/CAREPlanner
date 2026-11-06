#!/usr/bin/env python3
"""Check published sources and runtime weights without historical outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'experiments/phase_e_static_baseline_20261030'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--without-models', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((PACKAGE / 'source_manifest.json').read_text())
    tracked = set(subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '-z']).decode().split('\0'))
    errors = []
    names = [entry['path'] for entry in manifest['files']]
    if len(names) != len(set(names)):
        errors.append('Duplicate source manifest paths')
    for entry in manifest['files']:
        base = ROOT if entry.get('source') == 'repository' else PACKAGE / 'overlay'
        if entry['kind'] == 'symlink':
            link = ROOT / entry['path']
            if not link.is_symlink() or os.readlink(link) != entry['target'] or not link.exists():
                errors.append('Source symlink: ' + entry['path'])
            if entry['path'] not in tracked:
                errors.append('Not in Git: ' + entry['path'])
            continue
        source = base / entry['path']
        if not source.is_file() or digest(source) != entry['sha256']:
            errors.append('Source hash: ' + str(source.relative_to(ROOT)))
        if not entry['path'].startswith('src/third_party/piqp/'):
            if str(source.relative_to(ROOT)) not in tracked:
                errors.append('Not in Git: ' + str(source.relative_to(ROOT)))
            mirror = ROOT / entry['path']
            if entry['source'] == 'overlay' and (not mirror.is_file() or digest(mirror) != entry['sha256']):
                errors.append('Root/overlay mismatch: ' + entry['path'])
    models = json.loads((PACKAGE / 'model_assets.json').read_text())['models']
    if not args.without_models:
        for model in models:
            if not model['required']:
                continue
            path = ROOT / model['path']
            if not path.is_file() or digest(path) != model['sha256']:
                errors.append('Model missing/mismatched (update to the matching publication commit): ' + model['path'])
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps(dict(status='PASS',source_files=len(manifest['files']),
                         required_models_checked=0 if args.without_models else 2,
                         candidate=manifest['runtime_revision'])))


if __name__ == '__main__':
    main()
