#!/usr/bin/env python3
"""Prepare the pinned PIQP headers with CAREPlanner's finite-bound fix."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('The patched copy must be outside the PIQP source tree')
    metadata = json.loads((ROOT / 'patches/piqp/manifest.json').read_text())
    relative = Path('include/piqp/kkt_system.hpp')
    header = source / relative
    if not header.is_file():
        raise RuntimeError('PIQP is missing; run git submodule update --init --recursive')
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    if digest not in (metadata['upstream_header_sha256'], metadata['patched_header_sha256']):
        raise RuntimeError('Unexpected PIQP header version; refusing to apply a different patch')
    shutil.copytree(source / 'include', destination / 'include', dirs_exist_ok=True)
    if digest == metadata['upstream_header_sha256']:
        subprocess.run(['patch', '--batch', '--forward', '-p1', '-d', str(destination),
                        '-i', str(ROOT / 'patches/piqp/finite-bound-indices.patch')],
                       check=True, stdout=subprocess.DEVNULL)
    if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != metadata['patched_header_sha256']:
        raise RuntimeError('Patched PIQP digest mismatch')
    return destination / 'include'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'src/third_party/piqp')
    parser.add_argument('--destination', type=Path, default=ROOT / 'build/care_piqp')
    args = parser.parse_args()
    print(prepare(args.source, args.destination))


if __name__ == '__main__':
    main()
