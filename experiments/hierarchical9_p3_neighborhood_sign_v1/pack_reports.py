#!/usr/bin/env python3
"""Stdlib-only report ZIP; no checkpoint/data/archive overwrite, no GPU required."""
from pathlib import Path
import argparse
import hashlib
import json
import os
from zipfile import ZipFile, ZIP_DEFLATED


def pack(root):
    root=Path(root).resolve()
    files=['evaluation_p3/summary.md','evaluation_p3/report.json','evaluation_p3/manifest.json',
           'P3/validation.jsonl','P3/run.json','P3/metrics.jsonl']
    for name in files:
        f=root/name
        if not f.is_file() or not f.stat().st_size:
            raise FileNotFoundError(f)
    m=json.loads((root/'evaluation_p3/manifest.json').read_text())
    if m.get('status')!='COMPLETE' or m.get('mode')!='pilot' or m.get('pilot_updates')!=2000:
        raise ValueError('Not a complete P3 pilot evaluation')
    out=root/'p3_pilot_reports.zip'
    if out.exists():
        raise FileExistsError(f'No overwrite: {out}')
    tmp=root/f'.p3_reports_{os.getpid()}.zip'
    try:
        with ZipFile(tmp,'x',compression=ZIP_DEFLATED) as z:
            hashes=[]
            for name in files:
                data=(root/name).read_bytes()
                z.writestr(name,data)
                hashes.append(hashlib.sha256(data).hexdigest()+'  '+name)
            z.writestr('REPORT_SHA256SUMS','\n'.join(hashes)+'\n')
        with ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise RuntimeError('ZIP integrity failure')
        # Atomically publish without overwriting an archive created by another process.
        os.link(tmp,out)
    finally:
        if tmp.exists():tmp.unlink()
    print(f'[done] p3_reports_zip {out} bytes={out.stat().st_size}')
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-root',type=Path,required=True)
    args=ap.parse_args()
    pack(args.reference_root)
