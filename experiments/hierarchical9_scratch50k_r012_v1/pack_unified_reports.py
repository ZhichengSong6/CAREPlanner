#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--r012-root',type=Path,required=True)
    ap.add_argument('--reference-root',type=Path,required=True)
    args=ap.parse_args()
    root=args.r012_root.resolve(); ref=args.reference_root.resolve()
    ev=root/'evaluation_unified_dev'
    manifest=json.loads((ev/'manifest.json').read_text()) if (ev/'manifest.json').is_file() else {}
    if manifest.get('status')!='COMPLETE':
        raise SystemExit('Unified evaluation is not COMPLETE')
    files=[ev/'summary.md',ev/'report.json',ev/'manifest.json',ev/'solves.jsonl']
    for arm in ('R0','R1','R2'):
        files += [root/'formal'/arm/'run.json', root/'formal'/arm/'metrics.jsonl']
    for arm in ('E0','E1'):
        files += [ref/'e012_end2end'/arm/'run.json']
    missing=[str(p) for p in files if not p.is_file() or p.stat().st_size==0]
    if missing: raise SystemExit('Missing/empty report files:\n'+'\n'.join(missing))
    out=root/'r012_unified_dev_reports.zip'
    if out.exists(): raise SystemExit(f'Refusing overwrite: {out}')
    sums=[]
    with ZipFile(out,'x',ZIP_DEFLATED) as z:
        for p in files:
            if p.is_relative_to(root):
                arc=str(p.relative_to(root))
            else:
                arc='reference_root/'+str(p.relative_to(ref))
            z.write(p,arcname=arc)
            sums.append(f'{sha(p)}  {arc}')
        z.writestr('REPORT_SHA256SUMS','\n'.join(sums)+'\n')
    with ZipFile(out,'r') as z:
        bad=z.testzip()
        if bad is not None: raise SystemExit(f'ZIP integrity failure: {bad}')
    print(f'[done] r012_unified_reports_zip {out} bytes={out.stat().st_size}')

if __name__=='__main__': main()
