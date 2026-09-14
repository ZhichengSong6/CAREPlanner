#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import abc_protocol as proto


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference-root",type=Path,required=True)
    args=ap.parse_args()
    root=args.reference_root.resolve()
    evaldir=proto.evaluation_dir(root,"pilot")
    files=[
        evaldir/"summary.md",evaldir/"report.json",evaldir/"manifest.json",
    ]
    for arm in proto.ARMS:
        base=proto.output_dir(root,arm,"pilot")
        files += [base/"run.json",base/"validation.jsonl",base/"metrics.jsonl"]
    missing=[str(p) for p in files if not p.is_file() or p.stat().st_size==0]
    if missing:
        raise SystemExit("Missing/empty formal ABC reports:\n"+"\n".join(missing))
    output=root/"abc_capacity_routing_reports.zip"
    if output.exists():
        raise SystemExit(f"Refusing overwrite: {output}")
    sums=[]
    for p in files:
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        sums.append(f"{digest}  {p.relative_to(root)}")
    checksum="\n".join(sums)+"\n"
    with ZipFile(output,"x",ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,arcname=str(p.relative_to(root)))
        z.writestr("REPORT_SHA256SUMS",checksum)
    with ZipFile(output,"r") as z:
        bad=z.testzip()
        if bad is not None:
            raise SystemExit(f"ZIP integrity failure: {bad}")
    print(f"[done] abc_reports_zip {output} bytes={output.stat().st_size}")


if __name__=="__main__":
    main()
