#!/usr/bin/env python3
"""Offline only: prepare, base labels, declared audit, immutable paired export."""
import argparse
from pathlib import Path
from prepare import Sampling,prepare
from batch_pipeline import worker,merge
from common import open_run,old_io


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','worker','merge','inspect'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--urdf',type=Path)
    p.add_argument('--phase',choices=('base','audit'),default='base');p.add_argument('--device',default='cpu')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-tasks',type=int)
    p.add_argument('--sampling',type=Path,default=Path(__file__).with_name('sampling.json'))
    a=p.parse_args();urdf=a.urdf or a.repo/'src/arm_description/urdf/Arm.urdf'
    if a.stage=='prepare':
        if a.source is None:p.error('--source is required')
        prepare(a.source,a.out,a.repo,urdf,Sampling(**old_io.read_json(a.sampling)),a.resume)
    elif a.stage=='worker':worker(a.out,a.repo,urdf,a.device,a.rank,a.world,a.phase,a.max_tasks)
    elif a.stage=='merge':merge(a.out,a.phase=='audit')
    else:
        _,s,_,_,_=open_run(a.out)
        print(f'queries={s["query_count"]} base_tasks={s["supported_tasks"]} planned_audit_tasks={s["audit_tasks"]}')
        print((a.out/'sampling_report.json').read_text())

if __name__=='__main__':main()
