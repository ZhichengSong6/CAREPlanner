#!/usr/bin/env python3
"""Read-only CAREPlanner offline relabeler. Run --help for staged commands."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from cache_io import BankCache, import_bank, prepare_queries, read_json
from label_core import SolverConfig
from pipeline import prepare_preflight, worker, merge
from repo_oracle import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare',help='Extract numeric bank once and freeze paired queries; no geometry optimization')
    prep.add_argument('--data',required=True,type=Path)
    prep.add_argument('--bank-cache',required=True,type=Path)
    prep.add_argument('--out',required=True,type=Path)
    prep.add_argument('--train-x',type=int,default=8)
    prep.add_argument('--val-x',type=int,default=2)
    prep.add_argument('--uniform-per-x',type=int,default=2)
    prep.add_argument('--near-per-x',type=int,default=1)
    prep.add_argument('--near-std',type=float,default=.05)
    prep.add_argument('--seed',type=int,default=260921)
    prep.add_argument('--split-seed',type=int,default=0)
    prep.add_argument('--val-count',type=int,default=1000)
    prep.add_argument('--shard-size',type=int,default=4)
    check=sub.add_parser('preflight',help='Verify real repository FOV/masks/Jacobian/old-label parity and freeze solver options')
    for sp in (check,):
        sp.add_argument('--out',required=True,type=Path);sp.add_argument('--repo',required=True,type=Path)
        sp.add_argument('--urdf',type=Path);sp.add_argument('--device',default='cpu')
    check.add_argument('--solver-config',type=Path,default=Path(__file__).parent/'solver_config.json')
    work=sub.add_parser('worker',help='Label assigned fixed shards; rerun safely to resume completed shards')
    work.add_argument('--out',required=True,type=Path);work.add_argument('--repo',required=True,type=Path)
    work.add_argument('--urdf',type=Path);work.add_argument('--device',default='cpu')
    work.add_argument('--rank',type=int);work.add_argument('--world-size',type=int)
    work.add_argument('--max-shards',type=int)
    combine=sub.add_parser('merge',help='Verify all shards and produce dataset index / summary; refuses partial data')
    combine.add_argument('--out',required=True,type=Path)
    verify=sub.add_parser('verify-bank',help='Full SHA256 check of the extracted immutable bank')
    verify.add_argument('--bank-cache',required=True,type=Path)
    auditing=sub.add_parser('audit',help='Optional higher-budget spot check, including failed cases; no label rewriting')
    auditing.add_argument('--out',required=True,type=Path);auditing.add_argument('--repo',required=True,type=Path)
    auditing.add_argument('--urdf',type=Path);auditing.add_argument('--device',default='cpu')
    auditing.add_argument('--samples',type=int,default=8);auditing.add_argument('--starts',type=int,default=8)
    auditing.add_argument('--maxiter',type=int,default=200);auditing.add_argument('--seed',type=int,default=77491)
    a=p.parse_args()
    if a.command=='prepare':
        if a.out.exists(): raise FileExistsError(f'Output exists; do not re-prepare: {a.out}')
        print('[bank] extraction/hash may read the full archive; original file is untouched',flush=True)
        import_bank(a.data,a.bank_cache)
        result=prepare_queries(BankCache(a.bank_cache),a.out,a.train_x,a.val_x,a.uniform_per_x,a.near_per_x,
                               a.near_std,a.seed,a.split_seed,a.val_count,a.shard_size)
        print('[queries] READY',result['query_count'],a.out,flush=True)
    elif a.command=='preflight':
        cfg=SolverConfig(**read_json(a.solver_config));cfg.validate()
        prepare_preflight(a.out,a.repo,a.urdf or a.repo/'src/arm_description/urdf/Arm.urdf',a.device,cfg)
    elif a.command=='worker':
        rank=a.rank if a.rank is not None else int(os.environ.get('RANK',0))
        world=a.world_size if a.world_size is not None else int(os.environ.get('WORLD_SIZE',1))
        device=a.device
        if device=='cuda': device=f"cuda:{int(os.environ.get('LOCAL_RANK',0))}"
        worker(a.out,a.repo,a.urdf or a.repo/'src/arm_description/urdf/Arm.urdf',device,rank,world,a.max_shards)
    elif a.command=='merge':
        merge(a.out)
    elif a.command=='audit':
        from audit_budget import audit
        audit(a.out,a.repo,a.urdf or a.repo/'src/arm_description/urdf/Arm.urdf',a.device,
              a.samples,a.starts,a.maxiter,a.seed)
    elif a.command=='verify-bank':
        bank=BankCache(a.bank_cache)
        for key,digest in bank.manifest['array_sha256'].items():
            if sha256_file(bank.root/(key+'.npy'))!=digest: raise RuntimeError(f'SHA mismatch: {key}')
        print('[bank] full checksum PASS')


if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError,OSError) as exc:
        print(f'[ERROR] {exc}',file=sys.stderr,flush=True)
        sys.exit(2)
