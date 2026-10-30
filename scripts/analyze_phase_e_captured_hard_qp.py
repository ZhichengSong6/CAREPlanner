#!/usr/bin/env python3
"""Exact saved full hard-QP replay and LP constraint isolation; never executable."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from diagnose_phase_e_repair_conflicts import lp_check

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--solver',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 snapshot=json.loads(a.snapshot.read_text());d=snapshot['subproblem'];n=d['G']['shape'][0];nc=d['selected_cdf_rows'];na=d['acceleration_rows'];ng=d['primitive_rows'];ns=d['shell_penalty_rows']
 assert n==na+nc+ng+ns
 def replay(problem):
  p=subprocess.run([str(a.solver.resolve())],input=json.dumps(problem),text=True,capture_output=True,check=True);return json.loads(p.stdout)
 original=replay(d);groups={'bounds_and_dynamics':[], 'acceleration_only':list(range(na)), 'cdf_only':list(range(na,na+nc)), 'acceleration_and_cdf':list(range(na+nc)), 'geometry_only':list(range(na+nc,n)),'without_cdf':list(range(na))+list(range(na+nc,n)), 'full':list(range(n))}
 lp={name:lp_check(d,rows) for name,rows in groups.items()}
 full=lp['full'];classification='UNRESOLVED'
 if full['status']==2:
  classification='HARD_CONSTRAINT_SET_INFEASIBLE_AT_CAPTURED_LINEARIZATION'
 elif full['feasible'] and full['max_constraint_violation']<=1e-7:
  classification='LP_FEASIBLE_PIQP_SOLVED' if original['status']=='solved' else 'LP_FEASIBLE_PIQP_NUMERICAL_OR_ITERATION_FAILURE'
 report=dict(status=classification,executable=False,query_stamp_ns=d['query_stamp_ns'],recorded_status=d['status'],recorded_decision=snapshot['decision'],recorded_identity={k:snapshot[k] for k in ['plan_seq','mode_epoch','observation_token','repair_target_revision','repair_progress_epoch']},original_piqp=original,row_groups=dict(acceleration=na,cdf=nc,primitive=ng,shell_penalty=ns),lp_checks=lp,single_row_box_conflicts=sum(r['reachable_upper'] is not None and r['margin']>r['reachable_upper']+1e-8 for r in d['cdf_rows']),snapshot=str(a.snapshot),snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),solver_sha256=hashlib.sha256(a.solver.read_bytes()).hexdigest(),limitations=['Subset LP feasibility is only constraint isolation; removed rows are NEVER an executable relaxation.','Classification concerns this exact saved local QP, not global reachability or neural training error.','LP feasibility is numerical and reported with its residual; no runtime iteration limit, row, margin or budget changed.'])
 with a.output.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({k:report[k] for k in ['status','query_stamp_ns','recorded_status','single_row_box_conflicts']}))
if __name__=='__main__':main()
