#!/usr/bin/env python3
"""Aggregate the 12-case x 2 Phase-C2 VBC execution benchmark."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from typing import Any,Dict,Optional


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file(): return {}
    try: return json.loads(path.read_text())
    except Exception: return {}

def one_glob(root: Path, pattern: str) -> Optional[Path]:
    xs=sorted(root.glob(pattern))
    return xs[-1] if xs else None

def f(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except Exception: return None

def b(v):
    return v if isinstance(v,bool) else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--case-file',required=True); ap.add_argument('--output-root',required=True)
    a=ap.parse_args(); case_file=Path(a.case_file).resolve(); root=Path(a.output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(case_file.read_text()); cases={c['case_id']:c for c in frozen['cases']}; order=frozen['selected_case_ids']
    rows=[]
    for cid in order:
        c=cases[cid]
        for mode,weight in [('baseline',float(frozen.get('waypoint_weight_baseline',0.0))),('careplanner',float(frozen.get('waypoint_weight_careplanner',3000.0)))]:
            rd=root/'runs'/cid/mode
            status=read_json(rd/'run_status.json')
            valid=read_json(rd/'runtime_case_validation.json')
            initv=read_json(rd/'initial_q_validation.json')
            projv=read_json(rd/'runtime_projector_validation.json')
            exe=read_json(one_glob(rd,'executed_vbc_*.json'))
            gate=read_json(one_glob(rd,'execution_gate_*.json'))
            wp_path=one_glob(rd,'*_weight_*/summary.json'); wp=read_json(wp_path)
            row={
                'case_id':cid,'difficulty_bin':c.get('difficulty_bin'),'mode':mode,'waypoint_weight':weight,
                'run_status':status.get('status','missing'),'initial_q_match':initv.get('passed'),'runtime_case_match':valid.get('passed'),'runtime_projector_match':projv.get('passed'),
                'expected_sweep_time_s':c.get('nominal_sweep_time_s'),'d_q_l2':c.get('distance_qvis_from_nominal_l2'),
                'nominally_visible':c.get('nominally_visible'),'nominal_vbc_margin_s':c.get('nominal_vbc_margin_s'),
                'executed_seen_before_sweep':exe.get('seen_before_sweep'),
                'executed_vbc_margin_s':f(exe.get('executed_vbc_margin_s')),
                'executed_see_delay_s':f(exe.get('see_delay_from_target_s')),
                'executed_sweep_delay_s':f(exe.get('sweep_delay_from_target_s')),
                'min_clearance_all_m':f(exe.get('min_clearance_all_m')),
                'max_tracking_inf':f(wp.get('max_tracking_inf')),'max_pred_dev_inf':f(wp.get('max_pred_dev_inf')),
                'max_command_inf':f(wp.get('max_command_inf')),'solve_ms_mean':f(wp.get('solve_ms_mean')),
                'solve_ms_max':f(wp.get('solve_ms_max')),'min_waypoint_pred_error_inf':f(wp.get('min_waypoint_pred_error_inf')),
                'gate_released':gate.get('released'),'gate_master_duration_s':f(gate.get('master_duration_s')),
            }
            margin=row['executed_vbc_margin_s']; row['vbc_safe_0p30']=None if margin is None else margin>=float(frozen.get('safety_margin_s',0.30))
            rows.append(row)

    fields=list(rows[0]) if rows else []
    with (root/'benchmark_runs.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)

    paired=[]
    for cid in order:
        pair={r['mode']:r for r in rows if r['case_id']==cid}
        base=pair.get('baseline',{}); care=pair.get('careplanner',{})
        bm=base.get('executed_vbc_margin_s'); cm=care.get('executed_vbc_margin_s')
        paired.append({
            'case_id':cid,'difficulty_bin':cases[cid].get('difficulty_bin'),'d_q_l2':cases[cid].get('distance_qvis_from_nominal_l2'),
            'baseline_margin_s':bm,'careplanner_margin_s':cm,
            'margin_improvement_s':None if bm is None or cm is None else cm-bm,
            'baseline_safe_0p30':base.get('vbc_safe_0p30'),'careplanner_safe_0p30':care.get('vbc_safe_0p30'),
            'baseline_max_pred_dev_inf':base.get('max_pred_dev_inf'),'careplanner_max_pred_dev_inf':care.get('max_pred_dev_inf'),
            'baseline_max_tracking_inf':base.get('max_tracking_inf'),'careplanner_max_tracking_inf':care.get('max_tracking_inf'),
        })
    pf=list(paired[0]) if paired else []
    with (root/'paired_summary.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=pf); w.writeheader(); w.writerows(paired)

    def stats(mode, difficulty=None):
        rr=[r for r in rows if r['mode']==mode and (difficulty is None or r['difficulty_bin']==difficulty)]
        ok=[r for r in rr if r['run_status']=='ok' and r['initial_q_match'] is True and r['runtime_case_match'] is True and r['runtime_projector_match'] is True]
        margins=[r['executed_vbc_margin_s'] for r in ok if r['executed_vbc_margin_s'] is not None]
        safe=[r for r in ok if r['vbc_safe_0p30'] is True]
        seen=[r for r in ok if r['executed_seen_before_sweep'] is True]
        return {'num_expected_runs':len(rr),'num_valid_runs':len(ok),'num_seen_before_sweep':len(seen),'num_safe_margin_ge_0p30':len(safe),'safe_rate_ge_0p30_over_valid':None if not ok else len(safe)/len(ok),'safe_rate_ge_0p30_over_expected':None if not rr else len(safe)/len(rr),'margin_mean_s':None if not margins else sum(margins)/len(margins),'margin_min_s':None if not margins else min(margins),'margin_max_s':None if not margins else max(margins)}
    payload={'benchmark_name':frozen.get('benchmark_name'),'case_file':str(case_file),'safety_margin_s':frozen.get('safety_margin_s',0.30),'num_cases':len(order),'num_expected_runs':2*len(order),'baseline':stats('baseline'),'careplanner':stats('careplanner'),'by_difficulty':{d:{'baseline':stats('baseline',d),'careplanner':stats('careplanner',d)} for d in ('easy','medium','hard')},'selected_case_ids':order,'paired_cases':paired}
    (root/'benchmark_summary.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps({'baseline':payload['baseline'],'careplanner':payload['careplanner']},indent=2))

if __name__=='__main__': main()
