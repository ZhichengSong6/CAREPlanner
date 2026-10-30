#!/usr/bin/env python3
"""Actual generate() selection with recorded CASE018 scores and fixed callbacks."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import MethodType
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
from per_sensor_visibility_runtime import PerSensorVisibilityRuntime, DEFAULT_SENSOR_FRAMES

RUN = ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_018_r1_case018_retry_delivery_20261012/run'
BEFORE = ROOT/'outputs/phase_e_case018_mount_group_fix_20261012/before/src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py'


def controlled(cls, record, excluded=None, scores=None, accept=True, replacement=True, priority=True):
    h=record['hybrid']; obj=cls.__new__(cls); obj.device=torch.device('cpu')
    for k,v in h['config'].items(): setattr(obj,k,v)
    obj.checkpoint_path=h['config']['checkpoint']; obj.checkpoint={}
    obj.sensor_masks_np=np.array([[1,1,0,0,0,0,0]]*2+[[1,1,1,0,0,0,0]]*2+
                                [[1,1,1,1,0,0,0]]*2+[[1]*7]*2)
    rank_calls=[]; branches=[]; geometry_calls=[]
    def rank(self,points,q):
        rank_calls.append(True)
        values=h['scores_at_scalar_q_zero'] if len(rank_calls)==1 else scores if scores is not None else h['scores_at_branch_seed']
        return np.array(values),np.array(h['ranking'])
    def gradient(self,points,q,sid): return 0.,torch.ones((1,7))*(sid+1)
    def branch(self,points,q,sid,**kwargs):
        branches.append(sid)
        return dict(sensor_id=sid,sensor_frame=DEFAULT_SENSOR_FRAMES[sid],q_candidate=q.numpy().reshape(7).tolist())
    def geometry(self,points,q,sid,minimum_conservative_g=None):
        geometry_calls.append(dict(sensor=sid,threshold=minimum_conservative_g))
        return dict(accepted=accept,reject_reason='accepted' if accept else 'controlled_rejection')
    for name,method in [('rank_sensors',rank),('branch_value_and_grad',gradient),('_optimize_branch',branch),('_candidate_geometry',geometry)]:
        setattr(obj,name,MethodType(method,obj))
    excluded=h['replacement_group_priority']['tried_sensor_ids'] if excluded is None else excluded
    opts=dict(excluded=excluded,previous_q=h['selected_q_vis'],score_gap_override=.75,
              preferred_sensor_ids=[7,6] if priority else [],ee_sensor_priority_first=priority,
              minimum_conservative_g=0.) if replacement else None
    result=obj.generate(record['points'],h['scalar_q_zero'],record['measured_seed'],
                        replacement=opts,preferred_sensor_ids=[7,6] if priority else [])
    return result,branches,geometry_calls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    records=[json.loads(l) for l in (RUN/'projector_traces/candidate_replacement.jsonl').read_text().splitlines()]
    records=[r for r in records if r['event']=='applied_requires_existing_gates']
    spec=importlib.util.spec_from_file_location('before_group_fix_runtime',BEFORE)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    results=[]
    for r in records:
        before,b_calls,b_geo=controlled(old.PerSensorVisibilityRuntime,r)
        after,a_calls,a_geo=controlled(PerSensorVisibilityRuntime,r)
        assert b_calls==[r['hybrid']['selected_sensor_id']]
        expected=6 if r['request_id']==16 else r['hybrid']['selected_sensor_id']
        assert a_calls==[expected] and len(a_geo)==1 and a_geo[0]['threshold']==0.
        results.append(dict(request_id=r['request_id'],before=before['selected_sensor_id'],after=after['selected_sensor_id'],
                            score_pool=after['replacement_group_priority']['score_eligible_sensor_ids']))
    assert [r['before'] for r in results]==[0,4,3,5]
    assert [r['after'] for r in results]==[0,4,3,6]
    last=records[-1]
    scores=[-100.]*8;scores[5]=100.;scores[6]=-2.
    result,branches,_=controlled(PerSensorVisibilityRuntime,last,scores=scores)
    assert branches==[6], 'even a much higher mate score must stay outside the pool'
    result,branches,_=controlled(PerSensorVisibilityRuntime,last,excluded=[0,2,4,6,7],scores=scores,priority=False)
    assert result['replacement_group_priority']['score_phase']=='paired_sensor_fallback'
    assert branches==[5], 'mate fallback remains available after all groups tried'
    failed,branches,geometry=controlled(PerSensorVisibilityRuntime,last,accept=False)
    assert branches==[6] and not failed['accepted'] and failed['rejected_sensor_ids']==[6] and len(geometry)==1
    for r in records:
        b,bc,_=controlled(old.PerSensorVisibilityRuntime,r,replacement=False)
        a,ac,_=controlled(PerSensorVisibilityRuntime,r,replacement=False)
        assert bc==ac and b['selected_sensor_id']==a['selected_sensor_id'], 'ordinary generation changed'
    report=dict(status='PASS',snapshots=results,controls=dict(large_score_cannot_escape_group=True,
        mate_fallback_after_all_groups=True,geometric_failure_no_extra_branch=True,ordinary_generation_unchanged=True),
        qualification='Full production generate() before/after. Frozen rank scores; synthetic gradient, branch and geometry callbacks isolate selection. Does not certify a real path or execution.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
