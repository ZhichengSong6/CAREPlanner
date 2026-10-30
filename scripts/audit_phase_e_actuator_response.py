#!/usr/bin/env python3
"""Audit recorded multi-pose responses against conditional command-domain budgets."""
import argparse
import json
from pathlib import Path
import numpy as np
from phase_e_stop_envelope import stopping_budget


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=False)
    result=json.loads((args.input_dir/'result.json').read_text())
    states=json.loads((args.input_dir/'states.json').read_text())
    assert result['status']=='PASS' and result['input_hashes_unchanged'] and not result['cleanup']['remaining']
    trials=result['trials']; assert len(trials)==48
    labels={('joint%d_%+d'%(j,s)) for j in range(1,8) for s in (-1,1)}|{'coupled_-1','coupled_+1'}
    for pose in range(3): assert {r['label'] for r in trials if r['pose_index']==pose}==labels
    checks=[]; errors=[]; zero_drift=[]
    for trial in trials:
        rows=[r for r in states if trial['baseline_s']<=r['t']<=trial['confirmation_end_s']]
        q=np.array([r['q'] for r in rows]); integral=np.array([r['integral'] for r in rows])
        error=np.max(np.abs(q-q[0]-integral+integral[0]),axis=0); errors.append(error)
        assert np.max(np.abs(error-trial['sampled_trajectory_error_rad']))<1e-12
        assert all(r['integral_valid'] and not r['clock_fault'] for r in rows)
        assert max(b['t']-a['t'] for a,b in zip(rows,rows[1:]))<=.03
        assert np.max(np.abs(np.array(rows[0]['applied'])-trial['velocity']))<1e-9
        budget=stopping_budget(trial['velocity'],trial['velocity'],0.,trial['baseline_age_at_generation_s'],
            trial['max_update_gap_s'],[3,3,4,4,6,6,6],[2,2,2,2,2.5,2.5,2.5])
        displacement=np.max(np.abs(integral-integral[0]),axis=0)
        excess=float(np.max(displacement-budget.command_travel_rad))
        assert excess<=1e-10
        assert trial['command_zero_s']-trial['generation_s']<=budget.command_stop_s+1e-10
        # Settling is sustained across the full reported confirmation interval.
        rest=[r for r in rows if r['t']>=trial['measured_rest_s']]
        assert rest[-1]['t']-rest[0]['t']>=.1 and all(max(map(abs,r['measured']))<=.001 for r in rest)
        mask=np.array(trial['velocity'])==0
        if mask.any(): zero_drift.append(float(np.max(np.abs((q-q[0])[:,mask]))))
        checks.append(dict(pose=trial['pose_index'],label=trial['label'],
            command_displacement_rad=displacement.tolist(),conditional_command_budget_rad=budget.command_travel_rad.tolist(),
            max_command_excess_rad=excess))
    errors=np.array(errors)
    summary=dict(status='PASS',trials=48,command_budget_checks=48,
                 max_sampled_trajectory_error_rad=errors.max(axis=0).tolist(),
                 max_zero_command_joint_drift_rad=max(zero_drift),
                 max_observed_extra_settling_s=max(r['measured_extra_settling_s'] for r in trials),
                 max_observed_control_gap_s=max(r['max_update_gap_s'] for r in trials),
                 physical_bound_certified=False,runtime_admission=False,
                 scope='48 trajectories / 3 poses / 100 Hz state sampling, simulated direct-velocity plant only',
                 command_checks=checks)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
    for pose,color in enumerate(('tab:blue','tab:orange','tab:green')):
        select=np.array([t['pose_index']==pose for t in trials])
        axes[0].plot(np.arange(1,8),errors[select].max(axis=0)*1e6,'o-',color=color,label='pose %d'%pose)
    axes[0].set(xticks=range(1,8),xlabel='Joint',ylabel='Max sampled error (microrad)',title='Observed response error; not a certified physical bound')
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(range(1,49),[t['measured_extra_settling_s']*1e3 for t in trials],'o',markersize=3)
    axes[1].set(xlabel='Trial',ylabel='First sustained-rest sample\nafter command zero (ms)',ylim=(0,12))
    axes[1].grid(alpha=.25)
    fig.savefig(args.output_dir/'response_observations.png',dpi=160)
    print(json.dumps({k:v for k,v in summary.items() if k!='command_checks'},indent=2))

if __name__=='__main__': main()
