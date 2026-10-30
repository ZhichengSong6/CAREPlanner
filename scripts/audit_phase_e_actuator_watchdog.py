#!/usr/bin/env python3
"""Reproduce measured stop travel and formal-window watchdog statistics."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-dir',type=Path,default=ROOT/'outputs/phase_e_actuator_watchdog_20260930/gazebo_v2')
    parser.add_argument('--formal-run',type=Path,default=ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_006_r1_case006_actuator_watchdog_20261001/run')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/phase_e_actuator_watchdog_20260930/audit')
    args=parser.parse_args()
    diag=args.diagnostic_dir; run=args.formal_run; out=args.output_dir
    inputs=[diag/'result.json',diag/'joint_states.json',diag/'states.json',run/'goal_stop_status.json',run/'actuator_watchdog_state.csv']
    diagnostic=json.loads(inputs[0].read_text()); assert diagnostic['status']=='PASS'
    samples=json.loads(inputs[1].read_text()); states=json.loads(inputs[2].read_text())
    times=np.array([r['t'] for r in samples])
    qs=np.array([[r['q'][r['name'].index(j)] for j in NAMES] for r in samples])
    assert np.all(np.diff(times)>=0)
    def pose(t):
        assert times[0]<=t<=times[-1]
        return np.array([np.interp(t,times,qs[:,j]) for j in range(7)])
    trials=[]
    for e in diagnostic['trials']:
        generation,expiry,stop=e['final_valid_generation_s'],e['expiry_s'],e['measured_zero_s']
        trials.append(dict(e,interpolated_measured_displacement_from_generation_rad=(pose(stop)-pose(generation)).tolist(),
                           interpolated_measured_displacement_from_expiry_rad=(pose(stop)-pose(expiry)).tolist()))
    w=json.loads(inputs[3].read_text())['benchmark_window']
    with inputs[4].open() as stream:
        rows=[r for r in csv.DictReader(stream) if r.get('field.header.stamp','').isdigit() and
              w['start_ros_ns']<=int(r['field.header.stamp'])<=w['end_ros_ns']]
    assert rows
    summary=dict(window_ros_s=[w['start_ros_s'],w['end_ros_s']],samples=len(rows),
                 mode_counts=dict(Counter(r['field.mode'] for r in rows)),
                 max_observed_command_age_ms=max((int(r['field.header.stamp'])-int(r['field.command_stamp']))/1e6 for r in rows),
                 rejected_at_window_start=int(rows[0]['field.rejected_commands']),
                 rejected_at_window_end=int(rows[-1]['field.rejected_commands']),
                 clock_fault_samples=sum(r['field.clock_fault']!='0' for r in rows))
    result=dict(scope='Recorded joint-space travel with linear interpolation; no body swept-volume or worst-case proof',
                measured_stop_threshold_rad_s=.001,measured_stop_confirmation_s=.1,
                trials=trials,formal=summary,
                input_sha256={str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex='col',constrained_layout=True)
    for col,e in enumerate(trials):
        start=e['final_valid_generation_s']; end=e['measured_zero_s']+.11
        selected=[r for r in states if start-.03<=r['t']<=end]
        axes[0,col].plot([r['t']-start for r in selected],[r['applied'][1] for r in selected],label='Applied command')
        axes[0,col].plot([r['t']-start for r in selected],[r['measured'][1] for r in selected],label='Measured speed')
        selected_t=times[(times>=start-.03)&(times<=end)]
        axes[1,col].plot(selected_t-start,[pose(t)[1]-pose(start)[1] for t in selected_t],label='Measured joint travel')
        for row in range(2):
            axes[row,col].axvline(e['expiry_s']-start,color='red',linestyle='--',label='Expiry')
            axes[row,col].axvline(e['command_zero_s']-start,color='green',linestyle=':',label='Command zero')
            axes[row,col].grid(alpha=.25)
        axes[0,col].set_title('Joint 2: {} rad/s{}'.format(e['initial_joint2_command_rad_s'],' + replay' if e['replay_injected'] else ''))
        axes[0,col].set_ylabel('Speed (rad/s)'); axes[0,col].legend(fontsize=8)
        axes[1,col].set_ylabel('Travel from last generation (rad)')
        axes[1,col].set_xlabel('ROS time since last valid generation (s)')
    fig.savefig(out/'stop_trace.png',dpi=160); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
