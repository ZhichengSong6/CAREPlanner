#!/usr/bin/env python3
"""Replay recorded CASE006 stop commands and measured geometry; no dynamics claim."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase_e_case006_stopping'


def vector_series(path, prefix, stamp='%time'):
    from audit_phase_e_case006_prior_gap import rows, JOINTS
    result = []
    for r in rows(path):
        if not r.get(stamp, '').isdigit():
            continue
        if prefix == 'field.position' or prefix == 'field.velocity':
            names = [r[f'field.name{i}'] for i in range(7)]
            values = [float(r[prefix + str(names.index(j))]) for j in JOINTS]
        else:
            values = [float(r[prefix + str(j)]) for j in range(7)]
        result.append((int(r[stamp]), values))
    return np.array([x[0] for x in result], dtype=np.int64), np.array([x[1] for x in result])


def previous(times, values, stamp):
    index = np.searchsorted(times, stamp, side='right') - 1
    if index < 0:
        raise ValueError('no preceding sample')
    return values[index]


def first_zero(times, values, start, threshold):
    for i in np.flatnonzero(times >= start):
        # Require a sustained 100 ms window, avoiding a transient zero crossing.
        end = np.searchsorted(times, times[i] + 100_000_000)
        if end < len(times) and np.max(np.abs(values[i:end+1])) <= threshold:
            return int(times[i])
    raise ValueError('no sustained stop')


def main():
    import pinocchio as pin
    import yaml
    from audit_case001_motion_prior import (
        URDF, JOINTS, PLANNER, load_shapes, frame_placements, distance_matrix,
    )
    from audit_phase_e_case006_prior_gap import RUN, BREACH_POINT, rows, field
    actuator_t, actuator = vector_series(RUN/'actuator_command.csv', 'field.data')
    desired_t, desired = vector_series(RUN/'tracker_desired_velocity.csv', 'field.data')
    qt, qs = vector_series(RUN/'joint_states.csv', 'field.position', 'field.header.stamp')
    vt, vs = vector_series(RUN/'joint_states.csv', 'field.velocity', 'field.header.stamp')
    assert np.array_equal(qt, vt)
    hold = next(int(r['%time']) for r in rows(RUN/'tracker_summary.csv')
                if r['%time'].isdigit() and int(r['%time']) >= 21_203_000_000
                and field(r['field.data'], 'source') == 'external_vbc_hold')
    desired_stop = first_zero(desired_t, desired, hold, 1e-9)
    actuator_stop = first_zero(actuator_t, actuator, hold, 1e-9)
    measured_stop = first_zero(vt, vs, hold, 1e-4)
    q_hold = previous(qt, qs, hold)
    q_hold_stamp = int(qt[np.searchsorted(qt, hold, side='right')-1])
    q_hold_interpolated = np.array([np.interp(hold/1e9, qt/1e9, qs[:,j]) for j in range(7)])
    u_hold = previous(actuator_t, actuator, hold)
    v_hold = previous(vt, vs, hold)
    accel_cfg = yaml.safe_load(PLANNER.read_text())['mpc']['joint_acceleration_limits']
    accel = np.array([accel_cfg[j] for j in JOINTS])
    # Command-level ideal ramps only. They are not a guaranteed physical
    # deceleration model; all actual samples below remain the evidence.
    continuous_ramp_travel = np.abs(u_hold)**2/(2*accel)
    command_integral = np.zeros(7)
    absolute_command_integral = np.zeros(7)
    for i in range(len(actuator_t)-1):
        start = max(hold, int(actuator_t[i]))
        end = min(measured_stop, int(actuator_t[i+1]))
        if end > start:
            command_integral += actuator[i]*(end-start)/1e9
            absolute_command_integral += np.abs(actuator[i])*(end-start)/1e9
    stopped_q = previous(qt, qs, measured_stop)
    model = pin.buildModelFromUrdf(str(URDF)); data = model.createData()
    shapes = load_shapes(ET.parse(URDF).getroot())
    indices = [model.joints[model.getJointId(j)].idx_q for j in JOINTS]
    q = np.zeros(model.nq)
    trace = []
    for t, positions, velocities in zip(qt, qs, vs):
        if not 20_800_000_000 <= t <= 21_600_000_000:
            continue
        q[indices] = positions
        dist = float(distance_matrix(BREACH_POINT[None,:], shapes,
                     frame_placements(model, data, shapes, q)).min())
        command = previous(actuator_t, actuator, t)
        requested = previous(desired_t, desired, t)
        trace.append(dict(ros_s=float(t/1e9), distance_m=dist,
                          joint2_measured_rad_s=float(velocities[1]),
                          joint2_actuator_rad_s=float(command[1]),
                          joint2_desired_rad_s=float(requested[1])))
    near = (actuator_t >= hold-100_000_000) & (actuator_t <= measured_stop+100_000_000)
    result = dict(
        scope='Recorded samples; command integrals use receipt-time zero-order hold.',
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (URDF, PLANNER, RUN/'actuator_command.csv',
                                RUN/'tracker_desired_velocity.csv', RUN/'joint_states.csv',
                                RUN/'tracker_summary.csv')},
        hold_ros_ns=hold, desired_zero_ros_ns=desired_stop,
        actuator_zero_ros_ns=actuator_stop, measured_stop_ros_ns=measured_stop,
        actuator_stop_delay_ms=(actuator_stop-hold)/1e6,
        measured_stop_delay_ms=(measured_stop-hold)/1e6,
        local_actuator_max_gap_ms=float(np.diff(actuator_t[near]).max()/1e6),
        at_hold=dict(actuator_velocity_rad_s=u_hold.tolist(), measured_velocity_rad_s=v_hold.tolist()),
        continuous_command_ramp_travel_rad=continuous_ramp_travel.tolist(),
        recorded_command_integral_rad=command_integral.tolist(),
        recorded_absolute_command_integral_rad=absolute_command_integral.tolist(),
        last_pre_hold_measured_stamp_ns=q_hold_stamp,
        measured_displacement_from_pre_hold_sample_rad=(stopped_q-q_hold).tolist(),
        interpolated_displacement_from_hold_rad=(stopped_q-q_hold_interpolated).tolist(),
        limits=['CSV command timestamps are recorder receipts, not actuator application acknowledgements.',
                'Configured acceleration caps limit command changes; they do not certify physical minimum deceleration.',
                'One observed stopping duration is not a worst-case delay bound.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    with (OUT/'trace.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(trace[0])); writer.writeheader(); writer.writerows(trace)
    subprocess.run(['/usr/bin/python3', str(Path(__file__).resolve()), '--plot-only'], check=True)
    print(json.dumps(result,indent=2))


def plot():
    with (OUT/'trace.csv').open() as stream:
        trace = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    hold = json.loads((OUT/'audit.json').read_text())['hold_ros_ns']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(2,1,sharex=True,figsize=(8,5),constrained_layout=True)
    ts=[r['ros_s'] for r in trace]
    axes[0].plot(ts,[1000*r['distance_m'] for r in trace],label='Measured primitive distance')
    axes[0].axhline(20,color='red',linestyle='--',label='Required 20 mm')
    axes[0].set_ylabel('Distance (mm)'); axes[0].legend()
    for key,label in [('joint2_desired_rad_s','Tracker desired'),('joint2_actuator_rad_s','Rate limiter output'),('joint2_measured_rad_s','Measured')]:
        axes[1].plot(ts,[r[key] for r in trace],label=label)
    axes[1].set_ylabel('Joint 2 speed (rad/s)'); axes[1].set_xlabel('ROS time (s)'); axes[1].legend()
    for ax in axes:
        ax.axvline(hold/1e9,color='black',linestyle=':',label='Hold'); ax.grid(alpha=.2)
    fig.savefig(OUT/'stop_trace.png',dpi=170); plt.close(fig)


if __name__ == '__main__':
    plot() if '--plot-only' in sys.argv else main()
