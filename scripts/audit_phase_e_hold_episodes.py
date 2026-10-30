#!/usr/bin/env python3
"""Full-record hold episodes; censor stops interrupted by release or record end."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase_e_stopping_bounds_20260930'
BATCHES = [
    'r1_case006_bounded_owner_retry_fourth_20260924',
    'r1_case006_candidate40_execution20_formal_20260925',
    'r1_case006_candidate40_execution20_repeat_20260925',
    'r1_case006_limiter_hold_20260929',
]
JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']


def vectors(path, measured=False):
    samples = {}
    with path.open() as stream:
        for r in csv.DictReader(stream):
            stamp = r.get('field.header.stamp' if measured else '%time', '')
            if not stamp or not stamp.isdigit() or int(stamp) == 0:
                continue
            names = [r['field.name'+str(i)] for i in range(7)] if measured else JOINTS
            prefix = 'field.velocity' if measured else 'field.data'
            samples[int(stamp)] = [float(r[prefix+str(names.index(j))]) for j in JOINTS]
    times = np.array(sorted(samples), dtype=np.int64)
    return times, np.array([samples[t] for t in times])


def hold_states(path, limiter):
    samples = {}
    for line in path.read_text().splitlines():
        t, _, data = line.partition(',')  # string payload may contain unquoted commas
        if not t.isdigit() or int(t) == 0:
            continue
        if limiter:
            fields = dict(re.findall(r'(vbc_hold|gcdf_hold)=([01])\b', data))
            if len(fields) != 2:
                continue
            held = any(value == '1' for value in fields.values())
        else:
            match = re.search(r'\bsource=(\S+)', data)
            if not match:
                continue
            held = match.group(1) in ('external_vbc_hold', 'external_gcdf_hard_hold')
        samples[int(t)] = held
    return sorted(samples.items())


def sustained_zero(times, values, start, end, threshold):
    """Sampled 100 ms evidence, wholly within hold, with <=30 ms sample gaps."""
    for i in np.flatnonzero((times >= start) & (times < end)):
        j = np.searchsorted(times, times[i] + 100_000_000)
        if j >= len(times) or times[j] >= end:
            return None
        if (np.max(np.diff(times[i:j+1])) <= 30_000_000
                and np.max(np.abs(values[i:j+1])) <= threshold):
            return int(times[i])
    return None


def audit(batch):
    run = ROOT / 'outputs/c5_5_vbc_gcdf_regime' / ('phase_e_goal_006_'+batch) / 'run'
    limiter = batch == BATCHES[-1]
    source = run / ('rate_limiter_summary.csv' if limiter else 'tracker_summary.csv')
    states = hold_states(source, limiter)
    paths = [run/'tracker_desired_velocity.csv', run/'actuator_command.csv', run/'joint_states.csv']
    series = [vectors(p, measured=i == 2) for i,p in enumerate(paths)]
    start_record = max(states[0][0], *(int(t[0]) for t,v in series))
    end_record = min(states[-1][0], *(int(t[-1]) for t,v in series))
    episodes = []
    start = None
    for t, held in states + [(end_record+1, False)]:
        if t > end_record:
            if start is not None:
                episodes.append((start, end_record, 'record_end'))
            break
        if held and start is None:
            start = t
        elif not held and start is not None:
            episodes.append((start, t, 'release'))
            start = None
    results = []
    for start, end, reason in episodes:
        r = dict(start_ros_ns=start, end_ros_ns=end, end_reason=reason,
                 duration_ms=(end-start)/1e6)
        for name,(times,values),threshold in zip(('desired','command','measured'), series, (1e-9,1e-9,1e-4)):
            stop = sustained_zero(times, values, start, end, threshold)
            before = np.searchsorted(times, start, side='right')-1
            initial = float(np.max(np.abs(values[before]))) if before >= 0 else None
            r[name] = dict(stop_ros_ns=stop, delay_ms=None if stop is None else (stop-start)/1e6,
                           censored=stop is None, pre_hold_speed_inf_rad_s=initial,
                           pre_hold_sample_age_ms=None if before < 0 else (start-int(times[before]))/1e6,
                           already_zero_at_pre_hold=initial is not None and initial <= threshold)
        results.append(r)
    summary = {}
    for name,(times,values) in zip(('desired','command','measured'), series):
        completed = [r[name]['delay_ms'] for r in results if not r[name]['censored']]
        common = times[(times >= start_record) & (times <= end_record)]
        summary[name] = dict(completed=len(completed), censored=len(results)-len(completed),
                             max_observed_delay_ms=max(completed) if completed else None,
                             full_record_max_sample_gap_ms=float(np.max(np.diff(times)))/1e6,
                             common_record_max_sample_gap_ms=float(np.max(np.diff(common)))/1e6)
    return dict(batch=batch, hold_observation='limiter summary' if limiter else 'tracker source',
                episode_count=len(results), summary=summary, episodes=results,
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [source]+paths})


def main():
    # Prevent a later episode's stop from being attributed to an earlier hold.
    t = np.arange(0, 410_000_000, 10_000_000, dtype=np.int64)
    v = np.ones((len(t),7)); v[t >= 200_000_000] = 0.
    assert sustained_zero(t,v,0,150_000_000,1e-9) is None
    assert sustained_zero(t,v,200_000_000,400_000_000,1e-9) == 200_000_000
    assert sustained_zero(t[::4],v[::4],200_000_000,400_000_000,1e-9) is None
    result = dict(scope='Full recorded ROS-time diagnostics, not benchmark-window metrics',
                  definitions=['Hold transitions are sampled summaries, not callback acceptance timestamps',
                               'Desired/command use recorder receipt; measured uses joint-state header',
                               'Stop requires 100 ms sampled zero before release/end and gaps <=30 ms',
                               'Interrupted/short holds remain censored; zero before hold is separately marked',
                               'Observed maxima are not worst-case bounds; runs are not paired comparisons'],
                  runs=[audit(batch) for batch in BATCHES])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'episodes.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([dict(batch=r['batch'], episode_count=r['episode_count'], summary=r['summary'])
                      for r in result['runs']], indent=2))


if __name__ == '__main__':
    main()
