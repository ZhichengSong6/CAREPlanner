#!/usr/bin/env python3
"""Measure observation completion and task rejoin inside original score windows.

Read-only with respect to runs and scoring. A final observation-complete flag is
not evidence that the robot had time to converge after returning to NORMAL.
"""
import argparse
import json
from pathlib import Path
import re

FIELDS = re.compile(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\[[^\]]*\]|\S+)')


def string_rows(path):
    with Path(path).open() as stream:
        for line_no, line in enumerate(stream, 1):
            stamp, _, text = line.partition(',')
            if stamp.isdigit():
                yield dict(ros_ns=int(stamp), source_line=line_no,
                           **dict(FIELDS.findall(text)))


def state_intervals(events, key, start, end):
    """Clip carry-in state and transitions to [start,end]; preserve unknowns."""
    if end <= start:
        raise ValueError('Score window must have positive duration')
    active, since, result = None, start, []
    for event in sorted(events, key=lambda row: row['ros_ns']):
        stamp, value = event['ros_ns'], event.get(key)
        if stamp > end:
            break
        if stamp <= start:
            active = value
        elif value != active:
            result.append(dict(state=active, start_ros_ns=since, end_ros_ns=stamp,
                               duration_ros_s=(stamp-since)/1e9))
            active, since = value, stamp
    # Preserve a transition exactly at the deadline with zero remaining time.
    if end >= since:
        result.append(dict(state=active, start_ros_ns=since, end_ros_ns=end,
                           duration_ros_s=(end-since)/1e9))
    return result


def measure(regime_events, observation_events, start, end):
    regimes = state_intervals(regime_events, 'state', start, end)
    observations = state_intervals(observation_events, 'complete', start, end)
    final = observations[-1] if observations[-1]['state'] == '1' else None
    normal_after = []
    if final:
        for interval in regimes:
            a = max(interval['start_ros_ns'], final['start_ros_ns'])
            b = interval['end_ros_ns']
            if interval['state'] == 'NORMAL' and b > a:
                normal_after.append(dict(start_ros_ns=a, end_ros_ns=b, duration_ros_s=(b-a)/1e9))
    return dict(start_ros_ns=start, end_ros_ns=end, window_ros_s=(end-start)/1e9,
        regime_intervals=regimes, observation_intervals=observations,
        regime_duration_ros_s={state: sum(r['duration_ros_s'] for r in regimes if r['state']==state)
            for state in sorted({r['state'] for r in regimes}, key=str)},
        observation_complete_episodes=sum(r['state']=='1' for r in observations),
        observation_complete_at_end=bool(final),
        final_observation_complete_ros_ns=final['start_ros_ns'] if final else None,
        final_completion_present_at_window_start=bool(final and final['start_ros_ns']==start),
        time_remaining_after_final_observation_complete_ros_s=final['duration_ros_s'] if final else None,
        normal_intervals_after_final_observation=normal_after,
        available_normal_after_final_observation_ros_s=sum(r['duration_ros_s'] for r in normal_after) if final else None,
        interpretation='Descriptive ROS time, not wall time or a causal failure label; later reopenings supersede earlier completion.')


def measure_run(run, runtime):
    directory = Path(runtime) / run['run_directory']
    start, end = (run['benchmark_window'][k] for k in ('start_ros_ns','end_ros_ns'))
    return dict(tag=run['tag'], task_success=run.get('task_success'),
        **measure(list(string_rows(directory/'regime_summary.csv')),
                  list(string_rows(directory/'visibility_acquisition_summary.csv')), start, end))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path, help='Existing list of per-run results with benchmark_window')
    parser.add_argument('--runtime', required=True, type=Path, help='Root containing each run_directory')
    parser.add_argument('--case', help='Optional case number, e.g. 029')
    parser.add_argument('--output', required=True, type=Path, help='New JSON file; existing output is never overwritten')
    args = parser.parse_args()
    runs = json.loads(args.results.read_text())
    if args.case is not None:
        runs = [r for r in runs if str(r['case']).zfill(3)==args.case.zfill(3)]
    if not runs:
        parser.error('No matching runs')
    payload = dict(source_results=str(args.results.resolve()), runtime=str(args.runtime.resolve()),
                   runs=[measure_run(r,args.runtime) for r in runs])
    with args.output.open('x') as stream:
        json.dump(payload,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    print(f'Measured {len(runs)} existing windows: {args.output}')


if __name__=='__main__':
    main()
