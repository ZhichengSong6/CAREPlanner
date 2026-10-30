#!/usr/bin/env python3
"""Controlled scheduling faults against production limiter methods, no dynamics."""
import hashlib
import json
from pathlib import Path

import numpy as np

from test_phase_e_rate_limiter_hold import HoldTests, Stamp
from test_phase_e_shutdown_timers import message

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase_e_stopping_bounds_20260930'
ACCEL = np.array([3., 3., 4., 4., 6., 6., 6.])
VMAX = np.array([2., 2., 2., 2., 2.5, 2.5, 2.5])


def scenario(speed, sign, reaction, gaps):
    fixture = HoldTests()
    fixture.setUp()
    n = fixture.node
    n.acceleration_limits = ACCEL.copy()
    n.velocity_limits = VMAX.copy()
    n._last_output = sign * speed.copy()
    # During the injected reaction interval the previous command is held.
    # Start the braking clock at hold acceptance, conservatively discarding
    # time credit before acceptance. Production always has last_tick if moving.
    fixture.clock += reaction
    n._last_tick = Stamp(fixture.clock)
    n._vbc_hold_cb(message(data=True))
    elapsed = reaction
    travel = speed * reaction
    stopped = np.full(7, np.nan)
    previous = n._last_output.copy()
    for i in range(10000):
        gap = float(gaps[i % len(gaps)])
        travel += np.abs(previous) * gap  # previous output held until next tick
        fixture.clock += gap
        elapsed += gap
        n._desired_cb(message(data=(sign * VMAX).tolist()))
        n._timer_tick(None)
        current = n._last_output.copy()
        assert np.all(np.abs(current) <= np.abs(previous) + 1e-12)
        stopped[np.isnan(stopped) & (np.abs(current) < 1e-12)] = elapsed
        previous = current
        if np.all(np.isfinite(stopped)):
            break
    else:
        raise AssertionError('did not stop')
    h = float(max(gaps))
    effective_accel = ACCEL * min(1., .02 / h)
    time_bound = reaction + speed / effective_accel + h
    travel_bound = speed * reaction + speed**2 / (2 * effective_accel) + speed * h
    assert np.all(stopped <= time_bound + 1e-9), (stopped, time_bound)
    assert np.all(travel <= travel_bound + 1e-9), (travel, travel_bound)
    return dict(speed_rad_s=speed.tolist(), sign=sign, reaction_s=reaction,
                callback_gaps_s=list(map(float, gaps)), stop_s=stopped.tolist(),
                command_travel_rad=travel.tolist(), time_bound_s=time_bound.tolist(),
                command_travel_bound_rad=travel_bound.tolist())


def main():
    rng = np.random.RandomState(630)
    schedules = [[h] for h in (.005, .010, .020, .050, .100)]
    schedules += [list(rng.uniform(.001, .100, 50))]
    results = [scenario(speed, sign, reaction, gaps)
               for speed in (np.full(7, .12), np.full(7, .6), VMAX)
               for sign in (-1, 1) for reaction in (0., .01, .05, .1)
               for gaps in schedules]
    example = [r for r in results if r['speed_rad_s'][0] == .6
               and r['sign'] == 1 and r['reaction_s'] == 0.
               and len(r['callback_gaps_s']) == 1]
    # A concrete falsification of the nominal continuous-ramp assumption.
    delayed = next(r for r in example if r['callback_gaps_s'] == [.05])
    assert delayed['stop_s'][0] > .6 / 3. * 2
    source = ROOT / 'src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py'
    result = dict(status='PASS', scenario_count=len(results),
                  joint_scenario_count=7*len(results),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  nominal_continuous_stop_model='FALSIFIED under delayed callbacks',
                  conditional_command_bound='PASS for all injected schedules',
                  assumptions=['monotone clock equals elapsed time',
                               'every callback gap <= specified H',
                               'hold remains accepted until stopped',
                               'command applied immediately and held until next update',
                               'nonzero initial output has an initialized last_tick'],
                  exclusions=['No proof of a finite runtime H or reaction bound',
                              'No actuator response, transport bound or physical deceleration proof'],
                  example_joint1=[dict(gap_s=r['callback_gaps_s'][0],
                                       stop_s=r['stop_s'][0],
                                       travel_rad=r['command_travel_rad'][0],
                                       travel_bound_rad=r['command_travel_bound_rad'][0])
                                  for r in example], scenarios=results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'injection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'scenarios'}, indent=2))


if __name__ == '__main__':
    main()
