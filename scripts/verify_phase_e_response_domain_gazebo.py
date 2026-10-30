#!/usr/bin/env python3
"""Private Gazebo response/cadence experiments; synthetic diagnostic leases only."""
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import threading
import time
import traceback
import xml.etree.ElementTree as ET
import xmlrpc.client

from phase_e_owned_processes import identity, remember_descendants, reap_owned

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']
BASE = '/care_arm/arm_group_velocity_controller'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected():
    expected = json.loads((ROOT / 'outputs/phase_e_bound_policy_20261020/host_protected_baseline.json').read_text())['protected']
    actual = {}
    for pid in expected:
        p = Path('/proc') / pid
        actual[pid] = dict(starttime=(p / 'stat').read_text().rsplit(')', 1)[1].split()[19],
                           command_sha256=digest(p / 'cmdline'))
    assert actual == expected, 'protected process identity changed'
    return actual


def trial_matrix():
    trials = []
    for j in range(7):
        for sign in (-1, 1):
            q = [0.] * 7
            v = [0.] * 7
            q[j] = -sign * .8
            v[j] = sign * 2.
            trials.append(dict(label=f'saturated_{j}_{sign}', q=q, v=v, profile='saturated', stop='expiry'))
    for j in range(7):
        v = [0.] * 7
        v[j] = 2. * (-1 if j % 2 else 1)
        trials.append(dict(label=f'acceleration_revoke_{j}', q=[0.] * 7, v=v,
                           profile='acceleration', stop='revoke'))
    for j in (0, 3, 5):
        v = [0.] * 7
        v[j] = 1.
        q = [0.] * 7
        q[j] = -.4
        trials.append(dict(label=f'reversal_{j}', q=q, v=v, profile='reversal', stop='expiry'))
    for sign in (-1, 1):
        v = [sign * 2. * (-1 if j % 2 else 1) for j in range(7)]
        trials.append(dict(label=f'coupled_{sign}', q=[-.8 * (1 if x > 0 else -1) for x in v],
                           v=v, profile='saturated', stop='expiry'))
    for sign in (-1, 1):
        trials.append(dict(label=f'loaded_acceleration_{sign}', q=[.3, -.8, .7, -.5, .3, -.4, 0.],
                           v=[sign * 2. * (-1 if j % 2 else 1) for j in range(7)],
                           profile='acceleration', stop='revoke'))
    for sign in (-1, 1):
        q = [0.] * 7
        v = [0.] * 7
        q[6] = sign * 1.05
        v[6] = -sign * 2.
        trials.append(dict(label=f'wrist_limit_inward_{sign}', q=q, v=v,
                           profile='acceleration', stop='expiry'))
    from verify_phase_e_leased_response_gazebo import historical_profile
    q, v, source = historical_profile()
    for sign in (-1, 1):
        trials.append(dict(label=f'historical_010_direction_{sign}', q=q, v=[sign * x for x in v],
                           profile='acceleration', stop='expiry', historical_source=source))
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--controller-prefix', type=Path, required=True)
    parser.add_argument('--observer-library', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--scenario-set', choices=['smoke', 'matrix', 'lifecycle', 'hold'], default='matrix')
    parser.add_argument('--tail-hold-s', type=float, default=.3)
    parser.add_argument('--trial-label', action='append', default=[], help='Select named hold diagnostics')
    args = parser.parse_args()
    assert math.isfinite(args.tail_hold_s) and .101 <= args.tail_hold_s <= 30.
    assert not args.trial_label or args.scenario_set == 'hold'
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = protected()
    for port in (args.port, args.port + 1):
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', port)) != 0, 'private port busy'
    os.environ.update(ROS_MASTER_URI=f'http://127.0.0.1:{args.port}', ROS_HOSTNAME='127.0.0.1',
        GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port + 1}', GAZEBO_MODEL_DATABASE_URI='',
        ROS_HOME=str(out / 'ros_home'), ROS_LOG_DIR=str(out / 'ros_logs'))
    prefix = args.controller_prefix.resolve()
    paths = {Path(__file__).resolve(), ROOT / 'scripts/phase_e_response_domain_observer.cpp',
             ROOT / 'scripts/phase_e_response_contract.py', ROOT / 'scripts/phase_e_owned_processes.py',
             ROOT / 'scripts/phase_e_leased_stop_evidence.py', ROOT / 'scripts/phase_e_stopping_contract.py',
             ROOT / 'scripts/phase_e_stop_envelope.py', ROOT / 'scripts/verify_phase_e_leased_response_gazebo.py',
             args.observer_library.resolve(), prefix / 'lib/libcare_velocity_controller.so'}
    paths.update(p for p in (ROOT / 'src/care_velocity_controller').rglob('*') if p.is_file())
    paths.update((ROOT / 'src/arm_description/launch').glob('*.launch'))
    paths.update((ROOT / 'src/arm_description/meshes').glob('*'))
    paths.update((ROOT / 'src/arm_description/config').glob('*.yaml'))
    paths.update((ROOT / 'src/arm_description/urdf').glob('*.urdf'))
    paths.update((ROOT / 'src/arm_description/worlds').glob('*.world'))
    paths.update((prefix / 'lib/python3/dist-packages/care_velocity_controller/msg').glob('*.py'))
    paths = {p for p in paths if p.is_file()}
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    (out / 'input_sha256.json').write_text(json.dumps(inputs, indent=2) + '\n')
    with tarfile.open(out / 'input_sources.tar.gz', 'w:gz') as tar:
        for p in sorted(paths):
            data = p.read_bytes()
            info = tarfile.TarInfo(str(p.relative_to(ROOT)))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    result = dict(status='FAIL', scenario_set=args.scenario_set, runtime_admission=False,
        production_issuer_enabled=False, physical_bound_certified=False, new_formal_windows=0,
        synthetic_diagnostic_leases=True, trials=[], lifecycle=[], checks=[], expected_step_s=.001,
        tail_hold_s=args.tail_hold_s, selected_trial_labels=args.trial_label)
    processes, handles, owned = [], [], {}
    sender = None
    stop_sender = threading.Event()
    data_lock, send_lock = threading.RLock(), threading.RLock()
    state = [None]
    leases = []
    sender_errors = []
    trace = (out / 'controller_trace.jsonl').open('x')

    def event(kind, row):
        with data_lock:
            trace.write(json.dumps(dict(kind=kind, wall_s=time.monotonic(), **row), allow_nan=False) + '\n')

    def start(argv, label):
        handle = (out / (label + '.log')).open('x')
        handles.append(handle)
        p = subprocess.Popen(list(map(str, argv)), cwd=ROOT, stdout=handle,
                             stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(p)
        owned[p.pid] = identity(p.pid)
        return p

    last_scan = [0.]

    def wait(predicate, label, timeout=30.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if time.monotonic() - last_scan[0] > .5:
                remember_descendants({p.pid for p in processes if p.poll() is None}, owned)
                last_scan[0] = time.monotonic()
            if predicate():
                return
            if sender_errors:
                raise AssertionError(sender_errors)
            if any(p.poll() is not None for p in processes):
                raise AssertionError('child exited: ' + label)
            time.sleep(.003)
        raise AssertionError('timeout: ' + label)

    def last():
        with data_lock:
            return copy.deepcopy(state[0])

    def capture_loaded():
        remember_descendants({p.pid for p in processes if p.poll() is None}, owned)
        found = {}
        for pid in owned:
            try:
                mapping = (Path('/proc') / str(pid) / 'maps').read_text()
            except (FileNotFoundError, ProcessLookupError):
                continue
            entries = {}
            for line in mapping.splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) != 6 or not parts[5].startswith('/'):
                    continue
                path = Path(parts[5])
                if any(s in path.name for s in ('libgazebo_ode', 'libode.so', 'libgazebo_physics',
                    'libdefault_robot_hw_sim', 'libgazebo_ros_control', 'libcare_velocity_controller',
                    'libresponse_domain_observer')):
                    entries[str(path)] = digest(path)
            if entries:
                found[str(pid)] = entries
        (out / 'loaded_objects.json').write_text(json.dumps(found, indent=2) + '\n')
        assert any(str(prefix / 'lib/libcare_velocity_controller.so') in entries for entries in found.values())
        return found

    try:
        robot = ET.parse(ROOT / 'src/arm_description/urdf/Arm.urdf')
        observer = out / 'libresponse_domain_observer.so'
        observer.write_bytes(args.observer_library.read_bytes())
        plugin = ET.SubElement(ET.SubElement(robot.getroot(), 'gazebo'), 'plugin',
                               name='care_response_domain_observer', filename=str(observer))
        ET.SubElement(plugin, 'directory').text = str(out)
        robot.write(out / 'observed_robot.urdf', encoding='unicode')
        result['stage'] = 'master'
        start(['roscore', '-p', args.port], 'master')
        def ready():
            try:
                return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/response_domain')[0] == 1
            except OSError:
                return False
        wait(ready, 'private master', 20.)
        import rospy
        from care_velocity_controller.msg import CommandStatus, ExecutionLease, LeasedVelocityCommand
        from phase_e_leased_stop_evidence import status_record, command_record
        from std_srvs.srv import Trigger, Empty
        from gazebo_msgs.srv import GetPhysicsProperties, SetPhysicsProperties
        from controller_manager_msgs.srv import ListControllers, SwitchController
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('response_domain_study', disable_signals=True)
        def rt(ns):
            return rospy.Time(ns // 1000000000, ns % 1000000000)
        def on_state(msg):
            row = status_record(msg)
            row['integral'] = list(msg.applied_position_integral)
            row['rearm_successes'] = msg.rearm_successes
            with data_lock:
                state[0] = row
            event('state', row)
        rospy.Subscriber(BASE + '/state', CommandStatus, on_state, queue_size=2000, tcp_nodelay=True)
        authority = rospy.Publisher(BASE + '/execution_lease', ExecutionLease, queue_size=10, tcp_nodelay=True)
        direct = rospy.Publisher(BASE + '/command_leased', LeasedVelocityCommand, queue_size=10, tcp_nodelay=True)
        result['stage'] = 'Gazebo startup'
        start(['roslaunch', 'arm_description', 'gazebo_velocity_control.launch',
            'use_command_watchdog:=true', 'require_execution_lease:=true', 'gazebo_gui:=false',
            'use_rviz:=false', 'apply_initial_joint_overrides:=false',
            'robot_urdf:=' + str(out / 'observed_robot.urdf')], 'gazebo')
        wait(lambda: last() is not None and authority.get_num_connections() > 0 and direct.get_num_connections() > 0,
             'actual controller and publishers', 90.)
        selected = next(c for c in rospy.ServiceProxy('/care_arm/controller_manager/list_controllers', ListControllers)().controller
                        if c.name == 'arm_group_velocity_controller')
        assert selected.state == 'running' and selected.type == 'care_velocity_controller/ExpiringJointVelocityController'
        assert last()['lease_required'] and last()['mode'] == 0
        assert last()['velocity_limits'] == [2.] * 7
        assert last()['acceleration_limits'] == [3., 3., 4., 4., 6., 6., 6.]
        capture_loaded()
        (out / 'ros_parameters_initial.json').write_text(json.dumps(rospy.get_param('/'), indent=2) + '\n')
        get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        physics_initial = get_physics()
        assert physics_initial.success and abs(physics_initial.time_step - .001) < 1e-12
        (out / 'physics_service_initial.yaml').write_text(str(physics_initial) + '\n')
        control = dict(kind='velocity', value=[0.] * 7, renew=True, enabled=True,
                       token=rospy.Time.now().to_nsec())
        current = [None]
        def set_control(**values):
            with send_lock:
                control.update(values)
        def send():
            sequence, lease_sequence, last_generation = 0, 0, 0
            try:
                while not stop_sender.is_set():
                    s = last()
                    now = rospy.Time.now().to_nsec()
                    with send_lock:
                        c = control
                        if c['enabled'] and s and now > last_generation:
                            grant = current[0]
                            cap = [.5] * 7 if c['kind'] == 'pose' else [abs(v) for v in c['value']]
                            if c['renew'] and (not grant or grant.execution_stamp_ns != c['token'] or
                                    grant.controller_session != s['controller_session'] or
                                    list(grant.target_abs_velocity) != cap or now - grant.issued_at.to_nsec() >= 15000000):
                                grant = ExecutionLease()
                                grant.controller_session = s['controller_session']
                                grant.execution_stamp_ns = c['token']
                                lease_sequence += 1
                                grant.lease_sequence = lease_sequence
                                grant.issued_at, grant.valid_until = rt(now), rt(now + 100000000)
                                grant.certificate_sha256 = 'd' * 64
                                grant.target_range_enforced, grant.target_abs_velocity = True, cap
                                current[0] = grant
                                authority.publish(grant)
                                row = dict(session=grant.controller_session, token=c['token'], sequence=lease_sequence,
                                           issued_ns=now, until_ns=now + 100000000, target_abs_velocity=cap)
                                leases.append(row)
                                event('synthetic_lease', row)
                            if grant:
                                velocity = c['value'] if c['kind'] == 'velocity' else [
                                    max(-.5, min(.5, 3. * (t - q))) for t, q in zip(c['value'], s['q'])]
                                msg = LeasedVelocityCommand()
                                sequence += 1
                                msg.header.stamp, msg.source_sequence = rt(now), sequence
                                msg.controller_session = grant.controller_session
                                msg.execution_stamp_ns, msg.lease_sequence = grant.execution_stamp_ns, grant.lease_sequence
                                msg.valid_until, msg.certificate_sha256 = grant.valid_until, grant.certificate_sha256
                                msg.target_range_enforced, msg.target_abs_velocity = True, grant.target_abs_velocity
                                msg.joint_names, msg.velocity = NAMES, velocity
                                direct.publish(msg)
                                event('command', command_record(msg))
                                last_generation = now
                    time.sleep(.002)
            except BaseException:
                sender_errors.append(traceback.format_exc())
        sender = threading.Thread(target=send, daemon=True)
        sender.start()
        rearm = rospy.ServiceProxy(BASE + '/rearm', Trigger)
        def sim_wait(duration, label):
            start_ns = rospy.Time.now().to_nsec()
            wait(lambda: rospy.Time.now().to_nsec() >= start_ns + int(duration * 1e9), label)
        def move(target):
            set_control(kind='pose', value=list(target), renew=True)
            wait(lambda: max(abs(a-b) for a, b in zip(last()['q'], target)) < .003 and
                         max(map(abs, last()['measured'])) < .01, 'pose reached', 120.)
            set_control(kind='velocity', value=[0.] * 7)
            sim_wait(.12, 'zero hold at pose')
            wait(lambda: max(map(abs, last()['measured'])) <= .001, 'measured rest at pose', 15.)
            assert last()['mode'] == 1
        def arm_next():
            count = last()['rearm_successes']
            set_control(kind='velocity', value=[0.] * 7, renew=True, token=rospy.Time.now().to_nsec())
            sim_wait(.06, 'post-stop new token zero')
            assert rearm().success, 'explicit rearm rejected'
            wait(lambda: last()['rearm_successes'] > count, 'explicit rearm receipt')
        wait(lambda: last()['mode'] == 1, 'initial zero applied')
        result['checks'].append('actual_controller_and_synthetic_zero')
        if args.scenario_set in ('matrix', 'smoke', 'hold'):
            trials = trial_matrix()
            if args.scenario_set == 'smoke':
                trials = [trials[0], trials[14], trials[21]]
            elif args.scenario_set == 'hold':
                if args.trial_label:
                    labels = set(args.trial_label)
                    trials = [t for t in trials if t['label'] in labels]
                    assert len(labels) == len(trials), 'unknown hold trial label'
                else:
                    trials = trials[24:28]
            result['planned_trials'] = len(trials)
            (out / 'planned_trials.json').write_text(json.dumps(trials, indent=2) + '\n')
            for i, trial in enumerate(trials):
                result['stage'] = trial['label']
                move(trial['q'])
                baseline = last()
                event('trial_begin', dict(index=i, label=trial['label'], t_ns=baseline['t_ns']))
                set_control(kind='velocity', value=trial['v'])
                if trial['profile'] in ('saturated', 'reversal'):
                    wait(lambda: max(abs(a-b) for a, b in zip(last()['applied'], trial['v'])) < 1e-5,
                         'applied target reached', 30.)
                    sim_wait(.03, 'steady command interval')
                else:
                    sim_wait(.04, 'acceleration interval')
                reverse_ns = None
                if trial['profile'] == 'reversal':
                    reverse_ns = rospy.Time.now().to_nsec()
                    set_control(value=[-v for v in trial['v']])
                    sim_wait(.04, 'reverse during acceleration')
                decision = last()
                with send_lock:
                    control['renew'] = False
                    withdrawal_ns = rospy.Time.now().to_nsec()
                    rights = copy.deepcopy([r for r in leases if r['token'] == control['token'] and r['until_ns'] > decision['t_ns']])
                    if trial['stop'] == 'revoke':
                        revoke = copy.deepcopy(current[0])
                        revoke.revoke = True
                        authority.publish(revoke)
                    event('withdrawal', dict(label=trial['label'], t_ns=withdrawal_ns, method=trial['stop'], rights=rights))
                wait(lambda: last()['mode'] == 3, 'fixed lease or revoke command stop', 15.)
                stopped = last()
                sim_wait(args.tail_hold_s, 'physical tail observation')
                wait(lambda: max(map(abs, last()['measured'])) <= .001, 'physical rest after stop', 15.)
                end = last()
                assert end['integral_valid'] and not end['clock_fault']
                trial.update(index=i, baseline_ns=baseline['t_ns'], decision_ns=decision['t_ns'],
                    withdrawal_ns=withdrawal_ns, reverse_ns=reverse_ns, end_ns=end['t_ns'],
                    token=control['token'], controller_session=decision['controller_session'],
                    command_stopped_sample_ns=stopped['t_ns'], decision_state=decision, outstanding_rights=rights)
                result['trials'].append(trial)
                event('trial_end', dict(index=i, label=trial['label'], t_ns=end['t_ns']))
                (out / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(dict(index=i, label=trial['label'], completed=len(result['trials']))), flush=True)
                arm_next()
            move([0.] * 7)
            result['checks'].extend(['all_planned_trials_completed', 'new_token_zero_and_explicit_rearm_each_trial'])
        else:
            # Deliberately mutate a private diagnostic instance. Each mutation
            # is outside the proposed fixed-domain contract, not a qualified run.
            pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
            unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
            set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
            switch = rospy.ServiceProxy('/care_arm/controller_manager/switch_controller', SwitchController)
            reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            set_control(value=[.3, 0., 0., 0., 0., 0., 0.])
            wait(lambda: abs(last()['applied'][0]-.3) < 1e-5, 'pause test moving')
            pause()
            paused_ns = rospy.Time.now().to_nsec()
            old = last()
            event('lifecycle_pause', dict(t_ns=paused_ns))
            wall_start = time.monotonic()
            while time.monotonic()-wall_start < .35:
                assert rospy.Time.now().to_nsec() == paused_ns
                time.sleep(.01)
            unpause()
            sim_wait(.03, 'pause resume')
            result['lifecycle'].append(dict(kind='pause_resume', paused_ns=paused_ns,
                wall_pause_s=time.monotonic()-wall_start, previous_state=old, next_state=last(),
                domain_status='sim_time_frozen; no wall_clock_expiry_claim'))
            set_control(value=[0.] * 7)
            sim_wait(.2, 'rest before step mutation')
            mutation_ns = rospy.Time.now().to_nsec()
            event('lifecycle_step_change', dict(t_ns=mutation_ns, new_step_s=.002))
            response = set_physics(.002, physics_initial.max_update_rate, physics_initial.gravity, physics_initial.ode_config)
            assert response.success
            sim_wait(.15, 'changed physics step')
            changed = last()
            assert changed['max_update_gap'] >= .002-1e-8
            assert set_physics(physics_initial.time_step, physics_initial.max_update_rate,
                               physics_initial.gravity, physics_initial.ode_config).success
            sim_wait(.1, 'restored step')
            result['lifecycle'].append(dict(kind='physics_step_mutation', t_ns=mutation_ns,
                changed_state=changed, domain_status='outside_fixed_1ms_domain; service_not_prevented'))
            set_control(renew=False, enabled=False)
            sim_wait(.7, 'stop before lifecycle switch')
            assert last()['mode'] == 3
            old_session = last()['controller_session']
            switch_ns = rospy.Time.now().to_nsec()
            response = switch([], ['arm_group_velocity_controller'], 2, False, 2.)
            assert response.ok
            sim_wait(.12, 'controller inactive while physics advances')
            inactive = last()
            response = switch(['arm_group_velocity_controller'], [], 2, False, 2.)
            assert response.ok
            wait(lambda: last()['controller_session'] != old_session, 'new session after restart')
            assert last()['mode'] == 0 and max(map(abs, last()['applied'])) == 0.
            result['lifecycle'].append(dict(kind='controller_stop_start', t_ns=switch_ns,
                old_session=old_session, inactive_last_state=inactive, restarted_state=last(),
                domain_status='outside_uninterrupted_control_domain; new_session_waiting'))
            reset_before = rospy.Time.now().to_nsec()
            event('lifecycle_reset', dict(t_ns=reset_before))
            reset()
            wait(lambda: rospy.Time.now().to_nsec() < reset_before, 'simulation reset time', 10.)
            time.sleep(.1)
            result['lifecycle'].append(dict(kind='simulation_reset', previous_ns=reset_before,
                next_ns=rospy.Time.now().to_nsec(), next_state=last(),
                domain_status='new_epoch_required; reset_service_not_prevented'))
            result['checks'].append('four_lifecycle_mutations_observed_and_classified')
        result['stage'] = 'complete'
        result['status'] = 'PASS_DIAGNOSTIC_NOT_QUALIFICATION'
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        stop_sender.set()
        if sender:
            sender.join(3.)
        remember_descendants({p.pid for p in processes if p.poll() is None}, owned)
        result['owned_processes'] = {str(p): token for p, token in owned.items()}
        result['cleanup'] = reap_owned(owned)
        for process in processes:
            try:
                process.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                pass
        for handle in handles:
            handle.close()
        trace.close()
        result['protected_identity_equal'] = protected() == before
        result['inputs_unchanged'] = all(digest(ROOT / p) == h for p, h in inputs.items())
        result['sender_errors'] = sender_errors
        result['private_ports_released'] = {}
        for port in (args.port, args.port + 1):
            with socket.socket() as sock:
                result['private_ports_released'][str(port)] = sock.connect_ex(('127.0.0.1', port)) != 0
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        assert result['protected_identity_equal'] and result['inputs_unchanged']
        assert not result['cleanup']['remaining'] and all(result['private_ports_released'].values())


if __name__ == '__main__':
    main()
