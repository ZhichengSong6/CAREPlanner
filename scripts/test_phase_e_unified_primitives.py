#!/usr/bin/env python3
"""Combined non-actuating geometry acceptance using resolved production launch nodes.

Real map/services, one-shot prior, URDF FK, VBC and CUDA GCDF run together.
Only joint/trajectory/filtered-ray inputs are fixtures; no fake confidence/CDF.
No planner, tracker, Gazebo, actuator or learned visibility search is started.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import traceback
import xmlrpc.client
from phase_e_acceptance_scope import trajectory_workload

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT/'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch'
JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']


def candidate_summary_matches(data, stamp_ns):
    fields = dict(token.split('=', 1) for token in data.split() if '=' in token)
    return (fields.get('trajectory_stamp_ns') == str(stamp_ns) and
            fields.get('trajectory_source') == 'predicted' and fields.get('success') == '1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=['samples', 'primitive'], default='primitive')
    parser.add_argument('--trace-files', action='store_true', help='Audit file syscalls; timings are instrumented')
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--knots', type=int, choices=[2, 20, 21, 50], default=2,
                        help='21 states = mainline 20 intervals/1s; 50 = stress only; 2/20 = historical diagnostics')
    parser.add_argument('--execution-event-driven', action='store_true')
    parser.add_argument('--shadow-event-driven', action='store_true')
    parser.add_argument('--input-period-s', type=float, default=0., help='Test-only fixed publication cadence; 0 retains output-paced legacy fixture')
    parser.add_argument('--persistent-confidence', action='store_true', help='Opt-in VBC service transport reuse; no response caching')
    parser.add_argument('--fixture-tcp-nodelay', action='store_true', help='Diagnostic only: disable Nagle on fixture publishers/observers, not production nodes')
    parser.add_argument('--verification-tcp-nodelay', action='store_true', help='Enable production VBC input and continuity certificate transport option')
    parser.add_argument('--unknown-soak', action='store_true', help='Test workload: remove bootstrap before timed inputs and require VBC rejection on every input')
    parser.add_argument('--observer-history-limit', type=int, default=0, help='Bound retained diagnostic messages for soak; 0 preserves legacy recording')
    parser.add_argument('--closed-loop', action='store_true', help='Private simulated plant; real planner/gates/tracker, no robot')
    parser.add_argument('--install-prefix', type=Path, help='Use installed assets/executables; reject source/devel package resolution')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt', help='External fixed GCDF asset; never copied into deployment')
    args = parser.parse_args()
    prefix = args.install_prefix.resolve() if args.install_prefix else ROOT/'devel'
    def asset(package, relative):
        return (prefix/'share' if args.install_prefix else ROOT/'src')/package/relative
    launch = asset('egocentric_arm_planner', 'launch/phaseC4_4_verified_regime_planner.launch')
    checkpoint = args.checkpoint.resolve()
    if args.repetitions < 6:
        parser.error('at least six repetitions required')
    if args.observer_history_limit and args.observer_history_limit < 16:
        parser.error('observer history must be 0 (unbounded) or at least 16')
    if args.closed_loop and (args.unknown_soak or args.observer_history_limit):
        parser.error('soak options apply to the non-actuating trajectory fixture only')
    if not math.isfinite(args.input_period_s) or not 0 <= args.input_period_s <= 1:
        parser.error('input period must be finite and within [0, 1] seconds')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    missing = str(out/'DOES_NOT_EXIST_body_samples.yaml')
    samples = str(asset('care_confidence_map', 'config/body_samples.yaml'))
    body_file = missing if args.backend == 'primitive' else samples
    uri = 'http://127.0.0.1:'+str(port)
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost', ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    children, handles, timers = [], [], []
    report = dict(status='FAIL', backend=args.backend, checks={}, failure_stage='startup',
                  workload=trajectory_workload(args.knots, args.closed_loop),
                  scope=__doc__, instrumented=args.trace_files, body_samples_file=body_file)

    def start(cmd, label, trace=True):
        if args.trace_files and trace:
            cmd = ['strace', '-f', '-qq', '-e', 'trace=%file', '-o', str(out/(label+'.files'))]+cmd
        handle = (out/(label+'.log')).open('x'); handles.append(handle)
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(proc); return proc

    def wait(predicate, label, seconds=20):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError('Timeout: '+label)

    def check(name):
        report['checks'][name] = 'PASS'
        print('PASS '+name, flush=True)

    try:
        import roslaunch.config
        if args.install_prefix:
            import rospkg
            packages = ['arm_model', 'arm_trajectory', 'arm_description', 'care_confidence_map', 'care_collision_cdf', 'care_visibility_cdf', 'egocentric_arm_planner']
            resolved = {p: Path(rospkg.RosPack().get_path(p)).resolve() for p in packages}
            assert all(path == prefix/'share'/p for p, path in resolved.items()), resolved
            report['installed_packages'] = {p: str(path) for p, path in resolved.items()}
            check('install_space_package_resolution_no_source_fallback')
        flags = ['geometry_backend:='+args.backend, 'body_samples_file:='+body_file,
                 'verification_tcp_nodelay:='+str(args.verification_tcp_nodelay).lower(),
                 'vbc_confidence_query_persistent:='+str(args.persistent_confidence).lower(),
                 'execution_vbc_event_driven_eval:='+str(args.execution_event_driven).lower(),
                 'cdf_shadow_vbc_event_driven_eval:='+str(args.shadow_event_driven).lower(),
                 'confidence_map_config_file:='+str(asset('care_confidence_map', 'config/confidence_map_phase_e_ray.yaml')),
                 'use_local_sparse_scp:=true', 'final_executable_gcdf_enabled:=true',
                 'execution_gcdf_audit_enabled:=true', 'cdf_shadow_vbc_audit_enabled:=true',
                 'gcdf_body_inflation_m:=0.015',
                 'local_scp_gpu_socket:='+str(out/'gpu.sock'), 'local_scp_selector_jsonl:='+str(out/'selector.jsonl'),
                 # Input-only fixture namespace: never publish a committed/actuator topic.
                 'execution_vbc_trajectory_topic:=/unified_test/execution_trajectory']
        if args.closed_loop:
            report['scope'] = 'Private simulated plant, real planner/GCDF/VBC/commit/tracker/limiter and self-filter; no physical robot'
            flags += ['config_file:='+str(asset('egocentric_arm_planner', 'config/planner_c5_5_vbc_gcdf_regime.yaml')),
                      'tof_fusion_enabled:=true',
                      'commit_pipeline_candidate_topic:=/care_planner/local_planner/candidate_trajectory',
                      'committed_trajectory_topic:=/unified_test/committed',
                      'execution_vbc_trajectory_topic:=/unified_test/committed']
        cfg = roslaunch.config.load_config_default([(str(launch), flags)], None, verbose=False)
        if report['workload']['matches_mainline_horizon']:
            assert cfg.params['/local_sparse_scp_planner_node/local_planner/num_intervals'].value == 20
            assert cfg.params['/local_sparse_scp_planner_node/local_planner/horizon_duration'].value == 1.
        defaults = roslaunch.config.load_config_default([str(launch)], None, verbose=False)
        assert all(p.value == 'primitive' for k, p in defaults.params.items() if k.endswith('/geometry_backend'))
        assert all(p.value == args.backend for k, p in cfg.params.items() if k.endswith('/geometry_backend'))
        sample_params = [p.value for k, p in cfg.params.items() if k.endswith('/body_samples_file')]
        if args.backend == 'primitive':
            assert not sample_params, 'primitive launch leaked legacy YAML parameters'
        else:
            assert sample_params and all(value == body_file for value in sample_params)
        assert cfg.params['/c5_4_local_scp_cdf_selector/max_pairs'].value == 8000
        assert cfg.params['/confidence_map_node/confidence_map/query_dedicated_queue'].value is True
        report['map_query_dedicated_queue'] = True
        assert cfg.params['/c5_4_local_scp_cdf_selector/max_pairs_per_step'].value == 250
        assert cfg.params['/c5_4_local_scp_cdf_selector/proximity_margin'].value == .025
        assert cfg.params['/c5_4_local_scp_cdf_selector/execution_proximity_margin'].value == .12
        shadow_prefix = '/cdf_shadow_vbc/trajectory_vbc_selector_node/trajectory_vbc/'
        assert cfg.params[shadow_prefix+'event_driven_eval'].value == args.shadow_event_driven
        for name in ['/trajectory_vbc_selector_node/', '/execution_vbc_audit/trajectory_vbc_selector_node/',
                     '/cdf_shadow_vbc/trajectory_vbc_selector_node/']:
            assert cfg.params[name+'trajectory_vbc/tcp_nodelay'].value == args.verification_tcp_nodelay
            assert cfg.params[name+'trajectory_vbc/confidence_query_persistent'].value == args.persistent_confidence
        assert cfg.params['/optimized_trajectory_continuity/tcp_nodelay'].value == args.verification_tcp_nodelay
        assert cfg.params[shadow_prefix+'preserve_periodic_eval'].value is True
        assert cfg.params[shadow_prefix+'eval_rate'].value == 20.
        assert cfg.params[shadow_prefix+'min_visibility_before_sweep_margin_s'].value == .3
        # Exercise old per-path A/B overrides even with unified primitive default.
        mixed = roslaunch.config.load_config_default([(str(launch), ['geometry_backend:=primitive', 'vbc_geometry_backend:=samples'])], None, verbose=False)
        assert mixed.params['/trajectory_vbc_selector_node/trajectory_vbc/geometry_backend'].value == 'samples'
        assert mixed.params['/trajectory_risk_node/trajectory_risk/geometry_backend'].value == 'primitive'
        check('unified_launch_defaults_overrides_and_safety_constants')
        selected = [n for n in cfg.nodes if n.package == 'care_confidence_map' or n.name == 'c5_4_local_scp_cdf_selector'
                    or n.namespace == '/execution_gcdf_audit/' or (args.closed_loop and n.name in
                        ['local_sparse_scp_planner_node', 'optimized_trajectory_continuity'])]
        for extra in ['body_observability.launch', 'body_sample_model.launch']:
            extra_cfg = roslaunch.config.load_config_default([(str(asset('care_confidence_map', 'launch')/extra),
                ['geometry_backend:='+args.backend, 'body_samples_file:='+body_file])], None, verbose=False)
            cfg.params.update(extra_cfg.params); selected.extend(extra_cfg.nodes)
        if args.closed_loop:
            tracker_cfg = roslaunch.config.load_config_default([(str(asset('egocentric_arm_planner', 'launch/c4_3_low_level_tracker.launch')),
                ['config_file:='+str(asset('egocentric_arm_planner', 'config/planner_c5_5_vbc_gcdf_regime.yaml')),
                 'input_trajectory:=/unified_test/committed', 'output_velocity_command:=/unified_test/desired_velocity',
                 'use_acceleration_limiter:=true', 'rate_limiter_output_velocity_command:=/unified_test/velocity'])], None, verbose=False)
            cfg.params.update(tracker_cfg.params); selected.extend(tracker_cfg.nodes)
            for n in selected:
                n.remap_args.append(('/care_arm/arm_group_velocity_controller/command', '/unified_test/velocity'))
        allowed = {'confidence_map_node', 'initialize_body_prior_once.py', 'trajectory_risk_node',
                   'gcdf_primitive_anchor_node', 'cpp_forbidden_voxel_gpu_shadow_node',
                   'trajectory_vbc_selector_temporal_cluster_node', 'predicted_vbc_auditor_node',
                   'body_observability_node', 'body_sample_model_node',
                   'execution_gcdf_measured_state_trajectory_node.py', 'execution_gcdf_safety_monitor_node.py'}
        if args.closed_loop:
            allowed |= {'local_sparse_scp_planner_node', 'optimized_trajectory_continuity_node.py',
                        'trajectory_execution_manager_node', 'joint_velocity_rate_limiter_node.py', 'tof_fusion_self_filter_node'}
        assert all(n.type in allowed for n in selected), [n.type for n in selected]
        assert len(selected) == (20 if args.closed_loop else 15), [(n.namespace, n.type) for n in selected]
        report['nodes'] = [dict(name=n.namespace+n.name, type=n.type) for n in selected]
        # Only selected nodes' parameters are loaded, not the planner or execution owner.
        params = {k: p.value for k, p in cfg.params.items()
                  if any(k.startswith(n.namespace+n.name+'/') for n in selected)}
        for n in selected:
            if n.type == 'trajectory_vbc_selector_temporal_cluster_node':
                assert params[n.namespace+n.name+'/trajectory_vbc/confidence_query_persistent'] == args.persistent_confidence
        (out/'resolved_params.json').write_text(json.dumps(params, indent=2)+'\n')
        paths = [launch, Path(__file__).resolve(), ROOT/'scripts/phase_e_acceptance_scope.py', asset('arm_description', 'urdf/Arm.urdf'),
                 asset('egocentric_arm_planner', 'launch/c5_3a_cdf_shadow_vbc_audit.launch'),
                 asset('care_confidence_map', 'launch/trajectory_vbc_selector.launch'),
                 asset('arm_description', 'urdf/Arm_with_self_filter_collision.urdf'),
                 asset('care_confidence_map', 'config/confidence_map_phase_e_ray.yaml'),
                 asset('care_confidence_map', 'config/trajectory_risk.yaml'), checkpoint,
                 prefix/'lib/libtrajectory_risk_evaluator.so', prefix/'lib/libbody_sample_model.so']
        for n in selected:
            path = prefix/'lib'/n.package/n.type
            paths.append(path.resolve())
        if args.closed_loop:
            paths.append(ROOT/'scripts/phase_e_primitive_closed_loop.py')
        def hashes():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        report['sha256'] = hashes()
        start(['roscore', '-p', str(port)], 'master', trace=False)
        def master_ready():
            try:
                return xmlrpc.client.ServerProxy(uri).getPid('/unified_test')[0] == 1
            except OSError:
                return False
        wait(master_ready, 'private master')
        import rospy
        from std_msgs.msg import Header, Bool, String
        from sensor_msgs.msg import JointState, PointCloud2, PointField
        from sensor_msgs import point_cloud2
        from geometry_msgs.msg import Point, PointStamped
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from visualization_msgs.msg import MarkerArray, Marker
        from std_srvs.srv import Trigger
        from care_confidence_map.srv import QueryConfidence
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from care_collision_cdf.msg import CollisionCDFWitnessRequest as Req, CollisionCDFWitnessResponse as Resp
        rospy.init_node('unified_primitive_acceptance', disable_signals=True)
        for key, value in params.items():
            rospy.set_param(key, value)
        rospy.set_param('/robot_description', asset('arm_description', 'urdf/Arm.urdf').read_text())
        start(['rosrun', 'robot_state_publisher', 'robot_state_publisher', 'joint_states:=/care_arm/joint_states'], 'fk')
        report['failure_stage'] = 'CUDA worker startup'
        worker = prefix/'lib/care_collision_cdf/collision_cdf_gpu_worker.py' if args.install_prefix else ROOT/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py'
        start(['/home/zhicheng/miniconda3/envs/viscdf/bin/python', str(worker),
               '--checkpoint', str(checkpoint),
               '--device', 'cuda', '--socket', str(out/'gpu.sock'), '--warmup-pairs', '8'], 'worker')
        wait(lambda: (out/'gpu.sock').exists(), 'CUDA worker', 40)
        for n in selected:
            cmd = [str(prefix/'lib'/n.package/n.type), '__name:='+n.name, '__ns:='+n.namespace]
            cmd += shlex.split(n.args) + [a+':='+b for a, b in n.remap_args]
            start(cmd, (n.namespace+n.name).strip('/').replace('/', '_'))
        if args.closed_loop:
            from phase_e_primitive_closed_loop import run_closed_loop
            report['failure_stage'] = 'real planner/GCDF/VBC/commit/tracker loop'
            report['closed_loop'] = run_closed_loop(out)
            assert report['closed_loop']['status'] == 'PASS', report['closed_loop']
            assert report['sha256'] == hashes()
            if args.trace_files and args.backend == 'primitive':
                traces = list(out.glob('*.files'))
                assert len(traces) == len(selected)+2
                assert not any('body_samples.yaml' in p.read_text() or missing in p.read_text() for p in traces)
                assert not any('libbody_sample_model.so' in p.read_text() for p in traces)
                report['file_trace_count'] = len(traces)
                check('closed_loop_no_body_samples_YAML_access')
                check('closed_loop_no_legacy_body_library_access')
            report['status'] = 'PASS'; report['failure_stage'] = None
            return
        q = [0.002*(j+1) for j in range(7)]
        joint_pub = rospy.Publisher('/care_arm/joint_states', JointState, queue_size=1, tcp_nodelay=args.fixture_tcp_nodelay)
        def joints(_=None):
            joint_pub.publish(JointState(header=Header(stamp=rospy.Time.now()), name=JOINTS, position=q))
        joint_timer = rospy.Timer(rospy.Duration(.025), joints); timers.append(joint_timer)
        collected, subs, pubs, received = {}, [], {}, {}
        def subscribe(key, topic, kind):
            collected[key] = []
            received[key] = 0
            def record(message):
                collected[key].append((time.monotonic(), message))
                received[key] += 1
                # Retention only: ROS queue, evaluation and every latency row
                # remain unchanged. Keep small hold histories for slice checks.
                if args.observer_history_limit and key not in ['hold', 'safety']:
                    del collected[key][:-args.observer_history_limit]
            subs.append(rospy.Subscriber(topic, kind, record, queue_size=100, tcp_nodelay=args.fixture_tcp_nodelay))
        subscribe('map', '/care_planner/confidence_map/points', PointCloud2)
        subscribe('ready', '/care_planner/confidence_map/initial_prior_ready', Bool)
        subscribe('viz', '/care_planner/body_samples/world_markers', MarkerArray)
        subscribe('obs', '/care_planner/body_observability/summary', String)
        subscribe('risk', '/care_planner/trajectory_risk/summary', String)
        subscribe('audit', '/care_planner/execution/predicted_vbc_audit_summary', String)
        subscribe('candidate', '/care_planner/candidate_vbc/summary', String)
        subscribe('execution_vbc', '/care_planner/execution_vbc/summary', String)
        subscribe('shadow', '/care_planner/cdf_shadow_vbc/summary', String)
        subscribe('hold', '/care_planner/execution_gcdf/hard_hold', Bool)
        subscribe('safety', '/care_planner/execution_gcdf/safety_summary', String)
        channels = {
            'local': ('/care_planner/local_planner/scp_query_trajectory', '/care_planner/local_planner/body_sweep_anchors', '/care_planner/local_planner/cdf_constraint_batch'),
            'final': ('/care_planner/final_gcdf/query_trajectory', '/care_planner/final_gcdf/body_sweep_anchors', '/care_planner/final_gcdf/constraint_batch'),
            'execution': ('/care_planner/execution_gcdf/measured_trajectory', '/care_planner/execution_gcdf/body_sweep_anchors', '/care_planner/execution_gcdf/constraint_batch')}
        for channel, (topic, anchor, batch) in channels.items():
            subscribe(channel+'_anchors', anchor, PointCloud2); subscribe(channel+'_batch', batch, Batch)
            if channel != 'execution':
                pubs[channel] = rospy.Publisher(topic, JointTrajectory, queue_size=1, tcp_nodelay=args.fixture_tcp_nodelay)
        subscribe('witness', channels['local'][2]+'/witness_response', Resp)
        reqpub = rospy.Publisher(channels['local'][2]+'/witness_request', Req, queue_size=1)
        for key, topic in [('task', '/care_planner/task_trajectory'), ('candidate', '/care_planner/optimized_trajectory'),
                           ('execution_vbc', '/unified_test/execution_trajectory'), ('shadow', '/care_planner/mpc/cdf_shadow_predicted_trajectory')]:
            pubs[key] = rospy.Publisher(topic, JointTrajectory, queue_size=1, tcp_nodelay=args.fixture_tcp_nodelay)
        target_pub = rospy.Publisher('/care_planner/active_sensing/target_point', PointStamped, queue_size=1, latch=True)
        active_pub = rospy.Publisher('/care_planner/active_sensing/visibility_waypoint_active', Bool, queue_size=1, latch=True)
        report['failure_stage'] = 'real FK and cold-start'
        wait(lambda: collected['ready'] and collected['ready'][-1][1].data and collected['map'], 'actual FK cold-start ready')
        refresh = rospy.ServiceProxy('/care_planner/confidence_map/refresh_body_prior', Trigger)
        deact = rospy.ServiceProxy('/care_planner/confidence_map/deactivate_body_prior', Trigger)
        assert 'locked' in refresh().message
        wait(lambda: collected['viz'] and sum(m.action == Marker.ADD and m.type != Marker.TEXT_VIEW_FACING
             for m in collected['viz'][-1][1].markers) == (26 if args.backend == 'primitive' else 90), 'complete body markers')
        wait(lambda: all(p.get_num_connections() for p in pubs.values()) and reqpub.get_num_connections()
             and all(s.get_num_connections() for s in subs), 'all input/output subscribers')
        import rosgraph
        master = rosgraph.Master('/unified_primitive_acceptance')
        anchor_node = 'gcdf_primitive_anchors' if args.backend == 'primitive' else 'trajectory_risk_node'
        required_edges = {
            '/local_scp_pair_export/'+anchor_node: [channels['local'][0]],
            '/final_executable_gcdf_pair_export/'+anchor_node: [channels['final'][0]],
            '/c5_4_local_scp_cdf_selector': [channels['local'][1], channels['final'][1]],
            '/trajectory_vbc_selector_node': ['/care_planner/optimized_trajectory'],
            '/execution_vbc_audit/trajectory_vbc_selector_node': ['/unified_test/execution_trajectory'],
            '/cdf_shadow_vbc/trajectory_vbc_selector_node': ['/care_planner/mpc/cdf_shadow_predicted_trajectory'],
        }
        def production_edges_ready():
            for node, topics in required_edges.items():
                try:
                    code, _, rows = xmlrpc.client.ServerProxy(master.lookupNode(node)).getBusInfo('/unified_primitive_acceptance')
                except (OSError, rosgraph.MasterError):
                    return False
                connected = {r[4] for r in rows if r[2] == 'i' and r[5]} if code == 1 else set()
                if not set(topics) <= connected:
                    return False
            return True
        wait(production_edges_ready, 'production input/anchor TCPROS edges')
        def all_fixture_publishers_connected():
            subscriptions = dict(master.getSystemState()[1])
            return all(p.get_num_connections() >= len(subscriptions.get(p.resolved_name, []))
                       for p in list(pubs.values())+[reqpub])
        wait(all_fixture_publishers_connected, 'all fixture publisher-side handshakes')
        report['production_edges_ready'] = required_edges
        check('real_FK_map_one_shot_prior_locked_and_full_markers')
        target_pub.publish(PointStamped(header=Header(frame_id='base_link'), point=Point(.1, .1, .3)))
        active_pub.publish(Bool(True))
        def trajectory():
            return JointTrajectory(header=Header(stamp=rospy.Time.now(), frame_id='base_link'), joint_names=JOINTS[::-1],
                points=[JointTrajectoryPoint(positions=q[::-1], time_from_start=rospy.Duration(k*.05)) for k in range(args.knots)])
        def send(msg):
            reqpub.publish(Req(header=msg.header, plan_sequence=1, mode_epoch=2, target_revision=3, progress_epoch=4,
                               dof=7, original_timestep=[1], point_flat=[.1, .1, .3], q_flat=q))
            for key, pub in pubs.items():
                if key == 'task':
                    # Bootstrap and candidate are different logical inputs, even
                    # with identical q. Never accept a bootstrap timer verdict
                    # as the candidate query's stamp-matched latency result.
                    task = copy.deepcopy(msg)
                    task.header.stamp -= rospy.Duration(0, 1)
                    pub.publish(task)
                else:
                    pub.publish(msg)
        report['failure_stage'] = 'combined known/unknown trajectory evaluation'
        if args.unknown_soak:
            assert deact().success
            def fully_unknown():
                rows = point_cloud2.read_points(collected['map'][-1][1], field_names=['confidence'])
                return sum(value < .5 for (value,) in rows) >= 33000
            wait(fully_unknown, 'high UNKNOWN map after bootstrap removal')
            check('soak_starts_with_at_least_33000_unknown_voxels')
        latency = []; node_timings = []; lineage = []; shadow_lineage = []; input_schedule = []
        schedule_origin = time.monotonic()
        for iteration in range(args.repetitions+2):
            due = schedule_origin+iteration*args.input_period_s
            if args.input_period_s:
                time.sleep(max(0., due-time.monotonic()))
            msg = trajectory(); stamp = msg.header.stamp; started = time.monotonic(); send(msg)
            def matching(key):
                return next(((t, m) for t, m in collected[key] if
                    (m.header.stamp == stamp if hasattr(m, 'header') else
                     candidate_summary_matches(m.data, stamp.to_nsec()) if key in ['candidate', 'shadow'] else
                     'trajectory_stamp_ns='+str(stamp.to_nsec())+' ' in m.data+' ')), None)
            wanted = ['local_batch', 'final_batch', 'candidate', 'execution_vbc', 'shadow']
            wait(lambda: all(matching(key) for key in wanted), 'stamp-bound combined outputs')
            if iteration >= 2:
                latency.append({key: (matching(key)[0]-started)*1000 for key in wanted})
                row = {}
                for key in ['candidate', 'execution_vbc', 'shadow']:
                    fields = dict(token.split('=', 1) for token in matching(key)[1].data.split() if '=' in token)
                    row[key] = {name: float(fields[name]) for name in ['input_wait_ms', 'total_eval_ms',
                        'body_fk_ms', 'swept_voxel_build_ms', 'confidence_query_ms', 'confidence_ready_ms', 'confidence_rpc_ms']}
                    row[key]['query_points'] = int(fields['swept_voxel_query_count'])
                    row[key]['transport_and_publish_residual_ms'] = latency[-1][key]-row[key]['input_wait_ms']-row[key]['total_eval_ms']
                node_timings.append(row)
                fields = dict(token.split('=', 1) for token in matching('candidate')[1].data.split() if '=' in token)
                if args.unknown_soak:
                    assert fields.get('has_violation') == '1', 'UNKNOWN soak input incorrectly declared clear'
                lineage.append({key: fields[key] for key in ['trajectory_stamp_ns', 'trajectory_source', 'evaluation_trigger']})
                fields = dict(token.split('=', 1) for token in matching('shadow')[1].data.split() if '=' in token)
                shadow_lineage.append({key: fields[key] for key in ['trajectory_stamp_ns', 'trajectory_source', 'evaluation_trigger']})
                input_schedule.append(dict(offset_s=started-schedule_origin,
                    lateness_ms=max(0., started-due)*1000 if args.input_period_s else None))
            if iteration == 0:
                for key in ['local', 'final']:
                    cloud = next(m for _, m in collected[key+'_anchors'] if m.header.stamp == stamp)
                    assert ('primitive_schema' in {f.name for f in cloud.fields}) == (args.backend == 'primitive')
                    if args.backend == 'primitive':
                        assert cloud.width == 21*args.knots
                        assert all(max(abs(a-b) for a, b in zip(row, q)) < 1e-7 for row in
                            point_cloud2.read_points(cloud, field_names=['q'+str(j) for j in range(7)]))
                report['initial_summaries'] = {k: matching(k)[1].data for k in ['candidate', 'execution_vbc', 'shadow']}
        report['latency_rows_ms'] = latency
        report['knots'] = args.knots
        report['execution_event_driven'] = args.execution_event_driven
        report['persistent_confidence'] = args.persistent_confidence
        report['shadow_event_driven'] = args.shadow_event_driven
        report['fixture_tcp_nodelay'] = args.fixture_tcp_nodelay
        report['verification_tcp_nodelay'] = args.verification_tcp_nodelay
        report['unknown_soak'] = args.unknown_soak
        report['observer_history_limit'] = args.observer_history_limit
        report['timed_duration_s'] = time.monotonic()-schedule_origin
        report['input_period_s'] = args.input_period_s
        report['input_schedule'] = input_schedule
        report['measurement_schema'] = 'separate_bootstrap_candidate_shadow_predicted_v2'
        report['candidate_lineage_rows'] = lineage
        report['shadow_lineage_rows'] = shadow_lineage
        report['node_timing_rows_ms'] = node_timings
        def percentile(values, quantile):
            values = sorted(values); return values[max(0, math.ceil(quantile*len(values))-1)]
        report['latency_ms'] = {key: {p: percentile([r[key] for r in latency], v) for p, v in [('p50', .5), ('p95', .95)]}
                                for key in latency[0]}
        check('shared_real_map_three_GCDF_paths_three_VBC_paths_stamp_bound')
        wait(lambda: collected['obs'] and collected['risk'] and collected['audit'] and collected['execution_batch'], 'diagnostics and measured-q audit')
        report['diagnostic_summaries'] = {k: collected[k][-1][1].data for k in ['obs', 'risk', 'audit']}
        assert 'success=1' in report['diagnostic_summaries']['risk']
        assert 'diagnostic_only=1' in report['diagnostic_summaries']['audit']
        check('diagnostic_nodes_share_real_map_and_FK')
        report['failure_stage'] = 'ray occupancy and bootstrap deactivation'
        ray_pub = rospy.Publisher('/care_planner/perception/tof_ray_observations', PointCloud2, queue_size=1)
        wait(lambda: ray_pub.get_num_connections(), 'real map ray subscriber')
        fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(['x', 'y', 'z', 'origin_x', 'origin_y', 'origin_z'])]
        fields += [PointField(name='sensor_id', offset=24, datatype=PointField.INT32, count=1),
                   PointField(name='hit', offset=28, datatype=PointField.UINT8, count=1)]
        # Post-self-filter input fixture, not a fabricated map/certificate or a ToF self-filter test.
        ray_pub.publish(point_cloud2.create_cloud(Header(stamp=rospy.Time.now(), frame_id='base_link'), fields,
                                                 [[.1, .1, .3, .1, .1, .1, 0, 1]]))
        def occupied():
            return any(row[3] > .5 for row in point_cloud2.read_points(collected['map'][-1][1], field_names=['x', 'y', 'z', 'occupancy']))
        wait(occupied, 'real ray-created OCCUPIED voxel')
        assert deact().success and 'locked' in refresh().message
        time.sleep(.15)
        msg = trajectory(); send(msg)
        wait(lambda: any(m.header.stamp == msg.header.stamp for _, m in collected['final_batch']), 'post-deactivation GCDF')
        wait(lambda: any('trajectory_stamp_ns='+str(msg.header.stamp.to_nsec())+' ' in m.data+' ' and 'has_violation=1' in m.data
                         for _, m in collected['candidate']), 'UNKNOWN VBC violation, not SAFE')
        assert occupied(), 'deactivation erased genuine sensor occupancy'
        wait(lambda: any(b.num_pairs > 0 for _, b in collected['execution_batch']), 'nonempty measured OCCUPIED audit')
        wait(lambda: any(r.request.header.stamp == msg.header.stamp for _, r in collected['witness']), 'stamp-bound witness response')
        witness = next(r for _, r in collected['witness'] if r.request.header.stamp == msg.header.stamp)
        report['failure_stage'] = 'post-deactivation witness reevaluation'
        # Preserve negative evidence before asserting; a budget error must not
        # disappear behind a bare AssertionError or be relabelled as success.
        report['witness_status'] = list(witness.status)
        report['witness_diagnostic'] = dict(request_stamp_ns=witness.request.header.stamp.to_nsec(),
            batch_stamp_ns=witness.batch.header.stamp.to_nsec(), batch_pairs=witness.batch.num_pairs,
            original_timestep=list(witness.request.original_timestep), point=list(witness.request.point_flat),
            q=list(witness.request.q_flat))
        assert len(witness.status) == 1 and witness.status[0] in ['evaluated_unknown', 'evaluated_occupied', 'resolved_free'], report['witness_status']
        assert witness.batch.header.stamp == msg.header.stamp
        check('real_ray_occupancy_survives_deactivation_unknown_VBC_rejected')
        refs = []
        for key in ['local_batch', 'final_batch', 'execution_batch']:
            for _, batch in collected[key]:
                for i in range(batch.num_pairs):
                    if key == 'execution_batch':
                        assert batch.source_type[i] == 1, 'UNKNOWN entered measured collision hard-hold set'
                    refs.append(dict(point=list(batch.point_flat[3*i:3*i+3]), q=list(batch.q_linearization_flat[7*i:7*i+7]),
                                     distance=batch.distance[i], gradient=list(batch.gradient_flat[7*i:7*i+7])))
        unique = {tuple(r['q']+r['point']): r for r in refs}
        assert unique, 'vacuous zero-pair GCDF test'
        (out/'gpu_rows.json').write_text(json.dumps(list(unique.values()), indent=2)+'\n')
        verify = subprocess.run(['/home/zhicheng/miniconda3/envs/viscdf/bin/python', str(ROOT/'scripts/test_phase_e_witness_pipeline.py'),
                                 '--verify-json', str(out/'gpu_rows.json')], capture_output=True, text=True)
        (out/'gradient.log').write_text(verify.stdout+verify.stderr)
        assert verify.returncode == 0, 'raw 7D GCDF gradient oracle failed'
        report['gradient'] = json.loads(verify.stdout.splitlines()[-1]); check('real_CUDA_nonempty_full_7D_gradient_oracle')
        if args.backend == 'primitive':
            report['failure_stage'] = 'malformed final trajectory rejection'
            bad = trajectory(); bad.points[0].positions[0] = float('nan'); pubs['final'].publish(bad)
            time.sleep(.25)
            assert not any(m.header.stamp == bad.header.stamp for _, m in collected['final_batch'])
            assert not any(m.header.stamp == bad.header.stamp for _, m in collected['final_anchors'])
            check('invalid_final_q_no_anchor_no_SAFE_batch')
        report['failure_stage'] = 'fail-closed measured state staleness'
        joint_timer.shutdown(); count = len(collected['hold']); safety_count = len(collected['safety'])
        wait(lambda: any(m.data for _, m in collected['hold'][count:]), 'stale measured state hard hold')
        wait(lambda: any('stale_hold=1' in m.data for _, m in collected['safety'][safety_count:]), 'explicit stale hold reason')
        report['stale_summary'] = collected['safety'][-1][1].data
        check('unchanged_execution_monitor_fails_closed_on_stale_measured_q')
        topics = rospy.get_published_topics()
        assert not any('command' in t or 'committed_trajectory' in t for t, _ in topics), topics
        assert all(p.poll() is None for p in children), 'unexpected child exit'
        assert not Path(missing).exists() and report['sha256'] == hashes()
        check('no_actuator_no_committed_publisher_immutable_inputs')
        if args.trace_files and args.backend == 'primitive':
            traces = list(out.glob('*.files')); assert len(traces) == len(selected)+2
            bad = [line for path in traces for line in path.read_text().splitlines()
                   if 'body_samples.yaml' in line or missing in line or 'libbody_sample_model.so' in line]
            assert not bad, bad[:10]
            report['file_trace_count'] = len(traces); check('strace_no_body_samples_file_access_in_runtime_process_tree')
        report['status'] = 'PASS'; report['failure_stage'] = None
    except Exception as exc:
        report['error'] = repr(exc); report['traceback'] = traceback.format_exc()
        if 'collected' in locals():
            report['observed_counts'] = {k: len(v) for k, v in collected.items()}
            report['received_counts'] = received
            report['last_observed'] = {k: (v[-1][1].data if hasattr(v[-1][1], 'data') and
                isinstance(v[-1][1].data, str) else str(getattr(v[-1][1], 'header', ''))) for k, v in collected.items() if v}
        raise
    finally:
        for timer in timers:
            timer.shutdown()
        for proc in reversed(children):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
        for handle in handles:
            handle.close()
        (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ['sha256', 'latency_rows_ms', 'node_timing_rows_ms', 'closed_loop']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
