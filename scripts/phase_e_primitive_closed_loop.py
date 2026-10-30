"""Private simulated plant fixture; never connect its velocity topic to a robot.

Called only after test_phase_e_unified_primitives launches an allowlisted
production graph on a fresh private ROS master. Certificates are never mocked.
"""
import json
import math
import threading
import time


def run_closed_loop(out):
    import rospy
    import rosgraph
    from std_msgs.msg import Header, Bool, String, Float64MultiArray
    from sensor_msgs.msg import JointState, PointCloud2
    from sensor_msgs import point_cloud2
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_srvs.srv import Trigger
    from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
    joints = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']
    initial = [.002*(j+1) for j in range(7)]
    q = initial[:]; velocity = [0.]*7; lock = threading.Lock()
    observed, subscribers, timers = {}, [], []
    report = dict(status='FAIL', checks=[], scope='real planner/gates/tracker + synthetic velocity-integrating plant; not a physical robot')
    def sub(key, topic, kind):
        observed[key] = []
        subscribers.append(rospy.Subscriber(topic, kind, lambda m: observed[key].append((time.monotonic(), m)), queue_size=100))
    def wait(predicate, label, seconds=15):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            if predicate(): return
            time.sleep(.01)
        raise AssertionError('closed loop timeout: '+label)
    def event(key, predicate):
        return any(predicate(m) for _, m in observed[key])
    def velocity_cb(msg):
        if len(msg.data) == 7 and all(math.isfinite(v) for v in msg.data):
            with lock: velocity[:] = msg.data
    subscribers.append(rospy.Subscriber('/unified_test/velocity', Float64MultiArray, velocity_cb, queue_size=1))
    for key, topic, kind in [
        ('ready', '/care_planner/confidence_map/initial_prior_ready', Bool),
        ('candidate', '/care_planner/local_planner/candidate_trajectory', JointTrajectory),
        ('local_query', '/care_planner/local_planner/scp_query_trajectory', JointTrajectory),
        ('local_batch', '/care_planner/local_planner/cdf_constraint_batch', Batch),
        ('final_query', '/care_planner/final_gcdf/query_trajectory', JointTrajectory),
        ('final_batch', '/care_planner/final_gcdf/constraint_batch', Batch),
        ('verified', '/care_planner/optimized_trajectory', JointTrajectory),
        ('vbc', '/care_planner/candidate_vbc/summary', String),
        ('commit', '/unified_test/committed', JointTrajectory),
        ('outcome', '/care_planner/verification_outcome', String),
        ('planner', '/care_planner/local_planner/summary', String),
        ('tracker', '/care_planner/execution/tracker_summary', String),
        ('safety', '/care_planner/execution_gcdf/safety_summary', String),
        ('desired', '/unified_test/desired_velocity', Float64MultiArray),
        ('limited', '/unified_test/velocity', Float64MultiArray),
        ('rays', '/care_planner/perception/tof_ray_observations', PointCloud2),
        ('fusion', '/care_planner/perception/tof_fusion_summary', String)]:
        sub(key, topic, kind)
    sub('map_e3', '/care_planner/confidence_map/e3_summary', String)
    joint_pub = rospy.Publisher('/care_arm/joint_states', JointState, queue_size=1)
    reference_pub = rospy.Publisher('/care_planner/task_trajectory', JointTrajectory, queue_size=1)
    last = [time.monotonic()]
    def plant(_):
        now = time.monotonic(); dt = min(.02, now-last[0]); last[0] = now
        with lock:
            for j in range(7): q[j] += velocity[j]*dt
            state, speed = q[:], velocity[:]
        joint_pub.publish(JointState(header=Header(stamp=rospy.Time.now()), name=joints, position=state, velocity=speed))
    plant_timer = rospy.Timer(rospy.Duration(.01), plant); timers.append(plant_timer)
    def reference(delta):
        with lock: start = q[:]
        traj = JointTrajectory(header=Header(stamp=rospy.Time.now(), frame_id='base_link'), joint_names=joints)
        for k in range(21):
            point = start[:]; point[0] += delta*k/20
            traj.points.append(JointTrajectoryPoint(positions=point, velocities=[delta]+[0.]*6, time_from_start=rospy.Duration(k*.05)))
        reference_pub.publish(traj)
        return traj
    try:
        wait(lambda: event('ready', lambda m: m.data) and reference_pub.get_num_connections() >= 2, 'prior and reference consumers')
        wait(lambda: all(s.get_num_connections() for s in subscribers), 'complete graph subscriptions')
        # Our observers connecting does not imply that production consumers are
        # connected. These queries are one-shot, so wait for the actual TCPROS
        # edges before publishing the first reference (especially under strace).
        import xmlrpc.client
        master = rosgraph.Master('/unified_closed_loop_audit')
        required_edges = {
            '/local_scp_pair_export/gcdf_primitive_anchors': ['/care_planner/local_planner/scp_query_trajectory'],
            '/final_executable_gcdf_pair_export/gcdf_primitive_anchors': ['/care_planner/final_gcdf/query_trajectory'],
            '/c5_4_local_scp_cdf_selector': ['/care_planner/local_planner/body_sweep_anchors', '/care_planner/final_gcdf/body_sweep_anchors'],
            '/local_sparse_scp_planner_node': ['/care_planner/local_planner/cdf_constraint_batch'],
            '/optimized_trajectory_continuity': ['/care_planner/local_planner/candidate_trajectory', '/care_planner/final_gcdf/constraint_batch', '/care_planner/candidate_vbc/summary'],
            '/trajectory_vbc_selector_node': ['/care_planner/optimized_trajectory'],
            '/trajectory_execution_manager_node': ['/unified_test/committed'],
            '/joint_velocity_rate_limiter': ['/unified_test/desired_velocity'],
        }
        def production_edges_ready():
            for node, topics in required_edges.items():
                try:
                    code, _, rows = xmlrpc.client.ServerProxy(master.lookupNode(node)).getBusInfo('/unified_closed_loop_audit')
                except (OSError, rosgraph.MasterError):
                    return False
                connected = {row[4] for row in rows if row[2] == 'i' and row[5]} if code == 1 else set()
                if not set(topics) <= connected:
                    return False
            return True
        wait(production_edges_ready, 'production query/certificate/commit TCPROS edges')
        report['production_edges_ready'] = required_edges
        trajectory = reference(.01)
        wait(lambda: observed['candidate'], 'real hard-QP candidate')
        wait(lambda: observed['commit'], 'real final GCDF -> exact VBC -> commit')
        wait(lambda: event('tracker', lambda m: 'active=1' in m.data), 'tracker accepted commit')
        wait(lambda: q[0] > initial[0]+.0002, 'post-commit plant progress towards positive reference')
        assert event('limited', lambda m: len(m.data) == 7 and m.data[0] > 1e-4)
        assert event('outcome', lambda m: 'committed=1' in m.data)
        for key in ['local_batch', 'final_batch', 'vbc']:
            assert observed[key], 'missing production certificate '+key
        report['checks'].append('real 20-interval hard-QP -> real LOCAL/FINAL GCDF -> VBC -> commit -> tracker -> limiter -> measured-state feedback')
        report['q_after_commit'] = q[:]
        report['first_commit_stamp_ns'] = observed['commit'][0][1].header.stamp.to_nsec()
        report['committed_outcome'] = next(m.data for _, m in observed['outcome'] if 'committed=1' in m.data)
        # Exercise the unchanged self-filter with timestamped organized optical
        # clouds. No forged filtered rays or direct map writes.
        frames = ['link2_sensor1_tof_link', 'link2_sensor2_tof_link', 'link3_sensor1_tof_link', 'link3_sensor2_tof_link',
                  'link4_sensor1_tof_link', 'link4_sensor2_tof_link', 'EE_sensor1_tof_link', 'EE_sensor2_tof_link']
        topics = ['/link2_sensor1/tof/cloud', '/link2_sensor2/tof/cloud', '/link3_sensor1/tof/cloud', '/link3_sensor2/tof/cloud',
                  '/link4_sensor1/tof/cloud', '/link4_sensor2/tof/cloud', '/EE_sensor1/tof/cloud', '/EE_sensor2/tof/cloud']
        raw_pubs = [rospy.Publisher(t, PointCloud2, queue_size=1) for t in topics]
        wait(lambda: all(p.get_num_connections() for p in raw_pubs), '8 raw ToF subscribers')
        for _ in range(4):
            stamp = rospy.Time.now()-rospy.Duration(.03)
            for frame, pub in zip(frames, raw_pubs):
                cloud = point_cloud2.create_cloud_xyz32(Header(stamp=stamp, frame_id=frame), [[0., 0., .45]]*256)
                cloud.width = 16; cloud.height = 16; cloud.row_step = 16*cloud.point_step
                pub.publish(cloud)
            time.sleep(.09)
        wait(lambda: event('rays', lambda m: m.width*m.height > 0), 'real self-filter ray output')
        def fields(msg):
            return dict(x.split('=', 1) for x in msg.data.split() if '=' in x)
        wait(lambda: event('map_e3', lambda m: int(fields(m).get('last_ray_count', 0)) > 0 and
                          int(fields(m).get('last_free_cell_count', 0)) > 0), 'real map integrated filtered rays')
        report['checks'].append('all8 synthetic organized raw clouds -> unchanged TF/self-filter -> real map ray input')
        report['fusion_summary'] = observed['fusion'][-1][1].data if observed['fusion'] else ''
        report['map_ray_summary'] = next(m.data for _, m in reversed(observed['map_e3'])
            if int(fields(m).get('last_ray_count', 0)) > 0)
        # Do not wipe sensor evidence. Remove only the original bootstrap layer.
        assert rospy.ServiceProxy('/care_planner/confidence_map/deactivate_body_prior', Trigger)().success
        time.sleep(.15)
        before = len(observed['planner']); before_outcome = len(observed['outcome']); before_vbc = len(observed['vbc'])
        reference(.01)
        wait(lambda: any('has_violation=1' in m.data for _, m in observed['vbc'][before_vbc:]) or
             any('unsafe' in m.data or 'failed' in m.data or 'infeasible' in m.data for _, m in observed['planner'][before:]) or
             any('committed=0' in m.data for _, m in observed['outcome'][before_outcome:]), 'unknown/rejected planning diagnosis')
        report['checks'].append('bootstrap removal exercises real UNKNOWN/constraint rejection path (no forced success)')
        plant_timer.shutdown(); before = len(observed['safety'])
        wait(lambda: any('stale_hold=1' in m.data for _, m in observed['safety'][before:]), 'stale measured-state hold')
        wait(lambda: observed['limited'] and max(abs(v) for v in observed['limited'][-1][1].data) < 1e-8, 'limiter output zero after stale hold')
        report['checks'].append('actual execution monitor stale hold reaches tracker/limiter zero command')
        publishers = rosgraph.Master('/unified_closed_loop_audit').getSystemState()[0]
        owners = dict(publishers)
        assert owners.get('/unified_test/desired_velocity') == ['/trajectory_execution_manager_node']
        assert owners.get('/unified_test/velocity') == ['/joint_velocity_rate_limiter']
        assert owners.get('/unified_test/committed') == ['/optimized_trajectory_continuity']
        assert not any('/care_arm/' in topic and 'command' in topic for topic in owners)
        report['checks'].append('publisher ownership: gate owns commit, tracker owns desired command, limiter only bounds it; no physical actuator topic')
        report['status'] = 'PASS'
    except Exception as exc:
        report['error'] = repr(exc)
    finally:
        for timer in timers: timer.shutdown()
        report['counts'] = {k: len(v) for k, v in observed.items()}
        report['last_summaries'] = {k: [m.data[:4000] for _, m in v[-3:]] for k, v in observed.items()
                                  if v and isinstance(getattr(v[-1][1], 'data', None), str)}
        (out/'closed_loop.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
