"""Benchmark-only startup barrier. It cannot release any execution safety gate."""
import time
import uuid
import json

ARM_TOPIC = '/care_planner/benchmark/arm'


def prepare_arm(required_recorders, timeout_s=20.0, admission_ping_topic=None):
    import rosgraph
    import rospy
    from std_msgs.msg import String
    pub = rospy.Publisher(ARM_TOPIC, String, queue_size=1, latch=True)
    master = rosgraph.Master(rospy.get_name())
    deadline = time.monotonic() + timeout_s
    missing = set(required_recorders)
    arm_subscribers = []
    refreshed = False
    publisher_registered = False
    if admission_ping_topic is None:
        enabled = (rospy.get_param('/optimized_trajectory_continuity/candidate_admission_enabled', False)
                   if hasattr(rospy, 'get_param') else False)
        admission_ping_topic = (rospy.get_param('/care_planner/benchmark/candidate_admission_ping_topic', '')
                                if enabled else '')
        if enabled and not admission_ping_topic:
            raise RuntimeError('benchmark_not_armed: candidate admission enabled without ping topic')
    ready = dict(received=None)
    nonce = uuid.uuid4().hex
    ping_pub = ready_sub = None
    retry_pub = retry_sub = None
    retry_ready = dict(nonce=None, received=None, sent=None)
    if hasattr(rospy, 'get_param') and rospy.get_param('/care_planner/benchmark/qp_retry_readiness_required', False):
        def on_retry_ready(msg):
            if msg.data == retry_ready['nonce']:
                retry_ready['received'] = time.monotonic()
        retry_sub = rospy.Subscriber('/care_planner/benchmark/qp_retry_ping/ready',
                                     String, on_retry_ready, queue_size=4)
        retry_pub = rospy.Publisher('/care_planner/benchmark/qp_retry_ping',
                                    String, queue_size=4, latch=False)
    last_ping = -float('inf')
    actuator = None
    observation = None
    critical = None
    topics = rospy.get_param('/care_planner/benchmark/critical_recorder_topics', []) if hasattr(rospy, 'get_param') else []
    if topics:
        from phase_e_critical_recorders import CriticalRecorderReadiness
        critical = CriticalRecorderReadiness(required_recorders, topics, rospy, master.master_uri)
    if hasattr(rospy, 'get_param') and rospy.get_param('/care_planner/benchmark/observation_readiness_required', False):
        from phase_e_observation_readiness import ObservationReadiness
        observation = ObservationReadiness(required_recorders, rospy, master.master_uri)
    if hasattr(rospy, 'get_param') and rospy.get_param('/care_planner/benchmark/actuator_readiness_required', False):
        from phase_e_actuator_readiness import ActuatorReadiness
        actuator = ActuatorReadiness(rospy.get_param('/care_planner/benchmark/actuator_readiness'),
                                     rospy, master.master_uri)
    if admission_ping_topic:
        def on_ready(msg):
            fields = dict(w.split('=', 1) for w in msg.data.split() if '=' in w)
            if fields.get('ready_nonce') == nonce and fields.get('receiver_boot_id'):
                ready['received'] = time.monotonic()
        ready_sub = rospy.Subscriber(admission_ping_topic + '/ready', String, on_ready, queue_size=4)
        ping_pub = rospy.Publisher(admission_ping_topic, String, queue_size=4, latch=False)
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        from scene_guard_status import read as read_scene_guard
        scene_guard = read_scene_guard()
        if scene_guard.get('abort'):
            raise RuntimeError('static_scene_guard_abort: '+json.dumps(scene_guard['abort']))
        try:
            now = time.monotonic()
            if retry_pub is not None and (retry_ready['sent'] is None or now - retry_ready['sent'] >= .25):
                retry_ready['nonce'] = uuid.uuid4().hex
                retry_ready['sent'] = now
                retry_pub.publish(String(data=retry_ready['nonce']))
            control_ready = (retry_pub is None or (retry_ready['received'] is not None and
                                                  now - retry_ready['received'] <= .5))
            if ping_pub is not None and now - last_ping >= .25:
                last_ping = now
                ping_pub.publish(String(data=nonce))
            admission_ready = (ping_pub is None or
                (ready['received'] is not None and now - ready['received'] <= .50))
            publishers, subscribers, _ = master.getSystemState()
            registered = {name for _, nodes in subscribers for name in nodes}
            missing = set(required_recorders) - registered
            critical_ready = critical is None or critical.check(deadline, publishers, subscribers)
            actuator_ready = actuator is None or actuator.check(deadline)
            observation_ready = observation is None or observation.check(deadline, subscribers)
            if (not missing and pub.get_num_connections() > 0 and admission_ready and
                    actuator_ready and control_ready and observation_ready and critical_ready and scene_guard['ready'] and time.monotonic() < deadline):
                if critical is not None:
                    pub.care_critical_recorder_readiness = critical.last
                    print('[BENCHMARK CRITICAL RECORDERS READY] ' + json.dumps(critical.last), flush=True)
                if observation is not None:
                    pub.care_observation_readiness = observation.last
                    print('[BENCHMARK OBSERVATION READY] ' + json.dumps(observation.last), flush=True)
                if retry_sub is not None:
                    pub.care_qp_retry_readiness = dict(ready=True, path='benchmark->regime->planner->regime->benchmark',
                        acknowledgement_age_s=now-retry_ready['received'])
                    retry_sub.unregister()
                    retry_pub.unregister()
                if ready_sub is not None:
                    ready_sub.unregister()
                    ping_pub.unregister()
                if actuator is not None:
                    pub.care_actuator_readiness = actuator.last
                    print('[BENCHMARK ACTUATOR READY] ' + json.dumps(actuator.last), flush=True)
                    actuator.close()
                return pub
            topic = getattr(pub, 'resolved_name', ARM_TOPIC)
            arm_subscribers = [name for t, nodes in subscribers if t == topic for name in nodes]
            publisher_registered = any(t == topic and rospy.get_name() in nodes
                                       for t, nodes in publishers)
            if not missing and arm_subscribers and not refreshed:
                # Re-notify subscribers of this SAME local publisher. This
                # sends no arm token and cannot start the test: a real TCP
                # connection and all recorders are still required above.
                # Keep the original startup deadline and refresh at most once.
                refreshed = True
                master.registerPublisher(topic, String._type, rospy.get_node_uri())
                print('[BENCHMARK ARM] refreshed own publisher registration; '
                      'registered={} subscribers={}'.format(
                          publisher_registered, arm_subscribers), flush=True)
        except Exception:
            pass
        time.sleep(.05)
    if ready_sub is not None:
        ready_sub.unregister()
        ping_pub.unregister()
    if retry_sub is not None:
        retry_sub.unregister()
        retry_pub.unregister()
    if actuator is not None:
        actuator.close()
    raise RuntimeError('benchmark_not_armed: missing_recorders={} arm_subscribers={} '
                       'publisher_registered={} connections={} refreshed={} admission_ready={} actuator_evidence={} qp_retry_ready={} observation_evidence={} critical_recorder_evidence={}'.format(
                           sorted(missing), arm_subscribers, publisher_registered,
                           pub.get_num_connections(), refreshed,
                           ready['received'] is not None if ping_pub is not None else 'not_required',
                           actuator.last if actuator is not None else 'not_required',
                           retry_ready if retry_pub is not None else 'not_required',
                           observation.last if observation is not None else 'not_required',
                           critical.last if critical is not None else 'not_required'))
