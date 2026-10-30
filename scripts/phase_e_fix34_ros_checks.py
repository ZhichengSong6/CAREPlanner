"""Additional isolated fixtures; never imported by runtime nodes."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time


def run_checks(rospy, repo, out, config, start, wait_for):
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from care_collision_cdf.msg import CollisionCDFConstraintBatch
    from std_msgs.msg import Bool, String
    cfg = copy.deepcopy(config)
    cfg["topics"] = dict(joint_states="/fix34/joints", target_pose="/fix34/target",
                         task_trajectory="/fix34/reference", command_trajectory="/fix34/nominal")
    rospy.set_param("/fix34_ik", cfg)
    start([str(repo/"devel/lib/egocentric_arm_planner/receding_horizon_planner_node"), "__name:=fix34_ik"], "fix34_ik.log")
    refs = []; statuses = []
    rospy.Subscriber("/fix34/reference", JointTrajectory, lambda m: refs.append(m))
    rospy.Subscriber("/care_planner/task_reference_status", String,
                     lambda m: statuses.append(m.data) if m._connection_header.get("callerid") == "/fix34_ik" else None)
    states = rospy.Publisher("/fix34/joints", JointState, queue_size=1)
    targets = rospy.Publisher("/fix34/target", PoseStamped, queue_size=1)
    wait_for(lambda: states.get_num_connections() and targets.get_num_connections())
    measured = JointState(); measured.name = cfg["joint_names"]
    # Full precision historical Case008 approximate endpoint; original target.
    measured.position = [-.39457622528294967,.06971961707945959,.3822909039332365,
                         1.5478706758478369,1.6368529353383412,-.7931680567508005,-1.2]
    for _ in range(5):
        measured.header.stamp = rospy.Time.now(); states.publish(measured); time.sleep(.03)
    db = json.loads((repo/"outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json").read_text())
    case = next(c for c in db["cases"] if c["case_id"] == "phase_e_goal_008")
    target = PoseStamped(); target.header.frame_id = "base_link"
    target.pose.position.x,target.pose.position.y,target.pose.position.z = case["goal_position"]
    target.pose.orientation.x,target.pose.orientation.y,target.pose.orientation.z,target.pose.orientation.w = case["goal_orientation"]
    targets.publish(target)
    wait_for(lambda: any("status=exhausted" in s for s in statuses))
    approx = [s for s in statuses if "generation_IK_APPROXIMATE" in s]
    assert len(approx) == 3 and not refs, (statuses, len(refs))
    assert len(set(s.split("ik_position_error=")[-1] for s in approx)) > 1, "retry did not continue IK"
    # New exact measured-pose request must clear the previous IK seed.
    import evaluate_phase_d_run as ev
    from tf.transformations import quaternion_from_matrix
    helper = ev.load_fk_helper(str(repo)); robot = ev.URDF.from_xml_file(cfg["urdf_path"])
    chain = helper.find_chain_joints(robot,"base_link","EE_link")
    T = ev.fk_T(helper,chain,dict(zip(measured.name,measured.position)))
    target.pose.position.x,target.pose.position.y,target.pose.position.z = map(float,T[:3,3])
    target.pose.orientation.x,target.pose.orientation.y,target.pose.orientation.z,target.pose.orientation.w = map(float,quaternion_from_matrix(T))
    targets.publish(target)
    wait_for(lambda: len(refs) == 1)
    assert max(abs(a-b) for a,b in zip(refs[0].points[0].positions,measured.position)) < 1e-10

    # Deliberately tiny PIQP iteration cap ONLY on an isolated fixture node.
    local = copy.deepcopy(config)
    lp = local["local_planner"]
    lp.update(joint_states="/fix34/qp_joints", reference_trajectory="/fix34/qp_reference")
    lp["piqp"]["max_iterations"] = 1
    rospy.set_param("/fix34_qp", local)
    queries = []; candidates = []; stalls = []; failures = []
    batch_pub = rospy.Publisher(lp["cdf_batch_topic"], CollisionCDFConstraintBatch, queue_size=1)
    def query_cb(msg):
        queries.append(msg)
        batch = CollisionCDFConstraintBatch(); batch.header = msg.header; batch.dof = 7
        batch_pub.publish(batch)  # Explicit empty batch: no UNKNOWN or OCCUPIED blockers.
    rospy.Subscriber(lp["query_trajectory_topic"], JointTrajectory, query_cb)
    rospy.Subscriber(lp["candidate_trajectory_topic"], JointTrajectory, lambda m: candidates.append(m))
    rospy.Subscriber("/care_planner/local_planner/task_stall", String, lambda m: stalls.append(m.data))
    rospy.Subscriber("/care_planner/local_planner/task_uncertified", Bool, lambda m: failures.append(m))
    qp_states = rospy.Publisher("/fix34/qp_joints",JointState,queue_size=1)
    qp_refs = rospy.Publisher("/fix34/qp_reference",JointTrajectory,queue_size=1)
    replan = rospy.Publisher(lp["replan_request_topic"],Bool,queue_size=1)
    start([str(repo/"devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node"),"__name:=fix34_qp"],"fix34_qp.log")
    wait_for(lambda: qp_states.get_num_connections() and qp_refs.get_num_connections() and batch_pub.get_num_connections())
    measured.position = [0.]*7
    for _ in range(5):
        measured.header.stamp = rospy.Time.now(); qp_states.publish(measured); time.sleep(.03)
    reference = JointTrajectory(); reference.header.stamp = rospy.Time.now(); reference.joint_names = measured.name
    for t,q in ((0.,[0.]*7),(2.,[.4]+[0.]*6)):
        p = JointTrajectoryPoint(); p.positions = q; p.time_from_start = rospy.Duration(t); reference.points.append(p)
    qp_refs.publish(reference)
    for i in range(4):
        wait_for(lambda: len(failures) > i)
        replan.publish(Bool(data=True))
    wait_for(lambda: any("status=stalled" in s for s in stalls))
    assert not candidates
    count = len(queries)
    for _ in range(10): replan.publish(Bool(data=True)); time.sleep(.05)
    assert len(queries) == count == 5, (count,len(queries))
    snapshot = next(s for s in stalls if "status=stalled" in s).split("snapshot=")[1].split()[0]
    saved = out/"qp_snapshot.json"; shutil.copyfile(snapshot,saved)
    replay = subprocess.run(["/tmp/care_replay_qp",str(saved)],capture_output=True,text=True)
    (out/"qp_replay.json").write_text(replay.stdout)
    assert replay.returncode == 0, (replay.stdout,replay.stderr)
    measured.position = [.02]+[0.]*6
    measured.header.stamp = rospy.Time.now(); qp_states.publish(measured)
    wait_for(lambda: len(queries) > count)
    assert not candidates
    # Contradictory UNKNOWN + OCCUPIED rows: local primal infeasibility does
    # not prove that either point is an actual observation blocker.
    primal = copy.deepcopy(local)
    pl = primal["local_planner"]
    for key in ("cdf_batch_topic", "query_trajectory_topic", "candidate_trajectory_topic", "replan_request_topic", "summary_topic"):
        pl[key] = "/fix34/primal/" + key
    pl["reference_trajectory"] = "/fix34/primal/reference"
    pl["piqp"]["max_iterations"] = config["local_planner"]["piqp"]["max_iterations"]
    rospy.set_param("/fix34_primal", primal)
    primal_status = []; evidence = []; primal_candidates = []
    from std_msgs.msg import Float64MultiArray
    rospy.Subscriber("/care_planner/final_gcdf/recovery_visibility_event",Float64MultiArray,lambda m:evidence.append(m))
    rospy.Subscriber("/care_planner/local_planner/task_stall",String,
                     lambda m:primal_status.append(m.data) if m._connection_header.get("callerid") == "/fix34_primal" else None)
    rospy.Subscriber(pl["candidate_trajectory_topic"],JointTrajectory,lambda m:primal_candidates.append(m))
    primal_batch = rospy.Publisher(pl["cdf_batch_topic"],CollisionCDFConstraintBatch,queue_size=1)
    def contradictory(msg):
        b = CollisionCDFConstraintBatch(); b.header = msg.header; b.dof = 7; b.num_pairs = 2
        b.original_timestep = [1,1]; b.source_type = [0,1]; b.distance = [-1.,-1.]
        b.gradient_flat = [0.]*14; b.q_linearization_flat = list(msg.points[1].positions)*2
        b.point_flat = [.1,.2,.3,.4,.5,.6]; primal_batch.publish(b)
    rospy.Subscriber(pl["query_trajectory_topic"],JointTrajectory,contradictory)
    primal_ref = rospy.Publisher(pl["reference_trajectory"],JointTrajectory,queue_size=1)
    primal_retry = rospy.Publisher(pl["replan_request_topic"],Bool,queue_size=1)
    start([str(repo/"devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node"),"__name:=fix34_primal"],"fix34_primal.log")
    wait_for(lambda: primal_ref.get_num_connections() and primal_batch.get_num_connections())
    for _ in range(5):
        measured.header.stamp = rospy.Time.now(); qp_states.publish(measured); time.sleep(.03)
    reference.header.stamp = rospy.Time.now(); primal_ref.publish(reference)
    for i in range(4):
        wait_for(lambda: len([s for s in primal_status if "status=retry" in s]) > i)
        primal_retry.publish(Bool(data=True))
    wait_for(lambda: any("status=stalled" in s for s in primal_status))
    assert not evidence and not primal_candidates
    assert all("QP_PRIMAL_INFEASIBLE_UNATTRIBUTED" in s for s in primal_status if "status=retry" in s)
    primal_snapshot = next(s for s in primal_status if "status=stalled" in s).split("snapshot=")[1].split()[0]
    shutil.copyfile(primal_snapshot,out/"qp_primal_snapshot.json")
    r = subprocess.run(["/tmp/care_replay_qp",str(out/"qp_primal_snapshot.json")],capture_output=True,text=True)
    (out/"qp_primal_replay.json").write_text(r.stdout)
    assert r.returncode == 0
    # Bounded REPAIR feedback: five hard failures per target/motion episode.
    # No actuator, soft candidate, or inferred visibility blocker.
    repair_cfg = copy.deepcopy(local)  # isolated PIQP cap=1, never benchmark config
    rp = repair_cfg['local_planner']
    for key in ('cdf_batch_topic', 'query_trajectory_topic', 'candidate_trajectory_topic',
                'replan_request_topic', 'summary_topic', 'recovery_topic',
                'single_waypoint_q_topic', 'single_waypoint_active_topic'):
        rp[key] = '/evidence/repair/' + key
    rp['reference_trajectory'] = '/evidence/repair/reference'
    rospy.set_param('/evidence_repair', repair_cfg)
    repair_events = []; repair_candidates = []; repair_queries = []
    respond_to_repair = [True]
    repair_batch = rospy.Publisher(rp['cdf_batch_topic'], CollisionCDFConstraintBatch, queue_size=1)
    def repair_query(msg):
        repair_queries.append(msg)
        b = CollisionCDFConstraintBatch(); b.header = msg.header; b.dof = 7
        if respond_to_repair[0]: repair_batch.publish(b)
    rospy.Subscriber(rp['query_trajectory_topic'], JointTrajectory, repair_query)
    rospy.Subscriber(rp['candidate_trajectory_topic'], JointTrajectory, repair_candidates.append)
    rospy.Subscriber(rp['summary_topic'], String, lambda m: repair_events.append(m.data))
    recover = rospy.Publisher(rp['recovery_topic'], Bool, queue_size=1)
    active = rospy.Publisher(rp['single_waypoint_active_topic'], Bool, queue_size=1)
    target_q = rospy.Publisher(rp['single_waypoint_q_topic'], Float64MultiArray, queue_size=1)
    repair_reference = rospy.Publisher(rp['reference_trajectory'], JointTrajectory, queue_size=1)
    retry = rospy.Publisher(rp['replan_request_topic'], Bool, queue_size=1)
    start([str(repo/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'),
           '__name:=evidence_repair'], 'repair_no_candidate.log')
    wait_for(lambda: recover.get_num_connections() and active.get_num_connections()
             and target_q.get_num_connections() and repair_batch.get_num_connections()
             and repair_reference.get_num_connections())
    from std_msgs.msg import MultiArrayDimension
    qmsg = Float64MultiArray(data=[.4] + [0.] * 6)
    qmsg.layout.dim = [MultiArrayDimension(label='care_obs_v1_2_deadbeef', size=7, stride=7)]
    for _ in range(5):
        measured.header.stamp = rospy.Time.now(); qp_states.publish(measured); time.sleep(.03)
    target_q.publish(qmsg); active.publish(Bool(data=True)); time.sleep(.1)
    recover.publish(Bool(data=True))
    time.sleep(.1)
    reference.header.stamp = rospy.Time.now(); repair_reference.publish(reference)
    failures_of_repair = lambda: [e for e in repair_events if
        'event=repair_qp_retry_scheduled' in e or 'event=repair_qp_exhausted' in e]
    wait_for(lambda: len(failures_of_repair()) == 5)
    time.sleep(.5)
    assert len(repair_queries) == 5 and not repair_candidates and not evidence
    for _ in range(10): retry.publish(Bool(data=True)); time.sleep(.05)
    # A repeated reference and an activation toggle cannot replenish A.
    reference.header.stamp = rospy.Time.now(); repair_reference.publish(reference)
    active.publish(Bool(data=False)); time.sleep(.1); active.publish(Bool(data=True))
    time.sleep(.5)
    assert len(repair_queries) == 5
    qmsg.layout.dim[0].label = 'care_obs_v1_3_deadbeef'; qmsg.data[0] = .5
    target_q.publish(qmsg)
    wait_for(lambda: len(failures_of_repair()) == 10)
    qmsg.layout.dim[0].label = 'care_obs_v1_2_deadbeef'; qmsg.data[0] = .4
    target_q.publish(qmsg); retry.publish(Bool(data=True)); time.sleep(.5)
    assert len(repair_queries) == 10, 'revisiting exhausted A reopened its budget'
    # Real motion reopens an episode. A stale old batch must not become work.
    b = CollisionCDFConstraintBatch(); b.header = repair_queries[0].header; b.dof = 7
    repair_batch.publish(b); time.sleep(.1)
    assert len(repair_queries) == 10
    measured.position = [.04]+[0.]*6
    measured.header.stamp = rospy.Time.now(); qp_states.publish(measured)
    wait_for(lambda: len(failures_of_repair()) == 15)
    assert len(repair_queries) == 15 and not repair_candidates and not evidence
    replayed = []
    for i,e in enumerate(failures_of_repair()):
        fields = dict(part.split('=',1) for part in e.split() if '=' in part)
        assert fields['repair_failures'] == str(i%5+1), fields
        assert fields['repair'] == '1' and fields['solved'] == '0'
        assert fields['event'] == ('repair_qp_exhausted' if i%5==4 else 'repair_qp_retry_scheduled')
        if fields['snapshot'] != 'not_available':
            saved_repair = out/('repair_snapshot_%02d.json'%i)
            shutil.copyfile(fields['snapshot'],saved_repair)
            snap = json.loads(saved_repair.read_text())
            assert snap['repair'] and snap['diagnostic_only']
            assert str(snap['plan_sequence']) == fields['plan_seq']
            assert snap['observation_token'] == fields['observation_token']
            r = subprocess.run(['/tmp/care_replay_qp',str(saved_repair)],capture_output=True,text=True)
            (out/('repair_replay_%02d.json'%i)).write_text(r.stdout)
            assert r.returncode == 0, (r.stdout,r.stderr)
            replayed.append(str(saved_repair))
    assert len(replayed) == 6
    # Leaving/re-entering REPAIR may run NORMAL work but cannot refill A.
    recover.publish(Bool(data=False)); time.sleep(.4)
    recover.publish(Bool(data=True)); time.sleep(.5)
    assert len(failures_of_repair()) == 15
    # Replace C while its GCDF query is outstanding; C's returned hard-QP
    # failure is stale and cannot be charged to D or generate a candidate.
    respond_to_repair[0] = False
    nquery = len(repair_queries)
    qmsg.layout.dim[0].label = 'care_obs_v1_4_deadbeef'; qmsg.data[0] = .6
    target_q.publish(qmsg)
    wait_for(lambda: len(repair_queries) > nquery)
    stale_query = repair_queries[-1]
    qmsg.layout.dim[0].label = 'care_obs_v1_5_deadbeef'; qmsg.data[0] = .7
    target_q.publish(qmsg); time.sleep(.08)
    respond_to_repair[0] = True
    b = CollisionCDFConstraintBatch(); b.header = stale_query.header; b.dof = 7
    repair_batch.publish(b)
    wait_for(lambda: len(failures_of_repair()) == 20)
    assert all('observation_token=care_obs_v1_5_deadbeef' in e for e in failures_of_repair()[15:])
    assert not repair_candidates and not evidence
    # Missing GCDF is also bounded, with explicit abort feedback; no QP
    # snapshot exists when no assembled QP was actually solved.
    respond_to_repair[0] = False
    qmsg.layout.dim[0].label = 'care_obs_v1_6_deadbeef'; qmsg.data[0] = .8
    target_q.publish(qmsg)
    wait_for(lambda: any('event=repair_plan_exhausted' in e for e in repair_events))
    timeout_events = [e for e in repair_events if 'event=aborted_cdf_wait_timeout' in e]
    assert len(timeout_events) == 5, repair_events[-12:]
    count = len(repair_queries)
    retry.publish(Bool(data=True)); time.sleep(.5)
    assert len(repair_queries) == count and not repair_candidates and not evidence
    return dict(status="PASS", checks=["Case008 approximate IK continues for exactly three attempts without reference",
        "new measured-pose request succeeds with measured boundary", "QP stops after five failures despite replan pulses",
        "diagnostic soft solve never publishes candidate", "hard QP snapshot replays same status", "measured progress releases QP stall",
        "mixed UNKNOWN/OCCUPIED primal infeasibility never publishes observation evidence",
        "primal infeasibility snapshot replays same status",
        "REPAIR automatically retries exactly five times; repeated pulses/reference/activation cannot reset budget",
        "new target and real measured motion reopen opportunities; revisited exhausted target stays blocked",
        "stale GCDF batch discarded; no REPAIR candidate or false blocker on failed hard solve",
        "mode toggles do not refill target budgets; changed-target in-flight result is discarded",
        "missing GCDF exhausts exactly five attempts without QP snapshot or candidate",
        "all six first/last REPAIR snapshots retain plan/target identity and replay identical status"],
        repair_snapshots=replayed,
        repair_timeout_events=timeout_events,
        repair_no_candidate_events=failures_of_repair(),
        approximate_statuses=approx, qp_stall_statuses=stalls, primal_statuses=primal_status, snapshot=str(saved))
