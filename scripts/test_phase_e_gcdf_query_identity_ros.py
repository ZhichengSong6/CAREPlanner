#!/usr/bin/env python3
"""Bounded frozen-/clock fixture: real SCP + primitive exporter, fake empty GCDF.

Private ROS master, no Gazebo, tracker, actuator, or real safety certificate.
Run --expect-collision on the pre-fix binary to demonstrate the old failure.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xmlrpc.client


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--expect-collision', action='store_true')
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    os.environ.update(ROS_MASTER_URI=f'http://127.0.0.1:{port}', ROS_HOSTNAME='localhost',
                      ROS_LOG_DIR=str(out/'ros_logs'))
    children, handles = [], []
    result = dict(status='FAIL', fixture_only=True, expected_old_collision=args.expect_collision)
    result['planner_sha256'] = hashlib.sha256((root/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node').read_bytes()).hexdigest()
    def start(argv, name):
        f = (out/name).open('x'); handles.append(f)
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p)
    def wait(pred, timeout=8.):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if pred(): return
            time.sleep(.01)
        raise AssertionError('bounded fixture timeout')
    try:
        start(['roscore', '-p', str(port)], 'master.log')
        master = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
        def ready():
            try: return master.getPid('/identity_fixture')[0] == 1
            except OSError: return False
        wait(ready)
        import rospy
        import yaml
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState, PointCloud2
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch
        from std_msgs.msg import Bool, String
        rospy.set_param('/use_sim_time', True)
        rospy.init_node('identity_fixture', disable_signals=True)
        cfg = yaml.safe_load((root/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        cfg['urdf_path'] = str(root/'src/arm_description/urdf/Arm.urdf')
        lp = cfg['local_planner']
        for key in ['joint_states', 'reference_trajectory', 'query_trajectory_topic',
                    'cdf_batch_topic', 'candidate_trajectory_topic', 'summary_topic']:
            lp[key] = '/identity/'+key
        rospy.set_param('/identity_scp', cfg)
        rospy.set_param('/identity_anchors', dict(robot_urdf_file=cfg['urdf_path'],
            primitive_urdf_file=str(root/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            base_frame='base_link', input_trajectory_topic=lp['query_trajectory_topic'],
            output_topic='/identity/anchors', ignored_risk_links=['base_link', 'link1'],
            body_inflation_m=.015, max_eval_timesteps=50))
        queries, anchors, candidates, summaries, uncertified = [], [], [], [], []
        subs = [rospy.Subscriber(lp['query_trajectory_topic'], JointTrajectory, queries.append),
                rospy.Subscriber('/identity/anchors', PointCloud2, anchors.append),
                rospy.Subscriber(lp['candidate_trajectory_topic'], JointTrajectory, candidates.append),
                rospy.Subscriber(lp['summary_topic'], String, lambda m: summaries.append(m.data)),
                rospy.Subscriber('/care_planner/local_planner/task_uncertified', Bool, lambda m: uncertified.append(m.data))]
        clock = rospy.Publisher('/clock', Clock, queue_size=1, latch=True)
        states = rospy.Publisher(lp['joint_states'], JointState, queue_size=1)
        refs = rospy.Publisher(lp['reference_trajectory'], JointTrajectory, queue_size=1)
        batches = rospy.Publisher(lp['cdf_batch_topic'], CollisionCDFConstraintBatch, queue_size=10)
        clock.publish(Clock(clock=rospy.Time(10)))
        start([str(root/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'),
               '__name:=identity_scp'], 'planner.log')
        start([str(root/'devel/lib/care_confidence_map/gcdf_primitive_anchor_node'),
               '__name:=identity_anchors'], 'anchors.log')
        wait(lambda: states.get_num_connections() and refs.get_num_connections()
             and batches.get_num_connections() and len(subs) == 5)
        measured = JointState(); measured.name = cfg['joint_names']
        measured.position = [0.]*7; measured.velocity = [0.]*7
        for _ in range(5):
            measured.header.stamp = rospy.Time(10); states.publish(measured); time.sleep(.02)
        reference = JointTrajectory(); reference.header.stamp = rospy.Time(10)
        reference.header.frame_id = 'base_link'; reference.joint_names = measured.name
        for t, q in [(0., [0.]*7), (1., [.4]+[0.]*6)]:
            p = JointTrajectoryPoint(); p.positions = q; p.time_from_start = rospy.Duration(t)
            reference.points.append(p)
        refs.publish(reference); time.sleep(.1)
        for i in range(1, 11):
            if queries: break
            clock.publish(Clock(clock=rospy.Time.from_sec(10.+i*.05))); time.sleep(.05)
        wait(lambda: len(queries) >= 1 and len(anchors) >= 1)
        frozen = rospy.Time.now().to_nsec()
        def reply(header):
            b = CollisionCDFConstraintBatch(); b.header = copy.deepcopy(header); b.dof = 7
            batches.publish(b)
        reply(anchors[0].header)
        wait(lambda: len(queries) >= 2)
        if args.expect_collision:
            time.sleep(.1)
            assert queries[0].header.stamp == queries[1].header.stamp
            assert len(anchors) == 1 and not candidates
            result['checks'] = ['old producer repeats frozen ROS time', 'real primitive exporter drops second query']
        else:
            wait(lambda: len(anchors) >= 2)
            # Replay a stale response just 1 ns away: must not authorize work.
            reply(anchors[0].header); time.sleep(.1)
            assert len(queries) == 2 and not candidates, 'stale batch accepted'
            replied = 1
            deadline = time.monotonic()+5.
            while not candidates and time.monotonic() < deadline:
                if replied < len(anchors):
                    reply(anchors[replied].header); replied += 1
                time.sleep(.01)
            assert candidates, 'no candidate under fixed /clock'
            stamps = [q.header.stamp.to_nsec() for q in queries]
            assert len(stamps) >= 3 and all(b > a for a,b in zip(stamps,stamps[1:]))
            assert len(anchors) == len(queries) and rospy.Time.now().to_nsec() == frozen
            assert any('stamp_miss=1' in s for s in summaries)
            result['checks'] = ['strictly increasing query identities at frozen /clock',
                'every query exported by real primitive node', 'stale 1 ns-separated response rejected',
                'real SCP completes fixture candidate without clock advance']
            result['frozen_phase_query_stamps_ns'] = stamps
            # A genuine missing response must hold and report uncertified, not
            # silently certify or loop. Continue /clock only for this phase.
            previous_queries, previous_candidates = len(queries), len(candidates)
            reference.header.stamp = rospy.Time.from_sec(10.7)
            clock.publish(Clock(clock=reference.header.stamp)); time.sleep(.05)
            measured.header.stamp = reference.header.stamp; states.publish(measured)
            refs.publish(reference); time.sleep(.05)
            for i in range(1, 41):
                clock.publish(Clock(clock=rospy.Time.from_sec(10.7+i*.025)))
                time.sleep(.025)
            assert any('event=aborted_cdf_wait_timeout ' in s for s in summaries)
            assert any(uncertified), 'timeout not reported uncertified'
            assert len(queries) == previous_queries+1 and len(candidates) == previous_candidates
            result['checks'].append('missing response: explicit uncertified, no candidate or automatic retry')
        published = rospy.get_published_topics()
        assert not any('arm_group_velocity_controller/command' in t for t,_ in published)
        result.update(status='PASS', frozen_clock_ns=frozen,
            query_stamps_ns=[q.header.stamp.to_nsec() for q in queries],
            anchor_stamps_ns=[q.header.stamp.to_nsec() for q in anchors], candidates=len(candidates), summaries=summaries)
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=5)
        for f in handles: f.close()
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'summaries'}, indent=2))


if __name__ == '__main__': main()
