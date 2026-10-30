#!/usr/bin/env python3
"""Exercise certified NORMAL completion liveness in the built C++ planner."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import xmlrpc.client


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--port", type=int, default=11349)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    uri = f"http://127.0.0.1:{args.port}"
    try:
        xmlrpc.client.ServerProxy(uri).getPid("/normal_liveness_preflight")
    except OSError:
        pass
    else:
        raise RuntimeError("private ROS master port is occupied")
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME="localhost",
                      ROS_LOG_DIR=str(output / "ros_logs"), ROS_HOME=str(output / "ros_home"))
    processes = []
    streams = []

    def start(argv, name):
        stream = (output / name).open("x")
        streams.append(stream)
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        processes.append(process)
        return process

    def wait(predicate, name, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.02)
        raise AssertionError("timeout: " + name)

    try:
        start(["roscore", "-p", str(args.port)], "roscore.log")
        def master_ready():
            try:
                return xmlrpc.client.ServerProxy(uri).getPid("/normal_liveness_preflight")[0] == 1
            except OSError:
                return False
        wait(master_ready, "private ROS master")
        import rospy
        import yaml
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Header, String
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch

        rospy.init_node("normal_liveness_driver", disable_signals=True)
        config = copy.deepcopy(yaml.safe_load((repo / "src/egocentric_arm_planner/config/"
                                                 "planner_c5_5_vbc_gcdf_regime.yaml").read_text()))
        lp = config["local_planner"]
        for key in ("joint_states", "reference_trajectory", "cdf_batch_topic",
                    "query_trajectory_topic", "candidate_trajectory_topic", "summary_topic"):
            lp[key] = "/normal_liveness_test/" + key
        lp["cdf_wait_timeout"] = 5.0
        lp["normal_active_no_progress_duration_s"] = 0.5
        lp["normal_active_no_progress_min_executions"] = 5
        rospy.set_param("/normal_liveness_test", config)
        joints = ["joint1", "joint2", "joint3", "joint4",
                  "wrist_joint1", "wrist_joint2", "wrist_joint3"]
        joint_pub = rospy.Publisher(lp["joint_states"], JointState, queue_size=10)
        ref_pub = rospy.Publisher(lp["reference_trajectory"], JointTrajectory, queue_size=2)
        batch_pub = rospy.Publisher(lp["cdf_batch_topic"], Batch, queue_size=30)
        tracker_pub = rospy.Publisher(config["local_planner"]["execution_summary_topic"],
                                      String, queue_size=10)
        events, candidates, stalls, queries = [], [], [], []
        lock = threading.Lock()

        def append(target, item):
            with lock:
                target.append(item)

        def query_cb(msg):
            append(queries, msg.header.stamp.to_nsec())
            batch_pub.publish(Batch(header=msg.header, dof=7, num_pairs=0))

        subscriptions = [
            rospy.Subscriber(lp["summary_topic"], String,
                             lambda msg: append(events, msg.data)),
            rospy.Subscriber(lp["candidate_trajectory_topic"], JointTrajectory,
                             lambda msg: append(candidates, msg)),
            rospy.Subscriber(lp["query_trajectory_topic"], JointTrajectory, query_cb),
            rospy.Subscriber("/care_planner/local_planner/task_stall", String,
                             lambda msg: append(stalls, msg.data))]
        binary = repo / "devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node"
        start([str(binary), "__name:=normal_liveness_test"], "planner.log")
        wait(lambda: joint_pub.get_num_connections() and ref_pub.get_num_connections() and
             batch_pub.get_num_connections() and tracker_pub.get_num_connections(),
             "planner subscriptions")

        def measured():
            joint_pub.publish(JointState(header=Header(stamp=rospy.Time.now()),
                                         name=joints, position=[0.] * 7,
                                         velocity=[0.] * 7))

        measured()
        ref = JointTrajectory(joint_names=joints)
        ref.header.stamp = rospy.Time.now()
        ref.points = [JointTrajectoryPoint(positions=[0.] * 7,
                                           time_from_start=rospy.Duration(0)),
                      JointTrajectoryPoint(positions=[.4] + [0.] * 6,
                                           time_from_start=rospy.Duration(2))]
        ref_pub.publish(ref)
        for i in range(5):
            wait(lambda: len(candidates) > i, f"candidate {i}")
            measured()
            time.sleep(.03)
            stamp = candidates[i].header.stamp.to_nsec()
            tracker_pub.publish(String(data=(
                f"active=0 complete=1 execution_aborted=0 phase_s=1 "
                f"execution_stamp_ns={stamp}")))
            if i < 4:
                wait(lambda: len(candidates) > i + 1, f"replan after execution {i}")
        wait(lambda: any("classification=NORMAL_CERTIFIED_NO_MEASURED_PROGRESS" in s
                         for s in stalls), "certified measured stall")
        stall = next(s for s in stalls if
                     "classification=NORMAL_CERTIFIED_NO_MEASURED_PROGRESS" in s)
        assert "failures=5" in stall, stall
        assert "query_stamp_ns=0" not in stall, stall
        assert any("event=normal_certified_no_measured_progress " in s for s in events)
        assert len(candidates) == 5, len(candidates)

        # Reset the completed-execution stall with a fresh reference, then
        # simulate rapid certified handoffs that never reach complete=1.
        # This was the unobserved liveness path in CASE018.
        before = len(candidates)
        ref.header.stamp = rospy.Time.now()
        ref_pub.publish(ref)
        wait(lambda: len(candidates) > before, "fresh reference after completion stall")
        for _ in range(12):
            measured()
            time.sleep(.03)
            active_stamp = (rospy.Time.now() - rospy.Duration(.1)).to_nsec()
            tracker_pub.publish(String(data=(
                f"active=1 complete=0 execution_aborted=0 phase_s=.1 "
                f"execution_stamp_ns={active_stamp}")))
            if any("classification=NORMAL_ACTIVE_CERTIFIED_NO_MEASURED_PROGRESS" in s
                   for s in stalls):
                break
            time.sleep(.12)
        wait(lambda: any("classification=NORMAL_ACTIVE_CERTIFIED_NO_MEASURED_PROGRESS" in s
                         for s in stalls), "active measured stall")
        active_stall = next(s for s in stalls if
            "classification=NORMAL_ACTIVE_CERTIFIED_NO_MEASURED_PROGRESS" in s)
        assert "active_execution_stamps=" in active_stall, active_stall
        assert "query_stamp_ns=0" not in active_stall, active_stall
        result = dict(status="PASS", planner_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                      candidates=len(candidates), queries=len(queries), stall=stall,
                      active_stall=active_stall)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
