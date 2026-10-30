#!/usr/bin/env python3
"""Isolated ROS integration test: synthetic joint states, no Gazebo/actuators.

Runs its own roscore on port 11329 and only the nominal-reference planner.
Terminates only child process groups created here. Requires a built workspace.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xmlrpc.client


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--fix34", action="store_true", help="Also exercise strict IK and bounded QP failures")
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    uri="http://127.0.0.1:11329"
    try:
        xmlrpc.client.ServerProxy(uri).getPid("/fix12_probe")
    except OSError:
        pass
    else:
        raise RuntimeError("Port 11329 already serves a ROS master; refusing to interfere")
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME="localhost",ROS_LOG_DIR=str(out/"ros_logs"))
    children=[]; handles=[]
    def start(argv,name):
        f=(out/name).open("x"); handles.append(f)
        p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p); return p
    def wait_for(predicate,timeout=10):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate(): return
            time.sleep(.03)
        raise AssertionError("timed out waiting for test condition")
    try:
        start(["roscore","-p","11329"],"roscore.log")
        def master_ready():
            try: return xmlrpc.client.ServerProxy(uri).getPid("/fix12_probe")[0]==1
            except OSError: return False
        wait_for(master_ready)
        import rospy
        import yaml
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import String
        from trajectory_msgs.msg import JointTrajectory
        from rosgraph_msgs.msg import Log
        import evaluate_phase_d_run as ev
        from tf.transformations import quaternion_from_matrix
        rospy.init_node("fix12_retry_test",disable_signals=True)
        config=yaml.safe_load((repo/"src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml").read_text())
        config["urdf_path"]=str(repo/"src/arm_description/urdf/Arm.urdf")
        config["topics"]={"joint_states":"/fix12/joints","target_pose":"/fix12/target",
                          "task_trajectory":"/fix12/reference","command_trajectory":"/fix12/nominal_command"}
        config["receding_horizon"]["mpc_command_topic"]="/fix12/unused_command_history"
        rospy.set_param("/fix12_planner",config)
        statuses=[]; references=[]; planner_messages=[]
        rospy.Subscriber('/rosout',Log,lambda m:planner_messages.append(m.msg))
        rospy.Subscriber("/care_planner/task_reference_status",String,lambda m:statuses.append(m.data))
        rospy.Subscriber("/fix12/reference",JointTrajectory,lambda m:references.append(m))
        state_pub=rospy.Publisher("/fix12/joints",JointState,queue_size=1)
        goal_pub=rospy.Publisher("/fix12/target",PoseStamped,queue_size=1)
        binary=str(repo/"devel/lib/egocentric_arm_planner/receding_horizon_planner_node")
        start([binary,"__name:=fix12_planner"],"planner.log")
        wait_for(lambda: state_pub.get_num_connections()>0 and goal_pub.get_num_connections()>0)
        # Only synthetic measured state. No actuator publisher exists in this test.
        q=JointState(); q.name=ev.JOINTS; q.position=[0]*7; q.velocity=[0]*7
        for _ in range(5):
            q.header.stamp=rospy.Time.now(); state_pub.publish(q); time.sleep(.03)
        bad=PoseStamped(); bad.header.frame_id="base_link"; bad.header.stamp=rospy.Time.now()
        goal_pub.publish(bad)  # zero quaternion -> deterministic generation failure
        wait_for(lambda:any("request_id=1 " in s and "status=exhausted" in s for s in statuses))
        assert any("attempt=3 " in s and "status=exhausted" in s for s in statuses)
        assert not references
        helper=ev.load_fk_helper(str(repo)); robot=ev.URDF.from_xml_file(config["urdf_path"])
        chain=helper.find_chain_joints(robot,"base_link","EE_link")
        T=ev.fk_T(helper,chain,dict(zip(ev.JOINTS,[0]*7)))
        good=PoseStamped(); good.header.frame_id="base_link"; good.header.stamp=rospy.Time.now()
        good.pose.position.x,good.pose.position.y,good.pose.position.z=map(float,T[:3,3])
        quat=quaternion_from_matrix(T)
        good.pose.orientation.x,good.pose.orientation.y,good.pose.orientation.z,good.pose.orientation.w=map(float,quat)
        goal_pub.publish(good)
        wait_for(lambda:bool(references) and any("request_id=2 " in s and "status=succeeded" in s for s in statuses))
        count=len(references); time.sleep(.7); assert len(references)==count==1
        # Same pose, new measured boundary: the previously converged goal is
        # an IK seed, while the published trajectory must start at new q.
        q.position=[.1]+[0.]*6
        for _ in range(5):
            q.header.stamp=rospy.Time.now();state_pub.publish(q);time.sleep(.03)
        good.header.stamp=rospy.Time.now();goal_pub.publish(good)
        wait_for(lambda:len(references)==2 and any("request_id=3 " in s and "status=succeeded" in s for s in statuses))
        assert max(abs(a-b) for a,b in zip(references[-1].points[0].positions,q.position))<1e-8
        wait_for(lambda:any("request_id=3 ik_seed_source=cached_goal" in s for s in planner_messages))
        pubs=rospy.get_published_topics()
        assert not any("arm_group_velocity_controller/command" in topic for topic,_ in pubs)
        # A second planner with no measured input must timeout, not wait forever.
        cfg=dict(config); cfg["topics"]=dict(config["topics"],joint_states="/fix12/no_joints",target_pose="/fix12/timeout_target")
        cfg["receding_horizon"]=dict(config["receding_horizon"],reference_timeout_s=.3)
        rospy.set_param("/fix12_timeout_planner",cfg)
        timeout_pub=rospy.Publisher("/fix12/timeout_target",PoseStamped,queue_size=1)
        start([binary,"__name:=fix12_timeout_planner"],"timeout_planner.log")
        wait_for(lambda:timeout_pub.get_num_connections()>0)
        timeout_pub.publish(good)
        wait_for(lambda:any("reason=request_timeout" in s for s in statuses))
        # Fault injection is isolated to this node's private parameters: a
        # 50 ms duration cap must fail closed, not return an invalid SUCCESS.
        capped=dict(config)
        capped["topics"]=dict(config["topics"],target_pose="/fix12/capped_target",
                              task_trajectory="/fix12/capped_reference")
        capped["task_generator"]=dict(config["task_generator"],min_plan_duration=.05,max_plan_duration=.05)
        rospy.set_param("/fix12_capped_planner",capped)
        capped_refs=[]
        rospy.Subscriber("/fix12/capped_reference",JointTrajectory,lambda m:capped_refs.append(m))
        capped_pub=rospy.Publisher("/fix12/capped_target",PoseStamped,queue_size=1)
        start([binary,"__name:=fix12_capped_planner"],"capped_planner.log")
        wait_for(lambda:capped_pub.get_num_connections()>0)
        for _ in range(5):
            q.header.stamp=rospy.Time.now(); state_pub.publish(q); time.sleep(.03)
        target_T=ev.fk_T(helper,chain,dict(zip(ev.JOINTS,[.4]+[0]*6)))
        good.pose.position.x,good.pose.position.y,good.pose.position.z=map(float,target_T[:3,3])
        quat=quaternion_from_matrix(target_T)
        good.pose.orientation.x,good.pose.orientation.y,good.pose.orientation.z,good.pose.orientation.w=map(float,quat)
        previous=len(statuses); capped_pub.publish(good)
        wait_for(lambda:any("status=exhausted" in s for s in statuses[previous:]))
        assert not capped_refs
        assert "INVALID_TRAJECTORY" in (out/"capped_planner.log").read_text()
        report={"status":"PASS","checks":["generation failure exhausts exactly 3 attempts without reference",
            "new request succeeds after exhausted request","successful reference published once",
            "missing measured input times out","no actuator publisher",
            "duration cap returns INVALID_TRAJECTORY without publishing",
            "same-target cached IK goal rebuilds trajectory from changed measured boundary"],"statuses":statuses}
        if args.fix34:
            from phase_e_fix34_ros_checks import run_checks
            report["fix34"] = run_checks(rospy, repo, out, config, start, wait_for)
        (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM); p.wait(timeout=5)
        for f in handles:f.close()


if __name__=="__main__":main()
