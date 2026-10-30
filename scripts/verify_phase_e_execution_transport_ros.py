#!/usr/bin/env python3
"""Private ROS receiver faults and actual built planner response callbacks."""
import argparse
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
import xmlrpc.client

ROOT=Path(__file__).resolve().parents[1]

PEER=r'''
import os,time,rospy
from std_msgs.msg import Float64MultiArray
role=os.environ['CARE_TEST_PEER'];rospy.init_node(role,disable_signals=True)
sub=None
if role=='receipt_tracker':
    pub=rospy.Publisher('/receipt_test/desired',Float64MultiArray,queue_size=1)
while not rospy.is_shutdown():
    if role=='receipt_tracker':pub.publish(Float64MultiArray(data=[0.]*7))
    elif sub is None and rospy.get_param('/receipt_test/subscribe',False):
        sub=rospy.Subscriber('/receipt_test/command',Float64MultiArray,lambda m:None,queue_size=1,tcp_nodelay=True)
    time.sleep(.01)
'''

CPP_RECEIVER=r'''
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
void receive(const std_msgs::Float64MultiArray::ConstPtr&) {}
int main(int argc, char** argv) {
  ros::init(argc, argv, "receipt_controller");
  ros::NodeHandle nh;
  ros::Subscriber sub;
  ros::WallRate rate(100);
  while (ros::ok()) {
    bool subscribe=false;
    nh.getParam("/receipt_test/subscribe", subscribe);
    if (subscribe && !sub) {
      sub=nh.subscribe<std_msgs::Float64MultiArray>("/receipt_test/command",1,
        receive,
        ros::TransportHints().tcpNoDelay());
    }
    ros::spinOnce(); rate.sleep();
  }
}
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--port',type=int,default=11644)
    ap.add_argument('--receiver-implementation',choices=['roscpp','rospy'],default='roscpp')
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.receiver_implementation=='roscpp':
        cpp=out/'receipt_controller.cpp';cpp.write_text(CPP_RECEIVER)
        flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','roscpp'],text=True))
        with (out/'receipt_controller_build.log').open('x') as log:
            subprocess.run(['g++','-std=c++14','-O2',str(cpp),'-o',str(out/'receipt_controller')]+flags,stdout=log,stderr=subprocess.STDOUT,check=True)
    uri='http://127.0.0.1:'+str(args.port)
    import socket
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',args.port))!=0,'private port occupied'
    os.environ.update(ROS_MASTER_URI=uri,ROS_LOG_DIR=str(out/'ros_logs'),ROS_HOME=str(out/'ros_home'))
    procs=[];streams=[];paused=None
    def start(argv,name,extra=None):
        stream=(out/(name+'.log')).open('x');streams.append(stream)
        env=os.environ.copy();env.update(extra or {})
        p=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True);procs.append(p);return p
    def wait(test,name,timeout=10.):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if test():return
            time.sleep(.02)
        raise AssertionError('timeout '+name)
    try:
        start(['roscore','-p',str(args.port)],'roscore')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/receipt_test')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy,yaml
        from std_msgs.msg import String,Float64MultiArray,Bool,Header
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from care_collision_cdf.msg import CollisionCDFWitnessRequest,CollisionCDFWitnessResponse
        from phase_e_benchmark_arm import prepare_arm,ARM_TOPIC
        rospy.init_node('execution_transport_test_driver',disable_signals=True)
        config=dict(tracker_node='/receipt_tracker',limiter_node='/receipt_limiter',controller_node='/receipt_controller',
            desired_topic='/receipt_test/desired',command_topic='/receipt_test/command')
        rospy.set_param('/care_planner/benchmark/actuator_readiness_required',True)
        rospy.set_param('/care_planner/benchmark/actuator_readiness',config)
        rospy.set_param('/receipt_test/subscribe',False)
        rospy.set_param('/receipt_limiter/joint_names',['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3'])
        start(['/usr/bin/python3','-c',PEER],'tracker',{'CARE_TEST_PEER':'receipt_tracker'})
        controller=start([str(out/'receipt_controller')],'controller') if args.receiver_implementation=='roscpp' else start(['/usr/bin/python3','-c',PEER],'controller',{'CARE_TEST_PEER':'receipt_controller'})
        start(['/usr/bin/python3','src/egocentric_arm_planner/scripts/joint_velocity_rate_limiter_node.py',
               '__name:=receipt_limiter','_input_topic:=/receipt_test/desired','_output_topic:=/receipt_test/command',
               '_summary_topic:=/receipt_test/limiter_summary'],'limiter')
        armed=[];echo=[]
        subscriptions=[rospy.Subscriber(ARM_TOPIC,String,lambda m:armed.append(m.data)),
                       rospy.Subscriber('/receipt_test/command',Float64MultiArray,lambda m:echo.append(m.data))]
        wait(lambda:len(echo)>10,'limiter zero echo')
        begin=time.monotonic()
        try:prepare_arm([],timeout_s=.6,admission_ping_topic='')
        except RuntimeError as e:missing=str(e)
        else:raise AssertionError('echo-only graph armed')
        assert not armed and time.monotonic()-begin<1.
        rospy.set_param('/receipt_test/subscribe',True)
        pub=prepare_arm([],timeout_s=3.,admission_ping_topic='')
        good=pub.care_actuator_readiness;assert good['ready'] and not armed
        receiver_uri=next(row['receiver_uri'] for row in good['receiver_counts'] if row['receiver']==config['controller_node'])
        receiver_stats=xmlrpc.client.ServerProxy(receiver_uri).getBusStats('/execution_transport_test_driver')
        (out/'receiver_raw_bus_stats.json').write_text(json.dumps(receiver_stats,indent=2)+'\n')
        pub.unregister()
        os.killpg(controller.pid,signal.SIGSTOP);paused=controller
        begin=time.monotonic()
        try:prepare_arm([],timeout_s=.6,admission_ping_topic='')
        except RuntimeError as e:stopped=str(e)
        else:raise AssertionError('stopped receiver armed')
        assert time.monotonic()-begin<1. and not armed
        os.killpg(controller.pid,signal.SIGCONT);paused=None
        pub=prepare_arm([],timeout_s=3.,admission_ping_topic='');assert pub.care_actuator_readiness['ready'];pub.unregister()

        # Actual C++ planner callbacks. CDF replies, certification and measured
        # states are explicit synthetic controls; no robot/model is simulated.
        params=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        lp=params['local_planner']
        for key in ['joint_states','reference_trajectory','cdf_batch_topic','query_trajectory_topic','candidate_trajectory_topic','summary_topic','executed_command_topic','execution_summary_topic','desired_command_topic','limiter_summary_topic','vbc_obligation_points_topic','single_waypoint_active_topic','single_waypoint_q_topic']:
            lp[key]='/response_test/'+key
        rospy.set_param('/response_test',params)
        safety_topic='/response_test/vbc_hold'
        rospy.set_param('/response_test/execution/vbc_safety_hold_topic',safety_topic)
        safety_pub=rospy.Publisher(safety_topic,Bool,queue_size=1,latch=True)
        joint_pub=rospy.Publisher(lp['joint_states'],JointState,queue_size=1)
        ref_pub=rospy.Publisher(lp['reference_trajectory'],JointTrajectory,queue_size=1)
        batch_pub=rospy.Publisher(lp['cdf_batch_topic'],Batch,queue_size=10)
        witness_pub=rospy.Publisher(lp['cdf_batch_topic']+'/witness_response',CollisionCDFWitnessResponse,queue_size=10)
        tracker_pub=rospy.Publisher(lp['execution_summary_topic'],String,queue_size=10)
        desired_pub=rospy.Publisher(lp['desired_command_topic'],Float64MultiArray,queue_size=1)
        replan=rospy.Publisher(params['local_planner']['replan_request_topic'],Bool,queue_size=1)
        recovery_pub=rospy.Publisher(lp['recovery_topic'],Bool,queue_size=1,latch=True)
        frontier_pub=rospy.Publisher(lp['visibility_frontier_topic'],Float64MultiArray,queue_size=1,latch=True)
        obligation_pub=rospy.Publisher(lp['vbc_obligation_points_topic'],Float64MultiArray,queue_size=1,latch=True)
        waypoint_active=rospy.Publisher(lp['single_waypoint_active_topic'],Bool,queue_size=1,latch=True)
        waypoint_q=rospy.Publisher(lp['single_waypoint_q_topic'],Float64MultiArray,queue_size=1,latch=True)
        verified_pub=rospy.Publisher('/care_planner/verification_outcome',String,queue_size=10)
        events=[];candidates=[]
        def witness(request):
            reply=CollisionCDFWitnessResponse(request=request,batch=Batch(header=request.header,dof=7,num_pairs=0),
                map_stamp=rospy.Time.now(),status=['resolved_free']*len(request.original_timestep))
            witness_pub.publish(reply)
        subscriptions.extend([
            rospy.Subscriber(lp['summary_topic'],String,lambda m:events.append(m.data)),
            rospy.Subscriber(lp['candidate_trajectory_topic'],JointTrajectory,lambda m:candidates.append(m)),
            rospy.Subscriber(lp['query_trajectory_topic'],JointTrajectory,lambda m:batch_pub.publish(Batch(header=m.header,dof=7,num_pairs=0))),
            rospy.Subscriber(lp['cdf_batch_topic']+'/witness_request',CollisionCDFWitnessRequest,witness)])
        start([str(ROOT/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'),'__name:=response_test'],'planner')
        wait(lambda:joint_pub.get_num_connections() and ref_pub.get_num_connections() and
             batch_pub.get_num_connections() and tracker_pub.get_num_connections() and
             desired_pub.get_num_connections(),'planner input')
        names=params['joint_names'];pose=[0.]*7
        def measured():joint_pub.publish(JointState(header=Header(stamp=rospy.Time.now()),name=names,position=pose,velocity=[0.]*7))
        measured();ref=JointTrajectory(joint_names=names);ref.header.stamp=rospy.Time.now()
        ref.points=[JointTrajectoryPoint(positions=[0.]*7,time_from_start=rospy.Duration(0)),JointTrajectoryPoint(positions=[.3]+[0.]*6,time_from_start=rospy.Duration(2))]
        ref_pub.publish(ref);recovery_pub.publish(Bool(data=True))
        obligation_pub.publish(Float64MultiArray(data=[1.,1.,1.,.2,.1,.5]))
        waypoint_active.publish(Bool(data=True));waypoint_q.publish(Float64MultiArray(data=[.3]+[0.]*6))
        frontier_pub.publish(Float64MultiArray(data=[1.,1.,0.,.05,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.]))
        def candidate_ready():measured();return bool(candidates)
        wait(candidate_ready,'candidate')
        # Nonzero tracker requests during a deliberate safety hold are not
        # actuator nonresponse, and do not spend sensing-direction attempts.
        safety_pub.publish(Bool(data=True));time.sleep(.1)
        begin=time.monotonic()
        while time.monotonic()-begin<1.2:
            measured();desired_pub.publish(Float64MultiArray(data=[.1]+[0.]*6))
            token=rospy.Time.now().to_nsec()
            tracker_pub.publish(String(data=f'active=1 complete=0 execution_aborted=0 phase_s=.1 execution_stamp_ns={token}'))
            time.sleep(.02)
        assert not any('event=execution_command_no_response_hold ' in s for s in events)
        safety_pub.publish(Bool(data=False));time.sleep(.05)
        last_raw=None;execution=0;begin=time.monotonic();fault=None
        while time.monotonic()-begin<4.:
            measured();desired_pub.publish(Float64MultiArray(data=[.1]+[0.]*6))
            raw=candidates[-1].header.stamp.to_nsec()
            if raw!=last_raw:
                execution=rospy.Time.now().to_nsec();last_raw=raw
                verified_pub.publish(String(data=f'result=safe committed=1 safety_gate=vbc raw_candidate_stamp_ns={raw} execution_stamp_ns={execution}'))
            tracker_pub.publish(String(data=f'active=1 complete=0 execution_aborted=0 phase_s=.1 execution_stamp_ns={execution}'))
            replan.publish(Bool(data=True));time.sleep(.02)
            fault=next((s for s in events if 'event=execution_command_no_response_hold ' in s),None)
            if fault:break
        assert fault and 'safe_stall_attempts=0 ' in fault,fault
        before=len(candidates);time.sleep(.2)
        held=len(candidates)
        for _ in range(10):
            measured();desired_pub.publish(Float64MultiArray(data=[0.]*7));tracker_pub.publish(String(data=f'active=1 complete=0 execution_aborted=0 phase_s=.1 execution_stamp_ns={execution}'));replan.publish(Bool(data=True));time.sleep(.02)
        assert len(candidates)==held,'zero requests or replan released nonresponse hold'
        pose[0]=.002
        for _ in range(10):
            measured();tracker_pub.publish(String(data=f'active=1 complete=0 execution_aborted=0 phase_s=.1 execution_stamp_ns={execution}'));time.sleep(.02)
        wait(lambda:any('event=execution_response_restored ' in s for s in events),'measured response restoration')
        wait(lambda:len(candidates)>held,'fresh proposal after measured response')
        report=dict(status='PASS',receiver_implementation=args.receiver_implementation,
            raw_bus_stats_outer_types=[type(x).__name__ for x in receiver_stats],
            echo_only_rejected=missing,paused_receiver_rejected=stopped,actual_receiver_receipt=good,
            deliberate_safety_hold_not_classified_as_nonresponse=True,
            benchmark_arm_messages=len(armed),cpp_nonresponse_event=fault,cpp_candidates_while_held=held,
            cpp_response_restored=True,qualification='Real independent ROS processes and actual built planner. Synthetic zero-CDF certification and measured states test control flow; not physical collision qualification or a complete closed-loop alternative.')
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:
        if paused is not None:os.killpg(paused.pid,signal.SIGCONT)
        for p in reversed(procs):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        for s in streams:s.close()
        (out/'cleanup.json').write_text(json.dumps({'remaining':[p.pid for p in procs if p.poll() is None]})+'\n')

if __name__=='__main__':main()
