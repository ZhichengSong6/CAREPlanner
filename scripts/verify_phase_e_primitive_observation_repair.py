#!/usr/bin/env python3
"""Fixed production QP and tracker validation in private ROS, without actuators."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    # Sandbox restrictions are checked before allocating output paths.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    def save(name,data): (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    flags=(ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/flags.make').read_text()
    inc=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1])
    link=shlex.split((ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/link.txt').read_text())
    libs=link[link.index('-o')+2:]
    obj=ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src'
    header=(ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp').read_text()
    (out/'planner_diagnostic_access.hpp').write_text(header.replace('private:','public:'))
    with (out/'compile.log').open('x') as log:
        cmd=['c++','-std=c++14','-O2',*inc,'-I'+str(out),
             str(ROOT/'scripts/test_phase_e_primitive_observation_qp.cpp'),
             str(obj/'local_sparse_scp_planner.cpp.o'),str(obj/'primitive_observation_guard.cpp.o'),
             '-o',str(out/'qp_test'),*libs]
        assert subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode==0,'QP test compile failed'
    old=json.loads((ROOT/'outputs/phase_e_case026_margin_split_20261010/failure_analysis.json').read_text())
    point=old['experiment10']['last_rejection']['point']
    evidence=json.dumps(old['experiment10']['rejected_candidates'][-1]['candidate_evidence'],separators=(',',':'))
    q0=np.array([-.3766,.2381,-.7043,-.6656,.4890,-.8407,-.5084])
    # Counterfactual keeps exactly the same native CDF rows/objective/bounds.
    # Clear synthetic CDF rows isolate the newly added geometric mechanism.
    grad=np.array([.13281116799344905,.4794704609218866,-.01856917018813048,
                   .38250690526011455,.038735342720343235,.020710129289454315,.03805295283754867])
    target=q0-.03*grad/np.linalg.norm(grad)
    w=dict(point=point,margin=.05,evidence=evidence)
    rows=[dict(k=k,point=point,d=.5,g=[0.]*7,qlin=q0.tolist()) for k in range(1,21)]
    base=dict(q0=q0.tolist(),target=target.tolist(),witnesses=[w],rows=rows)
    cases=[dict(base,id='same_input_old_proposal',geometry_enabled=False),
           dict(base,id='same_input_new_proposal',geometry_enabled=True)]
    blocked=copy.deepcopy(base);blocked.update(id='initial_unknown_inside_20mm',geometry_enabled=True,q0=target.tolist())
    for row in blocked['rows']:row['qlin']=target.tolist()
    cases.append(blocked)
    cdf=old['baseline20'];unique={tuple(r['point']):r['margin'] for r in cdf['rows']}
    original=dict(q0=cdf['q0'],target=cdf['q0'],
        witnesses=[dict(point=p,margin=m) for p,m in unique.items()],rows=cdf['rows'])
    cases.extend([dict(original,id='original_160_hard_rows_old',geometry_enabled=False),
                  dict(original,id='original_160_hard_rows_new',geometry_enabled=True)])
    save('qp_inputs.json',cases)
    uri='http://127.0.0.1:'+str(port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',
        ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[];streams=[]
    def start(argv,name):
        stream=(out/(name+'.log')).open('x');streams.append(stream)
        p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);return p
    def until(predicate,timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate():return
            time.sleep(.01)
        raise AssertionError('private ROS condition timed out')
    report=dict(status='FAIL',qualification='Fixed subproblem and private tracker transport, synthetic CDF/joint states; no robot, physical collision claim, or original rejected-path reconstruction.')
    try:
        master=start(['roscore','-p',str(port)],'master')
        def master_ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/observation_test')[0]==1
            except OSError:return False
        until(master_ready,10)
        import rospy,yaml
        from std_msgs.msg import String,Bool,Float64MultiArray
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        rospy.init_node('primitive_observation_validation',disable_signals=True)
        cfg=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        fk=ROOT/'src/arm_description/urdf/Arm.urdf'
        primitive=ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
        cfg['local_planner']['primitive_observation_guard']=dict(enabled=True,fk_urdf=str(fk),
            primitive_urdf=str(primitive),base_frame='base_link',ignored_links=['base_link','link1'],margin_m=.02)
        rospy.set_param('/primitive_observation_qp',cfg)
        rospy.set_param('/robot_description',fk.read_text())
        proc=start([str(out/'qp_test'),str(out/'qp_inputs.json'),str(out/'qp_results.json')],'qp')
        assert proc.wait(timeout=30)==0,'production QP probe failed'
        results=json.loads((out/'qp_results.json').read_text())
        assert results[0]['solved'] and results[1]['solved']
        assert results[0]['cdf_rows']==results[1]['cdf_rows']==20
        assert results[0]['max_slack']==results[1]['max_slack']==0
        assert results[0]['geometry_rows']==0 and results[1]['geometry_rows']>=20
        assert not results[0]['outside_20mm_continuous']
        assert results[1]['outside_20mm_continuous'],'new proposal lacks continuous exclusion proof'
        assert results[2]['status']=='primitive_observation_start_blocked' and not results[2]['solved']
        assert all(not r['solved'] and r['cdf_rows']==160 and r['reachable_cdf_conflicts']==20 for r in results[3:])
        report['qp']=[{k:v for k,v in r.items() if k not in ('q','u')} for r in results]

        # Real actuator-facing tracker, isolated topics and synthetic measured
        # state. Its velocity publisher has no robot/controller subscriber.
        tracker_cfg=dict(joint_names=NAMES,execution=dict(rate=100.,control_dt=.01,
            input_trajectory='/test/committed',joint_states='/test/joints',
            output_velocity_command='/test/velocity',summary_topic='/test/tracker',
            reference_state='/test/reference',position_feedback_gain=0.,
            max_tracking_error=1.,reject_large_start_error=True,max_start_error=.01))
        rospy.set_param('/token_tracker',tracker_cfg)
        status=[];commands=[]
        def callback(msg):status.append((time.monotonic(),dict(x.split('=',1) for x in msg.data.split() if '=' in x)))
        rospy.Subscriber('/test/tracker',String,callback,queue_size=100)
        rospy.Subscriber('/test/velocity',Float64MultiArray,lambda m:commands.append(list(m.data)),queue_size=100)
        joint_pub=rospy.Publisher('/test/joints',JointState,queue_size=1)
        commit_pub=rospy.Publisher('/test/committed',JointTrajectory,queue_size=1)
        stop_pub=rospy.Publisher('/care_planner/execution/vbc_stop_request',String,queue_size=20)
        bool_pub=rospy.Publisher('/care_planner/execution/predicted_vbc_verification_hold',Bool,queue_size=1)
        tracker=start([str(ROOT/'devel/lib/egocentric_arm_planner/trajectory_execution_manager_node'),
                       '__name:=token_tracker'],'tracker')
        until(lambda:joint_pub.get_num_connections()>0 and commit_pub.get_num_connections()>0 and stop_pub.get_num_connections()>0)
        def joints(_):
            joint_pub.publish(JointState(header=__import__('std_msgs.msg',fromlist=['Header']).Header(stamp=rospy.Time.now()),
                name=NAMES,position=[0.]*7,velocity=[0.]*7))
        timer=rospy.Timer(rospy.Duration(.01),joints);joints(None)
        def commit(token):
            t=JointTrajectory();t.header.stamp=rospy.Time(0,token);t.joint_names=NAMES
            t.points=[JointTrajectoryPoint(positions=[0.]*7,velocities=[.02]+[0.]*6,time_from_start=rospy.Duration(0.)),
                JointTrajectoryPoint(positions=[.2]+[0.]*6,velocities=[.02]+[0.]*6,time_from_start=rospy.Duration(10.))]
            commit_pub.publish(t)
        until(lambda:bool(status))
        token=rospy.Time.now().to_nsec();commit(token)
        until(lambda:any(s.get('execution_stamp_ns')==str(token) and s.get('source')=='active_trajectory' for _,s in status))
        start_index=len(status);stop_pub.publish(String(data=f'execution_stamp_ns={token-1}'))
        time.sleep(.05)
        assert not any(s.get('source')=='external_vbc_hold' for _,s in status[start_index:])
        stop_pub.publish(String(data=f'execution_stamp_ns={token}'))
        until(lambda:status[-1][1].get('source')=='external_vbc_hold')
        first_hold=status[-1];phase=float(first_hold[1]['phase_s'])
        bool_pub.publish(Bool(data=False));time.sleep(.05)
        assert status[-1][1]['source']=='external_vbc_hold'
        assert abs(float(status[-1][1]['phase_s'])-phase)<1e-6
        assert commands[-1]==[0.]*7
        newer=rospy.Time.now().to_nsec();commit(newer)
        until(lambda:status[-1][1].get('execution_stamp_ns')==str(newer) and status[-1][1]['source']=='active_trajectory')
        index=len(status);stop_pub.publish(String(data=f'execution_stamp_ns={token}'));time.sleep(.05)
        assert not any(s.get('source')=='external_vbc_hold' for _,s in status[index:])
        timer.shutdown()
        report['tracker']=dict(matching_token_stopped=True,phase_frozen=True,zero_command=True,
            boolean_release_cannot_clear_token=True,new_certified_token_released=True,delayed_old_request_ignored=True)
        save('tracker_status.json',status)
        report['status']='PASS'
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGINT)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
        for stream in streams:stream.close()
        save('report.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
