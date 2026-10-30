#!/usr/bin/env python3
"""Built planner, synthetic exact CDF replies and certification/tracker protocol.

No Gazebo or control commands; tests scheduling, not geometric certification.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xmlrpc.client


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--port',type=int,default=11616)
    args=ap.parse_args(); out=args.output_dir.resolve()
    uri=f'http://127.0.0.1:{args.port}'
    import socket
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1',args.port)) != 0, 'port occupied'
    out.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='localhost',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[]; streams=[]
    def start(argv,name):
        f=(out/name).open('x');streams.append(f)
        p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p);return p
    def wait(fn,label,seconds=15):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if fn():return
            time.sleep(.02)
        raise AssertionError('timeout '+label)
    try:
        start(['roscore','-p',str(args.port)],'master.log')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/probe')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy,yaml
        from std_msgs.msg import String,Bool,Header,Float64MultiArray,MultiArrayDimension
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from care_collision_cdf.msg import CollisionCDFWitnessRequest as Request, CollisionCDFWitnessResponse as Response
        rospy.init_node('unknown_liveness_test_driver',disable_signals=True)
        config=copy.deepcopy(yaml.safe_load((repo/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text()))
        lp=config['local_planner']
        for k in ['joint_states','reference_trajectory','cdf_batch_topic','query_trajectory_topic','candidate_trajectory_topic','summary_topic']:
            lp[k]='/unknown_liveness/'+k
        rospy.set_param('/unknown_liveness_planner',config)
        joints=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
        jp=rospy.Publisher(lp['joint_states'],JointState,queue_size=10)
        rp=rospy.Publisher(lp['reference_trajectory'],JointTrajectory,queue_size=2)
        wp=rospy.Publisher(lp['cdf_batch_topic']+'/witness_response',Response,queue_size=20)
        active=rospy.Publisher('/care_planner/active_sensing/visibility_waypoint_active',Bool,queue_size=2,latch=True)
        target=rospy.Publisher('/care_planner/active_sensing/visibility_waypoint_q',Float64MultiArray,queue_size=2,latch=True)
        points=rospy.Publisher('/care_planner/active_sensing/current_visibility_obligation_points',Float64MultiArray,queue_size=2,latch=True)
        vp=rospy.Publisher('/care_planner/verification_outcome',String,queue_size=20)
        tp=rospy.Publisher(lp.get('execution_summary_topic','/care_planner/low_level_tracker/summary'),String,queue_size=20)
        replan=rospy.Publisher('/care_planner/local_planner/replan_request',Bool,queue_size=10)
        state=dict(source=0)
        candidates=[]; stalls=[]
        def query(req):
            n=len(req.original_timestep)
            b=Batch(header=req.header,num_pairs=n,dof=7,original_timestep=req.original_timestep,
                source_type=[state['source']]*n,point_flat=req.point_flat,q_linearization_flat=req.q_flat,
                gradient_flat=[x for _ in range(n) for x in [-1.,0,0,0,0,0,0]],
                distance=[.05-req.q_flat[i*7] for i in range(n)])
            wp.publish(Response(request=req,batch=b,map_stamp=rospy.Time.now(),
                status=['evaluated_unknown' if state['source']==0 else 'evaluated_occupied']*n))
        subs=[rospy.Subscriber(lp['cdf_batch_topic']+'/witness_request',Request,query),
              rospy.Subscriber(lp['candidate_trajectory_topic'],JointTrajectory,lambda m:candidates.append(m)),
              rospy.Subscriber('/care_planner/local_planner/task_stall',String,lambda m:stalls.append(m.data))]
        binary=repo/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'
        start([str(binary),'__name:=unknown_liveness_planner'],'planner.log')
        wait(lambda:all(p.get_num_connections() for p in [jp,rp,wp,active,target,points,vp,tp,replan]),'subscribers')
        results=[]
        for number,(label,source,certify) in enumerate([('occupied',1,True),('uncertified',0,False),('qualified',0,True)],1):
            state['source']=source
            active.publish(Bool(data=True))
            tgt=Float64MultiArray(data=[-.2,0,0,0,0,0,0]);tgt.layout.dim=[MultiArrayDimension(label=f'care_obs_v1_test_{number}',size=7,stride=7)]
            target.publish(tgt);points.publish(Float64MultiArray(data=[number,number,1,.2,0,.5]))
            jp.publish(JointState(header=Header(stamp=rospy.Time.now()),name=joints,position=[0.]*7,velocity=[0.]*7))
            time.sleep(.08)
            base=len(candidates); marker=len(stalls)
            ref=JointTrajectory(header=Header(stamp=rospy.Time.now()),joint_names=joints,
                points=[JointTrajectoryPoint(positions=[0.]*7,time_from_start=rospy.Duration(0)),
                        JointTrajectoryPoint(positions=[.4,0,0,0,0,0,0],time_from_start=rospy.Duration(2))])
            rp.publish(ref);wait(lambda:len(candidates)>base,label+' first candidate')
            start_time=time.monotonic(); next_request=0.; last_raw=0; execution=0
            while time.monotonic()-start_time<3.8:
                elapsed=time.monotonic()-start_time
                jp.publish(JointState(header=Header(stamp=rospy.Time.now()),name=joints,position=[0.]*7,velocity=[0.]*7))
                raw=candidates[-1].header.stamp.to_nsec()
                if raw!=last_raw:
                    last_raw=raw;execution=rospy.Time.now().to_nsec()
                    if certify:vp.publish(String(data=f'result=safe safety_gate=vbc committed=1 verification_view=full_horizon raw_candidate_stamp_ns={raw} execution_stamp_ns={execution}'))
                if execution:
                    age=(rospy.Time.now().to_nsec()-execution)/1e9
                    tp.publish(String(data=f'active=1 complete=0 execution_aborted=0 phase_s={age} execution_stamp_ns={execution}'))
                if elapsed>=next_request:
                    replan.publish(Bool(data=True));next_request=elapsed+.7
                time.sleep(.03)
            early=[s for s in stalls[marker:] if 'classification=NORMAL_UNKNOWN_CONSTRAINED' in s]
            assert bool(early)==(label=='qualified'),(label,early,stalls[marker:])
            if early:
                assert len(early)==1 and 'unknown_evidence_query_stamp_ns=' in early[0]
            results.append(dict(scenario=label,early_stalls=early,candidates=len(candidates)-base))
        result=dict(status='PASS',checks=results,planner_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        for f in streams:f.close()
        (out/'cleanup.json').write_text(json.dumps(dict(remaining=[p.pid for p in processes if p.poll() is None]))+'\n')

if __name__=='__main__':main()
