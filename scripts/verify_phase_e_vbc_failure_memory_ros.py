#!/usr/bin/env python3
"""Built planner and hard QP; synthetic CDF/VBC responses, no actuator nodes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import xmlrpc.client

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--port',type=int,default=11619)
    a=ap.parse_args();out=a.output_dir.resolve();repo=Path(__file__).resolve().parents[1]
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',a.port))!=0
    out.mkdir(parents=True,exist_ok=False)
    uri=f'http://127.0.0.1:{a.port}'
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='localhost',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[];streams=[];stop=threading.Event()
    def start(argv,name):
        f=(out/name).open('x');streams.append(f)
        p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p)
    def wait(fn,label,seconds=10):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if fn():return
            time.sleep(.01)
        raise AssertionError('timeout '+label)
    try:
        start(['roscore','-p',str(a.port)],'master.log')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/probe')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy,yaml
        from std_msgs.msg import String,Bool,Header,Float64MultiArray,MultiArrayDimension
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from care_collision_cdf.msg import CollisionCDFWitnessRequest as Request,CollisionCDFWitnessResponse as Response
        rospy.init_node('failure_memory_driver',disable_signals=True)
        config=copy.deepcopy(yaml.safe_load((repo/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text()))
        lp=config['local_planner']
        # Mainline launch supplies this opt-in separately from the YAML.
        lp['candidate_replacement_enabled']=True
        for k in ['joint_states','reference_trajectory','cdf_batch_topic','query_trajectory_topic','candidate_trajectory_topic','summary_topic']:
            lp[k]='/failure_memory/'+k
        rospy.set_param('/failure_memory_planner',config)
        joints=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
        jp=rospy.Publisher(lp['joint_states'],JointState,queue_size=10)
        rp=rospy.Publisher(lp['reference_trajectory'],JointTrajectory,queue_size=2)
        wp=rospy.Publisher(lp['cdf_batch_topic']+'/witness_response',Response,queue_size=20)
        active=rospy.Publisher('/care_planner/active_sensing/visibility_waypoint_active',Bool,queue_size=2,latch=True)
        target=rospy.Publisher('/care_planner/active_sensing/visibility_waypoint_q',Float64MultiArray,queue_size=2,latch=True)
        points=rospy.Publisher('/care_planner/active_sensing/current_visibility_obligation_points',Float64MultiArray,queue_size=2,latch=True)
        repair=rospy.Publisher(lp['recovery_topic'],Bool,queue_size=2,latch=True)
        vp=rospy.Publisher('/care_planner/verification_outcome',String,queue_size=20)
        replan=rospy.Publisher('/care_planner/local_planner/replan_request',Bool,queue_size=2)
        candidates=[];identities={};triggers=[];summaries=[];state=dict(q=[0.]*7)
        def query(req):
            n=len(req.original_timestep)
            batch=Batch(header=req.header,num_pairs=n,dof=7,original_timestep=req.original_timestep,
                source_type=[0]*n,point_flat=req.point_flat,q_linearization_flat=req.q_flat,
                gradient_flat=[0.]*(n*7),distance=[1.]*n)
            wp.publish(Response(request=req,batch=batch,map_stamp=rospy.Time.now(),status=['evaluated_unknown']*n))
        def identity(msg):
            fields=dict(word.split('=',1) for word in msg.data.split())
            identities[int(fields['raw_candidate_stamp_ns'])]=fields['observation_token']
        subs=[rospy.Subscriber(lp['cdf_batch_topic']+'/witness_request',Request,query),
              rospy.Subscriber(lp['candidate_trajectory_topic'],JointTrajectory,lambda m:candidates.append(m)),
              rospy.Subscriber('/care_planner/local_planner/observation_candidate_identity',String,identity),
              rospy.Subscriber('/care_planner/local_planner/candidate_replacement_trigger',String,lambda m:triggers.append(m.data)),
              rospy.Subscriber(lp['summary_topic'],String,lambda m:summaries.append(m.data))]
        binary=repo/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'
        start([str(binary),'__name:=failure_memory_planner'],'planner.log')
        wait(lambda:all(p.get_num_connections() for p in [jp,rp,wp,active,target,points,repair,vp,replan]),'connections')
        def joints_loop():
            while not stop.is_set():
                jp.publish(JointState(header=Header(stamp=rospy.Time.now()),name=joints,position=state['q'],velocity=[0.]*7))
                stop.wait(.02)
        worker=threading.Thread(target=joints_loop);worker.start()
        seq=0
        def publish_target(token,oid,ps,pose):
            nonlocal seq
            msg=Float64MultiArray(data=pose)
            msg.layout.dim=[MultiArrayDimension(label=token,size=7,stride=7),MultiArrayDimension(label='care_qmask_v1_1111111',size=7,stride=7)]
            target.publish(msg);seq+=1
            points.publish(Float64MultiArray(data=[seq,oid,len(ps)]+[x for p in ps for x in p]))
        def candidate(token,base):
            wait(lambda:any(identities.get(m.header.stamp.to_nsec())==token for m in candidates[base:]),'candidate '+token)
            return next(m for m in reversed(candidates[base:]) if identities.get(m.header.stamp.to_nsec())==token)
        def reject(msg,bad=False):
            raw=msg.header.stamp.to_nsec();audited=raw+1
            evidence=dict(trajectory_stamp_ns=str(audited+int(bad)),violations=[dict(point=[.05,-.25,.8])])
            text=f'result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns={raw} audited_trajectory_stamp_ns={audited} vbc_evidence='+json.dumps(evidence,separators=(',',':'))
            vp.publish(String(data=text));return text
        results=[]
        active.publish(Bool(data=True));repair.publish(Bool(data=True));time.sleep(.1)
        for i,label in enumerate(['qualified','pose_changed','geometry_shrunk','wrong_audited_evidence','motion_excursion'],11):
            oid=i;old=f'care_obs_v1_{oid}_old';new=f'care_obs_v1_{oid}_new'
            ps=[[.05,-.25,.8],[.05,-.25,.85]];pose=[.3,0,0,0,0,0,0]
            state['q']=[0.]*7;base=len(candidates);marker=len(triggers)
            publish_target(old,oid,ps,pose)
            rp.publish(JointTrajectory(header=Header(stamp=rospy.Time.now()),joint_names=joints,
                points=[JointTrajectoryPoint(positions=[0.]*7,time_from_start=rospy.Duration(0)),
                        JointTrajectoryPoint(positions=pose,time_from_start=rospy.Duration(2))]))
            first=candidate(old,base);first_text=reject(first)
            time.sleep(.06)
            assert len(triggers)==marker,(label,'unexpected first trigger')
            if label=='motion_excursion':
                state['q']=[.02,0,0,0,0,0,0];time.sleep(.06);state['q']=[0.]*7;time.sleep(.04)
            grown=ps+[[0.,-.25,.75]]
            if label=='geometry_shrunk':grown=[ps[0]]
            if label=='pose_changed':pose[0]+=.01
            base=len(candidates);publish_target(new,oid,grown,pose)
            second=candidate(new,base);second_text=reject(second,label=='wrong_audited_evidence')
            time.sleep(.10)
            found=[t for t in triggers[marker:] if f'observation_token={new} ' in t]
            assert bool(found)==(label=='qualified'),(label,found)
            if found:
                assert len(found)==1 and f'raw_candidate_stamp_ns={second.header.stamp.to_nsec()} ' in found[0]
                vp.publish(String(data=second_text));time.sleep(.04)
                assert len(triggers)==marker+1,'duplicate minted trigger'
            results.append(dict(scenario=label,first_raw=first.header.stamp.to_nsec(),
                second_raw=second.header.stamp.to_nsec(),triggers=found,
                injected_outcomes=[first_text,second_text]))
        result=dict(status='PASS',checks=results,planner_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    scope='Built planner/hard QP with synthetic exact CDF/VBC responses; no execution certification')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        stop.set()
        if 'worker' in locals():worker.join(2)
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        for f in streams:f.close()
        (out/'cleanup.json').write_text(json.dumps(dict(remaining=[p.pid for p in processes if p.poll() is None]))+'\n')

if __name__=='__main__':main()
