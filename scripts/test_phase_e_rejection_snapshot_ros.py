#!/usr/bin/env python3
"""Actual continuity ROS node + deterministic synthetic auditors; no actuator."""
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
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); out=args.output_dir.resolve()
    uri='http://127.0.0.1:11340'
    try: xmlrpc.client.ServerProxy(uri).getPid('/snapshot_test_probe')
    except OSError: pass
    else: raise RuntimeError('Private master port occupied')
    out.mkdir(parents=True,exist_ok=False)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='localhost',ROS_LOG_DIR=str(out/'logs'),ROS_HOME=str(out/'ros_home'))
    children=[]; handles=[]; report=dict(status='FAIL',checks=[])
    def start(argv,name):
        stream=(out/name).open('x');handles.append(stream)
        p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    def wait(pred,label,seconds=8):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if pred():return
            time.sleep(.005)
        raise AssertionError('timeout: '+label)
    try:
        start(['roscore','-p','11340'],'master.log')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/snapshot_test_probe')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy
        from std_msgs.msg import String,Bool
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from care_collision_cdf.msg import CollisionCDFRejectionFeedback
        rospy.init_node('snapshot_protocol_test',disable_signals=True)
        params=dict(input_topic='/snapshot/raw',output_topic='/snapshot/vbc_query',
            committed_output_topic='/snapshot/committed',global_summary_topic='/snapshot/vbc_result',
            verification_event_topic='/snapshot/outcome',final_gcdf_query_topic='/snapshot/gcdf_query',
            final_gcdf_batch_topic='/snapshot/gcdf_result',repair_active_topic='/snapshot/repair',
            joint_state_topic='/snapshot/measured',candidate_audit_context_topic='/snapshot/context',
            rejection_snapshot_dir=str(out/'snapshots'),final_gcdf_enabled=True,continuation_enabled=False)
        params.update(repair_prefix_verification_enabled=True,repair_execution_prefix_s=.25)
        rospy.set_param('/snapshot_continuity',params)
        pub={k:rospy.Publisher(topic,typ,queue_size=10) for k,topic,typ in [
            ('raw','/snapshot/raw',JointTrajectory),('gcdf','/snapshot/gcdf_result',Batch),
            ('vbc','/snapshot/vbc_result',String),('repair','/snapshot/repair',Bool),
            ('measured','/snapshot/measured',JointState),('context','/snapshot/context',String)]}
        gcdf=[];vbc=[];outcomes=[];commits=[];feedback=[]
        subs=[rospy.Subscriber(t,typ,target.append) for t,typ,target in [
            ('/snapshot/gcdf_query',JointTrajectory,gcdf),('/snapshot/vbc_query',JointTrajectory,vbc),
            ('/snapshot/outcome',String,outcomes),('/snapshot/committed',JointTrajectory,commits),
            ('/snapshot/raw/gcdf_rejection_feedback',CollisionCDFRejectionFeedback,feedback)]]
        start(['rosrun','egocentric_arm_planner','optimized_trajectory_continuity_node.py',
               '__name:=snapshot_continuity'],'continuity.log')
        wait(lambda:all(p.get_num_connections() for p in pub.values()),'subscriptions')
        wait(lambda:all(s.get_num_connections() for s in subs),'auditor/output connections')
        joints=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
        measured=JointState(name=joints,position=[0.]*7,velocity=[0.]*7)
        timer=rospy.Timer(rospy.Duration(.02),lambda _:pub['measured'].publish(measured))
        pub['repair'].publish(Bool(data=True));time.sleep(.06)
        raw=JointTrajectory(joint_names=joints);raw.header.stamp=rospy.Time.now();raw.header.frame_id='base_link'
        for i in range(21):
            raw.points.append(JointTrajectoryPoint(positions=[i*.0003]+[0.]*6,
                velocities=[.006]+[0.]*6,time_from_start=rospy.Duration(i*.05)))
        stamp=str(raw.header.stamp.to_nsec())
        pub['context'].publish(String(data=json.dumps(dict(schema=1,raw_candidate_stamp_ns=stamp,
            observation_token='synthetic_protocol',mode_epoch=1,target_revision=1,progress_epoch=1))))
        pub['raw'].publish(raw)
        wait(lambda:len(gcdf)==1,'real executable GCDF query')
        pub['gcdf'].publish(Batch(header=gcdf[0].header,dof=7,num_pairs=0))
        wait(lambda:len(vbc)==1,'same executable to VBC')
        vs=vbc[0].header.stamp.to_nsec()
        def verdict(s):
            evidence=dict(schema=1,trajectory_stamp_ns=str(s),bundle_seq=7,geometry_backend='synthetic_protocol',
                violations=[dict(point=[.1,-.2,.8],sweeping_link='wrist_link2',source_collision_index=0,
                    sweep_eval_k=2,latest_see_s=-.2)])
            pub['vbc'].publish(String(data='trajectory_source=predicted success=1 trajectory_stamp_ns=%d has_violation=1 vbc_evidence=%s'%(
                s,json.dumps(evidence,separators=(',',':')))))
        verdict(vs+1);time.sleep(.015);assert not outcomes
        verdict(vs);wait(lambda:len(outcomes)==1,'unsafe outcome')
        wait(lambda:len(list((out/'snapshots').glob('*.json')))==1,'snapshot file')
        saved=json.loads(next((out/'snapshots').glob('*.json')).read_text())
        assert saved['context_status']=='exact_raw_stamp'
        assert saved['raw_candidate_stamp_ns']==stamp and saved['audited_trajectory_stamp_ns']==str(vs)
        assert len(saved['raw_trajectory']['points'])==21
        assert saved['raw_trajectory']['points'][7]['positions'][0]==raw.points[7].positions[0]
        assert saved['final_gcdf_trajectory']['points']==saved['audited_trajectory']['points']
        assert saved['final_gcdf_batch']['header']['stamp']==saved['final_gcdf_trajectory']['stamp_ns']
        assert saved['handoff_trajectory'] and saved['measured_at_dispatch'] and saved['measured_at_rejection']
        assert 'result=unsafe' in outcomes[0].data and 'safety_gate=vbc' in outcomes[0].data and not commits
        verdict(vs);time.sleep(.06);assert len(outcomes)==1
        # A final-GCDF rejection must also save its exact executable, without
        # inventing a VBC audit epoch or publishing to the VBC/commit topics.
        raw.header.stamp=rospy.Time.now(); stamp2=str(raw.header.stamp.to_nsec())
        pub['context'].publish(String(data=json.dumps(dict(schema=1,raw_candidate_stamp_ns=stamp2,
            observation_token='synthetic_gcdf',mode_epoch=1,target_revision=2,progress_epoch=1))))
        pub['raw'].publish(raw);wait(lambda:len(gcdf)==2,'second executable')
        b=Batch(header=gcdf[1].header,dof=7,num_pairs=1,original_timestep=[1],source_type=[0],
            point_flat=[.2,-.15,.55],q_linearization_flat=gcdf[1].points[1].positions,
            distance=[-.01],gradient_flat=[0.,0.,0.,0.,0.,0.,1.])
        pub['gcdf'].publish(b);wait(lambda:len(outcomes)==2,'GCDF unsafe outcome')
        wait(lambda:len(list((out/'snapshots').glob('*.json')))==2,'GCDF snapshot')
        records=[json.loads(p.read_text()) for p in (out/'snapshots').glob('*.json')]
        saved_gcdf=next(r for r in records if r['safety_gate']=='gcdf_unknown')
        assert saved_gcdf['raw_candidate_stamp_ns']==stamp2
        assert saved_gcdf['audited_trajectory_stamp_ns']==str(b.header.stamp.to_nsec())
        row=saved_gcdf['gcdf_evidence']['violations'][0]
        assert row['candidate_q']==list(gcdf[1].points[1].positions) and row['gradient'][6]==1.
        assert row['source_type']==0 and len(vbc)==1 and not commits
        assert 'audited_trajectory_stamp_ns='+str(b.header.stamp.to_nsec()) in outcomes[1].data
        wait(lambda:len(feedback)==1,'bounded final-GCDF feedback')
        assert feedback[0].raw_candidate_stamp==raw.header.stamp
        assert feedback[0].header.stamp==b.header.stamp and not feedback[0].overflow
        assert list(feedback[0].point_flat)==list(b.point_flat)
        assert not any('/command' in t for t,_ in rospy.get_published_topics())
        report.update(status='PASS',checks=['actual catkin node startup',
            'raw -> executable GCDF -> exact VBC rejection -> full precision snapshot',
            'near-stamp rejected; duplicate ignored; exact context joined',
            'GCDF/VBC identical q/t; distinct epochs preserved; no commit or actuator',
            'GCDF UNKNOWN rejection captured with exact epoch, full 7D gradient, no VBC/commit',
            'negative point identities published with exact raw/audited stamps, independent of QP timestep'],
            outcome=outcomes[0].data,snapshot=str(next((out/'snapshots').glob('*.json'))))
        timer.shutdown();print(json.dumps(report,indent=2))
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        for h in handles:h.close()
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
