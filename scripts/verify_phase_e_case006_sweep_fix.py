#!/usr/bin/env python3
"""Production-library regression and optional isolated live VBC startup gates.

ROS mode starts a private master, synthetic point-confidence service and VBC
selectors only. It never starts a planner, controller, Gazebo or model worker.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import xmlrpc.client
from diagnose_phase_e_case006_o6_recovery import ROOT, digest

FIXTURE=ROOT/'outputs/phase_e_case006_observation_paths_20261006_v2/production_sweep/input.json'


def offline(out):
    library=ROOT/'devel/lib/libtrajectory_risk_evaluator.so'
    source=ROOT/'scripts/test_phase_e_case006_sweep_start.cpp'
    command=['g++','-std=c++14','-O0','-DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION','-DPINOCCHIO_WITH_URDFDOM',
        '-DPINOCCHIO_WITH_HPP_FCL','-DBOOST_MPL_LIMIT_LIST_SIZE=30','-DBOOST_MPL_LIMIT_VECTOR_SIZE=30',
        '-I'+str(ROOT/'src/care_confidence_map/include'),'-I'+str(ROOT/'devel/include'),
        '-I/opt/ros/noetic/include','-I/usr/include/eigen3',str(source),'-o',str(out/'test'),
        '-L'+str(ROOT/'devel/lib'),'-L/opt/ros/noetic/lib','-ltrajectory_risk_evaluator',
        '-lpinocchio_default','-lroscpp','-lrosconsole','-lrostime']
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'devel/lib')+':/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu')
    subprocess.run(command,env=env,check=True,capture_output=True,text=True)
    result=subprocess.run([str(out/'test'),str(FIXTURE)],env=env,capture_output=True,text=True,check=True)
    report=json.loads(result.stdout)
    report['input_sha256']={str(p.relative_to(ROOT)):digest(p) for p in [source,library,FIXTURE]}
    return report


def live(out,port,refinement_fixtures=None):
    uri=f'http://127.0.0.1:{port}'
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),
                      ROS_LOG_DIR=str(out/'ros_logs'))
    processes=[];files=[];summary={};mode={'known':False,'points':[(.05,-.2,.75)]};checks=[]
    def start(argv,name):
        stream=(out/(name+'.log')).open('w');files.append(stream)
        p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(p);return p
    def wait(test,timeout=12):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if test():return
            time.sleep(.025)
        raise AssertionError('condition timed out')
    def ready():
        try:return xmlrpc.client.ServerProxy(uri).getPid('/case006_sweep_test')[0]==1
        except OSError:return False
    try:
        start(['roscore','-p',str(port)],'master');wait(ready)
        import rospy
        import yaml
        from care_confidence_map.srv import QueryConfidence,QueryConfidenceResponse
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from std_msgs.msg import String
        from sensor_msgs.msg import JointState
        rospy.init_node('case006_sweep_test',disable_signals=True)
        def confidence(req):
            values=[1. if mode['known'] or not any(max(abs(p.x-x),abs(p.y-y),abs(p.z-z))<1e-8 for x,y,z in mode['points']) else 0.
                    for p in req.points]
            return QueryConfidenceResponse(values,[0.]*len(values),[1]*len(values))
        service=rospy.Service('/case006_sweep_test/confidence',QueryConfidence,confidence)
        fixture=json.loads(FIXTURE.read_text())
        config0=yaml.safe_load((ROOT/'src/care_confidence_map/config/trajectory_risk.yaml').read_text())['trajectory_vbc']
        binary=ROOT/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node'
        endpoints={};subscriptions=[]
        selectors=[('candidate',.04,None),('execution',.02,None),('disabled',.04,False)]
        if refinement_fixtures:selectors.append(('measured',.04,None))
        measured_pub=rospy.Publisher('/care_arm/joint_states',JointState,queue_size=1)
        measured_q={'q':fixture['trajectory'][0]['q']}
        def publish_measured(_):
            msg=JointState();msg.header.stamp=rospy.Time.now()
            msg.name=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
            msg.position=measured_q['q'];measured_pub.publish(msg)
        timer=rospy.Timer(rospy.Duration(.02),publish_measured)
        for label,margin,bounds in selectors:
            name='sweep_fix_'+label;topic='/'+name
            config=dict(config0,geometry_backend='primitive',robot_urdf_file=fixture['urdf'],
                primitive_urdf_file=fixture['primitive_urdf'],swept_volume_margin_m=margin,
                input_trajectory_topic=topic+'/task',predicted_trajectory_topic=topic+'/prediction',
                output_target_topic=topic+'/target',candidate_active_topic=topic+'/active',
                active_set_points_topic=topic+'/points',active_set_bundle_topic=topic+'/bundle',
                force_bootstrap_topic=topic+'/bootstrap',confidence_query_service='/case006_sweep_test/confidence',
                prefer_predicted_trajectory=True,predicted_trajectory_timeout=3.,event_driven_eval=True,
                preserve_periodic_eval=False,eval_rate=5.,min_visibility_before_sweep_margin_s=.3,
                map_resolution=.05,map_x_min=-.95,map_x_max=.95,map_y_min=-.95,map_y_max=.95,map_z_min=0.,map_z_max=1.15)
            config.pop('continuous_motion_bound_enabled',None)
            config['include_measured_body']=label=='measured'
            if bounds is not None:config['continuous_motion_bound_enabled']=bounds
            rospy.set_param('/'+name,{'trajectory_vbc':config})
            summary[label]=[]
            subscriptions.append(rospy.Subscriber(topic+'/summary',String,lambda m,k=label:summary[k].append(m.data)))
            process=start([str(binary),'__name:='+name,'/care_planner/trajectory_risk/vbc_summary:='+topic+'/summary'],label)
            if bounds is False:
                wait(lambda:process.poll() is not None)
                assert process.returncode!=0 and not summary[label]
                checks.append('explicit disabled enclosure refuses startup without safe verdict');continue
            pub=rospy.Publisher(topic+'/prediction',JointTrajectory,queue_size=1)
            wait(lambda:pub.get_num_connections()>0 or process.poll() is not None)
            assert process.poll() is None
            endpoints[label]=pub
        def evaluate(label,trajectory):
            msg=JointTrajectory();msg.header.stamp=rospy.Time.now();stamp=str(msg.header.stamp.to_nsec())
            msg.joint_names=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
            for row in trajectory:
                p=JointTrajectoryPoint(positions=row['q']);p.time_from_start=rospy.Duration(row['time_s']);msg.points.append(p)
            endpoints[label].publish(msg)
            def matching():return [s for s in summary[label] if f' trajectory_stamp_ns={stamp} ' in s]
            wait(lambda:bool(matching()))
            fields=dict(re.findall(r'(\w+)=([^\s]+)',matching()[-1]));fields['evidence']=json.loads(fields['vbc_evidence'])
            return fields
        original=evaluate('candidate',fixture['trajectory'])
        assert original['has_violation']=='1' and original['continuous_motion_bound_enabled']=='1'
        assert original['start_support_unknown_count']=='1'
        ev=original['evidence'];assert ev['spatial_semantics']=='voxel_centers_piecewise_linear_enclosure'
        assert len(ev['violations'])==1
        v=ev['violations'][0]
        assert v['initial_support_unknown'] and abs(v['initial_support_distance_m']-.03901809669363678)<1e-10
        assert v['sweep_s']==0 and v['latest_see_s']==-.3
        checks.append('original CASE006 path rejected at 40mm with exact initial-support evidence and start deadline')
        hold=[fixture['trajectory'][0],dict(time_s=.1,q=fixture['trajectory'][0]['q'])]
        assert evaluate('candidate',hold)['has_violation']=='1'
        assert evaluate('candidate',[fixture['trajectory'][0]])['has_violation']=='1'
        checks.append('stationary hold and single-pose candidate cannot erase unknown initial point')
        execution=evaluate('execution',fixture['trajectory']);assert execution['has_violation']=='0'
        checks.append('20mm execution shell unchanged for the isolated unknown-point fixture')
        mode['known']=True
        known=evaluate('candidate',fixture['trajectory']);assert known['has_violation']=='0'
        checks.append('fresh known-free evidence releases the same candidate at the same 40mm margin')
        mode['known']=False
        assert evaluate('candidate',fixture['trajectory'])['has_violation']=='1'
        checks.append('revoked confidence restores rejection on a fresh exact identity')
        refinement=[]
        if refinement_fixtures:
            for f in json.loads(refinement_fixtures.read_text())['fixtures']:
                mode['points']=f['request']['points']
                trajectory=f['request']['trajectory']
                fields=evaluate('candidate',trajectory)
                clear=f['audited_ns'] in ['25284000000','25434000000']
                assert (fields['has_violation']=='0')==clear,(f['audited_ns'],fields)
                assert 0<=int(fields['sweep_refinement_fk'])<=256
                if clear:
                    assert int(fields['sweep_refinement_excluded'])==len(mode['points'])
                    measured_q['q']=trajectory[0]['q'];time.sleep(.06)
                    measured=evaluate('measured',trajectory)
                    assert measured['has_violation']=='0' and measured['include_measured_body']=='1'
                    assert int(measured['sweep_refinement_excluded'])==len(mode['points'])
                else:measured=None
                refinement.append(dict(audited_ns=f['audited_ns'],candidate=fields,measured=measured))
            checks.append('seven exact CASE000 paths: two proven false positives released, five genuine rejections retained')
            checks.append('measured-body audit agrees on nominal exclusion when measured pose equals the certified initial pose')
        return dict(status='PASS',checks=checks,unknown=original,known=known,execution=execution,
            refinement=refinement,
            qualification='isolated production selectors with synthetic confidence; no planner/controller/simulator',
            input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [FIXTURE,binary,
                ROOT/'devel/lib/libtrajectory_risk_evaluator.so',Path(__file__).resolve()]})
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        for f in files:f.close()
        (out/'cleanup.json').write_text(json.dumps(dict(remaining=[p.pid for p in processes if p.poll() is None]))+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--ros',action='store_true')
    ap.add_argument('--refinement-fixtures',type=Path);args=ap.parse_args()
    port=None
    if args.ros:
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=live(args.output_dir.resolve(),port,args.refinement_fixtures) if args.ros else offline(args.output_dir.resolve())
    (args.output_dir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['unknown','known','execution','refinement','input_sha256']},indent=2))


if __name__=='__main__':main()
