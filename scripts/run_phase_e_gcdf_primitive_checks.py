#!/usr/bin/env python3
"""Primitive GCDF acceptance on private ROS: no case solver/Gazebo/actuator."""
import argparse
from collections import defaultdict
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

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--repetitions',type=int,default=12)
    ap.add_argument('--geometry-only',action='store_true')
    args=ap.parse_args()
    assert args.repetitions>=6
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    uri='http://127.0.0.1:'+str(port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='localhost',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[];handles=[];report=dict(status='FAIL',scope='offline geometry and isolated ROS; no cases or robot')
    files=[ROOT/p for p in ['devel/lib/care_collision_cdf/cpp_forbidden_voxel_gpu_shadow_node',
        'devel/lib/care_collision_cdf/test_gcdf_primitives','devel/lib/care_confidence_map/gcdf_primitive_anchor_node',
        'devel/lib/libtrajectory_risk_evaluator.so','devel/lib/libbody_sample_model.so',
        'src/care_confidence_map/config/body_samples.yaml','src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
        'src/care_confidence_map/include/care_confidence_map/gcdf_primitive_anchors.hpp',
        'src/care_confidence_map/include/care_confidence_map/primitive_geometry.hpp',
        'src/care_confidence_map/src/gcdf_primitive_anchor_node.cpp',
        'src/care_collision_cdf/src/cpp_forbidden_voxel_gpu_shadow_node.cpp',
        'scripts/test_phase_e_gcdf_primitives.cpp']]
    digest=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report['sha256']=digest()
    def start(argv,name):
        log=(out/name).open('x');handles.append(log)
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p
    def wait(pred,label,seconds=10):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if pred():return
            time.sleep(.01)
        raise AssertionError('timeout: '+label)
    try:
        start(['roscore','-p',str(port)],'master.log')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/gcdf_geometry_probe')[0]==1
            except OSError:return False
        wait(ready,'private ROS master')
        binary=ROOT/'devel/lib/care_collision_cdf/test_gcdf_primitives'
        p=start([str(binary),str(ROOT),str(out),str(args.repetitions)],'cpp.log')
        for i in range(90):
            try:p.wait(timeout=5);break
            except subprocess.TimeoutExpired:
                if i%6==5:print('Geometry/paired timing checks running',flush=True)
        assert p.poll()==0, 'actual C++ geometry/selector test failed; see cpp.log'
        rows=[json.loads(line) for line in (out/'timings.jsonl').read_text().splitlines()]
        assert len(rows)==12*args.repetitions*2
        groups=defaultdict(list)
        for row in rows:groups[(row['pose'],row['knots'],row['map'],row['backend'])].append(row)
        def dist(v):
            v=sorted(v);return dict(p50=v[int(.5*(len(v)-1))],p95=v[int(.95*(len(v)-1))])
        stats=[]
        for pose,knots,mapmode,backend in sorted(groups):
            if backend:continue
            old=groups[(pose,knots,mapmode,0)];new=groups[(pose,knots,mapmode,1)]
            metrics={name:{'samples':dist([r[name] for r in old]),'primitive':dist([r[name] for r in new])}
                     for name in ['selection_ms','total_ms']}
            passed=all(m['primitive'][q]<=m['samples'][q] for m in metrics.values() for q in ['p50','p95'])
            stats.append(dict(pose=pose,knots=knots,map=mapmode,timings=metrics,performance='PASS' if passed else 'FAIL',
                samples_anchors=old[0]['anchors'],primitive_anchors=new[0]['anchors'],
                samples_wire_bytes=old[0]['wire_bytes'],primitive_wire_bytes=new[0]['wire_bytes'],
                old_only=new[0]['old_only'],new_only=new[0]['new_only']))
        report.update(geometry='PASS',timing_rows=len(rows),stats=stats,
            performance='PASS' if all(s['performance']=='PASS' for s in stats) else 'FAIL')
        import roslaunch.config
        launch=str(ROOT/'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch')
        launch_checks={}
        namespaces={'/final_executable_gcdf_pair_export/','/local_scp_pair_export/','/execution_gcdf_audit/'}
        for backend in ['samples','primitive']:
            config=roslaunch.config.load_config_default([(launch,['gcdf_geometry_backend:='+backend,
                'use_local_sparse_scp:=true','final_executable_gcdf_enabled:=true','execution_gcdf_audit_enabled:=true'])],None,verbose=False)
            selected=[n.type for n in config.nodes if n.namespace in namespaces and n.package=='care_confidence_map']
            assert selected==(['trajectory_risk_node']*3 if backend=='samples' else ['gcdf_primitive_anchor_node']*3), selected
            launch_checks[backend]=selected
        report['launch_checks']=launch_checks
        if not args.geometry_only:
            import rospy
            from std_msgs.msg import Header
            from sensor_msgs.msg import PointCloud2,PointField
            from sensor_msgs import point_cloud2
            from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
            from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
            from care_collision_cdf.msg import CollisionCDFWitnessRequest as Req,CollisionCDFWitnessResponse as Resp
            rospy.init_node('primitive_gcdf_protocol_test',disable_signals=True)
            sock=str(out/'gpu.sock')
            worker=start(['/home/zhicheng/miniconda3/envs/viscdf/bin/python',
                str(ROOT/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py'),
                '--checkpoint',str(ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'),
                '--device','cuda','--socket',sock,'--warmup-pairs','8'],'worker.log')
            wait(lambda:Path(sock).exists(),'unchanged CUDA worker',30)
            prefix='/primitive_gcdf_test';clouds={};batches={};pubs={};subs=[]
            joints=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
            q=[-.22706325988282663,.4568144338672266,-.5396616798361127,-.8349629656679631,
                .23558290158128645,-.27101622546461174,-.5447906370079348]
            for channel in ['local','final','execution']:
                topic=prefix+'/'+channel;clouds[channel]=[];batches[channel]=[]
                params=dict(robot_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                    primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                    base_frame='base_link',body_inflation_m=.015,max_eval_timesteps=1 if channel=='execution' else 50,
                    ignored_risk_links=['base_link','link1'],input_trajectory_topic=topic+'/q',output_topic=topic+'/anchors',
                    body_samples_file='/deliberately/missing/body_samples.yaml')
                rospy.set_param('/primitive_'+channel,params)
                subs.append(rospy.Subscriber(topic+'/anchors',PointCloud2,clouds[channel].append))
                subs.append(rospy.Subscriber(topic+'/batch',Batch,batches[channel].append))
                pubs[channel]=rospy.Publisher(topic+'/q',JointTrajectory,queue_size=2)
                start([str(ROOT/'devel/lib/care_confidence_map/gcdf_primitive_anchor_node'),
                    '__name:=primitive_'+channel],channel+'.log')
            cfg=dict(geometry_backend='primitive',anchor_topic=prefix+'/local/anchors',
                final_anchor_topic=prefix+'/final/anchors',execution_anchor_topic=prefix+'/execution/anchors',
                map_topic=prefix+'/map',constraint_batch_topic=prefix+'/local/batch',
                final_constraint_batch_topic=prefix+'/final/batch',execution_constraint_batch_topic=prefix+'/execution/batch',
                gpu_socket=sock,output_jsonl=str(out/'selector.jsonl'),process_on_input_callback=True,
                local_witness_requests_required=True,proximity_margin=.025,execution_proximity_margin=.12)
            rospy.set_param('/primitive_selector',cfg)
            responses=[]
            subs.append(rospy.Subscriber(prefix+'/local/batch/witness_response',Resp,responses.append))
            reqpub=rospy.Publisher(prefix+'/local/batch/witness_request',Req,queue_size=2)
            mappub=rospy.Publisher(prefix+'/map',PointCloud2,queue_size=2)
            start([str(ROOT/'devel/lib/care_collision_cdf/cpp_forbidden_voxel_gpu_shadow_node'),
                '__name:=primitive_selector'],'selector.log')
            wait(lambda:all(p.get_num_connections() for p in list(pubs.values())+[reqpub,mappub]) and
                 all(s.get_num_connections() for s in subs),'complete connections')
            fs=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(
                ['x','y','z','confidence','current_visibility','occupancy'])]
            def publish_map():
                mappub.publish(point_cloud2.create_cloud(Header(stamp=rospy.Time.now(),frame_id='base_link'),fs,
                    [[.1,.1,.3,0.,0.,0.],[.1,.15,.3,1.,0.,1.],[.2,-.05,.5,0.,0.,0.]]))
            timer=rospy.Timer(rospy.Duration(.1),lambda _:publish_map());publish_map();time.sleep(.08)
            checks=[];refs=[]
            for channel in ['local','final','execution']:
                msg=JointTrajectory(header=Header(stamp=rospy.Time.now(),frame_id='base_link'),joint_names=joints[::-1])
                for k in range(1 if channel=='execution' else 2):
                    msg.points.append(JointTrajectoryPoint(positions=q[::-1],time_from_start=rospy.Duration(k*.05)))
                if channel=='local':
                    reqpub.publish(Req(header=msg.header,plan_sequence=1,mode_epoch=2,target_revision=3,progress_epoch=4,
                        dof=7,original_timestep=[1],point_flat=[.1,.1,.3],q_flat=q))
                pubs[channel].publish(msg)
                wait(lambda:bool(clouds[channel]),channel+' primitive cloud')
                cloud=clouds[channel][-1]
                assert cloud.header.stamp==msg.header.stamp and cloud.width==21*len(msg.points)
                decoded=list(point_cloud2.read_points(cloud,field_names=['q'+str(j) for j in range(7)],skip_nans=False))
                assert all(max(abs(a-b) for a,b in zip(row,q))<1e-6 for row in decoded)
                wait(lambda:any(b.header.stamp==msg.header.stamp for b in batches[channel]),channel+' original GCDF batch')
                batch=next(b for b in batches[channel] if b.header.stamp==msg.header.stamp)
                if channel=='local':
                    wait(lambda:bool(responses),'explicit primitive witness response')
                    assert list(responses[-1].status)==['evaluated_unknown'] and batch.num_pairs>=1
                if channel=='execution':assert all(x==1 for x in batch.source_type)
                for i in range(batch.num_pairs):
                    refs.append(dict(point=list(batch.point_flat[3*i:3*i+3]),q=list(batch.q_linearization_flat[7*i:7*i+7]),
                        distance=batch.distance[i],gradient=list(batch.gradient_flat[7*i:7*i+7])))
                checks.append(channel+': real URDF primitive producer -> selector -> unchanged CUDA worker; exact q/step/stamp')
                for fault in ['nan','joint','frame','duplicate','time']:
                    bad=copy.deepcopy(msg);bad.header.stamp=rospy.Time.now()
                    if fault=='nan':bad.points[0].positions[0]=float('nan')
                    if fault=='joint':bad.joint_names[0]='bad'
                    if fault=='frame':bad.header.frame_id='other'
                    if fault=='duplicate':bad.joint_names[0]=bad.joint_names[1]
                    if fault=='time':bad.points[0].time_from_start=rospy.Duration(-.1)
                    before=len(clouds[channel]);pubs[channel].publish(bad);time.sleep(.04)
                    assert len(clouds[channel])==before, 'malformed trajectory published'
            timer.shutdown()
            badmap=point_cloud2.create_cloud(Header(stamp=rospy.Time.now(),frame_id='wrong_frame'),fs,
                [[.1,.1,.3,0.,0.,0.]])
            mappub.publish(badmap);time.sleep(.08)
            wrong=JointTrajectory(header=Header(stamp=rospy.Time.now(),frame_id='base_link'),joint_names=joints)
            wrong.points=[JointTrajectoryPoint(positions=q,time_from_start=rospy.Duration(0.))]
            count=len(batches['final']);pubs['final'].publish(wrong);time.sleep(.15)
            assert len(batches['final'])==count,'wrong map frame became safe/ordinary batch'
            publish_map();time.sleep(.08)
            corrupt=copy.deepcopy(clouds['final'][-1]);corrupt.header.stamp=rospy.Time.now()
            corrupt.width-=1;corrupt.row_step-=corrupt.point_step;corrupt.data=corrupt.data[:-corrupt.point_step]
            corrupt_pub=rospy.Publisher(prefix+'/final/anchors',PointCloud2,queue_size=2)
            wait(lambda:corrupt_pub.get_num_connections()>0,'corrupt geometry test connection')
            corrupt_pub.publish(corrupt);time.sleep(.15)
            assert len(batches['final'])==count,'partial primitive set became a safety batch'
            checks.append('wrong map frame and partial primitive cloud fail closed, no empty SAFE batch')
            checks.append('invalid q/joints/frame/time fails closed on all three producers; no YAML required')
            (out/'gpu_rows.json').write_text(json.dumps(refs,indent=2)+'\n')
            verify=subprocess.run(['/home/zhicheng/miniconda3/envs/viscdf/bin/python',
                str(ROOT/'scripts/test_phase_e_witness_pipeline.py'),'--verify-json',str(out/'gpu_rows.json')],capture_output=True,text=True)
            assert verify.returncode==0,verify.stderr
            report['gpu_gradient_check']=json.loads(verify.stdout.splitlines()[-1])
            assert not any('/command' in t or '/committed_trajectory' in t for t,_ in rospy.get_published_topics())
            checks.append('full 7D gradients independently verified; no actuator or committed publisher')
            report['ros_checks']=checks
        else:report['ros_checks']='NOT_RUN geometry-only'
        assert digest()==report['sha256'],'code/model inputs changed during test'
        report['status']='PASS'
        print(json.dumps({k:v for k,v in report.items() if k not in ['sha256','stats']},indent=2))
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        for h in handles:h.close()
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
