#!/usr/bin/env python3
"""Primitive diagnostic/legacy migration tests on a private ROS master only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
import xmlrpc.client

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    uri='http://127.0.0.1:'+str(port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='localhost',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[];handles=[];report=dict(status='FAIL',checks=[],scope='isolated ROS and synthetic fixtures; no cases/CUDA/robot')
    binaries=['trajectory_risk_node','predicted_vbc_auditor_node','trajectory_vbc_selector_node',
              'trajectory_vbc_selector_legacy_node','body_observability_node','body_sample_model_node','test_diagnostic_primitives']
    files=[ROOT/'devel/lib/care_confidence_map'/n for n in binaries]+[
        ROOT/'devel/lib/libtrajectory_risk_evaluator.so',ROOT/'devel/lib/libbody_sample_model.so',
        ROOT/'src/care_confidence_map/config/body_samples.yaml',ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',
        ROOT/'src/arm_description/urdf/Arm.urdf']
    files += list((ROOT/'src/care_confidence_map/include/care_confidence_map').glob('*primitive*.hpp'))
    files += [ROOT/'src/care_confidence_map/src/trajectory_risk_evaluator.cpp',Path(__file__).resolve(),
              ROOT/'scripts/test_phase_e_diagnostic_primitives.cpp']
    files += [ROOT/'src/care_confidence_map/src'/(b+'.cpp') for b in binaries if b!='test_diagnostic_primitives']
    files += [ROOT/'src/care_confidence_map/include/care_confidence_map'/h for h in
              ['body_geometry_params.hpp','confidence_query_validation.hpp','trajectory_risk_evaluator.hpp']]
    digest=lambda:{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report['sha256']=digest()
    def start(cmd,name):
        h=(out/name).open('x');handles.append(h)
        p=subprocess.Popen(cmd,stdout=h,stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p
    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
    def wait(pred,label,seconds=20):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if pred():return
            time.sleep(.025)
        raise AssertionError('Timeout: '+label)
    try:
        import yaml
        cfg=yaml.safe_load((ROOT/'src/care_confidence_map/config/trajectory_risk.yaml').read_text())
        (out/'sensor_frames.txt').write_text('\n'.join(cfg['trajectory_vbc']['sensor_frames'])+'\n')
        p=start([str(ROOT/'devel/lib/care_confidence_map/test_diagnostic_primitives'),str(ROOT),str(out)],'cpp.log')
        wait(lambda:p.poll() is not None,'C++ oracle/performance',180)
        assert p.returncode==0,'C++ failed; see cpp.log'
        report['checks'].append('C++: grid oracle, exact audit distances, raw q, no YAML, bounded failure')
        rows=[json.loads(s) for s in (out/'timings.jsonl').read_text().splitlines()];assert len(rows)==240
        stats=[]
        for pose in range(3):
            item=dict(pose=pose)
            for metric in ['probe_ms','audit_ms']:
                pair={}
                for backend in [0,1]:
                    v=sorted(r[metric] for r in rows if r['pose']==pose and r['primitive']==backend)
                    pair['primitive' if backend else 'samples']=dict(p50=v[19],p95=v[37])
                pair['status']='PASS' if all(pair['primitive'][q]<=pair['samples'][q] for q in ['p50','p95']) else 'FAIL'
                item[metric]=pair
            stats.append(item)
        report['timing']=stats
        report['performance']='PASS' if all(s[m]['status']=='PASS' for s in stats for m in ['probe_ms','audit_ms']) else 'FAIL'
        report['coverage']=[json.loads(s) for s in (out/'coverage.jsonl').read_text().splitlines()]
        start(['roscore','-p',str(port)],'master.log')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/diagnostic_test')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy
        import roslaunch.config
        from std_msgs.msg import String
        from geometry_msgs.msg import TransformStamped,PointStamped
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import MarkerArray,Marker
        from tf2_ros import StaticTransformBroadcaster
        from care_confidence_map.srv import QueryConfidence,QueryConfidenceResponse
        rospy.init_node('diagnostic_primitive_test',disable_signals=True)
        mode=['free'];requests=[]
        def query(req):
            n=len(req.points);requests.append(n)
            if mode[0]=='bad_size':return QueryConfidenceResponse([],[],[])
            value=float('nan') if mode[0]=='nan' else 1. if mode[0]=='free' else 0.
            return QueryConfidenceResponse([value]*n,[0.]*n,[0 if mode[0]=='outside' else 1]*n)
        service=rospy.Service('/diagnostic_test/query',QueryConfidence,query)
        broadcaster=StaticTransformBroadcaster();transforms=[]
        for name in ['link1','link2','link3','link4','wrist_link1','wrist_link2','wrist_link3']:
            t=TransformStamped();t.header.frame_id='base_link';t.child_frame_id=name;t.header.stamp=rospy.Time.now();t.transform.rotation.w=1.;transforms.append(t)
        broadcaster.sendTransform(transforms)
        robot=str(ROOT/'src/arm_description/urdf/Arm.urdf');primitive=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
        joints=[j.get('name') for j in ET.parse(robot).getroot().findall('joint') if j.get('type') in ['revolute','continuous']]
        assert len(joints)==7
        for backend in ['samples','primitive']:
            prefix='/diagnostic_test/'+backend;summaries={};markers={};anchors=[];subs=[];procs=[]
            def collect(topic,kind,key,store):
                store[key]=[];subs.append(rospy.Subscriber(topic,kind,lambda msg:store[key].append(msg),queue_size=20))
            for binary,ns,key in [('trajectory_risk_node','trajectory_risk','risk'),('predicted_vbc_auditor_node','predicted_vbc','audit'),
                    ('trajectory_vbc_selector_legacy_node','trajectory_vbc','legacy'),('trajectory_vbc_selector_node','trajectory_vbc','layers'),
                    ('body_observability_node','body_observability','obs'),('body_sample_model_node','body_sample_model','viz')]:
                name='diagnostic_'+backend+'_'+key;params=copy.deepcopy(cfg)
                p=params.setdefault(ns,{})
                p.update(geometry_backend=backend,robot_urdf_file=robot,primitive_urdf_file=primitive,
                    body_samples_file=str(out/'MISSING.yaml') if backend=='primitive' else str(ROOT/'src/care_confidence_map/config/body_samples.yaml'),
                    base_frame='base_link',map_frame='base_link',input_trajectory_topic=prefix+'/trajectory',
                    prediction_topic=prefix+'/trajectory',predicted_trajectory_topic=prefix+'/trajectory',
                    confidence_query_service='/diagnostic_test/query',query_timeout=.1,
                    refresh_body_prior_before_query=False,require_active=False,target_topic=prefix+'/target',
                    output_namespace=prefix+'/risk',summary_topic=prefix+'/audit',
                    output_target_topic=prefix+'/'+key+'/target',candidate_active_topic=prefix+'/'+key+'/active',
                    marker_topic=prefix+'/viz',show_text_labels=False,event_driven_eval=True,
                    forbidden_space_pair_publish_enabled=True,forbidden_space_pair_topic=prefix+'/anchors')
                rospy.set_param('/'+name,params)
                cmd=[str(ROOT/'devel/lib/care_confidence_map'/binary),'__name:='+name]
                if key in ['legacy','layers']:cmd+=['/care_planner/trajectory_risk/vbc_summary:='+prefix+'/'+key]
                if key=='obs':cmd+=['/care_planner/body_observability/summary:='+prefix+'/obs']
                if key=='viz':collect(prefix+'/viz',MarkerArray,'viz',markers)
                else:collect(prefix+'/risk/summary' if key=='risk' else prefix+'/'+key,String,key,summaries)
                proc=start(cmd,name+'.log');procs.append(proc)
            subs.append(rospy.Subscriber(prefix+'/anchors',PointCloud2,lambda m:anchors.append(m)))
            pub=rospy.Publisher(prefix+'/trajectory',JointTrajectory,queue_size=1,latch=True)
            target_pub=rospy.Publisher(prefix+'/target',PointStamped,queue_size=1,latch=True)
            wait(lambda:pub.get_num_connections()>=4,'four trajectory consumers '+backend)
            assert all(p.poll() is None for p in procs),'node failed startup'
            target=PointStamped();target.header.frame_id='base_link';target.point.x=.1;target.point.z=.4;target_pub.publish(target)
            traj=JointTrajectory();traj.header.frame_id='base_link';traj.joint_names=list(reversed(joints))
            for k in range(2):
                point=JointTrajectoryPoint();point.positions=[.002*k*(7-j) for j in range(7)]
                point.time_from_start=rospy.Duration(.05*k);traj.points.append(point)
            def send():traj.header.stamp=rospy.Time.now();pub.publish(traj)
            mode[0]='free';send()
            wait(lambda:all(summaries[k] for k in summaries) and markers['viz'] and anchors,'all output interfaces '+backend)
            # The fail-closed visualizer may first publish DELETEALL while its
            # TF listener connects. Wait for complete geometry, not merely the
            # first message. Missing-TF rejection is checked separately below.
            wait(lambda:sum(m.action==Marker.ADD and m.type!=Marker.TEXT_VIEW_FACING
                            for m in markers['viz'][-1].markers)==(26 if backend=='primitive' else 90),
                 'complete visualization geometry '+backend)
            assert 'success=1' in summaries['risk'][-1].data
            for key in ['legacy','layers']:assert 'has_violation=0' in summaries[key][-1].data
            assert 'diagnostic_only=1' in summaries['audit'][-1].data
            solid=[m for m in markers['viz'][-1].markers if m.action==Marker.ADD and m.type!=Marker.TEXT_VIEW_FACING]
            if backend=='primitive':
                assert len(solid)==26 and {m.type for m in solid} >= {Marker.CUBE,Marker.CYLINDER}
                fields={f.name for f in anchors[-1].fields};assert 'primitive_schema' in fields and 'radius' not in fields
                assert anchors[-1].width==42 and anchors[-1].header.stamp==traj.header.stamp
                offsets={f.name:f.offset for f in anchors[-1].fields}
                seen_steps=set()
                for i in range(anchors[-1].width):
                    offset=i*anchors[-1].point_step
                    k=struct.unpack_from('<i',anchors[-1].data,offset+offsets['original_timestep'])[0];seen_steps.add(k)
                    assert k in [0,1]
                    for j in range(7):
                        value=struct.unpack_from('<f',anchors[-1].data,offset+offsets['q'+str(j)])[0]
                        assert abs(value-.002*k*(j+1))<1e-8,'raw q ordering changed'
                assert seen_steps=={0,1}
                assert all('geometry_backend=primitive' in summaries[k][-1].data for k in summaries)
            else:assert len(solid)==90 and all(m.type==Marker.SPHERE for m in solid)
            report['checks'].append(backend+': six live nodes; known map, exact markers, anchors, no missing YAML read')
            mode[0]='unknown';send()
            wait(lambda:all('has_violation=1' in summaries[k][-1].data for k in ['legacy','layers']),'unknown geometry candidates '+backend)
            report['checks'].append(backend+': both legacy selectors detect unknown obligations')
            if backend=='primitive':
                for bad in ['bad_size','nan','outside']:
                    mode[0]=bad;send()
                    wait(lambda:all('success=0' in summaries[k][-1].data for k in ['legacy','layers']),'invalid/outside legacy no-safe '+bad)
                    if bad!='outside':
                        wait(lambda:'success=0' in summaries['risk'][-1].data and 'invalid_confidence' in summaries['obs'][-1].data,'bad query diagnostics '+bad)
                report['checks'].append('primitive: malformed/nonfinite confidence and outside-map reject; no fresh SAFE')
                mode[0]='free';traj.points[0].positions[0]=float('nan');send()
                wait(lambda:'success=0' in summaries['risk'][-1].data and 'status=nonfinite_prediction' in summaries['audit'][-1].data,'nonfinite q rejection')
                report['checks'].append('primitive: nonfinite q rejects main risk and predicted auditor')
            for p in procs:stop(p)
            for sub in subs:sub.unregister()
            pub.unregister();target_pub.unregister()
        # Startup failures must not silently load spheres; no subscriptions or
        # untrusted geometry are allowed to become a fresh diagnostic result.
        for binary,ns in [('trajectory_risk_node','trajectory_risk'),('predicted_vbc_auditor_node','predicted_vbc'),
                ('trajectory_vbc_selector_legacy_node','trajectory_vbc'),('trajectory_vbc_selector_node','trajectory_vbc'),
                ('body_observability_node','body_observability'),('body_sample_model_node','body_sample_model')]:
            for fault in ['backend','mesh']:
                name='diagnostic_bad_'+binary+'_'+fault;params=copy.deepcopy(cfg)
                params.setdefault(ns,{}).update(geometry_backend='typo' if fault=='backend' else 'primitive',
                    robot_urdf_file=robot,primitive_urdf_file=robot,body_samples_file=str(out/'MISSING.yaml'))
                rospy.set_param('/'+name,params)
                p=start([str(ROOT/'devel/lib/care_confidence_map'/binary),'__name:='+name],name+'.log')
                wait(lambda:p.poll() is not None,'reject '+name)
                assert p.returncode!=0,'invalid backend/mesh initialized'
        report['checks'].append('six nodes reject unknown backend and mesh URDF without YAML fallback')
        # A missing TF must invalidate current observability, never show partial
        # green body geometry. Other tests use real complete TF transformations.
        rospy.set_param('/diagnostic_missing_tf',{'body_observability':dict(geometry_backend='primitive',
            primitive_urdf_file=primitive,body_samples_file=str(out/'MISSING.yaml'),map_frame='unavailable_frame',
            confidence_query_service='/diagnostic_test/query')})
        invalid=[]
        sub=rospy.Subscriber('/missing_tf/summary',String,lambda m:invalid.append(m.data))
        p=start([str(ROOT/'devel/lib/care_confidence_map/body_observability_node'),'__name:=diagnostic_missing_tf',
            '/care_planner/body_observability/summary:=/missing_tf/summary'],'missing_tf.log')
        wait(lambda:invalid and 'invalid_geometry' in invalid[-1],'missing TF fail-closed')
        stop(p);sub.unregister();report['checks'].append('body observability missing TF invalidates latched diagnostic')
        # Parse routes, do not launch planner/Gazebo. All actual ROS above is diagnostic-only.
        launch=str(ROOT/'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch')
        c=roslaunch.config.load_config_default([(launch,['diagnostic_geometry_backend:=primitive','cdf_selector_enabled:=true'])],None,verbose=False)
        for node_ns,param_ns in [('trajectory_risk_node','trajectory_risk'),('predicted_vbc_auditor_node','predicted_vbc')]:
            assert c.params['/'+node_ns+'/'+param_ns+'/geometry_backend'].value=='primitive'
        assert c.params['/c5_3a_cpp_forbidden_voxel_gpu_shadow/geometry_backend'].value=='primitive'
        viz=str(ROOT/'src/care_confidence_map/launch/body_samples_viz.launch')
        c=roslaunch.config.load_config_default([(viz,['geometry_backend:=primitive'])],None,verbose=False)
        assert [n.type for n in c.nodes]==['body_sample_model_node']
        report['checks'].append('parent + old visualization launch route to primitive')
        assert not any('/command' in t or 'committed_trajectory' in t for t,_ in rospy.get_published_topics())
        assert digest()==report['sha256'],'source/binary changed during run'
        report['status']='PASS'
    except Exception as e:
        report['error']=repr(e);raise
    finally:
        for p in reversed(children):stop(p)
        for h in handles:h.close()
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
