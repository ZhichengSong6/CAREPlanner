#!/usr/bin/env python3
"""Actual map-node age/order/reobservation controls with recorded robot poses.

TF/clock replay only; no actuator or lease publisher. Replayed observation poses
are diagnostic witnesses, not authorized recovery trajectories.
"""
import argparse,collections,copy,gzip,hashlib,json,math,os,socket,subprocess,sys,threading,time,traceback,xmlrpc.client
from pathlib import Path
sys.path.append('/usr/lib/python3/dist-packages')
import numpy as np
import pinocchio as pin
import yaml
from phase_e_owned_processes import identity,remember_descendants,reap_owned
from phase_e_observation_evidence import raster_indices,point_coverage,observation_window
from verify_phase_e_leased_response_gazebo import protected

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/phase_e_adaptive_ledger_20261019/gazebo_adaptive'
JOINTS=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']


class RobotFrames:
    def __init__(self,config):
        self.model=pin.buildModelFromUrdf(str(ROOT/'src/arm_description/urdf/Arm.urdf'))
        self.data=self.model.createData()
        self.indices=[self.model.joints[self.model.getJointId(j)].idx_q for j in JOINTS]
        self.sensors=[s['frame'] for s in config['sensor_model']['sensors']]
        self.links=['link1','link2','link3','link4','wrist_link1','wrist_link2','wrist_link3']

    def transforms(self,q):
        full=np.zeros(self.model.nq);full[self.indices]=q
        pin.forwardKinematics(self.model,self.data,full);pin.updateFramePlacements(self.model,self.data)
        base=self.data.oMf[self.model.getFrameId('base_link')]
        values=[]
        for frame in self.links+self.sensors:
            pose=base.inverse()*self.data.oMf[self.model.getFrameId(frame)]
            values.append(dict(frame=frame,translation=pose.translation.tolist(),
                quaternion_xyzw=pin.Quaternion(pose.rotation).coeffs().tolist()))
        return values


def select_witness(config,robot):
    model=dict(min_range=config['sensor_model']['tof_min_range'],max_range=config['sensor_model']['tof_max_range'],
        local_resolution=config['sensor_model']['local_resolution'],forward_axis=config['sensor_model']['forward_axis'],
        horizontal_fov_deg=config['sensor_model']['horizontal_fov_deg'],vertical_fov_deg=config['sensor_model']['vertical_fov_deg'])
    candidates=[];watch=set()
    for line in gzip.open(SOURCE/'shadow/point_evidence.jsonl.gz','rt'):
        r=json.loads(line);e=r['evidence'];refs=set(r['reference_pose_grid_indices'])
        bad=[j for j,g in enumerate(r['grid_indices']) if g in refs and e['confidence_at_horizon'][j]<.8]
        watch.update(r['grid_indices'][j] for j in bad)
        if bad:candidates.append((len(bad),r,bad))
    states=[json.loads(s) for s in (SOURCE/'controller_trace.jsonl').open()]
    states=[s for s in states if s['kind']=='state']
    for _,r,bad in sorted(candidates,key=lambda v:-v[0]):
        e=r['evidence'];model.update({k:e[k] for k in ['map_min','map_max','grid_size','resolution']})
        for j in bad:
            if e['provenance'][j]!=1:continue
            gid=r['grid_indices'][j];p=r['points'][j]
            problem=[t for t in robot.transforms(r['q']) if t['frame'] in robot.sensors]
            if any(gid in raster_indices(model,t) for t in problem):continue
            near=sorted(states,key=lambda s:abs(s['t_ns']-e['last_seen_s'][j]*1e9))[:12]
            for s in near:
                witness=[t for t in robot.transforms(s['q']) if t['frame'] in robot.sensors]
                seeing=[t['frame'] for t in witness if gid in raster_indices(model,t)]
                if seeing:
                    return dict(grid_index=gid,point=p,problem_q=r['q'],witness_q=s['q'],
                        witness_source_ns=s['t_ns'],historical_last_seen_s=e['last_seen_s'][j],
                        seeing_sensors=seeing,watch_grid_indices=sorted(watch),model=model,
                        source_record_sha256=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest())
    raise AssertionError('No archived sensor-only observation witness found')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--map-binary',type=Path,required=True)
    ap.add_argument('--port',type=int,default=12130)
    ap.add_argument('--only-tf-validity-controls',action='store_true')
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=args.map_binary.resolve();before=protected()
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',args.port))!=0
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml').read_text())
    robot=RobotFrames(config);selection=select_witness(config,robot)
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    result=dict(status='RUNNING',scope='Actual ideal_fov map with controlled TF/clock replay; no actuator commands',
        runtime_admission=False,new_formal_windows=0,conditions=[],checks=[],
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [binary,Path(__file__).resolve(),
            ROOT/'scripts/phase_e_observation_evidence.py',ROOT/'src/arm_description/urdf/Arm.urdf',
            ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml',
            SOURCE/'shadow/point_evidence.jsonl.gz',SOURCE/'controller_trace.jsonl']})
    children=[];handles=[];owned={};done=threading.Event();thread=None
    def start(cmd,label):
        f=(out/(label+'.log')).open('x');handles.append(f)
        p=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);owned[p.pid]=identity(p.pid);return p
    def wait(fn,label,timeout=20.):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            remember_descendants({p.pid for p in children if p.poll() is None},owned)
            if fn():return
            time.sleep(.01)
        raise AssertionError('timeout: '+label)
    def check(label,condition):
        assert condition,label
        result['checks'].append(label)
    try:
        start(['roscore','-p',args.port],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/observation_memory')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy,tf2_ros
        from geometry_msgs.msg import Point,TransformStamped
        from rosgraph_msgs.msg import Clock
        from std_srvs.srv import Trigger
        from care_confidence_map.srv import QueryConfidenceSnapshot,QueryConfidenceSnapshotRequest
        from phase_e_stop_shadow_node import snapshot_dict
        rospy.set_param('/use_sim_time',True);rospy.init_node('observation_memory',disable_signals=True)
        clock=rospy.Publisher('/clock',Clock,queue_size=1,latch=True);broadcaster=tf2_ros.TransformBroadcaster()
        sim=[10000000000]
        def publish_clock():
            while not done.is_set():
                ns=sim[0];clock.publish(Clock(clock=rospy.Time(ns//10**9,ns%10**9)));time.sleep(.01)
        thread=threading.Thread(target=publish_clock,daemon=True);thread.start()
        wait(lambda:rospy.Time.now().to_nsec()==sim[0],'clock')
        def frames(q,ns):
            msgs=[]
            for t in robot.transforms(q):
                m=TransformStamped();m.header.frame_id='base_link';m.child_frame_id=t['frame']
                m.header.stamp=rospy.Time(ns//10**9,ns%10**9)
                m.transform.translation.x,m.transform.translation.y,m.transform.translation.z=t['translation']
                m.transform.rotation.x,m.transform.rotation.y,m.transform.rotation.z,m.transform.rotation.w=t['quaternion_xyzw']
                msgs.append(m)
            return msgs
        query=None
        def tick(q,target=None):
            target=sim[0]+50000000 if target is None else target
            assert target>sim[0]
            msgs=frames(q,target)
            for _ in range(6):broadcaster.sendTransform(msgs);time.sleep(.01)
            sim[0]=target
            wait(lambda:rospy.Time.now().to_nsec()==target,'advanced clock')
            if query:
                wait(lambda:query(QueryConfidenceSnapshotRequest()).map_update_stamp.to_nsec()==target,'map update')
            return target
        def snapshot(horizon=.4):
            req=QueryConfidenceSnapshotRequest(points=[Point(*selection['point'])],horizon_s=horizon,minimum_confidence=.8)
            return snapshot_dict(query(req))
        orders=[('forward',[10.,21.5,22.,22.5,30.,60.]),('reverse',[60.,30.,22.5,22.,21.5,10.])]
        if args.only_tf_validity_controls:orders=[]
        condition=0
        for order,ages in orders:
            for age in ages:
                label='%02d_%s_age_%s'%(condition,order,age);condition+=1
                tick([0.]*7)
                namespace='/memory_'+str(condition)
                current=copy.deepcopy(config)
                current['confidence_map'].update(query_service=namespace+'/query',
                    observation_trace_file=str(out/(label+'_observations.jsonl')),
                    observation_trace_watch_grid_indices=selection['watch_grid_indices'])
                current['current_body_prior'].update(geometry_backend='primitive',
                    primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                    refresh_service=namespace+'/refresh',deactivate_service=namespace+'/deactivate')
                rospy.set_param(namespace,current)
                proc=start([binary,'__name:='+namespace[1:]],label)
                rospy.wait_for_service(namespace+'/query_snapshot',timeout=20.)
                query=rospy.ServiceProxy(namespace+'/query_snapshot',QueryConfidenceSnapshot)
                tick([0.]*7)
                refresh=rospy.ServiceProxy(namespace+'/refresh',Trigger)()
                check(label+' initial fixed bootstrap',refresh.success and 'skipped_samples=0' in refresh.message)
                tick(selection['witness_q']);observed=snapshot()
                check(label+' actual sensor-only witness',observed['provenance']==[1] and observed['confidence_now']==[1.])
                last_seen=observed['last_seen_s'][0]
                tick(selection['problem_q'],int(round((last_seen+age)*1e9)))
                now=snapshot(0.);future=snapshot(.4)
                expected=math.exp(-age/100.)
                check(label+' confidence matches actual age',abs(now['confidence_now'][0]-expected)<1e-6)
                check(label+' source is retained without reobservation',now['last_seen_s']==[last_seen] and now['provenance']==[1])
                record=dict(label=label,order=order,age_s=age,now=now,future=future,
                    observation_window=observation_window(future,future['snapshot_ns']*1e-9+.4))
                result['conditions'].append(record)
                # Replay a genuine sensor observation, rather than editing a
                # confidence value or refreshing the locked body prior.
                tick(selection['witness_q']);renewed=snapshot()
                check(label+' reobservation restores evidence',renewed['provenance']==[1] and renewed['confidence_now']==[1.] and renewed['model_free_at_horizon']==[1])
                record['reobserved']=renewed
                proc.send_signal(2);proc.wait(timeout=10.)
                query=None
        if orders:
            pairs=collections.defaultdict(list)
            for r in result['conditions']:pairs[r['age_s']].append(r)
            check('age-matched results independent of forward/reverse run order',all(
                a['now']['confidence_now']==b['now']['confidence_now'] and
                a['future']['model_free_at_horizon']==b['future']['model_free_at_horizon']
                for a,b in pairs.values()))
            early=next(r for r in result['conditions'] if r['age_s']==21.5)
            check('policy requests observation while stopping evidence still valid',
                early['future']['model_free_at_horizon']==[1] and early['observation_window']['action']=='request_reobservation')
        # A cached pose is not a new observation. This control reproduces the
        # pre-fix false refresh even though the queried point remains in FOV.
        tick([0.]*7);namespace='/memory_tf_validity';current=copy.deepcopy(config)
        current['confidence_map'].update(query_service=namespace+'/query',
            observation_trace_file=str(out/'tf_validity_observations.jsonl'),
            observation_trace_watch_grid_indices=selection['watch_grid_indices'])
        current['current_body_prior'].update(geometry_backend='primitive',
            primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
            refresh_service=namespace+'/refresh',deactivate_service=namespace+'/deactivate')
        rospy.set_param(namespace,current);proc=start([binary,'__name:=memory_tf_validity'],'tf_validity')
        rospy.wait_for_service(namespace+'/query_snapshot',timeout=20.)
        query=rospy.ServiceProxy(namespace+'/query_snapshot',QueryConfidenceSnapshot)
        tick([0.]*7)
        check('TF control initial bootstrap',rospy.ServiceProxy(namespace+'/refresh',Trigger)().success)
        tick(selection['witness_q']);initial=snapshot();last_seen=initial['last_seen_s'][0]
        check('TF control actual sensor witness',initial['provenance']==[1] and initial['confidence_now']==[1.])
        def clock_without_tf(value):
            sim[0]=int(round(value*1e9))
            wait(lambda:rospy.Time.now().to_nsec()==sim[0],'clock without TF')
            wait(lambda:query(QueryConfidenceSnapshotRequest()).map_update_stamp.to_nsec()==sim[0],'update without new TF')
        clock_without_tf(last_seen+.2);stale=snapshot()
        result['tf_validity']=dict(initial=initial,stale_tf=stale)
        check('cached TF cannot refresh last_seen',stale['last_seen_s']==[last_seen])
        clock_without_tf(last_seen+30.);expired=snapshot()
        result['tf_validity']['expired_tf']=expired
        check('cached TF evidence expires normally',expired['last_seen_s']==[last_seen] and expired['confidence_now'][0]<.8 and expired['model_free_at_horizon']==[0])
        future_ns=sim[0]+1000000000
        for _ in range(6):broadcaster.sendTransform(frames(selection['witness_q'],future_ns));time.sleep(.01)
        clock_without_tf(sim[0]*1e-9+.1);future=snapshot()
        result['tf_validity']['future_tf']=future
        check('future TF cannot create current observation evidence',future['last_seen_s']==[last_seen] and future['model_free_at_horizon']==[0])
        tick(selection['witness_q'],future_ns);recovered=snapshot()
        result['tf_validity']['recovered_tf']=recovered
        check('fresh valid TF resumes genuine observations',recovered['last_seen_s'][0]>last_seen and recovered['model_free_at_horizon']==[1])
        proc.send_signal(2);proc.wait(timeout=10.);query=None
        result['status']='PASS'
    except BaseException:
        result['status']='FAIL';result['error']=traceback.format_exc()
    finally:
        done.set()
        if thread:thread.join(2.)
        remember_descendants({p.pid for p in children if p.poll() is None},owned)
        result['cleanup']=reap_owned(owned)
        for p in children:
            try:p.wait(timeout=3.)
            except subprocess.TimeoutExpired:pass
        for f in handles:f.close()
        result['protected_identity_equal']=protected()==before
        result['input_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in result['input_sha256'].items())
        with socket.socket() as sock:result['port_released']=sock.connect_ex(('127.0.0.1',args.port))!=0
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('conditions','input_sha256')},indent=2),flush=True)
    return 0 if result['status']=='PASS' and result['protected_identity_equal'] and result['input_unchanged'] and not result['cleanup']['remaining'] and result['port_released'] else 1

if __name__=='__main__':raise SystemExit(main())
