"""Candidate-bound occupied-volume audit using an immutable live map cloud."""
from pathlib import Path
import threading,time,json,hashlib
import numpy as np
import rospy
import yaml
from sensor_msgs.msg import PointCloud2
from metric_curve import MetricCurve,NAMES

class LiveMetricCurve:
    def __init__(self,runtime):
        mount_file=Path(runtime)/'src/egocentric_arm_planner/config/known_ground_geometry.json'
        mount=json.loads(mount_file.read_text())
        from ground_support_policy import load_ground_policy
        load_ground_policy(Path(runtime)/'src/arm_description/urdf/Arm.urdf',mount_file)
        self.mount_sha256=hashlib.sha256(mount_file.read_bytes()).hexdigest()
        self.model=MetricCurve(runtime,mounted_base_contact=False);self.lock=threading.Lock();self.latest=None
        cfg=yaml.safe_load((Path(runtime)/'src/care_confidence_map/config/confidence_map_static_box_ray.yaml').read_text())['confidence_map']
        self.resolution=float(cfg['resolution'])
        axes=[cfg[n+'_min']+self.resolution*np.arange(int(np.floor((cfg[n+'_max']-cfg[n+'_min'])/self.resolution))+1) for n in ['x','y','z']]
        self.grid=np.array(np.meshgrid(*axes,indexing='ij')).reshape(3,-1).T
        self.subscription=rospy.Subscriber('/care_planner/confidence_map/points',PointCloud2,self.receive,queue_size=1)
    def receive(self,msg):
        with self.lock:self.latest=msg
    def check(self,candidate,deadline=None):
        started=time.monotonic()
        with self.lock:msg=self.latest
        if msg is None:return dict(valid=False,reason='metric_map_missing')
        now=rospy.Time.now();age=(now-msg.header.stamp).to_sec()
        identity=dict(candidate_stamp_ns=str(candidate.header.stamp.to_nsec()),candidate_sequence=int(candidate.header.seq),
            map_stamp_ns=str(msg.header.stamp.to_nsec()),map_frame=msg.header.frame_id,map_age_s=age)
        def reject(reason):return dict(identity,valid=False,reason=reason,elapsed_s=time.monotonic()-started)
        if msg.header.frame_id!='base_link' or candidate.header.frame_id!='base_link':return reject('metric_frame_mismatch')
        if msg.header.stamp.is_zero() or not 0<=age<=.05:return reject('metric_map_stale_or_future')
        if len(candidate.joint_names)!=7 or set(candidate.joint_names)!=set(NAMES):return reject('metric_joint_identity')
        formats={1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',8:'<f8'}
        try:
            if msg.height!=1 or msg.is_bigendian:return reject('metric_map_layout')
            dtype=np.dtype(dict(names=[f.name for f in msg.fields],formats=[formats[f.datatype] for f in msg.fields],
                offsets=[f.offset for f in msg.fields],itemsize=msg.point_step))
            a=np.frombuffer(msg.data,dtype=dtype,count=msg.width)
            xyz=np.column_stack([a[n] for n in ['x','y','z']])
            if xyz.shape!=self.grid.shape or not np.allclose(xyz,self.grid,rtol=0.,atol=1e-6):
                return reject('metric_map_grid_incomplete_or_changed')
            if not np.isfinite(a['occupancy']).all():return reject('metric_map_nonfinite')
            occupied=a[a['occupancy']>.5];points=np.column_stack([occupied[n] for n in ['x','y','z']])
            order=[candidate.joint_names.index(n) for n in NAMES]
            q=np.array([[p.positions[i] for i in order] for p in candidate.points])
            v=np.array([[p.velocities[i] for i in order] for p in candidate.points])
            t=np.array([p.time_from_start.to_sec() for p in candidate.points])
            time_ns=[p.time_from_start.to_nsec() for p in candidate.points]
            remaining=min(.03-(time.monotonic()-started),deadline-time.monotonic() if deadline is not None else .03)
            if remaining<=0:return reject('metric_admission_deadline')
            result=self.model.check_arrays(q,v,t,points,resolution=self.resolution,budget_s=remaining,time_ns=time_ns)
            result.update(identity,occupied_points=len(points),scope='occupied supports only; existing UNKNOWN visibility checks remain mandatory')
            result['existing_mount_policy_sha256']=self.mount_sha256
            if (rospy.Time.now()-msg.header.stamp).to_sec()>.05:
                result.update(valid=False,reason='metric_map_expired_during_check')
            return result
        except (ValueError,KeyError,TypeError,IndexError,OverflowError) as exc:
            return reject('metric_malformed_'+type(exc).__name__)
