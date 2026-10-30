"""Bind a full tracker curve to conservative occupied-volume exclusion."""
from pathlib import Path
import ctypes as ct,hashlib,json,math,time
import xml.etree.ElementTree as ET
import numpy as np
NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']

def curvature(q,v,t,time_ns=None):
    from tracker_curve import curve_data
    return curve_data(q,v,t,time_ns)[0]

class MetricCurve:
    def __init__(self,runtime,library=None,mounted_base_contact=False):
        self.runtime=Path(runtime)
        self.library=ct.CDLL(str(library or self.runtime/'devel/lib/egocentric_arm_planner/libmetric_curve.so'))
        self.library.care_metric_create.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_char_p,ct.c_int]
        self.library.care_metric_create.restype=ct.c_void_p
        self.library.care_metric_destroy.argtypes=[ct.c_void_p]
        fn=self.library.care_metric_check;dp=ct.POINTER(ct.c_double)
        fn.argtypes=[ct.c_void_p,ct.c_int,dp,dp,ct.c_int,dp,ct.c_double,ct.c_double,ct.c_double,ct.c_int,ct.c_double,dp,ct.c_char_p,ct.c_int]
        fn.restype=ct.c_int;self.fn=fn
        robot=self.runtime/'src/arm_description/urdf/Arm.urdf'
        primitive=self.runtime/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
        error=ct.create_string_buffer(512)
        self.handle=self.library.care_metric_create(str(robot).encode(),str(primitive).encode(),error,len(error))
        if not self.handle:raise ValueError(error.value.decode())
        self.mounted_base_contact=bool(mounted_base_contact)
        self.library.care_metric_mount_contact.argtypes=[ct.c_void_p,ct.c_int]
        self.library.care_metric_mount_contact(self.handle,int(self.mounted_base_contact))
        self.geometry_sha256=hashlib.sha256(robot.read_bytes()+primitive.read_bytes()).hexdigest()
        self.body_labels=[dict(link=link.get('name'),collision_index=i)
            for link in sorted(ET.parse(primitive).getroot().findall('link'),key=lambda link:link.get('name'))
            for i,_ in enumerate(link.findall('collision'))]
    def check_arrays(self,q,v,t,points,resolution=.05,physical=.02,tracking=.02,budget_s=.03,max_fk=512,time_ns=None):
        started=time.monotonic();q=np.ascontiguousarray(q,dtype=np.float64)
        from tracker_curve import curve_data
        c,_,canonical_time_ns=curve_data(q,v,t,time_ns)
        points=np.ascontiguousarray(points,dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('invalid_metric_map_points')
        deadline=started+budget_s;remaining=deadline-time.monotonic()
        if remaining<=0:return dict(valid=False,reason='metric_python_budget',elapsed_s=time.monotonic()-started)
        result=np.zeros(7);error=ct.create_string_buffer(512);dp=ct.POINTER(ct.c_double)
        code=self.fn(self.handle,len(q),q.ctypes.data_as(dp),c.ctypes.data_as(dp),len(points),points.ctypes.data_as(dp),
            resolution,physical,tracking,max_fk,remaining,result.ctypes.data_as(dp),error,len(error))
        elapsed=time.monotonic()-started
        body_index=int(result[5])
        return dict(valid=code==0 and elapsed<=budget_s,code=code,
            reason='metric_python_wall_budget' if code==0 and elapsed>budget_s else error.value.decode(),elapsed_s=elapsed,
            native_s=float(result[0]),fk=int(result[1]),refined=int(result[2]),minimum_slack_m=float(result[3]),failure_point_index=int(result[4]),
            failure_body_index=int(result[5]),failure_enclosure_frame=int(result[6]),
            failure_body=self.body_labels[body_index] if 0<=body_index<len(self.body_labels) else None,
            physical_target_m=physical,assumed_tracking_tube_m=tracking,required_reference_volume_m=physical+tracking,
            geometry_sha256=self.geometry_sha256,body_policy='all collision primitives; structurally fixed bodies have zero tracking displacement',
            mounted_base_nonpenetration_only=self.mounted_base_contact,
            curve_time_identity='ROS integer nanoseconds',curve_sha256=hashlib.sha256(q.tobytes()+c.tobytes()+canonical_time_ns.tobytes()).hexdigest())
    def close(self):
        if self.handle:self.library.care_metric_destroy(self.handle);self.handle=None
