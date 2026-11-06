"""ctypes binding for the fixed, double-precision known-floor numerical kernel."""
import ctypes as ct
from pathlib import Path
import numpy as np

class NativeGround:
    def __init__(self,frames,bodies,rotations,plane,body_floors=None,common_margin=0.,normal=None,offset=None):
        names=['base_link'];lookup={'base_link':0}
        parents=[-1];kinds=[0];indices=[-1];origins=[np.eye(4)];axes=[np.zeros(3)];rot=[np.zeros(18)]
        def add(name):
            if name in lookup:return lookup[name]
            parent,origin,kind,axis,index=frames[name];pi=add(parent)
            if kind not in ('fixed','revolute','continuous','prismatic'):raise ValueError('unsupported_ground_joint')
            k=1 if kind in ('revolute','continuous') else (2 if kind=='prismatic' else 0)
            if k and not 0<=index<7:raise ValueError('unknown_ground_joint')
            lookup[name]=len(names);names.append(name);parents.append(pi);kinds.append(k);indices.append(index)
            origins.append(origin);axes.append(axis)
            rot.append(np.concatenate([v.ravel() for v in rotations[name]]) if k==1 else np.zeros(18))
            return lookup[name]
        body_frames=[];starts=[0];vertices=[];radii=[];self.body_names=[]
        for name,_,points,bound in bodies:
            body_frames.append(add(name));starts.append(starts[-1]+len(points));vertices.extend(points)
            radii.append(bound);self.body_names.append(name)
        if not bodies:raise ValueError('empty_ground_bodies')
        self.arrays=[np.ascontiguousarray(a,dtype=d) for a,d in
            ((parents,np.int32),(kinds,np.int32),(indices,np.int32),(origins,np.float64),(axes,np.float64),
             (rot,np.float64),(body_frames,np.int32),(starts,np.int32),(vertices,np.float64),(radii,np.float64))]
        for a in self.arrays:a.flags.writeable=False
        self.nf=len(names);self.nb=len(bodies);self.plane=float(plane);self.calls=0
        path=Path(__file__).resolve().parents[3]/'devel/lib/egocentric_arm_planner/libknown_ground.so'
        self.library=ct.CDLL(str(path))
        self.offsets=np.ascontiguousarray([common_margin-x for x in body_floors],dtype=np.float64) if body_floors is not None else np.zeros(len(bodies))
        self.offsets.flags.writeable=False
        self.normal=np.ascontiguousarray([0.,0.,1.] if normal is None else normal,dtype=np.float64)
        self.offset=-self.plane if offset is None else float(offset)
        fn=self.library.known_analytic_plane_clearance
        ip=ct.POINTER(ct.c_int32);dp=ct.POINTER(ct.c_double)
        fn.argtypes=[ct.c_int,ip,ip,ip,dp,dp,dp,ct.c_int,ip,ip,dp,dp,dp,dp,ct.c_double,dp,ip,dp,dp,ct.c_double]
        fn.restype=ct.c_int;self.fn=fn
        self.pointers=[a.ctypes.data_as(ip if a.dtype==np.int32 else dp) for a in self.arrays]

    def clearance(self,q,radius):
        q=np.ascontiguousarray(q,dtype=np.float64);radius=np.ascontiguousarray(radius,dtype=np.float64)
        if q.shape!=(7,) or radius.shape!=(7,):raise ValueError('invalid_ground_state_shape')
        dp=ct.POINTER(ct.c_double);height=ct.c_double();body=ct.c_int32()
        p=self.pointers
        rc=self.fn(self.nf,*p[:6],self.nb,*p[6:],q.ctypes.data_as(dp),radius.ctypes.data_as(dp),
                   self.plane,ct.byref(height),ct.byref(body),self.offsets.ctypes.data_as(dp),self.normal.ctypes.data_as(dp),self.offset)
        self.calls+=1
        if rc:raise ValueError('native_ground_error_'+str(rc))
        return height.value,self.body_names[body.value]
