"""Evaluate live calibrated arm geometry for repeated route proposals in the numerical planner."""

import ctypes as C
from pathlib import Path

import numpy as np


class MeasuredArm:
    def __init__(self,body,grid,library=None):
        self.lib=C.CDLL(str(library or Path(__file__).with_name('measured_arm_geometry.so')))
        p=C.c_void_p
        i=C.c_int
        self.lib.measured_arm_create.argtypes=[i,p,p,p,p,p,i,p,p,p,p,p,p,C.c_double]
        self.lib.measured_arm_create.restype=p
        self.lib.measured_arm_destroy.argtypes=[p]
        self.lib.measured_arm_destroy.restype=None
        self.lib.measured_arm_cells.argtypes=[p,p,i,p,i,p]
        self.lib.measured_arm_cells.restype=i
        self.lib.measured_cloud_create.argtypes=[p,i]
        self.lib.measured_cloud_create.restype=p
        self.lib.measured_cloud_destroy.argtypes=[p]
        self.lib.measured_cloud_destroy.restype=None
        self.cloud=None
        self.cloud_source=None
        self.lib.measured_arm_surface_gaps.argtypes=[p,p,p,i,p,C.c_double,p]
        self.lib.measured_arm_surface_gaps.restype=None
        self.shape_distance=getattr(self.lib,'measured_arm_shape_gaps',None)
        if self.shape_distance is not None:
            self.shape_distance.argtypes=self.lib.measured_arm_surface_gaps.argtypes
            self.shape_distance.restype=None
        kin=body.task_model.kinematics
        names={'base_link':0}
        parents=[-1]; joints=[-1]; origins=[np.eye(4)]; outers=[np.zeros((3,3))]; skews=[outers[0]]
        def frame(name):
            if name not in names:
                parent,origin,kind,_,index=kin.frames[name]
                parent_id=frame(parent)
                names[name]=len(names)
                parents.append(parent_id)
                moving=kind in ('revolute','continuous')
                joints.append(index if moving else -1)
                outer,skew=kin.rotations[name] if moving else (outers[0],skews[0])
                origins.append(origin);outers.append(outer);skews.append(skew)
            return names[name]
        links=[frame(p['link']) for p in body.primitives]
        kinds=[]; halves=[]; offsets=[]
        for primitive in body.primitives:
            typ=primitive['type'];g=primitive['geometry']
            kinds.append({'box':0,'cylinder':1,'sphere':2}[typ])
            halves.append(np.asarray(g.size)*.5 if typ=='box' else
                          [g.radius,g.radius,.5*g.length] if typ=='cylinder' else [g.radius]*3)
            offsets.append(primitive['origin'])
        pitch=.05
        shape=np.rint((grid[-1]-grid[0])/pitch).astype(int)+1
        if np.prod(shape)!=len(grid):
            raise ValueError('Expected the calibrated regular workspace grid')
        pack=lambda arr,dtype:np.ascontiguousarray(arr,dtype=dtype)
        arrays=[pack(v,np.int32 if idx in (0,1,5,6,10) else np.float64)
                for idx,v in enumerate((parents,joints,origins,outers,skews,links,kinds,offsets,halves,grid[0],shape))]
        args=[v.ctypes.data for v in arrays]
        self.handle=self.lib.measured_arm_create(len(names),*args[:5],len(links),*args[5:],pitch)
        self.cells_per_pose=140

    def __del__(self):
        if getattr(self,'cloud',None):
            self.lib.measured_cloud_destroy(self.cloud)
        if getattr(self,'handle',None):
            self.lib.measured_arm_destroy(self.handle)

    def membership(self,poses):
        poses=np.ascontiguousarray(poses,dtype=np.float64).reshape(-1,7)
        offsets=np.empty(len(poses)+1,dtype=np.int32)
        ids=np.empty(len(poses)*self.cells_per_pose,dtype=np.int32)
        total=self.lib.measured_arm_cells(self.handle,poses.ctypes.data,len(poses),ids.ctypes.data,len(ids),offsets.ctypes.data)
        if total>len(ids):
            self.cells_per_pose=max(self.cells_per_pose,int(np.max(np.diff(offsets)))+1)
            ids=np.empty(total,dtype=np.int32)
            self.lib.measured_arm_cells(self.handle,poses.ctypes.data,len(poses),ids.ctypes.data,len(ids),offsets.ctypes.data)
        return [ids[a:b] for a,b in zip(offsets[:-1],offsets[1:])]

    def surface_gaps(self,root,poses,points,*,exact=False):
        q=np.ascontiguousarray(poses,dtype=np.float64).reshape(-1,7)
        start=np.ascontiguousarray(root,dtype=np.float64).reshape(7)
        if self.cloud_source is not points:
            cloud=np.ascontiguousarray(points,dtype=np.float64).reshape(-1,3)
            if self.cloud:
                self.lib.measured_cloud_destroy(self.cloud)
            self.cloud=self.lib.measured_cloud_create(cloud.ctypes.data,len(cloud))
            self.cloud_source=points
        gaps=np.empty(len(q),dtype=np.float64)
        distance=self.shape_distance if exact and self.shape_distance is not None else self.lib.measured_arm_surface_gaps
        distance(self.handle,start.ctypes.data,q.ctypes.data,len(q),self.cloud,.09,gaps.ctypes.data)
        return gaps


def calibrated(body,grid,library=None):
    arm=getattr(body,'measured_arm_numeric',None)
    if arm is None:
        arm=MeasuredArm(body,grid,library)
        body.measured_arm_numeric=arm
    return arm
