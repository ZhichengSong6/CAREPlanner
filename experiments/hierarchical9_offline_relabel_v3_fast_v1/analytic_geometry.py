"""Exact analytic FOV plane margins/Jacobian for one fixed (x,sensor).

Matches RepoOracle._planes semantics but differentiates rigid-body kinematics analytically,
avoiding six torch.autograd reverse passes per geometry evaluation.
"""
from __future__ import annotations
import math
import numpy as np
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
V1=HERE.parent/"hierarchical9_offline_relabel_v1"
if str(V1) not in sys.path:sys.path.insert(0,str(V1))
from repo_oracle import FOV

def _revolute(axis,q):
    a=np.asarray(axis,float);x,y,z=a;c=math.cos(q);s=math.sin(q);C=1-c
    return np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],
                     [y*x*C+z*s,c+y*y*C,y*z*C-x*s],
                     [z*x*C-y*s,z*y*C+x*s,c+z*z*C]],dtype=np.float64)

class AnalyticGeometry:
    def __init__(self,oracle,x,sensor):
        self.x=np.asarray(x,np.float64).reshape(3);self.s=int(sensor)
        self.specs=[]
        for spec in oracle.double_specs[self.s]:
            self.specs.append(dict(type=spec["type"],q_index=int(spec["q_index"]),
                origin=spec["origin"].detach().cpu().numpy().astype(np.float64),
                axis=spec["axis"].detach().cpu().numpy().astype(np.float64)))
        ax=math.tan(math.radians(FOV["horizontal_fov_deg"])/2)
        ay=math.tan(math.radians(FOV["vertical_fov_deg"])/2)
        nx=math.sqrt(1+ax*ax);ny=math.sqrt(1+ay*ay)
        self.plane_grad_p=np.array([[1/nx,0,ax/nx],[-1/nx,0,ax/nx],
                                    [0,1/ny,ay/ny],[0,-1/ny,ay/ny],
                                    [0,0,1],[0,0,-1]],dtype=np.float64)
        self.ax=ax;self.ay=ay;self.nx=nx;self.ny=ny
    def __call__(self,q):
        q=np.asarray(q,np.float64).reshape(7);T=np.eye(4,dtype=np.float64);joints=[]
        for spec in self.specs:
            T=T@spec["origin"];idx=spec["q_index"];kind=spec["type"]
            if idx<0 or kind=="fixed":continue
            axis_world=T[:3,:3]@spec["axis"];origin_world=T[:3,3].copy()
            joints.append((idx,kind,axis_world,origin_world))
            M=np.eye(4,dtype=np.float64)
            if kind in ("revolute","continuous"):
                M[:3,:3]=_revolute(spec["axis"],float(q[idx]))
            elif kind=="prismatic":
                M[:3,3]=spec["axis"]*float(q[idx])
            else:raise ValueError(kind)
            T=T@M
        R=T[:3,:3];t=T[:3,3];p=R.T@(self.x-t);px,py,pz=p
        h=np.array([(px+self.ax*pz)/self.nx,(-px+self.ax*pz)/self.nx,
                    (py+self.ay*pz)/self.ny,(-py+self.ay*pz)/self.ny,
                    pz-FOV["z_min"],FOV["z_max"]-pz],dtype=np.float64)-FOV["delta"]
        dp=np.zeros((3,7),dtype=np.float64)
        for idx,kind,axis,origin in joints:
            if kind in ("revolute","continuous"):
                dp[:,idx]+= -R.T@np.cross(axis,self.x-origin)
            else:
                dp[:,idx]+= -R.T@axis
        jac=self.plane_grad_p@dp
        return h,jac
