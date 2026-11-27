"""Sensor-evidence interval derived from a finite ToF return.

No-return rays are ambiguous in the existing fusion product: an invalid raw
depth is replaced with a maximum-range endpoint. Such endpoints cannot establish
FREE space in this adapter. Near clipping is axial optical depth, not radius.
"""
import numpy as np

def array(msg):
    types={1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',8:'<f8'}
    dtype=np.dtype(dict(names=[f.name for f in msg.fields],formats=[types[f.datatype] for f in msg.fields],offsets=[f.offset for f in msg.fields],itemsize=msg.point_step))
    assert not msg.is_bigendian and msg.row_step==msg.width*msg.point_step
    return np.frombuffer(msg.data,dtype=dtype,count=msg.width*msg.height)

def intervals(origin, endpoint, sensor_ids, hit, forward_axes, near=.15, far=.75, guard=.025):
    origin=np.asarray(origin,dtype=float);endpoint=np.asarray(endpoint,dtype=float)
    delta=endpoint-origin;length=np.linalg.norm(delta,axis=1)
    axes=np.asarray(forward_axes)[np.clip(sensor_ids,0,len(forward_axes)-1)]
    depth=np.einsum('ij,ij->i',delta,axes)
    valid=(hit!=0)&(sensor_ids>=0)&(sensor_ids<len(forward_axes))
    valid &= np.isfinite(origin).all(axis=1)&np.isfinite(endpoint).all(axis=1)
    valid &= (depth>=near)&(depth<=far)&(length>0)
    start=np.full(len(length),np.inf)
    start[valid]=near*length[valid]/depth[valid]
    end=length-guard
    return valid, start, end

def validate():
    # Axial near plane matters for diagonal rays; finite hits preserve evidence
    # beyond that plane, while invalid/no-return samples establish none.
    axes=np.array([[0.,0.,1.]])
    o=np.zeros((4,3));e=np.array([[0,0,.5],[.3,0,.4],[0,0,.75],[0,0,.1]])
    valid,start,end=intervals(o,e,np.zeros(4,dtype=int),np.array([1,1,0,1]),axes)
    assert valid.tolist()==[True,True,False,False]
    assert np.allclose(start[:2],[.15,.1875]) and np.allclose(end[:2],[.475,.475])
    assert start[1]*.4/.5>=.15-1e-12
    return True
