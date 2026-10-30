#!/usr/bin/env python3
"""Conditional sensor-memory deadlines and exact ideal-FOV diagnostics.

No observation, free cell, physical bound, or motion authorization is created.
Trace transforms belong to the map update, rather than a later measured pose.
"""
import math
import numpy as np


def sensor_expiry_s(last_seen_s, temporal_decay_time, threshold=.8):
    if (not all(math.isfinite(x) for x in (last_seen_s, temporal_decay_time, threshold)) or
            last_seen_s < 0 or temporal_decay_time <= 0 or not 0 < threshold <= 1):
        raise ValueError('invalid_sensor_memory_contract')
    return last_seen_s-temporal_decay_time*math.log(threshold)


def observation_window(evidence, required_until_s, reobserve_lead_s=1.):
    """Expose a planning deadline; frozen bootstrap remains a distinct layer.

    A reobservation request does not authorize moving to an observation pose.
    The complete stopping check must still cover that motion and its stop.
    """
    keys=('provenance','last_seen_s','confidence_at_horizon','model_free_at_horizon')
    if any(k not in evidence for k in keys) or 'temporal_decay_time' not in evidence:
        return dict(available=False,reason='missing_observation_metadata',runtime_admission=False)
    if (not math.isfinite(required_until_s) or not math.isfinite(reobserve_lead_s) or reobserve_lead_s<0 or
            len({len(evidence[k]) for k in keys})!=1):
        raise ValueError('invalid_observation_window')
    deadlines=[];unknown=0;bootstrap=0
    now=evidence['snapshot_ns']*1e-9
    for p,last,c,free in zip(*(evidence[k] for k in keys)):
        if p not in (1,2,3) or not math.isfinite(c) or c<.8 or free!=1:unknown+=1
        if p in (2,3):bootstrap+=1
        elif p==1:
            end=sensor_expiry_s(last,evidence['temporal_decay_time'])
            if last>now:raise ValueError('future_sensor_observation')
            deadlines.append(end)
    earliest=min(deadlines) if deadlines else None
    slack=earliest-required_until_s if earliest is not None else None
    action=('stop_missing_evidence' if unknown else
            'request_reobservation' if slack is not None and slack<=reobserve_lead_s else
            'within_conditional_observation_window')
    return dict(available=True,earliest_sensor_only_expiry_s=earliest,
        remaining_after_required_stop_s=slack,reobserve_lead_s=reobserve_lead_s,
        sensor_only_points=len(deadlines),frozen_bootstrap_points=bootstrap,
        uncovered_points=unknown,action=action,runtime_admission=False,
        future_environment_qualified=False)


def basis(axis):
    names={'x':(0,1),'y':(1,1),'z':(2,1),'-x':(0,-1),'-y':(1,-1),'-z':(2,-1)}
    if axis not in names:raise ValueError('unsupported_sensor_forward_axis')
    i,sign=names[axis];forward=np.eye(3)[i]*sign
    right=np.eye(3)[1 if i==0 else 0]
    up=np.eye(3)[1 if i==2 else 2]
    return forward,right,up


def local_samples(model):
    lo,hi,step=(model[k] for k in ('min_range','max_range','local_resolution'))
    if not all(math.isfinite(x) for x in (lo,hi,step)) or lo<0 or hi<=lo or step<=0:
        raise ValueError('invalid_sensor_sampling_model')
    forward,right,up=basis(model['forward_axis'])
    th=math.tan(math.radians(model['horizontal_fov_deg']/2))
    tv=math.tan(math.radians(model['vertical_fov_deg']/2));values=[]
    for i in range(math.floor((hi-lo)/step)+1):
        depth=lo+i*step
        if not lo<=depth<=hi:continue
        hmax,vmax=depth*th,depth*tv
        for ih in range(math.floor(2*hmax/step)+1):
            for iv in range(math.floor(2*vmax/step)+1):
                values.append(forward*depth+right*(-hmax+ih*step)+up*(-vmax+iv*step))
        values.append(forward*depth)
    return np.array(values)


def rotation_xyzw(quaternion):
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(float(q@q)-1)>1e-6:
        raise ValueError('invalid_sensor_trace_quaternion')
    x,y,z,w=map(float,q)
    # Follow tf2::Matrix3x3::setRotation, including its normalization.
    scale=2./(x*x+y*y+z*z+w*w)
    xs,ys,zs=x*scale,y*scale,z*scale
    wx,wy,wz=w*xs,w*ys,w*zs
    xx,xy,xz=x*xs,x*ys,x*zs
    yy,yz,zz=y*ys,y*zs,z*zs
    return np.array([[1-(yy+zz),xy-wz,xz+wy],
                     [xy+wz,1-(xx+zz),yz-wx],
                     [xz-wy,yz+wx,1-(xx+yy)]])


def raster_indices(model, transform, samples=None):
    samples=local_samples(model) if samples is None else samples
    rotation=rotation_xyzw(transform['quaternion_xyzw'])
    coordinates=samples@rotation.T+np.asarray(transform['translation'])
    grid=(coordinates-np.asarray(model['map_min']))/model['resolution']
    # C++ llround rounds halfway away from zero, unlike numpy.rint.
    grid=(np.sign(grid)*np.floor(np.abs(grid)+.5)).astype(np.int64)
    dims=np.asarray(model['grid_size'])
    grid=grid[np.all((grid>=0)&(grid<dims),axis=1)]
    return np.unique(grid[:,0]*dims[1]*dims[2]+grid[:,1]*dims[2]+grid[:,2])


def point_coverage(points, model, transform):
    local=(np.asarray(points)-np.asarray(transform['translation']))@rotation_xyzw(transform['quaternion_xyzw'])
    forward,right,up=basis(model['forward_axis']);depth=local@forward
    angle=(depth>0)&(np.abs(local@right)<=depth*math.tan(math.radians(model['horizontal_fov_deg']/2)))
    angle&=np.abs(local@up)<=depth*math.tan(math.radians(model['vertical_fov_deg']/2))
    return dict(visible=angle&(depth>=model['min_range'])&(depth<=model['max_range']),
        too_near=angle&(depth<model['min_range']),too_far=angle&(depth>model['max_range']))


def matched_update(evidence, model, updates):
    key=(evidence['epoch'],evidence['revision'])
    update=updates.get(key)
    if (evidence['instance']!=model['instance'] or update is None or
            update['instance']!=evidence['instance'] or
            update['map_update_ns']!=evidence['map_update_ns'] or
            update['observation_ns']>evidence['snapshot_ns']):
        raise ValueError('no_matching_map_update_trace')
    return update
