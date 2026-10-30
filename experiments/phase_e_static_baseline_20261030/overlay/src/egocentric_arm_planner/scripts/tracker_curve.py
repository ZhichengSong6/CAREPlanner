"""Canonical actuator curve, using ROS integer duration arithmetic."""
import numpy as np

def curve_data(q,v,t=None,time_ns=None):
    q=np.asarray(q,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=7 or len(q)<2 or v.shape!=q.shape or not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('malformed_tracker_curve')
    if time_ns is None:
        t=np.asarray(t,dtype=np.float64)
        if t.shape!=(len(q),) or not np.isfinite(t).all() or np.max(np.abs(t))>1e6:raise ValueError('invalid_tracker_times')
        time_ns=np.rint(t*1e9).astype(np.int64)
        canonical=time_ns//1000000000+(time_ns%1000000000)*1e-9
        if np.max(np.abs(t-canonical))>1e-10:raise ValueError('non_ros_tracker_duration')
    else:
        raw=np.asarray(time_ns)
        if raw.dtype.kind not in 'iu' or raw.shape!=(len(q),):raise ValueError('invalid_tracker_time_identity')
        time_ns=np.asarray(raw,dtype=np.int64)
    if time_ns[0]!=0 or np.any(time_ns<0) or np.any(time_ns>10**15) or np.any(np.diff(time_ns)<=0):
        raise ValueError('invalid_tracker_duration_order')
    if t is not None:
        supplied=np.asarray(t,dtype=np.float64)
        canonical=time_ns//1000000000+(time_ns%1000000000)*1e-9
        if supplied.shape!=(len(q),) or not np.isfinite(supplied).all() or np.max(np.abs(supplied-canonical))>1e-10:
            raise ValueError('tracker_time_identity_mismatch')
    duration=np.diff(time_ns)
    # ros::Duration::toSec(): double(sec) + 1e-9*double(nsec).
    h=(duration//1000000000).astype(np.float64)+(duration%1000000000).astype(np.float64)*1e-9
    integral=.5*h[:,None]*(v[:-1]+v[1:])
    consistent=np.all(np.abs(np.diff(q,axis=0)-integral)<=1e-9,axis=1)
    bends=np.ascontiguousarray(np.where(consistent[:,None],h[:,None]*(v[:-1]-v[1:])/8.,0.))
    return bends,h,np.ascontiguousarray(time_ns)
