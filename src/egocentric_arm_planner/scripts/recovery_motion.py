#!/usr/bin/env python3
"""Build one acceleration-feasible reference before either safety audit."""
import math


def interpolate_motion(q0, q1, v0, v1, dt, fraction):
    """Exact subsegment of a trapezoid-integrated acceleration ramp."""
    n=len(q0)
    if not (n and len(q1)==len(v0)==len(v1)==n and math.isfinite(dt) and dt>0):
        return None
    if not all(math.isfinite(x) for row in [q0,q1,v0,v1] for x in row):
        return None
    if any(abs(b-a-.5*dt*(v+w))>1e-9 for a,b,v,w in zip(q0,q1,v0,v1)):
        return None
    s=float(fraction)
    c=[dt*(v-w)/8. for v,w in zip(v0,v1)]
    return ([a+s*(b-a)+4.*s*(1.-s)*bend for a,b,bend in zip(q0,q1,c)],
            [(b-a)/dt+(4.-8.*s)*bend/dt for a,b,bend in zip(q0,q1,c)],
            [-8.*bend/(dt*dt) for bend in c])


def ramp_step(velocity, desired, acceleration, dt):
    if not math.isfinite(dt) or dt <= 0 or not (len(velocity)==len(desired)==len(acceleration)):
        raise ValueError('Invalid recovery ramp')
    delta=[float(y)-float(x) for x,y in zip(velocity,desired)]
    if not all(math.isfinite(x) for x in list(velocity)+list(desired)+list(acceleration)) or any(a<=0 for a in acceleration):
        raise ValueError('Invalid recovery state')
    # One fraction preserves the coordinated requested velocity change.
    fraction=min([1.]+[a*dt/abs(d) for a,d in zip(acceleration,delta) if abs(d)>1e-12])
    return [v+fraction*d for v,d in zip(velocity,delta)]


def has_nodal_motion(points, acceleration, tolerance=1e-6):
    """Recognize a solved ramp, never a synthetic terminal zero marker.

    The tolerance accommodates QP residuals; the executable is still rebuilt
    by exact integration and audited in full from the current measured state.
    """
    n = len(acceleration)
    if len(points) < 2 or not n:
        return False
    for a, b in zip(points, points[1:]):
        dt = (b.time_from_start-a.time_from_start).to_sec()
        if not math.isfinite(dt) or dt <= 0:
            return False
        if any(len(row) != n for row in [a.positions,b.positions,a.velocities,b.velocities]):
            return False
        if not all(math.isfinite(x) for row in [a.positions,b.positions,a.velocities,b.velocities] for x in row):
            return False
        if any(abs(y-x-.5*dt*(v+w)) > tolerance or abs(w-v) > limit*dt+tolerance
               for x,y,v,w,limit in zip(a.positions,b.positions,a.velocities,b.velocities,acceleration)):
            return False
    return True


def residual_motion_bound(reference_q, reference_v, q, v, radii, acceleration,
                          horizon_s, feedback_gain):
    """Bound changes ABOUT a proved motion; do not add its nominal braking."""
    if not (len(reference_q)==len(reference_v)==len(q)==len(v)==len(radii)==len(acceleration)):
        raise ValueError('Invalid residual dimensions')
    if not all(math.isfinite(x) for row in [reference_q,reference_v,q,v,radii,acceleration] for x in row):
        raise ValueError('Invalid residual state')
    if horizon_s<0 or not math.isfinite(horizon_s) or feedback_gain<0 or not math.isfinite(feedback_gain):
        raise ValueError('Invalid residual horizon')
    total=0.
    for qr,vr,qj,vj,r,a in zip(reference_q,reference_v,q,v,radii,acceleration):
        if r<0 or a<=0: raise ValueError('Invalid residual contract')
        stop_horizon=max(abs(vr),abs(vj))/a+.05
        # Clipping a scalar braking velocity is nonexpansive. Account for
        # incoming velocity error and position-feedback amplification separately.
        time=max(horizon_s,stop_horizon)
        total+=r*(abs(qj-qr)+time*abs(vj-vr))*math.exp(feedback_gain*time)
    return total
