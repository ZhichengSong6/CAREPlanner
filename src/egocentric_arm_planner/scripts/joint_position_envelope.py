"""Local actuator command envelope; no observation or planning policy."""
import math

def cap_velocity(proposed, previous, position, lower, upper, acceleration, velocity, dt, horizon):
    """Project into the existing slew interval and position-dependent speed cap.

    Position bounds already include the configured planner reserve. The speed
    cap solves v*horizon + v^2/(2*a) <= remaining distance. If the two envelopes
    have no intersection, apply the fastest available braking under the original
    acceleration limit. This is a kinematic command envelope, not a claim about
    unknown physical actuator dynamics.
    """
    values=(proposed,previous,position,lower,upper,acceleration,velocity,dt,horizon)
    if not all(math.isfinite(x) for x in values) or not (lower<upper and acceleration>0 and velocity>0 and dt>0 and horizon>=0):
        raise ValueError('Invalid position-envelope input')
    def speed(distance):
        ah=acceleration*horizon
        return min(velocity,math.sqrt(ah*ah+2*acceleration*max(0.,distance))-ah)
    pos_lo=-speed(position-lower);pos_hi=speed(upper-position)
    slew_lo=max(-velocity,previous-acceleration*dt)
    slew_hi=min(velocity,previous+acceleration*dt)
    lo=max(pos_lo,slew_lo);hi=min(pos_hi,slew_hi)
    feasible=lo<=hi
    if feasible:result=min(hi,max(lo,proposed))
    elif slew_hi<pos_lo:result=slew_hi
    else:result=slew_lo
    return result,dict(limited=abs(result-proposed)>1e-10,infeasible=not feasible,
                      position_lower_speed=pos_lo,position_upper_speed=pos_hi)
