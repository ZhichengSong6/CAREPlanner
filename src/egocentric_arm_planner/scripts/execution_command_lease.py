"""Identity-preserving limiter transport; authorization is a separate service.

No deadline or source timestamp is generated here. Repeated timer ticks cannot
turn one received tracker command into an indefinitely fresh stream.
"""
import copy
import math
import numpy as np


class LeasedCommandLimiter:
    def __init__(self, names, velocity, acceleration, source_timeout=.1):
        self.names=list(names)
        self.velocity=np.asarray(velocity,dtype=float)
        self.acceleration=np.asarray(acceleration,dtype=float)
        self.timeout=float(source_timeout)
        self.session=0; self.state_time=-1.; self.retired=0
        self.last_sequence=0; self.command=None; self.output=None
        self.output_sequence=0; self.last_time=None
        self.rejected=0

    def on_state(self, state, now):
        stamp=state.header.stamp.to_sec()
        if (not state.execution_lease_required or not state.controller_session or
                state.clock_fault or not 0<=now-stamp<=.05 or
                not np.allclose(state.velocity_limits,self.velocity,rtol=0.,atol=1e-12) or
                not np.allclose(state.acceleration_limits,self.acceleration,rtol=0.,atol=1e-12)):
            return False
        if stamp<=self.state_time:
            return False
        if state.controller_session!=self.session:
            self.session=state.controller_session; self.command=None; self.output=None
            self.last_sequence=self.output_sequence=0; self.last_time=None; self.retired=0
        self.state_time=stamp
        self.retired=max(self.retired,state.retired_execution_through_ns)
        return True

    def receive(self, msg, now):
        generated=msg.header.stamp.to_sec()
        cap=np.asarray(getattr(msg,'target_abs_velocity',[0.]*len(self.names)),dtype=float)
        bounded=getattr(msg,'target_range_enforced',False)
        range_valid=(cap.shape==self.velocity.shape and np.isfinite(cap).all() and
                     np.all(cap>=0.) and np.all(cap<=self.velocity))
        valid=(msg.controller_session==self.session and bool(self.session) and
               msg.execution_stamp_ns>self.retired and msg.lease_sequence>0 and
               msg.source_sequence>self.last_sequence and
               msg.joint_names==self.names and len(msg.velocity)==len(self.names) and
               all(math.isfinite(x) for x in msg.velocity) and
               np.all(np.abs(msg.velocity)<=self.velocity+1e-12) and
               range_valid and (not bounded or np.all(np.abs(msg.velocity)<=cap)) and
               0<=now-generated<self.timeout and 0<=now-self.state_time<=.05 and
               msg.valid_until.to_sec()>now and len(msg.certificate_sha256)==64 and
               all(x in '0123456789abcdef' for x in msg.certificate_sha256) and
               (self.command is None or msg.execution_stamp_ns>=self.command.execution_stamp_ns))
        if not valid:
            self.rejected+=1; return False
        self.last_sequence=msg.source_sequence
        self.command=copy.deepcopy(msg)
        if msg.stop:
            self.retired=max(self.retired,msg.execution_stamp_ns)
        return True

    def stop(self):
        if self.command:
            self.retired=max(self.retired,self.command.execution_stamp_ns)
            self.command.stop=True

    def tick(self, now, dt):
        if self.command is None:
            return None
        msg=copy.deepcopy(self.command)
        if (self.last_time is not None and now<self.last_time) or not 0<=now-self.state_time<=.05:
            self.stop()
        self.last_time=now
        if (now-msg.header.stamp.to_sec()>=self.timeout or now>=msg.valid_until.to_sec() or
                msg.execution_stamp_ns<=self.retired):
            self.stop()
        msg.stop=self.command.stop
        if msg.stop:
            msg.velocity=[0.]*len(self.names)
        elif msg.source_sequence!=self.output_sequence:
            previous=np.zeros(len(self.names)) if self.output is None else np.asarray(self.output.velocity)
            delta=np.clip(np.asarray(msg.velocity)-previous,-self.acceleration*dt,self.acceleration*dt)
            msg.velocity=np.clip(previous+delta,-self.velocity,self.velocity).tolist()
            if getattr(msg,'target_range_enforced',False):
                # This is a target bound, not an instantaneous physical output
                # bound. The controller retains its own bounded deceleration.
                msg.velocity=np.clip(msg.velocity,-np.asarray(msg.target_abs_velocity),msg.target_abs_velocity).tolist()
        elif self.output is not None:
            msg.velocity=list(self.output.velocity)
        self.output=copy.deepcopy(msg); self.output_sequence=msg.source_sequence
        return msg
