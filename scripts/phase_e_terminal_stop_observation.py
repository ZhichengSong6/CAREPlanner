"""Read-only terminal evidence; never publishes motion or changes task scoring."""
import math
import threading


class TerminalStopObservation:
    def __init__(self, hold_ns=100000000, velocity_limit=.05,
                 evidence_max_age_ns=100000000, sample_max_gap_ns=50000000):
        self.lock = threading.RLock()
        self.hold_ns = hold_ns
        self.velocity_limit = velocity_limit
        self.evidence_max_age_ns = evidence_max_age_ns
        self.sample_max_gap_ns = sample_max_gap_ns
        self.commit_stamp = 0
        self.duration_ns = 0
        self.tail_valid = False
        self.tail_completed = False
        self.completion_sample_ns = None
        self.evidence = {}
        self.last_sample_ns = None
        self.hold_start_ns = None
        self.hold_end_ns = None
        self.reason = 'waiting_committed_tail'
        self.resets = 0

    def _reset(self, reason):
        if self.hold_start_ns is not None:
            self.resets += 1
        self.hold_start_ns = self.hold_end_ns = None
        self.reason = reason

    def commit(self, stamp_ns, duration_ns, zero_tail_valid):
        with self.lock:
            if stamp_ns <= self.commit_stamp:
                return
            self.commit_stamp = stamp_ns
            self.duration_ns = duration_ns
            self.tail_valid = bool(zero_tail_valid)
            self.tail_completed = False
            self.completion_sample_ns = None
            self._reset('new_execution')

    def evidence_update(self, kind, receive_ros_ns, valid):
        with self.lock:
            previous = self.evidence.get(kind)
            self.evidence[kind] = (receive_ros_ns, bool(valid))
            if not valid or (previous and receive_ros_ns < previous[0]):
                self._reset('invalid_'+kind)

    def tracker(self, receive_ros_ns, token, phase_ns, source, complete, aborted, sample_ros_ns=None):
        with self.lock:
            matched = token == self.commit_stamp and token > 0
            normal_source = source in ('active_trajectory', 'trajectory_complete_hold')
            if aborted or (token > 0 and not matched):
                self.tail_completed = False
                self.completion_sample_ns = None
            if matched and normal_source and complete and phase_ns >= self.duration_ns:
                self.tail_completed = self.tail_valid and not aborted
                if self.tail_completed and self.completion_sample_ns is None:
                    self.completion_sample_ns = receive_ros_ns if sample_ros_ns is None else sample_ros_ns
            valid = (not aborted and
                     ((matched and normal_source) or
                      (token == 0 and source == 'last_reference_hold' and self.tail_completed)))
            self.evidence_update('tracker', receive_ros_ns, valid)

    def sample(self, stamp_ns, now_ros_ns, inside, velocity_inf):
        with self.lock:
            gap_ok = (self.last_sample_ns is None or
                      0 < stamp_ns-self.last_sample_ns <= self.sample_max_gap_ns)
            self.last_sample_ns = stamp_ns
            if not gap_ok:
                self._reset('noncontinuous_measured_samples')
                return False
            if not 0 <= now_ros_ns-stamp_ns <= self.sample_max_gap_ns:
                self._reset('stale_measured_sample')
                return False
            if not inside or not math.isfinite(velocity_inf) or velocity_inf > self.velocity_limit:
                self._reset('pose_or_speed_outside')
                return False
            if not self.tail_completed:
                self._reset('waiting_complete_certified_tail')
                return False
            if self.completion_sample_ns is None or stamp_ns < self.completion_sample_ns:
                self._reset('measured_sample_precedes_tail_completion')
                return False
            for kind in ('tracker', 'execution_vbc', 'regime', 'actuator'):
                t, valid = self.evidence.get(kind, (0, False))
                if not valid or not 0 <= now_ros_ns-t <= self.evidence_max_age_ns:
                    self._reset('missing_or_stale_'+kind)
                    return False
            if self.hold_start_ns is None:
                self.hold_start_ns = stamp_ns
            self.hold_end_ns = stamp_ns
            done = stamp_ns-self.hold_start_ns >= self.hold_ns
            self.reason = 'complete' if done else 'measuring_stopped_hold'
            return done

    def snapshot(self):
        with self.lock:
            return dict(protocol='terminal_tail_then_measured_hold_v2',
                completed=self.reason == 'complete', reason=self.reason,
                execution_stamp_ns=self.commit_stamp, tail_valid=self.tail_valid,
                tail_completed=self.tail_completed, duration_ns=self.duration_ns,
                completion_sample_ros_ns=self.completion_sample_ns,
                hold_start_ros_ns=self.hold_start_ns, hold_end_ros_ns=self.hold_end_ns,
                required_hold_ns=self.hold_ns, velocity_limit_rad_s=self.velocity_limit,
                evidence_max_age_ns=self.evidence_max_age_ns,
                measured_max_age_and_gap_ns=self.sample_max_gap_ns,
                reset_count=self.resets,
                evidence={k:dict(receive_ros_ns=t,valid=v) for k,(t,v) in self.evidence.items()},
                qualification='Read-only observation. Safety scores and exact audit/commit bindings require the saved-record review.')

    def invalid_measured_sample(self):
        with self.lock:
            self._reset('invalid_measured_sample')


def zero_hold_tail(trajectory, required_hold_ns):
    """Validate a static zero-speed tail on the received committed curve."""
    points = trajectory.points
    if len(points) < 2 or len(trajectory.joint_names) != 7:
        return False
    n = len(trajectory.joint_names)
    if len(set(trajectory.joint_names)) != n:
        return False
    end = points[-1]
    if len(end.positions) != n or not all(math.isfinite(x) for x in end.positions):
        return False
    start = end.time_from_start.to_nsec()-required_hold_ns
    if start < 0:
        return False
    tail = []
    for p in reversed(points):
        tail.append(p)
        if p.time_from_start.to_nsec() <= start:
            break
    if tail[-1].time_from_start.to_nsec() > start:
        return False
    for p in tail:
        if len(p.positions) != n or len(p.velocities) != n:
            return False
        if not all(math.isfinite(x) for x in (*p.positions, *p.velocities)):
            return False
        if max(abs(x) for x in p.velocities) > 1e-8:
            return False
        if max(abs(x-y) for x,y in zip(p.positions,end.positions)) > 1e-9:
            return False
    return all(b.time_from_start.to_nsec() > a.time_from_start.to_nsec()
               for a,b in zip(points,points[1:]))


def subscribe_terminal_observation(rospy, state):
    import re
    from std_msgs.msg import String, Float64MultiArray
    from trajectory_msgs.msg import JointTrajectory

    def fields(msg):
        return dict(re.findall(r'(\w+)=([^\s,]+)', msg.data))

    def tracker(msg):
        d=fields(msg)
        try:
            sample_time=int(d['sample_ros_ns'])
            if sample_time<=0:raise ValueError('missing positive tracker sample time')
            state.tracker(rospy.Time.now().to_nsec(), int(d.get('execution_stamp_ns',0)),
                round(float(d.get('phase_s',0))*1e9), d.get('source'),
                d.get('complete') == '1', d.get('execution_aborted') == '1',sample_time)
        except (KeyError, TypeError, ValueError, OverflowError):
            state.evidence_update('tracker',rospy.Time.now().to_nsec(),False)

    def commit(msg):
        state.commit(msg.header.stamp.to_nsec(),
                     msg.points[-1].time_from_start.to_nsec() if msg.points else 0,
                     zero_hold_tail(msg,state.hold_ns))

    def execution(msg):
        d=fields(msg)
        state.evidence_update('execution_vbc',rospy.Time.now().to_nsec(),
                              d.get('success') == '1' and d.get('has_violation') == '0')

    def regime(msg):
        state.evidence_update('regime',rospy.Time.now().to_nsec(),fields(msg).get('state') == 'NORMAL')

    def actuator(msg):
        valid = (len(msg.data) == 7 and all(math.isfinite(x) for x in msg.data)
                 and max(abs(x) for x in msg.data) <= state.velocity_limit)
        state.evidence_update('actuator',rospy.Time.now().to_nsec(),valid)

    return [rospy.Subscriber(topic,typ,callback,queue_size=100,tcp_nodelay=True)
            for topic,typ,callback in [
                ('/care_planner/committed_trajectory',JointTrajectory,commit),
                ('/care_planner/execution/tracker_summary',String,tracker),
                ('/care_planner/execution_vbc/summary',String,execution),
                ('/care_planner/c4_4/regime_summary',String,regime),
                ('/care_arm/arm_group_velocity_controller/command',Float64MultiArray,actuator)]]
