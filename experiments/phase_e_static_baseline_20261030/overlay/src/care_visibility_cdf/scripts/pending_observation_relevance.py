"""Retire superseded candidate discovery at a certified task terminal path.

This only changes pending steering work. It never completes an observation,
retires an execution blocker, changes a deadline, or authorizes motion.
"""
from collections import OrderedDict
import hashlib
import json
import math


def fields(text):
    return dict(word.split('=', 1) for word in text.split() if '=' in word)


class PendingObservationRelevance:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.goal = None
        self.revision = 0
        self.commits = OrderedDict()
        self.verdicts = OrderedDict()
        self.newest_commit = 0
        self.tracker = None
        self.execution_blocked = False

    def set_goal(self, goal):
        if goal != self.goal:
            self.goal = goal
            self.revision += 1
            self.commits.clear()
            self.verdicts.clear()
            self.newest_commit = 0
            self.tracker = None

    def _trim(self, values):
        while len(values) > self.capacity:
            values.popitem(last=False)

    def verdict(self, text, now):
        f = fields(text)
        try:
            audit = int(f['audited_trajectory_stamp_ns'])
            if audit <= 0:
                return
            record = dict(f, received=float(now), revision=self.revision)
            # A duplicate receipt must not refresh evidence freshness.
            if audit not in self.verdicts:
                self.verdicts[audit] = record
                self._trim(self.verdicts)
        except (KeyError, ValueError):
            return

    def commit(self, token, payload, terminal, now):
        token = int(token)
        if token <= 0 or token < self.newest_commit:
            return
        old = self.commits.get(token)
        if old is not None:
            if old['payload'] != payload:
                old['terminal'] = False
            return
        self.newest_commit = token
        self.commits[token] = dict(payload=payload, terminal=bool(terminal),
                                  revision=self.revision, received=float(now))
        self._trim(self.commits)

    def observe_tracker(self, text, now):
        self.tracker = dict(fields(text), received=float(now))

    def decide(self, work, now, measured_inside=False, existing_obligations=False):
        """Conservative terminal-only exclusion; uncertainty retains work."""
        if work.get('source') != 'candidate':
            return 'retain', 'execution_or_legacy_source'
        if not self.goal or work.get('task_revision') != self.revision:
            return 'retain', 'missing_or_changed_task_context'
        if existing_obligations or self.execution_blocked:
            return 'retain', 'live_observation_responsibility'
        identity = work.get('identity') or {}
        try:
            source = int(identity['trajectory_stamp_ns'])
            token = self.newest_commit
            old = self.verdicts[source]
            curve = self.commits[token]
            if old.get('result') != 'unsafe' or old.get('committed') != '0':
                return 'retain', 'source_not_confirmed_rejected'
            proof = next(v for v in reversed(list(self.verdicts.values()))
                         if int(v.get('execution_stamp_ns', 0)) == token and
                         v.get('committed') == '1' and v.get('result') == 'safe')
            audit = int(proof['audited_trajectory_stamp_ns'])
            if not source < audit < token or not curve['terminal']:
                return 'retain', 'no_newer_terminal_curve'
            if (proof.get('verification_view') != 'full_horizon' or
                    proof.get('safety_gate') != 'vbc' or
                    proof.get('normal_motion_consistent') != '1' or
                    curve['revision'] != self.revision or proof['revision'] != self.revision):
                return 'retain', 'terminal_proof_scope_mismatch'
            evidence = json.loads(proof['vbc_evidence'])
            if (evidence.get('schema') != 1 or
                    int(evidence.get('trajectory_stamp_ns', 0)) != audit or
                    evidence.get('violations') != []):
                return 'retain', 'terminal_vbc_evidence_mismatch'
            tracker = self.tracker or {}
            age = float(now) - tracker['received']
            if (not 0 <= age <= .1 or
                    int(tracker.get('execution_stamp_ns', 0)) != token or
                    tracker.get('execution_aborted') != '0' or
                    tracker.get('source') not in ('active_trajectory', 'trajectory_complete_hold') or
                    not measured_inside):
                return 'retain', 'terminal_execution_not_current'
        except (KeyError, ValueError, TypeError, StopIteration):
            return 'retain', 'incomplete_terminal_evidence'
        return 'obsolete', 'rejected_candidate_superseded_at_task_terminal'


def trajectory_payload(msg):
    # Header.seq is assigned by ROS publication, not by the trajectory owner.
    data = [msg.header.frame_id, list(msg.joint_names), [
        [list(p.positions), list(p.velocities), list(p.accelerations),
         list(p.effort), p.time_from_start.to_nsec()] for p in msg.points]]
    return hashlib.sha256(json.dumps(data, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def attach_pending_relevance(node):
    """Use the existing business identities and original terminal pose bounds."""
    import numpy as np
    import rospy
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from trajectory_msgs.msg import JointTrajectory
    from validate_visibility_oracle import URDF, find_chain_joints, fk_transform
    from evaluate_direct_vs_projection_ascent import DEFAULT_JOINT_NAMES

    robot = URDF.from_xml_file(str(node.urdf_path))
    chain = find_chain_joints(robot, 'base_link', 'EE_link')
    state = node._pending_relevance

    def pose_inside(q):
        if state.goal is None:
            return False
        goal = np.asarray(state.goal[1:], dtype=float)
        tf = fk_transform(chain, dict(zip(DEFAULT_JOINT_NAMES, q)))
        x, y, z, w = goal[3:]
        n = math.sqrt(x*x+y*y+z*z+w*w)
        if n < 1e-12:
            return False
        x, y, z, w = x/n, y/n, z/n, w/n
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        angle = math.acos(float(np.clip((np.trace(rotation.T @ tf[:3, :3])-1)/2, -1, 1)))
        return bool(np.linalg.norm(tf[:3, 3]-goal[:3]) <= .02 and angle <= .2)

    def goal_cb(msg):
        p, q = msg.pose.position, msg.pose.orientation
        value = (msg.header.frame_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if value[0] != 'base_link' or not all(math.isfinite(v) for v in value[1:]):
            value = None
        with node._coherent_bundle_lock:
            state.set_goal(value)

    def commit_cb(msg):
        try:
            mapping = node._trajectory_mapping(msg)
            end = msg.points[-1]
            threshold = end.time_from_start.to_nsec()-100000000
            tail = [p for p in msg.points if p.time_from_start.to_nsec() >= threshold]
            previous = [p for p in msg.points if p.time_from_start.to_nsec() <= threshold]
            if previous:
                tail.insert(0, previous[-1])
            zero_tail = (len(tail) >= 2 and threshold >= 0 and bool(previous) and
                         all(len(p.positions) == 7 and len(p.velocities) == 7 and
                             max(abs(a-b) for a, b in zip(p.positions, end.positions)) <= 1e-9 and
                             max(abs(v) for v in p.velocities) <= 1e-8 for p in tail))
            payload = trajectory_payload(msg)
            with node._coherent_bundle_lock:
                terminal = zero_tail and msg.header.frame_id == 'base_link' and pose_inside(
                    [end.positions[j] for j in mapping])
                state.commit(msg.header.stamp.to_nsec(), payload, terminal, rospy.Time.now().to_sec())
        except (IndexError, TypeError, ValueError):
            return

    def verdict_cb(msg):
        with node._coherent_bundle_lock:
            state.verdict(msg.data, rospy.Time.now().to_sec())

    def tracker_cb(msg):
        with node._coherent_bundle_lock:
            state.observe_tracker(msg.data, rospy.Time.now().to_sec())

    node._pending_relevance_pose_inside = pose_inside
    node._pending_relevance_subscribers = [
        rospy.Subscriber('/care_planner/ee_target_pose', PoseStamped, goal_cb, queue_size=1),
        rospy.Subscriber('/care_planner/committed_trajectory', JointTrajectory, commit_cb, queue_size=8),
        rospy.Subscriber('/care_planner/verification_outcome', String, verdict_cb, queue_size=32),
        rospy.Subscriber('/care_planner/execution/tracker_summary', String, tracker_cb, queue_size=8)]
