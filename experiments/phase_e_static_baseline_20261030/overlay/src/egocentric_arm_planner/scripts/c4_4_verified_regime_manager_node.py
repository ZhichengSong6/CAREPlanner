#!/usr/bin/env python3
"""Verified CAREPlanner candidate/execution regime manager.

C4.4 baseline semantics:
    NORMAL -> REPAIR -> PROBE_NORMAL -> NORMAL
and a safe committed REPAIR candidate immediately enters PROBE_NORMAL.

C4.7 optional visibility-acquisition gate:
    a safe REPAIR candidate is only permission to execute the active-sensing
    detour.  It does NOT prove that the visibility objective has been acquired.
    While the gate is enabled, safe repair candidates may keep committing and the
    planner stays in REPAIR.  PROBE_NORMAL is entered only after the real
    confidence/visibility layer reports acquisition_complete=True after first
    publishing False in the current REPAIR episode.

Candidate certification and committed-execution safety remain separate. A C5.8
candidate may be rejected by the final executable GCDF gate or by exact VBC.
PROBE success is NOT counted at commit time. It is counted only after the
tracker confirms that the certified task-prefix duration for the exact committed
header.stamp execution token has actually elapsed. The certified brake+hold tail
remains the fail-safe while the next probe is planned/certified. ROS Header.seq
is diagnostic only.
"""

import math
import json
import os
import re
import threading
import time
from collections import deque

import rospy
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import PoseStamped


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def _tokens(text):
    return {k: v for k, v in _TOKEN_RE.findall(text or "")}


def _as_bool(v):
    if v in ("1", "true", "True"):
        return True
    if v in ("0", "false", "False"):
        return False
    return None


def _as_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v, default=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class C44VerifiedRegimeManager:
    NORMAL = "NORMAL"
    REPAIR = "REPAIR"
    PROBE_NORMAL = "PROBE_NORMAL"

    def __init__(self):
        self._lock = threading.RLock()

        self.rate = float(rospy.get_param("~rate", 20.0))
        self.candidate_unsafe_required = int(rospy.get_param(
            "~candidate_unsafe_required", 2))
        self.execution_unsafe_required = int(rospy.get_param(
            "~execution_unsafe_required", 2))
        legacy_probe_required = int(rospy.get_param(
            "~probe_safe_commits_required", 3))
        self.probe_completed_prefixes_required = int(rospy.get_param(
            "~probe_completed_prefixes_required", legacy_probe_required))
        self.task_first_research = os.environ.get('CARE_TASK_FIRST_RESEARCH') == 'true'
        self.repair_goal_stopping = os.environ.get('CARE_REPAIR_GOAL_STOPPING') == 'true'
        self._goal_stop_execution = 0
        self._goal_stop_feedback_time = None
        self.task_first_seen_count = 0
        self.task_first_seen_point_count = 0
        self.task_first_observation_returns = 0
        self._rolling_release = None
        self._rolling_request_pub = rospy.Publisher('/care_planner/rolling_task/consumed_request', String, queue_size=8)
        self.matched_observation_release = os.environ.get('CARE_MATCHED_OBSERVATION_RELEASE') == 'true'
        if self.matched_observation_release:
            # catkin devel relays execute in a private namespace, so importing
            # a sibling relay does not export its classes. __file__ points at
            # this real source (or installed script); load the sibling itself.
            import importlib.util
            release_spec = importlib.util.spec_from_file_location(
                'care_observation_release_identity',
                os.path.join(os.path.dirname(__file__), 'observation_release_identity.py'))
            release_module = importlib.util.module_from_spec(release_spec)
            release_spec.loader.exec_module(release_module)
            self._observation_release_identity = release_module.ObservationReleaseIdentity(allow_completed_gain=os.environ.get('CARE_ROLLING_TASK_OBSERVATION')=='true')
            self._observation_handoff = release_module.ObservationHandoffIdentity()
            self._observation_waypoint_region = None
        if self.task_first_research:
            self.probe_completed_prefixes_required = 1
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.25))
        self.clear_pulse_s = float(rospy.get_param("~clear_pulse_s", 0.12))
        self.probe_ignore_s = float(rospy.get_param("~probe_ignore_s", 0.16))

        self.repair_completion_gate_enabled = bool(rospy.get_param(
            "~repair_completion_gate_enabled", False))
        self.research_opportunistic_probe = os.environ.get(
            'CARE_RESEARCH_OPPORTUNISTIC_PROBE','false').lower()=='true'
        self.repair_completion_topic = str(rospy.get_param(
            "~repair_completion_topic",
            "/care_planner/active_sensing/visibility_acquisition_complete"))

        if self.rate <= 0.0:
            raise ValueError("~rate must be positive")
        if self.candidate_unsafe_required < 1 or self.execution_unsafe_required < 1:
            raise ValueError("unsafe streak requirements must be >= 1")
        if self.probe_completed_prefixes_required < 1:
            raise ValueError("probe_completed_prefixes_required must be >= 1")
        if min(self.input_timeout,
               self.clear_pulse_s, self.probe_ignore_s) <= 0.0:
            raise ValueError("timeouts/pulses must be positive")

        self.candidate_outcome_topic = str(rospy.get_param(
            "~candidate_outcome_topic", "/care_planner/verification_outcome"))
        self.execution_summary_topic = str(rospy.get_param(
            "~execution_summary_topic", "/care_planner/execution_vbc/summary"))
        self.tracker_summary_topic = str(rospy.get_param(
            "~tracker_summary_topic", "/care_planner/execution/tracker_summary"))
        self.probe_single_flight_summary_topic = str(rospy.get_param(
            "~probe_single_flight_summary_topic",
            "/care_planner/execution/probe_single_flight_summary"))
        self.replan_request_topic = str(rospy.get_param(
            "~replan_request_topic", "/care_planner/local_planner/replan_request"))
        self.committed_trajectory_topic = str(rospy.get_param(
            "~committed_trajectory_topic", "/care_planner/committed_trajectory"))
        self.execution_ready_topic = str(rospy.get_param(
            "~execution_ready_topic", "/care_planner/execution/ready"))
        self.physical_deadline_topic = str(rospy.get_param(
            "~physical_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_rolling"))
        self.effective_deadline_topic = str(rospy.get_param(
            "~effective_deadline_topic",
            "/care_planner/active_sensing/visibility_waypoint_deadline_effective"))
        self.trigger_topic = str(rospy.get_param(
            "~trigger_topic",
            "/care_planner/execution/predicted_vbc_recovery_triggered"))
        self.clear_topic = str(rospy.get_param(
            "~clear_topic", "/care_planner/execution/predicted_vbc_recovery_clear"))
        self.verification_hold_topic = str(rospy.get_param(
            "~verification_hold_topic",
            "/care_planner/execution/predicted_vbc_verification_hold"))
        self.probe_active_topic = str(rospy.get_param(
            "~probe_active_topic",
            "/care_planner/c4_4/probe_active"))
        self.summary_topic = str(rospy.get_param(
            "~summary_topic", "/care_planner/c4_4/regime_summary"))
        self.task_infeasible_topic = str(rospy.get_param(
            "~task_infeasible_topic",
            "/care_planner/local_planner/task_infeasible"))
        self.task_obstacle_blocked_topic = str(rospy.get_param(
            "~task_obstacle_blocked_topic",
            "/care_planner/local_planner/task_obstacle_blocked"))
        self.task_uncertified_topic = str(rospy.get_param(
            "~task_uncertified_topic",
            "/care_planner/local_planner/task_uncertified"))
        self.visibility_waypoint_active_topic = str(rospy.get_param(
            "~visibility_waypoint_active_topic",
            "/care_planner/active_sensing/visibility_waypoint_active"))
        self.candidate_vbc_summary_topic = str(rospy.get_param(
            "~candidate_vbc_summary_topic",
            "/care_planner/candidate_vbc/summary"))
        self.force_vbc_bootstrap_topic = str(rospy.get_param(
            "~force_vbc_bootstrap_topic",
            "/care_planner/trajectory_risk/force_bootstrap"))
        self.normal_task_replan_topic = str(rospy.get_param(
            "~normal_task_replan_topic",
            "/care_planner/local_planner/normal_task_replan_request"))

        self.state = self.NORMAL
        self.execution_ready = False
        self.execution_ready_time = None

        self.candidate_outcome_time = None
        self.last_candidate_seq = 0
        self.last_candidate_result = "none"
        self.last_candidate_safety_gate = "none"
        self.last_candidate_verification_view = "none"
        self.candidate_outcome_pending_gate_replay = False
        self.candidate_gate_replay_count = 0
        self.candidate_last_unsafe = False
        self.candidate_unsafe_streak = 0
        self.candidate_unsafe_count = 0
        self.candidate_safe_count = 0
        self.candidate_timeout_count = 0
        self.candidate_protocol_invalid_count = 0
        self.candidate_event_count = 0
        # C5.22: NORMAL VBC-unsafe hysteresis must keep producing fresh
        # candidates; otherwise streak=1 can deadlock forever with no commit.
        self.normal_unsafe_retry_count = 0
        self.normal_unsafe_confirmed_repair_count = 0

        self.execution_summary_time = None
        self.execution_last_unsafe = False
        self.execution_unsafe_streak = 0
        self.execution_event_latched = False
        self.execution_safety_event_count = 0
        self.execution_safe_count = 0

        self.tracker_summary_time = None
        self.tracker_active = False
        self.tracker_complete = False
        self.tracker_execution_stamp_ns = 0
        self.tracker_source = "none"
        self.tracker_execution_state = "NONE"
        self.tracker_abort_count = 0
        self.last_tracker_aborted_stamp_ns = 0
        # A rejected VBC candidate can reveal a new UNKNOWN blocker while an
        # older certified execution is still in flight in any regime. Hold
        # that execution until a new certified token takes over.
        self.vbc_unsafe_execution_stamp_ns = 0
        self.execution_vbc_hold_tokenized = False
        self.execution_audit_identities = {}
        self.pending_execution_audits = {}
        self.last_execution_audit_stamp_ns = 0
        self.last_execution_audit_token = 0

        # C5.17 decision-level PROBE single-flight. Candidate admission is
        # serialized by probe_single_flight_gate_node; mirror its phase here so
        # later planner failures cannot preempt an already certified/in-flight
        # PROBE decision.
        self.probe_single_flight_phase = "IDLE"
        self.probe_single_flight_execution_stamp_ns = 0
        self.probe_single_flight_reason = "startup"
        self.probe_single_flight_summary_time = None
        self.probe_failure_suppressed_busy_count = 0
        self.probe_infeasible_suppressed_busy_count = 0
        self.probe_uncertified_suppressed_busy_count = 0

        self.last_committed_trajectory_time = None
        self.last_commit_count = 0
        self.last_commit_event_time = None
        self.probe_completed_prefix_streak = 0
        self.probe_completed_execution_count = 0
        self.pending_probe_candidate_seq = 0
        self.pending_probe_execution_stamp_ns = 0
        self.pending_probe_effective_prefix_s = math.nan
        self.last_tracker_complete_seq = 0
        self.last_tracker_complete_stamp_ns = 0
        self.repair_safe_commit_count = 0
        self.repair_final_vbc_hold = False

        self.repair_entry_count = 0
        self.probe_entry_count = 0
        self.normal_entry_count = 0
        self.normal_task_replan_count = 0
        self.probe_failure_count = 0
        self.candidate_repair_entry_count = 0
        self.gcdf_repair_entry_count = 0
        self.gcdf_occupied_replan_count = 0
        self.execution_repair_entry_count = 0
        self.task_infeasible_repair_entry_count = 0
        self.task_infeasible_pending = False
        self.task_obstacle_blocked_pending = False
        self.task_obstacle_blocked_count = 0
        self.task_obstacle_replan_count = 0
        self.task_uncertified_repair_entry_count = 0
        self.task_uncertified_pending = False

        # C4.7: a stale latched True from a previous acquisition episode must not
        # immediately clear a newly-entered REPAIR.  Each episode arms only after
        # observing completion=False, then accepts a later False->True.
        self.repair_completion = False
        self.repair_completion_armed = False
        self.repair_completion_time = None
        self.repair_completion_event_count = 0

        # C5.11/C5.43: solver failure and active-sensing demand are separate
        # facts. A failed NORMAL or PROBE task QP may request same-SCP-trajectory
        # blocker rediscovery, but REPAIR is entered only after a persistent
        # visibility waypoint/obligation exists.
        self.visibility_waypoint_active = False
        # Scheduling facts only: pair the existing immutable waypoint identity
        # with the current acquisition snapshot. Neither grants execution.
        self._observation_waypoint = None
        self._observation_demand = None
        self._observation_retired_tokens = deque(maxlen=64)
        self.observation_takeover_count = 0
        self.observation_takeover_token = "none"
        self.blocker_rediscovery_pending = False
        self.blocker_rediscovery_origin = "none"
        self.blocker_rediscovery_force_bootstrap = False
        self.blocker_rediscovery_count = 0
        self.blocker_rediscovery_vbc_unsafe_count = 0
        self.blocker_rediscovery_vbc_safe_count = 0
        self.task_stall_stamp_ns = 0
        self.task_stall_deadline = None
        self.task_stall_confirmed = False
        self.task_stall_outcome = "none"
        # Only failed-QP retries use this acknowledged channel. Transport
        # repeats carry the same planner-issued ticket and spend no QP budget.
        self.qp_retry_token = ""
        self.qp_retry_mode = ""
        self.qp_retry_last_send = -math.inf
        self.qp_retry_retired = deque(maxlen=64)
        self.qp_retry_send_count = 0
        self.qp_retry_ack_count = 0
        self.qp_retry_last_status = "none"
        self.qp_retry_ready_nonce = ""
        self.qp_retry_ready_time = -math.inf

        self.clear_until = None
        self.probe_ignore_until = None
        self.last_transition_reason = "startup"
        self.physical_deadline_abs = None

        self.trigger_pub = rospy.Publisher(
            self.trigger_topic, Bool, queue_size=1, latch=True)
        self.clear_pub = rospy.Publisher(
            self.clear_topic, Bool, queue_size=1, latch=True)
        self.hold_pub = rospy.Publisher(
            self.verification_hold_topic, Bool, queue_size=1, latch=True)
        self.execution_vbc_stop_pub = rospy.Publisher(
            "/care_planner/execution/vbc_stop_request", String, queue_size=20)
        self.probe_active_pub = rospy.Publisher(
            self.probe_active_topic, Bool, queue_size=1, latch=True)
        self.deadline_pub = rospy.Publisher(
            self.effective_deadline_topic, Float64, queue_size=1)
        self.summary_pub = rospy.Publisher(
            self.summary_topic, String, queue_size=1, latch=True)
        self.replan_request_pub = rospy.Publisher(
            self.replan_request_topic, Bool, queue_size=10, latch=False)
        self.qp_retry_pub = rospy.Publisher(
            self.replan_request_topic + "/qp_retry", String,
            queue_size=10, latch=False)
        rospy.Subscriber(self.replan_request_topic + "/qp_retry_ack", String,
                         self._qp_retry_ack_cb, queue_size=10)
        self.qp_retry_ready_pub = rospy.Publisher(
            "/care_planner/benchmark/qp_retry_ping/ready", String, queue_size=4)
        rospy.Subscriber("/care_planner/benchmark/qp_retry_ping", String,
                         self._qp_retry_ready_cb, queue_size=4)
        self.force_vbc_bootstrap_pub = rospy.Publisher(
            self.force_vbc_bootstrap_topic, Bool, queue_size=1, latch=True)
        self.normal_task_replan_pub = rospy.Publisher(
            self.normal_task_replan_topic, Bool, queue_size=10, latch=False)

        rospy.Subscriber(
            self.candidate_outcome_topic, String, self._candidate_outcome_cb,
            queue_size=20)
        rospy.Subscriber(
            self.execution_summary_topic, String, self._execution_summary_cb,
            queue_size=1)
        rospy.Subscriber(str(rospy.get_param("~execution_audit_identity_topic",
            "/care_planner/execution/audit_identity")), String,
            self._execution_audit_identity_cb, queue_size=20)
        rospy.Subscriber(
            self.tracker_summary_topic, String, self._tracker_summary_cb,
            queue_size=20)
        rospy.Subscriber(
            self.probe_single_flight_summary_topic, String,
            self._probe_single_flight_summary_cb, queue_size=20)
        rospy.Subscriber(
            self.committed_trajectory_topic, JointTrajectory,
            self._committed_trajectory_cb, queue_size=1)
        rospy.Subscriber(
            self.execution_ready_topic, Bool, self._execution_ready_cb,
            queue_size=1)
        rospy.Subscriber(
            self.physical_deadline_topic, Float64, self._deadline_cb,
            queue_size=1)
        rospy.Subscriber(
            self.task_infeasible_topic, Bool, self._task_infeasible_cb,
            queue_size=10)
        rospy.Subscriber(
            self.task_obstacle_blocked_topic, Bool,
            self._task_obstacle_blocked_cb, queue_size=10)
        rospy.Subscriber(
            self.task_uncertified_topic, Bool, self._task_uncertified_cb,
            queue_size=10)
        rospy.Subscriber(
            "/care_planner/local_planner/task_stall", String,
            self._task_stall_cb, queue_size=10)
        rospy.Subscriber(str(rospy.get_param("~local_planner_summary_topic",
            "/care_planner/local_planner/summary")), String,
            self._repair_final_vbc_feedback_cb, queue_size=20)
        rospy.Subscriber(
            self.visibility_waypoint_active_topic, Bool,
            self._visibility_waypoint_active_cb, queue_size=1)
        rospy.Subscriber(str(rospy.get_param("~visibility_waypoint_q_topic",
            "/care_planner/active_sensing/visibility_waypoint_q")),
            Float64MultiArray, self._observation_waypoint_cb, queue_size=1)
        rospy.Subscriber(str(rospy.get_param("~visibility_acquisition_summary_topic",
            "/care_planner/active_sensing/visibility_acquisition_summary")),
            String, self._observation_acquisition_cb, queue_size=1)
        rospy.Subscriber(
            self.candidate_vbc_summary_topic, String,
            self._candidate_vbc_summary_cb, queue_size=20)
        if self.repair_completion_gate_enabled or self.research_opportunistic_probe:
            rospy.Subscriber(
                self.repair_completion_topic, Bool,
                self._repair_completion_cb, queue_size=1)

        self.trigger_pub.publish(Bool(data=False))
        self.probe_active_pub.publish(Bool(data=False))
        self.clear_pub.publish(Bool(data=False))
        self.hold_pub.publish(Bool(data=False))
        self.force_vbc_bootstrap_pub.publish(Bool(data=False))
        self._occlusion_reference_stamps = deque(maxlen=32)
        self._occlusion_pending = None
        self._occlusion_goal = None
        self._occlusion_consumed_routes = set()
        if os.environ.get('CARE_OCCLUSION_REROUTE_ENABLED','false') == 'true':
            rospy.Subscriber('/care_planner/task_trajectory',JointTrajectory,self._occlusion_reference_cb,queue_size=8)
            rospy.Subscriber('/care_planner/ee_target_pose',PoseStamped,self._occlusion_goal_cb,queue_size=1)
            rospy.Subscriber('/care_planner/active_sensing/occlusion_reroute',String,self._occlusion_route_cb,queue_size=8)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._timer_cb)
        self._publish_summary()

        rospy.logwarn(
            "[c4_regime] NORMAL->REPAIR->PROBE_NORMAL->NORMAL; candidate=%s "
            "execution=%s probe_completed_prefixes=%d completion_gate=%d completion_topic=%s",
            self.candidate_outcome_topic, self.execution_summary_topic,
            self.probe_completed_prefixes_required,
            int(self.repair_completion_gate_enabled),
            self.repair_completion_topic)

    def _occlusion_goal_cb(self,msg):
        p=msg.pose.position;q=msg.pose.orientation
        with self._lock:
            goal=[msg.header.frame_id,p.x,p.y,p.z,q.x,q.y,q.z,q.w]
            if goal != self._occlusion_goal:
                self._occlusion_consumed_routes.clear()
            self._occlusion_goal=goal
            self._occlusion_pending=None

    def _occlusion_reference_cb(self,msg):
        with self._lock:
            self._occlusion_reference_stamps.append(msg.header.stamp.to_nsec())
            self._try_occlusion_route_locked()

    def _occlusion_route_cb(self,msg):
        try:
            event=json.loads(msg.data)
            if (event.get('schema')!=1 or event.get('action')!='try_alternative_reference' or
                    event.get('actual_observation_completed') is not False or
                    not isinstance(event.get('deferred_tokens'),list)):
                return
            stamp=int(event['reference_stamp_ns'])
            if stamp<=0:return
        except (ValueError,KeyError,TypeError):return
        with self._lock:
            self._occlusion_pending=(event,rospy.Time.now())
            self._try_occlusion_route_locked()

    def _try_occlusion_route_locked(self):
        pending=self._occlusion_pending
        if pending is None:return
        event,received=pending;now=rospy.Time.now()
        if (not 0 <= (now-received).to_sec() <= .5 or event.get('goal_key')!=self._occlusion_goal):
            self._occlusion_pending=None;return
        if int(event['reference_stamp_ns']) not in self._occlusion_reference_stamps:return
        self._occlusion_pending=None
        if self.task_first_research:
            route=(event.get('goal_revision'),event.get('route_generation'),
                   tuple(sorted(event['deferred_tokens'])))
            if route in self._occlusion_consumed_routes:return
            # A refreshed stamp alone is not new information. A newly retired
            # observation, however, must release its now-empty recovery episode.
            self._occlusion_consumed_routes.add(route)
        if not self.execution_ready or self.state!=self.REPAIR:return
        deferred=event['deferred_tokens'];demand=self._observation_demand
        if demand is not None and demand[0] not in deferred:return
        # Planning may try another reference while unresolved UNKNOWN remains
        # in the map. This is not a completion edge or execution permission.
        for token in deferred:
            if token not in self._observation_retired_tokens:self._observation_retired_tokens.append(token)
        self._observation_waypoint=None;self._observation_demand=None
        self.visibility_waypoint_active=False
        self._transition_locked(self.PROBE_NORMAL if self.task_first_research else self.NORMAL,
            'external_occlusion_alternative_reference',now)
        self.replan_request_pub.publish(Bool(data=True))

    def _transition_locked(self, new_state, reason, now):
        if os.environ.get('CARE_ACTIVE_OBSERVATION', 'true') == 'false' and new_state != self.NORMAL:
            # Component ablation: preserve NORMAL task motion and every safety
            # hold; do not admit a dedicated observation-motion episode.
            self.passive_observation_suppressed = getattr(self, 'passive_observation_suppressed', 0) + 1
            self.last_transition_reason = 'passive_observation_suppressed_' + reason
            return
        if new_state == self.state:
            self.last_transition_reason = reason
            return
        old = self.state
        if new_state not in (self.PROBE_NORMAL, self.NORMAL):
            self._rolling_release = None
        self._retire_qp_retry_locked()
        if old == self.REPAIR and hasattr(self, '_observation_handoff'):
            self._observation_handoff.clear()
        self.state = new_state
        self.last_transition_reason = reason
        # Blocker rediscovery is a fail-closed substate used by NORMAL and
        # PROBE_NORMAL task-QP failures. Any real regime transition terminates
        # that substate.
        self.blocker_rediscovery_pending = False
        self.blocker_rediscovery_origin = "none"
        if self.blocker_rediscovery_force_bootstrap:
            self.force_vbc_bootstrap_pub.publish(Bool(data=False))
        self.blocker_rediscovery_force_bootstrap = False

        # Publish planner modes immediately on transition so subscribers cannot
        # race against stale REPAIR/PROBE state before the next 20 Hz timer tick.
        if hasattr(self, "trigger_pub"):
            self.trigger_pub.publish(Bool(
                data=bool(self.execution_ready and new_state == self.REPAIR)))
        if hasattr(self, "probe_active_pub"):
            self.probe_active_pub.publish(Bool(
                data=bool(self.execution_ready and new_state == self.PROBE_NORMAL)))

        if new_state == self.REPAIR:
            self.repair_entry_count += 1
            if hasattr(self, '_observation_handoff'):
                self._observation_handoff.clear()
                if self._observation_waypoint is not None and self._observation_waypoint_region is not None:
                    self._observation_handoff.bind(self._observation_waypoint[0],self._observation_waypoint_region,
                        (self._occlusion_goal,self.repair_entry_count))
            self.probe_completed_prefix_streak = 0
            self.pending_probe_candidate_seq = 0
            self.pending_probe_execution_stamp_ns = 0
            self.pending_probe_effective_prefix_s = math.nan
            self.clear_until = None
            self.probe_ignore_until = None
            self.repair_completion = False
            self.repair_completion_armed = False
            if reason.startswith("execution_"):
                self.execution_repair_entry_count += 1
            elif reason.startswith("final_gcdf_"):
                self.gcdf_repair_entry_count += 1
            elif reason.startswith("task_"):
                # Task-QP infeasibility has its own counter and is neither a
                # candidate-VBC failure nor an execution-VBC failure.
                pass
            else:
                self.candidate_repair_entry_count += 1
        elif new_state == self.PROBE_NORMAL:
            self.probe_entry_count += 1
            # The REPAIR episode is complete only after its obligations are
            # gone. Locally clear the waypoint-active latch so a stale final
            # q_vis publication from that episode cannot authorize a future
            # PROBE->REPAIR transition.
            self.visibility_waypoint_active = False
            self.probe_completed_prefix_streak = 0
            self.pending_probe_candidate_seq = 0
            self.pending_probe_execution_stamp_ns = 0
            self.pending_probe_effective_prefix_s = math.nan
            self.clear_until = now + rospy.Duration(self.clear_pulse_s)
            self.probe_ignore_until = now + rospy.Duration(self.probe_ignore_s)
        elif new_state == self.NORMAL:
            self.normal_entry_count += 1
            self.probe_completed_prefix_streak = 0
            self.pending_probe_candidate_seq = 0
            self.pending_probe_execution_stamp_ns = 0
            self.pending_probe_effective_prefix_s = math.nan
            self.clear_until = None
            self.probe_ignore_until = None

            # C5.33: after active-sensing detours, the original one-shot
            # /task_trajectory time axis is stale. Regenerate the same EE task
            # from the current measured state before NORMAL planning resumes.
            # This refreshes only the task reference; it does not restore the
            # legacy RECOVERY_HOLD / replan-ready controller handshake.
            if old == self.PROBE_NORMAL and self.execution_ready:
                self.normal_task_replan_count += 1
                self.normal_task_replan_pub.publish(Bool(data=True))

        rospy.logwarn(
            "[c4_regime] %s -> %s reason=%s repair_entries=%d probe_entries=%d",
            old, new_state, reason, self.repair_entry_count, self.probe_entry_count)

    def _execution_ready_cb(self, msg):
        if msg is None:
            return
        with self._lock:
            was_ready = bool(self.execution_ready)
            new_ready = bool(msg.data)
            self.execution_ready = new_ready
            self.execution_ready_time = rospy.Time.now()

            if not new_ready:
                # Repeated startup/not-ready heartbeats must NOT erase a safety
                # outcome that arrived while the gate was closed. Only a true
                # ready->not-ready falling edge starts a new execution epoch.
                if was_ready:
                    self._observation_waypoint = None
                    self._observation_demand = None
                    self._retire_qp_retry_locked()
                    self.state = self.NORMAL
                    self.candidate_unsafe_streak = 0
                    self.candidate_outcome_pending_gate_replay = False
                    self.execution_unsafe_streak = 0
                    self.execution_event_latched = False
                    self.probe_completed_prefix_streak = 0
                    self.pending_probe_candidate_seq = 0
                    self.pending_probe_execution_stamp_ns = 0
                    self.pending_probe_effective_prefix_s = math.nan
                    self.repair_completion = False
                    self.repair_completion_armed = False
                    self.task_obstacle_blocked_pending = False
                    self.blocker_rediscovery_pending = False
                    self.blocker_rediscovery_origin = "none"
                    self.blocker_rediscovery_force_bootstrap = False
                    self.force_vbc_bootstrap_pub.publish(Bool(data=False))
                    self.clear_until = None
                    self.probe_ignore_until = None
                self.last_transition_reason = "execution_not_ready"
                return

            # Ignore ready=True heartbeats. Replay pending decisions only on the
            # actual not-ready -> ready edge.
            self._pump_qp_retry_locked()
            if self._maybe_enter_observation_repair_locked(
                    self.execution_ready_time, "execution_ready"):
                return
            if was_ready:
                return

            if self.task_obstacle_blocked_pending:
                self.task_obstacle_blocked_pending = False
                self.task_obstacle_replan_count += 1
                self.last_transition_reason = (
                    "task_obstacle_blocked_after_gate_release_replan")
                self.replan_request_pub.publish(Bool(data=True))
            elif self.task_infeasible_pending and self.state == self.NORMAL:
                self.task_infeasible_pending = False
                self.task_uncertified_pending = False
                # C5.43: UNKNOWN-only task infeasibility is evidence that the
                # failed SCP trajectory intersects low-confidence space, but it
                # is not yet a steering obligation. Stay NORMAL and rediscover
                # the blocker on the exact SCP query trajectory. REPAIR starts
                # only after visibility_waypoint_active confirms q_vis exists.
                if self.visibility_waypoint_active:
                    self.task_infeasible_repair_entry_count += 1
                    self._transition_locked(
                        self.REPAIR,
                        "task_normal_infeasible_gcdf_obligation_ready_after_gate_release",
                        self.execution_ready_time)
                else:
                    self._begin_blocker_rediscovery_locked(
                        "task_normal_infeasible_after_gate_release",
                        force_bootstrap=False)
            elif self.task_uncertified_pending and self.state == self.NORMAL:
                self.task_uncertified_pending = False
                self.task_infeasible_pending = False
                # Numerical non-certification is not evidence of a visibility
                # blocker. Stay NORMAL and request a fresh measured-state solve.
                # Only an actual GCDF/VBC/visibility fact may enter REPAIR.
                self.last_transition_reason = (
                    "task_planner_uncertified_after_gate_release_replan")
                self.replan_request_pub.publish(Bool(data=True))
            elif (self.candidate_outcome_pending_gate_replay and
                  self.state == self.NORMAL):
                self.candidate_outcome_pending_gate_replay = False
                self.candidate_gate_replay_count += 1
                self._handle_normal_candidate_result_locked(
                    self.last_candidate_result,
                    self.last_candidate_safety_gate,
                    self.execution_ready_time,
                    replay=True)

    def _repair_final_vbc_feedback_cb(self, msg):
        fields = _tokens(msg.data)
        hold = _as_bool(fields.get("final_vbc_hold"))
        if hold is None:
            return
        with self._lock:
            if getattr(self, 'repair_goal_stopping', False):
                now = rospy.Time.now()
                execution = _as_int(fields.get('goal_stop_execution_stamp_ns'), 0)
                sample = _as_int(fields.get('goal_stop_sample_ros_ns'), 0)
                tracker_fresh = (self.tracker_summary_time is not None and
                    0 <= (now-self.tracker_summary_time).to_sec() <= self.input_timeout)
                eligible = (execution > 0 and execution == self.tracker_execution_stamp_ns and
                    execution != self.last_tracker_aborted_stamp_ns and
                    execution != self.vbc_unsafe_execution_stamp_ns and tracker_fresh and
                    self.tracker_source in ('active_trajectory', 'trajectory_complete_hold') and
                    0 <= now.to_nsec()-sample <= 50000000)
                self._goal_stop_execution = execution if eligible else 0
                self._goal_stop_feedback_time = now
                if eligible and self.execution_ready and self.state == self.REPAIR:
                    self._transition_locked(self.NORMAL, 'certified_goal_stop_handoff', now)
            self.repair_final_vbc_hold = hold
            if hold and self.state == self.REPAIR:
                self.last_transition_reason = "repair_final_vbc_no_progress_hold"

    def _finish_task_stall_locked(self, outcome):
        self.task_stall_outcome = outcome
        self.task_stall_deadline = None
        if self.blocker_rediscovery_origin == "task_qp_no_progress":
            self.blocker_rediscovery_pending = False
            self.blocker_rediscovery_origin = "none"
            self.blocker_rediscovery_force_bootstrap = False
            self.force_vbc_bootstrap_pub.publish(Bool(data=False))
        self.last_transition_reason = outcome

    def _retire_qp_retry_locked(self):
        if self.qp_retry_token:
            self.qp_retry_retired.append(self.qp_retry_token)
        self.qp_retry_token = ""
        self.qp_retry_mode = ""

    def _pump_qp_retry_locked(self):
        if (not self.qp_retry_token or not self.execution_ready or
                self.task_stall_stamp_ns or self.state != self.qp_retry_mode or
                (self.state == self.PROBE_NORMAL and self._probe_single_flight_busy_locked())):
            return
        now = time.monotonic()
        if now - self.qp_retry_last_send < .25:
            return
        self.qp_retry_last_send = now
        self.qp_retry_send_count += 1
        self.qp_retry_last_status = "awaiting_ack"
        self.qp_retry_pub.publish(String(data=self.qp_retry_token))

    def _qp_retry_ready_cb(self, msg):
        if msg is None or not re.fullmatch(r"[a-f0-9]{32}", msg.data):
            return
        with self._lock:
            self.qp_retry_ready_nonce = msg.data
            self.qp_retry_ready_time = time.monotonic()
            self.qp_retry_pub.publish(String(data="ready_" + msg.data))

    def _qp_retry_ack_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        with self._lock:
            if f.get("status") == "ready":
                if (self.qp_retry_ready_nonce and
                        f.get("token") == "ready_" + self.qp_retry_ready_nonce and
                        time.monotonic() - self.qp_retry_ready_time <= .5):
                    self.qp_retry_ready_pub.publish(String(data=self.qp_retry_ready_nonce))
                    self.qp_retry_ready_nonce = ""
                return
            if (not self.qp_retry_token or f.get("token") != self.qp_retry_token or
                    f.get("status") not in ("accepted", "duplicate", "stale", "blocked")):
                return
            self.qp_retry_ack_count += 1
            self.qp_retry_last_status = f["status"]
            self._retire_qp_retry_locked()

    def _task_stall_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        with self._lock:
            if f.get("status") == "retry":
                if self.task_stall_stamp_ns or self.state not in (self.NORMAL, self.PROBE_NORMAL):
                    return
                token = f.get("token", "")
                if (not token or token in self.qp_retry_retired or
                        f.get("mode") != self.state):
                    return
                if token != self.qp_retry_token:
                    self._retire_qp_retry_locked()
                    self.qp_retry_token = token
                    self.qp_retry_mode = f["mode"]
                    self.qp_retry_last_send = -math.inf
                self.last_transition_reason = "local_qp_primal_infeasible_unattributed_retry"
                self._pump_qp_retry_locked()
                return
            if f.get("status") == "reset":
                self._retire_qp_retry_locked()
                if self.task_stall_stamp_ns:
                    self._finish_task_stall_locked("qp_stall_reset_" + f.get("reason", "unknown"))
                self.task_stall_stamp_ns = 0
                self.task_stall_confirmed = False
                return
            stamp = _as_int(f.get("query_stamp_ns"), 0)
            if (f.get("status") != "stalled" or stamp <= 0 or
                    stamp == self.task_stall_stamp_ns or
                    self.state not in (self.NORMAL, self.PROBE_NORMAL)):
                return
            self.task_stall_stamp_ns = stamp
            self._retire_qp_retry_locked()
            self.task_stall_confirmed = False
            self.task_stall_deadline = time.monotonic() + 5.0
            self.task_stall_outcome = "qp_no_progress_wait_exact_blocker"
            self._begin_blocker_rediscovery_locked("task_qp_no_progress")

    def _begin_blocker_rediscovery_locked(
            self, origin, force_bootstrap=True):
        """Fail closed until a real visibility obligation exists.

        C5.43 local hard-GCDF task failures publish exact UNKNOWN blocker voxels
        directly to sensing and therefore call this with force_bootstrap=False.
        Legacy/numerical rediscovery paths may still request same-SCP VBC with
        force_bootstrap=True. Exact GCDF evidence must never be replaced by a
        narrower swept-body reclassification.
        """
        self.blocker_rediscovery_pending = True
        self.blocker_rediscovery_origin = str(origin)
        self.blocker_rediscovery_force_bootstrap = bool(force_bootstrap)
        self.blocker_rediscovery_count += 1
        self.pending_probe_candidate_seq = 0
        self.pending_probe_execution_stamp_ns = 0
        self.pending_probe_effective_prefix_s = math.nan
        self.last_transition_reason = (
            "{}_wait_visibility_obligation".format(origin))
        self.force_vbc_bootstrap_pub.publish(
            Bool(data=bool(force_bootstrap)))

    def _retire_observation_demand_locked(self):
        # Completion and waypoint travel on different connections. An old
        # completion must not retire a newer pose that arrived ahead of its
        # acquisition snapshot.
        demand = self._observation_demand
        if demand is not None:
            if demand[0] not in self._observation_retired_tokens:
                self._observation_retired_tokens.append(demand[0])
            if self._observation_waypoint is not None and self._observation_waypoint[0] == demand[0]:
                self._observation_waypoint = None
        self._observation_demand = None

    def _observation_waypoint_cb(self, msg):
        """Read the same token and seven-value pose consumed by the planner."""
        now = rospy.Time.now()
        with self._lock:
            self._observation_waypoint = None
            if msg is None or len(msg.data) != 7 or not all(
                    math.isfinite(v) for v in msg.data):
                return
            labels = [d.label for d in msg.layout.dim]
            target = re.fullmatch(r"care_obs_v1_(\d+)_[0-9a-f]{24}", labels[0]) if labels else None
            regions = [re.fullmatch(r"care_region_v1_(\d+)_[0-9a-f]{24}", label)
                       for label in labels if label.startswith("care_region_v1_")]
            if (target is None or len(regions) != 1 or regions[0] is None or
                    target[1] != regions[0][1] or labels[0] in self._observation_retired_tokens):
                return
            self._observation_waypoint = (labels[0], now)
            self._observation_waypoint_region = next(label for label in labels if label.startswith('care_region_v1_'))
            self._maybe_enter_observation_repair_locked(now, "waypoint")
            if self.state == self.REPAIR and hasattr(self, '_observation_handoff'):
                self._observation_handoff.bind(labels[0],self._observation_waypoint_region,
                    (self._occlusion_goal,self.repair_entry_count))
            self._maybe_release_matched_observation_locked(now)

    def _observation_acquisition_cb(self, msg):
        now = rospy.Time.now()
        with self._lock:
            pairs = _TOKEN_RE.findall(msg.data or "") if msg is not None else []
            f = dict(pairs)
            if len(pairs) != len(f) or f.get("policy") != "visibility_acquisition":
                self._observation_demand = None
                return
            if getattr(self, 'matched_observation_release', False):
                self._observation_release_identity.observe(f, now.to_sec())
                if self._maybe_release_matched_observation_locked(now):
                    return
            if getattr(self, 'task_first_research', False):
                count = _as_int(f.get('seen_obligation_count'), 0)
                previous = self.task_first_seen_count
                self.task_first_seen_count = max(previous, count)
                points = _as_int(f.get('acquired_point_count'), 0)
                previous_points = self.task_first_seen_point_count
                self.task_first_seen_point_count = max(previous_points, points)
                if (not getattr(self, 'matched_observation_release', False) and
                        (count > previous or points > previous_points) and
                        self.state == self.REPAIR and self.execution_ready):
                    self.task_first_observation_returns += 1
                    # Pending regions remain UNKNOWN and can be rediscovered by
                    # the next task prefix. Completion does not require q_vis.
                    self._observation_release_waypoint = self._observation_waypoint[0] if self._observation_waypoint else 'none'
                    self._observation_release_gain_kind = 'point_gain' if points > previous_points else 'region_completion'
                    self._observation_release_snapshot = f.get("acquisition_snapshot_id", "none")
                    self._observation_release_stamp_ns = now.to_nsec()
                    self._transition_locked(self.PROBE_NORMAL,
                        'task_first_observation_acquired', now)
            if f.get("complete") == "1" and f.get("remaining_obligation_count") == "0":
                demand = self._observation_demand
                # A short observation may finish before a false Bool is
                # received inside REPAIR. The fresh, ordered acquisition
                # stream already supplies pending->complete for the exact
                # demand that entered REPAIR. Preserve that completion edge.
                completed_takeover = (
                    f.get("started") == "1" and demand is not None and
                    self.state == self.REPAIR and self.execution_ready and
                    self.last_transition_reason.startswith("task_visibility_obligation_ready_") and
                    demand[0] == self.observation_takeover_token and
                    0. <= (now-demand[1]).to_sec() <= self.input_timeout and
                    (self._observation_waypoint is None or
                     self._observation_waypoint[0] == demand[0]))
                self._retire_observation_demand_locked()
                if completed_takeover:
                    self.repair_completion = True
                    self.repair_completion_time = now
                    self.repair_completion_armed = True
                    self.repair_completion_event_count += 1
                    self._observation_release_snapshot = f.get("acquisition_snapshot_id", "none")
                    self._observation_release_stamp_ns = now.to_nsec()
                    self._transition_locked(self.PROBE_NORMAL,
                        "actual_visibility_acquisition_complete", now)
                return
            self._observation_demand = None
            token = f.get("active_q_vis_token", "")
            target = re.fullmatch(r"care_obs_v1_(\d+)_[0-9a-f]{24}", token)
            if (f.get("started") != "1" or f.get("complete") != "0" or
                    f.get("active_seen") != "0" or target is None or
                    f.get("active_obligation_id") != target[1] or
                    _as_int(f.get("remaining_obligation_count")) <= 0 or
                    _as_int(f.get("active_point_count")) <= 0 or
                    token in self._observation_retired_tokens):
                return
            self._observation_demand = (token, now)
            self._maybe_enter_observation_repair_locked(now, "acquisition")
            self._maybe_release_matched_observation_locked(now)

    def _maybe_release_matched_observation_locked(self, now):
        if (not getattr(self, 'matched_observation_release', False) or
                not self.task_first_research or self.state != self.REPAIR or not self.execution_ready):
            return False
        if os.environ.get('CARE_ROLLING_TASK_OBSERVATION') == 'true' and hasattr(self,'_observation_handoff'):
            event=self._observation_handoff.take(self._observation_release_identity,
                (self._occlusion_goal,self.repair_entry_count),self._observation_retired_tokens,now.to_sec())
            token=event['token'] if event is not None else None
        else:
            if self._observation_waypoint is None:return False
            token,_received=self._observation_waypoint
            event=self._observation_release_identity.take(token,self._observation_retired_tokens,now.to_sec())
        if event is None:
            return
        self.task_first_observation_returns += 1
        self._observation_release_waypoint = token
        self._observation_release_gain_kind = 'matched_point_gain'
        self._observation_release_snapshot = event['acquisition_snapshot_id']
        if os.environ.get('CARE_ROLLING_TASK_OBSERVATION') == 'true':
            self._rolling_release = event.copy()
        self._observation_release_stamp_ns = now.to_nsec()
        self._transition_locked(self.PROBE_NORMAL, 'task_first_observation_acquired', now)
        return True

    def _maybe_enter_observation_repair_locked(self, now, source, probe_boundary=False):
        """A live observation is a planning request in either arrival order.

        An in-flight PROBE remains single-flight. Its exact matched prefix
        callback may yield to sensing; all three prefixes are still required
        before returning to NORMAL without an observation.
        """
        if os.environ.get('CARE_ACTIVE_OBSERVATION', 'true') == 'false':
            return False
        if (getattr(self, 'repair_goal_stopping', False) and self.state == self.NORMAL and
                getattr(self, '_goal_stop_execution', 0) > 0 and
                self._goal_stop_execution == self.tracker_execution_stamp_ns and
                self._goal_stop_execution != self.last_tracker_aborted_stamp_ns and
                self._goal_stop_feedback_time is not None and
                0 <= (now-self._goal_stop_feedback_time).to_sec() <= self.input_timeout):
            return False

        if (getattr(self, "task_first_research", False) or
                getattr(self, "research_opportunistic_probe", False) or
                not getattr(self, "repair_completion_gate_enabled", False) or
                not self.execution_ready or not self.visibility_waypoint_active or
                self.state not in (self.NORMAL, self.PROBE_NORMAL)):
            return False
        waypoint = self._observation_waypoint
        demand = self._observation_demand
        if (waypoint is None or demand is None or waypoint[0] != demand[0] or
                not self._fresh(waypoint[1], now, self.input_timeout) or
                not self._fresh(demand[1], now, self.input_timeout)):
            return False
        if (self.state == self.PROBE_NORMAL and not probe_boundary and
                (self.pending_probe_candidate_seq > 0 or
                 self.pending_probe_execution_stamp_ns > 0 or
                 self._probe_single_flight_busy_locked())):
            return False
        self.observation_takeover_count += 1
        self.observation_takeover_token = demand[0]
        if self.task_stall_stamp_ns:
            self._finish_task_stall_locked("active_visibility_obligation_ready")
        self._transition_locked(self.REPAIR,
            "task_visibility_obligation_ready_" + source, now)
        return True

    def _visibility_waypoint_active_cb(self, msg):
        if msg is None:
            return
        value = bool(msg.data)
        now = rospy.Time.now()
        with self._lock:
            self.visibility_waypoint_active = value
            if self._maybe_enter_observation_repair_locked(now, "active"):
                return
            # Target generation may finish after SAME-SCP VBC SAFE retired
            # rediscovery. That verdict rules out a new blocker on the stalled
            # query; it does not cancel a real observation obligation being
            # generated from another certified blocker. Honor the same live
            # waypoint authority whether it arrives before or after SAFE.
            # Keep reset, gate, and in-flight PROBE boundaries intact. This
            # transition only requests planning; it authorizes no execution.
            if (value and self.execution_ready and self.task_stall_stamp_ns > 0 and
                    self.task_stall_outcome == "QP_NO_PROGRESS_NO_VBC_BLOCKER" and
                    self.state in (self.NORMAL, self.PROBE_NORMAL) and
                    not (self.state == self.PROBE_NORMAL and
                         self._probe_single_flight_busy_locked())):
                self._finish_task_stall_locked("qp_stall_late_visibility_obligation")
                self._transition_locked(self.REPAIR, self.task_stall_outcome, now)
                return
            if (self.blocker_rediscovery_origin == "task_qp_no_progress" and
                    not self.task_stall_confirmed):
                return  # An old waypoint cannot certify this stalled query.
            if (not value or not self.blocker_rediscovery_pending or
                    not self.execution_ready or
                    self.state not in (self.NORMAL, self.PROBE_NORMAL)):
                return

            origin = self.blocker_rediscovery_origin
            if origin == "task_qp_no_progress":
                self.task_stall_outcome = "qp_stall_exact_blocker_obligation_ready"
                self.task_stall_deadline = None
            force_bootstrap = self.blocker_rediscovery_force_bootstrap
            self.blocker_rediscovery_pending = False
            self.blocker_rediscovery_origin = "none"
            self.blocker_rediscovery_force_bootstrap = False
            if force_bootstrap:
                self.force_vbc_bootstrap_pub.publish(Bool(data=False))
            if "uncertified" in origin:
                self.task_uncertified_repair_entry_count += 1
            elif "infeasible" in origin:
                self.task_infeasible_repair_entry_count += 1
            self._transition_locked(
                self.REPAIR,
                "{}_visibility_obligation_ready".format(origin),
                now)

    def _candidate_vbc_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        # C5.30: solver-failure rediscovery is valid only on the latest
        # Sparse-SCP query trajectory: the same trajectory that generated the
        # local-GCDF batch. Task-bootstrap verdicts are not comparable here.
        if f.get("trajectory_source") != "scp_rediscovery":
            return
        has_violation = _as_bool(f.get("has_violation"))
        if has_violation is None:
            return

        request_replan = False
        clear_bootstrap = False
        with self._lock:
            if (not self.blocker_rediscovery_pending or
                    not self.blocker_rediscovery_force_bootstrap or
                    self.state not in (self.NORMAL, self.PROBE_NORMAL)):
                return

            if self.blocker_rediscovery_origin == "task_qp_no_progress":
                if (_as_int(f.get("trajectory_stamp_ns"), 0) != self.task_stall_stamp_ns or
                        not _as_bool(f.get("success"))):
                    return
                if not has_violation:
                    self.blocker_rediscovery_vbc_safe_count += 1
                    # A real visibility waypoint may already be active when
                    # the task QP stalls.  In that case the same-SCP VBC SAFE
                    # result only says that the failed nominal trajectory did
                    # not add a *new* swept-body blocker; it does not remove
                    # the existing UNKNOWN obligation.  Keep the active
                    # obligation as the recovery authority and enter REPAIR.
                    # Without an active waypoint, preserve the original
                    # conservative behavior and remain in NORMAL.
                    if self.visibility_waypoint_active and self.execution_ready:
                        self.task_stall_outcome = (
                            "qp_stall_existing_visibility_obligation")
                        self.task_stall_deadline = None
                        self.blocker_rediscovery_pending = False
                        self.blocker_rediscovery_origin = "none"
                        self.blocker_rediscovery_force_bootstrap = False
                        self._transition_locked(
                            self.REPAIR,
                            self.task_stall_outcome,
                            rospy.Time.now())
                        return
                    self._finish_task_stall_locked("QP_NO_PROGRESS_NO_VBC_BLOCKER")
                    return  # VBC SAFE is not a hard-QP feasibility certificate.
                self.task_stall_confirmed = True
                self.blocker_rediscovery_vbc_unsafe_count += 1
                self.task_stall_outcome = "qp_stall_exact_blocker_wait_obligation"
                if self.visibility_waypoint_active and self.execution_ready:
                    self._finish_task_stall_locked("qp_stall_exact_blocker_obligation_ready")
                    self._transition_locked(self.REPAIR, self.task_stall_outcome, rospy.Time.now())
                return

            if has_violation:
                self.blocker_rediscovery_vbc_unsafe_count += 1
                self.last_transition_reason = (
                    "blocker_rediscovery_confirmed_wait_visibility_waypoint")
                return

            # A fresh SAME-SCP-trajectory VBC SAFE verdict means this solver
            # failure is not attributable to a low-confidence swept volume on
            # the exact GCDF linearization. Stay in the current task regime
            # (NORMAL or PROBE_NORMAL) and replan; do not manufacture REPAIR.
            self.blocker_rediscovery_vbc_safe_count += 1
            self.blocker_rediscovery_pending = False
            self.blocker_rediscovery_origin = "none"
            self.blocker_rediscovery_force_bootstrap = False
            self.last_transition_reason = (
                "no_visibility_blocker_after_solver_failure_replan")
            request_replan = True
            clear_bootstrap = True

        if clear_bootstrap:
            self.force_vbc_bootstrap_pub.publish(Bool(data=False))
        if request_replan:
            self.replan_request_pub.publish(Bool(data=True))

    def _probe_single_flight_busy_locked(self):
        return self.probe_single_flight_phase in ("VERIFYING", "EXECUTING")

    def _probe_single_flight_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        phase = f.get("phase", "IDLE")
        if phase not in ("IDLE", "VERIFYING", "EXECUTING"):
            return
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        reason = f.get("reason", "none")
        with self._lock:
            self.probe_single_flight_phase = phase
            self.probe_single_flight_execution_stamp_ns = execution_stamp_ns
            self.probe_single_flight_reason = reason
            self.probe_single_flight_summary_time = rospy.Time.now()
            self._maybe_enter_observation_repair_locked(
                self.probe_single_flight_summary_time, "probe_idle")

    def _task_infeasible_cb(self, msg):
        if msg is None or not bool(msg.data):
            return
        now = rospy.Time.now()
        with self._lock:
            # NORMAL infeasibility starts the first sensing episode. A
            # PROBE_NORMAL infeasibility means the just-expanded confidence map
            # is still insufficient for the task, so it starts the next sensing
            # episode instead of leaving the state machine stuck in PROBE_NORMAL.
            if self.state not in (self.NORMAL, self.PROBE_NORMAL):
                return

            if not self.execution_ready:
                if self.state == self.NORMAL:
                    self.task_infeasible_pending = True
                    self.last_transition_reason = "task_planner_infeasible_pending"
                return

            self.task_infeasible_pending = False
            self.task_uncertified_pending = False

            if self.state == self.PROBE_NORMAL:
                if self._probe_single_flight_busy_locked():
                    # A prior PROBE decision already owns the downstream
                    # verification/execution flight. A later planner attempt
                    # may fail, but that stale failure must not cancel or
                    # reinterpret the certified flight currently in progress.
                    self.probe_failure_suppressed_busy_count += 1
                    self.probe_infeasible_suppressed_busy_count += 1
                    self.last_transition_reason = (
                        "task_probe_infeasible_suppressed_single_flight_{}".format(
                            self.probe_single_flight_phase.lower()))
                    self._publish_summary()
                    return
                self.probe_failure_count += 1
                # Local hard-GCDF publishes exact UNKNOWN blocker evidence for
                # PROBE exactly as for NORMAL. Wait for that obligation rather
                # than letting completion-gate configuration decide whether a
                # sensing target exists.
                if not self.visibility_waypoint_active:
                    self._begin_blocker_rediscovery_locked(
                        "task_probe_infeasible",
                        force_bootstrap=False)
                else:
                    self.task_infeasible_repair_entry_count += 1
                    self._transition_locked(
                        self.REPAIR,
                        "task_probe_infeasible_gcdf_obligation_ready",
                        now)
            else:
                # C5.43: local hard-GCDF now publishes its exact selected
                # UNKNOWN voxels directly into the GCDF recovery/sensing
                # channel. Do not ask the narrower VBC swept-body selector to
                # re-judge those points. If the obligation already arrived,
                # enter REPAIR immediately; otherwise wait fail-closed for the
                # direct GCDF obligation callback.
                if self.visibility_waypoint_active:
                    self.task_infeasible_repair_entry_count += 1
                    self._transition_locked(
                        self.REPAIR,
                        "task_normal_infeasible_gcdf_obligation_ready",
                        now)
                else:
                    self._begin_blocker_rediscovery_locked(
                        "task_normal_infeasible",
                        force_bootstrap=False)

    def _task_obstacle_blocked_cb(self, msg):
        """Known OCCUPIED geometry is a planning problem, never a sensing one."""
        if msg is None or not bool(msg.data):
            return
        now = rospy.Time.now()
        with self._lock:
            self.task_obstacle_blocked_count += 1

            if not self.execution_ready:
                self.task_obstacle_blocked_pending = True
                self.last_transition_reason = "task_obstacle_blocked_pending"
                return

            self.task_obstacle_blocked_pending = False

            if self.state == self.PROBE_NORMAL and self._probe_single_flight_busy_locked():
                self.probe_failure_suppressed_busy_count += 1
                self.last_transition_reason = (
                    "task_obstacle_blocked_suppressed_single_flight_{}".format(
                        self.probe_single_flight_phase.lower()))
                return

            # Stay in the current regime. In NORMAL this asks LocalSparseSCP to
            # find a collision-free task detour; in PROBE it retries the current
            # task probe without manufacturing a visibility obligation.
            self.task_obstacle_replan_count += 1
            self.last_transition_reason = (
                "task_obstacle_blocked_{}_replan".format(
                    self.state.lower()))
            self.replan_request_pub.publish(Bool(data=True))

    def _task_uncertified_cb(self, msg):
        if msg is None or not bool(msg.data):
            return
        now = rospy.Time.now()
        with self._lock:
            if self.task_stall_stamp_ns:
                return  # Late numerical failure must not restart stalled work.
            # "Uncertified" is deliberately distinct from mathematical
            # infeasibility. It means PIQP exhausted its iteration budget and
            # therefore did not provide a hard-GCDF-certified task candidate.
            # C5.11 no longer equates this numerical failure with a sensing
            # request: PROBE fails closed and asks VBC to rediscover a blocker.
            # Only a real visibility obligation is allowed to enter REPAIR.
            if self.state not in (self.NORMAL, self.PROBE_NORMAL):
                return

            if not self.execution_ready:
                if self.state == self.NORMAL:
                    self.task_uncertified_pending = True
                    self.last_transition_reason = "task_planner_uncertified_pending"
                return

            self.task_uncertified_pending = False
            self.task_infeasible_pending = False

            if self.state == self.PROBE_NORMAL:
                if self._probe_single_flight_busy_locked():
                    self.probe_failure_suppressed_busy_count += 1
                    self.probe_uncertified_suppressed_busy_count += 1
                    self.last_transition_reason = (
                        "task_probe_uncertified_suppressed_single_flight_{}".format(
                            self.probe_single_flight_phase.lower()))
                    self._publish_summary()
                    return
                self.probe_failure_count += 1
                if (self.repair_completion_gate_enabled and
                        not self.visibility_waypoint_active):
                    self._begin_blocker_rediscovery_locked(
                        "task_probe_uncertified")
                else:
                    self.task_uncertified_repair_entry_count += 1
                    self._transition_locked(
                        self.REPAIR, "task_probe_uncertified", now)
            else:
                # NORMAL numerical non-certification is not itself an
                # active-sensing request. Keep executing the previously
                # certified trajectory/fail-safe hold and retry from the latest
                # measured state. A genuine final-GCDF/VBC unsafe result or
                # task infeasibility still enters REPAIR through its own path.
                self.last_transition_reason = (
                    "task_planner_uncertified_replan")
                self.replan_request_pub.publish(Bool(data=True))

    def _deadline_cb(self, msg):
        if msg is None:
            return
        value = float(msg.data)
        if not math.isfinite(value) or value <= 0.0:
            return
        with self._lock:
            self.physical_deadline_abs = value
        self.deadline_pub.publish(Float64(data=value))

    def _committed_trajectory_cb(self, msg):
        with self._lock:
            self.last_committed_trajectory_time = rospy.Time.now()
            # The direct committed-trajectory audit retains the commit stamp.
            stamp = int(msg.header.stamp.to_nsec())
            if stamp > 0:
                self._remember_execution_audit_identity_locked(stamp, stamp)
                self._drain_execution_audits_locked(rospy.Time.now())

    def _repair_completion_cb(self, msg):
        if msg is None or not self.repair_completion_gate_enabled:
            return
        value = bool(msg.data)
        now = rospy.Time.now()
        with self._lock:
            self.repair_completion = value
            self.repair_completion_time = now
            if self.state != self.REPAIR or not self.execution_ready:
                return
            if not value:
                self.repair_completion_armed = True
                return
            if value and self.repair_completion_armed:
                self.repair_completion_event_count += 1
                self._transition_locked(
                    self.PROBE_NORMAL, "actual_visibility_acquisition_complete", now)

    def _handle_normal_candidate_result_locked(
            self, result, safety_gate, now, replay=False):
        """Apply NORMAL candidate safety semantics from live or replayed input.

        C5.22: ordinary VBC unsafe uses hysteresis, but every rejected candidate
        before the threshold must explicitly request a fresh NORMAL plan.
        Otherwise no candidate is committed and there may be no later event to
        advance candidate_unsafe_streak from 1 to the configured threshold.
        Final-GCDF unsafe remains fail-closed and transitions immediately.
        """
        source = "after_gate_release" if replay else "live"
        if result == "unsafe" and safety_gate == "gcdf_occupied":
            # Known physical geometry cannot be repaired by sensing.
            self.gcdf_occupied_replan_count += 1
            self.last_transition_reason = (
                "final_gcdf_occupied_normal_replan_{}".format(source))
            self.replan_request_pub.publish(Bool(data=True))
            return

        if (result == "unsafe" and
                safety_gate in ("gcdf", "gcdf_unknown", "gcdf_mixed")):
            self.gcdf_repair_entry_count += 1
            self._transition_locked(
                self.REPAIR,
                "final_{}_normal_unsafe_{}".format(safety_gate, source),
                now)
            return

        if result == "unsafe":
            if self.candidate_unsafe_streak >= self.candidate_unsafe_required:
                self.normal_unsafe_confirmed_repair_count += 1
                self._transition_locked(
                    self.REPAIR,
                    "candidate_unique_unsafe_confirmed_{}".format(source),
                    now)
                return

            # Hysteresis not yet confirmed: stay NORMAL, but the rejected
            # candidate produced no commit/tracker-complete trigger. Request a
            # fresh plan from the latest measured state so a subsequent safe
            # result can clear the streak or a second unsafe can confirm REPAIR.
            self.normal_unsafe_retry_count += 1
            self.last_transition_reason = (
                "candidate_unsafe_retry_{}/{}_{}".format(
                    self.candidate_unsafe_streak,
                    self.candidate_unsafe_required,
                    source))
            self.replan_request_pub.publish(Bool(data=True))
            return

        if result == "timeout":
            # No candidate was certified or committed. Stay in NORMAL and
            # explicitly request a fresh plan; never reuse stale execution.
            self.last_transition_reason = (
                "{}_normal_timeout_replan_{}".format(
                    safety_gate, source))
            self.replan_request_pub.publish(Bool(data=True))

    def _candidate_outcome_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        seq = _as_int(f.get("seq"), 0)
        result = f.get("result", "")
        committed = _as_bool(f.get("committed"))
        verification_view = f.get("verification_view", "none")
        safety_gate = f.get("safety_gate", "vbc")
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        probe_effective_prefix_s = _as_float(
            f.get("probe_effective_prefix_s"), math.nan)
        if seq <= 0 or result not in ("safe", "unsafe", "timeout", "protocol_invalid"):
            return

        now = rospy.Time.now()
        with self._lock:
            if seq <= self.last_candidate_seq:
                return
            self.last_candidate_seq = seq
            self.last_candidate_result = result
            self.last_candidate_safety_gate = safety_gate
            self.last_candidate_verification_view = verification_view
            self.candidate_outcome_time = now
            self.candidate_event_count += 1

            if result == 'protocol_invalid':
                # No geometry verdict: preserve unsafe/direction failure memory
                # and mode budgets. The uncertified candidate never executes.
                self.candidate_protocol_invalid_count += 1
                self.last_transition_reason = 'candidate_vbc_protocol_invalid_wait'
                # This request has terminated without a commit or a future
                # tracker completion. One fresh-plan request per new outcome
                # restores progress without changing geometric failure memory,
                # the current execution token, mode or task budget.
                if self.execution_ready:
                    self.last_transition_reason = 'candidate_vbc_protocol_invalid_replan'
                    self.replan_request_pub.publish(Bool(data=True))
                return

            if result == "unsafe":
                self.candidate_last_unsafe = True
                self.candidate_unsafe_count += 1
                self.candidate_unsafe_streak += 1
            elif result == "safe":
                self.candidate_last_unsafe = False
                self.candidate_safe_count += 1
                self.candidate_unsafe_streak = 0
            else:
                self.candidate_timeout_count += 1
                self.candidate_unsafe_streak = 0

            # A candidate verdict applies only to the rejected, uncommitted
            # motion. It cannot relabel the current execution or override its
            # tokenized stop. Execution VBC stops require the explicit audit
            # identity checked by _drain_execution_audits_locked; missing or
            # stale execution evidence remains covered by the watchdog.

            if not self.execution_ready:
                # Startup race protection: preserve the safety fact and replay
                # it when execution_ready rises.  The old code recorded the
                # unsafe result then returned forever if no later candidate
                # event arrived.
                self.candidate_outcome_pending_gate_replay = True
                self.last_transition_reason = (
                    "candidate_outcome_pending_gate_replay")
                return

            self.candidate_outcome_pending_gate_replay = False
            if self.state == self.NORMAL:
                self._handle_normal_candidate_result_locked(
                    result, safety_gate, now, replay=False)
                return

            if self.state == self.REPAIR:
                if result == "unsafe":
                    if getattr(self, 'repair_final_vbc_hold', False):
                        self.last_transition_reason = "repair_final_vbc_no_progress_hold"
                        return
                    # Final executable GCDF/VBC rejection means no trajectory
                    # was committed, so there will be no tracker-complete event
                    # to drive another REPAIR plan. Stay in REPAIR and explicitly
                    # request a fresh local plan.
                    self.last_transition_reason = "repair_candidate_unsafe_replan"
                    self.replan_request_pub.publish(Bool(data=True))
                    return
                if result == "timeout":
                    self.last_transition_reason = "repair_candidate_timeout_replan"
                    self.replan_request_pub.publish(Bool(data=True))
                    return
                if result == "safe" and committed is True:
                    self.last_commit_count += 1
                    self.repair_safe_commit_count += 1
                    self.last_commit_event_time = now
                    if self.repair_completion_gate_enabled and not self.research_opportunistic_probe:
                        # C4.7: executing a VBC-safe detour is progress toward
                        # seeing. Stay in REPAIR until real confidence confirms it.
                        self.last_transition_reason = (
                            "safe_repair_commit_continue_visibility_acquisition")
                    else:
                        self._transition_locked(
                            self.PROBE_NORMAL, "safe_repair_commit", now)
                return

            if self.state == self.PROBE_NORMAL:
                # A stale REPAIR verification may arrive just after the regime
                # transition. Only a candidate explicitly built as a PROBE
                # executable prefix may arm the PROBE execution handshake.
                if (result == "safe" and committed is True and
                        verification_view == "probe_prefix_brake_hold"):
                    if execution_stamp_ns <= 0:
                        self.pending_probe_candidate_seq = 0
                        self.pending_probe_execution_stamp_ns = 0
                        self.pending_probe_effective_prefix_s = math.nan
                        self.last_transition_reason = (
                            "probe_commit_missing_execution_token_replan")
                        self.replan_request_pub.publish(Bool(data=True))
                        return
                    self.last_commit_count += 1
                    self.last_commit_event_time = now
                    self.pending_probe_candidate_seq = seq
                    self.pending_probe_execution_stamp_ns = execution_stamp_ns
                    self.pending_probe_effective_prefix_s = (
                        probe_effective_prefix_s
                        if (math.isfinite(probe_effective_prefix_s) and
                            probe_effective_prefix_s > 0.0)
                        else math.nan)
                    self.last_transition_reason = (
                        "safe_probe_commit_wait_prefix_stamp_{}".format(
                            execution_stamp_ns))
                    return

                if result == "unsafe":
                    # A current PROBE prefix has an unambiguous view tag and
                    # must never be hidden by the short post-transition ignore
                    # window. The window only suppresses stale non-PROBE events.
                    if (verification_view != "probe_prefix_brake_hold" and
                            self.probe_ignore_until is not None and
                            now < self.probe_ignore_until):
                        return
                    self.pending_probe_candidate_seq = 0
                    self.pending_probe_execution_stamp_ns = 0
                    self.pending_probe_effective_prefix_s = math.nan

                    if safety_gate == "gcdf_occupied":
                        # The probe candidate hit a known physical obstacle.
                        # Stay in PROBE and ask for another collision-free task
                        # probe; never manufacture a visibility obligation.
                        self.gcdf_occupied_replan_count += 1
                        self.last_transition_reason = (
                            "final_gcdf_occupied_probe_replan")
                        self.replan_request_pub.publish(Bool(data=True))
                        return

                    # Only visibility/VBC failures count as active-sensing
                    # PROBE failures. A known obstacle is tracked separately.
                    self.probe_failure_count += 1

                    origin = (
                        "final_gcdf_probe_unsafe"
                        if safety_gate in
                        ("gcdf", "gcdf_unknown", "gcdf_mixed")
                        else "candidate_probe_unique_unsafe")

                    # C5.12 invariant:
                    #   PROBE -> REPAIR only after q_vis/obligation is active.
                    # Exact VBC already emitted an active set. Final GCDF now
                    # emits its own earliest rejected low-confidence voxels.
                    if (self.repair_completion_gate_enabled and
                            not self.visibility_waypoint_active):
                        self._begin_blocker_rediscovery_locked(
                            origin, force_bootstrap=False)
                    else:
                        self._transition_locked(
                            self.REPAIR, origin, now)
                    return

                if (result == "timeout" and
                        verification_view == "probe_prefix_brake_hold"):
                    # Timeout is not evidence that active sensing is required;
                    # stay in PROBE, fail closed, and request a fresh candidate.
                    self.pending_probe_candidate_seq = 0
                    self.pending_probe_execution_stamp_ns = 0
                    self.pending_probe_effective_prefix_s = math.nan
                    self.last_transition_reason = "probe_candidate_timeout_replan"
                    self.replan_request_pub.publish(Bool(data=True))
                    return

    def _tracker_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        active = _as_bool(f.get("active"))
        complete = _as_bool(f.get("complete"))
        seq = _as_int(f.get("seq"), 0)
        execution_stamp_ns = _as_int(f.get("execution_stamp_ns"), 0)
        source = f.get("source", "")
        phase_s = _as_float(f.get("phase_s"), math.nan)
        aborted = _as_bool(f.get("execution_aborted"))

        now = rospy.Time.now()
        request_next_probe = False
        with self._lock:
            # Tracker liveness/ownership is the execution freshness signal in
            # the single-publish architecture. A committed topic message is no
            # longer expected to refresh while a trajectory is running.
            self.tracker_summary_time = now
            self.tracker_active = bool(active) if active is not None else False
            self.tracker_complete = bool(complete) if complete is not None else False
            self.tracker_execution_stamp_ns = execution_stamp_ns
            self.tracker_source = source
            self.tracker_execution_state = f.get("execution_state", "UNKNOWN")
            if (execution_stamp_ns > 0 and
                    execution_stamp_ns !=
                    self.vbc_unsafe_execution_stamp_ns):
                self.vbc_unsafe_execution_stamp_ns = 0
                self.execution_vbc_hold_tokenized = False
            self._drain_execution_audits_locked(now)

            if (aborted is True and execution_stamp_ns > 0 and
                    execution_stamp_ns != self.last_tracker_aborted_stamp_ns):
                self.tracker_abort_count += 1
                self.last_tracker_aborted_stamp_ns = execution_stamp_ns

            # Preserve full-completion diagnostics independently of PROBE
            # progress counting.
            full_complete = (
                complete is True and execution_stamp_ns > 0 and
                source == "trajectory_complete_hold")
            if full_complete:
                self.last_tracker_complete_seq = seq
                if execution_stamp_ns != self.last_tracker_complete_stamp_ns:
                    self.last_tracker_complete_stamp_ns = execution_stamp_ns

            if self.state != self.PROBE_NORMAL:
                return
            if self.pending_probe_execution_stamp_ns <= 0:
                return
            if execution_stamp_ns != self.pending_probe_execution_stamp_ns:
                # Stable timestamp matching rejects stale status from an older
                # committed trajectory.
                return

            # Tracking protection cancels the committed token. It cannot count
            # as a completed PROBE prefix, even if the stop was reported at a
            # phase equal to the prefix duration. Clear the stale handshake and
            # ask the planner for a fresh candidate from measured state.
            if aborted is True:
                self.pending_probe_candidate_seq = 0
                self.pending_probe_execution_stamp_ns = 0
                self.pending_probe_effective_prefix_s = math.nan
                self.probe_completed_prefix_streak = 0
                self.last_transition_reason = (
                    "probe_execution_aborted_token_{}_replan".format(
                        execution_stamp_ns))
                request_next_probe = True
                # Do not evaluate prefix/full completion for a cancelled token.
                self._publish_summary()
            else:
                prefix_complete = (
                    math.isfinite(self.pending_probe_effective_prefix_s) and
                    self.pending_probe_effective_prefix_s > 0.0 and
                    math.isfinite(phase_s) and
                    phase_s + 1e-6 >= self.pending_probe_effective_prefix_s)

                # Metadata-missing fallback preserves the old fail-closed behavior:
                # only a full certified trajectory completion can count.
                if not prefix_complete and not full_complete:
                    return

                completed_stamp = self.pending_probe_execution_stamp_ns
                self.pending_probe_candidate_seq = 0
                self.pending_probe_execution_stamp_ns = 0
                self.pending_probe_effective_prefix_s = math.nan
                self.probe_completed_prefix_streak += 1
                # Compatibility counter: this now means an actually executed,
                # certified PROBE prefix (full completion remains separately logged
                # through last_tracker_complete_*).
                self.probe_completed_execution_count += 1
                release = self._rolling_release
                if release is not None and os.environ.get('CARE_ROLLING_TASK_OBSERVATION') == 'true':
                    self._rolling_request_pub.publish(String(data=json.dumps(dict(
                        observation_token=release['token'],
                        region_token=release['point_gain_event']['region_token'],
                        acquisition_snapshot_id=release['acquisition_snapshot_id'],
                        point_gain_event=release['point_gain_event'],
                        execution_stamp_ns=str(completed_stamp),
                        consumed_after_actual_task_prefix=True),separators=(',',':'))))
                    self._rolling_release = None

                if self._maybe_enter_observation_repair_locked(
                        now, "probe_prefix_complete", probe_boundary=True):
                    return

                if self.probe_completed_prefix_streak >= self.probe_completed_prefixes_required:
                    # A new visibility obligation may arrive while the last
                    # probe prefix is in flight. Do not pass through NORMAL in
                    # that case: the task QP would treat the fresh blocker as
                    # an ordinary UNKNOWN row and spend one cycle rediscovering
                    # what the acquisition layer already reported.
                    if (self.visibility_waypoint_active and
                            not self.repair_completion_gate_enabled and
                            not self.research_opportunistic_probe):
                        self._transition_locked(
                            self.REPAIR,
                            "probe_prefixes_complete_visibility_obligation",
                            now)
                    else:
                        self._transition_locked(
                            self.NORMAL, "probe_normal_completed_prefixes", now)
                else:
                    self.last_transition_reason = (
                        "probe_certified_prefix_complete_stamp_{}".format(
                            completed_stamp))
                    # Start the next measured-state PROBE now. The old committed
                    # trajectory remains safe through its certified brake+hold tail
                    # until the replacement itself passes final GCDF + exact VBC.
                    request_next_probe = True

        if request_next_probe:
            self.replan_request_pub.publish(Bool(data=True))

    def _remember_execution_audit_identity_locked(self, audit, execution):
        previous = self.execution_audit_identities.get(audit)
        if previous is not None and previous != execution:
            # An ambiguous origin cannot authorize an execution decision.
            self.execution_audit_identities[audit] = 0
        else:
            self.execution_audit_identities[audit] = execution
        while len(self.execution_audit_identities) > 256:
            del self.execution_audit_identities[next(iter(self.execution_audit_identities))]

    def _execution_audit_identity_cb(self, msg):
        if msg is None:
            return
        pairs = _TOKEN_RE.findall(msg.data or "")
        f = dict(pairs)
        if (len(pairs) != 2 or set(f) != {
                "audit_trajectory_stamp_ns", "execution_stamp_ns"}):
            return
        values = [f["audit_trajectory_stamp_ns"], f["execution_stamp_ns"]]
        if any(not v.isdigit() or not 0 < int(v) < 2**64 for v in values):
            return
        with self._lock:
            self._remember_execution_audit_identity_locked(*map(int, values))
            self._drain_execution_audits_locked(rospy.Time.now())

    def _drain_execution_audits_locked(self, now):
        # Identity and verdict travel on separate ROS connections. Retain a
        # bounded pending set so either arrival order has the same result.
        for stamp, (f, received) in list(self.pending_execution_audits.items()):
            if not self._fresh(received, now, self.input_timeout):
                del self.pending_execution_audits[stamp]
                continue
            execution = self.execution_audit_identities.get(stamp, 0)
            if (not execution or execution != self.tracker_execution_stamp_ns or
                    self.last_committed_trajectory_time is None):
                continue
            del self.pending_execution_audits[stamp]
            if execution != self.last_execution_audit_token:
                self.last_execution_audit_token = execution
                self.last_execution_audit_stamp_ns = 0
                self.execution_unsafe_streak = 0
                self.execution_event_latched = False
            if stamp < self.last_execution_audit_stamp_ns:
                continue
            self.last_execution_audit_stamp_ns = stamp
            self.execution_summary_time = received
            unsafe = _as_bool(f.get("has_violation")) is True
            self.execution_last_unsafe = unsafe
            if unsafe:
                self.execution_unsafe_streak += 1
                if self.tracker_active:
                    self.vbc_unsafe_execution_stamp_ns = execution
                    self.execution_vbc_hold_tokenized = True
                    self.execution_vbc_stop_pub.publish(String(data=(
                        "execution_stamp_ns={}".format(execution))))
            else:
                self.execution_safe_count += 1
                self.execution_unsafe_streak = 0
                self.execution_event_latched = False
            if (self.execution_ready and unsafe and
                    self.execution_unsafe_streak >= self.execution_unsafe_required and
                    not self.execution_event_latched):
                self.execution_event_latched = True
                self.execution_safety_event_count += 1
                if self.state != self.REPAIR:
                    self._transition_locked(
                        self.REPAIR, "execution_committed_unsafe", now)

    def _execution_summary_cb(self, msg):
        if msg is None:
            return
        f = _tokens(msg.data)
        if (f.get("trajectory_source") != "predicted" or
                f.get("success") != "1"):
            return
        unsafe = _as_bool(f.get("has_violation"))
        if unsafe is None:
            return
        now = rospy.Time.now()
        with self._lock:
            stamp = _as_int(f.get("trajectory_stamp_ns"), 0)
            if stamp <= 0:
                return
            if unsafe or stamp not in self.pending_execution_audits:
                self.pending_execution_audits[stamp] = (f, now)
            while len(self.pending_execution_audits) > 64:
                del self.pending_execution_audits[next(iter(self.pending_execution_audits))]
            self._drain_execution_audits_locked(now)

    @staticmethod
    def _fresh(stamp, now, timeout):
        if stamp is None:
            return False
        age = (now - stamp).to_sec()
        return 0.0 <= age <= timeout

    def _timer_cb(self, _event):
        now = rospy.Time.now()
        with self._lock:
            self._pump_qp_retry_locked()
            self._maybe_enter_observation_repair_locked(now, "timer")
            if (self.task_stall_deadline is not None and
                    time.monotonic() >= self.task_stall_deadline):
                self._finish_task_stall_locked(
                    "QP_NO_PROGRESS_OBLIGATION_TIMEOUT" if self.task_stall_confirmed
                    else "QP_NO_PROGRESS_EXACT_DIAGNOSTIC_TIMEOUT")
            trigger = self.execution_ready and self.state == self.REPAIR
            probe_active = self.execution_ready and self.state == self.PROBE_NORMAL
            clear = (
                self.execution_ready and self.state == self.PROBE_NORMAL and
                self.clear_until is not None and now <= self.clear_until)

            exec_summary_fresh = self._fresh(
                self.execution_summary_time, now, self.input_timeout)
            tracker_summary_fresh = self._fresh(
                self.tracker_summary_time, now, self.input_timeout)
            startup_grace = (
                self.execution_ready_time is not None and
                (now - self.execution_ready_time).to_sec() < self.input_timeout)
            execution_in_progress = bool(
                self.tracker_active and self.tracker_execution_stamp_ns > 0)
            hold = bool(
                execution_in_progress and
                ((self.execution_ready and not startup_grace and
                  (not exec_summary_fresh or not tracker_summary_fresh)) or
                 (self.vbc_unsafe_execution_stamp_ns > 0 and
                  not getattr(self, "execution_vbc_hold_tokenized", False) and
                  self.tracker_execution_stamp_ns ==
                  self.vbc_unsafe_execution_stamp_ns)))
            deadline = self.physical_deadline_abs

        self.trigger_pub.publish(Bool(data=trigger))
        self.probe_active_pub.publish(Bool(data=probe_active))
        self.clear_pub.publish(Bool(data=clear))
        self.hold_pub.publish(Bool(data=hold))
        if deadline is not None:
            self.deadline_pub.publish(Float64(data=deadline))
        self._publish_summary()

    def _publish_summary(self):
        with self._lock:
            now = rospy.Time.now()

            def age(stamp):
                if stamp is None:
                    return math.nan
                return max(0.0, (now - stamp).to_sec())

            msg = String()
            msg.data = " ".join([
                "state={}".format(self.state),
                "active_observation={}".format(int(os.environ.get('CARE_ACTIVE_OBSERVATION', 'true') != 'false')),
                "passive_observation_suppressed={}".format(getattr(self, 'passive_observation_suppressed', 0)),
                "goal_stop_execution_stamp_ns={}".format(getattr(self, '_goal_stop_execution', 0)),
                "reason={}".format(self.last_transition_reason),
                "execution_ready={}".format(int(self.execution_ready)),
                "qp_retry_pending={}".format(int(bool(self.qp_retry_token))),
                "qp_retry_token={}".format(self.qp_retry_token or "none"),
                "qp_retry_sends={}".format(self.qp_retry_send_count),
                "qp_retry_acks={}".format(self.qp_retry_ack_count),
                "qp_retry_status={}".format(self.qp_retry_last_status),
                "last_candidate_seq={}".format(self.last_candidate_seq),
                "last_candidate_result={}".format(self.last_candidate_result),
                "last_candidate_safety_gate={}".format(
                    self.last_candidate_safety_gate),
                "last_candidate_verification_view={}".format(
                    self.last_candidate_verification_view),
                "candidate_outcome_pending_gate_replay={}".format(
                    int(self.candidate_outcome_pending_gate_replay)),
                "candidate_gate_replay_count={}".format(
                    self.candidate_gate_replay_count),
                "candidate_event_count={}".format(self.candidate_event_count),
                "candidate_protocol_invalid_count={}".format(self.candidate_protocol_invalid_count),
                "candidate_last_unsafe={}".format(int(self.candidate_last_unsafe)),
                "candidate_unsafe_streak={}".format(self.candidate_unsafe_streak),
                "candidate_unsafe_count={}".format(self.candidate_unsafe_count),
                "candidate_safe_count={}".format(self.candidate_safe_count),
                "candidate_timeout_count={}".format(self.candidate_timeout_count),
                "normal_unsafe_retry_count={}".format(
                    self.normal_unsafe_retry_count),
                "normal_unsafe_confirmed_repair_count={}".format(
                    self.normal_unsafe_confirmed_repair_count),
                "execution_last_unsafe={}".format(int(self.execution_last_unsafe)),
                "execution_unsafe_streak={}".format(self.execution_unsafe_streak),
                "execution_event_latched={}".format(int(self.execution_event_latched)),
                "execution_safety_event_count={}".format(self.execution_safety_event_count),
                "tracker_active={}".format(int(self.tracker_active)),
                "tracker_complete={}".format(int(self.tracker_complete)),
                "tracker_execution_stamp_ns={}".format(
                    self.tracker_execution_stamp_ns),
                "vbc_unsafe_execution_stamp_ns={}".format(
                    self.vbc_unsafe_execution_stamp_ns),
                "execution_vbc_hold_tokenized={}".format(int(self.execution_vbc_hold_tokenized)),
                "last_execution_audit_token={}".format(self.last_execution_audit_token),
                "last_execution_audit_stamp_ns={}".format(self.last_execution_audit_stamp_ns),
                "vbc_unsafe_execution_hold={}".format(int(
                    self.tracker_active and
                    self.vbc_unsafe_execution_stamp_ns > 0 and
                    self.tracker_execution_stamp_ns ==
                    self.vbc_unsafe_execution_stamp_ns)),
                "tracker_source={}".format(self.tracker_source),
                "tracker_abort_count={}".format(self.tracker_abort_count),
                "last_tracker_aborted_stamp_ns={}".format(
                    self.last_tracker_aborted_stamp_ns),
                "tracker_execution_state={}".format(self.tracker_execution_state),
                "repair_entry_count={}".format(self.repair_entry_count),
                "candidate_repair_entry_count={}".format(self.candidate_repair_entry_count),
                "gcdf_repair_entry_count={}".format(self.gcdf_repair_entry_count),
                "gcdf_occupied_replan_count={}".format(
                    self.gcdf_occupied_replan_count),
                "execution_repair_entry_count={}".format(self.execution_repair_entry_count),
                "task_infeasible_repair_entry_count={}".format(
                    self.task_infeasible_repair_entry_count),
                "task_infeasible_pending={}".format(
                    int(self.task_infeasible_pending)),
                "task_obstacle_blocked_pending={}".format(
                    int(self.task_obstacle_blocked_pending)),
                "task_obstacle_blocked_count={}".format(
                    self.task_obstacle_blocked_count),
                "task_obstacle_replan_count={}".format(
                    self.task_obstacle_replan_count),
                "task_uncertified_repair_entry_count={}".format(
                    self.task_uncertified_repair_entry_count),
                "task_uncertified_pending={}".format(
                    int(self.task_uncertified_pending)),
                "probe_entry_count={}".format(self.probe_entry_count),
                "probe_failure_count={}".format(self.probe_failure_count),
                "probe_single_flight_phase={}".format(
                    self.probe_single_flight_phase),
                "probe_single_flight_execution_stamp_ns={}".format(
                    self.probe_single_flight_execution_stamp_ns),
                "probe_single_flight_reason={}".format(
                    self.probe_single_flight_reason),
                "probe_failure_suppressed_busy_count={}".format(
                    self.probe_failure_suppressed_busy_count),
                "probe_infeasible_suppressed_busy_count={}".format(
                    self.probe_infeasible_suppressed_busy_count),
                "probe_uncertified_suppressed_busy_count={}".format(
                    self.probe_uncertified_suppressed_busy_count),
                "probe_completed_prefix_streak={}".format(
                    self.probe_completed_prefix_streak),
                # Compatibility alias for older result parsers.
                "probe_safe_commit_streak={}".format(
                    self.probe_completed_prefix_streak),
                "probe_completed_prefixes_required={}".format(
                    self.probe_completed_prefixes_required),
                "probe_completed_execution_count={}".format(
                    self.probe_completed_execution_count),
                "pending_probe_candidate_seq={}".format(
                    self.pending_probe_candidate_seq),
                "pending_probe_execution_stamp_ns={}".format(
                    self.pending_probe_execution_stamp_ns),
                "pending_probe_effective_prefix_s={}".format(
                    "nan" if not math.isfinite(
                        self.pending_probe_effective_prefix_s)
                    else "{:.6f}".format(
                        self.pending_probe_effective_prefix_s)),
                "last_tracker_complete_seq={}".format(
                    self.last_tracker_complete_seq),
                "last_tracker_complete_stamp_ns={}".format(
                    self.last_tracker_complete_stamp_ns),
                "normal_entry_count={}".format(self.normal_entry_count),
                "normal_task_replan_count={}".format(
                    self.normal_task_replan_count),
                "commit_count={}".format(self.last_commit_count),
                "repair_safe_commit_count={}".format(self.repair_safe_commit_count),
                "repair_completion_gate_enabled={}".format(
                    int(self.repair_completion_gate_enabled)),
                "repair_completion={}".format(int(self.repair_completion)),
                "observation_return_policy={}".format('matched' if getattr(self, 'matched_observation_release', False) else 'legacy'),
                "observation_release_waypoint={}".format(getattr(self, '_observation_release_waypoint', 'none')),
                "observation_release_gain_kind={}".format(getattr(self, '_observation_release_gain_kind', 'none')),
                "observation_return_observed_gains={}".format(getattr(getattr(self, '_observation_release_identity', None), 'observed_gains', 0)),
                "observation_return_unbound_gains={}".format(getattr(getattr(self, '_observation_release_identity', None), 'unbound_gains', 0)),
                "observation_return_expired_gains={}".format(getattr(getattr(self, '_observation_release_identity', None), 'expired_gains', 0)),
                "observation_return_retired_gains={}".format(getattr(getattr(self, '_observation_release_identity', None), 'retired_gains', 0)),
                "observation_return_released_gains={}".format(getattr(getattr(self, '_observation_release_identity', None), 'released_gains', 0)),
                "observation_release_snapshot={}".format(getattr(self, '_observation_release_snapshot', 'none')),
                "observation_release_stamp_ns={}".format(getattr(self, '_observation_release_stamp_ns', 0)),
                "repair_completion_armed={}".format(
                    int(self.repair_completion_armed)),
                "repair_completion_event_count={}".format(
                    self.repair_completion_event_count),
                "visibility_waypoint_active={}".format(
                    int(self.visibility_waypoint_active)),
                "observation_takeover_count={}".format(self.observation_takeover_count),
                "observation_takeover_token={}".format(self.observation_takeover_token),
                "task_stall_outcome={}".format(self.task_stall_outcome),
                "task_stall_stamp_ns={}".format(self.task_stall_stamp_ns),
                "blocker_rediscovery_pending={}".format(
                    int(self.blocker_rediscovery_pending)),
                "blocker_rediscovery_origin={}".format(
                    self.blocker_rediscovery_origin),
                "blocker_rediscovery_force_bootstrap={}".format(
                    int(self.blocker_rediscovery_force_bootstrap)),
                "blocker_rediscovery_count={}".format(
                    self.blocker_rediscovery_count),
                "blocker_rediscovery_vbc_unsafe_count={}".format(
                    self.blocker_rediscovery_vbc_unsafe_count),
                "blocker_rediscovery_vbc_safe_count={}".format(
                    self.blocker_rediscovery_vbc_safe_count),
                "candidate_outcome_age_s={:.6f}".format(
                    age(self.candidate_outcome_time)),
                "execution_summary_age_s={:.6f}".format(
                    age(self.execution_summary_time)),
                "tracker_summary_age_s={:.6f}".format(
                    age(self.tracker_summary_time)),
                "committed_trajectory_age_s={:.6f}".format(
                    age(self.last_committed_trajectory_time)),
                "repair_completion_age_s={:.6f}".format(
                    age(self.repair_completion_time)),
            ])
        self.summary_pub.publish(msg)


def main():
    rospy.init_node("c4_4_verified_regime_manager")
    C44VerifiedRegimeManager()
    rospy.spin()


if __name__ == "__main__":
    main()
