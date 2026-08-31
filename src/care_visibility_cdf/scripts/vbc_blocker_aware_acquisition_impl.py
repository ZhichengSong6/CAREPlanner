#!/usr/bin/env python3
"""C4.9 blocker-aware recursive visibility acquisition.

Builds on C4.7/C4.8 semantics:
  * obligations clear only after actual confidence confirms they were seen;
  * REPAIR may abandon nominal timing;
  * downstream exact VBC remains the only commit authority;
  * C4.8 may verify only executable prefix + brake + hold.

C4.9 adds blocker-aware target scheduling.  The current visibility obligation is
kept active until seen, except when the earliest VBC temporal layer of the current
repair motion exposes another urgent spatial region.  That region is pushed as a
nested blocker.  Once actual confidence clears it, the previous obligation is
resumed.  Spatial proximity alone never causes preemption.
"""

from __future__ import annotations

import math
import threading
from typing import Dict, List

import numpy as np
import rospy
from std_msgs.msg import Bool, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

from vbc_visibility_acquisition_impl import VisibilityAcquisitionWaypointNode


class BlockerAwareVisibilityAcquisitionWaypointNode(VisibilityAcquisitionWaypointNode):
    def __init__(self) -> None:
        self._c49_ready = False
        self._repair_stack: List[int] = []
        self._pending_blocker_id = None
        self._pending_blocker_count = 0
        self._stack_push_count = 0
        self._stack_pop_count = 0
        self._stack_cycle_block_count = 0
        self._last_active_layer_ids: List[int] = []
        self._last_active_layer_sweep_s = math.nan
        self._last_switch_reason = "startup"
        self._process_attempt_count = 0
        self._process_success_count = 0
        self._last_process_reason = "startup"

        # C5.12 direct final-GCDF recovery evidence.  A rejected executable
        # trajectory can expose low-confidence voxels that the task-bootstrap
        # VBC does not report (notably braking-tail sweep).  Keep this evidence
        # on a separate persistent channel and convert it into ordinary
        # visibility obligations using the same VisCDF projector.
        self._gcdf_recovery_lock = threading.Lock()
        self._gcdf_recovery_trajectory = None
        self._gcdf_recovery_trajectory_received = None
        self._pending_gcdf_recovery_event = None
        self._processed_gcdf_recovery_seq = 0
        self._gcdf_recovery_event_count = 0
        self._gcdf_recovery_generated_count = 0
        self._gcdf_recovery_match_count = 0
        self._gcdf_recovery_drop_count = 0
        self._last_gcdf_recovery_reason = "startup"
        super().__init__()

        self.blocker_push_max_sweep_s = float(rospy.get_param(
            "~blocker_push_max_sweep_s", 0.30))
        self.blocker_confirmations = int(rospy.get_param(
            "~blocker_confirmations", 2))
        self.blocker_stack_summary_topic = str(rospy.get_param(
            "~blocker_stack_summary_topic",
            "/care_planner/active_sensing/blocker_stack_summary"))
        self.gcdf_recovery_trajectory_topic = str(rospy.get_param(
            "~gcdf_recovery_trajectory_topic",
            "/care_planner/final_gcdf/recovery_trajectory"))
        self.gcdf_recovery_event_topic = str(rospy.get_param(
            "~gcdf_recovery_event_topic",
            "/care_planner/final_gcdf/recovery_visibility_event"))
        if self.blocker_push_max_sweep_s <= 0.0:
            raise ValueError("~blocker_push_max_sweep_s must be positive")
        if self.blocker_confirmations < 1:
            raise ValueError("~blocker_confirmations must be >= 1")

        self.blocker_stack_summary_pub = rospy.Publisher(
            self.blocker_stack_summary_topic, String, queue_size=1, latch=True)
        self.gcdf_recovery_trajectory_sub = rospy.Subscriber(
            self.gcdf_recovery_trajectory_topic, JointTrajectory,
            self._gcdf_recovery_trajectory_cb, queue_size=1)
        self.gcdf_recovery_event_sub = rospy.Subscriber(
            self.gcdf_recovery_event_topic, Float64MultiArray,
            self._gcdf_recovery_event_cb, queue_size=1)
        self._c49_ready = True
        self._prune_or_initialize_stack()
        self._publish_schedule()
        self._publish_blocker_stack_summary()
        rospy.logwarn(
            "[vbc_blocker_stack] C4.9 ENABLED max_blocker_sweep=%.3fs confirmations=%d",
            self.blocker_push_max_sweep_s, self.blocker_confirmations)

    def _gcdf_recovery_trajectory_cb(self, msg) -> None:
        if msg is None or not msg.points:
            return
        with self._gcdf_recovery_lock:
            self._gcdf_recovery_trajectory = msg
            self._gcdf_recovery_trajectory_received = rospy.Time.now()

    def _gcdf_recovery_event_cb(self, msg) -> None:
        if msg is None:
            return
        values = list(msg.data)
        if len(values) < 4:
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "malformed_short_event"
            return

        try:
            seq = int(round(float(values[0])))
            sweep_s = float(values[1])
            timestep = int(round(float(values[2])))
            count = int(round(float(values[3])))
        except Exception:
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "malformed_header"
            return

        if (seq <= 0 or not math.isfinite(sweep_s) or sweep_s < 0.0 or
                timestep < 0 or count <= 0 or len(values) != 4 + 3 * count):
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "malformed_shape"
            return

        points = np.asarray(values[4:], dtype=np.float64).reshape(count, 3)
        if not np.all(np.isfinite(points)):
            self._gcdf_recovery_drop_count += 1
            self._last_gcdf_recovery_reason = "nonfinite_points"
            return

        with self._gcdf_recovery_lock:
            if seq <= self._processed_gcdf_recovery_seq:
                self._last_gcdf_recovery_reason = "stale_processed_event"
                return
            self._pending_gcdf_recovery_event = {
                "seq": seq,
                "sweep_s": sweep_s,
                "timestep": timestep,
                "points": points.copy(),
            }
            self._gcdf_recovery_event_count += 1
            self._last_gcdf_recovery_reason = "event_pending"

        # A fresh hard-GCDF blocker defines a new acquisition episode even
        # before q_vis generation finishes.  This makes the current latched
        # completion state truthful while the regime manager remains fail-closed.
        with self._obligation_lock:
            no_obligations = len(self._obligations) == 0
        if self._acquisition_complete and no_obligations:
            self._acquisition_started = True
            self._acquisition_complete = False
            self._visibility_episode_reopen_count += 1
            self.acquisition_complete_pub.publish(Bool(data=False))
            rospy.logwarn(
                "[vbc_blocker_stack] FINAL-GCDF blocker reopens acquisition "
                "episode seq=%d count=%d sweep=%.3fs",
                seq, count, sweep_s)

    def _process_gcdf_recovery_event(self) -> None:
        with self._gcdf_recovery_lock:
            event = (
                None if self._pending_gcdf_recovery_event is None
                else dict(self._pending_gcdf_recovery_event))
            trajectory = self._gcdf_recovery_trajectory
            trajectory_received = self._gcdf_recovery_trajectory_received

        if event is None:
            return
        seq = int(event["seq"])
        if seq <= self._processed_gcdf_recovery_seq:
            with self._gcdf_recovery_lock:
                self._pending_gcdf_recovery_event = None
            self._last_gcdf_recovery_reason = "already_processed"
            return
        if (trajectory is None or not trajectory.points or
                trajectory_received is None):
            self._last_gcdf_recovery_reason = "waiting_recovery_trajectory"
            return
        if int(getattr(trajectory.header, "seq", 0)) != seq:
            self._last_gcdf_recovery_reason = "waiting_matching_recovery_trajectory"
            return
        if self._latest_measured_q is None:
            self._last_gcdf_recovery_reason = "waiting_measured_q"
            return

        points = np.asarray(event["points"], dtype=np.float64).reshape(-1, 3)
        sweep_s = float(event["sweep_s"])
        regions = self._cluster_regions(points)
        active_ids: List[int] = []
        new_ids: List[int] = []
        all_regions_handled = True

        for region in regions:
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is not None:
                    matched["last_seen_ros_s"] = rospy.Time.now().to_sec()
                    matched["points"] = np.asarray(
                        region["points"], dtype=np.float64).copy()
                    matched["keys"] = tuple(region["keys"])
                    matched["centroid"] = np.asarray(
                        region["centroid"], dtype=np.float64).copy()
                    active_ids.append(int(matched["id"]))
                    self._schedule_matched_obligations += 1
                    self._gcdf_recovery_match_count += 1
                    continue
                if len(self._obligations) >= self.max_obligations:
                    all_regions_handled = False
                    self._last_gcdf_recovery_reason = "max_obligations"
                    continue

            try:
                new_ob = self._generate_new_obligation(
                    region,
                    trajectory,
                    sweep_s,
                    trajectory_received,
                    "final_gcdf_rejected")
            except Exception as exc:
                self._schedule_generation_failures += 1
                all_regions_handled = False
                self._last_gcdf_recovery_reason = "generation_failed"
                rospy.logerr(
                    "[vbc_blocker_stack] final-GCDF obligation generation "
                    "failed seq=%d: %s", seq, exc)
                continue

            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is None and len(self._obligations) < self.max_obligations:
                    self._obligations.append(new_ob)
                    self._schedule_new_obligations += 1
                    oid = int(new_ob["id"])
                    active_ids.append(oid)
                    new_ids.append(oid)
                    self._gcdf_recovery_generated_count += 1
                    rospy.logwarn(
                        "[vbc_blocker_stack] FINAL-GCDF ADD obligation=%d "
                        "seq=%d points=%d sweep=%.3fs",
                        oid, seq, len(region["points"]), sweep_s)
                elif matched is not None:
                    active_ids.append(int(matched["id"]))
                    self._gcdf_recovery_match_count += 1

        if not all_regions_handled:
            return

        with self._gcdf_recovery_lock:
            self._processed_gcdf_recovery_seq = max(
                self._processed_gcdf_recovery_seq, seq)
            if (self._pending_gcdf_recovery_event is not None and
                    int(self._pending_gcdf_recovery_event["seq"]) == seq):
                self._pending_gcdf_recovery_event = None
        self._last_gcdf_recovery_reason = "handled"

        self._prune_or_initialize_stack()
        self._consider_active_layer(active_ids, new_ids, sweep_s)
        self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _prune_or_initialize_stack(self) -> None:
        with self._obligation_lock:
            existing = {int(ob["id"]) for ob in self._obligations}
            before = list(self._repair_stack)
            self._repair_stack = [oid for oid in self._repair_stack if oid in existing]
            self._stack_pop_count += max(0, len(before) - len(self._repair_stack))
            if not self._repair_stack and existing:
                root = min(existing)
                self._repair_stack.append(root)
                self._last_switch_reason = "initialize_oldest_obligation"

    def _ordered_obligations(self):
        with self._obligation_lock:
            copied = [dict(ob) for ob in self._obligations]
            stack = list(self._repair_stack)
        if not copied:
            return []
        by_id = {int(ob["id"]): ob for ob in copied}
        valid_stack = [oid for oid in stack if oid in by_id]
        active_id = valid_stack[-1] if valid_stack else min(by_id)
        first = by_id[active_id]
        rest = [ob for oid, ob in sorted(by_id.items()) if oid != active_id]
        return [first] + rest

    def _select_nearest_qvis_id(self, ids: List[int]):
        with self._obligation_lock:
            by_id = {int(ob["id"]): dict(ob) for ob in self._obligations}
        ids = [oid for oid in ids if oid in by_id]
        if not ids:
            return None
        q0 = None if self._latest_measured_q is None else np.asarray(
            self._latest_measured_q, dtype=np.float64)
        def key(oid):
            if q0 is None:
                return (math.inf, oid)
            qv = np.asarray(by_id[oid]["q_vis"], dtype=np.float64)
            return (float(np.linalg.norm(qv - q0)), oid)
        return min(ids, key=key)

    def _consider_active_layer(self, active_ids: List[int], new_ids: List[int], sweep_s: float) -> None:
        self._prune_or_initialize_stack()
        active_ids = sorted(set(int(v) for v in active_ids))
        new_ids = sorted(set(int(v) for v in new_ids))
        self._last_active_layer_ids = active_ids
        self._last_active_layer_sweep_s = float(sweep_s)

        with self._obligation_lock:
            current = self._repair_stack[-1] if self._repair_stack else None
            stack = list(self._repair_stack)

        if current is None or not active_ids:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return
        if not math.isfinite(sweep_s) or sweep_s > self.blocker_push_max_sweep_s:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return

        # Prefer genuinely new regions in the earliest urgent layer.  Otherwise
        # preempt only when the current target is absent from that earliest layer.
        candidates = [oid for oid in new_ids if oid != current]
        if not candidates and current not in active_ids:
            candidates = [oid for oid in active_ids if oid != current]
        if not candidates:
            self._pending_blocker_id = None
            self._pending_blocker_count = 0
            return

        blocker = self._select_nearest_qvis_id(candidates)
        if blocker is None:
            return
        if blocker in stack:
            # Do not create O1->O4->O1 cycles from rejected hypothetical plans.
            self._stack_cycle_block_count += 1
            self._last_switch_reason = "existing_stack_cycle_not_pushed"
            return

        if self._pending_blocker_id == blocker:
            self._pending_blocker_count += 1
        else:
            self._pending_blocker_id = blocker
            self._pending_blocker_count = 1
        if self._pending_blocker_count < self.blocker_confirmations:
            return

        with self._obligation_lock:
            existing = {int(ob["id"]) for ob in self._obligations}
            if blocker in existing and blocker not in self._repair_stack:
                self._repair_stack.append(blocker)
                self._stack_push_count += 1
                self._last_switch_reason = "urgent_earliest_layer_blocker"
                rospy.logwarn(
                    "[vbc_blocker_stack] PUSH blocker=%d sweep=%.3fs stack=%s",
                    blocker, sweep_s,
                    ":".join(str(v) for v in self._repair_stack))
        self._pending_blocker_id = None
        self._pending_blocker_count = 0

    def _process_new_active_set(self) -> None:
        if self._c49_ready:
            self._process_gcdf_recovery_event()
        if not self._c49_ready:
            self._last_process_reason = "not_ready"
            return
        with self._obligation_lock:
            serial = self._raw_active_set_serial
            processed = self._processed_active_set_serial
            if serial == processed:
                self._last_process_reason = "duplicate_serial"
                return
            raw = self._raw_active_set.copy()

        self._process_attempt_count += 1
        if raw.shape[0] == 0:
            with self._obligation_lock:
                self._processed_active_set_serial = max(
                    self._processed_active_set_serial, serial)
            self._last_process_reason = "empty_active_set"
            return

        with self._lock:
            sweep = self._sweep_time_s
            trajectory, trajectory_received, trajectory_source = (
                self._preferred_trajectory_locked())
        if sweep is None:
            self._last_process_reason = "waiting_sweep"
            return
        if trajectory is None:
            self._last_process_reason = "waiting_trajectory"
            return
        if self._latest_measured_q is None:
            self._last_process_reason = "waiting_measured_q"
            return

        regions = self._cluster_regions(raw)
        active_ids: List[int] = []
        new_ids: List[int] = []
        all_regions_handled = True
        for region in regions:
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is not None:
                    matched["last_seen_ros_s"] = rospy.Time.now().to_sec()
                    matched["points"] = np.asarray(region["points"], dtype=np.float64).copy()
                    matched["keys"] = tuple(region["keys"])
                    matched["centroid"] = np.asarray(region["centroid"], dtype=np.float64).copy()
                    oid = int(matched["id"])
                    active_ids.append(oid)
                    self._schedule_matched_obligations += 1
                    continue
                if len(self._obligations) >= self.max_obligations:
                    rospy.logerr_throttle(
                        1.0, "[vbc_blocker_stack] max_obligations=%d reached",
                        self.max_obligations)
                    continue
            try:
                new_ob = self._generate_new_obligation(
                    region, trajectory, float(sweep), trajectory_received,
                    trajectory_source)
            except Exception as exc:
                self._schedule_generation_failures += 1
                all_regions_handled = False
                rospy.logerr(
                    "[vbc_blocker_stack] obligation generation failed; retrying: %s", exc)
                continue
            with self._obligation_lock:
                matched = self._match_existing(region)
                if matched is None and len(self._obligations) < self.max_obligations:
                    self._obligations.append(new_ob)
                    self._schedule_new_obligations += 1
                    oid = int(new_ob["id"])
                    active_ids.append(oid)
                    new_ids.append(oid)
                elif matched is not None:
                    active_ids.append(int(matched["id"]))

        if all_regions_handled:
            with self._obligation_lock:
                self._processed_active_set_serial = max(
                    self._processed_active_set_serial, serial)
            self._process_success_count += 1
            self._last_process_reason = "handled"
        else:
            self._last_process_reason = "generation_failed"
        self._prune_or_initialize_stack()
        self._consider_active_layer(active_ids, new_ids, float(sweep))
        self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _update_actual_visibility_completion(self) -> None:
        super()._update_actual_visibility_completion()
        self._prune_or_initialize_stack()
        self._publish_schedule()
        self._publish_blocker_stack_summary()

    def _publish_schedule(self) -> None:
        super()._publish_schedule()
        if getattr(self, "_c49_ready", False):
            self._publish_blocker_stack_summary()

    def _publish_blocker_stack_summary(self) -> None:
        if not getattr(self, "_c49_ready", False) or not hasattr(
                self, "blocker_stack_summary_pub"):
            return
        with self._obligation_lock:
            existing = sorted(int(ob["id"]) for ob in self._obligations)
            stack = [oid for oid in self._repair_stack if oid in set(existing)]
            qvis_times = [
                float(ob.get("q_vis_generation_ms", math.nan))
                for ob in self._obligations
                if math.isfinite(float(ob.get("q_vis_generation_ms", math.nan)))
            ]
        msg = String()
        msg.data = (
            "policy=blocker_aware_recursive"
            f" current_target_id={(stack[-1] if stack else -1)}"
            f" stack={':'.join(str(v) for v in stack) or 'none'}"
            f" pending_ids={':'.join(str(v) for v in existing) or 'none'}"
            f" earliest_layer_ids={':'.join(str(v) for v in self._last_active_layer_ids) or 'none'}"
            f" earliest_layer_sweep_s={self._last_active_layer_sweep_s:.6f}"
            f" pending_blocker_id={self._pending_blocker_id if self._pending_blocker_id is not None else -1}"
            f" pending_blocker_count={self._pending_blocker_count}"
            f" push_count={self._stack_push_count}"
            f" pop_count={self._stack_pop_count}"
            f" cycle_block_count={self._stack_cycle_block_count}"
            f" process_attempt_count={self._process_attempt_count}"
            f" process_success_count={self._process_success_count}"
            f" raw_active_set_serial={self._raw_active_set_serial}"
            f" processed_active_set_serial={self._processed_active_set_serial}"
            f" process_reason={self._last_process_reason}"
            f" gcdf_recovery_event_count={self._gcdf_recovery_event_count}"
            f" gcdf_recovery_generated_count={self._gcdf_recovery_generated_count}"
            f" gcdf_recovery_match_count={self._gcdf_recovery_match_count}"
            f" gcdf_recovery_drop_count={self._gcdf_recovery_drop_count}"
            f" gcdf_recovery_processed_seq={self._processed_gcdf_recovery_seq}"
            f" gcdf_recovery_reason={self._last_gcdf_recovery_reason}"
            f" q_vis_generation_last_ms="
            f"{(qvis_times[-1] if qvis_times else math.nan):.3f}"
            f" q_vis_generation_max_ms="
            f"{(max(qvis_times) if qvis_times else math.nan):.3f}"
            f" switch_reason={self._last_switch_reason}"
        )
        self.blocker_stack_summary_pub.publish(msg)
