#!/usr/bin/env python3
"""C4.7 visibility-acquisition repair semantics.

Key semantic correction
-----------------------
A deadline inherited from the nominal/task candidate is meaningful while judging
that candidate, but it is NOT a hard clock that a repair detour must preserve.
Once CAREPlanner enters REPAIR, nominal tracking is intentionally abandoned.  A
repair candidate may delay/avoid the original low-confidence sweep, so the old
nominal deadline is no longer a planning constraint.

C4.7 therefore treats VBC violations as persistent visibility obligations:

    O_r = (spatial region R_r, learned visible configuration q_vis^r)

No nominal absolute deadline is attached to the REPAIR objective.  The MPC uses
its legacy terminal q_vis repair objective and may need multiple receding-horizon
safe commits to reach q_vis.  Exact VBC still audits every repair candidate using
THAT candidate's own sweep times.  Thus the candidate is allowed to:

  * see a low-confidence point before it sweeps it, OR
  * delay/avoid that sweep while moving safely toward a visibility pose.

A safe repair candidate is only safe to EXECUTE; it does not mean the visibility
obligation is finished.  Obligations are removed only when the real confidence
map reports that all points in the region have actually been seen.  Only after
all obligations are observed does the node publish acquisition_complete=True,
which permits the regime manager to PROBE_NORMAL.
"""

from __future__ import annotations

import math
import threading
from typing import Dict, List

import numpy as np
import rospy
from care_confidence_map.srv import QueryConfidenceRequest
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float64, String

from vbc_deadline_waypoint_node import _vector_msg
from vbc_multi_deadline_obligation_impl import AccumulatedMultiDeadlineWaypointNode


class VisibilityAcquisitionWaypointNode(AccumulatedMultiDeadlineWaypointNode):
    """Persistent visibility goals; clear only on actual confidence acquisition."""

    def __init__(self) -> None:
        # Must exist before parent ROS subscribers/timer can call virtual methods.
        self._c47_ready = False
        self._acquisition_complete = False
        self._acquisition_started = False
        self._seen_obligation_count = 0
        self._last_visibility_check_s = -math.inf
        self._safe_repair_candidate_count = 0
        self._visibility_episode_reopen_count = 0
        # Several ROS subscriber callbacks and the rospy.Timer may all request
        # schedule publication. Serialize the publication transaction so
        # q_vis/q_zero/deadline messages from two snapshots cannot interleave.
        self._schedule_publish_lock = threading.RLock()
        self._timer_exception_count = 0
        super().__init__()

        self.visibility_check_rate = float(rospy.get_param(
            "~visibility_check_rate", 20.0))
        self.required_seen_fraction = float(rospy.get_param(
            "~required_seen_fraction", 1.0))
        self.acquisition_complete_topic = str(rospy.get_param(
            "~acquisition_complete_topic",
            "/care_planner/active_sensing/visibility_acquisition_complete"))
        self.acquisition_summary_topic = str(rospy.get_param(
            "~acquisition_summary_topic",
            "/care_planner/active_sensing/visibility_acquisition_summary"))

        if self.visibility_check_rate <= 0.0:
            raise ValueError("~visibility_check_rate must be positive")
        if not 0.0 < self.required_seen_fraction <= 1.0:
            raise ValueError("~required_seen_fraction must be in (0,1]")

        self.acquisition_complete_pub = rospy.Publisher(
            self.acquisition_complete_topic, Bool, queue_size=1, latch=True)
        self.acquisition_summary_pub = rospy.Publisher(
            self.acquisition_summary_topic, String, queue_size=1, latch=True)
        self.acquisition_complete_pub.publish(Bool(data=False))

        self._c47_ready = True
        rospy.logwarn(
            "[vbc_acquisition] C4.7 ENABLED: nominal deadlines ignored in REPAIR; "
            "obligations clear only from actual confidence, seen_threshold=%.3f "
            "required_fraction=%.2f",
            self.seen_threshold, self.required_seen_fraction)

    def _active_set_callback(self, msg) -> None:
        """A fresh unsafe active set starts a new acquisition episode.

        C4.7 completion is latched True after all current obligations are seen.
        If VBC later discovers a new non-empty unsafe active set, that True must
        no longer describe the current world state.  Reopen the episode
        immediately (False) while q_vis generation catches up.  The regime
        manager still waits for visibility_waypoint_active before entering
        REPAIR, so this edge cannot create a repair without a steering target.
        """
        super()._active_set_callback(msg)
        if msg is None or len(getattr(msg, "data", [])) == 0:
            return

        with self._obligation_lock:
            no_obligations = len(self._obligations) == 0
        if self._acquisition_complete and no_obligations:
            self._acquisition_started = True
            self._acquisition_complete = False
            self._visibility_episode_reopen_count += 1
            if hasattr(self, "acquisition_complete_pub"):
                self.acquisition_complete_pub.publish(Bool(data=False))
            rospy.logwarn(
                "[vbc_acquisition] NEW UNSAFE ACTIVE SET reopens acquisition "
                "episode count=%d while waiting for q_vis obligation",
                self._visibility_episode_reopen_count)

    # ------------------------------------------------------------------
    # Candidate VBC verdict semantics
    # ------------------------------------------------------------------
    def _candidate_vbc_summary_callback(self, msg: String) -> None:
        """Never clear a visibility goal merely because one candidate is safe."""
        if msg is None:
            return
        fields = dict(self._tokenize_summary(str(msg.data)))
        if fields.get("trajectory_source") != "predicted":
            return
        if fields.get("has_violation") == "0":
            self._safe_repair_candidate_count += 1
            rospy.loginfo_throttle(
                0.5,
                "[vbc_acquisition] predicted candidate SAFE, but visibility "
                "obligations persist until actual confidence confirms seen")

    @staticmethod
    def _tokenize_summary(text):
        # Avoid another module-level regex dependency.
        import re
        return re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", text or "")

    # ------------------------------------------------------------------
    # Actual confidence completion
    # ------------------------------------------------------------------
    def _query_region_seen(self, ob: Dict[str, object]):
        points = np.asarray(ob["points"], dtype=np.float64).reshape(-1, 3)
        if points.shape[0] == 0:
            return (
                False, 0.0, math.nan, math.nan, math.nan, 0,
                np.asarray([math.nan, math.nan, math.nan], dtype=np.float64),
                math.nan)
        req = QueryConfidenceRequest()
        req.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                      for p in points]
        try:
            res = self.confidence_client(req)
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0, "[vbc_acquisition] confidence query failed: %s", exc)
            return (
                False, 0.0, math.nan, math.nan, math.nan, int(points.shape[0]),
                np.asarray([math.nan, math.nan, math.nan], dtype=np.float64),
                math.nan)
        n = points.shape[0]
        if (len(res.confidence) != n or len(res.inside_map) != n or
                len(res.current_visibility) != n):
            return (
                False, 0.0, math.nan, math.nan, math.nan, int(n),
                np.asarray([math.nan, math.nan, math.nan], dtype=np.float64),
                math.nan)

        conf = np.asarray(res.confidence, dtype=np.float64)
        inside = np.asarray(res.inside_map, dtype=bool)
        current_vis = np.asarray(res.current_visibility, dtype=np.float64)
        finite_inside = inside & np.isfinite(conf)
        good = finite_inside & (conf >= self.seen_threshold)
        fraction = float(np.mean(good)) if n else 0.0
        seen = fraction + 1e-12 >= self.required_seen_fraction

        min_conf = (
            float(np.min(conf[finite_inside]))
            if np.any(finite_inside) else math.nan)
        mean_conf = (
            float(np.mean(conf[finite_inside]))
            if np.any(finite_inside) else math.nan)
        max_vis = (
            float(np.nanmax(current_vis))
            if current_vis.size else math.nan)

        worst_point = np.asarray(
            [math.nan, math.nan, math.nan], dtype=np.float64)
        if np.any(finite_inside):
            valid_indices = np.where(finite_inside)[0]
            worst_idx = int(valid_indices[np.argmin(conf[valid_indices])])
            worst_point = points[worst_idx].copy()

        q_dist_inf = math.nan
        measured = self._latest_measured_q
        q_vis = np.asarray(ob.get("q_vis", []), dtype=np.float64).reshape(-1)
        if (measured is not None and measured.shape == (7,) and
                q_vis.shape == (7,) and np.all(np.isfinite(q_vis))):
            q_dist_inf = float(np.max(np.abs(measured - q_vis)))

        return (
            bool(seen), fraction, min_conf, mean_conf, max_vis, int(n),
            worst_point, q_dist_inf)

    def _update_actual_visibility_completion(self) -> None:
        if not self._c47_ready:
            return
        now_s = rospy.Time.now().to_sec()
        if now_s - self._last_visibility_check_s < 1.0 / self.visibility_check_rate:
            return
        self._last_visibility_check_s = now_s

        with self._obligation_lock:
            snapshot = [dict(ob) for ob in self._obligations]
        if not snapshot:
            # Before the first violation/acquisition episode, do not advertise a
            # spurious completion pulse. Once an episode has started, empty means
            # every real visibility obligation has been seen.
            complete = bool(self._acquisition_started)
            self._acquisition_complete = complete
            self.acquisition_complete_pub.publish(Bool(data=complete))
            self._publish_acquisition_summary([])
            return

        self._acquisition_started = True
        seen_ids: List[int] = []
        diagnostics = []
        for ob in snapshot:
            (seen, frac, min_conf, mean_conf, max_vis, point_count,
             worst_point, q_dist_inf) = self._query_region_seen(ob)
            oid = int(ob["id"])
            diagnostics.append((
                oid, seen, point_count, frac, min_conf, mean_conf, max_vis,
                q_dist_inf, np.asarray(worst_point, dtype=np.float64).copy()))
            if seen:
                seen_ids.append(oid)

        if seen_ids:
            with self._obligation_lock:
                before = len(self._obligations)
                self._obligations = [
                    ob for ob in self._obligations
                    if int(ob["id"]) not in set(seen_ids)]
                removed = before - len(self._obligations)
            self._seen_obligation_count += removed
            rospy.logwarn(
                "[vbc_acquisition] ACTUAL VISIBILITY acquired obligations=%s "
                "removed=%d remaining=%d",
                ":".join(str(i) for i in seen_ids), removed,
                len(self._ordered_obligations()))
            self._publish_schedule()

        remaining = self._ordered_obligations()
        complete = bool(self._acquisition_started and not remaining)
        self._acquisition_complete = complete
        self.acquisition_complete_pub.publish(Bool(data=complete))
        self._publish_acquisition_summary(diagnostics)

    def _publish_acquisition_summary(self, diagnostics) -> None:
        remaining = self._ordered_obligations()
        msg = String()
        diag_text = ";".join(
            (
                f"{oid}:{int(seen)}:{point_count}:{frac:.3f}:"
                f"{min_conf:.3f}:{mean_conf:.3f}:{max_vis:.3f}:"
                f"{q_dist_inf:.4f}:"
                f"{float(worst_point[0]):.3f},"
                f"{float(worst_point[1]):.3f},"
                f"{float(worst_point[2]):.3f}"
            )
            for (oid, seen, point_count, frac, min_conf, mean_conf, max_vis,
                 q_dist_inf, worst_point) in diagnostics)

        if diagnostics:
            (active_oid, active_seen, active_point_count, active_frac,
             active_min_conf, active_mean_conf, active_max_vis,
             active_q_dist_inf, active_worst_point) = diagnostics[0]
        else:
            active_oid = -1
            active_seen = False
            active_point_count = 0
            active_frac = math.nan
            active_min_conf = math.nan
            active_mean_conf = math.nan
            active_max_vis = math.nan
            active_q_dist_inf = math.nan
            active_worst_point = np.asarray(
                [math.nan, math.nan, math.nan], dtype=np.float64)

        measured = self._latest_measured_q
        measured_text = (
            ",".join(f"{float(v):.4f}" for v in measured)
            if measured is not None and measured.shape == (7,)
            else "none")
        active_q_vis_text = "none"
        if remaining:
            q_vis = np.asarray(
                remaining[0].get("q_vis", []), dtype=np.float64).reshape(-1)
            if q_vis.shape == (7,) and np.all(np.isfinite(q_vis)):
                active_q_vis_text = ",".join(
                    f"{float(v):.4f}" for v in q_vis)

        msg.data = (
            "policy=visibility_acquisition"
            f" started={int(self._acquisition_started)}"
            f" complete={int(self._acquisition_complete)}"
            f" remaining_obligation_count={len(remaining)}"
            f" remaining_ids={':'.join(str(int(ob['id'])) for ob in remaining) or 'none'}"
            f" seen_obligation_count={self._seen_obligation_count}"
            f" safe_repair_candidate_count={self._safe_repair_candidate_count}"
            f" timer_exception_count={self._timer_exception_count}"
            f" episode_reopen_count={self._visibility_episode_reopen_count}"
            f" active_obligation_id={active_oid}"
            f" active_seen={int(active_seen)}"
            f" active_point_count={active_point_count}"
            f" active_seen_fraction={active_frac:.6f}"
            f" active_min_confidence={active_min_conf:.6f}"
            f" active_mean_confidence={active_mean_conf:.6f}"
            f" active_max_current_visibility={active_max_vis:.6f}"
            f" active_q_distance_inf={active_q_dist_inf:.6f}"
            f" active_worst_point_xyz="
            f"{float(active_worst_point[0]):.4f},"
            f"{float(active_worst_point[1]):.4f},"
            f"{float(active_worst_point[2]):.4f}"
            f" measured_q={measured_text}"
            f" active_q_vis={active_q_vis_text}"
            f" region_checks={diag_text or 'none'}"
        )
        self.acquisition_summary_pub.publish(msg)

    # ------------------------------------------------------------------
    # REPAIR steering publication: no nominal-deadline schedule
    # ------------------------------------------------------------------
    def _ordered_obligations(self):
        # Stable discovery order.  We do not reorder from hypothetical rejected
        # candidates.  The current target remains until it is actually seen.
        with self._obligation_lock:
            return sorted(
                [dict(ob) for ob in self._obligations],
                key=lambda ob: int(ob["id"]))

    def _publish_schedule(self) -> None:
        """Publish one coherent visibility-acquisition snapshot.

        Shared parent state is updated only while holding self._lock. All ROS
        messages are then published from local immutable copies created by this
        call; publication must never re-read a shared member after unlock.

        The legacy target callback may reset _deadline_abs_s to None concurrently
        when x* changes. Re-reading that member after unlock is the race that
        previously killed the timer thread. _schedule_publish_lock also prevents
        two ROS callback threads from interleaving the q_vis/q_zero/deadline
        triplet from different obligation snapshots.
        """
        if not getattr(self, "_c47_ready", False):
            return

        with self._schedule_publish_lock:
            obligations = self._ordered_obligations()
            if obligations:
                first = obligations[0]

                q_vis_snapshot = np.asarray(
                    first["q_vis"], dtype=np.float64).reshape(7).copy()
                q_zero_snapshot = np.asarray(
                    first["q_zero"], dtype=np.float64).reshape(7).copy()
                deadline_snapshot = float(rospy.Time.now().to_sec() + 1.0)
                final_f_snapshot = float(first["final_f_min"])

                with self._lock:
                    self._q_vis = q_vis_snapshot.copy()
                    self._q_zero = q_zero_snapshot.copy()
                    # Plumbing timestamp only. REPAIR does not optimize against
                    # this as a nominal task deadline.
                    self._deadline_abs_s = deadline_snapshot
                    self._deadline_from_start_s = math.nan
                    self._generation_success = True
                    self._shared_min_f = final_f_snapshot
                    self._shared_solution_mode = "c47_visibility_acquisition"
                    self._summary = "c47_acquisition_goal_ready"

                self.waypoint_pub.publish(_vector_msg(q_vis_snapshot))
                self.zero_pub.publish(_vector_msg(q_zero_snapshot))
                d = Float64()
                d.data = deadline_snapshot
                self.deadline_pub.publish(d)
            else:
                with self._lock:
                    self._q_vis = None
                    self._q_zero = None
                    self._deadline_abs_s = None
                    self._deadline_from_start_s = math.nan
                    self._generation_success = False
                    self._summary = "c47_no_visibility_obligations"

    def _maybe_generate(self) -> None:
        if not self._c47_ready:
            return
        # Reuse C4.6 accumulation/matching, but never clear from predicted SAFE.
        self._process_new_active_set()
        self._update_actual_visibility_completion()
        self._publish_schedule()

    def _timer_callback(self, _event) -> None:
        """Fail closed without permanently killing rospy's timer thread."""
        try:
            self._maybe_generate()
            self._publish_state()
        except Exception as exc:
            self._timer_exception_count += 1
            self._acquisition_complete = False
            try:
                self.acquisition_complete_pub.publish(Bool(data=False))
            except Exception:
                pass
            with self._lock:
                self._generation_success = False
                self._summary = "c47_timer_exception_fail_closed"
            rospy.logerr_throttle(
                1.0,
                "[vbc_acquisition] timer callback exception; keeping "
                "acquisition incomplete and retrying next cycle: %s",
                exc)

    def _publish_state(self) -> None:
        if not self._c47_ready:
            return

        # Use the same publication lock as _publish_schedule so the periodic
        # state publisher cannot interleave its waypoint/deadline pair with a
        # callback-driven schedule snapshot.
        with self._schedule_publish_lock:
            obligations = self._ordered_obligations()
            active = len(obligations) > 0
            a = Bool()
            a.data = active
            self.active_pub.publish(a)
            if obligations:
                q_snapshot = np.asarray(
                    obligations[0]["q_vis"], dtype=np.float64).reshape(7).copy()
                deadline_snapshot = float(rospy.Time.now().to_sec() + 1.0)
                self.waypoint_pub.publish(_vector_msg(q_snapshot))
                # Plumbing timestamp only; not a REPAIR deadline.
                d = Float64()
                d.data = deadline_snapshot
                self.deadline_pub.publish(d)

        msg = String()
        msg.data = (
            f"active={int(active)} seen={int(self._acquisition_complete)} "
            f"ready={int(active)} confidence=nan current_visibility=nan inside_map=1 "
            "deadline_from_start=nan deadline_remaining=nan "
            "reason=c47_visibility_acquisition "
            "steering_policy=visibility_acquisition "
            f"obligation_count={len(obligations)} "
            f"active_set_size={sum(len(ob['points']) for ob in obligations)} "
            "shared_solution_mode=c47_visibility_acquisition"
        )
        self.summary_pub.publish(msg)
        rospy.loginfo_throttle(0.5, "[vbc_acquisition] %s", msg.data)
