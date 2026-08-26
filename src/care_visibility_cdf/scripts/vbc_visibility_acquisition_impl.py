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
            return False, 0.0, math.nan, math.nan
        req = QueryConfidenceRequest()
        req.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                      for p in points]
        try:
            res = self.confidence_client(req)
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0, "[vbc_acquisition] confidence query failed: %s", exc)
            return False, 0.0, math.nan, math.nan
        n = points.shape[0]
        if (len(res.confidence) != n or len(res.inside_map) != n or
                len(res.current_visibility) != n):
            return False, 0.0, math.nan, math.nan

        conf = np.asarray(res.confidence, dtype=np.float64)
        inside = np.asarray(res.inside_map, dtype=bool)
        current_vis = np.asarray(res.current_visibility, dtype=np.float64)
        good = inside & np.isfinite(conf) & (conf >= self.seen_threshold)
        fraction = float(np.mean(good)) if n else 0.0
        seen = fraction + 1e-12 >= self.required_seen_fraction
        return (
            bool(seen), fraction,
            float(np.nanmin(conf)) if conf.size else math.nan,
            float(np.nanmax(current_vis)) if current_vis.size else math.nan,
        )

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
            seen, frac, min_conf, max_vis = self._query_region_seen(ob)
            oid = int(ob["id"])
            diagnostics.append((oid, seen, frac, min_conf, max_vis))
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
            f"{oid}:{int(seen)}:{frac:.3f}:{min_conf:.3f}:{max_vis:.3f}"
            for oid, seen, frac, min_conf, max_vis in diagnostics)
        msg.data = (
            "policy=visibility_acquisition"
            f" started={int(self._acquisition_started)}"
            f" complete={int(self._acquisition_complete)}"
            f" remaining_obligation_count={len(remaining)}"
            f" remaining_ids={':'.join(str(int(ob['id'])) for ob in remaining) or 'none'}"
            f" seen_obligation_count={self._seen_obligation_count}"
            f" safe_repair_candidate_count={self._safe_repair_candidate_count}"
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
        """Publish only the earliest persistent goal through the legacy q_vis path.

        C4.7 deliberately disables multi-deadline MPC.  q_vis is a visibility
        acquisition goal, not a point that must be reached at a nominal clock.
        The REPAIR QP therefore uses its terminal q_vis objective.  Receding safe
        commits can advance the real robot over multiple cycles until confidence
        confirms acquisition.
        """
        if not getattr(self, "_c47_ready", False):
            return
        obligations = self._ordered_obligations()
        if obligations:
            first = obligations[0]
            q = np.asarray(first["q_vis"], dtype=np.float64)
            with self._lock:
                self._q_vis = q.copy()
                self._q_zero = np.asarray(first["q_zero"], dtype=np.float64).copy()
                # Keep a valid timestamp for existing message plumbing only.
                # REPAIR single-waypoint MPC does not use it as a target time.
                self._deadline_abs_s = rospy.Time.now().to_sec() + 1.0
                self._deadline_from_start_s = math.nan
                self._generation_success = True
                self._shared_min_f = float(first["final_f_min"])
                self._shared_solution_mode = "c47_visibility_acquisition"
                self._summary = "c47_acquisition_goal_ready"
            self.waypoint_pub.publish(_vector_msg(q))
            self.zero_pub.publish(_vector_msg(np.asarray(first["q_zero"])))
            d = Float64(); d.data = float(self._deadline_abs_s)
            self.deadline_pub.publish(d)
        else:
            with self._lock:
                self._generation_success = False
                self._summary = "c47_no_visibility_obligations"

    def _maybe_generate(self) -> None:
        if not self._c47_ready:
            return
        # Reuse C4.6 accumulation/matching, but never clear from predicted SAFE.
        self._process_new_active_set()
        self._update_actual_visibility_completion()
        self._publish_schedule()

    def _publish_state(self) -> None:
        if not self._c47_ready:
            return
        obligations = self._ordered_obligations()
        active = len(obligations) > 0
        a = Bool(); a.data = active; self.active_pub.publish(a)
        if obligations:
            q = np.asarray(obligations[0]["q_vis"], dtype=np.float64)
            self.waypoint_pub.publish(_vector_msg(q))
            # Plumbing timestamp only; not a REPAIR deadline.
            d = Float64(); d.data = rospy.Time.now().to_sec() + 1.0
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
