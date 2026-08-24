#!/usr/bin/env python3
"""C4.3 rolling VBC waypoint generator with optional all-point active-set steering.

Legacy rolling mode keeps the validated single-target projection/root/ascent path.
C4.3 temporal-layer mode additionally subscribes to the full active point set
published by the selector. The active set is the union of all spatial regions in
the earliest unsafe temporal risk layer. A shared learned field

    F(q) = min_i f_theta(x_i, q)

is used for steering. Projection/root/ascent is attempted on F. If a common
zero-level set cannot be reached within the validated projection budget, the
node publishes a best-effort max-min ascent waypoint instead of blocking the
Recovery episode. Analytic visibility remains diagnostic only; global VBC is
still the safety truth.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vbc_deadline_waypoint_node import (  # noqa: E402
    VbcDeadlineWaypointNode,
    _finite,
    _fmt,
    _vector_msg,
)
from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    learned_ascent_step,
    learned_projection_step,
    model_value_and_grad_q,
)


class RollingVbcDeadlineWaypointNode(VbcDeadlineWaypointNode):
    def __init__(self) -> None:
        self._selection_active = False
        self._predicted_trajectory = None
        self._predicted_trajectory_received = None
        self._rolling_trajectory_source = "bootstrap"
        self.predicted_trajectory_timeout = 0.20
        self.target_cell_resolution = 0.05
        self._target_cell_key = None

        self.use_active_set = False
        self.active_set_points_topic = (
            "/care_planner/trajectory_risk/vbc_active_set_points")
        self._active_set_xyz = None
        self._active_set_key = None
        self._active_set_received = None
        self.shared_fallback_ascent_steps = 8
        self._shared_min_f = math.nan
        self._shared_solution_mode = "single_target"

        super().__init__()

        self.selection_active_topic = str(rospy.get_param(
            "~selection_active_topic",
            "/care_planner/active_sensing/target_selection_active"))
        self.predicted_trajectory_topic = str(rospy.get_param(
            "~predicted_trajectory_topic",
            "/care_planner/mpc/predicted_trajectory"))
        self.predicted_trajectory_timeout = float(rospy.get_param(
            "~predicted_trajectory_timeout", 0.20))
        self.target_cell_resolution = float(rospy.get_param(
            "~target_cell_resolution", 0.05))
        self.use_active_set = bool(rospy.get_param("~use_active_set", False))
        self.active_set_points_topic = str(rospy.get_param(
            "~active_set_points_topic",
            "/care_planner/trajectory_risk/vbc_active_set_points"))
        self.shared_fallback_ascent_steps = int(rospy.get_param(
            "~shared_fallback_ascent_steps", 8))

        if self.predicted_trajectory_timeout <= 0.0:
            raise ValueError("~predicted_trajectory_timeout must be positive")
        if self.target_cell_resolution <= 0.0:
            raise ValueError("~target_cell_resolution must be positive")
        if self.shared_fallback_ascent_steps < 1:
            raise ValueError("~shared_fallback_ascent_steps must be >= 1")

        self.selection_active_sub = rospy.Subscriber(
            self.selection_active_topic, Bool,
            self._selection_active_callback, queue_size=1)
        self.predicted_trajectory_sub = rospy.Subscriber(
            self.predicted_trajectory_topic, JointTrajectory,
            self._predicted_trajectory_callback, queue_size=1)
        self.active_set_sub = None
        if self.use_active_set:
            self.active_set_sub = rospy.Subscriber(
                self.active_set_points_topic, Float64MultiArray,
                self._active_set_callback, queue_size=1)

        rospy.logwarn(
            "[vbc_waypoint_rolling] C4.3 mode: selection_active=%s "
            "predicted=%s timeout=%.3fs target_cell=%.3fm active_set=%d topic=%s",
            self.selection_active_topic,
            self.predicted_trajectory_topic,
            self.predicted_trajectory_timeout,
            self.target_cell_resolution,
            int(self.use_active_set),
            self.active_set_points_topic)

    def _cell_key(self, xyz):
        scaled = np.rint(
            np.asarray(xyz, dtype=np.float64) / self.target_cell_resolution)
        return tuple(int(v) for v in scaled.tolist())

    def _active_set_callback(self, msg: Float64MultiArray) -> None:
        if msg is None:
            return
        values = np.asarray(list(msg.data), dtype=np.float64)
        if values.size == 0:
            points = np.zeros((0, 3), dtype=np.float64)
        elif values.size % 3 != 0:
            rospy.logwarn_throttle(
                1.0,
                "[vbc_waypoint_rolling] active-set message length %d is not divisible by 3",
                int(values.size))
            return
        else:
            points = values.reshape(-1, 3)
            if not _finite(points):
                return

        by_key = {}
        for point in points:
            by_key[self._cell_key(point)] = point.copy()
        ordered_keys = tuple(sorted(by_key.keys()))
        ordered_points = (
            np.asarray([by_key[k] for k in ordered_keys], dtype=np.float64)
            if ordered_keys else np.zeros((0, 3), dtype=np.float64))

        with self._lock:
            changed = self._active_set_key != ordered_keys
            self._active_set_xyz = ordered_points
            self._active_set_key = ordered_keys
            self._active_set_received = rospy.Time.now()
            if not self.use_active_set:
                return
            if not ordered_keys:
                self._generation_success = False
                self._generation_key = None
                self._q_zero = None
                self._q_vis = None
                self._deadline_abs_s = None
                self._deadline_from_start_s = None
                self._sweep_time_s = None
                self._shared_min_f = math.nan
                self._shared_solution_mode = "active_set_empty"
                self._summary = "active_set_empty"
            elif changed:
                self._generation_success = False
                self._generation_key = None
                self._q_zero = None
                self._q_vis = None
                self._deadline_abs_s = None
                self._deadline_from_start_s = None
                self._sweep_time_s = None
                self._shared_min_f = math.nan
                self._shared_solution_mode = "waiting_generation"
                self._summary = "new_active_set_waiting_sweep"
                rospy.logwarn(
                    "[vbc_waypoint_rolling] new active set: %d cells",
                    len(ordered_keys))

    def _selection_active_callback(self, msg: Bool) -> None:
        if msg is None:
            return
        active = bool(msg.data)
        with self._lock:
            was_active = self._selection_active
            self._selection_active = active
            if not active:
                self._generation_success = False
                self._generation_key = None
                self._summary = "selection_inactive"
            elif not was_active:
                self._seen_latched = False
                self._generation_key = None
                self._summary = "selection_active_waiting_pair"

    def _predicted_trajectory_callback(self, msg: JointTrajectory) -> None:
        if msg is None or not msg.points:
            return
        with self._lock:
            self._predicted_trajectory = msg
            self._predicted_trajectory_received = rospy.Time.now()

    def _target_callback(self, msg: PointStamped) -> None:
        if msg is None:
            return
        xyz = self._target_array(msg)
        if not _finite(xyz):
            return
        key = self._cell_key(xyz)

        with self._lock:
            changed_cell = self._target_cell_key != key
            self._target = msg
            self._target_xyz = xyz
            self._target_cell_key = key

            if self.use_active_set:
                return

            if changed_cell:
                self._seen_latched = False
                self._generation_success = False
                self._generation_key = None
                self._q_zero = None
                self._q_vis = None
                self._deadline_abs_s = None
                self._deadline_from_start_s = None
                self._sweep_time_s = None
                self._summary = "new_rolling_target_cell_waiting_sweep"
                rospy.logwarn(
                    "[vbc_waypoint_rolling] new target cell=%s x*=%s",
                    key, _fmt(xyz, 6))

    def _preferred_trajectory_locked(self):
        now = rospy.Time.now()
        if (self._predicted_trajectory is not None and
                self._predicted_trajectory_received is not None):
            age = (now - self._predicted_trajectory_received).to_sec()
            if 0.0 <= age <= self.predicted_trajectory_timeout:
                return (
                    self._predicted_trajectory,
                    self._predicted_trajectory_received,
                    "predicted")
        return self._trajectory, self._trajectory_received, "bootstrap"

    def _per_point_values(self, x: torch.Tensor, q: torch.Tensor):
        with torch.no_grad():
            q_batch = q.expand(x.shape[0], -1)
            inputs = torch.cat([x, q_batch], dim=-1)
            values = self.model(inputs).reshape(-1)
        return values.detach().cpu().numpy().astype(float)

    def _oracle_diag_set(self, x: torch.Tensor, q: torch.Tensor):
        if self.conservative_oracle is None or self.nominal_oracle is None:
            return {
                "g_conservative_min": None,
                "g_nominal_fov_min": None,
                "conservative_visible_fraction": None,
                "nominal_visible_fraction": None,
            }
        g_cons = []
        g_nom = []
        for i in range(x.shape[0]):
            diag = self._oracle_diag(x[i:i + 1], q)
            g_cons.append(float(diag["g_conservative"]))
            g_nom.append(float(diag["g_nominal_fov"]))
        return {
            "g_conservative_min": min(g_cons) if g_cons else None,
            "g_nominal_fov_min": min(g_nom) if g_nom else None,
            "conservative_visible_fraction": (
                float(np.mean(np.asarray(g_cons) >= 0.0)) if g_cons else None),
            "nominal_visible_fraction": (
                float(np.mean(np.asarray(g_nom) >= 0.0)) if g_nom else None),
        }

    def _generate_active_set_waypoint(
            self, points_xyz, trajectory, sweep_time_s, trajectory_received):
        points_np = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        if points_np.shape[0] < 1 or not _finite(points_np):
            raise ValueError("active set must contain finite xyz points")

        deadline_from_start = max(0.0, sweep_time_s - self.safety_margin_s)
        q_nom_np = self._sample_trajectory(trajectory, deadline_from_start)
        if not _finite(q_nom_np):
            raise ValueError("non-finite nominal deadline configuration")

        if trajectory.header.stamp.to_sec() > 0.0:
            epoch_s = trajectory.header.stamp.to_sec()
            epoch_source = "trajectory_header"
        elif trajectory_received is not None:
            epoch_s = trajectory_received.to_sec()
            epoch_source = "trajectory_receive_time_fallback"
        else:
            epoch_s = rospy.Time.now().to_sec()
            epoch_source = "now_fallback"
        deadline_abs_s = epoch_s + deadline_from_start

        x = torch.tensor(
            points_np, device=self.device, dtype=torch.float32)
        q0 = torch.tensor(
            q_nom_np.reshape(1, 7), device=self.device, dtype=torch.float32)
        q0, initial_clamped = self._clamp(q0)
        q = q0.clone()

        f_current, grad_current = self._learned(x, q)
        best_q = q.clone()
        best_f = float(f_current)
        projection_history = [{
            "iter": 0,
            "f_min": float(f_current),
            "grad_norm": float(torch.linalg.vector_norm(grad_current[0]).item()),
            "q": q[0].detach().cpu().numpy().astype(float).tolist(),
        }]
        root_history = []
        q_zero = None
        f_zero = math.nan
        root_source = "none"

        if f_current >= 0.0:
            q_zero = q.clone()
            f_zero = float(f_current)
            root_source = "initial_shared_positive"
        elif abs(f_current) <= self.projection_epsilon_f:
            q_zero = q.clone()
            f_zero = float(f_current)
            root_source = "initial_shared_tolerance"

        for iteration in range(1, self.projection_iters + 1):
            if q_zero is not None:
                break
            f_tensor, grad_tensor, _ = model_value_and_grad_q(x, q, self.model)
            q_next, diag = learned_projection_step(
                q=q,
                f=f_tensor,
                grad=grad_tensor,
                damping=self.projection_damping,
                max_step_norm=self.projection_max_step_norm,
                eps=self.math_eps)
            q_next, joint_clamped = self._clamp(q_next)
            f_next, grad_next = self._learned(x, q_next)
            if f_next > best_f:
                best_f = float(f_next)
                best_q = q_next.clone()
            projection_history.append({
                "iter": iteration,
                "f_min": float(f_next),
                "grad_norm": float(torch.linalg.vector_norm(grad_next[0]).item()),
                "raw_step_norm": float(diag["raw_step_norm"][0].item()),
                "applied_step_norm": float(diag["applied_step_norm"][0].item()),
                "algorithm_step_clipped": bool(diag["clipped"][0].item()),
                "joint_limit_clamped": joint_clamped,
                "q": q_next[0].detach().cpu().numpy().astype(float).tolist(),
            })

            if f_current * f_next <= 0.0 and f_current != f_next:
                q_zero, f_zero, root_history = self._refine_root_bisection(
                    x, q, f_current, q_next, f_next)
                root_source = "shared_sign_crossing_bisection"
                break
            if abs(f_next) <= self.projection_epsilon_f:
                q_zero = q_next.detach()
                f_zero = float(f_next)
                root_source = "shared_projection_tolerance"
                break

            q = q_next.detach()
            f_current = float(f_next)

        ascent_history = []
        if q_zero is not None:
            q_vis = q_zero.clone()
            for step in range(1, self.ascent_steps + 1):
                _, grad_tensor, _ = model_value_and_grad_q(
                    x, q_vis, self.model)
                q_next, diag = learned_ascent_step(
                    q=q_vis,
                    grad=grad_tensor,
                    step_size=self.ascent_step_size,
                    max_step_norm=self.ascent_max_step_norm,
                    eps=self.math_eps)
                q_next, joint_clamped = self._clamp(q_next)
                q_vis = q_next.detach()
                f_vis, grad_vis = self._learned(x, q_vis)
                ascent_history.append({
                    "step": step,
                    "f_min": float(f_vis),
                    "grad_norm": float(torch.linalg.vector_norm(grad_vis[0]).item()),
                    "applied_step_norm": float(diag["applied_step_norm"][0].item()),
                    "joint_limit_clamped": joint_clamped,
                    "q": q_vis[0].detach().cpu().numpy().astype(float).tolist(),
                })
            solution_mode = "shared_projection_root_ascent"
        else:
            q_vis = best_q.clone()
            best_shared_q = q_vis.clone()
            best_shared_f = best_f
            for step in range(1, self.shared_fallback_ascent_steps + 1):
                _, grad_tensor, _ = model_value_and_grad_q(
                    x, q_vis, self.model)
                q_next, diag = learned_ascent_step(
                    q=q_vis,
                    grad=grad_tensor,
                    step_size=self.ascent_step_size,
                    max_step_norm=self.ascent_max_step_norm,
                    eps=self.math_eps)
                q_next, joint_clamped = self._clamp(q_next)
                f_next, grad_next = self._learned(x, q_next)
                ascent_history.append({
                    "step": step,
                    "f_min": float(f_next),
                    "grad_norm": float(torch.linalg.vector_norm(grad_next[0]).item()),
                    "applied_step_norm": float(diag["applied_step_norm"][0].item()),
                    "joint_limit_clamped": joint_clamped,
                    "q": q_next[0].detach().cpu().numpy().astype(float).tolist(),
                    "fallback": True,
                })
                q_vis = q_next.detach()
                if f_next > best_shared_f:
                    best_shared_f = float(f_next)
                    best_shared_q = q_vis.clone()
            q_vis = best_shared_q
            q_zero = best_q.clone()
            f_zero = float(best_f)
            root_source = "shared_root_not_found"
            solution_mode = "best_effort_max_min_ascent"

        initial_values = self._per_point_values(x, q0)
        final_values = self._per_point_values(x, q_vis)
        f_final = float(np.min(final_values))
        initial_diag = self._oracle_diag_set(x, q0)
        final_diag = self._oracle_diag_set(x, q_vis)

        return {
            "active_set_points_xyz": points_np.tolist(),
            "active_set_size": int(points_np.shape[0]),
            "active_set_centroid_xyz": np.mean(points_np, axis=0).astype(float).tolist(),
            "nominal_sweep_time_s": float(sweep_time_s),
            "safety_margin_s": self.safety_margin_s,
            "deadline_from_start_s": deadline_from_start,
            "deadline_absolute_ros_s": deadline_abs_s,
            "deadline_epoch_source": epoch_source,
            "q_deadline_nominal": q0[0].detach().cpu().numpy().astype(float).tolist(),
            "q_deadline_initial_clamped": initial_clamped,
            "initial_f_min": float(np.min(initial_values)),
            "initial_f_mean": float(np.mean(initial_values)),
            "initial_f_per_point": initial_values.tolist(),
            "initial_oracle_diagnostic": initial_diag,
            "projection_root_source": root_source,
            "projection_zero_f": float(f_zero),
            "q_zero": q_zero[0].detach().cpu().numpy().astype(float).tolist(),
            "q_vis": q_vis[0].detach().cpu().numpy().astype(float).tolist(),
            "final_f_min": f_final,
            "final_f_mean": float(np.mean(final_values)),
            "final_f_per_point": final_values.tolist(),
            "shared_solution_mode": solution_mode,
            "shared_learned_all_positive": bool(f_final >= 0.0),
            "final_oracle_diagnostic": final_diag,
            "distance_qzero_from_nominal": float(
                torch.linalg.vector_norm(q_zero - q0).item()),
            "distance_qvis_from_nominal": float(
                torch.linalg.vector_norm(q_vis - q0).item()),
            "projection_history": projection_history,
            "root_refinement_history": root_history,
            "ascent_history": ascent_history,
            "config": {
                "projection_iters": self.projection_iters,
                "projection_damping": self.projection_damping,
                "projection_epsilon_f": self.projection_epsilon_f,
                "projection_max_step_norm": self.projection_max_step_norm,
                "root_refine_iters": self.root_refine_iters,
                "root_tolerance_f": self.root_tolerance_f,
                "ascent_steps": self.ascent_steps,
                "ascent_step_size": self.ascent_step_size,
                "ascent_max_step_norm": self.ascent_max_step_norm,
                "shared_fallback_ascent_steps": self.shared_fallback_ascent_steps,
            },
        }

    def _maybe_generate(self) -> None:
        with self._lock:
            selection_active = self._selection_active
            target = self._target
            target_xyz = (
                None if self._target_xyz is None else self._target_xyz.copy())
            target_cell_key = self._target_cell_key
            sweep = self._sweep_time_s
            old_key = self._generation_key
            active_set_xyz = (
                None if self._active_set_xyz is None
                else self._active_set_xyz.copy())
            active_set_key = self._active_set_key
            trajectory, trajectory_received, trajectory_source = (
                self._preferred_trajectory_locked())

        if not selection_active or trajectory is None or sweep is None:
            return

        if self.use_active_set:
            if (active_set_xyz is None or active_set_key is None or
                    len(active_set_key) == 0):
                return
            key = ("rolling_active_set", active_set_key)
            if old_key == key:
                return
            try:
                result = self._generate_active_set_waypoint(
                    active_set_xyz, trajectory, sweep, trajectory_received)
            except Exception as exc:
                with self._lock:
                    self._generation_key = key
                    self._generation_success = False
                    self._summary = (
                        "shared_generation_failed:" +
                        str(exc).replace(" ", "_"))
                rospy.logerr(
                    "[vbc_waypoint_rolling] shared generation failed: %s", exc)
                return

            q_zero = np.asarray(result["q_zero"], dtype=np.float64)
            q_vis = np.asarray(result["q_vis"], dtype=np.float64)
            deadline_abs = float(result["deadline_absolute_ros_s"])
            deadline_from_start = float(result["deadline_from_start_s"])
            result["rolling_trajectory_source"] = trajectory_source
            result["rolling_active_set_cell_keys"] = [
                list(k) for k in active_set_key]
            result["rolling_active_set_cell_resolution"] = (
                self.target_cell_resolution)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = self.output_root / f"vbc_visibility_waypoint_{stamp}.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=True))

            with self._lock:
                self._generation_key = key
                self._q_zero = q_zero
                self._q_vis = q_vis
                self._deadline_abs_s = deadline_abs
                self._deadline_from_start_s = deadline_from_start
                self._generation_success = True
                self._rolling_trajectory_source = trajectory_source
                self._shared_min_f = float(result["final_f_min"])
                self._shared_solution_mode = str(result["shared_solution_mode"])
                self._summary = "ready_shared_active_set"

            self.zero_pub.publish(_vector_msg(q_zero))
            self.waypoint_pub.publish(_vector_msg(q_vis))
            dmsg = Float64()
            dmsg.data = deadline_abs
            self.deadline_pub.publish(dmsg)
            rospy.logwarn(
                "[vbc_waypoint_rolling] SHARED WAYPOINT READY source=%s "
                "points=%d sweep=%.3f min_f=%+.5f mode=%s q_vis=%s",
                trajectory_source,
                int(result["active_set_size"]),
                sweep,
                float(result["final_f_min"]),
                result["shared_solution_mode"],
                _fmt(q_vis, 5))
            rospy.logwarn(
                "[vbc_waypoint_rolling] saved shared generation trace: %s",
                path)
            return

        if (target is None or target_xyz is None or target_cell_key is None):
            return
        key = ("rolling_cell", target_cell_key)
        if old_key == key:
            return
        try:
            result = self._generate_waypoint(
                target, trajectory, sweep, trajectory_received)
        except Exception as exc:
            with self._lock:
                self._generation_key = key
                self._generation_success = False
                self._summary = (
                    "generation_failed:" + str(exc).replace(" ", "_"))
            rospy.logerr(
                "[vbc_waypoint_rolling] generation failed: %s", exc)
            return

        q_zero = np.asarray(result["q_zero"], dtype=np.float64)
        q_vis = np.asarray(result["q_vis"], dtype=np.float64)
        deadline_abs = float(result["deadline_absolute_ros_s"])
        deadline_from_start = float(result["deadline_from_start_s"])
        result["rolling_trajectory_source"] = trajectory_source
        result["rolling_target_cell_key"] = list(target_cell_key)
        result["rolling_target_cell_resolution"] = self.target_cell_resolution
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_root / f"vbc_visibility_waypoint_{stamp}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=True))

        with self._lock:
            self._generation_key = key
            self._q_zero = q_zero
            self._q_vis = q_vis
            self._deadline_abs_s = deadline_abs
            self._deadline_from_start_s = deadline_from_start
            self._generation_success = True
            self._rolling_trajectory_source = trajectory_source
            self._shared_min_f = math.nan
            self._shared_solution_mode = "single_target"
            self._summary = "ready"

        self.zero_pub.publish(_vector_msg(q_zero))
        self.waypoint_pub.publish(_vector_msg(q_vis))
        dmsg = Float64()
        dmsg.data = deadline_abs
        self.deadline_pub.publish(dmsg)
        rospy.logwarn(
            "[vbc_waypoint_rolling] WAYPOINT READY source=%s cell=%s "
            "x*=%s sweep=%.3f q_vis=%s",
            trajectory_source,
            target_cell_key,
            _fmt(result["target_xyz"], 6),
            sweep,
            _fmt(q_vis, 5))
        rospy.logwarn(
            "[vbc_waypoint_rolling] saved generation trace: %s", path)

    def _publish_state(self) -> None:
        with self._lock:
            selection_active = self._selection_active
            target = self._target
            generation_success = self._generation_success
            q_vis = None if self._q_vis is None else self._q_vis.copy()
            deadline_abs = self._deadline_abs_s
            deadline_from_start = self._deadline_from_start_s
            seen = self._seen_latched
            summary_reason = self._summary
            trajectory_source = self._rolling_trajectory_source
            target_cell_key = self._target_cell_key
            active_set_size = (
                0 if self._active_set_xyz is None
                else int(self._active_set_xyz.shape[0]))
            shared_min_f = self._shared_min_f
            shared_solution_mode = self._shared_solution_mode

        if not selection_active:
            amsg = Bool()
            amsg.data = False
            self.active_pub.publish(amsg)
            msg = self._summary_message(
                selection_active=False,
                active=False,
                seen=False if self.use_active_set else seen,
                ready=False,
                confidence=math.nan,
                current_visibility=math.nan,
                inside=False,
                deadline_from_start=deadline_from_start,
                deadline_abs=deadline_abs,
                reason=summary_reason,
                trajectory_source=trajectory_source,
                target_cell_key=target_cell_key,
                active_set_size=active_set_size,
                shared_min_f=shared_min_f,
                shared_solution_mode=shared_solution_mode)
            self.summary_pub.publish(msg)
            return

        if self.use_active_set:
            active = bool(
                generation_success and q_vis is not None and
                deadline_abs is not None and active_set_size > 0)
            amsg = Bool()
            amsg.data = active
            self.active_pub.publish(amsg)
            if q_vis is not None:
                self.waypoint_pub.publish(_vector_msg(q_vis))
            if deadline_abs is not None:
                dmsg = Float64()
                dmsg.data = float(deadline_abs)
                self.deadline_pub.publish(dmsg)
            msg = self._summary_message(
                selection_active=True,
                active=active,
                seen=False,
                ready=generation_success,
                confidence=math.nan,
                current_visibility=math.nan,
                inside=False,
                deadline_from_start=deadline_from_start,
                deadline_abs=deadline_abs,
                reason=summary_reason,
                trajectory_source=trajectory_source,
                target_cell_key=target_cell_key,
                active_set_size=active_set_size,
                shared_min_f=shared_min_f,
                shared_solution_mode=shared_solution_mode)
            self.summary_pub.publish(msg)
            rospy.loginfo_throttle(
                0.5, "[vbc_waypoint_rolling] %s", msg.data)
            return

        confidence = math.nan
        current_visibility = math.nan
        inside = False
        if target is not None:
            response = self._query_confidence(target)
            if response is not None:
                confidence, current_visibility, inside = response
                if (inside and confidence >= self.seen_threshold and not seen):
                    with self._lock:
                        self._seen_latched = True
                    seen = True
                    rospy.logwarn(
                        "[vbc_waypoint_rolling] target CONFIRMED SEEN -> "
                        "waypoint objective OFF: confidence=%.4f",
                        confidence)

        active = bool(
            selection_active and generation_success and q_vis is not None and
            deadline_abs is not None and not seen)
        amsg = Bool()
        amsg.data = active
        self.active_pub.publish(amsg)
        if q_vis is not None:
            self.waypoint_pub.publish(_vector_msg(q_vis))
        if deadline_abs is not None:
            dmsg = Float64()
            dmsg.data = float(deadline_abs)
            self.deadline_pub.publish(dmsg)

        msg = self._summary_message(
            selection_active=selection_active,
            active=active,
            seen=seen,
            ready=generation_success,
            confidence=confidence,
            current_visibility=current_visibility,
            inside=inside,
            deadline_from_start=deadline_from_start,
            deadline_abs=deadline_abs,
            reason=summary_reason,
            trajectory_source=trajectory_source,
            target_cell_key=target_cell_key,
            active_set_size=0,
            shared_min_f=math.nan,
            shared_solution_mode="single_target")
        self.summary_pub.publish(msg)
        rospy.loginfo_throttle(
            0.5, "[vbc_waypoint_rolling] %s", msg.data)

    @staticmethod
    def _summary_message(
            selection_active, active, seen, ready, confidence,
            current_visibility, inside, deadline_from_start, deadline_abs,
            reason, trajectory_source, target_cell_key,
            active_set_size=0, shared_min_f=math.nan,
            shared_solution_mode="single_target"):
        now = rospy.Time.now().to_sec()
        remaining = math.nan if deadline_abs is None else deadline_abs - now
        cell = (
            "none" if target_cell_key is None
            else "[{},{},{}]".format(*target_cell_key))
        msg = String()
        msg.data = (
            f"active={int(active)} seen={int(seen)} ready={int(ready)} "
            f"selection_active={int(selection_active)} target_cell={cell} "
            f"active_set_size={int(active_set_size)} "
            f"shared_min_f={shared_min_f:.6f} "
            f"shared_solution_mode={shared_solution_mode} "
            f"confidence={confidence:.4f} "
            f"current_visibility={current_visibility:.4f} "
            f"inside_map={int(inside)} "
            f"deadline_from_start="
            f"{deadline_from_start if deadline_from_start is not None else math.nan:.6f} "
            f"deadline_remaining={remaining:.6f} "
            f"trajectory_source={trajectory_source} reason={reason}"
        )
        return msg


def main() -> None:
    rospy.init_node("vbc_deadline_waypoint_rolling")
    try:
        RollingVbcDeadlineWaypointNode()
    except Exception as exc:
        rospy.logfatal(
            "[vbc_waypoint_rolling] initialization failed: %s", exc)
        raise
    rospy.spin()


if __name__ == "__main__":
    main()
