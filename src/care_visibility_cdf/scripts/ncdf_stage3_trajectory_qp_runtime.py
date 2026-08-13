#!/usr/bin/env python3
"""Stage-III runtime node with ROS-safe trajectory writes and timing diagnostics.

This runtime variant intentionally does NOT force the end-of-horizon trajectory
correction back to zero.  In a receding-horizon controller only a short prefix
is executed before replanning, so a hard terminal equality can over-constrain
the current maneuver without guaranteeing closed-loop return.  Instead, the
terminal deviation is softly penalized while the whole horizon remains bounded
by the existing q/du trust regions and physical limits.
"""

from __future__ import annotations

import copy
import math
import time

import casadi as ca
import numpy as np
import rospy
from std_msgs.msg import Bool, Float32, String

from ncdf_stage3_trajectory_qp_core import (
    NcdfStage3TrajectoryQPNode,
    _format_vec,
    _value_grad,
)


class NcdfStage3TrajectoryQPRuntimeNode(NcdfStage3TrajectoryQPNode):
    def __init__(self):
        # Must be available before super().__init__(), because the base
        # constructor calls the overridden _build_qp_solver().
        self.terminal_q_weight = float(
            rospy.get_param("~terminal_q_weight", 20.0)
        )
        if self.terminal_q_weight < 0.0:
            raise ValueError("terminal_q_weight must be non-negative")

        super().__init__()

        rospy.loginfo(
            "[ncdf_stage3] terminal_q_weight=%.2f (soft terminal; no delta_q_H=0 constraint)",
            self.terminal_q_weight,
        )

    def _build_qp_solver(self):
        """Build Stage-III QP with a soft, rather than hard, terminal return."""
        n = 7
        K = self.num_intervals
        dt = self.stage_dt

        delta_q = ca.MX.sym("delta_q", n, K + 1)
        delta_u = ca.MX.sym("delta_u", n, K)

        p_len = n * (K + 1) + n * K + n * (K + 1) + n
        p = ca.MX.sym("p", p_len)
        off = 0
        q_nom = ca.reshape(p[off : off + n * (K + 1)], n, K + 1)
        off += n * (K + 1)
        u_nom = ca.reshape(p[off : off + n * K], n, K)
        off += n * K
        grad_dir = ca.reshape(p[off : off + n * (K + 1)], n, K + 1)
        off += n * (K + 1)
        dq_current = p[off : off + n]

        objective = (
            self.q_weight * ca.sumsqr(delta_q)
            + self.terminal_q_weight * ca.sumsqr(delta_q[:, K])
            + self.u_weight * ca.sumsqr(delta_u)
        )
        if K > 1:
            objective += self.smooth_weight * ca.sumsqr(
                delta_u[:, 1:] - delta_u[:, :-1]
            )

        # Since delta_q_K is now free, visibility also acts at the terminal
        # knot.  The terminal cost decides whether the visibility gain is worth
        # retaining a small nonzero deviation at the end of the horizon.
        for k in range(1, K + 1):
            objective -= self.visibility_weight * ca.dot(
                grad_dir[:, k], delta_q[:, k]
            )

        constraints = []
        lbg = []
        ubg = []

        def add_constraint(expr, lower, upper):
            constraints.append(ca.vec(expr))
            count = int(expr.numel())
            if np.ndim(lower) == 0:
                lbg.extend([float(lower)] * count)
            else:
                lbg.extend(
                    np.asarray(lower, dtype=np.float64)
                    .reshape(-1, order="F")
                    .tolist()
                )
            if np.ndim(upper) == 0:
                ubg.extend([float(upper)] * count)
            else:
                ubg.extend(
                    np.asarray(upper, dtype=np.float64)
                    .reshape(-1, order="F")
                    .tolist()
                )

        # The current state is fixed.  There is intentionally no hard terminal
        # delta_q[:, K] == 0 equality anymore.
        add_constraint(delta_q[:, 0], 0.0, 0.0)
        for k in range(K):
            add_constraint(
                delta_q[:, k + 1] - delta_q[:, k] - dt * delta_u[:, k],
                0.0,
                0.0,
            )

        # Bounded trajectory deviation and physical joint limits.
        add_constraint(
            delta_q,
            np.tile(-self.q_trust.reshape(7, 1), (1, K + 1)),
            np.tile(self.q_trust.reshape(7, 1), (1, K + 1)),
        )
        add_constraint(
            q_nom + delta_q,
            np.tile(self.q_min.reshape(7, 1), (1, K + 1)),
            np.tile(self.q_max.reshape(7, 1), (1, K + 1)),
        )

        # Bounded action deviation and total velocity limits.
        add_constraint(
            delta_u,
            np.tile(-self.du_trust.reshape(7, 1), (1, K)),
            np.tile(self.du_trust.reshape(7, 1), (1, K)),
        )
        add_constraint(
            u_nom + delta_u,
            np.tile(-self.velocity_limits.reshape(7, 1), (1, K)),
            np.tile(self.velocity_limits.reshape(7, 1), (1, K)),
        )

        # Total corrected-action acceleration limits.  The first knot is tied
        # to measured qdot; later knots constrain corrected velocity changes.
        add_constraint(
            u_nom[:, 0] + delta_u[:, 0] - dq_current,
            -self.acceleration_limits * self.control_dt,
            self.acceleration_limits * self.control_dt,
        )
        for k in range(1, K):
            add_constraint(
                (u_nom[:, k] + delta_u[:, k])
                - (u_nom[:, k - 1] + delta_u[:, k - 1]),
                -self.acceleration_limits * dt,
                self.acceleration_limits * dt,
            )

        x = ca.vertcat(ca.vec(delta_q), ca.vec(delta_u))
        g = ca.vertcat(*constraints)
        self._solver = ca.qpsol(
            "stage3_qp",
            "qrqp",
            {"x": x, "p": p, "f": objective, "g": g},
            {
                "print_header": False,
                "print_iter": False,
                "print_info": False,
                "error_on_fail": False,
            },
        )
        self._lbg = np.asarray(lbg, dtype=np.float64)
        self._ubg = np.asarray(ubg, dtype=np.float64)
        self._n_delta_q = n * (K + 1)

    def _apply_correction(self, traj, mapping, delta_q, delta_u):
        corrected = copy.deepcopy(traj)
        K = self.num_intervals
        dt = self.stage_dt

        for point in corrected.points:
            t = point.time_from_start.to_sec()
            if t <= 0.0:
                dq_corr = np.zeros(7, dtype=np.float64)
                du_corr = np.zeros(7, dtype=np.float64)
            elif t >= self.horizon_duration:
                # Keep the optimized terminal offset continuous beyond the
                # Stage-III horizon.  The offset is softly penalized and still
                # obeys q_trust; its derivative is zero, so nominal velocity is
                # retained beyond the horizon.
                dq_corr = delta_q[:, K]
                du_corr = np.zeros(7, dtype=np.float64)
            else:
                k = min(K - 1, int(math.floor(t / dt)))
                alpha = float((t - k * dt) / dt)
                dq_corr = (
                    (1.0 - alpha) * delta_q[:, k]
                    + alpha * delta_q[:, k + 1]
                )
                du_corr = delta_u[:, k]

            # rospy sequence fields may be tuples.  Convert to mutable lists,
            # modify them, then assign the whole sequence back to the message.
            positions = list(point.positions)
            for i, src in enumerate(mapping):
                if src >= len(positions):
                    raise RuntimeError(
                        f"trajectory point has no position index {src}"
                    )
                positions[src] = float(positions[src] + dq_corr[i])
            point.positions = positions

            if point.velocities:
                velocities = list(point.velocities)
                for i, src in enumerate(mapping):
                    if src < len(velocities):
                        velocities[src] = float(velocities[src] + du_corr[i])
                point.velocities = velocities

        return corrected

    def _trajectory_callback(self, traj):
        mapping = self._trajectory_mapping(traj)
        if mapping is None or not traj.points:
            rospy.logwarn_throttle(1.0, "[ncdf_stage3] invalid nominal trajectory")
            return

        with self._lock:
            target = copy.deepcopy(self._latest_target)
            target_received = self._target_received
            joint_state = copy.deepcopy(self._latest_joint_state)
            joint_received = self._joint_state_received

        target_fresh = (
            target is not None
            and self._age(target_received) <= self.target_timeout
            and (not target.header.frame_id or target.header.frame_id == self.base_frame)
        )
        if not target_fresh:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            return

        if joint_state is None or self._age(joint_received) > self.joint_state_timeout:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            rospy.logwarn_throttle(
                1.0, "[ncdf_stage3] stale JointState -> nominal pass-through"
            )
            return

        state = self._ordered_joint_state(joint_state)
        if state is None:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            return
        _, dq_current = state

        if traj.points[-1].time_from_start.to_sec() < self.horizon_duration - 1e-6:
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            rospy.logwarn_throttle(
                1.0,
                "[ncdf_stage3] nominal trajectory shorter than Stage-III horizon",
            )
            return

        x = np.asarray(
            [target.point.x, target.point.y, target.point.z], dtype=np.float64
        )
        if not np.all(np.isfinite(x)):
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            return

        K = self.num_intervals
        q_nom = np.zeros((7, K + 1), dtype=np.float64)
        u_nom = np.zeros((7, K), dtype=np.float64)
        grad_dir = np.zeros((7, K + 1), dtype=np.float64)
        f_nom = []

        total_tic = time.perf_counter()
        try:
            # -------- online part: nominal sampling + NCDF gradients --------
            ncdf_ms = 0.0
            for k in range(K + 1):
                sampled = self._sample_trajectory(
                    traj, mapping, k * self.stage_dt
                )
                if sampled is None:
                    raise RuntimeError("trajectory sampling failed")
                q_nom[:, k] = sampled[0]
                if k < K:
                    u_nom[:, k] = sampled[1]

                ncdf_tic = time.perf_counter()
                f_value, grad = _value_grad(
                    self.value_grad_fn, x, q_nom[:, k]
                )
                ncdf_ms += 1000.0 * (time.perf_counter() - ncdf_tic)

                f_nom.append(f_value)
                grad_norm = float(np.linalg.norm(grad))
                if grad_norm > self.grad_eps:
                    grad_dir[:, k] = grad / grad_norm

            parameters = self._pack_parameters(
                q_nom, u_nom, grad_dir, dq_current
            )

            qp_tic = time.perf_counter()
            solution = self._solver(
                p=parameters, lbg=self._lbg, ubg=self._ubg
            )
            qp_ms = 1000.0 * (time.perf_counter() - qp_tic)

            stats = self._solver.stats()
            if not bool(stats.get("success", True)):
                raise RuntimeError(
                    f"QP failed: {stats.get('return_status', 'unknown')}"
                )

            z = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
            delta_q = z[: self._n_delta_q].reshape(
                (7, K + 1), order="F"
            )
            delta_u = z[self._n_delta_q :].reshape((7, K), order="F")

            apply_tic = time.perf_counter()
            corrected = self._apply_correction(
                traj, mapping, delta_q, delta_u
            )
            apply_ms = 1000.0 * (time.perf_counter() - apply_tic)
            online_ms = 1000.0 * (time.perf_counter() - total_tic)

            # -------- diagnostics: not required to produce the command --------
            diag_ncdf_tic = time.perf_counter()
            learned_delta = []
            for k in range(1, K + 1):
                f_corrected, _ = _value_grad(
                    self.value_grad_fn, x, q_nom[:, k] + delta_q[:, k]
                )
                learned_delta.append(f_corrected - f_nom[k])
            diag_ncdf_ms = 1000.0 * (time.perf_counter() - diag_ncdf_tic)

            oracle_delta = []
            oracle_tic = time.perf_counter()
            if self.oracle is not None:
                for k in range(1, K + 1):
                    oracle_delta.append(
                        self._oracle_g(x, q_nom[:, k] + delta_q[:, k])
                        - self._oracle_g(x, q_nom[:, k])
                    )
            oracle_ms = 1000.0 * (time.perf_counter() - oracle_tic)

            total_ms = 1000.0 * (time.perf_counter() - total_tic)
            self._seq += 1
            max_delta_q = float(np.max(np.abs(delta_q)))
            max_delta_u = float(np.max(np.abs(delta_u)))
            terminal_norm = float(np.linalg.norm(delta_q[:, -1]))

            self.output_pub.publish(corrected)
            self.active_pub.publish(Bool(data=True))
            self.solve_time_pub.publish(Float32(data=online_ms))

            dg_mean = (
                float(np.mean(oracle_delta)) if oracle_delta else math.nan
            )
            dg_improve = (
                float(np.mean(np.asarray(oracle_delta) > 0.0))
                if oracle_delta
                else math.nan
            )
            summary = (
                f"seq={self._seq} x={_format_vec(x)} "
                f"max_dq={max_delta_q:.5f} max_du={max_delta_u:.5f} "
                f"terminal_dq={terminal_norm:.5f} "
                f"df_mean={float(np.mean(learned_delta)):+.5f} "
                f"df_min={float(np.min(learned_delta)):+.5f} "
                f"dg_mean={dg_mean:+.5f} dg_improve={dg_improve:.3f} "
                f"ncdf={ncdf_ms:.2f}ms qp={qp_ms:.2f}ms "
                f"apply={apply_ms:.2f}ms online={online_ms:.2f}ms "
                f"diag_ncdf={diag_ncdf_ms:.2f}ms "
                f"oracle={oracle_ms:.2f}ms total={total_ms:.2f}ms"
            )
            self.summary_pub.publish(String(data=summary))
            rospy.loginfo_throttle(0.5, "[ncdf_stage3] %s", summary)

        except Exception as exc:
            # Stage III is a filter. Any optimization problem falls back to the
            # untouched nominal trajectory; the executor remains downstream.
            self.output_pub.publish(traj)
            self.active_pub.publish(Bool(data=False))
            rospy.logerr_throttle(
                1.0,
                "[ncdf_stage3] QP failed -> nominal pass-through: %s",
                exc,
            )


def main():
    rospy.init_node("ncdf_stage3_trajectory_qp")
    NcdfStage3TrajectoryQPRuntimeNode()
    rospy.spin()
