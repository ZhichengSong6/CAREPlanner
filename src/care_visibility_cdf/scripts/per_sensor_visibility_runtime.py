#!/usr/bin/env python3
"""Runtime helper for hybrid scalar + per-sensor Visibility CDF steering.

The frozen scalar VisCDF remains the coarse union/manifold projector.  After the
scalar q_zero is available, the 8-head model preserves sensor modes:

    scalar q_zero
        -> rank the 8 learned sensor heads
        -> for each sensor, solve its OWN field exactly like the legacy scalar:
             projection -> sign-crossing root refinement -> small ascent
        -> conservative analytic FOV check
        -> zero-padding primitive self-occlusion check
        -> accept first certified branch, or fall back to scalar q_vis

Important runtime semantic:
  * ranking is evaluated at scalar q_zero;
  * each sensor branch is solved from the branch seed (Phase-E uses the current
    measured q through q_deadline_nominal), NOT from the S4-dominated scalar
    q_zero;
  * self-occlusion is checked only on the final branch candidate, never inside
    learned projection/ascent iterations;
  * final GCDF / exact VBC remain the execution safety authorities.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from urdf_parser_py.urdf import URDF

from evaluate_direct_vs_projection_ascent import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    YimingMLP,
    normalize_checkpoint_args,
    torch_load_checkpoint,
)
from validate_visibility_oracle import (
    find_chain_joints,
    fk_transform,
    sensor_margin,
)
from check_visibility_self_occlusion import (
    load_collision_primitives,
    q_row_to_map,
    raycast_self_occlusion,
)


class _RayArgs:
    """Frozen zero-padding runtime primitive LOS semantics."""

    min_ray_length = 1e-4
    ray_start_offset = 0.03
    point_end_offset = 0.005
    ignore_links: List[str] = []
    ignore_start_inside = True
    min_hit_distance = 0.0


def _parse_layers(raw) -> Tuple[int, ...]:
    if isinstance(raw, str):
        return tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(int(v) for v in raw)
    raise TypeError("unsupported mlp_layers type: {}".format(type(raw)))


def _parse_skips(raw) -> Tuple[int, ...]:
    if isinstance(raw, str):
        return tuple(int(v.strip()) for v in raw.split(",") if v.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(int(v) for v in raw)
    return ()


def build_per_sensor_model(checkpoint_path: str, device: torch.device):
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    semantics = str(ckpt.get("output_semantics", ""))
    if semantics and semantics != "per_sensor_signed_visibility_cdf":
        raise RuntimeError(
            "unexpected per-sensor checkpoint semantics: {!r}".format(semantics)
        )

    cargs = normalize_checkpoint_args(ckpt.get("args", {}))
    out_dim = int(ckpt.get("out_dim", 8))
    if out_dim != 8:
        raise RuntimeError(
            "per-sensor runtime requires out_dim=8, checkpoint has {}".format(
                out_dim
            )
        )

    model = YimingMLP(
        in_dim=10,
        out_dim=8,
        activation=cargs.get("activation", "relu"),
        model_arch=cargs.get("model_arch", "yiming"),
        mlp_layers=_parse_layers(
            cargs.get("mlp_layers", "1024,512,256,128,128")
        ),
        skips=_parse_skips(cargs.get("skips", "")),
        nerf=bool(cargs.get("nerf", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


class PerSensorVisibilityRuntime:
    """Branch-specific 8-head q_vis proposer with final geometry certification."""

    def __init__(
        self,
        checkpoint_path: str,
        reference_urdf_path: str,
        self_filter_urdf_path: str,
        device: torch.device,
        q_min: Sequence[float],
        q_max: Sequence[float],
        projection_iters: int = 10,
        projection_damping: float = 0.5,
        projection_epsilon_f: float = 0.03,
        projection_max_step_norm: float = 0.25,
        root_refine_iters: int = 12,
        root_tolerance_f: float = 0.002,
        branch_ascent_steps: int = 1,
        branch_step_size: float = 0.05,
        branch_max_step_norm: float = 0.25,
        branch_fallback_ascent_steps: int = 8,
        max_branch_attempts: int = 4,
        min_conservative_g: float = 0.0,
        conservative_hfov_deg: float = 50.0,
        conservative_vfov_deg: float = 66.0,
        conservative_z_min: float = 0.20,
        conservative_z_max: float = 0.70,
        conservative_delta: float = 0.01,
        nominal_hfov_deg: float = 55.0,
        nominal_vfov_deg: float = 72.0,
        nominal_z_min: float = 0.15,
        nominal_z_max: float = 0.75,
        require_primitive_los: bool = True,
    ) -> None:
        if projection_iters < 1:
            raise ValueError("projection_iters must be >= 1")
        if not 0.0 < projection_damping <= 1.0:
            raise ValueError("projection_damping must be in (0,1]")
        if projection_epsilon_f <= 0.0:
            raise ValueError("projection_epsilon_f must be positive")
        if projection_max_step_norm <= 0.0:
            raise ValueError("projection_max_step_norm must be positive")
        if root_refine_iters < 0:
            raise ValueError("root_refine_iters must be >= 0")
        if root_tolerance_f <= 0.0:
            raise ValueError("root_tolerance_f must be positive")
        if branch_ascent_steps < 1:
            raise ValueError("branch_ascent_steps must be >= 1")
        if branch_step_size <= 0.0:
            raise ValueError("branch_step_size must be positive")
        if branch_max_step_norm <= 0.0:
            raise ValueError("branch_max_step_norm must be positive")
        if branch_fallback_ascent_steps < 1:
            raise ValueError("branch_fallback_ascent_steps must be >= 1")
        if max_branch_attempts < 1 or max_branch_attempts > 8:
            raise ValueError("max_branch_attempts must be in [1,8]")

        self.device = device
        self.checkpoint_path = str(checkpoint_path)
        self.reference_urdf_path = str(reference_urdf_path)
        self.self_filter_urdf_path = str(self_filter_urdf_path)

        # Mirror the frozen scalar projector/root/ascent semantics.
        self.projection_iters = int(projection_iters)
        self.projection_damping = float(projection_damping)
        self.projection_epsilon_f = float(projection_epsilon_f)
        self.projection_max_step_norm = float(projection_max_step_norm)
        self.root_refine_iters = int(root_refine_iters)
        self.root_tolerance_f = float(root_tolerance_f)
        self.branch_ascent_steps = int(branch_ascent_steps)
        self.branch_step_size = float(branch_step_size)
        self.branch_max_step_norm = float(branch_max_step_norm)
        self.branch_fallback_ascent_steps = int(branch_fallback_ascent_steps)

        self.max_branch_attempts = int(max_branch_attempts)
        self.min_conservative_g = float(min_conservative_g)
        self.require_primitive_los = bool(require_primitive_los)

        self.cons_hfov = float(conservative_hfov_deg)
        self.cons_vfov = float(conservative_vfov_deg)
        self.cons_z_min = float(conservative_z_min)
        self.cons_z_max = float(conservative_z_max)
        self.cons_delta = float(conservative_delta)
        self.nom_hfov = float(nominal_hfov_deg)
        self.nom_vfov = float(nominal_vfov_deg)
        self.nom_z_min = float(nominal_z_min)
        self.nom_z_max = float(nominal_z_max)

        self.q_min_np = np.asarray(q_min, dtype=np.float64).reshape(7)
        self.q_max_np = np.asarray(q_max, dtype=np.float64).reshape(7)
        self.q_min = torch.tensor(
            self.q_min_np, device=self.device, dtype=torch.float32
        )
        self.q_max = torch.tensor(
            self.q_max_np, device=self.device, dtype=torch.float32
        )

        self.model, self.checkpoint = build_per_sensor_model(
            self.checkpoint_path, self.device
        )

        # Sensor poses come from the reference robot.  The dedicated self-filter
        # URDF intentionally contains only body collision primitives.
        self.reference_robot = URDF.from_xml_file(self.reference_urdf_path)
        self.self_filter_robot = URDF.from_xml_file(self.self_filter_urdf_path)
        self.sensor_chains = [
            find_chain_joints(self.reference_robot, "base_link", frame)
            for frame in DEFAULT_SENSOR_FRAMES
        ]
        self.primitives = load_collision_primitives(self.self_filter_robot)
        if not self.primitives:
            raise RuntimeError(
                "self-filter URDF contains no box/cylinder/sphere primitives"
            )
        self.collision_chains = [
            find_chain_joints(
                self.self_filter_robot, "base_link", primitive["link"]
            )
            for primitive in self.primitives
        ]

        joint_index = {
            str(name): idx for idx, name in enumerate(DEFAULT_JOINT_NAMES)
        }
        masks = np.zeros((8, 7), dtype=np.float32)
        for s, chain in enumerate(self.sensor_chains):
            for joint in chain:
                name = str(getattr(joint, "name", ""))
                if name in joint_index:
                    masks[s, joint_index[name]] = 1.0
        self.sensor_masks_np = masks
        self.sensor_masks = torch.tensor(
            masks, device=self.device, dtype=torch.float32
        )

    def warmup(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()
        x = torch.zeros((4, 3), device=self.device, dtype=torch.float32)
        q = 0.5 * (self.q_min + self.q_max)
        for _ in range(2):
            self.rank_sensors(x, q)
            self.branch_value_and_grad(x, q, 7)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return 1000.0 * (time.perf_counter() - tic)

    def _clamp(self, q: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        q_clamped = torch.maximum(
            torch.minimum(q, self.q_max[None, :]), self.q_min[None, :]
        )
        changed = bool(torch.any(torch.abs(q_clamped - q) > 1e-10).item())
        return q_clamped.detach(), changed

    @torch.no_grad()
    def _head_values(
        self, points: torch.Tensor, q: torch.Tensor
    ) -> torch.Tensor:
        points = points.reshape(-1, 3)
        q = q.reshape(1, 7)
        q_batch = q.expand(points.shape[0], -1)
        pred = self.model(torch.cat([points, q_batch], dim=-1))
        if pred.ndim != 2 or pred.shape[1] != 8:
            raise RuntimeError(
                "per-sensor model returned shape {}".format(tuple(pred.shape))
            )
        return pred

    @torch.no_grad()
    def rank_sensors(
        self, points: torch.Tensor, q: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        pred = self._head_values(points, q)
        scores = pred.min(dim=0).values
        order = torch.argsort(scores, descending=True)
        return (
            scores.detach().cpu().numpy().astype(np.float64),
            order.detach().cpu().numpy().astype(np.int64),
        )

    def branch_value_and_grad(
        self, points: torch.Tensor, q: torch.Tensor, sensor_id: int
    ) -> Tuple[float, torch.Tensor]:
        sensor_id = int(sensor_id)
        if sensor_id < 0 or sensor_id >= 8:
            raise ValueError("sensor_id out of range")

        points = points.detach().reshape(-1, 3)
        q_var = q.detach().clone().reshape(1, 7).requires_grad_(True)
        q_batch = q_var.expand(points.shape[0], -1)
        pred = self.model(torch.cat([points, q_batch], dim=-1))[:, sensor_id]
        value = torch.min(pred)
        grad = torch.autograd.grad(
            value,
            q_var,
            grad_outputs=torch.ones_like(value),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        # The target gradient for each sensor is chain-masked during training.
        # Enforce the same physical dependency online so approximation noise
        # cannot drive joints downstream of the selected sensor.
        grad = grad * self.sensor_masks[sensor_id : sensor_id + 1]
        return float(value.detach().item()), grad.detach()

    @torch.no_grad()
    def branch_score(
        self, points: torch.Tensor, q: torch.Tensor, sensor_id: int
    ) -> float:
        pred = self._head_values(points, q)[:, int(sensor_id)]
        return float(torch.min(pred).item())

    def branch_score_numpy(
        self, points_xyz, q_row, sensor_id: int
    ) -> float:
        points = torch.tensor(
            np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3),
            device=self.device,
            dtype=torch.float32,
        )
        q = torch.tensor(
            np.asarray(q_row, dtype=np.float32).reshape(1, 7),
            device=self.device,
            dtype=torch.float32,
        )
        return self.branch_score(points, q, int(sensor_id))

    def _projection_step(
        self,
        q: torch.Tensor,
        score: float,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        grad_norm_sq = torch.sum(grad * grad, dim=-1, keepdim=True)
        grad_norm_sq_value = float(grad_norm_sq[0, 0].item())
        if (not math.isfinite(grad_norm_sq_value)
                or grad_norm_sq_value < 1e-12):
            return q.detach(), {
                "degenerate": True,
                "grad_norm": math.sqrt(max(0.0, grad_norm_sq_value)),
                "raw_step_norm": math.nan,
                "applied_step_norm": 0.0,
                "clipped": False,
                "joint_limit_clamped": False,
            }

        raw_step = (
            float(score) * grad / torch.clamp(grad_norm_sq, min=1e-8)
        )
        raw_norm = float(torch.linalg.vector_norm(raw_step[0]).item())
        scale = min(
            1.0,
            self.projection_max_step_norm / max(raw_norm, 1e-8),
        )
        clipped_step = raw_step * scale
        applied = self.projection_damping * clipped_step
        q_next, joint_clamped = self._clamp(q - applied)
        return q_next, {
            "degenerate": False,
            "grad_norm": float(
                torch.linalg.vector_norm(grad[0]).item()
            ),
            "raw_step_norm": raw_norm,
            "applied_step_norm": float(
                torch.linalg.vector_norm(applied[0]).item()
            ),
            "clipped": bool(raw_norm > self.projection_max_step_norm),
            "joint_limit_clamped": bool(joint_clamped),
        }

    def _ascent_step(
        self,
        q: torch.Tensor,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        grad_norm = float(torch.linalg.vector_norm(grad[0]).item())
        if not math.isfinite(grad_norm) or grad_norm < 1e-8:
            return q.detach(), {
                "degenerate": True,
                "grad_norm": grad_norm,
                "applied_step_norm": 0.0,
                "joint_limit_clamped": False,
            }

        direction = grad / grad_norm
        raw = self.branch_step_size * direction
        raw_norm = float(torch.linalg.vector_norm(raw[0]).item())
        scale = min(
            1.0,
            self.branch_max_step_norm / max(raw_norm, 1e-8),
        )
        applied = raw * scale
        q_next, joint_clamped = self._clamp(q + applied)
        return q_next, {
            "degenerate": False,
            "grad_norm": grad_norm,
            "applied_step_norm": float(
                torch.linalg.vector_norm(applied[0]).item()
            ),
            "joint_limit_clamped": bool(joint_clamped),
        }

    def _refine_branch_root(
        self,
        points: torch.Tensor,
        sensor_id: int,
        qa: torch.Tensor,
        fa: float,
        qb: torch.Tensor,
        fb: float,
    ) -> Tuple[torch.Tensor, float, List[Dict[str, object]]]:
        lo = qa.detach().clone()
        hi = qb.detach().clone()
        flo = float(fa)
        fhi = float(fb)
        history: List[Dict[str, object]] = []

        if abs(flo) <= abs(fhi):
            best_q, best_f = lo.clone(), flo
        else:
            best_q, best_f = hi.clone(), fhi

        for iteration in range(1, self.root_refine_iters + 1):
            mid = 0.5 * (lo + hi)
            fm = self.branch_score(points, mid, sensor_id)
            history.append({
                "iter": int(iteration),
                "score": float(fm),
                "q": mid[0].detach().cpu().numpy().astype(float).tolist(),
            })
            if abs(fm) < abs(best_f):
                best_q, best_f = mid.detach().clone(), float(fm)
            if abs(fm) <= self.root_tolerance_f:
                return mid.detach(), float(fm), history

            if flo * fm <= 0.0:
                hi, fhi = mid.detach(), float(fm)
            else:
                lo, flo = mid.detach(), float(fm)

        return best_q.detach(), float(best_f), history

    def _optimize_branch(
        self, points: torch.Tensor, q_start: torch.Tensor, sensor_id: int
    ) -> Dict[str, object]:
        """Legacy-scalar-style per-head projection -> root -> ascent."""
        sensor_id = int(sensor_id)
        q0, initial_clamped = self._clamp(
            q_start.detach().clone().reshape(1, 7)
        )
        q = q0.clone()

        initial_score = self.branch_score(points, q, sensor_id)
        best_q = q.clone()
        best_score = float(initial_score)

        projection_history: List[Dict[str, object]] = [{
            "iter": 0,
            "score": float(initial_score),
            "grad_norm": math.nan,
            "joint_limit_clamped": bool(initial_clamped),
            "q": q[0].detach().cpu().numpy().astype(float).tolist(),
        }]
        root_history: List[Dict[str, object]] = []
        ascent_history: List[Dict[str, object]] = []

        q_zero = None
        f_zero = math.nan
        root_source = "none"
        f_current = float(initial_score)

        if f_current >= 0.0:
            q_zero = q.clone()
            f_zero = f_current
            root_source = "initial_branch_positive"
        elif abs(f_current) <= self.projection_epsilon_f:
            q_zero = q.clone()
            f_zero = f_current
            root_source = "initial_branch_tolerance"

        for iteration in range(1, self.projection_iters + 1):
            if q_zero is not None:
                break

            score, grad = self.branch_value_and_grad(points, q, sensor_id)
            q_next, diag = self._projection_step(q, score, grad)
            if bool(diag["degenerate"]):
                root_source = "degenerate_projection_gradient"
                break

            f_next = self.branch_score(points, q_next, sensor_id)
            if f_next > best_score:
                best_score = float(f_next)
                best_q = q_next.clone()

            projection_history.append({
                "iter": int(iteration),
                "score": float(f_next),
                "grad_norm": float(diag["grad_norm"]),
                "raw_step_norm": float(diag["raw_step_norm"]),
                "applied_step_norm": float(diag["applied_step_norm"]),
                "algorithm_step_clipped": bool(diag["clipped"]),
                "joint_limit_clamped": bool(diag["joint_limit_clamped"]),
                "q": q_next[0].detach().cpu().numpy().astype(float).tolist(),
            })

            if f_current * f_next <= 0.0 and f_current != f_next:
                q_zero, f_zero, root_history = self._refine_branch_root(
                    points, sensor_id, q, f_current, q_next, f_next
                )
                root_source = "branch_sign_crossing_bisection"
                break

            if abs(f_next) <= self.projection_epsilon_f:
                q_zero = q_next.detach()
                f_zero = float(f_next)
                root_source = "branch_projection_tolerance"
                break

            q = q_next.detach()
            f_current = float(f_next)

        if q_zero is not None:
            q_vis = q_zero.clone()
            for step in range(1, self.branch_ascent_steps + 1):
                score, grad = self.branch_value_and_grad(
                    points, q_vis, sensor_id
                )
                q_next, diag = self._ascent_step(q_vis, grad)
                if bool(diag["degenerate"]):
                    break
                q_vis = q_next.detach()
                f_next = self.branch_score(points, q_vis, sensor_id)
                best_score = max(best_score, float(f_next))
                ascent_history.append({
                    "step": int(step),
                    "score": float(f_next),
                    "grad_norm": float(diag["grad_norm"]),
                    "applied_step_norm": float(diag["applied_step_norm"]),
                    "joint_limit_clamped": bool(
                        diag["joint_limit_clamped"]
                    ),
                    "fallback": False,
                    "q": q_vis[0]
                    .detach().cpu().numpy().astype(float).tolist(),
                })
            solution_mode = "branch_projection_root_ascent"
        else:
            # Exactly mirror the legacy scalar active-set fallback: keep the
            # best projection iterate, then run a short best-effort ascent.
            q_vis = best_q.clone()
            best_shared_q = q_vis.clone()
            best_shared_score = float(best_score)
            for step in range(1, self.branch_fallback_ascent_steps + 1):
                score, grad = self.branch_value_and_grad(
                    points, q_vis, sensor_id
                )
                q_next, diag = self._ascent_step(q_vis, grad)
                if bool(diag["degenerate"]):
                    break
                f_next = self.branch_score(points, q_next, sensor_id)
                ascent_history.append({
                    "step": int(step),
                    "score": float(f_next),
                    "grad_norm": float(diag["grad_norm"]),
                    "applied_step_norm": float(diag["applied_step_norm"]),
                    "joint_limit_clamped": bool(
                        diag["joint_limit_clamped"]
                    ),
                    "fallback": True,
                    "q": q_next[0]
                    .detach().cpu().numpy().astype(float).tolist(),
                })
                q_vis = q_next.detach()
                if f_next > best_shared_score:
                    best_shared_score = float(f_next)
                    best_shared_q = q_vis.clone()

            q_vis = best_shared_q
            best_score = max(best_score, best_shared_score)
            q_zero = best_q.clone()
            f_zero = float(self.branch_score(points, q_zero, sensor_id))
            if root_source == "none":
                root_source = "branch_root_not_found"
            solution_mode = "branch_best_effort_ascent"

        final_score = self.branch_score(points, q_vis, sensor_id)

        return {
            "sensor_id": sensor_id,
            "sensor_frame": DEFAULT_SENSOR_FRAMES[sensor_id],
            "initial_score": float(initial_score),
            "best_score": float(best_score),
            "final_score": float(final_score),
            "q_start": q0[0].detach().cpu().numpy().astype(float).tolist(),
            "q_zero": q_zero[0]
                .detach().cpu().numpy().astype(float).tolist(),
            "f_zero": float(f_zero),
            "q_candidate": q_vis[0]
                .detach().cpu().numpy().astype(float).tolist(),
            "root_source": str(root_source),
            "solution_mode": str(solution_mode),
            "projection_history": projection_history,
            "root_history": root_history,
            "ascent_history": ascent_history,
        }

    def _candidate_geometry(
        self, points_xyz, q_row, sensor_id: int
    ) -> Dict[str, object]:
        points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        q = np.asarray(q_row, dtype=np.float64).reshape(7)
        q_map = q_row_to_map(DEFAULT_JOINT_NAMES, q)
        sensor_chain = self.sensor_chains[int(sensor_id)]
        sensor_transform = fk_transform(sensor_chain, q_map)

        point_rows = []
        min_cons_g = math.inf
        min_nom_margin = math.inf
        any_occluded = False

        for point in points:
            cons_margin, cons_plane, _ = sensor_margin(
                point,
                sensor_transform,
                self.cons_hfov,
                self.cons_vfov,
                self.cons_z_min,
                self.cons_z_max,
            )
            cons_g = float(cons_margin - self.cons_delta)
            nom_margin, nom_plane, _ = sensor_margin(
                point,
                sensor_transform,
                self.nom_hfov,
                self.nom_vfov,
                self.nom_z_min,
                self.nom_z_max,
            )

            occluded = False
            hit = None
            if self.require_primitive_los:
                occluded, hit = raycast_self_occlusion(
                    self.self_filter_robot,
                    self.collision_chains,
                    self.primitives,
                    sensor_transform,
                    point,
                    q_map,
                    _RayArgs(),
                )

            min_cons_g = min(min_cons_g, cons_g)
            min_nom_margin = min(min_nom_margin, float(nom_margin))
            any_occluded = bool(any_occluded or occluded)
            point_rows.append({
                "point_xyz": [float(v) for v in point],
                "conservative_g": cons_g,
                "conservative_plane": int(cons_plane),
                "nominal_margin": float(nom_margin),
                "nominal_plane": int(nom_plane),
                "primitive_self_occluded": bool(occluded),
                "primitive_hit": hit,
            })

        accepted = bool(
            len(point_rows) > 0
            and math.isfinite(min_cons_g)
            and min_cons_g + 1e-12 >= self.min_conservative_g
            and (not self.require_primitive_los or not any_occluded)
        )
        if min_cons_g < self.min_conservative_g:
            reject_reason = "conservative_fov"
        elif self.require_primitive_los and any_occluded:
            reject_reason = "primitive_self_occlusion"
        else:
            reject_reason = "accepted"

        return {
            "accepted": accepted,
            "reject_reason": reject_reason,
            "min_conservative_g": float(min_cons_g),
            "min_nominal_margin": float(min_nom_margin),
            "any_primitive_self_occluded": bool(any_occluded),
            "per_point": point_rows,
        }

    def generate(
        self,
        points_xyz,
        q_zero_row,
        branch_seed_row=None,
    ) -> Dict[str, object]:
        """Return first FOV+LOS-certified sensor branch.

        Sensor ranking is performed at scalar q_zero.  Every branch solve starts
        independently from branch_seed_row; if omitted, q_zero is used.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tic = time.perf_counter()

        points_np = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        q_zero_np = np.asarray(q_zero_row, dtype=np.float64).reshape(7)
        branch_seed_np = (
            q_zero_np.copy()
            if branch_seed_row is None
            else np.asarray(branch_seed_row, dtype=np.float64).reshape(7)
        )

        points = torch.tensor(
            points_np, device=self.device, dtype=torch.float32
        )
        q_zero = torch.tensor(
            q_zero_np.reshape(1, 7), device=self.device, dtype=torch.float32
        )
        branch_seed = torch.tensor(
            branch_seed_np.reshape(1, 7),
            device=self.device,
            dtype=torch.float32,
        )

        scores, order = self.rank_sensors(points, q_zero)
        seed_scores, _ = self.rank_sensors(points, branch_seed)
        attempts = []
        selected = None

        for rank, sensor_id in enumerate(
            order[: self.max_branch_attempts].tolist(), start=1
        ):
            branch = self._optimize_branch(
                points, branch_seed, int(sensor_id)
            )
            geometry = self._candidate_geometry(
                points_np, branch["q_candidate"], int(sensor_id)
            )
            attempt = dict(branch)
            attempt["rank"] = int(rank)
            attempt["geometry"] = geometry
            attempts.append(attempt)
            if bool(geometry["accepted"]):
                selected = attempt
                break

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        compute_ms = 1000.0 * (time.perf_counter() - tic)

        return {
            "enabled": True,
            "accepted": bool(selected is not None),
            "selected_sensor_id": (
                int(selected["sensor_id"]) if selected is not None else -1
            ),
            "selected_sensor_frame": (
                str(selected["sensor_frame"]) if selected is not None else "none"
            ),
            "selected_rank": (
                int(selected["rank"]) if selected is not None else -1
            ),
            "selected_q_vis": (
                list(selected["q_candidate"]) if selected is not None else None
            ),
            "scalar_q_zero": q_zero_np.astype(float).tolist(),
            "branch_seed_q": branch_seed_np.astype(float).tolist(),
            "scores_at_scalar_q_zero": [float(v) for v in scores.tolist()],
            "scores_at_branch_seed": [float(v) for v in seed_scores.tolist()],
            "ranking": [int(v) for v in order.tolist()],
            "ranking_frames": [
                DEFAULT_SENSOR_FRAMES[int(v)] for v in order.tolist()
            ],
            "attempts": attempts,
            "rejected_sensor_ids": [
                int(a["sensor_id"])
                for a in attempts
                if not bool(a["geometry"]["accepted"])
            ],
            "compute_ms": float(compute_ms),
            "config": {
                "branch_solver": (
                    "per_head_projection_root_ascent_from_branch_seed"
                ),
                "projection_iters": self.projection_iters,
                "projection_damping": self.projection_damping,
                "projection_epsilon_f": self.projection_epsilon_f,
                "projection_max_step_norm": (
                    self.projection_max_step_norm
                ),
                "root_refine_iters": self.root_refine_iters,
                "root_tolerance_f": self.root_tolerance_f,
                "branch_ascent_steps": self.branch_ascent_steps,
                "branch_step_size": self.branch_step_size,
                "branch_max_step_norm": self.branch_max_step_norm,
                "branch_fallback_ascent_steps": (
                    self.branch_fallback_ascent_steps
                ),
                "max_branch_attempts": self.max_branch_attempts,
                "min_conservative_g": self.min_conservative_g,
                "require_primitive_los": self.require_primitive_los,
                "self_filter_padding_m": 0.0,
                "checkpoint": self.checkpoint_path,
                "checkpoint_step": int(self.checkpoint.get("step", -1)),
            },
        }
