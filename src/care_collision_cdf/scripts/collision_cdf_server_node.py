#!/usr/bin/env python3
"""ROS batch inference server for CAREPlanner collision/forbidden-space CDF."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import rospy
import torch

from care_collision_cdf.srv import (
    QueryCollisionCDF,
    QueryCollisionCDFResponse,
    QueryCollisionCDFPairs,
    QueryCollisionCDFPairsResponse,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from collision_cdf_model import CollisionCDF  # noqa: E402


class CollisionCDFServer:
    def __init__(self):
        cfg = rospy.get_param("~collision_cdf", {})
        checkpoint = str(rospy.get_param("~checkpoint", "")).strip()
        if not checkpoint:
            raise ValueError("~checkpoint is required")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"collision CDF checkpoint not found: {checkpoint}. "
                "Place it under care_collision_cdf/checkpoints/yiming_cdf/model_dict.pt"
            )

        device = str(rospy.get_param("~device", cfg.get("device", "cuda")))
        checkpoint_key = str(
            rospy.get_param("~checkpoint_key", cfg.get("checkpoint_key", "latest"))
        )
        hidden = cfg.get("hidden_layers", [1024, 512, 256, 128, 128])
        self.max_points = int(cfg.get("max_points", 4096))
        self.max_q = int(cfg.get("max_q", 64))
        self.max_pairs = int(cfg.get("max_pairs", 8000))
        self.service_name = str(
            cfg.get("service_name", "/care_planner/collision_cdf/query")
        )
        self.pair_service_name = str(
            cfg.get(
                "pair_service_name",
                "/care_planner/collision_cdf/query_pairs",
            )
        )

        self.cdf = CollisionCDF(
            checkpoint_path=checkpoint,
            device=device,
            checkpoint_key=checkpoint_key,
            input_dims=int(cfg.get("input_dims", 10)),
            output_dims=int(cfg.get("output_dims", 1)),
            hidden_layers=hidden,
            nerf=bool(cfg.get("nerf", True)),
            activation=str(cfg.get("activation", "gelu")),
        )

        # Warm up both forward and autograd before accepting MPC queries.
        with torch.enable_grad():
            p = torch.zeros((4, 3), device=self.cdf.device)
            q = torch.zeros((2, 7), device=self.cdf.device)
            self.cdf.scene_distance_and_gradient(p, q)
            self.cdf.pair_distance_and_gradient(
                torch.zeros((8, 3), device=self.cdf.device),
                torch.zeros((8, 7), device=self.cdf.device),
            )
        if self.cdf.device.type == "cuda":
            torch.cuda.synchronize(self.cdf.device)

        self.server = rospy.Service(
            self.service_name, QueryCollisionCDF, self._query_callback
        )
        self.pair_server = rospy.Service(
            self.pair_service_name,
            QueryCollisionCDFPairs,
            self._pair_query_callback,
        )
        rospy.logwarn(
            "[collision_cdf] READY checkpoint=%s selected=%s device=%s "
            "activation=%s scene_service=%s pair_service=%s max_pairs=%d",
            checkpoint,
            self.cdf.selected_checkpoint,
            str(self.cdf.device),
            self.cdf.architecture.get("activation", "unknown"),
            self.service_name,
            self.pair_service_name,
            self.max_pairs,
        )

    def _query_callback(self, req):
        res = QueryCollisionCDFResponse()
        n_points = len(req.points)
        n_q = int(req.num_q)
        if n_points <= 0:
            res.success = False
            res.message = "points must be non-empty"
            return res
        if n_points > self.max_points:
            res.success = False
            res.message = f"too many points: {n_points} > {self.max_points}"
            return res
        if n_q <= 0 or n_q > self.max_q:
            res.success = False
            res.message = f"num_q must be in [1,{self.max_q}]"
            return res
        if len(req.q_flat) != n_q * 7:
            res.success = False
            res.message = f"q_flat length {len(req.q_flat)} != num_q*7={n_q*7}"
            return res

        points = np.asarray(
            [[p.x, p.y, p.z] for p in req.points], dtype=np.float32
        )
        q = np.asarray(req.q_flat, dtype=np.float32).reshape(n_q, 7)
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(q)):
            res.success = False
            res.message = "non-finite point or q input"
            return res

        try:
            if self.cdf.device.type == "cuda":
                torch.cuda.synchronize(self.cdf.device)
            t0 = time.perf_counter()
            with torch.enable_grad():
                d, grad, argmin = self.cdf.scene_distance_and_gradient(
                    torch.from_numpy(points), torch.from_numpy(q)
                )
            if self.cdf.device.type == "cuda":
                torch.cuda.synchronize(self.cdf.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            res.success = True
            res.message = "ok"
            res.min_distance = d.cpu().numpy().astype(np.float64).tolist()
            res.min_gradient_flat = (
                grad.cpu().numpy().astype(np.float64).reshape(-1).tolist()
            )
            res.argmin_point_index = argmin.cpu().numpy().astype(np.int32).tolist()
            res.inference_ms = float(elapsed_ms)
            return res
        except Exception as exc:
            rospy.logerr_throttle(1.0, "[collision_cdf] query failed: %s", exc)
            res.success = False
            res.message = str(exc)
            return res



    def _pair_query_callback(self, req):
        res = QueryCollisionCDFPairsResponse()
        n_pairs = int(req.num_pairs)

        if n_pairs <= 0 or n_pairs > self.max_pairs:
            res.success = False
            res.message = f"num_pairs must be in [1,{self.max_pairs}]"
            return res
        if len(req.points) != n_pairs:
            res.success = False
            res.message = (
                f"points length {len(req.points)} != num_pairs={n_pairs}"
            )
            return res
        if len(req.q_flat) != n_pairs * 7:
            res.success = False
            res.message = (
                f"q_flat length {len(req.q_flat)} != "
                f"num_pairs*7={n_pairs*7}"
            )
            return res

        points = np.asarray(
            [[p.x, p.y, p.z] for p in req.points], dtype=np.float32
        )
        q = np.asarray(req.q_flat, dtype=np.float32).reshape(n_pairs, 7)

        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(q)):
            res.success = False
            res.message = "non-finite point or q input"
            return res

        try:
            if self.cdf.device.type == "cuda":
                torch.cuda.synchronize(self.cdf.device)
            t0 = time.perf_counter()
            with torch.enable_grad():
                d, grad = self.cdf.pair_distance_and_gradient(
                    torch.from_numpy(points), torch.from_numpy(q)
                )
            if self.cdf.device.type == "cuda":
                torch.cuda.synchronize(self.cdf.device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            res.success = True
            res.message = "ok"
            res.distance = d.cpu().numpy().astype(np.float64).tolist()
            res.gradient_flat = (
                grad.cpu().numpy().astype(np.float64).reshape(-1).tolist()
            )
            res.inference_ms = float(elapsed_ms)
            return res
        except Exception as exc:
            rospy.logerr_throttle(
                1.0, "[collision_cdf] pair query failed: %s", exc
            )
            res.success = False
            res.message = str(exc)
            return res


def main():
    rospy.init_node("collision_cdf_server")
    CollisionCDFServer()
    rospy.spin()


if __name__ == "__main__":
    main()
