#!/usr/bin/env python3

import argparse
import json
import math
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "wrist_joint1",
    "wrist_joint2",
    "wrist_joint3",
]

DEFAULT_SENSOR_FRAMES = [
    "link2_sensor1_tof_link",
    "link2_sensor2_tof_link",
    "link3_sensor1_tof_link",
    "link3_sensor2_tof_link",
    "link4_sensor1_tof_link",
    "link4_sensor2_tof_link",
    "EE_sensor1_tof_link",
    "EE_sensor2_tof_link",
]


def sync_if_cuda(device: torch.device):
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()


def _parse_mlp_layers(value):
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


class MLP(nn.Module):
    """
    Yiming CDF MLPRegression-compatible model.

    Important details copied from yimingli1998/cdf/frankaemika/mlp.py:
      - if nerf=True: x -> cat([x, sin(x), cos(x)]) only, no multi-frequency bands
      - input_dims becomes 3 * input_dims
      - mlp_layers default [1024, 512, 256, 128, 128]
      - activation default ReLU
      - no manual reset/init is called by default, matching the training script behavior
      - skip connections are supported but default skips=[] as in nn_cdf.py
    """
    def __init__(
        self,
        in_dim: int = 10,
        hidden_dim: int = 256,              # kept only for backward-compatible argparse/checkpoints
        num_hidden_layers: int = 4,         # kept only for backward-compatible argparse/checkpoints
        out_dim: int = 1,
        activation: str = "relu",
        model_arch: str = "yiming",
        mlp_layers=(1024, 512, 256, 128, 128),
        skips=(),
        nerf: bool = True,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.model_arch = model_arch
        self.nerf = bool(nerf)
        self.skips = tuple(int(s) for s in skips)
        self.mlp_layers = _parse_mlp_layers(mlp_layers)

        if activation != "relu":
            raise ValueError("For Yiming-compatible training, activation must be 'relu'.")
        act_fn = nn.ReLU

        encoded_dim = 3 * self.in_dim if self.nerf else self.in_dim
        self.encoded_dim = encoded_dim

        mlp_arr = []
        layers = list(self.mlp_layers)
        skips = list(self.skips)

        # Same structural logic as Yiming's MLPRegression. For our default skips=[],
        # this reduces to a plain sequential MLP: encoded_dim -> layers -> out_dim.
        if len(skips) > 0:
            mlp_arr.append(layers[0:skips[0]])
            mlp_arr[0][-1] -= encoded_dim
            for si in range(1, len(skips)):
                mlp_arr.append(layers[skips[si - 1]:skips[si]])
                mlp_arr[-1][-1] -= encoded_dim
            mlp_arr.append(layers[skips[-1]:])
        else:
            mlp_arr.append(layers)

        mlp_arr[-1].append(self.out_dim)
        mlp_arr[0].insert(0, encoded_dim)

        self.layers = nn.ModuleList()
        for arr in mlp_arr[:-1]:
            self.layers.append(self._make_mlp(arr, act_fn=act_fn, islast=False))
        self.layers.append(self._make_mlp(mlp_arr[-1], act_fn=act_fn, islast=True))

    @staticmethod
    def _make_mlp(channels, act_fn=nn.ReLU, islast=False):
        blocks = []
        if not islast:
            for i in range(1, len(channels)):
                blocks.append(nn.Sequential(nn.Linear(channels[i - 1], channels[i]), act_fn()))
        else:
            for i in range(1, len(channels) - 1):
                blocks.append(nn.Sequential(nn.Linear(channels[i - 1], channels[i]), act_fn()))
            blocks.append(nn.Sequential(nn.Linear(channels[-2], channels[-1])))
        return nn.Sequential(*blocks)

    def encode(self, x):
        if self.nerf:
            return torch.cat((x, torch.sin(x), torch.cos(x)), dim=-1)
        return x

    def forward(self, x):
        x_nerf = self.encode(x)
        y = self.layers[0](x_nerf)
        for layer in self.layers[1:]:
            y = layer(torch.cat((y, x_nerf), dim=1))
        return y


class VisibilityQ0Dataset:
    def __init__(self, path: str, val_count: int, seed: int):
        print(f"[data] loading {path}")
        d = np.load(path, allow_pickle=True)

        required = [
            "x",
            "q",
            "k",
            "valid_fov",
            "sensor_chain_masks",
        ]
        for key in required:
            if key not in d.files:
                raise RuntimeError(f"Dataset missing key: {key}")

        x = d["x"].astype(np.float32)
        qlib = d["q"].astype(np.float32)
        valid_fov = d["valid_fov"].astype(np.bool_)
        masks = d["sensor_chain_masks"].astype(np.float32)

        if "q_min" in d.files and "q_max" in d.files:
            q_min = d["q_min"].astype(np.float32)
            q_max = d["q_max"].astype(np.float32)
        else:
            q_min = np.full((7,), -math.pi, dtype=np.float32)
            q_max = np.full((7,), math.pi, dtype=np.float32)
            print("[WARN] q_min/q_max not found in dataset. Using [-pi, pi].")

        valid_point = np.any(valid_fov, axis=(1, 2))
        valid_indices = np.where(valid_point)[0]
        if len(valid_indices) == 0:
            raise RuntimeError("No valid FOV point found.")

        rng = np.random.default_rng(seed)
        rng.shuffle(valid_indices)

        val_count = min(val_count, max(1, len(valid_indices) // 10))
        self.val_indices_np = valid_indices[:val_count].copy()
        self.train_indices_np = valid_indices[val_count:].copy()

        if len(self.train_indices_np) == 0:
            raise RuntimeError("No training points after validation split.")

        self.x_cpu = torch.from_numpy(x)
        self.qlib_cpu = torch.from_numpy(qlib)
        self.valid_cpu = torch.from_numpy(valid_fov)
        self.sensor_masks_cpu = torch.from_numpy(masks)
        self.q_min_cpu = torch.from_numpy(q_min)
        self.q_max_cpu = torch.from_numpy(q_max)

        self.train_indices_cpu = torch.from_numpy(self.train_indices_np.astype(np.int64))
        self.val_indices_cpu = torch.from_numpy(self.val_indices_np.astype(np.int64))

        self.P = x.shape[0]
        self.K = qlib.shape[1]
        self.J = qlib.shape[2]
        self.S = qlib.shape[3]

        finite_q_slots = np.isfinite(qlib).all(axis=2).sum()
        valid_slots = valid_fov.sum()

        print("")
        print("=== Dataset Summary ===")
        print(f"path:                 {path}")
        print(f"x:                    {x.shape}")
        print(f"q:                    {qlib.shape}")
        print(f"valid_fov:            {valid_fov.shape}")
        print(f"sensor_chain_masks:   {masks.shape}")
        print(f"total points:         {self.P}")
        print(f"valid points:         {len(valid_indices)}")
        print(f"train points:         {len(self.train_indices_np)}")
        print(f"val points:           {len(self.val_indices_np)}")
        print(f"finite q slots:       {int(finite_q_slots)}")
        print(f"valid_fov slots:      {int(valid_slots)}")
        print(f"q_min:                {q_min.tolist()}")
        print(f"q_max:                {q_max.tolist()}")
        print("=======================")
        print("")

    def sample_x_batch(self, batch_x: int, split: str, device: torch.device):
        if split == "train":
            pool = self.train_indices_cpu
        elif split == "val":
            pool = self.val_indices_cpu
        else:
            raise ValueError(split)

        ridx = torch.randint(0, len(pool), (batch_x,), dtype=torch.long)
        idx = pool[ridx]

        x = self.x_cpu[idx].to(device=device, non_blocking=True)
        qlib = self.qlib_cpu[idx].to(device=device, non_blocking=True)
        valid = self.valid_cpu[idx].to(device=device, non_blocking=True)

        return x, qlib, valid, idx.to(device=device, non_blocking=True)

    def sensor_masks(self, device: torch.device):
        return self.sensor_masks_cpu.to(device=device)

    def q_limits(self, device: torch.device):
        return (
            self.q_min_cpu.to(device=device),
            self.q_max_cpu.to(device=device),
        )


class PinocchioFOVOracle:
    """
    FOV-only online sign oracle.

    This class intentionally reuses the same oracle used during q0 data generation:
        prepare_chain_specs(...)
        visibility_g_batch(...)

    This keeps signed labels consistent with the zero-level-set library.
    """

    def __init__(
        self,
        urdf_path: str,
        joint_names,
        sensor_frames,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        z_min: float,
        z_max: float,
        delta: float,
        base_frame: str = "base_link",
    ):
        from urdf_parser_py.urdf import URDF
        from extract_visibility_zero_level_sets import prepare_chain_specs, visibility_g_batch

        self.URDF = URDF
        self.prepare_chain_specs = prepare_chain_specs
        self.visibility_g_batch = visibility_g_batch

        self.urdf_path = urdf_path
        self.base_frame = base_frame
        self.joint_names = list(joint_names)
        self.sensor_frames = list(sensor_frames)

        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.vertical_fov_deg = float(vertical_fov_deg)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.delta = float(delta)

        self.robot = URDF.from_xml_file(urdf_path)

        self._chain_specs_cache = {}
        self._printed_fov_fast_path = False
        self._printed_fov_fallback = False
        self._printed_fov_exception = False

        print("")
        print("=== FOV Oracle ===")
        print(f"backend:               existing visibility_g_batch")
        print(f"urdf:                  {urdf_path}")
        print(f"base_frame:            {base_frame}")
        print(f"joint_names:           {self.joint_names}")
        print(f"sensor_frames:         {self.sensor_frames}")
        print(f"horizontal_fov_deg:    {horizontal_fov_deg}")
        print(f"vertical_fov_deg:      {vertical_fov_deg}")
        print(f"z_min/z_max:           {z_min}, {z_max}")
        print(f"delta:                 {delta}")
        print("==================")
        print("")

    def _get_chain_specs(self, device):
        key = str(device)
        if key not in self._chain_specs_cache:
            self._chain_specs_cache[key] = self.prepare_chain_specs(
                self.robot,
                self.base_frame,
                self.sensor_frames,
                self.joint_names,
                device,
            )
        return self._chain_specs_cache[key]

    @torch.no_grad()
    def signed_fov_margins(self, p_world: torch.Tensor, q_query: torch.Tensor):
        """
        p_world: [B_x, 3]
        q_query: [B_q, 7]

        returns:
            raw_margin: [B_x, B_q, S]
            sign:       [B_x, B_q, S]

        visibility_g_batch returns raw sensor margins.
        signed condition:
            raw_margin - delta >= 0
        """

        device = p_world.device
        all_chain_specs = self._get_chain_specs(device)

        Bx = p_world.shape[0]
        Bq = q_query.shape[0]
        S = len(self.sensor_frames)

        # Fast path: try batched p_world.
        try:
            _, sensor_margins, _, _ = self.visibility_g_batch(
                p_world,
                q_query,
                all_chain_specs,
                self.horizontal_fov_deg,
                self.vertical_fov_deg,
                self.z_min,
                self.z_max,
                self.delta,
            )

            if sensor_margins.ndim == 3:
                # Expected shape [B_x, B_q, S]
                if sensor_margins.shape[0] == Bx and sensor_margins.shape[1] == Bq:
                    raw_margin = sensor_margins.contiguous()
                    if not self._printed_fov_fast_path:
                        print(
                            f"[FOV Oracle] using batched fast path, raw_margin={tuple(raw_margin.shape)}",
                            flush=True,
                        )
                        self._printed_fov_fast_path = True
                    g = raw_margin - self.delta
                    sign = torch.where(g >= 0.0, torch.ones_like(g), -torch.ones_like(g))
                    return raw_margin, sign

                # Possible shape [B_q, B_x, S]
                if sensor_margins.shape[0] == Bq and sensor_margins.shape[1] == Bx:
                    raw_margin = sensor_margins.permute(1, 0, 2).contiguous()
                    if not self._printed_fov_fast_path:
                        print(
                            f"[FOV Oracle] using batched fast path with permute, raw_margin={tuple(raw_margin.shape)}",
                            flush=True,
                        )
                        self._printed_fov_fast_path = True
                    g = raw_margin - self.delta
                    sign = torch.where(g >= 0.0, torch.ones_like(g), -torch.ones_like(g))
                    return raw_margin, sign

            if sensor_margins.ndim == 2 and Bx == 1:
                raw_margin = sensor_margins[None, :, :].contiguous()
                if not self._printed_fov_fast_path:
                    print(
                        f"[FOV Oracle] using single-point fast path, raw_margin={tuple(raw_margin.shape)}",
                        flush=True,
                    )
                    self._printed_fov_fast_path = True
                g = raw_margin - self.delta
                sign = torch.where(g >= 0.0, torch.ones_like(g), -torch.ones_like(g))
                return raw_margin, sign

            if not self._printed_fov_exception:
                print(
                    f"[FOV Oracle] batched call returned unsupported sensor_margins shape "
                    f"{tuple(sensor_margins.shape)} for Bx={Bx}, Bq={Bq}; falling back.",
                    flush=True,
                )
                self._printed_fov_exception = True

        except Exception as e:
            if not self._printed_fov_exception:
                print(
                    f"[FOV Oracle] batched fast path failed with {type(e).__name__}: {e}; falling back.",
                    flush=True,
                )
                self._printed_fov_exception = True

        # Safe fallback:
        # Original visibility_g_batch definitely supports one point and a q batch.
        if not self._printed_fov_fallback:
            print(
                f"[FOV Oracle] using POINT-WISE FALLBACK, Bx={Bx}, Bq={Bq}. "
                f"This calls visibility_g_batch {Bx} times per batch and is expected to be slow.",
                flush=True,
            )
            self._printed_fov_fallback = True

        margins = []
        for i in range(Bx):
            _, sensor_margins_i, _, _ = self.visibility_g_batch(
                p_world[i],
                q_query,
                all_chain_specs,
                self.horizontal_fov_deg,
                self.vertical_fov_deg,
                self.z_min,
                self.z_max,
                self.delta,
            )

            if sensor_margins_i.ndim != 2:
                raise RuntimeError(
                    f"Expected sensor_margins_i shape [B_q,S], got {tuple(sensor_margins_i.shape)}"
                )

            if sensor_margins_i.shape[0] != Bq:
                raise RuntimeError(
                    f"Expected sensor_margins_i first dim B_q={Bq}, got {tuple(sensor_margins_i.shape)}"
                )

            margins.append(sensor_margins_i)

        raw_margin = torch.stack(margins, dim=0).contiguous()  # [B_x,B_q,S]

        if raw_margin.shape[-1] != S:
            raise RuntimeError(
                f"Expected last dim S={S}, got raw_margin shape {tuple(raw_margin.shape)}"
            )

        g = raw_margin - self.delta
        sign = torch.where(g >= 0.0, torch.ones_like(g), -torch.ones_like(g))

        return raw_margin, sign


@torch.no_grad()
def decode_per_sensor_distance_and_grad(
    qlib: torch.Tensor,
    valid: torch.Tensor,
    q_query: torch.Tensor,
    sensor_masks: torch.Tensor,
    x_chunk: int = 256,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    qlib:         [B_x, K, 7, 8]
    valid:        [B_x, K, 8]
    q_query:      [B_q, 7]
    sensor_masks: [8, 7]

    returns:
        d_s:         [B_x, B_q, 8]
        grad_d_s:    [B_x, B_q, 8, 7]
        has_sensor:  [B_x, 8]
    """

    Bx, K, J, S = qlib.shape
    Bq = q_query.shape[0]
    device = q_query.device

    d_all = torch.full((Bx, Bq, S), float("inf"), device=device, dtype=torch.float32)
    grad_all = torch.zeros((Bx, Bq, S, J), device=device, dtype=torch.float32)
    has_sensor = torch.any(valid, dim=1)  # [Bx,S]

    for start in range(0, Bx, x_chunk):
        end = min(start + x_chunk, Bx)
        C = end - start

        q0_c = qlib[start:end]       # [C,K,J,S]
        valid_c = valid[start:end]   # [C,K,S]

        for s in range(S):
            valid_s = valid_c[:, :, s]  # [C,K]
            if not torch.any(valid_s):
                continue

            q0_s = q0_c[:, :, :, s]     # [C,K,J]
            mask_s = sensor_masks[s]    # [J]

            diff = (q_query[None, :, None, :] - q0_s[:, None, :, :]) * mask_s[None, None, None, :]
            d2 = torch.sum(diff * diff, dim=-1)  # [C,Bq,K]
            d2 = d2.masked_fill(~valid_s[:, None, :], float("inf"))

            min_d2, min_k = torch.min(d2, dim=-1)  # [C,Bq]
            dist = torch.sqrt(torch.clamp(min_d2, min=eps))

            q0_exp = q0_s[:, None, :, :].expand(C, Bq, K, J).reshape(C * Bq, K, J)
            min_k_flat = min_k.reshape(C * Bq)
            q0_nearest = q0_exp[torch.arange(C * Bq, device=device), min_k_flat]
            q0_nearest = q0_nearest.reshape(C, Bq, J)

            diff_nearest = (q_query[None, :, :] - q0_nearest) * mask_s[None, None, :]
            grad = diff_nearest / dist[:, :, None]

            invalid = ~torch.isfinite(dist)
            if torch.any(invalid):
                dist = torch.where(invalid, torch.full_like(dist, float("inf")), dist)
                grad = torch.where(invalid[:, :, None], torch.zeros_like(grad), grad)

            d_all[start:end, :, s] = dist
            grad_all[start:end, :, s, :] = grad

    return d_all, grad_all, has_sensor


def sample_random_q(dataset: VisibilityQ0Dataset, batch_q: int, device: torch.device):
    q_min, q_max = dataset.q_limits(device=device)
    u = torch.rand((batch_q, dataset.J), device=device)
    return q_min[None, :] + u * (q_max - q_min)[None, :]


@torch.no_grad()
def sample_q_mixture(
    dataset: VisibilityQ0Dataset,
    qlib: torch.Tensor,
    valid: torch.Tensor,
    sensor_masks: torch.Tensor,
    batch_q: int,
    device: torch.device,
    near_zero_ratio: float,
    near_zero_std: float,
):
    """
    Yiming-style default is pure random q.

    Optional near-zero perturbation:
        q = q0 + N(0, near_zero_std) on the active chain of the selected sensor.
    """

    q = sample_random_q(dataset, batch_q, device=device)

    if near_zero_ratio <= 0.0:
        return q

    n_near = int(round(batch_q * near_zero_ratio))
    n_near = max(0, min(batch_q, n_near))
    if n_near == 0:
        return q

    valid_slots = torch.nonzero(valid, as_tuple=False)  # [N,3] = bx,k,s
    if valid_slots.numel() == 0:
        return q

    choice = torch.randint(0, valid_slots.shape[0], (n_near,), device=device)
    slots = valid_slots[choice]

    bx = slots[:, 0]
    kk = slots[:, 1]
    ss = slots[:, 2]

    q0 = qlib[bx, kk, :, ss]  # [n_near,7]
    masks = sensor_masks[ss]  # [n_near,7]

    noise = torch.randn_like(q0) * near_zero_std * masks
    q_near = q0 + noise

    q_min, q_max = dataset.q_limits(device=device)
    q_near = torch.maximum(torch.minimum(q_near, q_max[None, :]), q_min[None, :])

    q[:n_near] = q_near

    perm = torch.randperm(batch_q, device=device)
    return q[perm]


def make_input_pairs(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    Bx = x.shape[0]
    Bq = q.shape[0]

    x_rep = x[:, None, :].expand(Bx, Bq, 3)
    q_rep = q[None, :, :].expand(Bx, Bq, 7)

    return torch.cat([x_rep, q_rep], dim=-1).reshape(Bx * Bq, 10)



def make_pairwise_input_pairs(x: torch.Tensor, q_pair: torch.Tensor) -> torch.Tensor:
    """
    x:      [B_x, 3]
    q_pair: [B_x, R, 7]

    returns:
        inp: [B_x * R, 10], each row is (x_i, q_i_r)
    """
    Bx, R, J = q_pair.shape
    x_rep = x[:, None, :].expand(Bx, R, 3)
    return torch.cat([x_rep, q_pair], dim=-1).reshape(Bx * R, 10)


@torch.no_grad()
def sample_pairwise_near_q_per_x(
    dataset: VisibilityQ0Dataset,
    qlib: torch.Tensor,
    valid: torch.Tensor,
    sensor_masks: torch.Tensor,
    near_per_x: int,
    device: torch.device,
    near_zero_std: float,
):
    """
    For each x_i, sample near_per_x valid q0 slots from its own q0 library,
    then perturb only the active sensor chain.

    qlib:  [B_x, K, 7, S]
    valid: [B_x, K, S]

    returns:
        q_near: [B_x, near_per_x, 7]
    """
    Bx, K, J, S = qlib.shape
    if near_per_x <= 0:
        return torch.empty((Bx, 0, J), device=device, dtype=qlib.dtype)

    flat_valid = valid.reshape(Bx, K * S).float()
    row_sum = flat_valid.sum(dim=1)
    if torch.any(row_sum <= 0):
        raise RuntimeError("sample_pairwise_near_q_per_x got an x with no valid q0 slots.")

    # replacement=True is intentional: different near samples may start from the same q0.
    slot_flat = torch.multinomial(flat_valid, num_samples=near_per_x, replacement=True)  # [Bx,R]
    kk = torch.div(slot_flat, S, rounding_mode="floor")
    ss = slot_flat.remainder(S)

    bx_idx = torch.arange(Bx, device=device)[:, None].expand(Bx, near_per_x)
    q0 = qlib[bx_idx, kk, :, ss]      # [Bx,R,7]
    masks = sensor_masks[ss]          # [Bx,R,7]

    noise = torch.randn_like(q0) * near_zero_std * masks
    q_near = q0 + noise

    q_min, q_max = dataset.q_limits(device=device)
    q_near = torch.maximum(torch.minimum(q_near, q_max[None, None, :]), q_min[None, None, :])

    return q_near.contiguous()


@torch.no_grad()
def decode_pairwise_per_sensor_distance_and_grad(
    qlib: torch.Tensor,
    valid: torch.Tensor,
    q_pair: torch.Tensor,
    sensor_masks: torch.Tensor,
    x_chunk: int = 128,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    qlib:         [B_x, K, 7, S]
    valid:        [B_x, K, S]
    q_pair:       [B_x, R, 7]
    sensor_masks: [S, 7]

    returns:
        d_s:         [B_x, R, S]
        grad_d_s:    [B_x, R, S, 7]
        has_sensor:  [B_x, S]
    """
    Bx, K, J, S = qlib.shape
    R = q_pair.shape[1]
    device = q_pair.device

    d_all = torch.full((Bx, R, S), float("inf"), device=device, dtype=torch.float32)
    grad_all = torch.zeros((Bx, R, S, J), device=device, dtype=torch.float32)
    has_sensor = torch.any(valid, dim=1)  # [Bx,S]

    if R == 0:
        return d_all, grad_all, has_sensor

    for start in range(0, Bx, x_chunk):
        end = min(start + x_chunk, Bx)
        C = end - start

        q0_c = qlib[start:end]       # [C,K,J,S]
        valid_c = valid[start:end]   # [C,K,S]
        q_pair_c = q_pair[start:end] # [C,R,J]

        for s in range(S):
            valid_s = valid_c[:, :, s]  # [C,K]
            if not torch.any(valid_s):
                continue

            q0_s = q0_c[:, :, :, s]     # [C,K,J]
            mask_s = sensor_masks[s]    # [J]

            diff = (q_pair_c[:, :, None, :] - q0_s[:, None, :, :]) * mask_s[None, None, None, :]
            d2 = torch.sum(diff * diff, dim=-1)  # [C,R,K]
            d2 = d2.masked_fill(~valid_s[:, None, :], float("inf"))

            min_d2, min_k = torch.min(d2, dim=-1)  # [C,R]
            dist = torch.sqrt(torch.clamp(min_d2, min=eps))

            q0_exp = q0_s[:, None, :, :].expand(C, R, K, J).reshape(C * R, K, J)
            min_k_flat = min_k.reshape(C * R)
            q0_nearest = q0_exp[torch.arange(C * R, device=device), min_k_flat]
            q0_nearest = q0_nearest.reshape(C, R, J)

            diff_nearest = (q_pair_c - q0_nearest) * mask_s[None, None, :]
            grad = diff_nearest / dist[:, :, None]

            invalid = ~torch.isfinite(dist)
            if torch.any(invalid):
                dist = torch.where(invalid, torch.full_like(dist, float("inf")), dist)
                grad = torch.where(invalid[:, :, None], torch.zeros_like(grad), grad)

            d_all[start:end, :, s] = dist
            grad_all[start:end, :, s, :] = grad

    return d_all, grad_all, has_sensor


@torch.no_grad()
def pairwise_signed_fov_margins(
    fov_oracle: PinocchioFOVOracle,
    x: torch.Tensor,
    q_pair: torch.Tensor,
    pair_chunk: int = 512,
):
    """
    x:      [B_x, 3]
    q_pair: [B_x, R, 7]

    returns:
        raw_margin: [B_x, R, S]
        sign:       [B_x, R, S]

    Implementation detail:
        visibility_g_batch supports Cartesian p x q.  For pairwise pairs we call it
        on chunks of flattened pairs and take the diagonal of the returned [C,C,S].
    """
    Bx, R, J = q_pair.shape
    S = len(fov_oracle.sensor_frames)
    device = x.device

    raw_all = torch.empty((Bx * R, S), device=device, dtype=torch.float32)
    if R == 0:
        raw = raw_all.reshape(Bx, R, S)
        sign = torch.empty_like(raw)
        return raw, sign

    p_flat = x[:, None, :].expand(Bx, R, 3).reshape(Bx * R, 3).contiguous()
    q_flat = q_pair.reshape(Bx * R, J).contiguous()

    all_chain_specs = fov_oracle._get_chain_specs(device)

    for start in range(0, Bx * R, pair_chunk):
        end = min(start + pair_chunk, Bx * R)
        C = end - start
        p_c = p_flat[start:end]
        q_c = q_flat[start:end]

        _, sensor_margins, _, _ = fov_oracle.visibility_g_batch(
            p_c,
            q_c,
            all_chain_specs,
            fov_oracle.horizontal_fov_deg,
            fov_oracle.vertical_fov_deg,
            fov_oracle.z_min,
            fov_oracle.z_max,
            fov_oracle.delta,
        )

        if sensor_margins.ndim == 3:
            if (
                sensor_margins.shape[0] == C
                and sensor_margins.shape[1] == C
                and sensor_margins.shape[2] == S
            ):
                idx = torch.arange(C, device=device)
                raw_c = sensor_margins[idx, idx, :]
            else:
                raise RuntimeError(
                    f"Unsupported pairwise sensor_margins shape {tuple(sensor_margins.shape)} "
                    f"for C={C}, S={S}"
                )
        elif sensor_margins.ndim == 2 and C == 1:
            raw_c = sensor_margins.reshape(1, S)
        else:
            raise RuntimeError(
                f"Unsupported pairwise sensor_margins shape {tuple(sensor_margins.shape)} for C={C}"
            )

        raw_all[start:end] = raw_c

    raw = raw_all.reshape(Bx, R, S).contiguous()
    g = raw - fov_oracle.delta
    sign = torch.where(g >= 0.0, torch.ones_like(g), -torch.ones_like(g))
    return raw, sign


def union_signed_targets(
    d_s: torch.Tensor,
    grad_d_s: torch.Tensor,
    sign_s: torch.Tensor,
    has_sensor: torch.Tensor,
):
    """
    d_s:        [B_x, B_q_or_R, S]
    grad_d_s:   [B_x, B_q_or_R, S, 7]
    sign_s:     [B_x, B_q_or_R, S]
    has_sensor: [B_x, S]

    returns:
        target:      [B_x, B_q_or_R]
        target_grad: [B_x, B_q_or_R, 7]
    """
    sensor_available = has_sensor[:, None, :]
    f_s = sign_s * d_s
    f_s = torch.where(sensor_available, f_s, torch.full_like(f_s, -float("inf")))

    target, s_star = torch.max(f_s, dim=-1)

    Bx, Bn, S, J = grad_d_s.shape
    gather_idx = s_star[:, :, None, None].expand(Bx, Bn, 1, J)
    grad_win = torch.gather(grad_d_s, dim=2, index=gather_idx).squeeze(2)
    sign_win = torch.gather(sign_s, dim=2, index=s_star[:, :, None]).squeeze(2)
    target_grad = sign_win[:, :, None] * grad_win

    invalid = ~torch.isfinite(target)
    if torch.any(invalid):
        target = torch.where(invalid, torch.zeros_like(target), target)
        target_grad = torch.where(invalid[:, :, None], torch.zeros_like(target_grad), target_grad)

    return target, target_grad


def compute_losses(
    model: nn.Module,
    inp: torch.Tensor,
    target: torch.Tensor,
    target_grad: torch.Tensor,
    weights: Dict[str, float],
):
    """
    Yiming CDF-compatible loss, with only the target field changed.

    Matches nn_cdf.py:
      d_loss        = MSE(pred, target)
      grad_loss     = 1 - cosine_similarity(d_grad_pred, gt_grad)
      eikonal_loss  = abs(||d_grad_pred||_2 - 1).mean()
      tension_loss  = square(second derivative wrt q).sum(-1).mean()
      total         = 5*d + 0.01*eikonal + 0.01*tension + 0.1*grad

    We compute gradients directly w.r.t. q_inputs, matching Yiming's implementation,
    instead of differentiating w.r.t. the concatenated [x,q] tensor and slicing.
    """
    x_inputs = inp[:, :3].detach()
    q_inputs = inp[:, 3:10].detach().clone().requires_grad_(True)
    model_inputs = torch.cat([x_inputs, q_inputs], dim=-1)

    pred = model(model_inputs).reshape(-1)
    outputs = target.reshape(-1)
    gt_grad = target_grad.reshape(-1, 7)

    d_loss = ((pred - outputs) ** 2).mean()

    d_grad_pred = torch.autograd.grad(
        pred,
        q_inputs,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1.0).mean()

    dd_grad_pred = torch.autograd.grad(
        d_grad_pred,
        q_inputs,
        grad_outputs=torch.ones_like(d_grad_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    cosloss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    grad_loss = (1.0 - cosloss(d_grad_pred, gt_grad)).mean()
    tension_loss = dd_grad_pred.square().sum(dim=-1).mean()

    loss = (
        weights["sdf"] * d_loss
        + weights["eikonal"] * eikonal_loss
        + weights["tension"] * tension_loss
        + weights["grad"] * grad_loss
    )

    stats = {
        "loss": float(loss.detach().cpu()),
        "sdf_loss": float(d_loss.detach().cpu()),
        "eikonal_loss": float(eikonal_loss.detach().cpu()),
        "tension_loss": float(tension_loss.detach().cpu()),
        "grad_loss": float(grad_loss.detach().cpu()),
        "pred_mean": float(pred.detach().mean().cpu()),
        "pred_abs_mean": float(pred.detach().abs().mean().cpu()),
        "grad_norm_mean": float(d_grad_pred.detach().norm(2, dim=-1).mean().cpu()),
        "encoded_dim": float(getattr(model, "encoded_dim", 10)),
    }

    return loss, stats


def run_batch(
    model: nn.Module,
    dataset: VisibilityQ0Dataset,
    fov_oracle: PinocchioFOVOracle,
    device: torch.device,
    batch_x: int,
    batch_q: int,
    split: str,
    weights: Dict[str, float],
    decode_x_chunk: int,
    near_zero_ratio: float,
    near_zero_std: float,
    profile: bool = False,
):
    """
    Pairwise replace near-zero training.

    For train split and near_zero_ratio > 0:
        batch_q=100, ratio=0.2 means
            80 shared random q's -> Cartesian [B_x,80]
            20 per-x near q's   -> pairwise  [B_x,20]
        Total training pairs remain B_x * batch_q.

    For validation, keep the original pure Cartesian random-q evaluation.
    """
    timings = {}

    t0 = time.perf_counter()
    x, qlib, valid, _ = dataset.sample_x_batch(batch_x, split=split, device=device)
    sensor_masks = dataset.sensor_masks(device=device)
    sync_if_cuda(device)
    timings["sample_x"] = time.perf_counter() - t0

    Bx = x.shape[0]
    use_pairwise_replace = (split == "train" and near_zero_ratio > 0.0)
    near_per_x = 0
    rand_q_count = batch_q
    if use_pairwise_replace:
        near_per_x = int(round(batch_q * near_zero_ratio))
        near_per_x = max(0, min(batch_q, near_per_x))
        rand_q_count = batch_q - near_per_x
        if near_per_x == 0:
            use_pairwise_replace = False
            rand_q_count = batch_q

    t1 = time.perf_counter()
    q_rand = sample_random_q(dataset, rand_q_count, device=device)
    if use_pairwise_replace:
        q_near = sample_pairwise_near_q_per_x(
            dataset=dataset,
            qlib=qlib,
            valid=valid,
            sensor_masks=sensor_masks,
            near_per_x=near_per_x,
            device=device,
            near_zero_std=near_zero_std,
        )
    else:
        q_near = None
    sync_if_cuda(device)
    timings["sample_q"] = time.perf_counter() - t1

    t2 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1) Cartesian random-q branch: [B_x, rand_q_count]
    # ------------------------------------------------------------------
    td0 = time.perf_counter()
    d_cart, grad_cart, has_sensor = decode_per_sensor_distance_and_grad(
        qlib=qlib,
        valid=valid,
        q_query=q_rand,
        sensor_masks=sensor_masks,
        x_chunk=decode_x_chunk,
    )
    sync_if_cuda(device)
    timings["decode_cart"] = time.perf_counter() - td0

    tf0 = time.perf_counter()
    _, sign_cart = fov_oracle.signed_fov_margins(x, q_rand)
    sync_if_cuda(device)
    timings["fov_sign_cart"] = time.perf_counter() - tf0

    tu0 = time.perf_counter()
    target_cart, grad_target_cart = union_signed_targets(
        d_s=d_cart,
        grad_d_s=grad_cart,
        sign_s=sign_cart,
        has_sensor=has_sensor,
    )
    sync_if_cuda(device)
    timings["union_cart"] = time.perf_counter() - tu0

    ti0 = time.perf_counter()
    inp_cart = make_input_pairs(x, q_rand)
    target_cart_flat = target_cart.reshape(-1)
    grad_cart_flat = grad_target_cart.reshape(-1, 7)
    sync_if_cuda(device)
    timings["make_input_cart"] = time.perf_counter() - ti0

    # ------------------------------------------------------------------
    # 2) Pairwise near-zero branch: [B_x, near_per_x]
    # ------------------------------------------------------------------
    if use_pairwise_replace:
        tdn0 = time.perf_counter()
        d_near, grad_near, has_sensor_near = decode_pairwise_per_sensor_distance_and_grad(
            qlib=qlib,
            valid=valid,
            q_pair=q_near,
            sensor_masks=sensor_masks,
            x_chunk=max(1, min(decode_x_chunk, 128)),
        )
        sync_if_cuda(device)
        timings["decode_near"] = time.perf_counter() - tdn0

        tfn0 = time.perf_counter()
        _, sign_near = pairwise_signed_fov_margins(
            fov_oracle=fov_oracle,
            x=x,
            q_pair=q_near,
            pair_chunk=512,
        )
        sync_if_cuda(device)
        timings["fov_sign_near"] = time.perf_counter() - tfn0

        tun0 = time.perf_counter()
        target_near, grad_target_near = union_signed_targets(
            d_s=d_near,
            grad_d_s=grad_near,
            sign_s=sign_near,
            has_sensor=has_sensor_near,
        )
        sync_if_cuda(device)
        timings["union_near"] = time.perf_counter() - tun0

        tin0 = time.perf_counter()
        inp_near = make_pairwise_input_pairs(x, q_near)
        target_near_flat = target_near.reshape(-1)
        grad_near_flat = grad_target_near.reshape(-1, 7)
        sync_if_cuda(device)
        timings["make_input_near"] = time.perf_counter() - tin0

        inp = torch.cat([inp_cart, inp_near], dim=0)
        target = torch.cat([target_cart_flat, target_near_flat], dim=0)
        target_grad = torch.cat([grad_cart_flat, grad_near_flat], dim=0)

        sign_pos_num = (sign_cart > 0).float().sum() + (sign_near > 0).float().sum()
        sign_pos_den = sign_cart.numel() + sign_near.numel()

        near_target_abs = float(target_near.abs().mean().detach().cpu())
        near_positive_ratio = float((target_near > 0).float().mean().detach().cpu())
        cart_target_abs = float(target_cart.abs().mean().detach().cpu())
        cart_positive_ratio = float((target_cart > 0).float().mean().detach().cpu())
    else:
        inp = inp_cart
        target = target_cart_flat
        target_grad = grad_cart_flat
        sign_pos_num = (sign_cart > 0).float().sum()
        sign_pos_den = sign_cart.numel()
        near_target_abs = 0.0
        near_positive_ratio = 0.0
        cart_target_abs = float(target_cart.abs().mean().detach().cpu())
        cart_positive_ratio = float((target_cart > 0).float().mean().detach().cpu())

    target_stats = {
        "target_mean": float(target.mean().detach().cpu()),
        "target_abs_mean": float(target.abs().mean().detach().cpu()),
        "positive_ratio": float((target > 0).float().mean().detach().cpu()),
        "sign_positive_ratio": float((sign_pos_num / max(1, sign_pos_den)).detach().cpu()),
        "cart_target_abs_mean": cart_target_abs,
        "cart_positive_ratio": cart_positive_ratio,
        "near_target_abs_mean": near_target_abs,
        "near_positive_ratio": near_positive_ratio,
        "near_per_x": float(near_per_x),
        "rand_q_count": float(rand_q_count),
    }

    sync_if_cuda(device)
    timings["build_target"] = time.perf_counter() - t2

    tl0 = time.perf_counter()
    loss, stats = compute_losses(
        model=model,
        inp=inp,
        target=target,
        target_grad=target_grad,
        weights=weights,
    )
    sync_if_cuda(device)
    timings["loss_forward"] = time.perf_counter() - tl0

    stats.update(target_stats)

    if profile:
        # Keep old profile keys approximately meaningful.
        stats["time_decode"] = float(timings.get("decode_cart", 0.0) + timings.get("decode_near", 0.0))
        stats["time_fov_sign"] = float(timings.get("fov_sign_cart", 0.0) + timings.get("fov_sign_near", 0.0))
        stats["time_union"] = float(timings.get("union_cart", 0.0) + timings.get("union_near", 0.0))
        stats["time_make_input"] = float(timings.get("make_input_cart", 0.0) + timings.get("make_input_near", 0.0))
        for k, v in timings.items():
            stats[f"time_{k}"] = float(v)

    return loss, stats

def save_checkpoint(path, model, optimizer, scheduler, args, step, best_val, stats):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "step": step,
            "best_val": best_val,
            "stats": stats,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    return int(ckpt.get("step", 0)), float(ckpt.get("best_val", float("inf")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="FOV-only signed visibility CDF training with Yiming-style q0 library."
    )

    parser.add_argument(
        "--data",
        default="src/care_visibility_cdf/data/visibility_yiming_style_grid30_q20000_k500_fovonly.npz",
    )
    parser.add_argument(
        "--urdf",
        default="src/arm_description/urdf/Arm.urdf",
    )
    parser.add_argument(
        "--out-dir",
        default="src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed",
    )

    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-x", type=int, default=4000)
    parser.add_argument("--batch-q", type=int, default=100)
    parser.add_argument("--val-batch-x", type=int, default=512)
    parser.add_argument("--val-batch-q", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=1000)

    parser.add_argument("--decode-x-chunk", type=int, default=64)

    # Yiming CDF-compatible network defaults. hidden_dim/num_hidden_layers are kept
    # only for backward compatibility with older checkpoints/scripts.
    parser.add_argument("--model-arch", choices=["yiming"], default="yiming")
    parser.add_argument("--mlp-layers", default="1024,512,256,128,128")
    parser.add_argument("--skips", default="")
    parser.add_argument("--nerf", action=argparse.BooleanOptionalAction, default=True,
                        help="Yiming-style encoding: [x, sin(x), cos(x)], not multi-frequency NeRF.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--activation", choices=["relu"], default="relu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Use torch.cuda.amp autocast + GradScaler, matching Yiming training.")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5000)
    parser.add_argument("--scheduler-threshold", type=float, default=0.01)
    parser.add_argument("--scheduler-eps", type=float, default=1e-4)

    parser.add_argument("--weight-sdf", type=float, default=5.0)
    parser.add_argument("--weight-eikonal", type=float, default=0.01)
    parser.add_argument("--weight-tension", type=float, default=0.01)
    parser.add_argument("--weight-grad", type=float, default=0.1)

    parser.add_argument("--horizontal-fov-deg", type=float, default=50.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=66.0)
    parser.add_argument("--z-min", type=float, default=0.20)
    parser.add_argument("--z-max", type=float, default=0.70)
    parser.add_argument("--delta", type=float, default=0.01)

    parser.add_argument("--near-zero-ratio", type=float, default=0.0)
    parser.add_argument("--near-zero-std", type=float, default=0.03)

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument("--profile", action="store_true", help="Print timing breakdown for profiling.")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="care_visibility_cdf")
    parser.add_argument("--wandb-name", default="exp1_yiming_k500_fov_signed")
    parser.add_argument("--wandb-group", default="exp1_yiming_k500_fov_signed")
    parser.add_argument("--wandb-tags", default="exp1,k500,fov,signed,yiming")
    parser.add_argument("--wandb-log-every", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "--wandb was requested, but the wandb package is not installed."
            ) from exc

        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            group=args.wandb_group or None,
            tags=tags,
            config=vars(args),
            dir=args.out_dir,
        )

    dataset = VisibilityQ0Dataset(
        path=args.data,
        val_count=args.val_count,
        seed=args.seed,
    )

    fov_oracle = PinocchioFOVOracle(
        urdf_path=args.urdf,
        joint_names=DEFAULT_JOINT_NAMES,
        sensor_frames=DEFAULT_SENSOR_FRAMES,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        z_min=args.z_min,
        z_max=args.z_max,
        delta=args.delta,
    )

    mlp_layers = _parse_mlp_layers(args.mlp_layers)
    skips = tuple(int(v.strip()) for v in args.skips.split(",") if v.strip())

    model = MLP(
        in_dim=10,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        out_dim=1,
        activation=args.activation,
        model_arch=args.model_arch,
        mlp_layers=mlp_layers,
        skips=skips,
        nerf=args.nerf,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=args.scheduler_eps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    weights = {
        "sdf": args.weight_sdf,
        "eikonal": args.weight_eikonal,
        "tension": args.weight_tension,
        "grad": args.weight_grad,
    }

    start_step = 0
    best_val = float("inf")

    if args.resume:
        start_step, best_val = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"[resume] {args.resume}")
        print(f"[resume] start_step={start_step}, best_val={best_val}")

    print("")
    print("=== Training Config ===")
    print(f"device:              {device}")
    print(f"data:                {args.data}")
    print(f"urdf:                {args.urdf}")
    print(f"out_dir:             {args.out_dir}")
    print(f"steps:               {args.steps}")
    print(f"batch_x:             {args.batch_x}")
    print(f"batch_q:             {args.batch_q}")
    print(f"val_batch_x:         {args.val_batch_x}")
    print(f"val_batch_q:         {args.val_batch_q}")
    print(f"decode_x_chunk:      {args.decode_x_chunk}")
    print(f"near_zero_ratio:     {args.near_zero_ratio}")
    print(f"near_zero_std:       {args.near_zero_std}")
    print(f"lr:                  {args.lr}")
    print(f"weights:             {weights}")
    print(f"model_arch:          {args.model_arch}")
    print(f"mlp_layers:          {args.mlp_layers}")
    print(f"skips:               {args.skips}")
    print(f"nerf:                {args.nerf}  # Yiming: [x, sin(x), cos(x)]")
    print(f"encoded_dim:         {getattr(model, 'encoded_dim', 'unknown')}")
    print(f"activation:          {args.activation}")
    print(f"amp:                 {args.amp}")
    print(f"hidden_dim_legacy:   {args.hidden_dim}")
    print(f"num_hidden_layers_legacy: {args.num_hidden_layers}")
    print(f"profile:             {args.profile}")
    print(f"wandb:               {args.wandb}")
    if args.wandb:
        print(f"wandb_project:       {args.wandb_project}")
        print(f"wandb_name:          {args.wandb_name}")
        print(f"wandb_group:         {args.wandb_group}")
        print(f"wandb_tags:          {args.wandb_tags}")
        print(f"wandb_log_every:     {args.wandb_log_every}")
    print("=======================")
    print("")

    t0 = time.time()
    last_train_stats = {}

    for step in range(start_step + 1, args.steps + 1):
        model.train()

        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            loss, train_stats = run_batch(
                model=model,
                dataset=dataset,
                fov_oracle=fov_oracle,
                device=device,
                batch_x=args.batch_x,
                batch_q=args.batch_q,
                split="train",
                weights=weights,
                decode_x_chunk=args.decode_x_chunk,
                near_zero_ratio=args.near_zero_ratio,
                near_zero_std=args.near_zero_std,
                profile=args.profile,
            )

        tb0 = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step(loss.detach())
        sync_if_cuda(device)
        if args.profile:
            train_stats["time_backward_step"] = float(time.perf_counter() - tb0)

        last_train_stats = train_stats

        if wandb_run is not None and (step % args.wandb_log_every == 0 or step == 1):
            wandb_payload = {f"train/{k}": v for k, v in train_stats.items()}
            wandb_payload["train/lr"] = optimizer.param_groups[0]["lr"]
            wandb_payload["step"] = step
            wandb_run.log(wandb_payload, step=step)

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[train] step={step:06d} "
                f"loss={train_stats['loss']:.6f} "
                f"sdf={train_stats['sdf_loss']:.6f} "
                f"eik={train_stats['eikonal_loss']:.6f} "
                f"tension={train_stats['tension_loss']:.6f} "
                f"grad={train_stats['grad_loss']:.6f} "
                f"pred_mean={train_stats['pred_mean']:.6f} "
                f"target_mean={train_stats['target_mean']:.6f} "
                f"target_abs={train_stats['target_abs_mean']:.6f} "
                f"target_pos={train_stats['positive_ratio']:.4f} "
                f"sign_pos={train_stats['sign_positive_ratio']:.4f} "
                f"grad_norm={train_stats['grad_norm_mean']:.6f} "
                f"lr={lr_now:.3e} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            if args.profile:
                print(
                    f"[profile] step={step:06d} "
                    f"sample_x={train_stats.get('time_sample_x', -1):.3f}s "
                    f"sample_q={train_stats.get('time_sample_q', -1):.3f}s "
                    f"build_target={train_stats.get('time_build_target', -1):.3f}s "
                    f"decode={train_stats.get('time_decode', -1):.3f}s "
                    f"fov_sign={train_stats.get('time_fov_sign', -1):.3f}s "
                    f"union={train_stats.get('time_union', -1):.3f}s "
                    f"make_input={train_stats.get('time_make_input', -1):.3f}s "
                    f"loss_forward={train_stats.get('time_loss_forward', -1):.3f}s "
                    f"backward_step={train_stats.get('time_backward_step', -1):.3f}s",
                    flush=True,
                )

        if step % args.val_every == 0 or step == 1:
            model.eval()

            val_loss, val_stats = run_batch(
                model=model,
                dataset=dataset,
                fov_oracle=fov_oracle,
                device=device,
                batch_x=args.val_batch_x,
                batch_q=args.val_batch_q,
                split="val",
                weights=weights,
                decode_x_chunk=args.decode_x_chunk,
                near_zero_ratio=0.0,
                near_zero_std=args.near_zero_std,
                profile=args.profile,
            )

            val_scalar = val_stats["sdf_loss"]

            if wandb_run is not None:
                wandb_payload = {f"val/{k}": v for k, v in val_stats.items()}
                wandb_payload["val/best_sdf_loss"] = min(best_val, val_scalar)
                wandb_payload["step"] = step
                wandb_run.log(wandb_payload, step=step)

            print(
                f"[val]   step={step:06d} "
                f"loss={val_stats['loss']:.6f} "
                f"sdf={val_stats['sdf_loss']:.6f} "
                f"eik={val_stats['eikonal_loss']:.6f} "
                f"tension={val_stats['tension_loss']:.6f} "
                f"grad={val_stats['grad_loss']:.6f} "
                f"pred_mean={val_stats['pred_mean']:.6f} "
                f"target_mean={val_stats['target_mean']:.6f} "
                f"target_abs={val_stats['target_abs_mean']:.6f} "
                f"target_pos={val_stats['positive_ratio']:.4f} "
                f"sign_pos={val_stats['sign_positive_ratio']:.4f} "
                f"grad_norm={val_stats['grad_norm_mean']:.6f}",
                flush=True,
            )

            if args.profile:
                print(
                    f"[profile-val] step={step:06d} "
                    f"sample_x={val_stats.get('time_sample_x', -1):.3f}s "
                    f"sample_q={val_stats.get('time_sample_q', -1):.3f}s "
                    f"build_target={val_stats.get('time_build_target', -1):.3f}s "
                    f"decode={val_stats.get('time_decode', -1):.3f}s "
                    f"fov_sign={val_stats.get('time_fov_sign', -1):.3f}s "
                    f"union={val_stats.get('time_union', -1):.3f}s "
                    f"make_input={val_stats.get('time_make_input', -1):.3f}s "
                    f"loss_forward={val_stats.get('time_loss_forward', -1):.3f}s",
                    flush=True,
                )

            if val_scalar < best_val:
                best_val = val_scalar
                best_path = os.path.join(args.out_dir, "best.pt")
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    step=step,
                    best_val=best_val,
                    stats={"train": train_stats, "val": val_stats},
                )
                print(f"[save] best checkpoint: {best_path}, best_val={best_val:.6f}", flush=True)

        if step % args.save_every == 0:
            path = os.path.join(args.out_dir, f"step_{step:06d}.pt")
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                step=step,
                best_val=best_val,
                stats={"train": train_stats},
            )
            print(f"[save] checkpoint: {path}", flush=True)

    final_path = os.path.join(args.out_dir, "final.pt")
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        step=args.steps,
        best_val=best_val,
        stats={"train": last_train_stats},
    )
    print(f"[save] final checkpoint: {final_path}")
    print(f"[done] best_val={best_val:.6f}")

    if wandb_run is not None:
        wandb_run.summary["best_val_sdf"] = best_val
        wandb_run.summary["final_step"] = args.steps
        wandb_run.finish()


if __name__ == "__main__":
    main()
