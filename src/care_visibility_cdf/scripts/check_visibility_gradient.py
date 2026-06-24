#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import torch
from urdf_parser_py.urdf import URDF

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from validate_visibility_oracle import (  # noqa: E402
    DEFAULT_JOINT_NAMES,
    DEFAULT_SENSOR_FRAMES,
    get_joint_limits,
)
from extract_visibility_zero_level_sets import (  # noqa: E402
    prepare_chain_specs,
    torch_tensor,
    visibility_g_batch,
)


def names_from_npz(data, key, default):
    if key in data:
        return [str(name) for name in data[key]]
    return list(default)


def scalar_from_npz(data, key, override):
    if override is not None:
        return float(override)
    if key not in data:
        raise RuntimeError(f"Missing required metadata key '{key}'. Pass it explicitly.")
    return float(data[key])


def sample_pairs_from_raw(raw, num_samples, seed):
    rng = np.random.default_rng(seed)
    n = len(raw["p"])
    count = min(num_samples, n)
    idx = rng.choice(n, size=count, replace=False)
    return raw["p"][idx].astype(np.float32), raw["q"][idx].astype(np.float32)


def sample_pairs_from_q0(q0_data, num_samples, seed):
    rng = np.random.default_rng(seed)
    grid_points = q0_data["grid_points"].astype(np.float32)
    q0 = q0_data["q0_templates"].astype(np.float32)
    q0_g = q0_data["q0_g"]
    active_sensor = q0_data["q0_active_sensor"]

    valid = np.isfinite(q0_g) & (active_sensor >= 0)
    ii, jj = np.where(valid)
    if len(ii) == 0:
        raise RuntimeError("No valid q0 entries found.")

    count = min(num_samples, len(ii))
    pick = rng.choice(len(ii), size=count, replace=False)
    pi = ii[pick]
    qi = jj[pick]
    return grid_points[pi], q0[pi, qi]


def compute_g_sensor_plane(point_np, q_np, all_chain_specs, args, device):
    with torch.no_grad():
        g, _, active_sensor, active_planes = visibility_g_batch(
            torch_tensor(point_np, device),
            torch_tensor(q_np.reshape(1, -1), device),
            all_chain_specs,
            args.horizontal_fov_deg,
            args.vertical_fov_deg,
            args.z_min,
            args.z_max,
            args.delta,
        )
    sensor = int(active_sensor.detach().cpu().numpy()[0])
    planes = active_planes.detach().cpu().numpy()[0]
    best_plane = int(planes[sensor])
    return float(g.detach().cpu().numpy()[0]), sensor, best_plane


def autograd_grad(point_np, q_np, all_chain_specs, args, device):
    q = torch_tensor(q_np.reshape(1, -1), device).clone().detach().requires_grad_(True)
    g, _, active_sensor, active_planes = visibility_g_batch(
        torch_tensor(point_np, device),
        q,
        all_chain_specs,
        args.horizontal_fov_deg,
        args.vertical_fov_deg,
        args.z_min,
        args.z_max,
        args.delta,
    )
    g[0].backward()
    grad = q.grad.detach().cpu().numpy()[0].astype(np.float64)
    sensor = int(active_sensor.detach().cpu().numpy()[0])
    planes = active_planes.detach().cpu().numpy()[0]
    best_plane = int(planes[sensor])
    return float(g.detach().cpu().numpy()[0]), grad, sensor, best_plane


def finite_difference_grad(point_np, q_np, q_min, q_max, all_chain_specs, args, device, fd_eps):
    grad_fd = np.full((len(q_np),), np.nan, dtype=np.float64)
    valid_dim = np.zeros((len(q_np),), dtype=np.bool_)

    g0, sensor0, plane0 = compute_g_sensor_plane(point_np, q_np, all_chain_specs, args, device)

    for joint_idx in range(len(q_np)):
        if q_np[joint_idx] - fd_eps < q_min[joint_idx]:
            continue
        if q_np[joint_idx] + fd_eps > q_max[joint_idx]:
            continue

        q_plus = q_np.copy()
        q_minus = q_np.copy()
        q_plus[joint_idx] += fd_eps
        q_minus[joint_idx] -= fd_eps

        g_plus, sensor_plus, plane_plus = compute_g_sensor_plane(
            point_np, q_plus, all_chain_specs, args, device
        )
        g_minus, sensor_minus, plane_minus = compute_g_sensor_plane(
            point_np, q_minus, all_chain_specs, args, device
        )

        # Hard max over sensors and hard min over planes are only piecewise smooth.
        # Skip dimensions where the active branch changes under finite perturbation.
        if sensor_plus != sensor0 or sensor_minus != sensor0:
            continue
        if plane_plus != plane0 or plane_minus != plane0:
            continue

        grad_fd[joint_idx] = (g_plus - g_minus) / (2.0 * fd_eps)
        valid_dim[joint_idx] = True

    return g0, grad_fd, valid_dim, sensor0, plane0


def print_error_stats(name, values):
    if len(values) == 0:
        print(f"{name}: no valid values")
        return
    pct = np.percentile(values, [0, 50, 90, 95, 99, 100])
    print(f"{name}: min/median/p90/p95/p99/max = "
          f"{pct[0]:.3e} / {pct[1]:.3e} / {pct[2]:.3e} / "
          f"{pct[3]:.3e} / {pct[4]:.3e} / {pct[5]:.3e}")


def main():
    parser = argparse.ArgumentParser(
        description="Check dg/dq from torch autograd against central finite differences."
    )
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--raw", default="", help="Optional M3 raw visibility .npz.")
    parser.add_argument("--q0", default="", help="Optional M4 q0 visibility .npz.")
    parser.add_argument("--source", choices=["raw", "q0"], default="q0")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument("--abs-tol", type=float, default=2e-3)
    parser.add_argument("--rel-tol", type=float, default=5e-2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--joint-names", nargs="*", default=None)
    parser.add_argument("--sensor-frames", nargs="*", default=None)
    parser.add_argument("--horizontal-fov-deg", type=float, default=None)
    parser.add_argument("--vertical-fov-deg", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    if args.source == "raw" and not args.raw:
        raise RuntimeError("--source raw requires --raw.")
    if args.source == "q0" and not args.q0:
        raise RuntimeError("--source q0 requires --q0.")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    data = np.load(args.q0 if args.source == "q0" else args.raw, allow_pickle=True)
    joint_names = args.joint_names or names_from_npz(data, "joint_names", DEFAULT_JOINT_NAMES)
    sensor_frames = args.sensor_frames or names_from_npz(data, "sensor_frames", DEFAULT_SENSOR_FRAMES)

    args.horizontal_fov_deg = scalar_from_npz(data, "horizontal_fov_deg", args.horizontal_fov_deg)
    args.vertical_fov_deg = scalar_from_npz(data, "vertical_fov_deg", args.vertical_fov_deg)
    args.z_min = scalar_from_npz(data, "z_min", args.z_min)
    args.z_max = scalar_from_npz(data, "z_max", args.z_max)
    args.delta = scalar_from_npz(data, "delta", args.delta)

    if args.source == "q0":
        p_batch, q_batch = sample_pairs_from_q0(data, args.num_samples, args.seed)
    else:
        p_batch, q_batch = sample_pairs_from_raw(data, args.num_samples, args.seed)

    robot = URDF.from_xml_file(args.urdf)
    joint_limits = get_joint_limits(robot, joint_names)
    q_min = np.asarray([joint_limits[name][0] for name in joint_names], dtype=np.float32)
    q_max = np.asarray([joint_limits[name][1] for name in joint_names], dtype=np.float32)
    all_chain_specs = prepare_chain_specs(robot, args.base_frame, sensor_frames, joint_names, device)

    abs_errors = []
    rel_errors = []
    compared_dims = 0
    skipped_dims = 0
    bad_dims = 0
    sensor_counts = np.zeros((len(sensor_frames),), dtype=np.int64)
    plane_counts = np.zeros((6,), dtype=np.int64)

    print("")
    print("=== Visibility Gradient Check Config ===")
    print(f"urdf:        {args.urdf}")
    print(f"source:      {args.source}")
    print(f"raw:         {args.raw if args.raw else '<none>'}")
    print(f"q0:          {args.q0 if args.q0 else '<none>'}")
    print(f"device:      {device}")
    print(f"samples:     {len(p_batch)}")
    print(f"fd_eps:      {args.fd_eps}")
    print(f"abs_tol:     {args.abs_tol}")
    print(f"rel_tol:     {args.rel_tol}")
    print(f"fov/range:   h={args.horizontal_fov_deg}, v={args.vertical_fov_deg}, z=[{args.z_min}, {args.z_max}]")
    print(f"delta:       {args.delta}")

    for point_np, q_np in zip(p_batch, q_batch):
        _, grad_auto, sensor, plane = autograd_grad(point_np, q_np, all_chain_specs, args, device)
        _, grad_fd, valid_dim, _, _ = finite_difference_grad(
            point_np, q_np, q_min, q_max, all_chain_specs, args, device, args.fd_eps
        )

        sensor_counts[sensor] += 1
        plane_counts[plane] += 1

        skipped_dims += int(np.count_nonzero(~valid_dim))
        for joint_idx in np.where(valid_dim)[0]:
            a = float(grad_auto[joint_idx])
            f = float(grad_fd[joint_idx])
            abs_err = abs(a - f)
            rel_err = abs_err / max(1.0, abs(a), abs(f))
            abs_errors.append(abs_err)
            rel_errors.append(rel_err)
            compared_dims += 1
            if abs_err > args.abs_tol and rel_err > args.rel_tol:
                bad_dims += 1

    abs_errors = np.asarray(abs_errors, dtype=np.float64)
    rel_errors = np.asarray(rel_errors, dtype=np.float64)

    print("")
    print("=== Branch Statistics ===")
    print("active sensor hist:", sensor_counts)
    print("active plane hist: ", plane_counts)
    print("plane ids: 0 left, 1 right, 2 bottom, 3 top, 4 near, 5 far")

    print("")
    print("=== Gradient Error Statistics ===")
    print(f"compared_dims: {compared_dims}")
    print(f"skipped_dims:  {skipped_dims}")
    print(f"bad_dims:      {bad_dims}")
    print_error_stats("abs_error", abs_errors)
    print_error_stats("rel_error", rel_errors)

    if compared_dims == 0:
        print("")
        print("[FAIL] No differentiable dimensions were checked. Try larger --fd-eps or more samples.")
        sys.exit(1)

    bad_ratio = bad_dims / max(compared_dims, 1)
    print(f"bad_ratio:     {bad_ratio:.6f}")

    if bad_ratio > 0.01:
        print("")
        print("[FAIL] Gradient check failed.")
        sys.exit(1)

    print("")
    print("[OK] Autograd dg/dq matches finite differences away from branch switches.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
