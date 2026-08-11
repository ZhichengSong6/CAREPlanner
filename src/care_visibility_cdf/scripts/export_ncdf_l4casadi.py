#!/usr/bin/env python3
"""Build the trained CARE visibility NCDF as an L4CasADi/CasADi function.

This is deliberately a conversion + verification tool, not a ROS runtime node.
It reconstructs the exact YimingMLP used by the frozen Exp1 checkpoint, wraps
it with L4CasADi, creates CasADi value/Jacobian functions, and checks them
against native PyTorch autograd on identical (x, q) samples.

The CasADi convention used here is

    x : 1 x 3
    q : 1 x 7
    z = [x, q] : 1 x 10
    f_theta(x, q) : 1 x 1
    d f_theta / d q : 1 x 7

L4CasADi v2 preserves the exact two-dimensional CasADi input shape, so the
1 x 10 shape is intentional and matches YimingMLP.forward().
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = (
    PACKAGE_DIR
    / "checkpoints"
    / "exp1_yiming_k500_fov_signed"
    / "final.pt"
)
DEFAULT_BUILD_DIR = PACKAGE_DIR / "generated" / "l4casadi_ncdf"

# Reuse the exact checkpoint/model reconstruction already used by the frozen
# evaluator instead of maintaining a second architecture definition.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_direct_vs_projection_ascent import (  # noqa: E402
    build_model_from_checkpoint,
    torch_load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap the frozen CARE visibility NCDF with L4CasADi and verify value/gradient parity."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the signed Yiming NCDF checkpoint (default: frozen Exp1 final.pt).",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used by the L4CasADi-backed PyTorch model.",
    )
    parser.add_argument(
        "--build-dir",
        default=str(DEFAULT_BUILD_DIR),
        help="Directory for L4CasADi generated sources/libraries.",
    )
    parser.add_argument(
        "--model-name",
        default="care_visibility_ncdf",
        help="Unique L4CasADi external-function name.",
    )
    parser.add_argument("--num-checks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-atol", type=float, default=1e-5)
    parser.add_argument("--grad-atol", type=float, default=2e-4)
    parser.add_argument("--grad-rtol", type=float, default=2e-3)
    return parser.parse_args()


def require_l4casadi():
    try:
        import casadi as cs
    except ImportError as exc:
        raise RuntimeError(
            "CasADi is not importable in this Python environment. Install/configure CasADi before running this script."
        ) from exc

    try:
        import l4casadi as l4c
    except ImportError as exc:
        raise RuntimeError(
            "l4casadi is not importable in this Python environment. Install L4CasADi before running this script."
        ) from exc

    return cs, l4c


def build_casadi_functions(model: torch.nn.Module, device: str, build_dir: Path, model_name: str):
    cs, l4c = require_l4casadi()

    build_dir.mkdir(parents=True, exist_ok=True)

    # We need first derivatives w.r.t. q. A full Hessian is unnecessary for the
    # first CAREPlanner integration, so keep generate_jac_jac disabled.
    l4c_model = l4c.L4CasADi(
        model,
        device=device,
        name=model_name,
        build_dir=str(build_dir),
        generate_jac=True,
        generate_adj1=True,
        generate_jac_adj1=True,
        generate_jac_jac=False,
        scripting=True,
        mutable=False,
    )

    # L4CasADi v2 preserves input dimensions. YimingMLP expects a 2-D tensor
    # whose last dimension is 10, hence explicit row-vector symbols.
    x_sym = cs.MX.sym("x", 1, 3)
    q_sym = cs.MX.sym("q", 1, 7)
    z_sym = cs.horzcat(x_sym, q_sym)

    f_sym = l4c_model(z_sym)
    grad_q_sym = cs.jacobian(f_sym, q_sym)

    f_fn = cs.Function(
        "care_ncdf",
        [x_sym, q_sym],
        [f_sym],
        ["x", "q"],
        ["f"],
    )
    grad_q_fn = cs.Function(
        "care_ncdf_grad_q",
        [x_sym, q_sym],
        [grad_q_sym],
        ["x", "q"],
        ["grad_q"],
    )
    value_grad_q_fn = cs.Function(
        "care_ncdf_value_grad_q",
        [x_sym, q_sym],
        [f_sym, grad_q_sym],
        ["x", "q"],
        ["f", "grad_q"],
    )

    return l4c_model, f_fn, grad_q_fn, value_grad_q_fn


def pytorch_value_grad_q(model: torch.nn.Module, x_np: np.ndarray, q_np: np.ndarray, device: torch.device):
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device).reshape(1, 3)
    q = torch.as_tensor(q_np, dtype=torch.float32, device=device).reshape(1, 7)
    q = q.clone().detach().requires_grad_(True)

    z = torch.cat([x, q], dim=1)
    f = model(z).reshape(())
    grad_q = torch.autograd.grad(f, q, create_graph=False, retain_graph=False)[0]

    return float(f.detach().cpu()), grad_q.detach().cpu().numpy().reshape(7)


def run_parity_checks(
    model: torch.nn.Module,
    value_grad_q_fn,
    device: torch.device,
    num_checks: int,
    seed: int,
    value_atol: float,
    grad_atol: float,
    grad_rtol: float,
):
    rng = np.random.default_rng(seed)

    max_value_abs = 0.0
    max_grad_abs = 0.0
    max_grad_rel = 0.0
    failures = []

    # Parity does not depend on whether a sample is physically useful. We use
    # the same broad region as the existing evaluator: x/y around the base,
    # z in [0,1], q in [-pi,pi].
    for index in range(num_checks):
        x_np = np.asarray(
            [
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.0, 1.0),
            ],
            dtype=np.float32,
        )
        q_np = rng.uniform(-np.pi, np.pi, size=(7,)).astype(np.float32)

        torch_f, torch_grad = pytorch_value_grad_q(model, x_np, q_np, device)

        casadi_out = value_grad_q_fn(x_np.reshape(1, 3), q_np.reshape(1, 7))
        casadi_f = float(np.asarray(casadi_out[0]).reshape(()))
        casadi_grad = np.asarray(casadi_out[1], dtype=np.float64).reshape(7)

        value_abs = abs(casadi_f - torch_f)
        grad_abs_vec = np.abs(casadi_grad - torch_grad.astype(np.float64))
        grad_abs = float(np.max(grad_abs_vec))
        denom = np.maximum(np.abs(torch_grad.astype(np.float64)), 1e-8)
        grad_rel = float(np.max(grad_abs_vec / denom))

        max_value_abs = max(max_value_abs, value_abs)
        max_grad_abs = max(max_grad_abs, grad_abs)
        max_grad_rel = max(max_grad_rel, grad_rel)

        value_ok = value_abs <= value_atol
        grad_ok = np.allclose(
            casadi_grad,
            torch_grad.astype(np.float64),
            atol=grad_atol,
            rtol=grad_rtol,
        )
        if not (value_ok and grad_ok):
            failures.append(
                {
                    "index": index,
                    "value_abs": value_abs,
                    "grad_abs": grad_abs,
                    "grad_rel": grad_rel,
                    "torch_f": torch_f,
                    "casadi_f": casadi_f,
                }
            )

    print("[parity] checks             :", num_checks)
    print(f"[parity] max |f_l4c-f_torch| : {max_value_abs:.6e}")
    print(f"[parity] max |dg_l4c-dg_torch|: {max_grad_abs:.6e}")
    print(f"[parity] max grad rel error   : {max_grad_rel:.6e}")

    if failures:
        print(f"[parity] FAILED: {len(failures)}/{num_checks} samples exceeded tolerances")
        for failure in failures[:5]:
            print("  ", failure)
        raise RuntimeError("L4CasADi/PyTorch NCDF parity check failed.")

    print("[parity] PASS: L4CasADi value and q-gradient agree with PyTorch.")


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NCDF checkpoint not found: {checkpoint_path}")
    if args.num_checks <= 0:
        raise ValueError("--num-checks must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False.")

    torch_device = torch.device(args.device)

    print(f"[checkpoint] {checkpoint_path}")
    print(f"[device]     {torch_device}")
    print(f"[build_dir]  {build_dir}")
    print(f"[model_name] {args.model_name}")

    ckpt = torch_load_checkpoint(str(checkpoint_path), torch_device)
    model, ckpt_args = build_model_from_checkpoint(ckpt, torch_device)

    print(
        "[model] checkpoint args: model_arch={}, nerf={}, activation={}".format(
            ckpt_args.get("model_arch", "yiming"),
            ckpt_args.get("nerf", True),
            ckpt_args.get("activation", "relu"),
        )
    )

    l4c_model, f_fn, grad_q_fn, value_grad_q_fn = build_casadi_functions(
        model=model,
        device=args.device,
        build_dir=build_dir,
        model_name=args.model_name,
    )

    # The first call triggers/builds L4CasADi artifacts. Subsequent calls are
    # the relevant steady-state path for optimization.
    run_parity_checks(
        model=model,
        value_grad_q_fn=value_grad_q_fn,
        device=torch_device,
        num_checks=args.num_checks,
        seed=args.seed,
        value_atol=args.value_atol,
        grad_atol=args.grad_atol,
        grad_rtol=args.grad_rtol,
    )

    print("[casadi] value function    :", f_fn)
    print("[casadi] q-gradient function:", grad_q_fn)
    print("[casadi] combined function :", value_grad_q_fn)
    print("[l4casadi] shared_lib_dir  :", l4c_model.shared_lib_dir)
    print("[done] NCDF is ready for the next CasADi optimization-integration step.")


if __name__ == "__main__":
    main()
