#!/usr/bin/env python3
"""Export CDF or RDF to the same CasADi-generated, Python-free C ABI.

This is an offline exporter. It does not select a planner's physical margins,
change ROS parameters, deploy a library, or certify collision avoidance.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import time

import casadi as ca
import numpy as np
import torch

from collision_cdf_model import CollisionCDF
from rdf_model import RobotRDF, load_bundle, sha256


def cdf_expression(net, z):
    model = net.model
    if len(model.layers) != 1:
        raise ValueError("Only the current no-skip collision CDF layout is supported")
    y = ca.vertcat(z.T, ca.sin(z.T), ca.cos(z.T)) if model.nerf else z.T
    for layer in model.layers[0]:
        linear = layer[0]
        y = ca.mtimes(ca.DM(linear.weight.detach().cpu().numpy()), y) + ca.DM(linear.bias.detach().cpu().numpy())
        for activation in list(layer)[1:]:
            if isinstance(activation, torch.nn.GELU) and activation.approximate == "none":
                y = 0.5*y*(1+ca.erf(y/math.sqrt(2.0)))
            elif isinstance(activation, torch.nn.ReLU):
                y = ca.if_else(y > 0, y, 0)
            else:
                raise ValueError("Unsupported CDF activation: " + str(activation))
    if y.numel() != 1:
        raise ValueError("Expected a scalar collision CDF")
    return y


def rdf_expression(bundle, z, excluded_links):
    contract, n = bundle["contract"], int(bundle["n_func"])
    transforms = [(ca.DM.eye(3), ca.DM.zeros(3, 1))]
    for joint in contract["joints"]:
        rotation, translation = transforms[joint["parent"]]
        origin, axis = ca.DM(joint["origin"]), ca.DM(joint["axis"])
        t0 = translation + ca.mtimes(rotation, origin[:3, 3])
        r0 = ca.mtimes(rotation, origin[:3, :3])
        if joint["q_index"] >= 0:
            q = z[3+joint["q_index"]]
            if joint["kind"] == "prismatic":
                t0 += ca.mtimes(r0, axis)*q
            else:
                a = joint["axis"]
                skew = ca.DM([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                r0 = ca.mtimes(r0, ca.DM.eye(3)+ca.sin(q)*skew+(1-ca.cos(q))*ca.mtimes(skew, skew))
        transforms.append((r0, t0))
    fields, best = {}, None
    # Build each polynomial once. Repeated sensor geometry shares coefficients.
    for key, field in bundle["fields"].items():
        local = ca.MX.sym("local", 3)
        normalized = (local-ca.DM(field["offset"].numpy()))/float(field["scale"])
        bounded = ca.fmin(ca.fmax(normalized, -0.99), 0.99)
        t = ca.fmin(ca.fmax((bounded+1)*0.5, 1e-4), 1-1e-4)
        axes = [ca.vertcat(*[math.comb(n-1, i)*(1-t[j])**(n-1-i)*t[j]**i for i in range(n)]) for j in range(3)]
        basis = ca.kron(ca.kron(axes[0], axes[1]), axes[2])
        squared = ca.sumsqr(normalized-bounded)
        # Guard the sqrt's input as well as its result: gradient is zero for
        # exactly zero residual, matching torch.linalg.vector_norm at zero.
        norm = ca.if_else(squared > 0, ca.sqrt(ca.if_else(squared > 0, squared, 1)), 0)
        value = (ca.dot(ca.DM(field["weights"].numpy()), basis)+norm)*float(field["scale"])
        fields[key] = ca.Function("link_field_"+str(len(fields)), [local], [value])
    for body in contract["bodies"]:
        if body["link"] in excluded_links:
            continue
        rotation, translation = transforms[body["frame_index"]]
        fixed = ca.DM(body["collision_origin"])
        translation = translation + ca.mtimes(rotation, fixed[:3, 3])
        rotation = ca.mtimes(rotation, fixed[:3, :3])
        local = ca.mtimes(rotation.T, z[:, :3].T-translation)
        value = fields[body["field_key"]](local)
        # Keep the first body at an exact tie, as torch.min does.
        best = value if best is None else ca.if_else(value < best, value, best)
    if best is None:
        raise ValueError("No active RDF collision bodies")
    return best


class SharedDistanceField:
    """ctypes reader for parity tests; same layout as the C++ wrapper."""
    def __init__(self, path, expected_units):
        self.library = ctypes.CDLL(str(Path(path).resolve()))
        integer = ctypes.c_longlong
        real = ctypes.POINTER(ctypes.c_double)
        self.fun = self.library.care_distance
        self.fun.argtypes = [ctypes.POINTER(real), ctypes.POINTER(real), ctypes.POINTER(integer), real, ctypes.c_int]
        self.fun.restype = ctypes.c_int
        self.library.care_distance_output_units.restype = ctypes.c_char_p
        self.units = self.library.care_distance_output_units().decode()
        if self.units != expected_units:
            raise ValueError("Distance units mismatch: " + self.units + " != " + expected_units)
        work = self.library.care_distance_work
        work.argtypes = [ctypes.POINTER(integer)]*4
        sizes = [integer() for _ in range(4)]
        if work(*[ctypes.byref(s) for s in sizes]):
            raise ValueError("CasADi work query failed")
        self.sizes = [s.value for s in sizes]
        self.integer, self.real = integer, real

    def __call__(self, inputs):
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != 10 or not np.isfinite(inputs).all():
            raise ValueError("Expected finite paired [B,10] inputs")
        na, nr, ni, nw = self.sizes
        iw, w = (self.integer*max(1, ni))(), (ctypes.c_double*max(1, nw))()
        args, res = (self.real*na)(), (self.real*nr)()
        distances, gradients = np.empty(len(inputs)), np.empty((len(inputs), 7))
        for i, row in enumerate(inputs):
            args[0] = row.ctypes.data_as(self.real)
            res[0] = distances[i:i+1].ctypes.data_as(self.real)
            res[1] = gradients[i].ctypes.data_as(self.real)
            if self.fun(args, res, iw, w, 0):
                raise RuntimeError("Generated distance evaluation failed")
        if not np.isfinite(distances).all() or not np.isfinite(gradients).all():
            raise RuntimeError("Nonfinite generated distance/gradient")
        return distances, gradients


def export(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()
    torch.set_num_threads(1)
    z = ca.MX.sym("x", 1, 10)
    if args.backend == "rdf":
        bundle = load_bundle(checkpoint)
        model = RobotRDF(bundle, args.exclude_link, dtype=torch.float64).eval()
        y = rdf_expression(bundle, z, args.exclude_link)
        units = "m"
        contract = dict(bundle["contract"], excluded_links=args.exclude_link)
        qlimits = np.asarray(contract["joint_limits"])
    else:
        if args.exclude_link:
            raise ValueError("CDF link coverage is fixed by its training data")
        net = CollisionCDF(str(checkpoint), device="cpu", activation=args.activation, checkpoint_key=args.checkpoint_key)
        model = net.model.double().eval()
        y = cdf_expression(net, z)
        units = "configuration_space_model"
        contract = dict(architecture=net.architecture, checkpoint_key=net.selected_checkpoint,
                        limitation="Original collision-CDF training geometry and label scale are not independently verified")
        qlimits = np.tile([-2.5, 2.5], (7, 1))
    gradient = ca.densify(ca.jacobian(y, z)[:, 3:])
    fn = ca.Function("care_distance", [z], [y, gradient], ["x"], ["distance", "gradient_q"])
    value_fn = ca.Function("care_distance_value", [z], [y], ["x"], ["distance"])
    generator = ca.CodeGenerator("distance_field.c", {"with_header": True, "casadi_int": "long long int"})
    generator.add(fn)
    generator.add(value_fn)
    generator.add(value_fn.jacobian())
    generator.generate(str(out)+os.sep)
    fn.save(str(out/"distance_field.casadi"))
    (out/"backend_metadata.c").write_text(
        '#include <stddef.h>\n'
        'int care_distance_abi_version(void) { return 1; }\n'
        'const char* care_distance_backend(void) { return "'+args.backend+'"; }\n'
        'const char* care_distance_output_units(void) { return "'+units+'"; }\n'
        'const char* care_distance_checkpoint_sha256(void) { return "'+sha256(checkpoint)+'"; }\n')
    command = ["gcc", "-O3", "-fPIC", "-shared", "-fno-fast-math", "distance_field.c", "backend_metadata.c", "-lm", "-o", "libcare_distance.so"]
    before = time.monotonic()
    result = subprocess.run(command, cwd=out, capture_output=True, text=True)
    (out/"compile.log").write_text(result.stdout+result.stderr)
    result.check_returncode()
    compile_seconds = time.monotonic()-before
    library = SharedDistanceField(out/"libcare_distance.so", units)
    rng = np.random.default_rng(2631)
    points = rng.uniform([-.8, -.8, .0], [.8, .8, 1.1], (args.checks, 3))
    qs = rng.uniform(qlimits[:, 0], qlimits[:, 1], (args.checks, 7))
    x = np.column_stack((points, qs))
    tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    values = model(tensor).reshape(-1)
    gradients = torch.autograd.grad(values.sum(), tensor)[0][:, 3:].detach().numpy()
    reference = values.detach().numpy()
    actual, actual_grad = library(x)
    # Also check loading through CasADi external, not only the raw C ABI.
    external = ca.external("care_distance", str(out/"libcare_distance.so"))
    ext_values, ext_grad = [], []
    for row in x:
        d, g = external(row.reshape(1, 10))
        ext_values.append(float(d)); ext_grad.append(np.asarray(g).reshape(7))
    errors = dict(value_abs_max=float(np.max(np.abs(actual-reference))),
                  gradient_abs_max=float(np.max(np.abs(actual_grad-gradients))),
                  external_value_abs_max=float(np.max(np.abs(actual-ext_values))),
                  external_gradient_abs_max=float(np.max(np.abs(actual_grad-ext_grad))))
    passed = bool(np.allclose(actual, reference, atol=1e-8, rtol=1e-7) and
                  np.allclose(actual_grad, gradients, atol=1e-7, rtol=1e-6) and
                  errors["external_value_abs_max"] < 1e-12 and errors["external_gradient_abs_max"] < 1e-12)
    np.savez_compressed(out/"parity_inputs_outputs.npz", x=x, distance=actual, gradient=actual_grad,
                        torch_distance=reference, torch_gradient=gradients)
    report = dict(schema=1, backend=args.backend, output_units=units, input="raw xyz metres + q7 radians; [B,10] paired",
                  abi_version=1, function="care_distance", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  library_sha256=sha256(out/"libcare_distance.so"), casadi=ca.__version__, torch=torch.__version__,
                  contract=contract, compile_seconds=compile_seconds, compile_command=command,
                  checks=args.checks, errors=errors, parity_passed=passed,
                  deployment="offline artifact only; planner margins and closed-loop qualification are separate")
    (out/"manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(output=str(out), passed=passed, errors=errors, compile_seconds=compile_seconds)), flush=True)
    if not passed:
        raise RuntimeError("Export value/gradient parity failed; library is not qualified")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cdf", "rdf"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New output directory (existing directories are rejected)")
    parser.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    parser.add_argument("--checkpoint-key", default="latest")
    parser.add_argument("--exclude-link", action="append", default=[])
    parser.add_argument("--checks", type=int, default=64)
    args = parser.parse_args()
    if args.checks < 16:
        parser.error("At least 16 parity probes are required")
    export(args)
