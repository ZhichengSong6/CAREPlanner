#!/usr/bin/env python3
"""Cross-check the real trained RDF, upstream implementation, generated C and C++."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src/care_collision_cdf/scripts"))
import numpy as np
import torch
from rdf_model import RobotRDF, load_bundle, sha256
from export_distance_field_casadi import SharedDistanceField
from collision_cdf_model import CollisionCDF


def main(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    run = Path(args.rdf_run).resolve()
    rdf_library = Path(args.rdf_library).resolve() if args.rdf_library else run/"casadi/libcare_distance.so"
    bundle = load_bundle(run/"care_arm_rdf.pth")
    rdf = RobotRDF(bundle, dtype=torch.float64).eval()
    rng = np.random.default_rng(7421)
    limits = np.asarray(bundle["contract"]["joint_limits"])
    # Include home, near-limit and varied poses, plus actual local mesh vertices
    # mapped by FK; these exercise the surface neighbourhood, not just far space.
    q = rng.uniform(limits[:, 0]+.01, limits[:, 1]-.01, (96, 7))
    q[:8] = 0
    p = rng.uniform([-.7, -.7, .05], [.7, .7, 1.05], (96, 3))
    transforms = rdf.transformations(torch.tensor(q))
    for i in range(48):
        body = rdf.bodies[i % len(rdf.bodies)]
        import trimesh
        mesh = trimesh.load(body["mesh_path"], force="mesh", process=True)
        vertex = mesh.vertices[(i*73) % len(mesh.vertices)]*np.asarray(body["mesh_scale"])
        fixed = np.asarray(body["collision_origin"])
        vertex = fixed[:3, :3]@vertex+fixed[:3, 3]
        rotation, translation = transforms[body["frame_index"]]
        p[i] = rotation[i].numpy()@vertex+translation[i].numpy()+rng.normal(0, .002, 3)
    x = np.column_stack((p, q))
    native_d, native_g = rdf.pair_distance_and_gradient(p, q)
    native_d, native_g = native_d.numpy(), native_g.numpy()
    shared = SharedDistanceField(rdf_library, "m")
    compiled_d, compiled_g = shared(x)
    value_error = float(np.max(np.abs(compiled_d-native_d)))
    grad_error = float(np.max(np.abs(compiled_g-native_g)))
    if value_error > 1e-8 or grad_error > 1e-7:
        raise RuntimeError("RDF native/generated mismatch")
    # Derivative of the scalar itself, away from discontinuities. Check two
    # scales, and explicitly count any probes crossing a nondifferentiable set.
    derivatives = []
    for epsilon in (1e-5, 2e-6):
        columns = []
        for j in range(7):
            plus, minus = x.copy(), x.copy()
            plus[:, 3+j] += epsilon; minus[:, 3+j] -= epsilon
            columns.append((shared(plus)[0]-shared(minus)[0])/(2*epsilon))
        finite_difference = np.column_stack(columns)
        error = np.abs(finite_difference-compiled_g)
        derivatives.append(dict(epsilon=epsilon, max_absolute_error=float(error.max()),
                                failed_components=int((error > (2e-5+2e-3*np.abs(compiled_g))).sum())))
    if derivatives[-1]["failed_components"]:
        raise RuntimeError("RDF scalar finite-difference check failed: " + str(derivatives))
    # Original upstream forward implementation, with our independently checked
    # URDF transformations replacing its Panda-specific kinematic layer.
    sys.path.insert(0, args.rdf_repo)
    from bf_sdf import BPSDF
    rdf32 = RobotRDF(bundle).eval()
    class Layer:
        def get_transformations_each_link(self, pose, theta):
            matrices = rdf32.transformations(theta)
            out_t = []
            for body in rdf32.bodies:
                rotation, translation = matrices[body["frame_index"]]
                t = torch.cat((torch.cat((rotation, translation[:, :, None]), dim=2),
                               torch.tensor([0.,0.,0.,1.]).reshape(1,1,4).expand(len(theta),1,4)), dim=1)
                t = t@torch.tensor(body["collision_origin"], dtype=torch.float32)
                out_t.append(pose@t)
            return out_t
    upstream_model = torch.load(run/"upstream_link_fields.pth", weights_only=True, map_location="cpu")
    upstream = BPSDF(bundle["n_func"], -1., 1., Layer(), "cpu")
    points32, qs32 = torch.tensor(p[:32], dtype=torch.float32), torch.tensor(q[[0, 12]], dtype=torch.float32)
    with torch.no_grad():
        original, _ = upstream.get_whole_body_sdf_batch(points32, torch.eye(4).repeat(2,1,1), qs32,
                                                       upstream_model, use_derivative=False,
                                                       used_links=list(range(len(rdf32.bodies))))
        matched = rdf32(torch.cat((points32.repeat(2,1), qs32.repeat_interleave(32,dim=0)),dim=1)).reshape(2,32)
    upstream_error = float((original-matched).abs().max())
    if upstream_error > 2e-5:
        raise RuntimeError("Upstream RDF forward mismatch: " + str(upstream_error))
    np.savetxt(out/"cpp_inputs.txt", x, fmt="%.17g")
    cpp = Path(args.cpp_reader).resolve()
    result = subprocess.run([str(cpp), str(rdf_library), "m", str(out/"cpp_inputs.txt")],
                            text=True, capture_output=True, check=True)
    (out/"cpp_outputs.txt").write_text(result.stdout)
    (out/"cpp_stderr.txt").write_text(result.stderr)
    cpp_values = np.loadtxt(out/"cpp_outputs.txt")
    cpp_error = float(np.max(np.abs(cpp_values-np.column_stack((compiled_d, compiled_g)))))
    if cpp_error > 1e-12:
        raise RuntimeError("C++ ABI result mismatch")
    # Actual runtime FP32 CDF, not just a double-precision reconstructed network.
    cdf = CollisionCDF(args.cdf_checkpoint, device="cpu")
    old_d, old_g = cdf.pair_distance_and_gradient(torch.as_tensor(p[:32]), torch.as_tensor(q[:32]))
    cdf_shared = SharedDistanceField(args.cdf_library, "configuration_space_model")
    cdf_d, cdf_g = cdf_shared(x[:32])
    cdf_errors = dict(value_abs_max=float(np.max(np.abs(cdf_d-old_d.numpy()))),
                      gradient_abs_max=float(np.max(np.abs(cdf_g-old_g.numpy()))))
    if cdf_errors["value_abs_max"] > 1e-4 or cdf_errors["gradient_abs_max"] > 3e-4:
        raise RuntimeError("Original FP32 CDF export mismatch")
    timings = {}
    for name, field in (("cdf", cdf_shared), ("rdf", shared)):
        field(x[:1])
        samples = []
        for repeat in range(3):
            start = time.perf_counter(); field(x[:32]); samples.append((time.perf_counter()-start)*1000)
        timings[name] = dict(batch_pairs=32, repeats=3, batch_ms=samples, median_batch_ms=float(np.median(samples)),
                            scope="serial CPU generated C, including ctypes dispatch; not the GPU online pipeline")
    np.savez_compressed(out/"values_and_gradients.npz", x=x, native_distance=native_d, native_gradient=native_g,
                        compiled_distance=compiled_d, compiled_gradient=compiled_g)
    report = dict(status="PASS_COMPONENT_PARITY", pairs=len(x), rdf_value_max_error=value_error,
                  rdf_gradient_max_error=grad_error, scalar_finite_differences=derivatives,
                  upstream_rdf_fp32_value_max_error=upstream_error, cpp_max_error=cpp_error,
                  original_cdf_fp32_errors=cdf_errors, timings=timings,
                  rdf_checkpoint_sha256=sha256(run/"care_arm_rdf.pth"),
                  limitation="No physical signed-label error bound, complete occupied-voxel test, online throughput qualification or closed-loop trial")
    (out/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rdf-run",required=True)
    p.add_argument("--rdf-library",help="Optional library outside the training directory")
    p.add_argument("--rdf-repo",default="/home/zhicheng/Project/RDF")
    p.add_argument("--cdf-checkpoint",default=str(ROOT/"src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt"))
    p.add_argument("--cdf-library",required=True)
    p.add_argument("--cpp-reader",required=True)
    p.add_argument("--output",required=True)
    main(p.parse_args())
