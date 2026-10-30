#!/usr/bin/env python3
"""Train link Bernstein RDF fields for CARE's actual URDF collision meshes.

Uses the upstream mesh_to_sdf sampling/normal-vote labels. Raw meshes are never
repaired or substituted. Non-watertight meshes and approximation errors are
reported explicitly; the result is an offline model, not a safety certificate.
"""
from __future__ import annotations

import argparse
import gc
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
import trimesh
import mesh_to_sdf
from rdf_model import robot_contract, bernstein_basis, RobotRDF, sha256


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")


def stats(prediction, truth, scale):
    error = (prediction-truth)*scale
    inside = truth*scale < -0.001
    outside = truth*scale > 0.001
    return dict(count=len(truth), mean_absolute_m=float(np.abs(error).mean()),
                p95_absolute_m=float(np.quantile(np.abs(error), .95)),
                max_absolute_m=float(np.abs(error).max()),
                max_overestimate_m=float(error.max()),
                inside_1mm_count=int(inside.sum()), inside_1mm_predicted_nonnegative=int(((prediction>=0)&inside).sum()),
                outside_1mm_count=int(outside.sum()), outside_1mm_predicted_nonpositive=int(((prediction<=0)&outside).sum()))


def fit(points, labels, n, ridge, device):
    """Solve RDF's linear ridge-regression objective using blocked normal equations.

    This avoids repeating the upstream recursive 1280x1280 matrix inversion.
    It uses the same tensor-product Bernstein representation, with a finite,
    saved training set and float64 accumulation/solve for conditioning.
    """
    size = n**3
    a = torch.eye(size, device=device, dtype=torch.float64)*ridge
    b = torch.zeros(size, device=device, dtype=torch.float64)
    for start in range(0, len(points), 2048):
        p = torch.tensor(points[start:start+2048], device=device, dtype=torch.float64)
        y = torch.tensor(labels[start:start+2048], device=device, dtype=torch.float64)
        phi = bernstein_basis(p, n)
        a.addmm_(phi.T, phi)
        b.addmv_(phi.T, y)
    chol, info = torch.linalg.cholesky_ex(a)
    if int(info) != 0:
        raise RuntimeError("RDF ridge system is not positive definite")
    weights = torch.cholesky_solve(b[:, None], chol).reshape(-1)
    residual = float(torch.linalg.vector_norm(a@weights-b)/torch.linalg.vector_norm(b).clamp_min(1e-15))
    if not torch.isfinite(weights).all() or residual > 1e-7:
        raise RuntimeError("Invalid RDF coefficient solve")
    return weights.detach().cpu(), residual


def evaluate(points, weights, n, device):
    out = []
    weights = weights.to(device)
    for start in range(0, len(points), 2048):
        p = torch.tensor(points[start:start+2048], device=device, dtype=torch.float64)
        bounded = p.clamp(-.99, .99)
        value = bernstein_basis(bounded, n)@weights + torch.linalg.vector_norm(p-bounded, dim=1)
        out.append(value.cpu().numpy())
    return np.concatenate(out)


def geometry_labels(mesh, cloud, points, original, closed_components):
    """Use closed-solid ray containment, or explicit depth-scan approximation.

    A point outside the mesh AABB cannot be inside the bounded robot solid.
    This independently justified correction prevents normal-vote outliers from
    becoming negative training labels far outside small/thin components.
    """
    outside_bounds = np.any((points < mesh.bounds[0]-1e-8) | (points > mesh.bounds[1]+1e-8), axis=1)
    if mesh.is_watertight:
        inside = np.zeros(len(points), dtype=bool)
        for component in closed_components:
            candidates = np.flatnonzero(~outside_bounds & ~inside &
                                        np.all(points >= component.bounds[0]-1e-8, axis=1) &
                                        np.all(points <= component.bounds[1]+1e-8, axis=1))
            for start in range(0, len(candidates), 8192):
                ids = candidates[start:start+8192]
                inside[ids] |= component.contains(points[ids])
        method = "union_of_watertight_components_ray_containment"
    else:
        if cloud.scans is None:
            raise ValueError("Non-watertight geometry labels require depth scans")
        inside = ~cloud.is_outside(points)
        method = "depth_scan_approximation_non_watertight"
    inside[outside_bounds] = False
    corrected = np.where(inside, -np.abs(original), np.abs(original))
    return corrected, dict(method=method, total=len(points),
                           normal_sign_disagreements=int(((original < 0) != inside).sum()),
                           negative_normal_labels_outside_aabb=int(((original < 0)&outside_bounds).sum()),
                           inside=int(inside.sum()))


def fk_check(bundle):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(bundle["contract"]["urdf_path"])
    data = model.createData()
    rdf = RobotRDF(bundle, dtype=torch.float64)
    rng = np.random.default_rng(421)
    limits = np.asarray(bundle["contract"]["joint_limits"])
    qs = rng.uniform(limits[:, 0], limits[:, 1], (32, 7))
    transformations = rdf.transformations(torch.tensor(qs))
    errors = []
    for i, q in enumerate(qs):
        qp = pin.neutral(model)
        for j, name in enumerate(bundle["contract"]["joint_order"]):
            joint_id = model.getJointId(name)
            if model.nqs[joint_id] != 1:
                raise ValueError("Pinocchio oracle expects scalar revolute joints")
            qp[model.idx_qs[joint_id]] = q[j]
        pin.forwardKinematics(model, data, qp)
        pin.updateFramePlacements(model, data)
        base = data.oMf[model.getFrameId(bundle["contract"]["base_frame"])].inverse()
        for k, name in enumerate(bundle["contract"]["frames"]):
            expected = base*data.oMf[model.getFrameId(name)]
            rotation, translation = transformations[k]
            errors.extend([float(np.max(np.abs(expected.rotation-rotation[i].numpy()))),
                           float(np.max(np.abs(expected.translation-translation[i].numpy())))])
    maximum = max(errors)
    if maximum > 1e-10:
        raise RuntimeError("Independent Pinocchio FK mismatch: " + str(maximum))
    return dict(poses=len(qs), frames=len(transformations), max_matrix_element_error=maximum, passed=True)


def main(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/"data").mkdir()
    repo = Path(args.rdf_repo).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if not (repo/"bf_sdf.py").is_file():
        raise ValueError("Missing upstream RDF source")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU training fallback")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    contract = robot_contract(args.urdf, args.package_root)
    if args.only_link:
        selected = [b for b in contract["bodies"] if b["link"] == args.only_link]
        if not selected:
            raise ValueError("Unknown probe collision link")
        contract["bodies"] = selected
    settings = vars(args).copy()
    settings.update(rdf_commit=commit, rdf_basis_sha256=sha256(repo/"bf_sdf.py"), torch=str(torch.__version__),
                    training_script_sha256=sha256(__file__), rdf_model_sha256=sha256(ROOT/"src/care_collision_cdf/scripts/rdf_model.py"),
                    device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                    geometry="URDF collision meshes with their declared scales/origins; no mesh replacement",
                    labels=("mesh_to_sdf sampled-surface magnitude; ray sign for closed component unions, depth-scan sign for non-watertight meshes, proven outside-AABB sign" if args.label_method == "geometry" else "mesh_to_sdf normal-majority vote"),
                    fit="blocked float64 ridge normal equations for upstream Bernstein representation")
    save_json(out/"training_settings.json", settings)
    save_json(out/"contract_before_training.json", contract)
    # Check the basis against the actual cloned implementation, not a second
    # local copy of its formula. This comparison does not validate mesh labels.
    sys.path.insert(0, str(repo))
    from bf_sdf import BPSDF
    upstream = BPSDF(args.n_func, -1., 1., None, args.device)
    probes = torch.rand((256, 3), device=args.device)*1.98-.99
    up, _ = upstream.build_basis_function_from_points(probes, False)
    local = bernstein_basis(probes, args.n_func)
    basis_error = float((up-local).abs().max())
    if basis_error > 2e-6:
        raise RuntimeError("Upstream Bernstein basis mismatch")
    del upstream, probes, up, local
    fields, metrics = {}, []
    started = time.monotonic()
    for index, body in enumerate(contract["bodies"]):
        then = time.monotonic()
        print("SAMPLING", index+1, len(contract["bodies"]), body["link"], flush=True)
        mesh = trimesh.load(body["mesh_path"], force="mesh", process=True)
        mesh.vertices = mesh.vertices*np.asarray(body["mesh_scale"])
        offset = np.asarray(mesh.bounding_box.centroid)
        scale = float(np.linalg.norm(mesh.vertices-offset, axis=1).max())
        topology = dict(watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                        vertices=len(mesh.vertices), faces=len(mesh.faces))
        if not topology["winding_consistent"] or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Mesh orientation/scale is invalid: " + body["link"])
        mesh.vertices = (mesh.vertices-offset)/scale
        components = mesh.split(only_watertight=False) if topology["watertight"] and args.label_method == "geometry" else []
        if topology["watertight"] and args.label_method == "geometry" and (not len(components) or not all(c.is_watertight for c in components)):
            raise ValueError("Closed mesh split produced an open component")
        cloud = mesh_to_sdf.get_surface_point_cloud(
            mesh, surface_point_method=args.surface_method, bounding_radius=1.,
            scan_count=args.scan_count, scan_resolution=args.scan_resolution,
            sample_point_count=args.surface_points, calculate_normals=True)
        # Independent random training and validation coordinates. Both use the
        # same label oracle: these are fit errors, not physical-geometry bounds.
        near, near_y = cloud.sample_sdf_near_surface(args.near_points, use_scans=args.surface_method=="scan",
                                                    sign_method="normal", normal_sample_count=11)
        random = np.random.uniform(-.99, .99, (args.far_points, 3)).astype(np.float32)
        far_y = cloud.get_sdf_in_batches(random, sample_count=11, batch_size=8192)
        test_near, test_y = cloud.sample_sdf_near_surface(args.test_points, use_scans=args.surface_method=="scan",
                                                        sign_method="normal", normal_sample_count=11)
        test_random = np.random.uniform(-.99, .99, (args.test_points, 3)).astype(np.float32)
        test_random_y = cloud.get_sdf_in_batches(test_random, sample_count=11, batch_size=8192)
        raw_labels = dict(raw_train_near=near_y.copy(), raw_train_random=far_y.copy(),
                          raw_test_near=test_y.copy(), raw_test_random=test_random_y.copy())
        label_checks = []
        if args.label_method == "geometry":
            near_y, check = geometry_labels(mesh, cloud, near, near_y, components);label_checks.append(dict(split="train_near", **check))
            far_y, check = geometry_labels(mesh, cloud, random, far_y, components);label_checks.append(dict(split="train_random", **check))
            test_y, check = geometry_labels(mesh, cloud, test_near, test_y, components);label_checks.append(dict(split="test_near", **check))
            test_random_y, check = geometry_labels(mesh, cloud, test_random, test_random_y, components);label_checks.append(dict(split="test_random", **check))
        points = np.concatenate((near, random))
        labels = np.concatenate((near_y, far_y))
        # Match the inference extension on occasional near-surface probes that
        # leave its bounded cube; do not train a different clamped function.
        bounded = np.clip(points, -.99, .99)
        labels_bounded = labels-np.linalg.norm(points-bounded, axis=1)
        data_path = out/"data"/(body["link"]+".npz")
        np.savez_compressed(data_path, train_points=points, train_labels=labels,
                            test_near=test_near, test_near_labels=test_y,
                            test_random=test_random, test_random_labels=test_random_y, **raw_labels)
        print("FITTING", body["link"], "samples", len(points), "basis", args.n_func**3, flush=True)
        weights, residual = fit(bounded, labels_bounded, args.n_func, args.ridge, args.device)
        near_pred = evaluate(test_near, weights, args.n_func, args.device)
        far_pred = evaluate(test_random, weights, args.n_func, args.device)
        key = "field_"+str(index)
        body["field_key"] = key
        fields[key] = dict(weights=weights, offset=torch.tensor(offset, dtype=torch.float64), scale=scale)
        metric = dict(link=body["link"], topology=topology, scale_m=scale,
                      label_checks=label_checks,
                      near=stats(near_pred, test_y, scale), random=stats(far_pred, test_random_y, scale),
                      normal_equation_residual=residual, dataset_sha256=sha256(data_path),
                      elapsed_seconds=time.monotonic()-then)
        metrics.append(metric)
        np.savez_compressed(out/"data"/(body["link"]+"_validation.npz"), near_prediction=near_pred, random_prediction=far_pred)
        save_json(out/"link_metrics.json", metrics)
        torch.save(dict(fields=fields, completed_links=[m["link"] for m in metrics], n_func=args.n_func), out/"partial_weights.pth")
        print("DONE", body["link"], "near_p95_mm", round(metric["near"]["p95_absolute_m"]*1000, 4),
              "seconds", round(metric["elapsed_seconds"], 2), flush=True)
        del cloud, mesh, near, near_y, random, far_y, points, labels, bounded, labels_bounded
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    bundle = dict(contract=contract, fields=fields, n_func=args.n_func, training=settings)
    fk = fk_check(bundle)
    checkpoint = out/"care_arm_rdf.pth"
    torch.save(bundle, checkpoint)
    # Upstream-compatible field dictionary for independent forward comparisons.
    torch.save({i: dict(mesh_name=b["link"], weights=fields[b["field_key"]]["weights"].float(),
                       offset=fields[b["field_key"]]["offset"], scale=fields[b["field_key"]]["scale"])
                for i, b in enumerate(contract["bodies"])}, out/"upstream_link_fields.pth")
    report = dict(status="TRAINED_OFFLINE_CANDIDATE", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  bodies=len(contract["bodies"]), n_func=args.n_func, basis_upstream_max_error=basis_error,
                  fk_validation=fk, elapsed_seconds=time.monotonic()-started,
                  non_watertight_links=[m["link"] for m in metrics if not m["topology"]["watertight"]],
                  max_near_p95_m=max(m["near"]["p95_absolute_m"] for m in metrics),
                  max_overestimate_m=max(m[g]["max_overestimate_m"] for m in metrics for g in ("near", "random")),
                  limitation="Validation labels use the same approximate mesh_to_sdf oracle; no certified geometry error bound or online qualification")
    save_json(out/"training_report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rdf-repo", default="/home/zhicheng/Project/RDF")
    p.add_argument("--urdf", default=str(ROOT/"src/arm_description/urdf/Arm.urdf"))
    p.add_argument("--package-root", default=str(ROOT/"src/arm_description"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--n-func", type=int, default=12)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--near-points", type=int, default=32768)
    p.add_argument("--far-points", type=int, default=8192)
    p.add_argument("--test-points", type=int, default=4096)
    p.add_argument("--surface-method", choices=("scan", "sample"), default="scan")
    p.add_argument("--label-method", choices=("normal", "geometry"), default="geometry")
    p.add_argument("--scan-count", type=int, default=32)
    p.add_argument("--scan-resolution", type=int, default=256)
    p.add_argument("--surface-points", type=int, default=500000)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--only-link", help="Isolated sampler/training probe; never a whole-arm model")
    a = p.parse_args()
    if not 2 <= a.n_func <= 24 or a.ridge <= 0 or min(a.near_points,a.far_points,a.test_points) < 128 or a.scan_count < 2:
        p.error("Invalid training sizes or ridge coefficient")
    main(a)
