#!/usr/bin/env python3
"""CARE robot RDF: frozen link Bernstein fields composed through URDF FK.

The representation and bounded-domain extension follow yimingli1998/RDF
(MIT, Idiap Research Institute). This module uses paired queries, with exact
autograd of the scalar field, rather than the upstream Cartesian product and
finite-difference joint derivative. A checkpoint includes the FK and geometry
contract; weights alone do not define a robot distance field.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn

JOINT_ORDER = ["joint1", "joint2", "joint3", "joint4", "wrist_joint1",
               "wrist_joint2", "wrist_joint3"]
SCHEMA = "care_rdf_bernstein_v1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def origin_matrix(node):
    xyz = np.zeros(3) if node is None else np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    rpy = np.zeros(3) if node is None else np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
        raise ValueError("Invalid URDF origin")
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    result = np.eye(4)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                      [-sp, cp*sr, cp*cr]]
    result[:3, 3] = xyz
    return result


def robot_contract(urdf, package_root, base_frame="base_link", joint_order=JOINT_ORDER):
    """Resolve every collision mesh, including fixed sensors and the end effector."""
    urdf, package_root = Path(urdf).resolve(), Path(package_root).resolve()
    root = ET.parse(urdf).getroot()
    links = {node.get("name"): node for node in root.findall("link")}
    if base_frame not in links or len(joint_order) != 7 or len(set(joint_order)) != 7:
        raise ValueError("Invalid base frame or seven-joint order")
    pending = list(root.findall("joint"))
    frames, joints, limits = [base_frame], [], {}
    while True:
        found = next((j for j in pending if j.find("parent").get("link") in frames), None)
        if found is None:
            break
        pending.remove(found)
        name, kind = found.get("name"), found.get("type")
        if kind not in ("fixed", "revolute", "continuous", "prismatic") or found.find("mimic") is not None:
            raise ValueError("Unsupported URDF joint: " + name)
        parent, child = found.find("parent").get("link"), found.find("child").get("link")
        if child in frames:
            raise ValueError("URDF is not a tree")
        qi = -1 if kind == "fixed" else joint_order.index(name)
        axis_node = found.find("axis")
        axis = np.fromstring(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0", sep=" ")
        if kind == "fixed":
            axis = np.array([1.0, 0.0, 0.0])  # URDF fixed joints may specify a zero axis.
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError("Invalid URDF joint axis")
        axis /= np.linalg.norm(axis)
        joints.append(dict(name=name, kind=kind, parent=frames.index(parent), child=len(frames),
                           q_index=qi, axis=axis.tolist(), origin=origin_matrix(found.find("origin")).tolist()))
        frames.append(child)
        if qi >= 0:
            limit = found.find("limit")
            limits[name] = [-math.pi, math.pi] if kind == "continuous" else [float(limit.get("lower")), float(limit.get("upper"))]
    if set(limits) != set(joint_order):
        raise ValueError("URDF movable joints differ from the input contract")
    bodies = []
    for frame in frames:
        for index, collision in enumerate(links[frame].findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                raise ValueError("Mesh training requires an explicit mesh for " + frame)
            uri = mesh.get("filename")
            if uri.startswith("package://"):
                relative = uri[len("package://"):]
                package, relative = relative.split("/", 1)
                if package != package_root.name:
                    raise ValueError("Unresolved mesh package: " + package)
                path = package_root / relative
            else:
                path = (urdf.parent / uri).resolve()
            scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("Invalid mesh scale")
            bodies.append(dict(name=frame + ":" + str(index), link=frame, frame_index=frames.index(frame),
                               mesh_path=str(path), mesh_sha256=sha256(path), mesh_scale=scale.tolist(),
                               collision_origin=origin_matrix(collision.find("origin")).tolist()))
    return dict(schema=SCHEMA, base_frame=base_frame, joint_order=list(joint_order),
                joint_limits=[limits[j] for j in joint_order], frames=frames, joints=joints, bodies=bodies,
                urdf_path=str(urdf), urdf_sha256=sha256(urdf), point_units="m", joint_units="rad",
                output_units="m", sign="outside_positive", domain=[-1.0, 1.0], clamp=0.99)


def bernstein_basis(points, n_func):
    """Same x-major, z-minor coefficient ordering as RDF's BPSDF."""
    t = ((points + 1.0) * 0.5).clamp(1e-4, 1.0-1e-4)
    i = torch.arange(n_func, device=points.device, dtype=points.dtype)
    n = n_func - 1
    coefficients = torch.tensor([math.comb(n, k) for k in range(n_func)],
                                device=points.device, dtype=points.dtype)
    basis = coefficients * (1.0-t[..., None]) ** (n-i) * t[..., None] ** i
    return torch.einsum("bi,bj,bk->bijk", basis[:, 0], basis[:, 1], basis[:, 2]).reshape(-1, n_func**3)


def load_bundle(path):
    # torch.__version__ may be a TorchVersion (a str subclass) in provenance.
    # Permit this one known metadata type without enabling arbitrary pickle.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    contract = bundle.get("contract", {})
    if contract.get("schema") != SCHEMA or contract.get("output_units") != "m":
        raise ValueError("Not a CARE RDF checkpoint with a metre-valued geometry contract")
    if contract.get("joint_order") != JOINT_ORDER or contract.get("domain") != [-1.0, 1.0] or contract.get("clamp") != 0.99:
        raise ValueError("Unsupported RDF input/domain contract")
    n = int(bundle["n_func"])
    if not 2 <= n <= 32 or not contract["bodies"]:
        raise ValueError("Invalid RDF basis size or empty geometry")
    for body in contract["bodies"]:
        field = bundle["fields"][body["field_key"]]
        if field["weights"].shape != (n**3,) or field["offset"].shape != (3,):
            raise ValueError("Malformed RDF coefficient dimensions")
        if not torch.isfinite(field["weights"]).all() or not torch.isfinite(field["offset"]).all() or not math.isfinite(field["scale"]) or field["scale"] <= 0:
            raise ValueError("Nonfinite RDF field or invalid scale")
    return bundle


class RobotRDF(nn.Module):
    def __init__(self, bundle, excluded_links=(), dtype=torch.float32):
        super().__init__()
        self.contract = bundle["contract"]
        self.n_func = int(bundle["n_func"])
        self.excluded_links = tuple(excluded_links)
        if not set(excluded_links).issubset({b["link"] for b in self.contract["bodies"]}):
            raise ValueError("Unknown excluded collision link")
        self.bodies = [b for b in self.contract["bodies"] if b["link"] not in excluded_links]
        if not self.bodies:
            raise ValueError("Cannot exclude every collision body")
        self.field_keys = list(bundle["fields"])
        for k, key in enumerate(self.field_keys):
            self.register_buffer("weights_" + str(k), bundle["fields"][key]["weights"].to(dtype).clone())
            self.register_buffer("offset_" + str(k), bundle["fields"][key]["offset"].to(dtype).clone())
            self.register_buffer("scale_" + str(k), torch.tensor(float(bundle["fields"][key]["scale"]), dtype=dtype))
        for k, joint in enumerate(self.contract["joints"]):
            self.register_buffer("origin_" + str(k), torch.tensor(joint["origin"], dtype=dtype))
            self.register_buffer("axis_" + str(k), torch.tensor(joint["axis"], dtype=dtype))
        for k, body in enumerate(self.bodies):
            self.register_buffer("collision_" + str(k), torch.tensor(body["collision_origin"], dtype=dtype))

    def transformations(self, q):
        batch = len(q)
        eye = torch.eye(3, dtype=q.dtype, device=q.device).expand(batch, 3, 3)
        transforms = [(eye, torch.zeros(batch, 3, dtype=q.dtype, device=q.device))]
        for k, joint in enumerate(self.contract["joints"]):
            rp, tp = transforms[joint["parent"]]
            origin, axis = getattr(self, "origin_"+str(k)), getattr(self, "axis_"+str(k))
            r0 = rp @ origin[:3, :3]
            t0 = tp + (rp @ origin[:3, 3, None]).squeeze(-1)
            if joint["q_index"] >= 0:
                value = q[:, joint["q_index"]]
                if joint["kind"] == "prismatic":
                    t0 = t0 + (r0 @ axis).reshape(batch, 3)*value[:, None]
                else:
                    zero = axis[0]*0
                    skew = torch.stack((zero, -axis[2], axis[1], axis[2], zero, -axis[0], -axis[1], axis[0], zero)).reshape(3, 3)
                    rotation = eye + value.sin()[:, None, None]*skew + (1-value.cos())[:, None, None]*(skew@skew)
                    r0 = r0 @ rotation
            transforms.append((r0, t0))
        return transforms

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != 10:
            raise ValueError("RDF expects paired [B,10] raw xyz + q7")
        transforms = self.transformations(x[:, 3:])
        values = []
        for k, body in enumerate(self.bodies):
            rotation, translation = transforms[body["frame_index"]]
            fixed = getattr(self, "collision_"+str(k))
            translation = translation + (rotation @ fixed[:3, 3, None]).squeeze(-1)
            rotation = rotation @ fixed[:3, :3]
            local = torch.einsum("bi,bij->bj", x[:, :3]-translation, rotation)
            fi = self.field_keys.index(body["field_key"])
            scale, offset, weights = (getattr(self, what+str(fi)) for what in ("scale_", "offset_", "weights_"))
            normalized = (local-offset)/scale
            bounded = normalized.clamp(-0.99, 0.99)
            residual = torch.linalg.vector_norm(normalized-bounded, dim=1)
            values.append((bernstein_basis(bounded, self.n_func) @ weights + residual)*scale)
        return torch.stack(values, dim=1).min(dim=1).values[:, None]

    def pair_distance_and_gradient(self, points_xyz, q):
        dtype, device = next(self.buffers()).dtype, next(self.buffers()).device
        p = torch.as_tensor(points_xyz, dtype=dtype, device=device)
        q = torch.as_tensor(q, dtype=dtype, device=device).detach().clone().requires_grad_(True)
        if p.ndim != 2 or q.ndim != 2 or p.shape != (len(q), 3) or q.shape[1] != 7 or len(q) == 0:
            raise ValueError("Expected nonempty paired [B,3] points and [B,7] joints")
        value = self(torch.cat((p, q), dim=1)).reshape(-1)
        gradient = torch.autograd.grad(value.sum(), q)[0]
        return value.detach(), gradient.detach()
