"""Read-only provenance, dependency loading and deterministic numeric utilities."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
PS_REL = "experiments/hierarchical9_paired_slope_scratch_v1"
R012_REL = "experiments/hierarchical9_scratch50k_r012_v1"
AUDIT_REL = "experiments/hierarchical9_boundary_audit_v1"
BRANCH = "mainline-b/h9-scratch50k-r012-v1"
# Git blob identities read from the actual repository; fail closed if semantics drift.
PINNED_BLOBS = {
    PS_REL + "/model.py": "0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b",
    R012_REL + "/model.py": "0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b",
    PS_REL + "/data.py": "c5fc55461f6794960bc319a5c24f798b105f7e19",
    PS_REL + "/pairdata.py": "7e21d030d284bb2bd5fce67cf00f58745c98a624",
    PS_REL + "/objective.py": "acaf0c9101038df151ddc304eb4147f8d68f85cf",
    PS_REL + "/protocol.json": "c273a00c8fa71febb8598c251f6c16239558e617",
    AUDIT_REL + "/runtime_probe.py": "ea8c0875f6e996bcd1159dbd14fccf4555f9cbb6",
    AUDIT_REL + "/oracle.py": "de6d93c02753da9f347a314643c2feca0748629e",
}


def read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def blob_sha(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    """Only callers' NEW output paths are written. No writes into source caches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def stats(values) -> dict:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return {"n": int(len(a)), "finite_n": 0, "nonfinite_n": int(len(a)),
                "mean": None, "p50": None, "p95": None, "max": None}
    return {"n": int(len(a)), "finite_n": int(len(finite)),
            "nonfinite_n": int(len(a)-len(finite)), "mean": float(finite.mean()),
            "p50": float(np.median(finite)), "p95": float(np.quantile(finite, .95)),
            "max": float(finite.max())}


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor contents, not serialization containers (best/final may differ there)."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(a.dtype), list(a.shape)], separators=(",", ":")).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def cohort(spec: dict) -> str:
    group = spec["group"]
    if "/local_" in group:
        return "local"
    if group.endswith("uniform_outside"):
        return "uniform"
    raise ValueError(f"Unknown frozen cohort: {group}")


def checked_starts(path: Path, cfg: dict) -> list[dict]:
    if sha(path) != cfg["starts_sha256"]:
        raise ValueError("Frozen starts SHA256 mismatch; no resampling is allowed")
    if read_json(path.parent / "manifest.json").get("starts_sha256") != cfg["starts_sha256"]:
        raise ValueError("Frozen starts source manifest mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != cfg["solver_cases"]:
        raise ValueError("Incomplete starts")
    for row in rows:
        if not 0 <= int(row["sensor"]) < 8:
            raise ValueError("Invalid sensor in frozen starts")
        if np.shape(row["q_init"]) != (7,) or np.shape(row["x"]) != (3,):
            raise ValueError("Bad start dimensions")
        if not np.isfinite(row["q_init"]).all() or not np.isfinite(row["x"]).all():
            raise ValueError("Nonfinite start")
        cohort(row)
    if sum(cohort(r) == "local" for r in rows) != 683:
        raise ValueError("Frozen cohort composition changed")
    return rows


def source_fingerprints(repo: Path) -> dict:
    for rel, expected in PINNED_BLOBS.items():
        if blob_sha(repo / rel) != expected:
            raise ValueError(f"Reviewed upstream dependency changed: {rel}")
    # Hash all local Python dependencies in these directories, not just the entrypoint.
    roots = [repo / "src/care_visibility_cdf/scripts", repo / AUDIT_REL, repo / PS_REL,
             repo / R012_REL, repo / "src/arm_description/urdf"]
    paths = set()
    for root in roots:
        paths.update(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".urdf"))
    return {str(p.relative_to(repo)): sha(p) for p in sorted(paths)}


def check_source_manifest(repo: Path, expected: dict) -> None:
    for rel, digest in expected.items():
        if sha(repo / rel) != digest:
            raise ValueError(f"Repository dependency changed while queued/running: {rel}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def aliases(mapping):
    previous = {k: sys.modules.get(k) for k in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class Dependencies:
    def __init__(self, repo: Path):
        self.repo = repo
        self.model = load_module("joint_audit_r1_model", repo / R012_REL / "model.py")
        self.data = load_module("joint_audit_ps_data", repo / PS_REL / "data.py")
        with aliases({"data": self.data}):
            self.pairdata = load_module("joint_audit_pairdata", repo / PS_REL / "pairdata.py")
        self.objective = load_module("joint_audit_objective", repo / PS_REL / "objective.py")

    def runtime(self):
        scripts = str(self.repo / "src/care_visibility_cdf/scripts")
        if scripts not in sys.path:
            # Keep this evaluation package first; prevent generic module-name collisions.
            sys.path.insert(1, scripts)
        self.probe = load_module("joint_audit_runtime_probe", self.repo / AUDIT_REL / "runtime_probe.py")
        self.oracle = load_module("joint_audit_oracle", self.repo / AUDIT_REL / "oracle.py")
        from hierarchical_visibility_cdf_model import HierarchicalSensorView
        from train_signed_visibility_cdf_pairwise_replace import (
            VisibilityQ0Dataset, DEFAULT_JOINT_NAMES, DEFAULT_SENSOR_FRAMES)
        self.sensor_view = HierarchicalSensorView
        self.dataset_class = VisibilityQ0Dataset
        self.joint_names = tuple(DEFAULT_JOINT_NAMES)
        self.sensor_frames = tuple(DEFAULT_SENSOR_FRAMES)
        return self


def checkpoint_manifest(paths: dict, cfg: dict, deps: Dependencies):
    r1root = Path(paths["r012_root"]) / "formal/R1"
    psroot = Path(paths["ps_root"]) / "formal"
    rr = read_json(r1root / "run.json")
    pr = read_json(psroot / "run.json")
    if not (rr.get("status") == "COMPLETE" and rr.get("arm") == "R1" and rr.get("successful_updates") == 50000):
        raise ValueError("R1 training provenance invalid")
    if not (pr.get("status") == "COMPLETE" and pr.get("successful_updates") == 50000 and
            pr.get("initialization") == "random_from_scratch"):
        raise ValueError("PS training provenance invalid")
    if pr["cache_identity"] != cfg["cache_identity"]:
        raise ValueError("Wrong PS source cache")
    specs = {"R1": (r1root / "final.pt", cfg["r1_sha256"], "care_h9_scratch50k_r012_v1", 50000),
             "PS_BEST": (psroot / "best_val.pt", cfg["ps_best_sha256"], "care_h9_paired_slope_scratch_v1", pr["best_step"]),
             "PS_FINAL": (psroot / "final.pt", cfg["ps_final_sha256"], "care_h9_paired_slope_scratch_v1", 50000)}
    manifest = {}
    for label, (file, expected, fmt, step) in specs.items():
        digest = sha(file)
        run = rr if label == "R1" else pr
        key = "best_val_sha256" if label == "PS_BEST" else "final_sha256"
        if digest != expected or run.get(key) != digest:
            raise ValueError(f"Wrong/corrupt checkpoint: {label}")
        if label != "R1":
            guard = read_json(file.with_suffix(file.suffix + ".sha256.json"))
            if guard.get("sha256") != digest:
                raise ValueError(f"Checkpoint checksum guard mismatch: {label}")
        # These are the user's own, explicitly hash-pinned checkpoints, not arbitrary uploads.
        cp = torch.load(file, map_location="cpu", weights_only=False)
        if cp.get("format") != fmt or cp.get("step") != step or cp.get("initialization") != "random_from_scratch":
            raise ValueError(f"Checkpoint metadata mismatch: {label}")
        if label == "R1":
            if cp.get("arm") != "R1" or cp.get("completed") is not True:
                raise ValueError("Not the original completed R1")
        else:
            for key in ("cache_identity", "index_identity", "pair_identity", "package_identity", "protocol"):
                if cp.get(key) != pr.get(key):
                    raise ValueError(f"PS {key} mismatch")
        model = deps.model.build_model("R1")
        model.load_state_dict(cp["model_state"], strict=True)
        manifest[label] = {"file": str(file), "file_sha256": digest, "step": step,
                           "state_sha256": state_digest(cp["model_state"])}
        del model, cp
    alias = {k: k for k in manifest}
    if manifest["PS_BEST"]["state_sha256"] == manifest["PS_FINAL"]["state_sha256"]:
        alias["PS_BEST"] = "PS_FINAL"
    return manifest, alias, pr


def load_models(manifest: dict, alias: dict, deps: Dependencies, device):
    models = {}
    for name in sorted(set(alias.values()), key=lambda n: (n != "R1", n)):
        meta = manifest[name]
        if sha(Path(meta["file"])) != meta["file_sha256"]:
            raise ValueError("Checkpoint changed after preflight")
        cp = torch.load(meta["file"], map_location="cpu", weights_only=False)
        model = deps.model.build_model("R1")
        model.load_state_dict(cp["model_state"], strict=True)
        if state_digest(model.state_dict()) != meta["state_sha256"]:
            raise ValueError("Model-state mismatch")
        models[name] = model.to(device).eval().requires_grad_(False)
        del cp
    return models


def assert_unchanged(models, manifest):
    for name, model in models.items():
        if state_digest(model.state_dict()) != manifest[name]["state_sha256"]:
            raise RuntimeError(f"Read-only audit unexpectedly changed weights: {name}")


def assert_disjoint_outputs(out: Path, sources: list[Path]):
    out = out.resolve()
    for source in sources:
        source = source.resolve()
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError(f"Audit output overlaps protected source: {source}")


def urdf_masks(urdf: Path, joints: tuple, frames: tuple) -> np.ndarray:
    """Ancestry-only, independent of cached masks and FK arithmetic."""
    import xml.etree.ElementTree as ET
    parents = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        parents[joint.find("child").attrib["link"]] = (joint.find("parent").attrib["link"], joint.attrib["name"])
    result = np.zeros((len(frames), len(joints)), dtype=np.float32)
    for s, frame in enumerate(frames):
        seen = set()
        while frame != "base_link":
            if frame in seen or frame not in parents:
                raise ValueError("Missing/cyclic URDF ancestry")
            seen.add(frame)
            frame, jname = parents[frame]
            if jname in joints:
                result[s, joints.index(jname)] = 1
    return result
