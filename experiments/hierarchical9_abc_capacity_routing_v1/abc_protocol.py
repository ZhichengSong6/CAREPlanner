"""Protocol and lineage checks for the hierarchical9 A/B/C capacity-routing diagnostic.

A/B/C all start from the same completed P0 checkpoint.  Existing experiments,
runtime, URDF and safety code remain read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P2_DIR = REPO / "experiments/hierarchical9_p2_no_boundary_eikonal_v1"
SCRATCH = REPO / "experiments/hierarchical9_scratch_v1"
P3_DIR = REPO / "experiments/hierarchical9_p3_neighborhood_sign_v1"
FORMAT = "care_h9_abc_capacity_routing_v1"
BASE_COMMIT = "ad7950d86c4db5a1e348944bd9584aff48a04dcd"
ARMS = ("A", "B", "C")
RNG_TAG = 74117
FIXED = dict(
    seed=0,
    lr=1e-4,
    steps=2000,
    global_batch_x=4000,
    batch_q=100,
    microbatch_x=250,
    decode_x_chunk=64,
    val_global_batch_x=512,
    val_batch_q=100,
    val_microbatch_x=128,
    val_every=500,
    log_every=100,
    amp="fp16",
    max_amp_retries=16,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p2 = _load_module("abc_p2_protocol", P2_DIR / "protocol.py")
old = p2.old
old.setup_paths()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    old.write_json(Path(path), value)


def fingerprints() -> dict[str, str]:
    paths = sorted(
        p for p in HERE.iterdir()
        if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".md")
    )
    return {str(p.relative_to(REPO)): sha256(p) for p in paths}


def load_reference_root(root: Path):
    """Return immutable cache, completed P0 control and its checkpoint digest."""
    root = Path(root).resolve()
    cache, p0, _p1, refs = p2.load_references(root)
    run = json.loads((root / "P0/run.json").read_text())
    if run.get("status") != "COMPLETE" or run.get("successful_updates", 2000) != 2000:
        if run.get("args", {}).get("steps") != 2000:
            raise ValueError("P0 is not the completed 2000-update control")
    expected = dict(
        format=old.PILOT_FORMAT,
        arm="P0",
        completed=True,
        parent_sha256=old.V1_SHA,
        parent_updates=50000,
        pilot_updates=2000,
        total_updates=52000,
        cache_manifest_sha256=cache.identity,
        out_dim=9,
        frozen_parameters=0,
    )
    for key, value in expected.items():
        if p0.get(key) != value:
            raise ValueError(f"P0 reference mismatch: {key}")
    return cache, p0, refs["P0"]


def p0_model(p0: dict, device: torch.device | str):
    model_cls = old.module("model", old.SCRATCH).HierarchicalVisibilityCDF
    model = model_cls()
    model.load_state_dict(p0["model_state"], strict=True)
    return model.to(device=device, dtype=torch.float32)


def abc_seed_for_update(seed: int, step: int) -> int:
    import numpy as np
    if step <= 0:
        raise ValueError("step must be positive")
    return int(np.random.SeedSequence([seed, step, RNG_TAG]).generate_state(1)[0])


def output_dir(root: Path, arm: str, mode: str) -> Path:
    if arm not in ARMS or mode not in ("smoke", "pilot"):
        raise ValueError((arm, mode))
    base = "abc_capacity_routing_smoke" if mode == "smoke" else "abc_capacity_routing"
    return Path(root).resolve() / base / arm


def evaluation_dir(root: Path, mode: str) -> Path:
    if mode not in ("smoke", "pilot"):
        raise ValueError(mode)
    name = "evaluation_abc_capacity_routing_smoke" if mode == "smoke" else "evaluation_abc_capacity_routing"
    return Path(root).resolve() / name


def make_args(p0: dict, output: Path, arm: str, mode: str) -> argparse.Namespace:
    if arm not in ARMS or mode not in ("smoke", "pilot"):
        raise ValueError((arm, mode))
    cfg = dict(FIXED)
    cfg["steps"] = 2 if mode == "smoke" else FIXED["steps"]
    cfg.update(
        arm=arm,
        mode=mode,
        output=str(Path(output).resolve()),
        artifact_root=p0["args"]["artifact_root"],
        data=p0["args"].get("data"),
        urdf=p0["args"].get("urdf"),
    )
    return argparse.Namespace(**cfg)


def load_abc_checkpoint(path: Path):
    path = Path(path).resolve()
    run_path = path.parent / "run.json"
    if not path.is_file() or not run_path.is_file():
        raise FileNotFoundError(path)
    run = json.loads(run_path.read_text())
    digest = sha256(path)
    if path.name != "final.pt" or run.get("status") != "COMPLETE" or run.get("final_sha256") != digest:
        raise ValueError(f"Incomplete or changed ABC checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False), digest


def assert_checkpoint(cp: dict, p0: dict, p0_sha: str, cache_identity: str, *, require_pilot: bool) -> None:
    arm = cp.get("arm")
    mode = cp.get("mode")
    if arm not in ARMS or mode not in ("smoke", "pilot"):
        raise ValueError("Invalid ABC arm/mode")
    steps = 2 if mode == "smoke" else 2000
    if require_pilot and mode != "pilot":
        raise ValueError("Smoke checkpoint cannot be used as a formal result")
    expected = dict(
        format=FORMAT,
        completed=True,
        parent="P0",
        parent_sha256=p0_sha,
        parent_updates=52000,
        extra_updates=steps,
        total_updates=52000 + steps,
        cache_manifest_sha256=cache_identity,
        optimizer_policy="fresh_Adam_sensor_specific_only_constant_lr_no_scheduler_no_clipping",
        training_objective="original_per_sensor_global_objective_only_no_union_no_consistency",
        union_path_frozen=True,
        rng_tag=RNG_TAG,
    )
    for key, value in expected.items():
        if cp.get(key) != value:
            raise ValueError(f"ABC checkpoint mismatch {arm}: {key}")
    args = vars(make_args(p0, Path(cp["args"]["output"]), arm, mode))
    if cp.get("args") != args:
        raise ValueError(f"ABC {arm} config differs from fixed protocol")
    if cp.get("abc_source_sha256") != fingerprints():
        raise ValueError("ABC source files changed since checkpoint creation")


def load_model_from_checkpoint(cp: dict, p0: dict, device: torch.device | str):
    from abc_model import build_arm
    from eval_compat import legacy_compatible
    base = p0_model(p0, device)
    model = build_arm(base, cp["arm"])
    model.load_state_dict(cp["model_state"], strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    # The legacy planning benchmark assumes one shared feature tensor.  Arm C
    # intentionally has per-sensor private tails, so present a narrow evaluation
    # compatibility interface that calls each real forward_sensor branch exactly.
    return legacy_compatible(model)
