"""Frozen protocol for E0/E1/E2 end-to-end hierarchical9 training."""
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
ABC_DIR = REPO / "experiments/hierarchical9_abc_capacity_routing_v1"
SCRATCH = REPO / "experiments/hierarchical9_scratch_v1"
P3_DIR = REPO / "experiments/hierarchical9_p3_neighborhood_sign_v1"
FORMAT = "care_h9_e012_end2end_v1"
BASE_COMMIT = "1265a42d2ee3730612d338717f6c1bc22d9db099"
ARMS = ("E0", "E1", "E2")
GLOBAL_RNG_TAG = 880031
MINER_RNG_TAG = 880037

FIXED = dict(
    seed=0,
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
    lr_early=1e-5,
    lr_union=5e-5,
    lr_sensor=1e-4,
    hard_warmup=500,
    hard_weight=1.0,
    hard_fn_ratio=0.25,
    replay_capacity_per_sensor=4096,
    replay_sample_per_sensor=64,
    mine_every=20,
    mine_x_per_sensor=4,
    mine_q_per_x=8,
    mine_projection_iters=10,
    mine_projection_damping=0.5,
    mine_projection_max_step=0.25,
    mine_ambiguous_g_m=1e-5,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


abc = _load("e012_abc_protocol", ABC_DIR / "abc_protocol.py")
old = abc.old
old.setup_paths()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    old.write_json(Path(path), value)


def training_fingerprints() -> dict[str, str]:
    names = ("e012_protocol.py", "e2e_model.py", "routed_objective.py", "replay.py", "train_arm.py")
    return {str((HERE/n).relative_to(REPO)): sha256(HERE/n) for n in names}


def all_fingerprints() -> dict[str, str]:
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(HERE.iterdir())
            if p.is_file() and p.suffix in (".py", ".sh", ".sbatch", ".md")}


def load_reference_root(root: Path):
    return abc.load_reference_root(Path(root).resolve())


def p0_model(p0: dict, device):
    return abc.p0_model(p0, device)


def seed_for_update(seed: int, step: int) -> int:
    import numpy as np
    if step <= 0:
        raise ValueError(step)
    return int(np.random.SeedSequence([seed, step, GLOBAL_RNG_TAG]).generate_state(1)[0])


def selected_shared_sensor(step: int) -> int:
    if step <= 0:
        raise ValueError(step)
    return (step - 1) % 8


def output_dir(root: Path, arm: str, mode: str) -> Path:
    if arm not in ARMS or mode not in ("smoke", "pilot"):
        raise ValueError((arm, mode))
    parent = "e012_end2end_smoke" if mode == "smoke" else "e012_end2end"
    return Path(root).resolve() / parent / arm


def evaluation_dir(root: Path, mode: str) -> Path:
    if mode not in ("smoke", "pilot"):
        raise ValueError(mode)
    return Path(root).resolve() / ("evaluation_e012_end2end_smoke" if mode == "smoke" else "evaluation_e012_end2end")


def make_args(p0: dict, output: Path, arm: str, mode: str) -> argparse.Namespace:
    if arm not in ARMS or mode not in ("smoke", "pilot"):
        raise ValueError((arm, mode))
    cfg = dict(FIXED)
    if mode == "smoke":
        cfg.update(steps=3, val_every=1, log_every=1, mine_every=1,
                   mine_x_per_sensor=2, mine_q_per_x=2,
                   replay_sample_per_sensor=8, hard_warmup=2)
    cfg.update(
        arm=arm, mode=mode, output=str(Path(output).resolve()),
        artifact_root=p0["args"]["artifact_root"],
        data=p0["args"].get("data"), urdf=p0["args"].get("urdf"),
    )
    return argparse.Namespace(**cfg)


def arm_definition(arm: str) -> dict:
    if arm == "E0":
        return {"shared_routing": "normal_all_head_sum", "runtime_hard_replay": False}
    if arm == "E1":
        return {"shared_routing": "union_priority_plus_one_sensor_round_robin", "runtime_hard_replay": False}
    if arm == "E2":
        return {"shared_routing": "union_priority_plus_one_sensor_round_robin", "runtime_hard_replay": True}
    raise ValueError(arm)


def load_checkpoint(path: Path):
    path = Path(path).resolve()
    run_path = path.parent / "run.json"
    if not path.is_file() or not run_path.is_file():
        raise FileNotFoundError(path)
    run = json.loads(run_path.read_text())
    digest = sha256(path)
    if path.name != "final.pt" or run.get("status") != "COMPLETE" or run.get("final_sha256") != digest:
        raise ValueError(f"Incomplete or changed checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False), digest


def assert_checkpoint(cp: dict, p0: dict, p0_sha: str, cache_identity: str, *, require_pilot: bool) -> None:
    arm, mode = cp.get("arm"), cp.get("mode")
    if arm not in ARMS or mode not in ("smoke", "pilot"):
        raise ValueError("Invalid E012 arm/mode")
    args = make_args(p0, Path(cp["args"]["output"]), arm, mode)
    expected = dict(
        format=FORMAT, completed=True, parent="P0", parent_sha256=p0_sha,
        parent_updates=52000, extra_updates=args.steps, total_updates=52000+args.steps,
        cache_manifest_sha256=cache_identity,
        optimizer_policy="fresh_Adam_param_groups_constant_lr_no_scheduler_no_clipping",
        original_global_objective=True,
        all_model_parameters_trainable=True,
        arm_definition=arm_definition(arm),
    )
    if require_pilot and mode != "pilot":
        raise ValueError("Smoke checkpoint is not formal")
    for key, value in expected.items():
        if cp.get(key) != value:
            raise ValueError(f"Checkpoint mismatch {arm}: {key}")
    if cp.get("args") != vars(args):
        raise ValueError(f"Checkpoint config mismatch: {arm}")
    if cp.get("training_source_sha256") != training_fingerprints():
        raise ValueError("Training source changed since checkpoint creation")


def load_model_from_checkpoint(cp: dict, p0: dict, device):
    from e2e_model import build_from_p0
    model = build_from_p0(p0_model(p0, device))
    model.load_state_dict(cp["model_state"], strict=True)
    return model.to(device=device, dtype=torch.float32).eval()
