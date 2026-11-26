#!/usr/bin/env python3
"""Verified runtime loader for full-Grid25 20k-step V3-distance H9 R1.

Strictly opt-in: this loader is reached only for either completed full-Grid25
V3-distance 20k baseline or controlled zero-boundary-weighted run. It checks the frozen
run.json, SHA256 and model dimensions before exposing sensors 0..7.

The 9-output model's union is a *proxy* and is not used by the runtime
sensor-only view. V3 distances are best-found FOV boundaries; no FD-qualified
gradient or LOS/self-occlusion safety certificate is implied.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from hierarchical_visibility_cdf_model import (
    HierarchicalSensorView,
    build_private_tail_from_checkpoint_architecture,
)

CHECKPOINT_FORMAT = "care_grid25_h9_r1_full_124seg_v3_distance_training_v1"
BOUNDARY_FORMAT = "care_grid25_h9_r1_full_124seg_v3_boundary_focus_training_v1"
PAIRS_FORMAT = "care_grid25_h9_r1_full_124seg_v3_boundary_pair_aug_training_v1"
SUPPORTED_FORMATS = (CHECKPOINT_FORMAT, BOUNDARY_FORMAT, PAIRS_FORMAT)
TRAINING_DATASET_FORMAT = "care_grid25_full_v3only_distance_snapshot_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("[GRID25 V3 H9 RUNTIME STOP] " + message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_verified_grid25_v3_sensor_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    load_checkpoint,
):
    """Return (8-sensor model, original checkpoint), preserving input gradients.

    load_checkpoint is the existing trusted local PyTorch loader. A file must
    match its sibling frozen run.json's BEST SHA before unpickling any bytes.
    Only the best.pt champion is accepted; final.pt is deliberately excluded.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    require(checkpoint_path.is_file(), f"Checkpoint missing: {checkpoint_path}")
    require(checkpoint_path.name == "best.pt",
            "Only the formally selected best.pt checkpoint is supported")
    run_path = checkpoint_path.parent / "run.json"
    require(run_path.is_file(), f"Training provenance missing: {run_path}")
    meta = json.loads(run_path.read_text())
    require(meta.get("format") in SUPPORTED_FORMATS
            and meta.get("status") == "COMPLETE"
            and meta.get("v3_distance_full_grid25_training") is True
            and meta.get("no_fd_gradient_supervision") is True
            and meta.get("union_proxy_not_exact") is True
            and meta.get("ready_for_original_fd_h9_training") is False,
            "Not a completed, explicitly declared V3-only full-grid H9 training")
    cfg = meta.get("config") or {}
    require(cfg.get("steps") == 20000
            and meta.get("successful_updates") == 20000
            and meta.get("best_step") in range(1, 20001)
            and cfg.get("batch") == 1024
            and cfg.get("seed") == 0
            and math.isclose(float(cfg.get("union_weight", -1)), 0.2),
            "Unexpected full Grid25 training configuration")
    if meta["format"] == BOUNDARY_FORMAT:
        require(cfg.get("boundary_band") == 0.10
                and cfg.get("boundary_boost") == 2.0
                and cfg.get("sensor_loss") == "masked_distance_mse_boundary_weighted"
                and cfg.get("checkpoint_selector") == "val_sensor_rmse"
                and meta.get("boundary_only_change_from_baseline") is True,
                "Not the frozen controlled boundary-focus training experiment")
    if meta["format"] == PAIRS_FORMAT:
        require(math.isclose(float(cfg.get("new_boundary_sample_fraction", -1)),
                             0.25, rel_tol=0, abs_tol=1e-9)
                and cfg.get("checkpoint_selector")=="original_frozen_val_sensor_rmse"
                and cfg.get("no_new_loss_terms") is True
                and meta.get("original_124_segment_dataset_unchanged") is True,
                "Wrong third-round V3 pair-augmentation training protocol")
        augmented_dir=checkpoint_path.parent.parent
        augmented_manifest=augmented_dir/"augmentation_manifest.json"
        require(augmented_manifest.is_file()
                and sha256_file(augmented_manifest)==meta.get("augmentation_manifest_sha256"),
                "Missing or altered qualified V3 boundary-pair augmentation")
        augmented=json.loads(augmented_manifest.read_text())
        require(augmented.get("format")=="care_grid25_v3_boundary_pairs_relabel_v1"
                and augmented.get("status")=="FROZEN"
                and augmented.get("no_los") is True
                and augmented.get("no_fd_gradient_supervision") is True
                and augmented.get("train_pairs_sha256")==meta.get("augmentation_train_sha256")
                and augmented.get("val_pairs_sha256")==meta.get("augmentation_val_sha256"),
                "Third-round augmentation identity/constraints changed")
    run_dataset = Path(meta.get("dataset", "")).expanduser().resolve()
    manifest = run_dataset / "manifest.json"
    require(manifest.is_file(), f"Dataset identity missing: {manifest}")
    frozen = json.loads(manifest.read_text())
    require(frozen.get("format") == TRAINING_DATASET_FORMAT
            and frozen.get("status") == "FROZEN"
            and frozen.get("ready_for_v3_distance_training") is True
            and frozen.get("ready_for_original_fd_h9_training") is False,
            "Wrong or incomplete V3 distance-only frozen dataset")
    counts = frozen.get("counts") or {}
    require(counts.get("segments") == 124
            and counts.get("eligible_x") == 126201
            and counts.get("train_x") == 109490
            and counts.get("val_x") == 16711,
            "Full-grid dataset counts/split changed")
    require(meta.get("dataset_sha256") == frozen.get("labels_sha256"),
            "Training dataset identity differs from frozen label manifest")
    require(isinstance(meta.get("best_sha256"), str)
            and sha256_file(checkpoint_path) == meta["best_sha256"],
            "best.pt checksum mismatch: refuse untrusted or altered checkpoint")
    # Loading a torch checkpoint is pickle-capable in older PyTorch. Accept
    # only locally trusted production files with independently checked SHA.
    ckpt = load_checkpoint(str(checkpoint_path), device)
    require(isinstance(ckpt, dict)
            and ckpt.get("format") == meta["format"]
            and ckpt.get("step") == meta["best_step"]
            and ckpt.get("source_v3_sha256") == frozen["labels_sha256"]
            and ckpt.get("model_source_sha256") == meta.get("model_source_sha256")
            and ckpt.get("trainer_sha256") == meta.get("trainer_sha256")
            and ckpt.get("no_fd_gradient_supervision") is True
            and ckpt.get("union_proxy_not_exact") is True,
            "Checkpoint identity, training provenance or supervision mismatches")
    require(math.isfinite(float(ckpt.get("val_sensor_rmse", math.nan)))
            and math.isclose(
                float(ckpt["val_sensor_rmse"]),
                float(meta.get("best_val_sensor_rmse", math.nan)),
                rel_tol=1e-6, abs_tol=1e-8,
            ),
            "Checkpoint validation metric does not match run.json")
    if meta["format"] == PAIRS_FORMAT:
        require(ckpt.get("augmented_train_sha256")==meta.get("augmentation_train_sha256")
                and ckpt.get("augmented_val_sha256")==meta.get("augmentation_val_sha256")
                and math.isclose(float(ckpt.get("val_new_boundary_rmse", math.nan)),
                                 float(meta.get("best_val_new_boundary_rmse", math.nan)),
                                 rel_tol=1e-6, abs_tol=1e-8),
                "Third-round checkpoint does not match frozen augmented pairs")
    if meta["format"] == BOUNDARY_FORMAT:
        require(math.isclose(
                    float(ckpt.get("val_boundary_sensor_rmse", math.nan)),
                    float(meta.get("best_val_boundary_sensor_rmse", math.nan)),
                    rel_tol=1e-6, abs_tol=1e-8)
                and math.isclose(
                    float(ckpt.get("val_boundary_sensor_sign_accuracy", math.nan)),
                    float(meta.get("best_val_boundary_sign_accuracy", math.nan)),
                    rel_tol=1e-6, abs_tol=1e-8),
                "Boundary-focus checkpoint metrics disagree with completed run")
    architecture = meta.get("model_architecture") or {}
    require(architecture.get("name") == "private_tail_h9"
            and architecture.get("total_parameters", 0) > 0
            and architecture.get("all_parameters_trainable") is True,
            "Unexpected H9 R1 architecture")
    full = build_private_tail_from_checkpoint_architecture(architecture).to(device)
    require(full.parameter_count() == architecture["total_parameters"],
            "Runtime H9 architecture has different parameter count")
    full.load_state_dict(ckpt["model_state"], strict=True)
    full.eval().requires_grad_(False)
    # Evaluate all nine outputs for finite result; do not disable q autograd.
    with torch.no_grad():
        sample = torch.zeros((1, 10), device=device, dtype=torch.float32)
        prediction = full(sample)
        require(prediction.shape == (1, 9)
                and bool(torch.isfinite(prediction).all().item()),
                "Loaded H9 contains nonfinite or malformed output")
    return HierarchicalSensorView(full).to(device).eval(), ckpt
