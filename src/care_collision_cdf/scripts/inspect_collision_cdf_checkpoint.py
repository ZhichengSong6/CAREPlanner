#!/usr/bin/env python3
"""Inspect and validate the structure of a trained collision CDF checkpoint."""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from collision_cdf_model import MLPRegression, extract_state_dict, infer_mlp_architecture, resolve_activation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--checkpoint-key", default="latest")
    parser.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    print("checkpoint:", os.path.abspath(args.checkpoint))
    print("payload type:", type(payload).__name__)
    if isinstance(payload, dict):
        keys = list(payload.keys())
        print("top-level keys (first 20):", keys[:20])
        numeric = []
        for k in keys:
            try:
                numeric.append(int(k))
            except (TypeError, ValueError):
                pass
        if numeric:
            print("numeric iteration range:", min(numeric), "..", max(numeric))

    state, selected = extract_state_dict(payload, args.checkpoint_key)
    print("selected:", selected)
    print("state tensors:", len(state))
    print("first tensor keys:", list(state.keys())[:10])

    print("all parameter shapes:")
    for key, value in state.items():
        print(" ", key, tuple(value.shape))

    arch = infer_mlp_architecture(state, raw_input_dims=10)
    print("inferred raw input dims:", arch["input_dims"])
    print("inferred encoded input dims:", arch["encoded_input_dims"])
    print("inferred hidden layers:", arch["hidden_layers"])
    print("inferred output dims:", arch["output_dims"])
    print("inferred nerf:", arch["nerf"])
    print("linear dims:", arch["linear_dims"])
    print("activation (must match training):", args.activation)

    model = MLPRegression(
        input_dims=arch["input_dims"],
        output_dims=arch["output_dims"],
        mlp_layers=arch["hidden_layers"],
        act_fn=resolve_activation(args.activation),
        nerf=arch["nerf"],
    )
    model.load_state_dict(state, strict=True)
    print("architecture load: OK")
    x = torch.zeros((4, 10), dtype=torch.float32)
    y = model(x)
    print("forward shape:", tuple(y.shape))
    print("CHECKPOINT OK")


if __name__ == "__main__":
    main()
