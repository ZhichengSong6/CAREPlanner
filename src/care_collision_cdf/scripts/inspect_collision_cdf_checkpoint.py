#!/usr/bin/env python3
"""Inspect and validate the structure of a trained collision CDF checkpoint."""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from collision_cdf_model import MLPRegression, extract_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--checkpoint-key", default="latest")
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

    model = MLPRegression(
        input_dims=10,
        output_dims=1,
        mlp_layers=[1024, 512, 256, 128, 128],
        nerf=True,
    )
    model.load_state_dict(state, strict=True)
    print("architecture load: OK")
    x = torch.zeros((4, 10), dtype=torch.float32)
    y = model(x)
    print("forward shape:", tuple(y.shape))
    print("CHECKPOINT OK")


if __name__ == "__main__":
    main()
