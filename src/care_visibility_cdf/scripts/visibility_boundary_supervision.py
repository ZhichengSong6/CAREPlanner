"""Signed-distance intervals from an independently checked sign-change bracket.

No globally closest boundary is claimed. For a continuous FOV margin, opposite
signs imply a boundary somewhere on the segment. Convexity of the joint metric
bounds query-to-boundary distance by the farther bracket endpoint.
"""
import numpy as np


def boundary_intervals(minus_q, plus_q, root_q, minus_margin, plus_margin, mask):
    vectors = [np.asarray(v, dtype=float) for v in (minus_q, plus_q, root_q, mask)]
    if any(v.shape != (7,) or not np.all(np.isfinite(v)) for v in vectors):
        raise ValueError('invalid seven-joint boundary bracket')
    minus, plus, root, mask = vectors
    if (not np.isfinite(minus_margin) or not np.isfinite(plus_margin) or
            not minus_margin < 0. < plus_margin or
            np.any((mask != 0.) & (mask != 1.)) or not np.any(mask)):
        raise ValueError('a checked continuous sign-change bracket is required')
    width = float(np.linalg.norm((plus-minus)*mask))
    if width <= 0.:
        raise ValueError('empty effective-joint bracket')
    root_bound = float(max(np.linalg.norm((root-minus)*mask), np.linalg.norm((root-plus)*mask)))
    return dict(minus=(-width, 0.), plus=(0., width), root=(-root_bound, root_bound),
                bracket_width_rad=width, metric='active-joint Euclidean radians',
                supervision='distance intervals, not globally exact regression targets')


def interval_loss(prediction, lower, upper):
    import torch
    if (prediction.shape != lower.shape or lower.shape != upper.shape or
            not torch.isfinite(prediction).all() or not torch.isfinite(lower).all() or
            not torch.isfinite(upper).all() or torch.any(lower > upper)):
        raise ValueError('invalid interval supervision')
    return (torch.relu(lower-prediction).square()+torch.relu(prediction-upper).square()).mean()
