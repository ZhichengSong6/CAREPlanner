"""Metric occupied-voxel contract shared by final certification and E5."""
import math


def voxel_volume_clearance(center_clearance, resolution):
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('Invalid occupied voxel resolution')
    return float(center_clearance) - 0.5 * math.sqrt(3.0) * resolution


def occupied_violations(batch, resolution, hard_margin):
    """Return violating row indices and minimum clearance; malformed fails closed.

    The selector must retain its global minimum occupied row before any pair
    truncation. Learned distances are intentionally not interpreted as meters.
    """
    if not math.isfinite(hard_margin):
        raise ValueError('Invalid occupied hard margin')
    n = int(batch.num_pairs)
    sources = list(batch.source_type)
    clearances = list(batch.approx_body_clearance_m)
    if n < 0 or len(sources) != n or len(clearances) != n:
        raise ValueError('Incomplete occupied geometry batch')
    minimum = math.inf
    violated = []
    for i, (source, clearance) in enumerate(zip(sources, clearances)):
        if source not in (0, 1) or not math.isfinite(clearance):
            raise ValueError('Invalid occupied geometry row')
        if source == 1:
            d = voxel_volume_clearance(clearance, resolution)
            minimum = min(minimum, d)
            if d < hard_margin:
                violated.append(i)
    return violated, minimum
