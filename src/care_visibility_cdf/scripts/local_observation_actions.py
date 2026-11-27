"""Group physical R1 steering directions for one observation region and sensor."""

import math
import threading

import numpy as np


class LocalObservationActions:
    def __init__(self):
        self.regions = {}
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.regions.clear()

    def identify(self, region, view, target, mask):
        proposal = view.get("per_sensor_hybrid") or {}
        if proposal.get("proposal_kind") != "local_r1_direction":
            return None
        sensor = view.get("per_sensor_selected_sensor_id")
        if sensor not in range(8) or sensor != proposal.get("selected_sensor_id"):
            return None
        if not any(c.get("sensor_id") == sensor and np.array_equal(c.get("q"), target)
                   for c in proposal.get("candidates", ())):
            return None
        seed = np.asarray(view.get("q_zero"), dtype=float)
        pose, mask = np.asarray(target, dtype=float), np.asarray(mask, dtype=float)
        if any(v.shape != (7,) for v in (seed, pose, mask)):
            return None
        step = (pose - seed) * mask
        length = np.linalg.norm(step)
        if not np.isfinite(length) or length <= 1.e-6:
            return None
        direction = step / length
        with self.lock:
            return self._match(region, sensor, direction)

    def _match(self, region, sensor, direction):
        if region not in self.regions and len(self.regions) >= 64:
            return None
        directions = self.regions.setdefault(region, {}).setdefault(sensor, [])
        if directions:
            scores = np.dot(directions, direction)
            closest = int(np.argmax(scores))
            if scores[closest] >= math.cos(math.pi / 6) or len(directions) >= 8:
                return closest
        directions.append(direction)
        return len(directions) - 1
