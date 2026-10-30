"""Bounded local observation objective; no map completion or budget mutation."""
import math
import hashlib
import numpy as np


class RollingVisibilityGuidance:
    def __init__(self):
        self.sequence = 0
        self.cached = None

    def evaluate(self, runtime, snapshot, points, q, stamp, now, feedback,
                 step_limit, drive_enabled=False, gradient_weight=.35):
        q = np.asarray(q, dtype=float).reshape(7)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        sid = int(snapshot.get('per_sensor_selected_sensor_id', -1))
        mask = np.asarray(snapshot['q_vis_joint_mask'], dtype=float).reshape(7)
        goal = np.asarray(snapshot['q_vis'], dtype=float).reshape(7)
        if (runtime is None or not 0 <= sid < 8 or not 0 <= now-stamp <= .2 or
                not len(points) or not np.all(np.isfinite(points)) or
                not np.all(np.isfinite(q)) or not np.all(np.isfinite(mask)) or
                np.any((mask < 0) | (mask > 1)) or not np.all(np.isfinite(goal)) or
                not math.isfinite(step_limit) or not 0 < step_limit <= .2 or
                not math.isfinite(gradient_weight) or not 0 <= gradient_weight <= .5):
            return None
        # Only exact, fresh per-point map feedback may exclude already seen points.
        subset = points
        if (feedback is not None and feedback['token'] == snapshot['observation_token'] and
                feedback['region'] == snapshot['region_token'] and
                0 <= now-feedback['query_time'] <= .5 and
                np.array_equal(feedback['points'], points) and
                len(feedback['point_seen']) == len(points)):
            unseen = ~np.asarray(feedback['point_seen'], dtype=bool)
            if not np.any(unseen):
                return None
            subset = points[unseen]
        key = (snapshot['observation_token'], sid, tuple(subset.ravel()),
               tuple(mask), float(step_limit), bool(drive_enabled), float(gradient_weight))
        old = self.cached
        # Exact q reuse is safe; a changed q always evaluates the existing field.
        if old is not None and old['key'] == key and np.array_equal(old['q_eval'], q):
            if float(stamp) > old['measured_stamp']:
                self.sequence += 1
            result = dict(old, measured_stamp=float(stamp), sequence=self.sequence)
            self.cached = result
            return result
        value, gradient = runtime.branch_value_and_grad_numpy(subset, q, sid)
        gradient = np.asarray(gradient, dtype=float).reshape(7)*mask
        if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
            return None
        grad_norm = float(np.max(np.abs(gradient)))
        forward = mask*(goal-q)
        gap = float(np.max(np.abs(forward)))
        direction = forward/max(gap, 1e-6)
        if drive_enabled and grad_norm > 1e-6:
            direction += gradient_weight*gradient/grad_norm
        norm = float(np.max(np.abs(direction)))
        step = min(step_limit, max(gap, .01 if drive_enabled and grad_norm > 1e-6 else 0.))
        target = q + (step*direction/norm if norm > 1e-6 else np.zeros(7))
        proposal_value = float(value)
        backoff = 0
        if drive_enabled and grad_norm > 1e-6:
            # A derivative is local. Check at most two bounded finite steps,
            # rather than trusting it across a ReLU/min-head switch.
            proposal_value = runtime.branch_score_numpy(subset, target, sid)
            if not math.isfinite(proposal_value) or proposal_value < value:
                target = q + .5*(target-q)
                backoff = 1
                proposal_value = runtime.branch_score_numpy(subset, target, sid)
                if not math.isfinite(proposal_value) or proposal_value < value:
                    target = q + forward*min(1., step_limit/max(gap, 1e-6))
                    backoff = 2
        self.sequence += 1
        result = dict(key=key, sequence=self.sequence, q_eval=q.copy(), target=target,
            gradient=gradient.copy(), value=float(value), sensor_id=sid,
            measured_stamp=float(stamp), observation_token=snapshot['observation_token'],
            region_token=snapshot['region_token'], joint_mask=mask.copy(),
            drive_enabled=bool(drive_enabled), point_count=len(subset))
        # One sensor-pose FK on the exact unseen set. Geometry is a progress
        # hint; only the original map queries can establish actual observation.
        result.update(geometry_valid=False, geometry_signature='none', conservative_fov_margin=0.)
        if hasattr(runtime, '_fov_objective'):
            margin = float(runtime._fov_objective(subset, q, sid))
            if math.isfinite(margin):
                result.update(geometry_valid=True, conservative_fov_margin=margin,
                    geometry_signature=hashlib.sha256(np.asarray(subset, dtype='<f8').tobytes()).hexdigest())
        result['proposal_value'], result['backoff'] = float(proposal_value), backoff
        if old is not None and old['key'] == key:
            result['predicted_gain'] = float(old['gradient'] @ (q-old['q_eval']))
            result['model_gain'] = float(value-old['value'])
        self.cached = result
        return result
