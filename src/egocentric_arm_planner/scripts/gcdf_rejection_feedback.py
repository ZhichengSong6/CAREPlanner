"""Bounded point identities from an exact executable rejection (no ROS needed)."""
import math
import struct


def rejection_points(batch, candidate, margin):
    """Validate the association before returning identities, never old QP rows.

    Executable braking/time-scaled indices are deliberately NOT mapped to local
    indices. Consumers must evaluate these points at their own current q.
    """
    n = int(batch.num_pairs)
    if (batch.header.stamp != candidate.header.stamp or
            batch.header.frame_id != candidate.header.frame_id or
            batch.dof != 7 or len(candidate.joint_names) != 7 or n < 1 or
            len(batch.distance) != n or len(batch.original_timestep) != n or
            len(batch.source_type) != n or len(batch.point_flat) != 3*n or
            len(batch.gradient_flat) != 7*n or len(batch.q_linearization_flat) != 7*n or
            not math.isfinite(margin)):
        raise ValueError('invalid executable batch identity/dimensions')
    points = {}
    for i in range(n):
        k = batch.original_timestep[i]
        p = list(batch.point_flat[3*i:3*i+3])
        q = list(batch.q_linearization_flat[7*i:7*i+7])
        g = list(batch.gradient_flat[7*i:7*i+7])
        d = batch.distance[i]
        if (k < 0 or k >= len(candidate.points) or batch.source_type[i] not in (0, 1) or
                not all(math.isfinite(x) for x in p+q+g+[d])):
            raise ValueError('invalid executable row')
        actual = list(candidate.points[k].positions)
        if (len(actual) != 7 or not all(math.isfinite(x) for x in actual) or
                # GPU transport uses float32. Compare in that exact domain.
                struct.pack('!7f', *q) != struct.pack('!7f', *actual)):
            raise ValueError('executable q mismatch')
        if d < margin:
            key = struct.pack('!3f', *p)
            if key not in points:
                if len(points) == 32:
                    return [], True
                points[key] = p
    return [x for p in points.values() for x in p], False
