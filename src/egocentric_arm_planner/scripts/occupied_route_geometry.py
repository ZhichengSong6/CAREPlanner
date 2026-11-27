"""Recheck a measured-state sweep against occupied points from final CDF."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def load_body():
    root = Path(__file__).resolve().parents[3]
    for directory in (root / 'scripts', root / 'src/care_visibility_cdf/scripts',
                      root / 'src/egocentric_arm_planner/scripts'):
        sys.path.append(str(directory))
    from check_visibility_self_occlusion import load_collision_primitives
    from known_ground import load_robot_frames

    robot, names, frames, rotations = load_robot_frames(str(
        root / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'))
    kin = SimpleNamespace(frames=frames, rotations=rotations)
    body = SimpleNamespace(task_model=SimpleNamespace(kinematics=kin),
                           primitives=load_collision_primitives(robot))
    return names, body


def joint_segment(start, end):
    samples = max(2, 1 + int(np.ceil(np.max(abs(end - start)) / .005)))
    return np.linspace(start, end, samples)


def occupied_conflict(body, poses, points, margin):
    from batched_body import point_clearances

    if not len(points):
        return False
    clearance = point_clearances(body, poses, points)
    # A cell already inside the padded envelope may only be left, without
    # going deeper or reentering. An initially clear cell must stay clear.
    floors = np.minimum(margin, np.maximum.accumulate(clearance, axis=0)[:-1])
    return bool(np.any(clearance[1:] < floors - 1e-7))
