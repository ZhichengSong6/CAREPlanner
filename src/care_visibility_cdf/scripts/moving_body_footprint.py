"""Separate invariant mounting geometry from space demanded by actuated links."""

from copy import copy

import numpy as np

from measured_arm_numeric import MeasuredArm


def moving_footprint(body,grid,library=None):
    cached=getattr(body,'moving_course_geometry',None)
    if cached is not None:return cached
    moving,fixed=copy(body),copy(body)
    links=body.task_model.kinematics.frames
    def changes(frame):
        while frame!='base_link':
            frame,_,mode,_,_=links[frame]
            if mode in ('revolute','continuous','prismatic'):return True
        return False
    moving.primitives=[piece for piece in body.primitives if changes(piece['link'])]
    fixed.primitives=[piece for piece in body.primitives if not changes(piece['link'])]
    arm=MeasuredArm(moving,grid,library)
    static=MeasuredArm(fixed,grid,library).membership(np.zeros((1,7)))[0] if fixed.primitives else np.empty(0,int)
    body.moving_course_geometry=arm,static
    return arm,static
