#!/usr/bin/env python3
"""Offline conditional stopping enclosures. This module never releases commands.

The command model assumes no further accepted command, monotone simulation time,
applied command held between updates, and period == elapsed update time <= H.
Physical error and settling bounds must be supplied separately; observations do
not silently become bounds. Geometry uses the production URDF triangle radii.
"""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

JOINTS=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']


def vector(value, name, nonnegative=False):
    a=np.asarray(value,dtype=float)
    if a.shape!=(7,) or not np.all(np.isfinite(a)) or (nonnegative and np.any(a<0)):
        raise ValueError('invalid '+name)
    return a


@dataclass
class StopBudget:
    command_travel_rad: np.ndarray
    command_stop_s: float
    speed_bound_rad_s: np.ndarray
    remaining_lease_s: float
    physical_travel_rad: object = None
    physical_stop_s: object = None


def stopping_budget(applied, target, command_age_s, measurement_age_s,
                    max_update_gap_s, acceleration, velocity_limits,
                    timeout_s=.1, max_step_s=.02,
                    physical_error_rad=None, physical_settle_s=None):
    """Absolute displacement enclosure about the LAST MEASURED configuration.

    physical_error_rad must bound the total measurement/integrated actuator
    error at EVERY instant from measurement through physical rest, including
    motion after commanded zero. It is not just an endpoint tracking error.
    physical_settle_s bounds extra time after commanded zero. Both are required.
    """
    u=vector(applied,'applied'); target=vector(target,'target')
    a=vector(acceleration,'acceleration',True); vmax=vector(velocity_limits,'velocity',True)
    for value,name,positive in [(command_age_s,'command age',False),(measurement_age_s,'measurement age',False),
                                (max_update_gap_s,'update gap',True),(timeout_s,'timeout',True),(max_step_s,'step cap',True)]:
        if not math.isfinite(value) or value<0 or (positive and value==0): raise ValueError('invalid '+name)
    if np.any(a<=0) or np.any(vmax<=0) or np.any(np.abs(u)>vmax+1e-12) or np.any(np.abs(target)>vmax+1e-12):
        raise ValueError('nonpositive limits or speed exceeds limit')
    h=max_update_gap_s
    remaining=max(0.,timeout_s-command_age_s)
    # Magnitude never exceeds both endpoints while tracking a fixed target,
    # including a reversal. Do not use only current speed while still accelerating.
    speed=np.maximum(np.abs(u),np.abs(target))
    effective=a*min(1.,max_step_s/h)
    # One H detects expiry, a second H covers the discrete braking hold.
    travel=vmax*measurement_age_s + speed*(remaining+2*h) + speed**2/(2*effective)
    horizon=remaining+2*h+float(np.max(speed/effective))
    b=StopBudget(travel,horizon,speed,remaining)
    if (physical_error_rad is None)!=(physical_settle_s is None):
        raise ValueError('physical error and settling time must be supplied together')
    if physical_error_rad is not None:
        error=vector(physical_error_rad,'physical error',True)
        if not math.isfinite(physical_settle_s) or physical_settle_s<0: raise ValueError('invalid physical settling time')
        b.physical_travel_rad=travel+error
        b.physical_stop_s=horizon+physical_settle_s
    return b


def primitive_radii(urdf, shapes):
    """Same global joint-axis triangle radii as VbcPrimitiveModel::load.

    R[p,j] bounds distance of EVERY solid point from joint j in ANY configuration.
    Then R @ joint_displacement bounds the complete solid at every configuration
    inside the joint box, not merely sampled vertices of one nominal path.
    """
    root=ET.parse(urdf).getroot()
    parents={j.find('child').attrib['link']:j for j in root.findall('joint')}
    radii=np.zeros((len(shapes),7))
    for i,shape in enumerate(shapes):
        radius=float(np.linalg.norm(shape['local'].translation))+shape['circumradius']
        link=shape['link']; visited=set()
        while link in parents:
            if link in visited: raise ValueError('cyclic kinematics')
            visited.add(link); joint=parents[link]; kind=joint.attrib['type']
            if joint.find('mimic') is not None or kind not in ('fixed','revolute','continuous'):
                raise ValueError('unsupported joint '+joint.attrib['name'])
            if kind!='fixed': radii[i,JOINTS.index(joint.attrib['name'])]=radius
            origin=joint.find('origin')
            if origin is not None:
                xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ')
                if xyz.shape!=(3,) or not np.all(np.isfinite(xyz)): raise ValueError('invalid joint origin')
                radius+=float(np.linalg.norm(xyz))
            link=joint.find('parent').attrib['link']
    return radii


def coverage_check(*, has_physical_contract, state_age_s, max_state_age_s,
                   stop_horizon_s, evidence_valid_for_s, complete_snapshot,
                   inside_map, free_mask):
    """Offline prerequisites; True is conditional coverage, never runtime admission.

    The caller is responsible for evidence provenance, snapshot/version binding,
    static/dynamic-world assumptions and the physical bounds' validity.
    """
    reasons=[]
    if not has_physical_contract: reasons.append('missing_physical_response_bound')
    if (not math.isfinite(state_age_s) or state_age_s<0 or
        not math.isfinite(max_state_age_s) or max_state_age_s<0 or state_age_s>max_state_age_s):
        reasons.append('stale_or_invalid_measured_state')
    if not complete_snapshot: reasons.append('missing_complete_evidence_snapshot')
    if not math.isfinite(stop_horizon_s) or stop_horizon_s<0: reasons.append('invalid_stop_horizon')
    if (not math.isfinite(evidence_valid_for_s) or evidence_valid_for_s<0 or
        evidence_valid_for_s<stop_horizon_s): reasons.append('evidence_expires_before_stop')
    if not inside_map: reasons.append('enclosure_may_leave_map')
    free=np.asarray(free_mask)
    if free.ndim!=1 or not free.size or free.dtype!=np.dtype(bool) or not np.all(free):
        reasons.append('uncovered_or_invalid_grid_centers')
    return dict(conditional_coverage=not reasons,runtime_admission=False,reasons=reasons)
