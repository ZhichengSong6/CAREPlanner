#!/usr/bin/env python3
"""Conservative joint-box FK, intersected with the existing global enclosure.

This changes geometry only. It never assumes a nominal future path or weakens
the authorized command set. Current URDF joints must be fixed or rotate about
local +Z; unsupported geometry fails closed. Intervals include roundoff padding.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from phase_e_stop_envelope import JOINTS, vector
from phase_e_stop_shadow import PrimitiveEnclosure, geometry_key


def outward(low, high):
    return np.nextafter(low, -np.inf), np.nextafter(high, np.inf)


def exact(value):
    value = np.asarray(value, dtype=float)
    return value, value


def add(a, b):
    return outward(a[0] + b[0], a[1] + b[1])


def multiply(a, b):
    products = np.stack([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
    return outward(products.min(axis=0), products.max(axis=0))


def matmul(a, b):
    terms = multiply((a[0][:, :, None], a[1][:, :, None]),
                     (b[0][None, :, :], b[1][None, :, :]))
    result = exact(np.zeros((a[0].shape[0], b[0].shape[1])))
    for k in range(a[0].shape[1]):
        result = add(result, (terms[0][:, k, :], terms[1][:, k, :]))
    return result


def trig(low, high, cosine=False):
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError('invalid_trigonometric_interval')
    if high-low >= 2*math.pi:
        return tuple(float(x) for x in outward(-1., 1.))
    shift = 0. if cosine else math.pi/2
    f = math.cos if cosine else math.sin
    values = [f(low), f(high)]
    # Padding covers libm error and critical-point arithmetic near an endpoint.
    first = math.ceil((low-shift)/math.pi-1e-14)
    last = math.floor((high-shift)/math.pi+1e-14)
    values.extend(1. if k % 2 == 0 else -1. for k in range(first, last+1))
    return tuple(float(x) for x in outward(max(-1., min(values)-1e-14), min(1., max(values)+1e-14)))


def rz(low, high):
    s, c = trig(low, high), trig(low, high, True)
    lower = np.array([[c[0], -s[1], 0.], [s[0], c[0], 0.], [0., 0., 1.]])
    upper = np.array([[c[1], -s[0], 0.], [s[1], c[1], 0.], [0., 0., 1.]])
    return lower, upper


def origin_interval(element):
    xyz = np.zeros(3); angles = np.zeros(3)
    if element is not None:
        xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        angles = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    if xyz.shape != (3,) or angles.shape != (3,) or not np.isfinite(np.r_[xyz, angles]).all():
        raise ValueError('invalid_origin')
    roll, pitch, yaw = angles
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])
    return outward(rotation-1e-14, rotation+1e-14), exact(xyz[:, None])


def compose(a, b):
    return matmul(a[0], b[0]), add(matmul(a[0], b[1]), a[1])


class LocalPrimitiveEnclosure(PrimitiveEnclosure):
    """A union of interval FK boxes intersected with the global solid dilation.

    Each sub-box covers a portion of q +/- absolute travel; their union covers
    the complete original joint box. Intersection of two enclosures preserves
    containment. Resource limits raise instead of dropping shapes or points.
    """
    nested_joint_box_coverage = True
    supports_refinement_workspace = True
    def __init__(self, urdf, max_grid_cells=250000, subdivisions=8):
        super().__init__(urdf, max_grid_cells)
        if not isinstance(subdivisions, int) or not 1 <= subdivisions <= 128:
            raise ValueError('invalid_subdivision_limit')
        self.subdivisions = subdivisions
        root = ET.parse(urdf).getroot()
        self.parents = {}
        self.configuration_limits = [None]*len(JOINTS)
        for joint in root.findall('joint'):
            name, kind = joint.get('name'), joint.get('type')
            if joint.find('mimic') is not None or kind not in ('fixed', 'revolute', 'continuous'):
                raise ValueError('unsupported_joint_'+str(name))
            index = None
            if kind != 'fixed':
                index = JOINTS.index(name)
                limit=joint.find('limit')
                if kind=='continuous':bounds=(-math.inf,math.inf)
                elif limit is None:raise ValueError('missing_joint_limits_'+name)
                else:bounds=(float(limit.get('lower')),float(limit.get('upper')))
                if bounds[0]>=bounds[1]:raise ValueError('invalid_joint_limits_'+name)
                self.configuration_limits[index]=bounds
                axis = joint.find('axis')
                value = np.fromstring(axis.get('xyz', '1 0 0') if axis is not None else '1 0 0', sep=' ')
                if value.shape != (3,) or not np.array_equal(value, [0., 0., 1.]):
                    raise ValueError('unsupported_joint_axis_'+name)
            child = joint.find('child').get('link')
            if child in self.parents:
                raise ValueError('multiple_joint_parents')
            self.parents[child] = (joint.find('parent').get('link'), index,
                                   origin_interval(joint.find('origin')))
        if any(v is None for v in self.configuration_limits):raise ValueError('incomplete_joint_limits')
        self.locals = []
        for shape in self.shapes:
            # Parse the same collision-origin transform independently of FK.
            link = root.find("link[@name='"+shape['link']+"']")
            collision = next(c for c in link.findall('collision') if c.get('name', '') == shape['name'])
            self.locals.append(origin_interval(collision.find('origin')))
        self.ancestors = {}
        for link in self.parents:
            indices=[];current=link;seen=set()
            while current in self.parents:
                if current in seen:raise ValueError('cyclic_kinematics')
                seen.add(current)
                current,index,_=self.parents[current]
                if index is not None:indices.append(index)
            self.ancestors[link]=tuple(sorted(indices))

    def partition(self, q, travel, subdivisions=None):
        q, travel = vector(q, 'q'), vector(travel, 'travel', True)
        count=self.subdivisions if subdivisions is None else subdivisions
        if not isinstance(count,int) or not 1<=count<=128:
            raise ValueError('invalid_subdivision_limit')
        low, high = outward(q-travel, q+travel)
        boxes = [(low, high)]
        weights = np.max(self.radii[self.active], axis=0)
        while len(boxes) < count:
            scores = [np.max((hi-lo)*weights) for lo, hi in boxes]
            k = int(np.argmax(scores))
            if scores[k] < 1e-12:
                break
            lo, hi = boxes.pop(k); j = int(np.argmax((hi-lo)*weights))
            middle = .5*lo[j]+.5*hi[j]
            a, b = hi.copy(), lo.copy(); a[j] = middle; b[j] = middle
            boxes.extend([(lo.copy(), a), (b, hi.copy())])
        return boxes

    def boxes(self, q, travel, margin=.020, subdivisions=None, check_budget=None, workspace=None):
        if not math.isfinite(margin) or margin < 0:
            raise ValueError('invalid_margin')
        all_boxes = [[] for _ in self.shapes]
        shared=None if workspace is None else workspace.setdefault('interval_poses',{})
        for low, high in self.partition(q, travel, subdivisions):
            if check_budget is not None:check_budget()
            cache = {}
            visiting = set()
            def pose(link):
                if link in cache:
                    return cache[link]
                if link in visiting:
                    raise ValueError('cyclic_kinematics')
                indices=self.ancestors.get(link,())
                key=(link,tuple(low[list(indices)]),tuple(high[list(indices)]))
                if shared is not None and key in shared:
                    workspace['interval_pose_hits']=workspace.get('interval_pose_hits',0)+1
                    cache[link]=shared[key]
                    return shared[key]
                visiting.add(link)
                if link not in self.parents:
                    result = exact(np.eye(3)), exact(np.zeros((3, 1)))
                else:
                    parent, index, origin = self.parents[link]
                    result = compose(pose(parent), origin)
                    if index is not None:
                        result = compose(result, (rz(low[index], high[index]), exact(np.zeros((3, 1)))))
                visiting.remove(link); cache[link] = result
                if shared is not None:
                    shared[key]=result
                    workspace['interval_pose_misses']=workspace.get('interval_pose_misses',0)+1
                return result
            for i in self.active:
                shape = self.shapes[i]
                rotation, translation = compose(pose(shape['link']), self.locals[i])
                absolute = np.maximum(np.abs(rotation[0]), np.abs(rotation[1]))
                if shape['kind'] == 'box':
                    extent = np.nextafter(absolute @ shape['half_size'], np.inf)
                elif shape['kind'] == 'cylinder':
                    extent = shape['radius']*np.sqrt(absolute[:, 0]**2+absolute[:, 1]**2)+shape['half_length']*absolute[:, 2]
                elif shape['kind'] == 'sphere':
                    extent = np.full(3, shape['radius'])
                else:
                    raise ValueError('unsupported_primitive')
                # An explicit spatial pad also covers reduction/sqrt roundoff.
                extent = np.nextafter(extent+margin+1e-12, np.inf)
                all_boxes[i].append(outward(translation[0][:, 0]-extent, translation[1][:, 0]+extent))
        return all_boxes

    def raster(self, q, travel, meta, margin=.020, subdivisions=None, check_budget=None, workspace=None):
        from audit_case001_motion_prior import frame_placements, collision_sdf
        import hashlib
        q,travel=vector(q,'q'),vector(travel,'travel',True)
        low,high,dims,resolution=self.prepare_grid(meta)
        # The workspace is owned by ONE adaptive check. It caches geometry,
        # never observations, and cannot cross q/travel/map-geometry bindings.
        context=None
        if workspace is not None:
            binding=(id(self),q.tobytes(),travel.tobytes(),float(margin),
                tuple(low),tuple(high),tuple(dims),float(resolution))
            context=workspace.setdefault(binding,{})
        boxes=self.boxes(q,travel,margin,subdivisions,check_budget,context)
        bounds=self.radii@travel
        points=self.points;mask=np.zeros(len(points),dtype=bool)
        reference_mask=np.zeros(len(points),dtype=bool)
        if context is None or 'placements' not in context:
            full = np.zeros(self.model.nq); full[self.qindices] = q
            placements = frame_placements(self.model, self.data, self.shapes, full)
            if context is not None:
                context['placements']=placements
                context['distances']={}
        else:placements=context['placements']
        inside = True; minima = []; maxima = []
        for i in self.active:
            if check_budget is not None:check_budget()
            shape, placement = self.shapes[i], placements[i]
            inflation = margin+float(bounds[i])
            rotation = np.asarray(placement.rotation)
            if shape['kind'] == 'box':
                extent = np.abs(rotation) @ shape['half_size']
            elif shape['kind'] == 'cylinder':
                extent = shape['radius']*np.sqrt(rotation[:, 0]**2+rotation[:, 1]**2)+shape['half_length']*np.abs(rotation[:, 2])
            else:
                extent = np.full(3, shape['radius'])
            g_low = placement.translation-extent-inflation-1e-12
            g_high = placement.translation+extent+inflation+1e-12
            # Evaluate each primitive's SDF only on its local candidate cells.
            # The exact coordinate comparisons below retain the old mask at
            # floating-point grid boundaries; integer padding only adds work.
            candidates=np.zeros(tuple(dims),dtype=bool)
            clipped=[]
            for a, b in boxes[i]:
                a, b = np.maximum(a, g_low), np.minimum(b, g_high)
                if np.any(a > b):
                    continue
                inside = inside and bool(np.all(a >= low) and np.all(b <= high))
                minima.append(a); maxima.append(b)
                start=np.maximum(0,np.minimum(dims,np.ceil((a-low)/resolution-1e-10))).astype(int)
                end=np.maximum(0,np.minimum(dims,np.floor((b-low)/resolution+1e-10)+1)).astype(int)
                if np.all(start<end):
                    candidates[tuple(slice(int(x),int(y)) for x,y in zip(start,end))]=True
                    clipped.append((a,b))
            ids=np.flatnonzero(candidates.ravel())
            if len(ids):
                local=points[ids];keep=np.zeros(len(ids),dtype=bool)
                for a,b in clipped:
                    keep|=np.all(local>=a,axis=1)&np.all(local<=b,axis=1)
                ids=ids[keep]
                if len(ids):
                    if context is None:distance=collision_sdf(points[ids],shape,placement)
                    else:
                        if i not in context['distances']:
                            context['distances'][i]=np.full(len(points),np.nan)
                        saved=context['distances'][i];missing=ids[np.isnan(saved[ids])]
                        if len(missing):saved[missing]=collision_sdf(points[missing],shape,placement)
                        context['sdf_reused_points']=context.get('sdf_reused_points',0)+len(ids)-len(missing)
                        distance=saved[ids]
                    mask[ids]|=distance<=inflation+1e-12
                    # The reference pose belongs to every joint box. These
                    # interior margin points survive every valid refinement.
                    reference_mask[ids]|=distance<=margin-1e-10
        if not minima:raise ValueError('empty_stopping_enclosure')
        ids=np.flatnonzero(mask);selected = points[ids]
        return dict(points=selected, grid_indices=ids.tolist(),
            inside=inside, primitive_bounds_m=bounds.tolist(),
            reference_pose_grid_indices=np.flatnonzero(reference_mask).tolist(),
            point_set_sha256=hashlib.sha256(np.asarray(selected, dtype='<f8').tobytes()).hexdigest(),
            enclosure_method='global_solid_intersect_interval_fk_boxes',
            minimum=np.min(minima, axis=0).tolist(), maximum=np.max(maxima, axis=0).tolist(),
            subdivisions=self.subdivisions if subdivisions is None else subdivisions)
