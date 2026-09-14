#!/usr/bin/env python3
"""Exact point-to-URDF-solid geometry for the observational Python logger.

No sphere proxy, grid sampling, YAML or FK library. TF supplies measured link
poses; each target is transformed once per link, then into collision-local axes.
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass


def numbers(text, count):
    values = tuple(float(x) for x in text.split())
    if len(values) != count or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid/nonfinite primitive geometry')
    return values


def rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))


def inverse_point(point, center, rotation):
    x, y, z = (point[i]-center[i] for i in range(3))
    return tuple(rotation[0][i]*x+rotation[1][i]*y+rotation[2][i]*z for i in range(3))


def target_in_link(point, transform):
    t, q = transform.translation, transform.rotation
    values = (t.x, t.y, t.z, q.x, q.y, q.z, q.w)+tuple(point)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite TF or target')
    norm = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    if norm < 1e-12 or not math.isfinite(norm):
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = (v/norm for v in (q.x, q.y, q.z, q.w))
    rotation = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    return inverse_point(point, (t.x, t.y, t.z), rotation)


@dataclass(frozen=True)
class Primitive:
    link: str
    collision_index: int
    kind: str
    center: tuple
    rotation: tuple
    dimensions: tuple

    def distance(self, point):
        x, y, z = inverse_point(point, self.center, self.rotation)
        if self.kind == 'sphere':
            return math.sqrt(x*x+y*y+z*z)-self.dimensions[0]
        if self.kind == 'cylinder':
            a, b = math.hypot(x, y)-self.dimensions[0], abs(z)-self.dimensions[1]/2
            return math.hypot(max(a, 0.), max(b, 0.))+min(max(a, b), 0.)
        gaps = tuple(abs(v)-s/2 for v, s in zip((x, y, z), self.dimensions))
        return math.sqrt(sum(max(v, 0.)**2 for v in gaps))+min(max(gaps), 0.)


def load_primitives(path, ignored_links=()):
    root = ET.parse(path).getroot()
    if root.tag != 'robot':
        raise ValueError('Expected URDF robot')
    out, names = {}, set()
    for link in root.findall('link'):
        name = link.get('name', '')
        if not name or name in names:
            raise ValueError('Missing/duplicate link name')
        names.add(name)
        for index, collision in enumerate(link.findall('collision')):
            geometries = collision.findall('geometry')
            origins = collision.findall('origin')
            if len(geometries) != 1 or len(geometries[0]) != 1 or len(origins) > 1:
                raise ValueError('Missing/ambiguous collision geometry or origin: '+name)
            shape = geometries[0][0]
            if shape.tag == 'box':
                dims = numbers(shape.attrib['size'], 3)
            elif shape.tag == 'cylinder':
                dims = numbers(shape.attrib['radius']+' '+shape.attrib['length'], 2)
            elif shape.tag == 'sphere':
                dims = numbers(shape.attrib['radius'], 1)
            else:
                raise ValueError('Unsupported collision primitive (no fallback): '+shape.tag)
            if any(v <= 0 for v in dims):
                raise ValueError('Nonpositive primitive dimensions')
            origin = origins[0].attrib if origins else {}
            center = numbers(origin.get('xyz', '0 0 0'), 3)
            rotation = rotation_rpy(numbers(origin.get('rpy', '0 0 0'), 3))
            if name not in ignored_links:
                out.setdefault(name, []).append(Primitive(name, index, shape.tag, center, rotation, dims))
    if not out:
        raise ValueError('No active collision primitives')
    return out
