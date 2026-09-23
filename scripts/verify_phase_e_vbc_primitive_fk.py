#!/usr/bin/env python3
"""Independent Python URDF FK + Coal distance oracle for C++ primitive export."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import coal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src/care_visibility_cdf/scripts'))
from validate_visibility_oracle import URDF, find_chain_joints, fk_transform, rpy_to_rot


def main():
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    urdf = REPO / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    robot = URDF.from_xml_file(str(urdf))
    links = {l.attrib['name']: l for l in ET.parse(urdf).getroot().findall('link')}
    pose_count = distance_count = 0
    max_pose_error = max_distance_error = 0.
    for frame in data['frames']:
        qmap = dict(zip(data['joint_names'], frame['q']))
        for p in frame['primitives']:
            c = links[p['link']].findall('collision')[p['collision_index']]
            assert c.attrib['name'] == p['collision_name']
            origin = c.find('origin')
            xyz = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')
            tf = fk_transform(find_chain_joints(robot, 'base_link', p['link']), qmap)
            expected_rotation = tf[:3, :3] @ rpy_to_rot(*rpy)
            expected_center = tf[:3, :3] @ xyz + tf[:3, 3]
            rotation = np.array(p['rotation']).reshape(3, 3)
            center = np.array(p['center'])
            error = max(np.max(np.abs(expected_rotation - rotation)), np.max(np.abs(expected_center - center)))
            assert error < 1e-10, (p['collision_name'], error)
            max_pose_error = max(max_pose_error, error)
            pose_count += 1
            g = list(c.find('geometry'))[0]
            if g.tag == 'box':
                shape = coal.Box(*np.fromstring(g.attrib['size'], sep=' '))
            elif g.tag == 'cylinder':
                shape = coal.Cylinder(float(g.attrib['radius']), float(g.attrib['length']))
            else:
                shape = coal.Sphere(float(g.attrib['radius']))
            for query in p['queries']:
                # Coal outside distance has an unambiguous point-solid oracle;
                # inside sign/cap/corner cases are separately checked in C++.
                if query['distance'] <= 1e-8:
                    continue
                actual = coal.distance(shape, coal.Transform3s(rotation, center), coal.Sphere(0.),
                                       coal.Transform3s(np.eye(3), np.array(query['point'])),
                                       coal.DistanceRequest(), coal.DistanceResult())
                error = abs(actual - query['distance'])
                assert error < 1e-6, (p['collision_name'], actual, query['distance'])
                max_distance_error = max(max_distance_error, error)
                distance_count += 1
    assert pose_count == 84 and distance_count > 1000
    report = dict(status='PASS', poses=pose_count, coal_outside_distances=distance_count,
                  max_pose_error=float(max_pose_error), max_distance_error=float(max_distance_error))
    (path.parent / 'independent_oracle.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
