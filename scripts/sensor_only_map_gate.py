#!/usr/bin/env python3
"""Check the bounded startup map and forward measured rays after task start."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import sys

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import String
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/care_visibility_cdf/scripts'))
from validate_visibility_oracle import URDF, find_chain_joints, fk_transform, DEFAULT_JOINT_NAMES
from check_visibility_self_occlusion import load_collision_primitives

WRIST_LINKS = {'wrist_link1', 'wrist_link2', 'wrist_link3', 'EE_link'}


def main():
    rospy.init_node('sensor_only_map_gate', disable_signals=True)
    token = os.environ['CARE_BENCHMARK_ARM_TOKEN']
    assert token and os.environ['CARE_SENSOR_ONLY'] == 'true'
    assert os.environ['CARE_KNOWN_GROUND_ENABLED'] == 'false'
    assert os.environ['CARE_ANALYTIC_GROUND_APPLY'] == 'false'
    self_volume = os.environ.get('CARE_SELF_VOLUME_ONLY') == 'true'
    wrist_clearance = float(os.environ.get('CARE_INITIAL_WRIST_CLEARANCE_M', '0'))
    link_clearances = json.loads(os.environ.get('CARE_INITIAL_LINK_CLEARANCE_M', '{}'))
    assert np.isfinite(wrist_clearance) and wrist_clearance >= 0. and not (wrist_clearance and link_clearances)
    assert all(np.isfinite(x) and x >= 0. for x in link_clearances.values())
    inflation = link_clearances or (dict.fromkeys(WRIST_LINKS, wrist_clearance) if wrist_clearance else {})
    bounded = any(inflation.values())
    assert self_volume or not bounded
    robot = URDF.from_xml_file(str(ROOT / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'))
    primitives = load_collision_primitives(robot)
    assert link_clearances.keys() <= {p['link'] for p in primitives}
    chains = [find_chain_joints(robot, 'base_link', p['link']) for p in primitives]
    out = Path(os.environ['CARE_SENSOR_GATE_TRACE'])
    lock = threading.RLock()
    state = dict(armed=False, arm_ros_ns=None, forwarded=0, discarded=0,
                 prearm_map_samples=0, prearm_free_cells=None, prearm_occupied_cells=None,
                 prearm_environment_free_cells=None, prearm_max_free_self_sdf_m=None,
                 prearm_out_of_envelope_cells=None)
    measured = {}
    pub = rospy.Publisher('/care_planner/perception/task_tof_ray_observations', PointCloud2, queue_size=1)
    trace = out.open('x', buffering=1)

    def emit(event, **kw):
        trace.write(json.dumps(dict(event=event, monotonic_ns=time.monotonic_ns(),
                                    ros_ns=rospy.Time.now().to_nsec(), **kw)) + '\n')

    def map_cb(msg):
        with lock:
            if state['armed']:
                return
            fields = {f.name: f.offset for f in msg.fields}
            names = ['x', 'y', 'z', 'confidence', 'occupancy']
            if not set(names) <= fields.keys():
                return
            a = np.ndarray((msg.width * msg.height,), dtype=np.dtype(dict(
                names=names, formats=['<f4'] * 5,
                offsets=[fields[n] for n in names], itemsize=msg.point_step)), buffer=msg.data)
            free = (a['confidence'] > .5) & (a['occupancy'] <= .5)
            points = np.column_stack([a[n][free] for n in ('x', 'y', 'z')])
            outside = len(points)
            out_of_envelope = outside
            maximum = None
            if self_volume and points.size and all(n in measured for n in DEFAULT_JOINT_NAMES):
                minimum = np.full(len(points), np.inf)
                envelope = np.full(len(points), np.inf)
                cache = {}
                for p, chain in zip(primitives, chains):
                    if p['link'] not in cache:
                        cache[p['link']] = fk_transform(chain, measured)
                    pose = cache[p['link']] @ p['origin']
                    local = (points - pose[:3, 3]) @ pose[:3, :3]
                    g = p['geometry']
                    if p['type'] == 'sphere':
                        sd = np.linalg.norm(local, axis=1) - g.radius
                    else:
                        delta = np.abs(local) - np.asarray(g.size) / 2 if p['type'] == 'box' else np.column_stack((np.linalg.norm(local[:, :2], axis=1) - g.radius, np.abs(local[:, 2]) - g.length / 2))
                        sd = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
                    minimum = np.minimum(minimum, sd)
                    envelope = np.minimum(envelope, sd - inflation.get(p['link'], 0.))
                maximum = float(minimum.max())
                outside = int(np.sum(minimum > 2e-6))
                out_of_envelope = int(np.sum(envelope > 2e-6))
            state.update(prearm_map_samples=state['prearm_map_samples'] + 1,
                         prearm_free_cells=int(free.sum()), prearm_occupied_cells=int(np.sum(a['occupancy'] > .5)),
                         prearm_environment_free_cells=outside, prearm_max_free_self_sdf_m=maximum,
                         prearm_out_of_envelope_cells=out_of_envelope)

    def joint_cb(msg):
        with lock:
            measured.update(zip(msg.name, msg.position))

    def arm_cb(msg):
        with lock:
            if msg.data != token or state['armed']:
                return
            external = state['prearm_environment_free_cells']
            if (not state['prearm_map_samples'] or state['prearm_out_of_envelope_cells'] != 0
                    or state['prearm_occupied_cells'] != 0 or bounded != (external > 0)):
                emit('invalid_initial_map', state=dict(state))
                return
            state.update(armed=True, arm_ros_ns=rospy.Time.now().to_nsec())
            event = 'armed_bounded_initial_prior' if bounded else 'armed_empty_environment'
            emit(event, state=dict(state))

    def rays_cb(msg):
        with lock:
            if not state['armed'] or msg.header.stamp.to_nsec() < state['arm_ros_ns']:
                state['discarded'] += 1
                return
            pub.publish(msg)
            state['forwarded'] += 1
            emit('forwarded', stamp_ns=msg.header.stamp.to_nsec(),
                 sha256=hashlib.sha256(msg.data).hexdigest(), points=msg.width * msg.height)

    emit('started', environment_bootstrap=bounded, initial_robot_volume_known=self_volume,
         initial_wrist_clearance_m=wrist_clearance, initial_link_clearance_m=link_clearances,
         known_ground=False, environment_source='unchanged_link_ToF_rays')
    subs = [rospy.Subscriber('/care_planner/confidence_map/points', PointCloud2, map_cb, queue_size=1, buff_size=2**23),
            rospy.Subscriber('/care_planner/benchmark/arm', String, arm_cb, queue_size=1),
            rospy.Subscriber('/care_arm/joint_states', JointState, joint_cb, queue_size=1),
            rospy.Subscriber('/care_planner/perception/tof_ray_observations', PointCloud2, rays_cb, queue_size=1, buff_size=2**23)]
    try:
        rospy.spin()
    finally:
        with lock:
            emit('finished', state=state)
            trace.close()


if __name__ == '__main__':
    main()
