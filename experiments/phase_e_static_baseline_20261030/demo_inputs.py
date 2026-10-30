"""Generate demo inputs from source configuration, without past run records."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out, task):
    runtime = out / 'fixed/runtime'
    urdf = runtime / 'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    helper = runtime / 'src/care_confidence_map/scripts/primitive_point_geometry.py'
    spec = importlib.util.spec_from_file_location('demo_primitive_geometry', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solids = module.load_primitives(urdf)
    tree = ET.parse(urdf).getroot()
    joints = task['joint_names']
    q = dict(zip(joints, task['initial_q']))
    if len(q) != 7 or any(v != 0 for v in q.values()):
        raise ValueError('The provided demo uses the zero-joint start configuration')
    parents = {}
    for node in tree.findall('joint'):
        origin = node.find('origin')
        attrs = origin.attrib if origin is not None else {}
        translation = np.array(module.numbers(attrs.get('xyz', '0 0 0'), 3))
        rotation = np.array(module.rotation_rpy(module.numbers(attrs.get('rpy', '0 0 0'), 3)))
        if node.get('type') != 'fixed' and node.get('name') not in q:
            raise ValueError('Unexpected movable joint: ' + node.get('name'))
        parents[node.find('child').get('link')] = (
            node.find('parent').get('link'), node.get('name'), rotation, translation)
    roots = {n.get('name') for n in tree.findall('link')} - set(parents)
    if len(roots) != 1 or not roots.issubset({'base_link', 'world'}):
        raise ValueError('Expected a single base_link or world root frame')
    poses = {next(iter(roots)): (np.eye(3), np.zeros(3))}

    def pose(link):
        if link not in poses:
            parent, _, rotation, translation = parents[link]
            r, t = pose(parent)
            poses[link] = (r @ rotation, t + r @ translation)
        return poses[link]

    base_rotation, base_translation = pose('base_link')
    if not (np.allclose(base_rotation, np.eye(3), atol=1e-12, rtol=0) and
            np.allclose(base_translation, 0, atol=1e-12, rtol=0)):
        raise ValueError('The demo requires coincident world and base_link frames')

    config = runtime / 'src/care_confidence_map/config/confidence_map_static_box_ray.yaml'
    mapping = yaml.safe_load(config.read_text())
    prior = mapping['current_body_prior']
    if not (prior['enabled'] and not prior['risk_samples_only'] and
            not prior['refresh_on_startup'] and prior['lock_after_complete_refresh']):
        raise ValueError('Unexpected initial body-prior policy')
    reserve = math.sqrt(3) * mapping['confidence_map']['resolution'] / 2
    deviation = float(task['maximum_prearm_joint_deviation_rad'])
    prepared = []
    for link, primitives in solids.items():
        r, t = pose(link)
        for p in primitives:
            rotation = r @ np.asarray(p.rotation)
            center = t + r @ np.asarray(p.center)
            if p.kind == 'box':
                extent = np.abs(rotation) @ (np.asarray(p.dimensions) / 2)
                radius = float(np.linalg.norm(np.asarray(p.dimensions) / 2))
            elif p.kind == 'sphere':
                radius = p.dimensions[0]
                extent = np.full(3, radius)
            else:
                rad, length = p.dimensions
                extent = rad * np.linalg.norm(rotation[:, :2], axis=1) + length / 2 * np.abs(rotation[:, 2])
                radius = math.hypot(rad, length / 2)
            radius += float(np.linalg.norm(p.center))
            bound = 0.0
            ancestor = link
            while ancestor in parents:
                parent, name, _, translation = parents[ancestor]
                if name in q:
                    bound += radius * deviation
                radius += float(np.linalg.norm(translation))
                ancestor = parent
            inflation = prior['link_inflation_radius'].get(link, prior['inflation_radius'])
            prepared.append((p, center, extent, inflation, bound))
    scenes, checks = {}, {}
    for name in task['scenes']:
        world = out / 'fixed/scenes' / (name + '.world')
        model = ET.parse(world).find(".//model[@name='care_static_box']")
        box = None
        rows = []
        if model is not None:
            values = [float(x) for x in model.findtext('pose').split()]
            if model.findtext('static') != 'true' or any(values[3:]):
                raise ValueError('Demo obstacles must be static, axis-aligned boxes')
            center = np.array(values[:3])
            size = np.array([float(x) for x in model.findtext('link/collision/geometry/box/size').split()])
            box = dict(center_m=center.tolist(), size_m=size.tolist())
            for p, c, extent, inflation, motion_bound in prepared:
                # Distance between enclosing AABBs is a lower bound on solid
                # separation; subtract the full prior, voxel and motion reserve.
                lower_bound = float(np.linalg.norm(np.maximum(np.abs(c - center) - extent - size / 2, 0)))
                slack = lower_bound - inflation - reserve - motion_bound
                rows.append(dict(link=p.link, collision_index=p.collision_index,
                                 clearance_lower_bound_m=lower_bound, certified_slack_m=slack))
            if min(row['certified_slack_m'] for row in rows) <= 0:
                raise ValueError('Initial trusted-free envelope intersects scene: ' + name)
        scenes[name] = dict(box=box, world=str(world), world_sha256=sha(world))
        checks[name] = dict(passed=True, primitives=rows)
    scene_path = out / 'fixed/scenes.json'
    save(scene_path, dict(configurations=scenes))
    save(out / 'fixed/bootstrap_gate.json', dict(
        status='PASS', method='conservative_urdf_aabb_bound_computed_at_prepare',
        config_sha256=sha(config), geometry_sha256=sha(urdf), scenes_sha256=sha(scene_path),
        joint_names=joints, initial_q=task['initial_q'],
        maximum_prearm_joint_deviation_rad=deviation,
        voxel_half_diagonal_m=reserve, configurations=checks))
    case = {k: task[k] for k in ('case_id', 'goal_position', 'goal_orientation', 'initial_q')}
    case['difficulty_bin'] = 'demo'
    goals = runtime / 'outputs/phase_e_goal_sampling/phase_e_obstacle_goal_pool_30.json'
    save(goals, dict(benchmark_name='standalone_demo', cases=[case]))
    save(out / 'fixed/demo_configuration.json', dict(
        mode='demo', cases_sha256=sha(goals), case_ids=[task['case_id']],
        qualification='Standalone source-configured demo, not a historical benchmark rerun.'))
    save(out / 'fixed/plan.json', dict(mode='demo', task_wall_budget_s=60, runs=[]))
