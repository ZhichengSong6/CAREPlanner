#!/usr/bin/env python3
"""Nominal rigid-body feasibility diagnostics, not a Gazebo response proof."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin

ROOT = Path(__file__).resolve().parents[1]


def gravity_triangle_bound(urdf, gravity):
    tree = ET.parse(urdf).getroot()
    links = {link.get('name'): link for link in tree.findall('link')}
    children = {}
    joints = tree.findall('joint')
    for joint in joints:
        children.setdefault(joint.find('parent').get('link'), []).append(joint)
    def origin_length(element):
        origin = element.find('origin')
        return float(np.linalg.norm([float(x) for x in origin.get('xyz', '0 0 0').split()])) if origin is not None else 0.
    def moment(link, reach):
        inertial = links[link].find('inertial')
        value = 0.
        if inertial is not None:
            mass = float(inertial.find('mass').get('value'))
            assert mass >= 0 and np.isfinite(mass)
            value = mass * (reach + origin_length(inertial))
        return value + sum(moment(j.find('child').get('link'), reach + origin_length(j))
                           for j in children.get(link, []))
    return {j.get('name'): gravity*moment(j.find('child').get('link'), 0.)
            for j in joints if j.get('type') != 'fixed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', action='append', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples-per-run', type=int, default=1000)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    urdf = ROOT / 'src/arm_description/urdf/Arm.urdf'
    model = pin.buildModelFromUrdf(str(urdf))
    assert model.nq == model.nv == 7
    model.gravity.linear[:] = [0., 0., -9.8]
    data = model.createData()
    limits = model.effortLimit.copy()
    runs = []
    input_sha = {str(urdf): hashlib.sha256(urdf.read_bytes()).hexdigest(),
                 str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for source in args.input_dir:
        result = json.loads((source / 'result.json').read_text())
        assert result['status'] == 'PASS_DIAGNOSTIC_NOT_QUALIFICATION' and result['scenario_set'] == 'matrix'
        path = source / 'physics_steps.jsonl'
        rows = [json.loads(line) for line in path.open()]
        ids = np.unique(np.linspace(1, len(rows)-1, min(args.samples_per_run, len(rows)-1), dtype=int))
        metrics = []
        for index in ids:
            row, previous = rows[index], rows[index-1]
            assert row['t_ns']-previous['t_ns'] == 1000000
            q, v = np.asarray(row['q_before']), np.asarray(row['v_before'])
            acceleration = (np.asarray(row['motor'])-previous['motor'])/.001
            mass = pin.crba(model, data, q)
            mass = np.triu(mass)+np.triu(mass, 1).T
            eig = np.linalg.eigvalsh(mass)
            assert np.min(eig) > 0
            gravity = pin.computeGeneralizedGravity(model, data, q).copy()
            torque = pin.rnea(model, data, q, v, acceleration).copy()
            metrics.append(dict(index=int(index), t_ns=row['t_ns'], q=q.tolist(),
                desired_acceleration=acceleration.tolist(), nominal_required_torque=torque.tolist(),
                gravity_torque=gravity.tolist(), mass_min_eigenvalue=float(eig[0]),
                mass_condition_number=float(eig[-1]/eig[0])))
        runs.append(dict(run=str(source.resolve()), sampled_states=len(metrics),
            max_nominal_required_torque=np.max(np.abs([r['nominal_required_torque'] for r in metrics]), axis=0).tolist(),
            min_sampled_effort_margin=np.min(limits-np.abs([r['nominal_required_torque'] for r in metrics]), axis=0).tolist(),
            max_nominal_gravity_torque=np.max(np.abs([r['gravity_torque'] for r in metrics]), axis=0).tolist(),
            minimum_sampled_mass_eigenvalue=min(r['mass_min_eigenvalue'] for r in metrics),
            maximum_sampled_mass_condition=max(r['mass_condition_number'] for r in metrics),
            states=metrics))
        input_sha[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = dict(status='NOMINAL_DYNAMICS_DIAGNOSTIC_NOT_RESPONSE_QUALIFICATION',
        pinocchio_version=pin.__version__, gravity=[0., 0., -9.8],
        runtime_admission=False, physical_bound_certified=False, new_formal_windows=0,
        model_effort_limits=limits.tolist(),
        nominal_pose_independent_gravity_triangle_bound=gravity_triangle_bound(urdf, 9.8),
        gravity_bound_scope='URDF rigid-body gravity moment triangle inequality; excludes contacts, inertia, Coriolis and solver defects',
        sample_scope='Uniformly selected observed states; nominal inverse dynamics excludes ODE constraint errors and contact impulses',
        response_gaps=['uniform motor/constraint solution error over reachable domain',
                       'uniform angle-integration/projection error',
                       'residual travel and persistent hold after command zero',
                       'future lifecycle/domain enforcement'],
        runs=runs, input_sha256=input_sha)
    (out / 'nominal_dynamics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'input_sha256')}, indent=2))
    for r in runs:
        print(json.dumps({k: v for k, v in r.items() if k != 'states'}))


if __name__ == '__main__':
    main()
