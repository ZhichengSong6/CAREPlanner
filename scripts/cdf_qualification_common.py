"""Pure checks and recorded-QP transforms for offline collision CDF qualification."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']
BOX_MIN = np.array([-.8, -.8, 0.])
BOX_MAX = np.array([.8, .8, 1.])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def new_output_dir(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent / 'artifact_sha256.json').exists():
            raise ValueError('Refusing to write inside a sealed artifact directory: ' + str(parent))
    path.mkdir(parents=True, exist_ok=False)
    return path


def in_box(point, lower=BOX_MIN, upper=BOX_MAX):
    p = np.asarray(point, dtype=float)
    return bool(p.shape == (3,) and np.isfinite(p).all() and
                np.all(p >= np.asarray(lower) - 1e-9) and np.all(p <= np.asarray(upper) + 1e-9))


def matrix(encoded):
    a = np.asarray(encoded['coo'], dtype=float).reshape(-1, 3)
    return sparse.coo_matrix((a[:, 2], (a[:, 0].astype(int), a[:, 1].astype(int))),
                             shape=encoded['shape']).tocsr()


def encode(a):
    a = sparse.coo_matrix(a)
    return dict(shape=list(a.shape), coo=np.column_stack((a.row, a.col, a.data)).tolist())


def bar_state(s):
    x = np.zeros(len(s['x_l']))
    q = np.asarray(s['q_bar'])
    u = np.asarray(s['u_bar'])
    n = (len(q) - 1) * 7
    x[:n] = q[1:].reshape(-1)
    x[n:n + u.size] = u.reshape(-1)
    return x


def sample_jacobian(s, row):
    """Recover the saved point-sample affine map; fail if it is ambiguous."""
    old = matrix(s['G']).getrow(row['matrix_row']).toarray()[0]
    g = np.asarray(row['gradient'])
    m = np.zeros((7, len(old)))
    k = int(row['timestep'])
    horizon = len(s['q_bar']) - 1
    state_cols = horizon * 7
    control_cols = np.asarray(s['u_bar']).size
    if 1 <= k <= horizon:
        m[:, (k - 1) * 7:k * 7] = np.eye(7)
        center = np.asarray(row['q_linearization'])
    elif k > horizon:
        for j, value in enumerate(g):
            if abs(value) <= 1e-12:
                raise ValueError('Cannot reconstruct executable sample from a zero saved derivative')
            cols = np.arange(j, state_cols + control_cols, 7)
            m[j, cols] = old[cols] / value
        center = m @ bar_state(s)
    else:
        raise ValueError('Unsupported fixed-start CDF matrix row')
    if np.max(np.abs(g @ m - old)) > 1e-8:
        raise ValueError('Saved affine map does not reproduce the full original row')
    return m, center


def replace_cdf_rows(s, values):
    """Replace only saved selected CDF rows; keep their original model margins."""
    out = copy.deepcopy(s)
    g_matrix = matrix(s['G']).tolil()
    for number, (row, value) in enumerate(zip(s['cdf_rows'], values)):
        m, center = sample_jacobian(s, row)
        d, gradient = float(value['distance']), np.asarray(value['gradient'], dtype=float)
        if gradient.shape != (7,) or not np.isfinite(gradient).all() or not np.isfinite(d):
            raise ValueError('Nonfinite/invalid CDF result')
        index = int(row['matrix_row'])
        g_matrix[index, :] = gradient @ m
        out['h_l'][index] = float(row['margin'] - d + gradient @ center)
        out['cdf_rows'][number].update(distance=d, gradient=gradient.tolist(), reachable_upper=None)
        # These selected CDF rows have one lower bound and no upper bound.
        if s['h_u'][index] < 1e25:
            raise ValueError('Unexpected upper bound on selected CDF row')
    if len(values) != len(s['cdf_rows']):
        raise ValueError('CDF row count mismatch')
    out['G'] = encode(g_matrix.tocsr())
    return out


def feasibility(s):
    a, g = matrix(s['A']), matrix(s['G'])
    lo, hi = np.asarray(s['h_l']), np.asarray(s['h_u'])
    fl, fu = np.isfinite(lo) & (lo > -1e25), np.isfinite(hi) & (hi < 1e25)
    ub = sparse.vstack((g[fu], -g[fl]), format='csr')
    rhs = np.r_[hi[fu], -lo[fl]]
    bounds = [(None if x < -1e25 else x, None if y > 1e25 else y)
              for x, y in zip(s['x_l'], s['x_u'])]
    r = linprog(np.zeros(len(bounds)), A_ub=ub, b_ub=rhs, A_eq=a, b_eq=s['b'],
                bounds=bounds, method='highs', options={'threads': 1,
                'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    if int(r.status) not in (0, 2):
        raise ValueError('Independent feasibility unresolved: ' + str(r.message))
    result = dict(status=int(r.status), success=bool(r.success), message=str(r.message))
    if r.success:
        residual = max(float(np.max(np.abs(a @ r.x - s['b']))), float(np.max(ub @ r.x - rhs)),
                       float(np.max(np.asarray(s['x_l']) - r.x)), float(np.max(r.x - np.asarray(s['x_u']))))
        if residual > 1e-7:
            raise ValueError('Independent feasible solution violates recorded constraints')
        result['max_constraint_residual'] = residual
    return result


def check_manifest(manifest, checkpoint, bundle, manifest_dir, activation):
    """Never infer activation, units, labels or geometry from tensor weights."""
    if not isinstance(manifest, dict):
        return dict(status='UNRESOLVED_TRAINING_CONTRACT', errors=['No training manifest supplied'], compatible=False)
    errors = []
    expected = dict(point_frame='base_link', point_units='metres', joint_units='radians',
                    joint_order=JOINTS, normalization='raw xyz followed by raw q7; positional encoding x,sin(x),cos(x)')
    runtime_input = manifest.get('runtime_input')
    runtime_input = runtime_input if isinstance(runtime_input, dict) else {}
    for key, value in expected.items():
        if runtime_input.get(key) != value:
            errors.append('runtime_input.' + key)
    for key, value in dict(schema=1, checkpoint_sha256=sha(checkpoint), activation=activation,
                           output_units='configuration_space_model', positive_means='outside_collision').items():
        if manifest.get(key) != value:
            errors.append(key)
    if manifest.get('excluded_links') != ['base_link', 'link1']:
        errors.append('excluded_links')
    for key in ('training_urdf', 'label_generator', 'dataset_manifest'):
        evidence = manifest.get(key) or {}
        if not isinstance(evidence, dict) or not isinstance(evidence.get('path'), str) or not evidence['path']:
            errors.append(key + '.evidence')
            continue
        path = Path(evidence['path'])
        if not path.is_absolute():
            path = Path(manifest_dir) / path
        if not path.is_file() or evidence.get('sha256') != sha(path):
            errors.append(key + '.evidence')
    evidence = manifest.get('training_urdf')
    geometry = evidence.get('sha256') if isinstance(evidence, dict) else None
    # Exact primitive geometry is the automatic comparison target. Mesh labels
    # can still be evaluated, but need explicit geometry review before a gate.
    if geometry != bundle['geometry']['urdf_sha256']:
        errors.append('training_geometry_requires_review')
    if not isinstance(manifest.get('label_definition'), str) or not manifest.get('label_definition', '').strip():
        errors.append('label_definition')
    domain = manifest.get('training_domain')
    domain = domain if isinstance(domain, dict) else {}
    if domain.get('xyz_min_m') != BOX_MIN.tolist() or domain.get('xyz_max_m') != BOX_MAX.tolist():
        errors.append('training_domain')
    return dict(status='COMPATIBLE_DECLARED_CONTRACT' if not errors else 'CONTRACT_REVIEW_REQUIRED',
                errors=errors, compatible=not errors,
                limitation='Hashed declarations are provenance, not independent validation of training labels or a global accuracy proof')


def eligible_for_targeted_test(contract, comparison, diagnostic_only, is_old):
    return bool(not diagnostic_only and contract['compatible'] and comparison is not None and not is_old
                and comparison['qps_newly_feasible'] > 0 and comparison['qps_newly_infeasible'] == 0
                and not comparison['new_normal_model_margin_blocks']
                and not comparison['inside_current_body_positive_sign'])
