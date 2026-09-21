"""Float64 offline geometry using the repository's URDF chain specifications.

Upstream batched FK allocates some matrices in FP32. This module has a
 dtype-safe FK for numerical optimization, and MUST pass an upstream FP32
margin parity check plus a finite-difference Jacobian check before labeling.
It does not change any repository module or runtime setting.
"""
from __future__ import annotations
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

DEFAULT_JOINTS = ['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
DEFAULT_SENSORS = ['link2_sensor1_tof_link','link2_sensor2_tof_link',
                   'link3_sensor1_tof_link','link3_sensor2_tof_link',
                   'link4_sensor1_tof_link','link4_sensor2_tof_link',
                   'EE_sensor1_tof_link','EE_sensor2_tof_link']
FOV = dict(horizontal_fov_deg=50., vertical_fov_deg=66., z_min=.2, z_max=.7, delta=.01)
DEPENDENCIES = ['extract_visibility_zero_level_sets.py', 'validate_visibility_oracle.py',
                'check_visibility_self_occlusion.py',
                'train_signed_visibility_cdf_pairwise_replace.py',
                'train_per_sensor_visibility_cdf.py']
REFERENCE_COMMIT = 'e9ada9d502fd622418f5fc1c28a8a52beb863364'


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_identity(repo, urdf):
    repo, urdf = Path(repo).resolve(), Path(urdf).resolve()
    scripts = repo / 'src/care_visibility_cdf/scripts'
    hashes = {str(Path('src/care_visibility_cdf/scripts')/p): sha256_file(scripts/p) for p in DEPENDENCIES}
    try:
        head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = 'UNKNOWN'
    observed_urdf_sha=sha256_file(urdf)
    # Compare with the known R1 code baseline when that object exists locally.
    # No fetch/network operation is performed. A missing object is recorded, not
    # falsely treated as proof of the historical dataset's URDF provenance.
    baseline_sha=None
    try:
        data=subprocess.check_output(['git','-C',str(repo),'show',
            REFERENCE_COMMIT+':src/arm_description/urdf/Arm.urdf'],stderr=subprocess.DEVNULL)
        baseline_sha=hashlib.sha256(data).hexdigest()
    except (OSError,subprocess.CalledProcessError):
        pass
    if baseline_sha is not None and observed_urdf_sha!=baseline_sha:
        raise RuntimeError('Selected URDF differs from the fixed R1 reference commit; no automatic geometry change')
    return dict(repo=str(repo), git_head=head, urdf=str(urdf), urdf_sha256=observed_urdf_sha,
                source_sha256=hashes, reference_commit=REFERENCE_COMMIT,
                baseline_urdf_sha256=baseline_sha,
                baseline_urdf_check='MATCH' if baseline_sha is not None else 'REFERENCE_OBJECT_NOT_AVAILABLE')


class RepoOracle:
    def __init__(self, repo, urdf, device='cpu', joint_names=None, sensor_frames=None):
        self.repo, self.urdf = Path(repo).resolve(), Path(urdf).resolve()
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; select cpu explicitly')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.joints, self.sensors = list(joint_names or DEFAULT_JOINTS), list(sensor_frames or DEFAULT_SENSORS)
        if self.joints != DEFAULT_JOINTS or self.sensors != DEFAULT_SENSORS:
            raise ValueError('Joint/sensor order differs from frozen R1 definition')
        scripts = self.repo/'src/care_visibility_cdf/scripts'
        sys.path.insert(0, str(scripts))
        self.upstream = importlib.import_module('extract_visibility_zero_level_sets')
        self.legacy = importlib.import_module('train_signed_visibility_cdf_pairwise_replace')
        self.per = importlib.import_module('train_per_sensor_visibility_cdf')
        for mod in (self.upstream, self.legacy, self.per):
            if Path(mod.__file__).resolve().parent != scripts:
                raise RuntimeError(f'Wrong repository import: {mod.__file__}')
        from urdf_parser_py.urdf import URDF
        robot = URDF.from_xml_file(str(self.urdf))
        self.specs = self.upstream.prepare_chain_specs(robot, 'base_link', self.sensors, self.joints, self.device)
        limits = self.upstream.get_joint_limits(robot, self.joints)
        self.urdf_lo = np.asarray([limits[n][0] for n in self.joints], dtype=float)
        self.urdf_hi = np.asarray([limits[n][1] for n in self.joints], dtype=float)
        self.masks = np.zeros((8,7), dtype=np.float32)
        self.double_specs = []
        for s, chain in enumerate(self.specs):
            out = []
            for spec in chain:
                if spec['q_index'] >= 0 and spec['type'] != 'fixed':
                    if spec['type'] not in ('revolute', 'continuous', 'prismatic'):
                        raise ValueError(f'Unsupported joint {spec}')
                    self.masks[s, spec['q_index']] = 1
                out.append({**spec, 'origin':spec['origin'].double(), 'axis':spec['axis'].double()})
            self.double_specs.append(out)
        self.identity = source_identity(self.repo, self.urdf)
        if self.identity['baseline_urdf_check']!='MATCH':
            print('[WARN] reference git object unavailable: historical URDF lineage NOT_VERIFIED; inspect preflight.json before training',flush=True)

    def _fk64(self, q, s):
        dtype, device = q.dtype, q.device
        t = torch.eye(4, dtype=dtype, device=device)
        eye = torch.eye(3, dtype=dtype, device=device)
        bottom = torch.tensor([[0.,0.,0.,1.]], dtype=dtype, device=device)
        for spec in self.double_specs[s]:
            t = t @ spec['origin']
            idx, kind = spec['q_index'], spec['type']
            if idx < 0 or kind == 'fixed':
                continue
            a, angle = spec['axis'], q[idx]
            if kind in ('revolute','continuous'):
                zero = a.new_zeros(())
                skew = torch.stack((zero,-a[2],a[1], a[2],zero,-a[0], -a[1],a[0],zero)).reshape(3,3)
                c, si = torch.cos(angle), torch.sin(angle)
                r = c*eye + (1-c)*a[:,None]*a[None,:] + si*skew
                translation = a.new_zeros((3,1))
            elif kind == 'prismatic':
                r, translation = eye, (a*angle).reshape(3,1)
            else:
                raise ValueError(kind)
            motion = torch.cat((torch.cat((r,translation),dim=1),bottom),dim=0)
            t = t @ motion
        return t

    def _planes(self, x, q, s):
        t = self._fk64(q, s)
        p = t[:3,:3].T @ (x-t[:3,3])
        a = np.tan(np.deg2rad(FOV['horizontal_fov_deg']/2))
        b = np.tan(np.deg2rad(FOV['vertical_fov_deg']/2))
        xx, yy, zz = p.unbind()
        return torch.stack(((xx+a*zz)/np.sqrt(1+a*a),(-xx+a*zz)/np.sqrt(1+a*a),
                            (yy+b*zz)/np.sqrt(1+b*b),(-yy+b*zz)/np.sqrt(1+b*b),
                            zz-FOV['z_min'],FOV['z_max']-zz))-FOV['delta']

    def geometry(self, x, s):
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=self.device)
        def evaluate(q):
            with torch.enable_grad():
                qt = torch.tensor(np.asarray(q), dtype=torch.float64, device=self.device, requires_grad=True)
                h = self._planes(xt, qt, int(s))
                js = [torch.autograd.grad(h[i], qt, retain_graph=(i < len(h)-1))[0] for i in range(len(h))]
                return h.detach().cpu().numpy(), torch.stack(js).detach().cpu().numpy()
        return evaluate

    def reference_margins(self, x, qs):
        qs = np.asarray(qs, dtype=np.float32).reshape(-1,7)
        with torch.no_grad():
            _, margins, _, _ = self.upstream.visibility_g_batch(
                torch.tensor(np.asarray(x),dtype=torch.float32,device=self.device),
                torch.tensor(qs,dtype=torch.float32,device=self.device),self.specs,
                FOV['horizontal_fov_deg'],FOV['vertical_fov_deg'],FOV['z_min'],FOV['z_max'],FOV['delta'])
        return (margins-FOV['delta']).detach().cpu().numpy().astype(np.float64)

    def verify(self, bank, query_x_indices, query_q):
        """Numerical gates are measurement checks, not changed runtime thresholds."""
        from label_core import old_bank_label
        if np.max(np.abs(bank.lo-self.urdf_lo))>1e-6 or np.max(np.abs(bank.hi-self.urdf_hi))>1e-6:
            raise RuntimeError('Dataset joint limits disagree with the selected URDF')
        if not np.array_equal(bank.masks, self.masks):
            raise RuntimeError('Dataset sensor_chain_masks disagree with URDF chains')
        selected = list(dict.fromkeys(int(i) for i in query_x_indices))[:2]
        if not selected:
            raise RuntimeError('Empty query set')
        rng = np.random.default_rng(882614)
        qs = np.asarray(rng.uniform(bank.lo, bank.hi, (4,7)), np.float32)
        margins_error, fd_error, old_value_error, old_grad_error = 0.,0.,0.,0.
        for xi in selected:
            x = np.asarray(bank.x[xi],np.float64)
            reference = self.reference_margins(x,qs)
            for s in range(8):
                geo = self.geometry(x,s)
                for j,q in enumerate(qs):
                    h,jac = geo(q)
                    margins_error = max(margins_error,abs(float(h.min())-reference[j,s]))
                    if j == 0:
                        fd = np.zeros_like(jac)
                        for k in np.flatnonzero(self.masks[s]):
                            delta = np.zeros(7); delta[k]=1e-5
                            fd[:,k]=(geo(q.astype(float)+delta)[0]-geo(q.astype(float)-delta)[0])/(2e-5)
                        fd_error=max(fd_error,float(np.linalg.norm(fd-jac)/max(np.linalg.norm(jac),1e-8)))
            qlib=torch.tensor(np.array(bank.q[xi:xi+1]),dtype=torch.float32,device=self.device)
            valid=torch.tensor(np.array(bank.valid[xi:xi+1]),dtype=torch.bool,device=self.device)
            qt=torch.tensor(qs,dtype=torch.float32,device=self.device)
            masks=torch.tensor(bank.masks,dtype=torch.float32,device=self.device)
            with torch.no_grad():
                ds,dg,has=self.legacy.decode_per_sensor_distance_and_grad(qlib,valid,qt,masks)
                signs=torch.tensor(np.where(reference>=0,1.,-1.)[None],dtype=torch.float32,device=self.device)
                yt,gt,mt=self.per.per_sensor_signed_targets(ds,dg,signs,has)
            yt,gt,mt=yt.cpu().numpy()[0],gt.cpu().numpy()[0],mt.cpu().numpy()[0]
            for s in range(8):
                points=bank.sensor_bank(xi,s)
                for j,q in enumerate(qs):
                    result=old_bank_label(q,points,bank.masks[s],1 if reference[j,s]>=0 else -1)
                    if result['valid'] != bool(mt[j,s]):
                        raise RuntimeError('Legacy validity parity failed')
                    if result['valid']:
                        old_value_error=max(old_value_error,abs(result['value']-float(yt[j,s])))
                        old_grad_error=max(old_grad_error,float(np.max(np.abs(result['grad']-gt[j,s]))))
        report=dict(margin_max_abs_error_m=margins_error, jacobian_fd_relative_max=fd_error,
                    old_value_max_abs_error=old_value_error,old_gradient_max_abs_error=old_grad_error,
                    status='PASS',device=str(self.device),torch=torch.__version__,identity=self.identity)
        if margins_error>2e-6 or fd_error>2e-4 or old_value_error>5e-6 or old_grad_error>3e-5:
            raise RuntimeError(f'Preflight parity failed: {report}')
        return report
