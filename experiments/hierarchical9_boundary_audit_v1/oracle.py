"""Differentiable per-sensor FOV. Reuse upstream FK, verify margins against upstream oracle."""
from __future__ import annotations
import math
import torch


class SensorOracle:
    def __init__(self, urdf, device, joint_names, sensor_frames):
        from urdf_parser_py.urdf import URDF
        from extract_visibility_zero_level_sets import prepare_chain_specs, fk_sensor_batch, visibility_g_batch
        self.specs = prepare_chain_specs(URDF.from_xml_file(str(urdf)), "base_link",
                                        sensor_frames, joint_names, device)
        self.fk, self.reference = fk_sensor_batch, visibility_g_batch
        self.hfov, self.vfov, self.zmin, self.zmax, self.delta = 50., 66., .2, .7, .01

    def planes(self, x, q, s):
        transform = self.fk(self.specs[s], q.reshape(-1, 7))
        diff = x.reshape(1, 3) - transform[:, :3, 3]
        p = torch.einsum("nji,nj->ni", transform[:, :3, :3], diff)
        px, py, pz = p.unbind(-1)
        ax, ay = math.tan(math.radians(self.hfov)/2), math.tan(math.radians(self.vfov)/2)
        nx, ny = math.sqrt(1+ax*ax), math.sqrt(1+ay*ay)
        # Match normalized plane margins and subtract delta ONCE.
        return torch.stack([(px+pz*ax)/nx, (-px+pz*ax)/nx,
                            (py+pz*ay)/ny, (-py+pz*ay)/ny,
                            pz-self.zmin, self.zmax-pz], -1) - self.delta

    def value(self, x, q, s):
        with torch.no_grad():
            return float(self.planes(x, q.reshape(1, 7), s)[0].min())

    def value_grad(self, x, q, s):
        with torch.enable_grad(), torch.autocast(q.device.type, enabled=False):
            qv = q.detach().float().reshape(1, 7).clone().requires_grad_(True)
            g = self.planes(x.float(), qv, s)[0].min(dim=-1).values
            grad = torch.autograd.grad(g, qv)[0][0]
        return float(g.detach()), grad.detach()

    def verify_against_upstream(self, x, qs):
        with torch.no_grad():
            _, margins, _, planes = self.reference(x, qs, self.specs, self.hfov,
                                                   self.vfov, self.zmin, self.zmax, self.delta)
            values = torch.stack([self.planes(x, qs, s).min(-1).values for s in range(8)], -1)
            idx = torch.stack([self.planes(x, qs, s).argmin(-1) for s in range(8)], -1)
            error = float((values - (margins-self.delta)).abs().max())
            if error > 2e-6 or not torch.equal(idx, planes):
                raise RuntimeError(f"FOV semantics mismatch: error={error}")
            return {"status": "PASS", "max_abs_error_m": error, "active_planes_equal": True}
