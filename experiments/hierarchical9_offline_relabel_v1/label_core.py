"""Offline labels only. No neural network, ROS, or execution code.

The 'new' value is a best-found, quality-screened continuous-boundary distance,
NOT a globally certified signed distance. All units are raw active-joint radians
and normalized FOV plane margins in metres. No angular wrapping is introduced.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable
import math
import time
import numpy as np
from scipy.optimize import minimize, lsq_linear


@dataclass(frozen=True)
class SolverConfig:
    starts: int = 4               # Query itself + nearest/diverse bank seeds.
    seed_pool: int = 32
    maxiter: int = 100
    ftol: float = 1e-10
    boundary_tol_m: float = 1e-5
    feasibility_tol_m: float = 1e-6
    bound_tol_rad: float = 1e-8
    active_tol_m: float = 2e-5
    bound_active_tol_rad: float = 2e-5
    stationarity_relative_tol: float = 2e-3
    plane_gap_m: float = 1e-4
    joint_margin_rad: float = 0.002
    min_normal: float = 1e-8
    sign_guard_m: float = 1e-6
    exact_boundary_tol_m: float = 1e-10
    zero_distance_rad: float = 1e-8
    tie_abs_rad: float = 1e-4
    tie_rel: float = 1e-3
    distinct_point_rad: float = 1e-3
    normal_cosine_min: float = 0.98
    fd_step_rad: float = 1e-5
    fd_relative_tol: float = 2e-4

    def validate(self):
        if self.starts < 2 or self.seed_pool < 1 or self.maxiter < 1:
            raise ValueError('starts>=2, seed_pool>=1, maxiter>=1 required')
        for key, val in asdict(self).items():
            if not math.isfinite(float(val)) or val <= 0:
                raise ValueError(f'Invalid solver option {key}={val}')
        if not 0 < self.normal_cosine_min <= 1:
            raise ValueError('normal_cosine_min must be in (0,1]')


def old_bank_label(q, bank, mask, sign):
    """Match the upstream FP32 nearest-bank arithmetic, including its 1e-8 floor.

    bank MUST contain only valid finite rows for this x/sensor, in stored order.
    old_valid follows the legacy finite-label rule. old_grad_regular is a
    separate diagnostic; it does not silently rewrite legacy validity.
    """
    q = np.asarray(q, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if len(bank) == 0:
        return dict(value=np.nan, grad=np.full(q.shape, np.nan), valid=False,
                    grad_regular=False, distance=np.nan, nearest=-1, gap=np.nan)
    if not np.isfinite(bank).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite valid bank/query')
    delta = (q[None] - bank) * mask[None]
    d2 = np.sum(delta * delta, axis=1, dtype=np.float32)
    k = int(np.argmin(d2))
    dist = np.sqrt(np.maximum(d2[k], np.float32(1e-8)))
    exact = float(np.sqrt(np.maximum(d2[k], np.float32(0))))
    distances = np.sqrt(np.maximum(d2, np.float32(0)))
    ordered = np.sort(distances, kind='stable')
    gap = float(ordered[1] - ordered[0]) if len(ordered) > 1 else math.inf
    grad = np.float32(sign) * delta[k] / dist
    return dict(value=float(np.float32(sign) * dist), grad=grad.astype(np.float64),
                valid=True, grad_regular=bool(exact > 1e-4 and gap > 1e-6),
                distance=exact, nearest=k, gap=gap)


class _MemoGeometry:
    def __init__(self, geometry: Callable, base, active):
        self.geometry, self.base, self.active = geometry, base.copy(), active
        self.last, self.h, self.jac = None, None, None
        self.calls = 0

    def full_q(self, z):
        q = self.base.copy()
        q[self.active] = z
        return q

    def evaluate(self, z):
        z = np.asarray(z, dtype=np.float64)
        if self.last is None or not np.array_equal(z, self.last):
            h, jac = self.geometry(self.full_q(z))
            h, jac = np.asarray(h, dtype=np.float64), np.asarray(jac, dtype=np.float64)
            if h.ndim != 1 or jac.shape != (len(h), len(self.base)):
                raise ValueError('geometry must return margins [F] and Jacobian [F,J]')
            if not np.isfinite(h).all() or not np.isfinite(jac).all():
                raise FloatingPointError('Nonfinite analytic geometry')
            self.last, self.h, self.jac = z.copy(), h, jac[:, self.active]
            self.calls += 1
        return self.h, self.jac


def _seeds(q, bank, active, lo, hi, cfg):
    rows = [q[active].copy()]
    bank = np.asarray(bank, dtype=np.float64)
    if len(bank):
        z = bank[:, active]
        ok = np.isfinite(z).all(1) & (z >= lo).all(1) & (z <= hi).all(1)
        z = z[ok]
        if len(z):
            order = np.argsort(np.linalg.norm(z - q[active], axis=1), kind='stable')
            pool = z[order[:cfg.seed_pool]]
            rows.append(pool[0].copy())
            while len(rows) < cfg.starts:
                separation = np.min(np.stack([np.linalg.norm(pool - r, axis=1) for r in rows]), axis=0)
                i = int(separation.argmax())
                if separation[i] <= 1e-8:
                    break
                rows.append(pool[i].copy())
    # Avoid identical starts; the query is not evidence that it is on a boundary.
    unique = []
    for row in rows:
        if not any(np.linalg.norm(row-u) < 1e-9 for u in unique):
            unique.append(row)
    return unique


def _kkt_residual(z, query, margins, jac, plane, lo, hi, cfg):
    """Necessary first-order constrained optimality check; NOT sufficiency/globality.

    h_plane=0 has a free multiplier. h_other>=0 and bounds have nonnegative
    multipliers with the appropriate outward normals.
    """
    a = [jac[plane]]
    lower, upper = [-np.inf], [np.inf]
    for j in range(len(margins)):
        if j != plane and margins[j] <= cfg.active_tol_m:
            a.append(-jac[j]); lower.append(0.); upper.append(np.inf)
    for k in range(len(z)):
        e = np.eye(len(z))[k]
        if z[k] - lo[k] <= cfg.bound_active_tol_rad:
            a.append(-e); lower.append(0.); upper.append(np.inf)
        if hi[k] - z[k] <= cfg.bound_active_tol_rad:
            a.append(e); lower.append(0.); upper.append(np.inf)
    mat = np.stack(a, axis=1)
    # Drop numerically zero normals, otherwise degenerate constraints can look valid.
    keep = np.linalg.norm(mat, axis=0) > cfg.min_normal
    if not keep[0]:
        return math.inf
    try:
        sol = lsq_linear(mat[:, keep], -(z-query),
                         bounds=(np.asarray(lower)[keep], np.asarray(upper)[keep]),
                         tol=1e-12, max_iter=200)
        return float(np.linalg.norm(mat[:, keep] @ sol.x + z-query) /
                     max(np.linalg.norm(z-query), 1e-6))
    except (ValueError, np.linalg.LinAlgError):
        return math.inf


def label_continuous(q, bank, mask, q_min, q_max, geometry: Callable,
                     cfg: SolverConfig | None = None):
    """Enumerate EVERY FOV face with all other margins constrained nonnegative.

    geometry(q_full) -> (margins[F], d_margins/d_q[F,J]), in float64.
    Only active joints move; inactive joints remain exactly equal to query q.
    Invalid/uncertain labels remain NaN or are accompanied by false validity.
    """
    cfg = cfg or SolverConfig()
    cfg.validate()
    begin = time.perf_counter()
    q = np.asarray(q, dtype=np.float64)
    mask = np.asarray(mask)
    q_min, q_max = np.asarray(q_min, float), np.asarray(q_max, float)
    if q.ndim != 1 or mask.shape != q.shape or q_min.shape != q.shape or q_max.shape != q.shape:
        raise ValueError('q/mask/limits shape mismatch')
    if not np.isfinite(q).all() or np.any(q < q_min) or np.any(q > q_max):
        raise ValueError('Query outside limits or nonfinite; no silent clamping')
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Only the existing binary active-chain metric is supported')
    active = np.flatnonzero(mask)
    if len(active) == 0:
        raise ValueError('Empty active chain')
    lo, hi, zq = q_min[active], q_max[active], q[active]
    memo = _MemoGeometry(geometry, q, active)
    hq, _ = memo.evaluate(zq)
    query_g = float(hq.min())
    sign = 1. if query_g >= 0 else -1.
    seeds = _seeds(q, bank, active, lo, hi, cfg)
    attempts, feasible = [], []
    for plane in range(len(hq)):
        other = np.arange(len(hq)) != plane
        constraints = [dict(type='eq',
                            fun=lambda z, p=plane: memo.evaluate(z)[0][p],
                            jac=lambda z, p=plane: memo.evaluate(z)[1][p])]
        if other.any():
            constraints.append(dict(type='ineq',
                                    fun=lambda z, keep=other: memo.evaluate(z)[0][keep],
                                    jac=lambda z, keep=other: memo.evaluate(z)[1][keep]))
        for seed_id, seed in enumerate(seeds):
            try:
                result = minimize(lambda z: 0.5 * np.sum((z-zq)**2), seed,
                                  jac=lambda z: z-zq, method='SLSQP',
                                  bounds=list(zip(lo, hi)), constraints=constraints,
                                  options=dict(maxiter=cfg.maxiter, ftol=cfg.ftol, disp=False))
                z = np.asarray(result.x, float)
                h, jac = memo.evaluate(z)
                feasible_here = bool(abs(h[plane]) <= cfg.boundary_tol_m
                                     and h.min() >= -cfg.feasibility_tol_m
                                     and np.all(z >= lo-cfg.bound_tol_rad)
                                     and np.all(z <= hi+cfg.bound_tol_rad))
                kkt = _kkt_residual(z, zq, h, jac, plane, lo, hi, cfg) if feasible_here else math.inf
                row = dict(plane=plane, seed_id=seed_id, q_seed=memo.full_q(seed).tolist(), success=bool(result.success),
                           optimizer_status=int(result.status), message=str(result.message),
                           iterations=int(result.nit), feasible=feasible_here,
                           distance_rad=float(np.linalg.norm(z-zq)),
                           min_margin_m=float(h.min()), equality_residual_m=float(abs(h[plane])),
                           stationarity_relative=kkt, q_star=memo.full_q(z).tolist())
                attempts.append(row)
                if feasible_here:
                    feasible.append((row, z.copy(), h.copy(), jac.copy()))
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                attempts.append(dict(plane=plane, seed_id=seed_id, success=False,
                                     feasible=False, optimizer_status=-99, message=repr(exc)))

    out = dict(value=math.nan, grad=np.full(len(q), np.nan), q_star=np.full(len(q), np.nan),
               value_valid=False, grad_valid=False, status='NO_FEASIBLE_BOUNDARY_FOUND',
               query_g_m=query_g, distance_rad=math.nan, boundary_residual_m=math.nan,
               stationarity_relative=math.nan, normal_cosine=math.nan,
               fd_relative_error=math.nan, plane_gap_m=math.nan, joint_margin_rad=math.nan,
               ambiguity=False, best_plane=-1, optimizer_success=False,
               starts_used=len(seeds), attempts=attempts, global_nearest_certified=False)
    if feasible:
        # Do not discard a closer low-quality solution and quietly call a farther one 'nearest'.
        feasible.sort(key=lambda a: (a[0]['distance_rad'], not a[0]['success']))
        row, z, h, jac = feasible[0]
        d = row['distance_rad']
        # If a numerically identical candidate converged, prefer its diagnostics.
        for item in feasible[1:]:
            if item[0]['distance_rad'] > d + 1e-8:
                break
            if item[0]['success'] and np.linalg.norm(item[1]-z) <= 1e-7:
                row, z, h, jac = item
                d = row['distance_rad']; break
        best_plane = int(np.argmin(h))
        normal = jac[best_plane]
        normal_norm = float(np.linalg.norm(normal))
        gap = float(np.sort(h)[1] - np.sort(h)[0]) if len(h) > 1 else math.inf
        margin = float(np.minimum(z-lo, hi-z).min())
        near_equal = cfg.tie_abs_rad + cfg.tie_rel*d
        ambiguous = any(abs(item[0]['distance_rad']-d) <= near_equal
                        and np.linalg.norm(item[1]-z) > cfg.distinct_point_rad for item in feasible[1:])
        on_boundary = abs(query_g) <= cfg.exact_boundary_tol_m and d <= cfg.zero_distance_rad
        sign_ok = abs(query_g) > cfg.sign_guard_m or on_boundary
        value_ok = bool(row['success'] and row['stationarity_relative'] <= cfg.stationarity_relative_tol and sign_ok)
        value = 0. if on_boundary else sign*d
        grad = np.full(len(q), np.nan)
        alignment, fd_err = math.nan, math.nan
        regular = gap > cfg.plane_gap_m and margin > cfg.joint_margin_rad and normal_norm > cfg.min_normal
        if regular:
            normal = normal / normal_norm
            direction = normal if on_boundary else (sign*(zq-z)/d if d > cfg.zero_distance_rad else None)
            if direction is not None:
                alignment = float(np.dot(direction, normal))
                fd = []
                for j in range(len(z)):
                    dz = np.zeros(len(z)); dz[j] = cfg.fd_step_rad
                    hp, _ = memo.evaluate(z+dz)
                    hm, _ = memo.evaluate(z-dz)
                    fd.append((hp[best_plane]-hm[best_plane])/(2*cfg.fd_step_rad))
                fd = np.asarray(fd)
                fd_err = float(np.linalg.norm(fd-jac[best_plane])/max(normal_norm, cfg.min_normal))
                if value_ok and not ambiguous and alignment >= cfg.normal_cosine_min and fd_err <= cfg.fd_relative_tol:
                    grad = np.zeros(len(q)); grad[active] = direction
        grad_ok = bool(np.isfinite(grad).all())
        out.update(value=value, grad=grad, q_star=memo.full_q(z), value_valid=value_ok,
                   grad_valid=grad_ok, status=('APPROX_VALUE_AND_GRAD' if grad_ok else
                                               'APPROX_VALUE_ONLY' if value_ok else 'FEASIBLE_LOW_CONFIDENCE'),
                   distance_rad=d, boundary_residual_m=float(abs(h.min())),
                   stationarity_relative=row['stationarity_relative'], normal_cosine=alignment,
                   fd_relative_error=fd_err, plane_gap_m=gap, joint_margin_rad=margin,
                   ambiguity=ambiguous, best_plane=best_plane, optimizer_success=row['success'])
    out.update(geometry_calls=memo.calls, elapsed_ms=1000*(time.perf_counter()-begin))
    return out
