"""Per-sensor approximate distance labels with explicit, fail-closed verification.

v1 stays immutable. Six faces are internal constraints, NOT six datasets/heads.
KKT + finite differences are numerical evidence, NEVER a global projection proof.
All tolerances below are labeling diagnostics, not execution/FOV safety settings.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize

V1 = Path(__file__).resolve().parents[1] / 'hierarchical9_offline_relabel_v1'
sys.path.insert(0, str(V1))
import label_core as legacy
if Path(legacy.__file__).resolve() != V1 / 'label_core.py':
    raise ImportError('Wrong v1 label_core import')
SolverConfig = legacy.SolverConfig


@dataclass(frozen=True)
class VerifyConfig:
    equivalent_point_rad: float = 2e-7
    equivalent_distance_rad: float = 2e-7
    equivalent_distance_relative: float = 1e-9
    fd_step_rad: float = 2e-4
    fd_min_step_rad: float = 2e-6
    fd_absolute_tol: float = 0.02
    fd_relative_tol: float = 0.02
    probe_starts: int = 4
    probe_maxiter: int = 100

    def validate(self):
        for k, v in asdict(self).items():
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f'Invalid verification option {k}={v}')
        if self.probe_starts < 2 or self.fd_min_step_rad >= self.fd_step_rad:
            raise ValueError('Need >=2 probe starts and min step < nominal step')


def select_equivalent(candidates, cfg: SolverConfig, vc: VerifyConfig):
    """Group against fixed representatives (no transitive clustering drift).

    Input: rechecked feasible candidates with distance_rad/q_star/qualified.
    Never replace a genuinely closer uncertain cluster with a farther good one.
    Unconverged distinct candidates are uncertainty, not asserted true ambiguity.
    """
    groups = []
    for c in sorted(candidates, key=lambda c: (c['distance_rad'], c['attempt_id'])):
        for group in groups:
            anchor = group[0]
            tol = vc.equivalent_distance_rad + vc.equivalent_distance_relative * anchor['distance_rad']
            if (abs(c['distance_rad'] - anchor['distance_rad']) <= tol and
                    np.linalg.norm(np.asarray(c['q_star']) - anchor['q_star']) <= vc.equivalent_point_rad):
                group.append(c)
                break
        else:
            groups.append([c])
    if not groups:
        return None, [], False, False
    representatives = [min(g, key=lambda c: (not c['qualified'],
                          c['stationarity_relative'], c['equality_residual_m'], c['distance_rad']))
                       for g in groups]
    best = representatives[0]  # groups ordered by their closest member
    ambiguity = uncertain = False
    tie = cfg.tie_abs_rad + cfg.tie_rel * best['distance_rad']
    for c in representatives[1:]:
        if (abs(c['distance_rad'] - best['distance_rad']) <= tie and
                np.linalg.norm(np.asarray(c['q_star']) - best['q_star']) > cfg.distinct_point_rad):
            if c['qualified']:
                ambiguity = True
            else:
                uncertain = True
    return best, representatives, ambiguity, uncertain


def screen_candidates(q, mask, lo, hi, geometry, attempts, cfg=None, vc=None):
    """Recompute endpoint feasibility/KKT; never trust saved 'feasible' flags."""
    cfg, vc = cfg or SolverConfig(), vc or VerifyConfig()
    cfg.validate(); vc.validate()
    q, mask, lo, hi = (np.asarray(v, float) for v in (q, mask, lo, hi))
    if q.ndim != 1 or any(v.shape != q.shape for v in (mask, lo, hi)):
        raise ValueError('q/mask/limit shapes')
    if (not all(np.isfinite(v).all() for v in (q, mask, lo, hi)) or
            not np.isin(mask, [0, 1]).all() or not (lo < hi).all() or
            np.any(q < lo) or np.any(q > hi)):
        raise ValueError('Invalid query/mask/limits; no clamping')
    active = np.flatnonzero(mask)
    if not len(active):
        raise ValueError('Empty active chain')
    hq, jq = geometry(q)
    hq, jq = np.asarray(hq, float), np.asarray(jq, float)
    if hq.ndim != 1 or jq.shape != (len(hq), len(q)) or not np.isfinite(hq).all() or not np.isfinite(jq).all():
        raise ValueError('Nonfinite or malformed geometry')
    g = float(hq.min())
    checked, rejected = [], []
    for aid, attempt in enumerate(attempts):
        if 'q_star' not in attempt:
            rejected.append(dict(attempt_id=aid, reason='NO_ENDPOINT')); continue
        p = np.asarray(attempt['q_star'], float)
        plane = int(attempt.get('plane', -1))
        if (p.shape != q.shape or not np.isfinite(p).all() or not 0 <= plane < len(hq) or
                np.any(np.abs((p-q)[mask == 0]) > 1e-10)):
            rejected.append(dict(attempt_id=aid, reason='INVALID_ENDPOINT_OR_INACTIVE_JOINT')); continue
        h, j = geometry(p); h, j = np.asarray(h, float), np.asarray(j, float)
        if h.shape != hq.shape or j.shape != jq.shape or not np.isfinite(h).all() or not np.isfinite(j).all():
            raise ValueError('Malformed endpoint geometry')
        feasible = (abs(h[plane]) <= cfg.boundary_tol_m and h.min() >= -cfg.feasibility_tol_m and
                    np.all(p >= lo-cfg.bound_tol_rad) and np.all(p <= hi+cfg.bound_tol_rad))
        if not feasible:
            rejected.append(dict(attempt_id=aid, reason='ENDPOINT_NOT_FEASIBLE')); continue
        stationarity = legacy._kkt_residual(p[active], q[active], h, j[:, active], plane,
                                           lo[active], hi[active], cfg)
        qualified = bool(attempt.get('success', False) and stationarity <= cfg.stationarity_relative_tol)
        checked.append(dict(attempt_id=aid, plane=plane, q_star=p.tolist(),
            distance_rad=float(np.linalg.norm((q-p)[active])), qualified=qualified,
            optimizer_success=bool(attempt.get('success', False)),
            equality_residual_m=float(abs(h[plane])), min_margin_m=float(h.min()),
            stationarity_relative=float(stationarity),
            active_faces=np.flatnonzero(np.abs(h) <= cfg.active_tol_m).tolist(),
            lower_bound_joints=active[(p-lo)[active] <= cfg.bound_active_tol_rad].tolist(),
            upper_bound_joints=active[(hi-p)[active] <= cfg.bound_active_tol_rad].tolist()))
    best, representatives, ambiguity, uncertain = select_equivalent(checked, cfg, vc)
    out = dict(value=np.nan, grad=np.full(len(q), np.nan), gradient_candidate=np.full(len(q), np.nan),
        q_star=np.full(len(q), np.nan), value_valid=False, grad_valid=False,
        query_g_m=g, distance_rad=np.nan, selected=best, clusters=representatives,
        checked_candidates=checked, rejected_candidates=rejected, ambiguity=ambiguity,
        uncertain_competitor=uncertain, gradient_reasons=[], verification=None,
        global_nearest_certified=False, method='best_found_continuous_boundary_v2')
    reasons = out['gradient_reasons']
    if best is None:
        reasons.append('NO_FEASIBLE_BOUNDARY_FOUND'); return out
    p = np.asarray(best['q_star']); d = best['distance_rad']
    on_boundary = abs(g) <= cfg.exact_boundary_tol_m and d <= cfg.zero_distance_rad
    sign_ok = abs(g) > cfg.sign_guard_m or on_boundary
    out.update(q_star=p, distance_rad=d, value=0. if on_boundary else (1 if g >= 0 else -1)*d,
               value_valid=bool(best['qualified'] and sign_ok))
    if not best['qualified']: reasons.append('BEST_CLUSTER_NOT_CONVERGED_OR_STATIONARY')
    if not sign_ok: reasons.append('QUERY_SIGN_UNRESOLVED')
    if ambiguity: reasons.append('DISTINCT_QUALIFIED_NEAR_EQUAL_PROJECTIONS')
    if uncertain: reasons.append('DISTINCT_UNCERTAIN_NEAR_EQUAL_CANDIDATE')
    if on_boundary:
        # At distance zero displacement/d is undefined. Do not invent corner normals.
        order = np.argsort(hq)
        gap = hq[order[1]] - hq[order[0]] if len(hq)>1 else np.inf
        n = jq[order[0]] * mask
        if (gap <= cfg.plane_gap_m or np.linalg.norm(n) <= cfg.min_normal or
                best['lower_bound_joints'] or best['upper_bound_joints']):
            reasons.append('ZERO_DISTANCE_NONREGULAR_BOUNDARY')
        else:
            out['gradient_candidate'] = n / np.linalg.norm(n)
    elif d > cfg.zero_distance_rad:
        out['gradient_candidate'] = (1 if g >= 0 else -1)*(q-p)*mask/d
    else:
        reasons.append('DISTANCE_TOO_SMALL')
    # No automatic single-face/joint-margin gate for NONZERO distances.
    if not reasons: reasons.append('DISTANCE_DERIVATIVE_NOT_VERIFIED')
    return out


def solve_attempts(q, bank, mask, lo, hi, geometry, cfg=None, warm=()):
    """Full per-sensor FOV constraints; local SLSQP, no globality claim.

    Additional warm seeds are hints. Query and original bank seeds are retained.
    Every trial logs elapsed time and geometry-call count, including failures.
    """
    cfg = cfg or SolverConfig(); cfg.validate()
    q, mask, lo, hi = (np.asarray(v, float) for v in (q, mask, lo, hi))
    if not np.isin(mask, [0, 1]).all() or np.any(q<lo) or np.any(q>hi):
        raise ValueError('Bad query/mask; no silent clamping')
    active = np.flatnonzero(mask)
    memo = legacy._MemoGeometry(geometry, q, active)
    seeds = legacy._seeds(q, np.asarray(bank, float).reshape(-1, len(q)), active,
                          lo[active], hi[active], cfg)
    for p in warm:
        p = np.asarray(p, float)[active]
        if np.all(p >= lo[active]) and np.all(p <= hi[active]) and not any(np.linalg.norm(p-v)<1e-8 for v in seeds):
            seeds.append(p.copy())
    h, _ = memo.evaluate(q[active]); attempts = []
    for plane in range(len(h)):
        other = np.arange(len(h)) != plane
        cons = [dict(type='eq', fun=lambda z,p=plane: memo.evaluate(z)[0][p],
                     jac=lambda z,p=plane: memo.evaluate(z)[1][p])]
        if other.any():
            cons.append(dict(type='ineq', fun=lambda z,k=other: memo.evaluate(z)[0][k],
                             jac=lambda z,k=other: memo.evaluate(z)[1][k]))
        for sid, seed in enumerate(seeds):
            t0, calls = time.perf_counter(), memo.calls
            try:
                r = minimize(lambda z: .5*np.sum((z-q[active])**2), seed,
                    jac=lambda z: z-q[active], method='SLSQP',
                    constraints=cons, bounds=list(zip(lo[active], hi[active])),
                    options=dict(ftol=cfg.ftol, maxiter=cfg.maxiter, disp=False))
                row = dict(plane=plane, seed_id=sid, q_seed=memo.full_q(seed).tolist(),
                    q_star=memo.full_q(r.x).tolist(), success=bool(r.success),
                    optimizer_status=int(r.status), message=str(r.message), iterations=int(r.nit))
            except (FloatingPointError, np.linalg.LinAlgError) as exc:
                row = dict(plane=plane, seed_id=sid, success=False, message=repr(exc))
            row.update(elapsed_ms=1000*(time.perf_counter()-t0), geometry_calls=memo.calls-calls)
            attempts.append(row)
    return attempts


def verify_gradient(base, q, mask, lo, hi, solve_probe, cfg=None, vc=None, progress=None):
    """Two-sided, TWO-SCALE finite differences of reoptimized SIGNED distances.

    solve_probe must run a new constrained solve and return screen_candidates.
    All active coordinate axes + two deterministic mixed directions are tested.
    One-sided slopes are checked too, avoiding symmetric-FD false positives at cusps.
    Failure/uncertainty => masked gradient. This verifies a local numerical label,
    not an unseen global projection and not execution feasibility.
    """
    cfg, vc = cfg or SolverConfig(), vc or VerifyConfig(); vc.validate()
    out = dict(base); out['grad'] = np.full(len(q), np.nan); out['grad_valid'] = False
    reasons = [s for s in base['gradient_reasons'] if s != 'DISTANCE_DERIVATIVE_NOT_VERIFIED']
    out['gradient_reasons'] = reasons
    if reasons:
        out['verification'] = dict(status='SKIPPED_BASE_GATE', probes=[]); return out
    q, mask, lo, hi = (np.asarray(v, float) for v in (q, mask, lo, hi))
    active = np.flatnonzero(mask); candidate = np.asarray(base['gradient_candidate'])
    margin = float(np.minimum(q-lo, hi-q)[active].min())
    d = float(base['distance_rad'])
    h = min(vc.fd_step_rad, .2*margin)
    if d > cfg.zero_distance_rad: h = min(h, .05*d)
    if h/2 < vc.fd_min_step_rad or not np.isfinite(candidate).all():
        reasons.append('QUERY_TOO_CLOSE_TO_BOUNDARY_OR_LIMIT_FOR_TWO_SIDED_FD')
        out['verification'] = dict(status='NOT_TESTABLE', probes=[], step_rad=h); return out
    dirs = [np.eye(len(q))[j] for j in active]
    for a in (np.ones(len(active)), np.where(np.arange(len(active))%2, -1., 1.)):
        v = np.zeros(len(q)); v[active] = a/np.linalg.norm(a)
        if not any(abs(np.dot(v,w)) > 1-1e-12 for w in dirs): dirs.append(v)
    probes, slopes, errors = [], {}, []
    tol_closer = 2*(vc.equivalent_distance_rad + vc.equivalent_distance_relative*d)
    def fail(reason):
        if reason not in reasons: reasons.append(reason)
    for direction_id, direction in enumerate(dirs):
        expected = float(candidate @ direction)
        allowed = vc.fd_absolute_tol + vc.fd_relative_tol*abs(expected)
        for scale in (1., .5):
            step = h*scale; values = {}
            for side in (-1, 1):
                trial = q+side*step*direction
                if progress: progress(f'direction={direction_id+1}/{len(dirs)} scale={scale} side={side:+d}')
                r = solve_probe(trial)
                good = bool(r['value_valid'] and not r['ambiguity'] and not r['uncertain_competitor'])
                sign_ok = base['value']==0 or np.sign(r['value'])==np.sign(base['value'])
                entry = dict(direction=direction_id, scale=scale, side=side, q=trial.tolist(),
                    value=r['value'], value_valid=r['value_valid'], ambiguity=r['ambiguity'],
                    uncertain_competitor=r['uncertain_competitor'], q_star=r['q_star'],
                    selected=r['selected'], attempts=r.get('attempts', []))
                probes.append(entry)
                if np.isfinite(r['q_star']).all():
                    center_distance = float(np.linalg.norm((q-r['q_star'])[active]))
                    if center_distance < d-tol_closer:
                        fail('CLOSER_BASE_CANDIDATE_DISCOVERED'); out['value_valid'] = False
                if not good or not sign_ok:
                    fail('PERTURBED_SOLVE_UNRELIABLE_OR_SIGN_CHANGED')
                else:
                    values[side] = float(r['value'])
            if len(values)==2:
                center = float(base['value'])
                central = (values[1]-values[-1])/(2*step)
                left, right = (center-values[-1])/step, (values[1]-center)/step
                err = max(abs(central-expected), abs(left-expected), abs(right-expected))
                errors.append(err); slopes[direction_id,scale] = central
                probes[-1]['difference_check'] = dict(expected=expected, central=central,
                    left=left, right=right, max_error=err, tolerance=allowed)
                if err > allowed: fail('DISTANCE_FD_MISMATCH')
            if reasons: break  # Fail closed, avoid spending all probes on an invalid base.
        if reasons: break
        if abs(slopes[direction_id,1.] - slopes[direction_id,.5]) > allowed:
            fail('DISTANCE_FD_SCALE_UNSTABLE'); break
    expected_count = 4*len(dirs)
    passed = not reasons and len(probes)==expected_count
    if passed: out.update(grad=candidate.copy(), grad_valid=True)
    out['verification'] = dict(status='PASS' if passed else 'HOLD', step_rad=h,
        directions=len(dirs), expected_probes=expected_count, completed_probes=len(probes),
        max_error=max(errors) if errors else None, probes=probes)
    return out


def label_verified(q, bank, mask, lo, hi, geometry, cfg=None, vc=None,
                   attempts=None, progress=None):
    """Use supplied original attempts for central query (no central re-solve)."""
    from dataclasses import replace
    cfg, vc = cfg or SolverConfig(), vc or VerifyConfig()
    reused = attempts is not None
    attempts = solve_attempts(q, bank, mask, lo, hi, geometry, cfg) if attempts is None else attempts
    base = screen_candidates(q, mask, lo, hi, geometry, attempts, cfg, vc)
    warm = [c['q_star'] for c in base['clusters'][:2] if c['qualified']]
    probe_cfg = replace(cfg, starts=vc.probe_starts, maxiter=vc.probe_maxiter)
    def probe(trial):
        a = solve_attempts(trial, bank, mask, lo, hi, geometry, probe_cfg, warm)
        r = screen_candidates(trial, mask, lo, hi, geometry, a, cfg, vc)
        r['attempts'] = a
        return r
    result = verify_gradient(base, q, mask, lo, hi, probe, cfg, vc, progress)
    result['central_attempts_reused'] = reused
    return result
