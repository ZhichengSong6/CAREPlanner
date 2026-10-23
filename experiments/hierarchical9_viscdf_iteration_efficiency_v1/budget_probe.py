"""Budget the EXISTING runtime; do not copy/reimplement its branching solver.

K is the number of branch_value_and_grad calls allowed, not solver invocations.
Value-only root refinement is allowed and counted separately. On exhaustion,
return the configuration at which the next gradient was requested. No oracle
is visible to the solver, and no analytic best-iterate selection is performed.
"""
from __future__ import annotations
import math
import time
import torch


class BudgetExhausted(Exception):
    def __init__(self, q):
        super().__init__('next gradient query would exceed the fixed budget')
        self.q = q.detach().clone()


def instrument(probe):
    """Wrap a fresh upstream Probe without changing its optimizer methods."""
    base = type(probe)
    class BudgetProbe(base):
        def reset_budget(self, limit):
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError('K must be a positive integer, or None for legacy FULL')
            self.limit = limit
            self.value_calls = self.gradient_calls = 0
            self.projection_calls = self.ascent_calls = 0
            self.noop_calls = self.root_value_calls = self.root_depth = 0
            self.visits = []

        def branch_value_and_grad(self, points, q, sensor_id):
            if self.limit is not None and self.gradient_calls >= self.limit:
                raise BudgetExhausted(q)
            return super().branch_value_and_grad(points, q, sensor_id)

        def branch_score(self, points, q, sensor_id):
            if self.root_depth:
                self.root_value_calls += 1
            return super().branch_score(points, q, sensor_id)

        def record(self, kind, q, degenerate=False):
            self.visits.append((kind, self.gradient_calls, q.detach().clone(), bool(degenerate)))

        def _projection_step(self, q, score, grad):
            out, diag = super()._projection_step(q, score, grad)
            self.projection_calls += 1
            degenerate = bool(diag.get('degenerate', False))
            self.noop_calls += int(degenerate or torch.equal(out, q))
            self.record('projection', out, degenerate)
            return out, diag

        def _ascent_step(self, q, grad):
            out, diag = super()._ascent_step(q, grad)
            self.ascent_calls += 1
            degenerate = bool(diag.get('degenerate', False))
            self.noop_calls += int(degenerate or torch.equal(out, q))
            self.record('ascent', out, degenerate)
            return out, diag

        def _refine_branch_root(self, *args, **kwargs):
            self.root_depth += 1
            try:
                out = super()._refine_branch_root(*args, **kwargs)
            finally:
                self.root_depth -= 1
            # Only the returned root is an accepted state. Do not promote every
            # bisection probe into an accepted trajectory configuration.
            self.record('refined_root', out[0])
            return out

    # The upstream factory builds a dynamic class with a no-argument constructor.
    wrapped = BudgetProbe()
    wrapped.reset_budget(None)
    return wrapped


def execute(probe, x, q, sensor, budget):
    """Only learned solver work is timed. Analytic checks occur AFTER return."""
    probe.reset_budget(budget)
    device = q.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    full = None
    try:
        full = probe._optimize_branch(x.reshape(1, 3), q.reshape(1, 7).clone(), int(sensor))
        candidate = torch.tensor(full['q_candidate'], device=device, dtype=q.dtype).reshape(1, 7)
        termination = 'legacy_completed'
    except BudgetExhausted as stop:
        candidate = stop.q.reshape(1, 7)
        termination = 'gradient_budget_exhausted'
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    ms = 1000. * (time.perf_counter() - start)
    if not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite returned configuration; do not drop the case')
    if probe.projection_calls + probe.ascent_calls != probe.gradient_calls:
        raise RuntimeError('Upstream gradient/update contract changed')
    if budget is not None and probe.gradient_calls > budget:
        raise RuntimeError('Budget overrun')
    trace = [{'kind': kind, 'gradient_calls': count,
              'q': v.reshape(-1).cpu().tolist(), 'degenerate': bad}
             for kind, count, v, bad in probe.visits]
    result = dict(budget=budget, termination=termination,
                  q_candidate=candidate.reshape(-1).cpu().tolist(),
                  solver_ms=ms, gradient_calls=probe.gradient_calls,
                  value_calls=probe.value_calls, root_value_calls=probe.root_value_calls,
                  projection_calls=probe.projection_calls, ascent_calls=probe.ascent_calls,
                  noop_calls=probe.noop_calls,
                  root_source=full.get('root_source') if full is not None else None,
                  solution_mode=full.get('solution_mode') if full is not None else None)
    return result, trace


def geometric_result(q, margin, lo, hi):
    if len(q) != 7 or not all(math.isfinite(float(v)) for v in q) or not math.isfinite(float(margin)):
        raise ValueError('Invalid candidate geometry')
    inside = all(float(a) <= float(v) <= float(b) for v, a, b in zip(q, lo, hi))
    return dict(candidate_g_m=float(margin), candidate_in_limits=inside,
                fov_pass=bool(inside and margin >= 0),
                violation_m=max(0., -float(margin)))


def first_hit(trace, margins, lo, hi):
    """DESCRIPTIVE full-run first accepted-state hit; NOT an online selector."""
    if len(trace) != len(margins):
        raise ValueError('Trace/margin length mismatch')
    annotated = []
    hit = None
    for record, margin in zip(trace, margins):
        r = dict(record, **geometric_result(record['q'], margin, lo, hi))
        annotated.append(r)
        if hit is None and r['fov_pass']:
            hit = r['gradient_calls']
    return hit, annotated
