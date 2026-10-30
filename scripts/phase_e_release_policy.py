"""Predeclared finite-fixture acceptance, not a hard-real-time guarantee."""
import math
from compare_phase_e_unified_primitives import compare

POLICY = dict(name='mainline21_relaxed_v1', trajectory_states=21,
              absolute_regression_ms=1., relative_regression=.10, p95_ceiling_ms=50.)


def compare_release(old, new):
    for report in (old, new):
        assert all(math.isfinite(value[q]) and value[q] > 0
                   for value in report['latency_ms'].values() for q in ('p50', 'p95')), 'invalid timing'
    # Reuse all safety, provenance, scheduling and workload comparison guards.
    strict = compare(old, new)
    assert old.get('knots') == new.get('knots') == 21, 'release gate requires mainline21'
    assert set(strict['metrics']) == {'local_batch', 'final_batch', 'candidate', 'execution_vbc', 'shadow'}
    metrics = {}
    for path, values in strict['metrics'].items():
        a, b = values['samples_ms'], values['primitive_ms']
        assert all(math.isfinite(x[q]) and x[q] > 0 for x in (a, b) for q in ('p50', 'p95')), 'invalid timing'
        limits = {q: a[q] + max(POLICY['absolute_regression_ms'], a[q]*POLICY['relative_regression'])
                  for q in ('p50', 'p95')}
        metrics[path] = dict(samples_ms=a, primitive_ms=b, allowed_primitive_ms=limits,
            status='PASS' if all(b[q] <= limits[q] for q in limits)
            and max(a['p95'], b['p95']) <= POLICY['p95_ceiling_ms'] else 'FAIL')
    return dict(status='PASS' if all(v['status'] == 'PASS' for v in metrics.values()) else 'FAIL',
                policy=dict(POLICY), metrics=metrics)
