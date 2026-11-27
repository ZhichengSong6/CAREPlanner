"""Bounded observation proposal ranking. No map, completion, or safety authority."""
import math
import os
import time
import numpy as np


class OptionalSearchExpired(RuntimeError):
    """Optional ranking work expired; retain the checked first proposal."""


class OptionalSearchWork:
    """Per-generation measured compute units; no extra model calls or I/O."""
    def __init__(self):
        self.max_s = {}
        self.calls = {}

    def estimate(self, kind):
        return self.max_s.get(kind, max(self.max_s.values(), default=0.))

    def call(self, guard, stage, kind, function, *args, **kwargs):
        started = time.monotonic()
        if guard is not None:
            guard.admit(stage, kind, self.estimate(kind), started)
        try:
            result = function(*args, **kwargs)
        except BaseException:
            ended = time.monotonic()
            self.observe(kind, ended-started)
            if guard is not None:
                guard.record(stage, kind, started, ended, returned=False)
            raise
        ended = time.monotonic()
        self.observe(kind, ended-started)
        if guard is not None:
            guard.record(stage, kind, started, ended, returned=True)
            guard.check(stage + '_end', now=ended)
        return result

    def observe(self, kind, elapsed):
        self.max_s[kind] = max(self.max_s.get(kind, 0.), elapsed)
        self.calls[kind] = self.calls.get(kind, 0) + 1


class OptionalSearchBudget:
    """One unchanged deadline, with measured admission and post-call checks.

    Twice the largest observed unit of this kind plus 50ms is an empirical
    reserve, not a hard realtime bound. A longer atomic call remains visible
    in the original exit_overrun_s and must fail the original 150ms audit.
    """
    def __init__(self, context, limit_s=1.5, work=None):
        if not math.isfinite(limit_s) or limit_s < 0.:
            raise ValueError('invalid optional search limit')
        self.limit_s = limit_s
        self.work = OptionalSearchWork() if work is None else work
        self.started = time.monotonic()
        self.deadline = self.started + limit_s
        for source in (context, context.get('completion') or {}):
            value = source.get('deadline_monotonic_s')
            if isinstance(value, (int, float)) and math.isfinite(value):
                self.deadline = min(self.deadline, value)
        self.last_stage = 'first_geometry_accepted'
        self.expired_stage = None
        self.stop_reason = None
        self.stop_stage = None
        self.exit_reserve_s = .05
        self.cost_multiplier = 2.
        self.last_units = []
        self.estimated_next_unit_s = None

    def remaining(self):
        return max(0., self.deadline-time.monotonic())

    def check(self, stage, now=None):
        self.last_stage = stage
        if (time.monotonic() if now is None else now) >= self.deadline:
            if self.expired_stage is None:
                self.expired_stage = stage
            if self.stop_reason is None:
                self.stop_reason, self.stop_stage = 'deadline', stage
            raise OptionalSearchExpired(stage)

    def admit(self, stage, kind, measured_cost_s, now):
        self.check(stage + '_start', now=now)
        required = self.cost_multiplier * measured_cost_s + self.exit_reserve_s
        self.estimated_next_unit_s = required
        if self.deadline-now <= required:
            self.stop_reason, self.stop_stage = 'insufficient_remaining_time', stage
            raise OptionalSearchExpired(stage)

    def record(self, stage, kind, started, ended, returned):
        self.last_units.append(dict(stage=stage, kind=kind,
            start_s=started-self.started, end_s=ended-self.started,
            wall_s=ended-started, returned=returned))
        if len(self.last_units) > 8:
            del self.last_units[0]

    def diagnostic(self):
        now = time.monotonic()
        return dict(limit_s=self.limit_s, effective_budget_s=max(0., self.deadline-self.started),
                    elapsed_s=now-self.started, expired=self.expired_stage is not None,
                    expired_stage=self.expired_stage, last_stage=self.last_stage,
                    exit_overrun_s=max(0., now-self.deadline),
                    stop_reason=self.stop_reason, stop_stage=self.stop_stage,
                    last_units=list(self.last_units), measured_unit_max_s=dict(self.work.max_s),
                    unit_calls=dict(self.work.calls), estimated_next_unit_s=self.estimated_next_unit_s,
                    cost_multiplier=self.cost_multiplier, exit_reserve_s=self.exit_reserve_s,
                    clock='monotonic', enforcement='cooperative_before_and_after_compute_units')


def task_rejoin_context(q_start, q_task, velocity, acceleration, source=None):
    arrays = [np.asarray(a, dtype=float).reshape(7)
              for a in (q_start, q_task, velocity, acceleration)]
    if (not all(np.all(np.isfinite(a)) for a in arrays) or
            np.any(arrays[2] <= 0.) or np.any(arrays[3] <= 0.)):
        raise ValueError('invalid observation task-return context')
    return dict(q_start=arrays[0].tolist(), q_task=arrays[1].tolist(),
                velocity=arrays[2].tolist(), acceleration=arrays[3].tolist(),
                source=source)


def _rest_time(q0, q1, context):
    distance = np.abs(np.asarray(q1)-np.asarray(q0))
    v, a = np.asarray(context['velocity']), np.asarray(context['acceleration'])
    return float(np.max(np.where(distance <= v*v/a,
        2.*np.sqrt(distance/a), distance/v+v/a)))


def task_pose_ranking_enabled():
    return os.environ.get('CARE_OBSERVATION_SELECTOR','task')=='task' and os.environ.get('CARE_TASK_POSE_RANKING') == 'true'


def task_departure_penalty(cost):
    """Soft time-equivalent preference; neither reachability nor completion proof.

    Fixed research weights: 0.20 m and 1 rad each cost one second in ranking.
    A necessary retreat may still win through shorter travel or observation order.
    """
    position = float(cost['task_position_m'])
    angle = float(cost['task_angle_rad'])
    if not all(math.isfinite(x) and x >= 0. for x in (position, angle)):
        raise ValueError('invalid task pose cost')
    return position / .20 + angle


def ranking_total(cost, completion=False):
    total = (cost['completion']['estimate_wall_s'] if completion else cost['total_s'])
    penalty = task_departure_penalty(cost) if task_pose_ranking_enabled() else 0.
    return total + (penalty / cost.get('ros_per_wall_s', 1.) if completion else penalty)


def candidate_cost(q, context, pose, task_pose, budget=None):
    if budget is not None:budget.check('candidate_cost_start')
    q = np.asarray(q, dtype=float).reshape(7)
    if not np.all(np.isfinite(q)):
        raise ValueError('nonfinite observation candidate')
    approach = _rest_time(context['q_start'], q, context)
    rejoin = _rest_time(q, context['q_task'], context)
    rate=float(context.get('ros_per_wall_s',1.))
    if not math.isfinite(rate) or rate<=0.:raise ValueError('invalid task-return clock rate')
    position = float(np.linalg.norm(pose[:3, 3]-task_pose[:3, 3]))
    cosine = float((np.trace(task_pose[:3, :3].T @ pose[:3, :3])-1.)*.5)
    angle = math.acos(min(1., max(-1., cosine)))
    result=dict(approach_s=approach, rejoin_s=rejoin, total_s=approach+rejoin,
                time_basis='physical_seconds', ros_per_wall_s=rate,
                task_position_m=position, task_angle_rad=angle,
                task_pose_error=max(position/.02, angle/.2))
    evaluator=context.get('corridor_evaluator')
    if evaluator is not None:
        if budget is not None:budget.check('corridor_query_start')
        query_context = dict(context, _search_budget=budget) if budget is not None else context
        try:preview=evaluator(q,query_context)
        except OptionalSearchExpired:raise
        except Exception:preview=dict(status='unavailable')
        if budget is not None:budget.check('corridor_query_end')
        result['corridor_preview']=preview
        result['corridor_used']=preview.get('status')=='valid'
        if result['corridor_used']:
            penalty=approach*preview['approach']['risk_fraction']+rejoin*preview['rejoin']['risk_fraction']
            result['new_observation_proxy_s']=penalty
            result['total_s']+=penalty
    completion = context.get('completion')
    if completion is not None:
        # This remains a calibrated ranking estimate, never a feasibility bound.
        estimate = ((approach * completion['approach_scale'] +
                    rejoin * completion['rejoin_scale'] +
                    result.get('new_observation_proxy_s', 0.)) / rate +
                    completion['acquisition_wall_s'] +
                    completion.get('return_overhead_wall_s', 0.) +
                    (completion['tail_ros_s'] + completion['hold_ros_s']) / rate)
        result['completion'] = dict(estimate_wall_s=estimate,
            deadline_monotonic_s=completion['deadline_monotonic_s'],
            profile_id=completion['profile_id'], arm_token=completion['arm_token'])
    if task_pose_ranking_enabled():
        result['task_departure_penalty_s'] = task_departure_penalty(result)
        result['ranking_total_s'] = ranking_total(result)
        result['ranking_weights'] = dict(position_m_per_s=.20, angle_rad_per_s=1.)
    return result


def alternative_can_improve(baseline, context, extra_compute_s, now=None):
    """Prune only when even a zero-risk direct-return proxy cannot win.

    The stop-to-stop time is subadditive: travel via any observation q cannot
    cost less than direct travel. This bounds the existing ranking proxy, not
    the physical path. A first candidate needing a detour remains searchable
    because an alternative can win by observation order alone.
    """
    if os.environ.get('CARE_TASK_FIRST_RESEARCH') == 'true':
        # One additional original branch, including unsuccessful geometry.
        # The first accepted proposal remains available when search stops.
        calls = context.get('_task_first_comparisons', 0)
        context['_task_first_comparisons'] = calls + 1
        if calls >= 1 or extra_compute_s >= 1.5:
            return False
    if os.environ.get('CARE_OBSERVATION_SELECTOR','task')=='observation':
        return True
    if baseline.get('approach_requires_detour') is not False:
        return True
    try:
        rate = float(context.get('ros_per_wall_s', 1.))
        elapsed = float(extra_compute_s)
        total = float(ranking_total(baseline))
        win_fraction = 1. if task_pose_ranking_enabled() else .95
        if not all(math.isfinite(x) for x in (rate, elapsed, total)) or rate <= 0. or min(elapsed, total) < 0.:
            return True
        lower = _rest_time(context['q_start'], context['q_task'], context)
        completion = context.get('completion')
        if completion is None:
            return lower + elapsed * rate < win_fraction * total
        if baseline.get('completion') is None:
            return True
        lower = (lower / rate * min(completion['approach_scale'], completion['rejoin_scale']) +
                 completion['acquisition_wall_s'] +
                 completion.get('return_overhead_wall_s', 0.) +
                 (completion['tail_ros_s'] + completion['hold_ros_s']) / rate)
        total = ranking_total(baseline, completion=True)
        remaining = completion['deadline_monotonic_s'] - (time.monotonic() if now is None else now)
        if remaining <= 0.:
            return False
        if lower + elapsed <= remaining < total:
            return True
        return lower + elapsed < win_fraction * total
    except (KeyError, ValueError, TypeError, FloatingPointError):
        return True


def improves_candidate(candidate, baseline, extra_compute_s):
    """Compare total task-return proxies after observation geometry passes.

    Times are stop-to-stop joint proxies, not collision-free travel times.
    Prefer sampled observation before body entry when both paths were checked.
    Within that class, charge extra computation and require a five-percent
    proxy gain before replacing the original fallback.
    """
    keys = ('approach_s', 'rejoin_s', 'total_s', 'task_pose_error')
    values = [float(c[k]) for c in (candidate, baseline) for k in keys]
    if not all(math.isfinite(v) and v >= 0. for v in values):
        return False
    if not math.isfinite(extra_compute_s) or extra_compute_s < 0.:
        return False
    research = os.environ.get('CARE_TASK_FIRST_RESEARCH') == 'true'
    if research:
        # Geometry order is independent of the optional corridor RPC.
        if ('approach_requires_detour' in candidate and 'approach_requires_detour' in baseline and
                candidate['approach_requires_detour'] != baseline['approach_requires_detour']):
            return not candidate['approach_requires_detour']
        comparable = (candidate.get('corridor_used', False) == baseline.get('corridor_used', False) and
            (not candidate.get('corridor_used', False) or
             same_corridor_snapshot(candidate['corridor_preview'], baseline['corridor_preview'])))
        if not comparable:
            # Use common information on BOTH sides, never missing=free for one.
            candidate, baseline = dict(candidate), dict(baseline)
            for cost in (candidate, baseline):
                cost['total_s'] -= cost.get('new_observation_proxy_s', 0.)
                cost['corridor_used'] = False
                if cost.get('completion') is not None:
                    cost['completion'] = dict(cost['completion'])
                    cost['completion']['estimate_wall_s'] -= cost.get('new_observation_proxy_s', 0.) / cost.get('ros_per_wall_s', 1.)
    # Missing information must not look cheaper than a queried corridor.
    if candidate.get('corridor_used',False)!=baseline.get('corridor_used',False):return False
    if candidate.get('corridor_used',False) and (
            not same_corridor_snapshot(candidate['corridor_preview'], baseline['corridor_preview'])):return False
    # Within the original candidate pool, prefer an approach that can observe
    # the actual target points before crossing the execution floor. Keep the
    # first accepted proposal when no such alternative exists. Sampling only
    # ranks proposals; every path still needs the original complete audits.
    if ('approach_requires_detour' in candidate and 'approach_requires_detour' in baseline and
            candidate['approach_requires_detour']!=baseline['approach_requires_detour']):
        return not candidate['approach_requires_detour']
    if os.environ.get('CARE_OBSERVATION_SELECTOR','task')=='observation':
        def observation_cost(cost):
            risk=cost.get('corridor_preview',{}).get('approach',{}).get('risk_fraction',0.) if cost.get('corridor_used') else 0.
            return cost['approach_s']*(1.+risk)
        return observation_cost(candidate)+extra_compute_s*candidate.get('ros_per_wall_s',1.) < .95*observation_cost(baseline)
    c, b = candidate.get('completion'), baseline.get('completion')
    if c is not None or b is not None:
        if c is None or b is None:
            return False
        if any(c.get(k) != b.get(k) for k in ('deadline_monotonic_s', 'profile_id', 'arm_token')):
            return False
        cv, bv = c.get('estimate_wall_s'), b.get('estimate_wall_s')
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0.
               for x in (cv, bv, c.get('deadline_monotonic_s'))):
            return False
        remaining = c['deadline_monotonic_s'] - time.monotonic()
        if remaining <= 0.:
            return False
        # Candidate computation has already elapsed. Charge it as opportunity
        # cost when comparing with the original immediately executable fallback.
        if (cv + extra_compute_s <= remaining) != (bv <= remaining):
            return cv + extra_compute_s <= remaining
        if task_pose_ranking_enabled():
            return ranking_total(candidate, True) + extra_compute_s < ranking_total(baseline, True) - 1e-6
        return cv + extra_compute_s < .95 * bv
    # FK endpoint distance is not a path cost or a safety constraint. Likewise
    # one longer leg may be outweighed by the other leg. Full execution audits
    # remain mandatory for every selected proposal.
    if task_pose_ranking_enabled():
        return ranking_total(candidate)+extra_compute_s*candidate.get('ros_per_wall_s', 1.) < ranking_total(baseline)-1e-6
    return candidate['total_s']+extra_compute_s*candidate.get('ros_per_wall_s', 1.) < .95*baseline['total_s']


def same_corridor_snapshot(a, b):
    """Identity, revision and query time are all needed for a fair comparison."""
    keys = ('map_identity', 'map_revision', 'snapshot_stamp_ns')
    return all(key in a and key in b and a[key] == b[key] for key in keys)
