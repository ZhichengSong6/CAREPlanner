"""Rank original-generator replacement proposals; retain the same q0 and gradient.

Corridor sampling is a cost proxy, not an executable path. Native recovery
separately scores geometry at measured states of certified executed prefixes.
"""
import copy
import time
import numpy as np
from candidate_replacement_runtime import sensor_mount_group_id
from observation_task_rejoin import candidate_cost, improves_candidate, alternative_can_improve
from observation_corridor_cost import approach_visibility_order


def proposal_cost(runtime, points, hybrid, context):
    # Lazy imports keep this scheduler independently testable without torch/ROS.
    from validate_visibility_oracle import fk_transform
    from check_visibility_self_occlusion import q_row_to_map
    from evaluate_direct_vs_projection_ascent import DEFAULT_JOINT_NAMES
    q = hybrid['selected_q_vis']
    pose = fk_transform(runtime.task_ee_chain, q_row_to_map(DEFAULT_JOINT_NAMES, q))
    task_pose = fk_transform(runtime.task_ee_chain,
                             q_row_to_map(DEFAULT_JOINT_NAMES, context['q_task']))
    cost = candidate_cost(q, context, pose, task_pose)
    order = approach_visibility_order(runtime, points, context['q_start'], q,
                                      int(hybrid['selected_sensor_id']))
    cost['approach_visibility_order'] = order
    if order.get('status') == 'valid':
        cost['approach_requires_detour'] = order['requires_detour']
    return cost


def select_replacement(runtime, points, q_zero, seed, first, options, context,
                       first_compute_s, clock=time.monotonic, cost_fn=proposal_cost):
    """At most one additional existing-generator call, at the identical seed.

    Cost scoring may fail soft. It must never throw away an accepted first
    proposal merely because an alternative or optional map preview failed.
    Only the selected head is charged to the persistent tried-head ledger.
    """
    diagnostic = dict(enabled=True, selected_by='first_accepted', candidates=[],
                      extra_calls=0, seed_policy='same_measured_q0',
                      execution_evaluation='native_certified_measured_prefixes')
    selected = first
    other = None
    try:
        if context is None or not first.get('accepted'):
            diagnostic['reason'] = 'no_context_or_first_candidate'
            return selected
        start = clock()
        baseline = cost_fn(runtime, points, first, context)
        diagnostic['candidates'].append(dict(sensor_id=first['selected_sensor_id'], cost=baseline))
        remaining = float(options['replacement']['compute_deadline_monotonic_s'])-clock()
        # Reserve the observed first-call time with headroom, plus publication.
        # This limits optional work within the existing six-second lease.
        estimate = max(.5, 1.5*float(first_compute_s))+.25
        diagnostic.update(remaining_compute_s=remaining, next_call_estimate_s=estimate)
        if remaining < estimate or not alternative_can_improve(baseline, context, clock()-start):
            diagnostic['reason'] = 'bounded_search_pruned'
            return selected
        other_options = copy.deepcopy(options)
        override = other_options['replacement']
        excluded = sorted(set(override['excluded']) | {int(first['selected_sensor_id'])})
        if len(excluded) >= 8:
            diagnostic['reason'] = 'no_remaining_head'
            return selected
        override.update(excluded=excluded,
            tried_mount_groups=sorted({sensor_mount_group_id(s) for s in excluded}))
        diagnostic['extra_calls'] = 1
        other = runtime.generate(points, q_zero, branch_seed_row=np.asarray(seed).copy(), **other_options)
        if not other.get('accepted'):
            diagnostic['alternative_rejected'] = other.get('attempts', [])
            return selected
        alternative = cost_fn(runtime, points, other, context)
        diagnostic['candidates'].append(dict(sensor_id=other['selected_sensor_id'], cost=alternative))
        # Both proposals have now been computed; compare their remaining costs.
        if improves_candidate(alternative, baseline, 0.):
            selected = other
            diagnostic['selected_by'] = 'observation_order_and_task_return'
        diagnostic['extra_compute_s'] = clock()-start
        return selected
    except Exception as exc:
        diagnostic['ranking_error'] = str(exc)
        return selected
    finally:
        selected['replacement_action_selection'] = diagnostic
        selected['native_alternatives'] = [dict(accepted=r.get('accepted'),selected_sensor_id=r.get('selected_sensor_id'),selected_q_vis=r.get('selected_q_vis'),selected_joint_mask=r.get('selected_joint_mask')) for r in (first,other) if r is not None]
