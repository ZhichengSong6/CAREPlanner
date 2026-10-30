"""Immutable, task-bound recovery policy carried by recovery_snapshot.

Hashes bind content; this planning protocol never grants motion authority.
Target identity is stable across heartbeats. Policy expiry is anchored to the
actual measurement, not the publication time.
"""
import copy
import hashlib
import json
import math
import re

from task_observation_binding import canonical


def content_sha(kind, content):
    return hashlib.sha256(('care_recovery_'+kind+'_v1\0'+content).encode()).hexdigest()


def target_sha(target):
    return content_sha('target', canonical(target))


def vector(value, size=7):
    return (isinstance(value, list) and len(value) == size and
            all(type(x) in (int, float) and math.isfinite(x) for x in value))


def validate_policy(policy, target, binding=None):
    required = {'schema', 'task_context', 'task_regime', 'target_sha256',
                'measured_ns', 'valid_until_ns', 'q_eval', 'frontier', 'guidance'}
    if (not isinstance(policy, dict) or set(policy) != required or
            type(policy['schema']) is not int or policy['schema'] != 1 or target is None or
            not isinstance(policy['task_context'], str) or
            (binding is not None and policy['task_context'] != binding) or
            not isinstance(policy['task_regime'], str) or
            re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}:[1-9][0-9]*:(NORMAL|REPAIR|PROBE_NORMAL)',
                         policy['task_regime']) is None or
            policy['target_sha256'] != target_sha(target) or
            any(type(policy[k]) is not int or not 0 < policy[k] < 2**64
                for k in ('measured_ns', 'valid_until_ns')) or
            not 0 < policy['valid_until_ns']-policy['measured_ns'] <= 200000000 or
            not vector(policy['q_eval'])):
        raise ValueError('invalid_recovery_policy_binding')
    f = policy['frontier']
    if (not isinstance(f, dict) or set(f) != {
            'target', 'joint_mask', 'frontier_weight_scale', 'qvis_weight_scale'} or
            not vector(f['target']) or f['joint_mask'] != target['joint_mask'] or
            not vector(f['joint_mask']) or any(type(f[k]) not in (int, float) or
                not math.isfinite(f[k]) or not 0 <= f[k] <= 1e6
                for k in ('frontier_weight_scale', 'qvis_weight_scale'))):
        raise ValueError('invalid_recovery_frontier')
    g = policy['guidance']
    if g is not None:
        if (not isinstance(g, dict) or set(g) != {
                'sequence', 'sensor_id', 'value', 'target', 'gradient', 'drive_enabled',
                'geometry_valid', 'geometry_signature', 'conservative_fov_margin'} or
                type(g['sequence']) is not int or not 0 < g['sequence'] < 2**53 or
                type(g['sensor_id']) is not int or not 0 <= g['sensor_id'] < 8 or
                g['sensor_id'] != target['sensor_id'] or
                not vector(g['target']) or not vector(g['gradient']) or
                any(type(g[k]) is not bool for k in ('drive_enabled', 'geometry_valid')) or
                not isinstance(g['geometry_signature'], str) or len(g['geometry_signature']) > 512 or
                any(type(g[k]) not in (int, float) or not math.isfinite(g[k])
                    for k in ('value', 'conservative_fov_margin')) or
                max(abs(a-b) for a,b in zip(g['target'], policy['q_eval'])) > .2+1e-9 or
                any(abs((1-m)*v) > 1e-9 for m,v in zip(f['joint_mask'],g['gradient']))):
            raise ValueError('invalid_recovery_guidance')
    canonical(policy)
    return policy


def selection_record(target, policy=None):
    raw = canonical(target)
    if policy is not None:
        validate_policy(policy, target)
    policy_raw = '' if policy is None else canonical(policy)
    return dict(schema=1, target_content=raw, target_sha256=content_sha('target', raw),
                policy_content=policy_raw,
                policy_sha256=content_sha('policy', policy_raw) if policy_raw else '')


def validate_selection(selection, target, binding=None):
    if (not isinstance(selection, dict) or set(selection) != {
            'schema', 'target_content', 'target_sha256', 'policy_content', 'policy_sha256'} or
            type(selection['schema']) is not int or selection['schema'] != 1 or
            selection['target_content'] != canonical(target) or
            selection['target_sha256'] != content_sha('target', selection['target_content']) or
            not isinstance(selection['policy_content'], str)):
        raise ValueError('invalid_recovery_selection')
    raw = selection['policy_content']
    if not raw:
        if selection['policy_sha256']:
            raise ValueError('invalid_empty_recovery_policy')
        return None
    if selection['policy_sha256'] != content_sha('policy', raw):
        raise ValueError('recovery_policy_digest_mismatch')
    policy = json.loads(raw)
    if canonical(policy) != raw:
        raise ValueError('noncanonical_recovery_policy')
    return validate_policy(policy, target, binding)


def build_policy(target, result, context, control, now_ns, diagnostic=None):
    """Return a measured, acknowledged policy, or a target-only bootstrap."""
    def skip(reason):
        if diagnostic is not None:diagnostic['reason']=reason
        return None
    if target is None:return skip('no_target')
    if result is None:return skip('no_frontier')
    if control is None:return skip('no_control')
    from task_recovery_events import regime_identity
    if control['task_context'] != context['candidate_binding']:return skip('control_task_mismatch')
    if not control['payload']['ready']:return skip('control_not_ready')
    if control['payload']['mode'] != 'REPAIR':return skip('control_not_repair')
    if control['payload'].get('recovery_target_sha256') != target_sha(target):return skip('target_not_acknowledged')
    if not control['stamp_ns'] <= now_ns < control['valid_until_ns']:return skip('control_not_current')
    if result.get('observation_token') != target['observation_token']:return skip('frontier_target_mismatch')
    if result.get('region_token') != target['region_token']:return skip('frontier_region_mismatch')
    measured = result.get('q_eval')
    stamp = result.get('measured_stamp', float('nan'))
    if measured is None or not math.isfinite(stamp):
        return skip('missing_measured_pose_or_time')
    measured_ns = round(stamp*1e9)
    until = min(measured_ns+200000000, context['valid_until_ns'],
                control['valid_until_ns'])
    if not context['started_ns'] <= measured_ns <= now_ns < until:
        return skip('measurement_or_local_window_expired')
    guide = result.get('guidance')
    guidance = None
    if guide is not None:
        if (guide['observation_token'] != target['observation_token'] or
                guide['region_token'] != target['region_token'] or
                round(guide['measured_stamp']*1e9) != measured_ns or
                list(guide['q_eval']) != list(measured) or
                list(guide['joint_mask']) != target['joint_mask']):
            return skip('guidance_snapshot_mismatch')
        guidance = {k:copy.deepcopy(guide[k]) for k in (
            'sequence','sensor_id','value','drive_enabled','geometry_valid',
            'geometry_signature','conservative_fov_margin')}
        guidance.update(target=[float(x) for x in guide['target']],
                        gradient=[float(x) for x in guide['gradient']])
    policy = dict(schema=1, task_context=context['candidate_binding'],
        task_regime=regime_identity(control), target_sha256=target_sha(target),
        measured_ns=measured_ns, valid_until_ns=until, q_eval=[float(x) for x in measured],
        frontier=dict(target=[float(x) for x in result.get('legacy_target',result['target'])],
            joint_mask=[float(x) for x in result.get('q_vis_joint_mask',target['joint_mask'])],
            frontier_weight_scale=float(result['frontier_weight_scale']),
            qvis_weight_scale=float(result['qvis_weight_scale'])), guidance=guidance)
    result=validate_policy(policy, target, context['candidate_binding'])
    if diagnostic is not None:diagnostic['reason']='bound_measured_policy'
    return result
