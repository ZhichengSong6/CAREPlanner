#!/usr/bin/env python3
"""Frozen before/current scheduler + publication callbacks, controlled seen input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import test_phase_e_path_target_lock as path_fixture
import test_phase_e_dependency_publish_chain as publish_fixture

ROOT = Path(__file__).resolve().parents[1]


def scenario(source_root, stored_mask, published_mask):
    path_fixture.REPO = source_root
    publish_fixture.REPO = source_root
    helper = publish_fixture.PublishChainTests()
    node, fixture, clock = helper.node()
    parent = node._obligations[0]
    if stored_mask is not None:
        parent['q_vis_joint_mask'] = np.array(stored_mask, dtype=float)
    shared = dict(parent, q_vis=np.arange(7)*.1)
    if published_mask is None:
        shared.pop('q_vis_joint_mask', None)
    else:
        shared['q_vis_joint_mask'] = np.array(published_mask, dtype=float)
    node._traced_waypoint(shared, shared['q_vis'])
    original_token = node._trace_published_target['observation_token']
    fixture.event(node)
    helper.bounded(node._process_observation_dependency)
    assert node._repair_stack == [1, 2]
    # Unseen child cannot pop just because motion or more messages occurred.
    helper.bounded(node._update_actual_visibility_completion)
    assert node._repair_stack == [1, 2]
    node.seen_ids.add(2)
    clock.now += .1
    helper.bounded(node._update_actual_visibility_completion)
    assert node._repair_stack == [1]
    target = node._trace_published_target
    expected_mask = [1.]*7 if published_mask is None else published_mask
    assert np.array_equal(target['q_vis'], shared['q_vis'])
    assert list(node._dependency_attempts.values()) == [1]
    assert node._seen_obligation_count == 1
    return dict(pose_preserved=True,
        mask_preserved=bool(np.array_equal(target['q_vis_joint_mask'], expected_mask)),
        token_preserved=target['observation_token'] == original_token,
        original_token=original_token, resumed_token=target['observation_token'],
        resumed_mask=target['q_vis_joint_mask'].tolist(),
        dependency_attempts=list(node._dependency_attempts.values()),
        synthetic_seen_children=node._seen_obligation_count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    short = [1, 1, 0, 0, 0, 0, 0]
    full = [1]*7
    cases = [('shared_shorter_mask', full, short, True),
             ('shared_longer_mask', short, full, True),
             ('legacy_stored_masked_published', None, short, True),
             ('masked_stored_legacy_published', short, None, True),
             ('same_mask', short, short, False),
             ('legacy_both', None, None, False)]
    results = []
    for name, stored, published, old_fails in cases:
        before = scenario(args.before_root, stored, published)
        after = scenario(ROOT, stored, published)
        assert after['mask_preserved'] and after['token_preserved'], (name, after)
        assert (before['mask_preserved'] and before['token_preserved']) != old_fails, (name, before)
        results.append(dict(name=name, before=before, after=after))
    paths = ['src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py',
             'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py']
    report = dict(status='PASS', scenarios=results,
        source_sha256={str(root/rel): hashlib.sha256((root/rel).read_bytes()).hexdigest()
                       for root in [args.before_root, ROOT] for rel in paths},
        qualification='Real frozen before/current production methods with actual Lock/RLock and '
          'publication/token/completion chain. Shared target and map seen inputs controlled. '
          'No native planner, real inference, GCDF/VBC certificate, robot or task success claim.')
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print('PASS: six before/after scenarios; four old identity/mask counterexamples, two controls.')


if __name__ == '__main__':
    main()
