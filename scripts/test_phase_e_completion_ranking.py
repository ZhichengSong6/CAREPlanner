"""Completion estimates cannot change geometry, deadlines, or fallback authority."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/care_visibility_cdf/scripts'))
from observation_task_rejoin import (candidate_cost, improves_candidate, task_rejoin_context,
                                     alternative_can_improve)
from observation_research_trace import (ObservationResearchTrace, completion_context,
                                        publish_task_window)


class CompletionRankingTests(unittest.TestCase):
    def costs(self):
        old = dict(approach_s=1., rejoin_s=2., total_s=3., task_pose_error=0.)
        new = dict(approach_s=1.5, rejoin_s=1., total_s=2.5, task_pose_error=0.)
        shared = dict(deadline_monotonic_s=110., profile_id='development_v1', arm_token='a')
        old['completion'] = dict(shared, estimate_wall_s=4.)
        new['completion'] = dict(shared, estimate_wall_s=2.5)
        return new, old

    def test_complete_estimate_includes_measured_rate_and_tail(self):
        c = task_rejoin_context(np.zeros(7), np.ones(7), np.ones(7), np.ones(7))
        c['ros_per_wall_s'] = .5
        c['completion'] = dict(approach_scale=2., rejoin_scale=3., acquisition_wall_s=.4,
            tail_ros_s=1.1, hold_ros_s=.1, deadline_monotonic_s=110., profile_id='p', arm_token='a')
        result = candidate_cost(np.full(7, .25), c, np.eye(4), np.eye(4))
        expected = 2. * result['approach_s'] + 3. * result['rejoin_s'] + .4 + 1.2/.5
        self.assertAlmostEqual(result['completion']['estimate_wall_s'], expected)
        self.assertNotIn('execution_authorized', result)

    def test_prefer_estimated_completion_within_original_deadline(self):
        new, old = self.costs()
        with patch('observation_task_rejoin.time.monotonic', return_value=107.):
            self.assertTrue(improves_candidate(new, old, .1))
            self.assertFalse(improves_candidate(old, new, .1))

    def test_charge_extra_compute_and_reject_expired(self):
        new, old = self.costs()
        with patch('observation_task_rejoin.time.monotonic', return_value=107.):
            self.assertFalse(improves_candidate(new, old, 2.))
        with patch('observation_task_rejoin.time.monotonic', return_value=110.):
            self.assertFalse(improves_candidate(new, old, 0.))

    def test_no_mixed_profiles_or_task_deadlines(self):
        for key, value in [('profile_id', 'other'), ('arm_token', 'other'),
                           ('deadline_monotonic_s', 111.)]:
            new, old = self.costs()
            new['completion'][key] = value
            self.assertFalse(improves_candidate(new, old, 0.))
        new, old = self.costs()
        del new['completion']
        self.assertFalse(improves_candidate(new, old, 0.))

    def test_invalid_estimate_does_not_replace_fallback(self):
        for value in [float('nan'), float('inf'), -1., 'fast']:
            new, old = self.costs()
            new['completion']['estimate_wall_s'] = value
            self.assertFalse(improves_candidate(new, old, 0.))

    def test_legacy_ranking_without_profile_unchanged(self):
        new, old = self.costs()
        del new['completion']; del old['completion']
        self.assertTrue(improves_candidate(new, old, .1))
        self.assertFalse(improves_candidate(new, old, 1.))

    def test_profile_requires_explicit_enable_and_real_development_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            w, p = Path(directory)/'window.json', Path(directory)/'profile.json'
            profile = dict(schema=1, profile_id='p', training_episode_ids=['dev_1'],
                approach_scale=1., rejoin_scale=1., acquisition_wall_s=.1,
                tail_ros_s=1.1, hold_ros_s=.1)
            p.write_text(json.dumps(profile))
            env = dict(CARE_TASK_WINDOW_FILE=str(w), CARE_COMPLETION_PROFILE_FILE=str(p))
            with patch.dict(os.environ, env, clear=True):
                publish_task_window(100., 60., 5000000000, 'arm')
                self.assertIsNone(completion_context(now=101.))
                os.environ['CARE_COMPLETION_RANKING_ENABLED'] = 'true'
                context = completion_context(now=101.)
                self.assertEqual(context['deadline_monotonic_s'], 160.)
                self.assertIsNone(completion_context(now=160.))
                self.assertIsNone(completion_context(now=99.))
                profile['training_episode_ids'] = []
                p.write_text(json.dumps(profile))
                self.assertIsNone(completion_context(now=101.))

    def test_trace_preserves_values_and_reports_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trace.jsonl'
            trace = ObservationResearchTrace(str(path))
            trace.emit('hybrid_call', 1., seed=np.arange(7.), invalid=float('nan'))
            trace.emit('seen_query', 2., seen=True)
            trace.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r['sequence'] for r in rows], [1, 2])
            self.assertEqual(rows[0]['fields']['seed'], list(range(7)))
            self.assertIsNone(rows[0]['fields']['invalid'])
            status = json.loads(Path(str(path)+'.status.json').read_text())
            self.assertEqual(status['dropped'], 0)
            self.assertTrue(status['drained'])
            with self.assertRaises(FileExistsError):
                ObservationResearchTrace(str(path))

    def test_pruning_lower_bound_preserves_winners_over_independent_candidates(self):
        rng = np.random.default_rng(281)
        pruned = 0
        for _ in range(60):
            c = task_rejoin_context(rng.uniform(-2., 2., 7), rng.uniform(-2., 2., 7),
                                   rng.uniform(.5, 2., 7), rng.uniform(1., 6., 7))
            c['ros_per_wall_s'] = rng.uniform(.3, 1.)
            baseline = candidate_cost(rng.uniform(-2., 2., 7), c, np.eye(4), np.eye(4))
            baseline['approach_requires_detour'] = False
            elapsed = rng.uniform(0., 10.)
            if alternative_can_improve(baseline, c, elapsed):
                continue
            pruned += 1
            for _ in range(80):
                alternative = candidate_cost(rng.uniform(-3., 3., 7), c, np.eye(4), np.eye(4))
                alternative['approach_requires_detour'] = False
                self.assertFalse(improves_candidate(alternative, baseline, elapsed))
        self.assertGreater(pruned, 10)

    def test_detour_improvement_and_deadline_fit_are_not_pruned(self):
        c = task_rejoin_context(np.zeros(7), np.ones(7), np.ones(7), np.ones(7))
        baseline = dict(total_s=2., approach_requires_detour=True)
        self.assertTrue(alternative_can_improve(baseline, c, 20.))
        c['completion'] = dict(approach_scale=1., rejoin_scale=1., acquisition_wall_s=0.,
            tail_ros_s=0., hold_ros_s=0., deadline_monotonic_s=103.)
        baseline = dict(total_s=3., approach_requires_detour=False,
                        completion=dict(estimate_wall_s=3.1))
        self.assertTrue(alternative_can_improve(baseline, c, .96, now=100.))


if __name__ == '__main__':
    unittest.main()
