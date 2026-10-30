#!/usr/bin/env python3
"""Pure orchestration tests; never starts ROS, CUDA, or any simulation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_phase_e_repeatability as subject


class RepeatabilityTests(unittest.TestCase):
    def test_refuses_source_change_addition_and_removal(self):
        for actual in ({'a': 'changed'}, {}, {'a': 'same', 'new': 'hash'}):
            with patch.object(subject, 'fingerprint', return_value=actual):
                with self.assertRaises(RuntimeError):
                    subject.assert_frozen({'a': 'same'})
        with patch.object(subject, 'fingerprint', return_value={'a': 'same'}):
            subject.assert_frozen({'a': 'same'})

    def test_refuses_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            subject.save(path, {'first': True})
            with self.assertRaises(FileExistsError):
                subject.save(path, {'first': False})
            self.assertIn('true', path.read_text())

    def test_failure_is_counted_not_dropped(self):
        base = dict(model='h9', case_id='phase_e_goal_026', overall_safe=True,
            startup_order_pass=True, final_position_error_m=0.3,
            final_orientation_error_rad=0.8, normal_box_reseed_count=2,
            observation=dict(stage_counts={'CERTIFICATION_REJECTED': 2},
                unmatched_verification_count=0, failed_generations=[], unfinished_generations=[]))
        rows = [dict(base, task_success=value) for value in (False, True, False)]
        result = subject.summarize(rows)[0]
        self.assertEqual((result['repeats'], result['task_pass'], result['safety_pass']), (3, 1, 3))
        self.assertEqual(result['observation_stages']['CERTIFICATION_REJECTED'], 6)
        self.assertEqual(subject.summarize([]), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
