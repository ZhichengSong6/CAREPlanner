#!/usr/bin/env python3
"""Recorder extraction must retain exact joint values, integer times and identity."""
import csv
from pathlib import Path
import tempfile
import unittest

from prepare_phase_e_vbc_real_trajectories import extract


class ReplayInputTests(unittest.TestCase):
    def fixture(self, rows):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)/'trajectory.csv'
        fields = ['%time','field.header.stamp'] + ['field.joint_names'+str(i) for i in range(7)]
        fields += ['field.points0.positions'+str(i) for i in range(7)] + ['field.points0.time_from_start']
        with path.open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for receipt, stamp, value in rows:
                row = dict(zip(fields, [receipt, stamp] + ['joint'+str(7-i) for i in range(7)] +
                               [value]*7 + ['123456789']))
                writer.writerow(row)
        return path

    def test_exact_payload_and_bounded_carry(self):
        path = self.fixture([(90, 81, '0.12345678901234567'), (100, 92, '0.2'), (120, 117, '0.3')])
        result = extract(path, dict(start_ros_ns=100, end_ros_ns=110))
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0]['carry_in'])
        self.assertEqual(result[0]['stamp_ns'], 81)
        self.assertEqual(result[0]['joint_names'][0], 'joint7')
        self.assertEqual(result[0]['points'][0]['q'][0], float('0.12345678901234567'))
        self.assertEqual(result[0]['points'][0]['time_ns'], 123456789)
        self.assertFalse(result[1]['carry_in'])
        self.assertEqual(result[1]['source_line'], 3)

    def test_nonfinite_not_sanitized(self):
        path = self.fixture([(100, 90, 'nan')])
        with self.assertRaises(AssertionError):
            extract(path, dict(start_ros_ns=100, end_ros_ns=110))

    def test_fractional_identity_rejected(self):
        path = self.fixture([(100, '90.1', '0')])
        with self.assertRaises(ValueError):
            extract(path, dict(start_ros_ns=100, end_ros_ns=110))


if __name__ == '__main__':
    unittest.main()
