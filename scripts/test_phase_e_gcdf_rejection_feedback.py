import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/egocentric_arm_planner/scripts'))
from gcdf_rejection_feedback import rejection_points


class FeedbackTest(unittest.TestCase):
    def fixture(self, n=1):
        h = NS(stamp=123, frame_id='base_link')
        trajectory = NS(header=h, joint_names=list(range(7)),
                        points=[NS(positions=[0.]*7) for _ in range(21)])
        batch = NS(header=h, dof=7, num_pairs=n, distance=[-.1]*n,
                   original_timestep=[8]*n, source_type=[0]*n,
                   point_flat=[v for i in range(n) for v in [i*.05, .1, .3]],
                   gradient_flat=[1.]*7*n, q_linearization_flat=[0.]*7*n)
        return batch, trajectory

    def test_tail_identity_not_index_or_gradient(self):
        b, t = self.fixture()
        self.assertEqual(rejection_points(b, t, 0.), ([0., .1, .3], False))

    def test_duplicate_points_and_occupied(self):
        b, t = self.fixture(2)
        b.point_flat[3:] = b.point_flat[:3]
        b.source_type[1] = 1
        b.original_timestep[1] = 20
        self.assertEqual(rejection_points(b, t, 0.), ([0., .1, .3], False))

    def test_safe_rows_not_feedback(self):
        b, t = self.fixture(); b.distance = [0.]
        self.assertEqual(rejection_points(b, t, 0.), ([], False))

    def test_capacity_no_partial_result(self):
        b, t = self.fixture(33)
        self.assertEqual(rejection_points(b, t, 0.), ([], True))

    def test_malformed_or_wrong_q_rejected(self):
        for field, value in [('dof', 8), ('source_type', []), ('source_type', [2]),
                             ('gradient_flat', []), ('point_flat', [float('nan')]*3),
                             ('distance', [float('nan')]), ('original_timestep', [21]),
                             ('q_linearization_flat', [0.]*6+[.001]),
                             ('header', NS(stamp=124, frame_id='base_link'))]:
            b, t = self.fixture()
            setattr(b, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                rejection_points(b, t, 0.)

    def test_real_case026_boundary_snapshot(self):
        path = ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_026_empty_qualification_20260914-unknown-recovery-01_dcde98b2/run/rejection_snapshots/rejection_04_da6b653bba73.json'
        self.assertTrue(path.is_file(), 'required real regression fixture missing')
        r = json.loads(path.read_text())
        b = NS(**r['final_gcdf_batch']); b.header = NS(**b.header)
        tr = r['audited_trajectory']
        t = NS(header=NS(stamp=tr['stamp_ns'], frame_id=tr['frame_id']),
               joint_names=tr['joint_names'], points=[NS(**p) for p in tr['points']])
        points, overflow = rejection_points(b, t, 0.)
        p = [.10000000149011612, .10000000149011612, .30000001192092896]
        self.assertFalse(overflow)
        self.assertIn(p, [points[i:i+3] for i in range(0, len(points), 3)])


if __name__ == '__main__':
    unittest.main()
