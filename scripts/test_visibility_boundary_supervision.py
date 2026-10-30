import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from visibility_boundary_supervision import boundary_intervals, interval_loss


class BoundarySupervisionTests(unittest.TestCase):
    def test_true_boundary_need_not_equal_numerical_root(self):
        minus = np.zeros(7); plus = np.zeros(7); plus[0] = .002
        root = (minus+plus)/2.
        result = boundary_intervals(minus, plus, root, -.0004, .0016, np.ones(7))
        # True boundary is at q0=.0004, not the saved approximate midpoint.
        distances = {'minus':-.0004, 'plus':.0016, 'root':.0006}
        for name, value in distances.items():
            low, high = result[name]
            self.assertLessEqual(low, value); self.assertGreaterEqual(high, value)
        self.assertEqual(result['plus'][1], .002)

    def test_missing_sign_change_and_invalid_metric_are_rejected(self):
        for lo, hi, mask in [(1., 2., np.ones(7)), (-1., -2., np.ones(7)),
                             (-1., 2., np.zeros(7)), (-1., 2., np.full(7, .5))]:
            with self.assertRaises(ValueError):
                boundary_intervals(np.zeros(7), np.ones(7), np.ones(7)/2, lo, hi, mask)

    def test_valid_values_receive_no_invented_exact_distance_gradient(self):
        prediction = torch.tensor([.001, -.001], requires_grad=True)
        loss = interval_loss(prediction, torch.tensor([0., -.002]), torch.tensor([.002, 0.]))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(prediction.grad.tolist(), [0., 0.])

    def test_outside_value_gradient_points_into_valid_interval(self):
        prediction = torch.tensor([.3, -.3], requires_grad=True)
        loss = interval_loss(prediction, torch.tensor([0., -.002]), torch.tensor([.002, 0.]))
        loss.backward()
        self.assertGreater(prediction.grad[0], 0.)
        self.assertLess(prediction.grad[1], 0.)


if __name__ == '__main__':
    unittest.main()
