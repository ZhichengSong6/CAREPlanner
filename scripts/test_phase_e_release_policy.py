import copy
import unittest
from test_phase_e_candidate_tail_pairs import fixture
from phase_e_release_policy import compare_release


def full(backend='samples', value=2.):
    result = fixture(backend, value)
    result['knots'] = 21
    result['latency_ms'] = {k: dict(p50=value, p95=value) for k in
                            ('local_batch', 'final_batch', 'candidate', 'execution_vbc', 'shadow')}
    return result


class ReleasePolicyTests(unittest.TestCase):
    def test_boundary(self):
        self.assertEqual(compare_release(full(), full('primitive', 3.))['status'], 'PASS')
        self.assertEqual(compare_release(full(), full('primitive', 3.00001))['status'], 'FAIL')

    def test_relative(self):
        self.assertEqual(compare_release(full(value=20.), full('primitive', 22.))['status'], 'PASS')
        self.assertEqual(compare_release(full(value=20.), full('primitive', 22.001))['status'], 'FAIL')

    def test_period_ceiling(self):
        self.assertEqual(compare_release(full(value=49.), full('primitive', 50.1))['status'], 'FAIL')

    def test_no_safety_relaxation(self):
        a, b = full(), full('primitive')
        b['status'] = 'FAIL'
        with self.assertRaises(AssertionError): compare_release(a, b)

    def test_no_stress_gate(self):
        a, b = full(), full('primitive')
        a['knots'] = b['knots'] = 50
        with self.assertRaises(AssertionError): compare_release(a, b)

    def test_invalid_latency(self):
        for value in (float('nan'), float('inf'), 0., -1.):
            with self.assertRaises(AssertionError): compare_release(full(), full('primitive', value))
            with self.assertRaises(AssertionError): compare_release(full(value=value), full('primitive'))

    def test_missing_path(self):
        a, b = full(), full('primitive')
        del a['latency_ms']['shadow']; del b['latency_ms']['shadow']
        with self.assertRaises(AssertionError): compare_release(a, b)


if __name__ == '__main__': unittest.main()
