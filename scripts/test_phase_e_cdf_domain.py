#!/usr/bin/env python3
"""Controls for historical domain classification and model-A/B preregistration."""
import math
from collections import Counter
import unittest

import audit_phase_e_cdf_domain as audit


class DomainControls(unittest.TestCase):
    def test_float32_face_and_real_overflow(self):
        self.assertTrue(audit.in_domain([.800000011920929, -.8, 1.]))
        self.assertFalse(audit.in_domain([.80001, 0., .5]))
        self.assertFalse(audit.in_domain([0., 0., 1.001]))
        self.assertFalse(audit.in_domain([0., 0., -.001]))

    def test_invalid_evidence_cannot_disappear(self):
        for point in ([0, math.nan, 0], [0, math.inf, 0], [0, 0]):
            with self.assertRaises(ValueError):
                audit.Points().add(point, 'invalid')

    def test_outside_counts_and_provenance(self):
        p = audit.Points()
        p.add([0, 0, 1.05], {'source': 'example.csv', 'line': 2})
        p.add([0, 0, .99], {'source': 'example.csv', 'line': 3})
        r = p.report()
        self.assertEqual(r['outside'], 1)
        self.assertEqual(r['inside_within_25mm_of_face'], 1)
        self.assertEqual(r['outside_faces'], {'z_upper': 1})
        self.assertEqual(r['outside_examples'][0]['evidence']['line'], 2)

    def test_empty_is_unmeasured(self):
        r = audit.Points().report()
        self.assertIsNone(r['bounds_min_m'])
        self.assertIsNone(r['minimum_signed_face_distance_m'])

    def test_unquoted_ros_arrays(self):
        stamp, row = audit.parse_line('123,C5 status=evaluated_unknown point=[0.1,0.2,0.3] gradient=[1,2,3,4,5,6,7] distance=-.2\n')
        self.assertEqual(stamp, 123)
        self.assertEqual(row['point'], [.1, .2, .3])
        self.assertEqual(len(row['gradient']), 7)
        self.assertIsNone(audit.parse_line('Is /clock being published?\n')[0])
        self.assertEqual(audit.parse_line('%time,field.data\n'), (None, {'recorder_header': True}))
        self.assertIsNone(audit.parse_line('WARNING: no messages received and simulated time is active.\n')[0])
        with self.assertRaises(ValueError):
            audit.parse_line('bad,C5 point=[0,0,0]')

    def test_both_seal_formats(self):
        self.assertEqual(audit.seal_hashes({'a': 'abc'}), {'a': 'abc'})
        self.assertEqual(audit.seal_hashes({'status': 'COMPLETE', 'artifacts':
                                         {'a': {'size': 1, 'sha256': 'abc'}}}), {'a': 'abc'})

    def test_normal_and_repair_qp_formats(self):
        payload = dict(query_stamp_ns=123, cdf_rows=[], G={})
        self.assertEqual(audit.qp_payload(payload), payload)
        self.assertEqual(audit.qp_payload({'subproblem': payload}), payload)
        with self.assertRaises(ValueError):
            audit.qp_payload({'unrecognized': True})

    def test_balanced_complete_schedule(self):
        rows = audit.make_schedule(['008', '029', '025', '026'])
        self.assertEqual(len(rows), 24)
        self.assertEqual(set(Counter((r['case'], r['model']) for r in rows).values()), {3})
        self.assertEqual(Counter(r['model'] for r in rows[::2]), {'old': 6, 'new': 6})
        for a, b in zip(rows[::2], rows[1::2]):
            self.assertEqual((a['case'], a['repeat']), (b['case'], b['repeat']))
            self.assertNotEqual(a['model'], b['model'])

    def test_control_selection_excludes_failed_and_fault_cases(self):
        cases = [dict(case_id='phase_e_goal_'+k, goal_position=[v, 0., 0.])
                 for k, v in [('008', 100), ('005', 0), ('006', 1), ('007', 3), ('009', 99)]]
        summary = {'cases': {c['case_id'][-3:]: dict(completed=3, valid_windows=3,
                   task_success=3, strict_hold=3, overall_safe=3, execution_unsafe=0) for c in cases}}
        summary['cases']['009']['strict_hold'] = 2
        eligible, selected = audit.choose_controls(cases, summary)
        self.assertEqual([c['case_id'][-3:] for c in selected], ['005', '007'])
        self.assertEqual(len(eligible), 3)


if __name__ == '__main__':
    unittest.main()
