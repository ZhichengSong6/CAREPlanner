import json
from pathlib import Path
import tempfile
import unittest
from run_phase_e_vbc_geometry_pair import validate_planner_report


class BindingTests(unittest.TestCase):
    def test_pass_and_invalid_reports(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json'
            frozen={'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node':'actual'}
            for status,sha,ok in [('PASS','actual',True),('FAIL','actual',False),('PASS','old',False)]:
                p.write_text(json.dumps(dict(status=status,binary_sha256=sha)))
                if ok:self.assertEqual(validate_planner_report(p,frozen)['binary_sha256'],'actual')
                else:
                    with self.assertRaises(AssertionError):validate_planner_report(p,frozen)
            p.write_text('{}')
            with self.assertRaises(KeyError):validate_planner_report(p,frozen)


if __name__=='__main__':unittest.main()
