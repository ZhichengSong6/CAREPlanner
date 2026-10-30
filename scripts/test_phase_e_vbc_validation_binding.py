#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_phase_e_vbc_geometry_pair import REPO, validate_geometry_report


class ValidationBindingTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((REPO/'outputs/phase_e_vbc_optimization/geometry-final/result.json').read_text())
        self.frozen = dict(self.report['sha256'])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def validate(self, report, frozen=None):
        path = Path(self.tmp.name)/'report.json'
        path.write_text(json.dumps(report))
        return validate_geometry_report(path, self.frozen if frozen is None else frozen)

    def test_complete_matching_pass(self):
        self.assertEqual(self.validate(self.report)['validated_artifacts'], self.frozen)

    def test_failed_report_rejected(self):
        report = copy.deepcopy(self.report); report['status'] = 'FAIL'
        with self.assertRaises(AssertionError): self.validate(report)

    def test_empty_identity_rejected(self):
        report = copy.deepcopy(self.report); report['sha256'] = {}
        with self.assertRaises(AssertionError): self.validate(report)

    def test_binary_drift_rejected(self):
        frozen = dict(self.frozen)
        frozen['devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node'] = 'changed'
        with self.assertRaises(AssertionError): self.validate(self.report, frozen)

    def test_unavailable_artifact_rejected(self):
        report = copy.deepcopy(self.report); report['sha256']['missing'] = 'sha'
        with self.assertRaises(AssertionError): self.validate(report)


if __name__ == '__main__':
    unittest.main()
