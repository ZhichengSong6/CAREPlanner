#!/usr/bin/env python3
"""Corpus, contract, affine-row and rejection controls for offline qualification."""
import argparse
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from cdf_qualification_common import (check_manifest, eligible_for_targeted_test, in_box, matrix,
                                      new_output_dir, read, replace_cdf_rows, sample_jacobian, sha)
from qualify_phase_e_cdf import compare_results, derivative_check, evaluate, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/phase_e_cdf_qualification_20261023'


class QualificationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = STUDY / 'bundle'
        cls.cfg = read(cls.bundle / 'bundle.json')
        cls.queries = read(cls.bundle / 'queries.json')

    def test_bundle_identity_and_original_fault_binding(self):
        verify_bundle(self.bundle)
        old = read(self.bundle / 'original_fault_queries.json')
        faults = [r for r in self.queries if r['group'] == 'retained_fault']
        self.assertEqual(len(faults), 1248)
        for a, b in zip(old['queries'], faults):
            self.assertEqual(a, {k: v for k, v in b.items() if k not in ('group', 'expected_input_valid')})
        self.assertEqual(len(self.cfg['qps']), 28)

    def test_controls_cover_declared_geometry_and_domain_groups(self):
        counts = {}
        for q in self.queries:
            counts[q['group']] = counts.get(q['group'], 0) + 1
            self.assertEqual(in_box(q['point_m']), q['expected_input_valid'])
            d = q['current_primitive_point_distance_m']
            if q['group'] == 'inside_current_body':
                self.assertLess(d, -.001)
            elif q['group'] == 'normal_clear':
                self.assertGreaterEqual(d, .05)
            elif q['group'] == 'clearance_5mm':
                self.assertTrue(.003 <= d <= .007)
            elif q['group'] == 'clearance_15mm':
                self.assertTrue(.013 <= d <= .017)
        self.assertEqual(counts, self.cfg['groups'])
        self.assertEqual(counts['out_of_domain_probe'], 6)
        self.assertEqual(len({q['id'] for q in self.queries}), len(self.queries))

    def test_measured_pose_selection_is_inside_original_success_windows(self):
        poses = read(self.bundle / 'control_poses.json')
        self.assertEqual(len(poses), 24)
        self.assertEqual(len({p['id'].rsplit('_', 1)[0] for p in poses}), 8)
        for p in poses:
            self.assertTrue(p['window']['start'] <= p['receipt_ros_ns'] <= p['window']['end'])
            self.assertTrue(np.isfinite(p['q_rad']).all())

    def test_all_recorded_qps_round_trip_and_preserve_non_cdf_constraints(self):
        extra_samples = 0
        for item in self.cfg['qps']:
            s = read(self.bundle / item['file'])['subproblem']
            same = replace_cdf_rows(s, s['cdf_rows'])
            np.testing.assert_allclose(matrix(same['G']).toarray(), matrix(s['G']).toarray(), atol=1e-8, rtol=0)
            np.testing.assert_allclose(same['h_l'], s['h_l'], atol=1e-8, rtol=0)
            values = [dict(distance=r['distance'] + .01, gradient=(np.array(r['gradient']) * 1.1).tolist()) for r in s['cdf_rows']]
            changed = replace_cdf_rows(s, values)
            selected = {r['matrix_row'] for r in s['cdf_rows']}
            others = sorted(set(range(len(s['h_l']))) - selected)
            np.testing.assert_array_equal(matrix(changed['G'])[others].toarray(), matrix(s['G'])[others].toarray())
            np.testing.assert_array_equal(np.array(changed['h_l'])[others], np.array(s['h_l'])[others])
            for key in ('P', 'c', 'A', 'b', 'x_l', 'x_u', 'h_u', 'q_bar', 'u_bar'):
                self.assertEqual(changed[key], s[key])
            extra_samples += sum(r['timestep'] > len(s['q_bar']) - 1 for r in s['cdf_rows'])
        self.assertGreater(extra_samples, 0)

    def test_ambiguous_executable_sample_map_is_rejected(self):
        for item in self.cfg['qps']:
            s = read(self.bundle / item['file'])['subproblem']
            row = next((r for r in s['cdf_rows'] if r['timestep'] > len(s['q_bar']) - 1), None)
            if row:
                bad = copy.deepcopy(row)
                bad['gradient'][0] = 0.
                with self.assertRaisesRegex(ValueError, 'zero saved derivative'):
                    sample_jacobian(s, bad)
                return
        self.fail('No executable sample control')

    def manifest_fixture(self, directory):
        template = read(self.bundle / 'training_manifest_template.json')
        checkpoint = directory / 'checkpoint_fixture'
        checkpoint.write_bytes(b'synthetic test evidence; not a trained checkpoint')
        template['checkpoint_sha256'] = sha(checkpoint)
        for key in ('training_urdf', 'label_generator', 'dataset_manifest'):
            p = directory / key
            p.write_bytes((self.bundle / 'runtime/Arm_with_self_filter_collision.urdf').read_bytes()
                          if key == 'training_urdf' else b'synthetic contract test evidence')
            template[key] = dict(path=p.name, sha256=sha(p))
        template['label_definition'] = 'Synthetic fixture declaration only'
        return checkpoint, template

    def test_contract_requires_evidence_and_rejects_convention_mismatches(self):
        with tempfile.TemporaryDirectory(prefix='care_cdf_contract_') as td:
            directory = Path(td)
            checkpoint, valid = self.manifest_fixture(directory)
            self.assertTrue(check_manifest(valid, checkpoint, self.cfg, directory, 'gelu')['compatible'])
            changes = [dict(checkpoint_sha256='0' * 64), dict(activation='relu'), dict(output_units='metres'),
                       dict(excluded_links=[]), dict(label_definition=None), dict(training_urdf=None),
                       dict(runtime_input=None), dict(training_domain=None)]
            for key, value in [('joint_units', 'degrees'), ('joint_order', list(reversed(valid['runtime_input']['joint_order']))),
                               ('point_frame', 'world'), ('normalization', 'scaled_xyz')]:
                inputs = dict(valid['runtime_input'], **{key: value})
                changes.append(dict(runtime_input=inputs))
            for update in changes:
                with self.subTest(update=update):
                    bad = copy.deepcopy(valid)
                    bad.update(update)
                    self.assertFalse(check_manifest(bad, checkpoint, self.cfg, directory, 'gelu')['compatible'])
            self.assertFalse(check_manifest(None, checkpoint, self.cfg, directory, 'gelu')['compatible'])

    def test_all_positive_prediction_cannot_pass_despite_feasible_qp(self):
        rows = [dict(id='collision', group='inside_current_body', point_m=[0, 0, .3], q_rad=[0] * 7, distance=-.1)]
        old_qp = dict(index=0, repeats=[dict(lp=dict(success=False))])
        new_qp = dict(index=0, repeats=[dict(lp=dict(success=True))])
        new = [dict(rows[0], distance=.2)]
        comparison = compare_results(new, [new_qp], dict(queries=rows, qps=[old_qp]))
        self.assertEqual(comparison['qps_newly_feasible'], 1)
        self.assertEqual(comparison['newly_inside_positive_sign'], ['collision'])
        self.assertFalse(eligible_for_targeted_test(dict(compatible=True), comparison, False, False))

    def test_eligibility_requires_model_change_progress_and_no_control_regression(self):
        comparison = dict(qps_newly_feasible=1, qps_newly_infeasible=0,
                          new_normal_model_margin_blocks=[], inside_current_body_positive_sign=[])
        self.assertTrue(eligible_for_targeted_test(dict(compatible=True), comparison, False, False))
        for fields in [dict(qps_newly_feasible=0), dict(qps_newly_infeasible=1),
                       dict(new_normal_model_margin_blocks=['normal']), dict(inside_current_body_positive_sign=['collision'])]:
            self.assertFalse(eligible_for_targeted_test(dict(compatible=True), dict(comparison, **fields), False, False))
        self.assertFalse(eligible_for_targeted_test(dict(compatible=False), comparison, False, False))
        self.assertFalse(eligible_for_targeted_test(dict(compatible=True), comparison, True, False))
        self.assertFalse(eligible_for_targeted_test(dict(compatible=True), comparison, False, True))

    def test_sealed_output_and_overwriting_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='care_cdf_output_') as td:
            directory = Path(td)
            (directory / 'artifact_sha256.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'sealed'):
                new_output_dir(directory / 'new')
            self.assertFalse((directory / 'new').exists())
        with tempfile.TemporaryDirectory(prefix='care_cdf_output_') as td:
            with self.assertRaises(FileExistsError):
                new_output_dir(td)

    def test_nonfinite_model_output_is_rejected(self):
        net = SimpleNamespace(pair_distance_and_gradient=lambda p, q: (torch.full((len(p),), float('nan')), torch.zeros((len(p), 7))))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            evaluate(net, np.zeros((1, 3)), np.zeros((1, 7)))

    def test_finite_difference_check_detects_wrong_gradient(self):
        net = SimpleNamespace(model=lambda inputs: inputs[:, 3:].sum(dim=1, keepdim=True))
        cfg = dict(finite_difference_scales=[.001, .0005], derivative_atol=.005, derivative_rtol=.02)
        good = derivative_check(net, np.zeros((2, 3)), np.zeros((2, 7)), np.ones((2, 7)), cfg)
        self.assertTrue(all(r['failed_components'] == 0 for r in good))
        bad = derivative_check(net, np.zeros((2, 3)), np.zeros((2, 7)), np.zeros((2, 7)), cfg)
        self.assertTrue(all(r['failed_components'] == 14 for r in bad))

    def test_real_old_model_workflow_and_same_model_comparison(self):
        baseline = read(STUDY / 'baseline/report.json')
        same = read(STUDY / 'self_comparison/report.json')
        self.assertEqual(baseline['input_queries'], 1872)
        self.assertEqual(baseline['rejected_domain_probes'], 6)
        self.assertEqual(len(baseline['qps']), 28)
        self.assertTrue(all(not r['repeats'][0]['lp']['success'] for r in baseline['qps']))
        self.assertTrue(all(r['recorded_status'] == r['repeats'][0]['native']['status'] for r in baseline['qps']))
        self.assertFalse(same['ready_for_targeted_model_test'])
        c = same['comparison']
        self.assertEqual(c['qps_newly_feasible'], 0)
        self.assertEqual(c['new_normal_model_margin_blocks'], [])
        self.assertTrue(all(r['model_margin_pass_to_fail'] == r['model_margin_fail_to_pass'] == 0 for r in c['group_changes'].values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--study-dir', type=Path, default=STUDY)
    args, rest = parser.parse_known_args()
    STUDY = args.study_dir.resolve()
    unittest.main(argv=[__file__] + rest, verbosity=2)
