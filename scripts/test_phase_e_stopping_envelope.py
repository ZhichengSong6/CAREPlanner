#!/usr/bin/env python3
"""Stop-budget/coverage regressions plus 300 actual-controller schedule replays."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

import numpy as np

from phase_e_stop_envelope import stopping_budget, coverage_check

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase_e_stop_envelope_20261001/tests'
A=[3.,3.,4.,4.,6.,6.,6.]; V=[2.,2.,2.,2.,2.5,2.5,2.5]


class Bounds(unittest.TestCase):
    def budget(self,**overrides):
        args=dict(applied=[.3]*7,target=[.3]*7,command_age_s=0.,measurement_age_s=0.,
                  max_update_gap_s=.001,acceleration=A,velocity_limits=V)
        args.update(overrides); return stopping_budget(**args)

    def test_live_target_acceleration_must_be_included(self):
        b=self.budget(applied=[0.]*7,target=[1.]*7)
        self.assertGreater(b.command_travel_rad[0],.26)
        self.assertIsNone(b.physical_travel_rad)

    def test_measurement_age_and_dynamic_error_are_separate(self):
        base=self.budget()
        extra=self.budget(measurement_age_s=.01,physical_error_rad=[.002]*7,physical_settle_s=.04)
        np.testing.assert_allclose(extra.physical_travel_rad-base.command_travel_rad,np.array(V)*.01+.002)
        self.assertAlmostEqual(extra.physical_stop_s,base.command_stop_s+.04)

    def test_invalid_inputs_and_partial_physical_contract(self):
        for override in [dict(command_age_s=-1),dict(max_update_gap_s=0),dict(applied=[float('nan')]*7),
                         dict(target=[3.]*7),dict(physical_error_rad=[0.]*7),dict(acceleration=[0.]*7)]:
            with self.subTest(override=override),self.assertRaises(ValueError): self.budget(**override)

    def test_long_callback_gaps_increase_stopping_reserve(self):
        self.assertGreater(self.budget(max_update_gap_s=.1).command_travel_rad[0],self.budget().command_travel_rad[0]*2)

    def test_production_policy_replays_inside_independent_formula(self):
        OUT.mkdir(parents=True,exist_ok=True)
        source=ROOT/'scripts/simulate_phase_e_stopping_envelope.cpp'
        header=ROOT/'src/care_velocity_controller/include/care_velocity_controller/expiring_velocity_core.hpp'
        binary=OUT/'controller_replay'
        subprocess.run(['g++','-std=c++14','-O2','-I'+str(header.parents[1]),str(source),'-o',str(binary)],check=True)
        raw=subprocess.check_output([str(binary)],text=True)
        (OUT/'controller_replays.jsonl').write_text(raw)
        cases=[json.loads(line) for line in raw.splitlines()]
        self.assertEqual(len(cases),300)
        for case in cases:
            b=self.budget(applied=case['applied'],target=case['target'],command_age_s=case['age'],max_update_gap_s=case['gap'])
            self.assertTrue(np.all(np.array(case['absolute_command_travel'])<=b.command_travel_rad+1e-10),case)
            self.assertLessEqual(case['observed_stop_s'],b.command_stop_s+1e-10)
        (OUT/'result.json').write_text(json.dumps(dict(status='PASS',cases=300,joint_cases=2100,
            scope='Conditional command-domain bound; no actuator dynamics',
            sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,header,ROOT/'scripts/phase_e_stop_envelope.py']}),indent=2)+'\n')


class Coverage(unittest.TestCase):
    def check(self,**overrides):
        args=dict(has_physical_contract=True,state_age_s=.001,max_state_age_s=.02,stop_horizon_s=.3,
                  evidence_valid_for_s=.4,complete_snapshot=True,inside_map=True,free_mask=[True,True])
        args.update(overrides); return coverage_check(**args)

    def test_synthetic_complete_evidence_is_only_conditional(self):
        r=self.check(); self.assertTrue(r['conditional_coverage']); self.assertFalse(r['runtime_admission'])

    def test_missing_physics_map_age_validity_or_domain_refuses_coverage(self):
        for override in [dict(has_physical_contract=False),dict(complete_snapshot=False),dict(state_age_s=.03),
                         dict(evidence_valid_for_s=.29),dict(inside_map=False),dict(free_mask=[True,False]),
                         dict(free_mask=[]),dict(free_mask=[float('nan')])]:
            with self.subTest(override=override): self.assertFalse(self.check(**override)['conditional_coverage'])


if __name__=='__main__': unittest.main()
