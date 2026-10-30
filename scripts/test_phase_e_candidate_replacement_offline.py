#!/usr/bin/env python3
"""Offline replacement-spec regression using exp74 data and inert production MRO."""
import argparse
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

from phase_e_candidate_replacement_spec import Context, Ledger, ReplacementSpec, region_key, observation_token, ROOT
from test_phase_e_dependency_publish_chain import PublishChainTests
from test_phase_e_observation_dependency import obligation

INPUT=ROOT/'outputs/phase_e_r1_o1_o6_offline/20260921_cpu'
SEGMENT=ROOT/'outputs/phase_e_r1_segment_offline/20260921_effective_prefix/report.json'
EVIDENCE=json.loads((INPUT/'snapshots.json').read_text())
DIAGNOSTIC=json.loads((INPUT/'diagnostic.json').read_text())
SNAP=next(s for s in EVIDENCE['snapshots'] if s['query_stamp_ns']==22635000000)
REAL=next(r for r in DIAGNOSTIC['candidates'] if r['snapshot']==SNAP['label'] and
          r['obligation']==6 and r['model']=='R1' and r['sensor']==7)
SHA=next(g for g in DIAGNOSTIC['groups'] if g['snapshot']==SNAP['label'] and g['obligation']==6)['models']['R1']['checkpoint']['checkpoint_sha256']


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.fixture=PublishChainTests()
        n,_,clock=self.fixture.node();self.node=n;self.clock=clock;clock.now=22.635
        n._obligations=[]
        for oid in [1,6]:
            trace=EVIDENCE['traces'][str(oid)]
            ob=obligation(oid)
            ob.update(points=np.array(trace['active_set_points_xyz']),q_vis=np.array(trace['q_vis']),
                q_zero=np.array(trace['q_zero']),generation_event_id=trace['generation_event_id'])
            ob['keys']=tuple(n._cell_key(p) for p in ob['points'])
            n._obligations.append(ob)
            self.assertEqual(observation_token(ob),trace['observation_token'])
        n._repair_stack=[1,6];n._dependency_edges={6:1}
        n._dependency_pins={ob['id']:dict(region=region_key(ob['points']),q_vis=ob['q_vis'].copy()) for ob in n._obligations}
        n._latest_measured_q=np.array(SNAP['measured_q'])
        child=n._obligations[1]
        self.context=Context(2,22635000000,SNAP['map_stamp_ns'],6,1,observation_token(child),
                             region_key(child['points']),tuple(SNAP['measured_q']))
        n._trace_published_target=dict(observation_token=self.context.token,q_vis=child['q_vis'].copy())
        n.seen_threshold=.5
        n.confidence_client=lambda req:NS(confidence=[float(p.x<0 and 6 in n.seen_ids) for p in req.points],
                                         inside_map=bytes([1]*len(req.points)),current_visibility=[0.]*len(req.points))
        self.parent=n._dependency_pins[1]['q_vis'].copy()
        self.ledger=Ledger(attempts=1,segments=0)  # inherited exp74 stage, not a fresh allowance
        self.policy=ReplacementSpec(self.context,self.ledger)
        self.candidate=dict(checkpoint_sha=SHA,sensor=7,q_vis=REAL['branch']['q_candidate'],
            q_zero=REAL['branch']['q_zero'],geometry_accepted=REAL['geometry']['accepted'],
            local_progress_evidence=True)

    def reserve(self,context=None,family=None):
        c=context or self.context
        return self.policy.reserve(c,self.node._repair_stack,self.node._dependency_edges,
                                   family or (SHA,7),(c.mode_epoch,c.query_stamp_ns,c.token))

    def apply(self):
        reason,ticket=self.reserve();self.assertEqual(reason,'reserved')
        self.fixture.bounded(lambda:self.assertEqual(self.policy.apply(ticket,self.context,self.candidate,self.node),
                                                    'steering_replaced_requires_existing_gates'))
        return ticket

    def assert_live(self):
        self.assertEqual(self.node._repair_stack,[1,6])
        self.assertEqual({ob['id'] for ob in self.node._obligations},{1,6})
        np.testing.assert_array_equal(self.node._dependency_pins[1]['q_vis'],self.parent)

    def test_real_s7_updates_child_pin_and_publication_without_parent_mutation(self):
        self.apply();self.assert_live()
        actual=self.node._ordered_obligations()[0]
        np.testing.assert_array_equal(actual['q_vis'],self.candidate['q_vis'])
        self.assertNotEqual(self.node._trace_published_target['observation_token'],self.context.token)
        self.assertEqual(self.ledger.attempts,2)
        self.assertEqual(self.node._obligations[1]['generation_event_id'],EVIDENCE['traces']['6']['generation_event_id'])
        self.assertEqual([t for t,m in self.node.messages],['q','zero','deadline','points'])
        self.node._publish_schedule()
        np.testing.assert_array_equal(self.node._trace_published_target['q_vis'],self.candidate['q_vis'])

    def test_real_partial_diagnostics_never_claim_execution_or_seen(self):
        segment=json.loads(SEGMENT.read_text())
        row=next(r for r in segment['experiments'] if r['label']=='S7_first_0.03')
        self.assertFalse(row['prefix_brake_hold']['execution_authorized'])
        self.assertEqual(row['prefix_brake_hold']['complete_map_gcdf_vbc'],'NOT_RUN')
        self.apply();self.policy.outcome('partial_offline_check')
        self.node._update_actual_visibility_completion();self.assert_live()
        self.assertEqual(self.node._dependency_resumes,0)

    def test_synthetic_safe_and_motion_events_do_not_complete_child(self):
        self.apply()
        for event in ['safe_candidate','motion']:
            self.policy.outcome(event)
            self.node._latest_measured_q+=.02;self.clock.now+=.1
            self.node._update_actual_visibility_completion();self.assert_live()

    def test_synthetic_confidence_completion_resumes_exact_original_parent(self):
        self.apply();self.node.seen_ids.add(6);self.clock.now+=.1
        self.fixture.bounded(self.node._update_actual_visibility_completion)
        self.assertEqual(self.node._repair_stack,[1]);self.assertEqual(self.node._dependency_resumes,1)
        np.testing.assert_array_equal(self.node._trace_published_target['q_vis'],self.parent)
        self.assertEqual(self.ledger.attempts,2)

    def test_failed_generation_is_charged_and_not_retried_after_motion_or_token_change(self):
        _,ticket=self.reserve();self.policy.reject(ticket,'generation_failed')
        c=replace(self.context,token='new-token',query_stamp_ns=23000000000,measured_q=(.2,)*7)
        self.assertEqual(self.reserve(c)[0],'family_already_attempted')
        self.assertEqual(self.ledger.attempts,2)

    def test_inherited_last_slot_then_exhausted_no_new_five_attempts(self):
        self.ledger.attempts=4
        _,ticket=self.reserve();self.policy.reject(ticket,'generation_failed')
        self.assertEqual(self.ledger.attempts,5)
        self.assertEqual(self.reserve(family=(SHA,6))[0],'budget_exhausted')
        self.assert_live()

    def test_already_exhausted_exp74_tail_cannot_start_replacement(self):
        self.ledger.attempts=5
        self.assertEqual(self.reserve()[0],'budget_exhausted');self.assert_live()

    def test_segment_budget_survives_replacement_and_monotone_merge(self):
        self.ledger.segments=5;self.apply();self.policy.inherit_progress(0,6)
        self.policy.inherit_progress(0,0)
        self.assertEqual((self.ledger.attempts,self.ledger.segments),(2,6))
        self.assertEqual(self.reserve(family=(SHA,6))[0],'budget_exhausted')

    def test_pending_generation_has_one_owner_and_duplicate_callback_is_inert(self):
        _,ticket=self.reserve();self.assertEqual(self.reserve(family=(SHA,6))[0],'generation_in_flight')
        self.assertEqual(self.ledger.attempts,2)
        self.policy.apply(ticket,self.context,self.candidate,self.node)
        count=len(self.node.messages)
        self.assertEqual(self.policy.apply(ticket,self.context,self.candidate,self.node),'stale_ticket')
        self.assertEqual(len(self.node.messages),count)

    def test_changed_query_map_measurement_or_token_discards_async_reply(self):
        for field,value in [('query_stamp_ns',22666000000),('map_stamp_ns',22583000000),
                            ('measured_q',(.01,)*7),('token','different')]:
            with self.subTest(field=field):
                self.setUp();_,ticket=self.reserve()
                self.assertEqual(self.policy.apply(ticket,replace(self.context,**{field:value}),self.candidate,self.node),
                                 'generation_context_changed')
                self.assertEqual(self.ledger.attempts,2);self.assert_live()
                self.assertEqual(self.node.messages,[])

    def test_preemption_during_generation_discards_result_and_keeps_budget(self):
        _,ticket=self.reserve();self.node._obligations.append(obligation(9))
        self.node._repair_stack.append(9);self.node._dependency_edges[9]=6
        self.assertEqual(self.policy.apply(ticket,self.context,self.candidate,self.node),'active_dependency_changed')
        self.node._obligations.pop();self.node._repair_stack.pop();self.node._dependency_edges.pop(9)
        self.assertEqual(self.reserve()[0],'family_already_attempted');self.assert_live()

    def test_new_wrapper_with_same_ledger_cannot_reset_failures(self):
        _,ticket=self.reserve();self.policy.reject(ticket,'timeout')
        self.policy=ReplacementSpec(replace(self.context,token='changed'),self.ledger)
        self.assertEqual(self.reserve()[0],'family_already_attempted')

    def test_stale_cycle_and_wrong_parent_never_reserve(self):
        c=self.context
        self.assertEqual(self.policy.reserve(c,[1,6],{6:1},(SHA,7),(2,c.query_stamp_ns,'old'))[0],
                         'no_current_ancestor_cycle')
        self.assertEqual(self.policy.reserve(c,[1,6],{6:3},(SHA,7),(2,c.query_stamp_ns,c.token))[0],
                         'no_current_ancestor_cycle')
        self.assertEqual(self.ledger.attempts,1)

    def test_changed_epoch_or_point_set_requires_new_explicit_context_not_budget_reset(self):
        for c in [replace(self.context,mode_epoch=3),replace(self.context,points_key=region_key([[0,0,.9]]))]:
            self.assertEqual(self.reserve(c)[0],'wrong_lineage')
        self.assertEqual(self.ledger.attempts,1)

    def test_duplicate_pose_under_another_head_is_rejected(self):
        self.ledger.attempted_poses.append(tuple(self.candidate['q_vis']))
        _,ticket=self.reserve()
        self.assertEqual(self.policy.apply(ticket,self.context,self.candidate,self.node),'pose_already_attempted')
        self.assertEqual(self.ledger.attempts,2);self.assert_live()

    def test_geometry_failure_nan_and_wrong_model_never_mutate_target(self):
        for change in [dict(geometry_accepted=False),dict(local_progress_evidence=False),
                       dict(q_vis=[float('nan')]*7),dict(checkpoint_sha='wrong')]:
            with self.subTest(change=change):
                self.setUp();_,ticket=self.reserve()
                self.assertNotEqual(self.policy.apply(ticket,self.context,{**self.candidate,**change},self.node),
                                    'steering_replaced_requires_existing_gates')
                self.assertEqual(self.node.messages,[]);self.assert_live()

    def test_rejected_timeout_and_abort_preserve_obligations_and_failure_memory(self):
        self.apply()
        for event in ['unsafe','timeout','aborted']:
            self.policy.outcome(event);self.assert_live()
            self.assertEqual(self.ledger.attempts,2)
        self.assertEqual(self.reserve()[0],'family_already_attempted')

    def test_case001_native_attempt_limit_covers_five_mount_groups(self):
        self.assertEqual((self.policy.max_attempts,self.policy.max_segments),(5,6))

    def test_budget_exhausted_while_generation_is_pending_does_not_apply(self):
        _,ticket=self.reserve();self.policy.inherit_progress(2,6)
        self.assertEqual(self.policy.apply(ticket,self.context,self.candidate,self.node),
                         'budget_exhausted_during_generation')
        self.assert_live();self.assertEqual(self.node.messages,[])

    def test_actual_measured_state_rechecked_even_if_caller_context_is_stale(self):
        _,ticket=self.reserve();self.node._latest_measured_q+=.02
        self.assertEqual(self.policy.apply(ticket,self.context,self.candidate,self.node),'measured_state_changed')
        self.assert_live();self.assertEqual(self.node.messages,[])

    def test_inconsistent_child_pin_is_not_silently_replaced(self):
        _,ticket=self.reserve();self.node._dependency_pins[6]['q_vis']+=.02
        self.assertEqual(self.policy.apply(ticket,self.context,self.candidate,self.node),'active_target_changed')
        self.assert_live();self.assertEqual(self.node.messages,[])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    protected=list((ROOT/'src/care_visibility_cdf/scripts').glob('*.py'))
    protected+=list((ROOT/'src/care_visibility_cdf/checkpoints').rglob('*.pt'))
    protected+=list((ROOT/'src/care_collision_cdf/checkpoints').rglob('*.pt'))
    protected+=[ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',
                ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/safe_frontier_recovery.hpp']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReplacementTests))
    text=stream.getvalue();print(text)
    (args.output_dir/'tests.txt').write_text(text)
    assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    evidence_sources=[INPUT/'snapshots.json',INPUT/'diagnostic.json',SEGMENT,Path(__file__),ROOT/'scripts/phase_e_candidate_replacement_spec.py']
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),qualification='OFFLINE_STATE_MACHINE_SPEC_ONLY',
        production_code_unchanged=True,model_weights_unchanged=True,source_sha256=hashes,
        fixture_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_sources},
        actual_data='exp74 22.635s O1/O6 identity, R1/S7 endpoint, prior local segment evidence',
        synthetic_events=['seen confidence responses','safe/unsafe/timeout/abort events','token/map changes and temporary preemption'],
        not_run=['production policy integration','full safety gate certification','ROS/Gazebo','actual seen/parent resume'])
    with (args.output_dir/'report.json').open('x') as f:json.dump(report,f,indent=2)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
