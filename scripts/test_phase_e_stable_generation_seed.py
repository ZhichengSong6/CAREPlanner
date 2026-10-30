#!/usr/bin/env python3
"""Generation scheduling, bounded waiting and unchanged application guards."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from candidate_replacement_runtime import StableGenerationSeed
from test_phase_e_candidate_replacement_runtime import RuntimeReplacementTests

class StableSeedTests(unittest.TestCase):
    def samples(self,g,start=1.,q=0.,active=0):
        for i in range(7):
            t=start+i*.02;g.observe_measured(np.full(7,q),t)
            g.observe_tracker(f'active={active} execution_stamp_ns=1',t)
        return start+.12

    def test_active_execution_blocks_even_stationary_measured_hold(self):
        g=StableGenerationSeed();now=self.samples(g,active=1)
        ready,d=g.ready(now,1.);self.assertFalse(ready)
        self.assertEqual(d['reason'],'certified_execution_active')

    def test_only_post_grant_fresh_samples_establish_stability(self):
        g=StableGenerationSeed();now=self.samples(g)
        self.assertTrue(g.ready(now,1.)[0])
        self.assertFalse(g.ready(now,now-.05)[0])
        self.assertFalse(g.ready(now+.051,1.)[0])
        self.assertFalse(g.ready(now+.21,1.)[0])

    def test_motion_excursion_is_not_hidden_by_return_to_origin(self):
        g=StableGenerationSeed()
        for i in range(7):
            t=1.+i*.02;g.observe_measured(np.full(7,.02 if i==3 else 0.),t)
            g.observe_tracker('active=0 execution_stamp_ns=0',t)
        self.assertEqual(g.ready(1.12,1.)[1]['reason'],'measured_still_moving')

    def test_duplicates_backward_time_and_gaps_do_not_prove_stability(self):
        g=StableGenerationSeed()
        for _ in range(30):g.observe_measured(np.zeros(7),1.)
        g.observe_tracker('active=0 execution_stamp_ns=0',1.)
        self.assertFalse(g.ready(1.,1.)[0])
        self.samples(g,1.02);g.observe_measured(np.zeros(7),1.01)
        self.assertFalse(g.ready(1.14,1.)[0])
        g=StableGenerationSeed()
        for t in [1.,1.07,1.12]:g.observe_measured(np.zeros(7),t)
        g.observe_tracker('active=0 execution_stamp_ns=0',1.12)
        self.assertEqual(g.ready(1.12,1.)[1]['reason'],'measured_window_gap')

    def test_invalid_tracker_and_measurement_fail_closed(self):
        for text in ['active=0 active=0 execution_stamp_ns=0','active=x execution_stamp_ns=0','active=0']:
            g=StableGenerationSeed();now=self.samples(g);g.observe_tracker(text,now)
            self.assertFalse(g.ready(now,1.)[0])
        g=StableGenerationSeed();now=self.samples(g);g.observe_measured([float('nan')]*7,now+.01)
        self.assertFalse(g.ready(now+.01,1.)[0])

    def test_raw_production_tracker_prefix_is_parsed(self):
        import csv
        p=Path(__file__).resolve().parents[1]/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_024_r1_case024_owner_fallback_20261009/run/tracker_summary.csv'
        with p.open() as f:raw=next(csv.DictReader(f))['field.data']
        self.assertTrue(raw.startswith('TRACKER active=0 '))
        g=StableGenerationSeed();now=self.samples(g);g.observe_tracker(raw,now)
        self.assertTrue(g.ready(now,1.)[0])

    def test_suspended_trajectory_requires_explicit_hold_and_measured_settling(self):
        for source in ['external_vbc_hold', 'external_gcdf_hard_hold']:
            g=StableGenerationSeed();now=self.samples(g,active=1)
            g.observe_tracker(f'TRACKER active=1 execution_stamp_ns=1 source={source}',now)
            self.assertTrue(g.ready(now,1.)[0])
            g.observe_measured(np.full(7,.02),now+.01)
            self.assertEqual(g.ready(now+.01,1.)[1]['reason'],'measured_still_moving')
            g.observe_tracker('TRACKER active=1 execution_stamp_ns=1 source=active_trajectory',now+.01)
            self.assertEqual(g.ready(now+.01,1.)[1]['reason'],'certified_execution_active')

    def test_recorded_v2_safety_hold_settles_before_original_deadline(self):
        import csv
        p=Path(__file__).resolve().parents[1]/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_024_r1_case024_stable_seed_20261009_v2/run'
        timeline=[]
        for name,kind in [('joint_states',0),('tracker_summary',1)]:
            with (p/(name+'.csv')).open() as f:
                for r in csv.DictReader(f):
                    t=int(r['%time'])/1e9
                    if 16.5<=t<=18.615:timeline.append((t,kind,r))
        g=StableGenerationSeed();first=None
        for t,kind,r in sorted(timeline,key=lambda row:row[:2]):
            if kind==0:g.observe_measured([float(r['field.position'+str(j)]) for j in range(7)],int(r['field.header.stamp'])/1e9)
            else:g.observe_tracker(r['field.data'],t)
            ready,detail=g.ready(t,16.615)
            if ready:first=(t,detail);break
        self.assertIsNotNone(first)
        self.assertLess(first[0],18.615)
        self.assertTrue(first[1]['tracker']['active'])
        self.assertEqual(first[1]['tracker']['source'],'external_vbc_hold')

    def fixture(self):
        h=RuntimeReplacementTests();data=h.fixture()
        n,r,e,s,events,clock,helper=data;r.offer(e)
        r.receive('request_id=1 granted=1 mode_epoch=3 query_stamp_ns=1000000000 attempts=2 segments=3 deadline_ns=2000000000')
        r.seed_gate.observe_commit_ack('request_id=1 confirmed=1 mode_epoch=3 query_stamp_ns=1000000000 deadline_ns=2000000000 last_execution_stamp_ns=0', clock.now)
        return data

    def feed(self,data,t,q,active):
        n,r,e,s,events,clock,helper=data;clock.now=t
        n._trace_measured=(np.full(7,q),t)
        r.seed_gate.observe_measured(n._trace_measured[0],t)
        r.seed_gate.observe_tracker(f'active={active} execution_stamp_ns=99',t)

    def test_wait_keeps_one_reservation_then_uses_latest_settled_seed(self):
        data=self.fixture();n,r,e,s,events,clock,helper=data
        calls=[];original=n._generate_active_set_waypoint
        def generate(*a):calls.append(n._seed_override.copy());return original(*a)
        n._generate_active_set_waypoint=generate
        for i in range(8):self.feed(data,1.+i*.02,i*.003,1);r.process()
        self.assertEqual(len(calls),0);self.assertEqual(len(s),1)
        # A duplicate granted response cannot reset the stable window/deadline.
        r.receive('request_id=1 granted=1 mode_epoch=3 query_stamp_ns=1000000000 attempts=2 segments=3 deadline_ns=99000000000')
        for i in range(9):
            self.feed(data,1.16+i*.02,.03,0);r.process()
        self.assertEqual(len(calls),1);np.testing.assert_equal(calls[0],np.full(7,.03))
        self.assertEqual(len(s),2);self.assertIn('action=finish',s[-1])
        self.assertIsNone(r.pending)
        self.assertEqual(events[-1]['grant']['attempts'],'2')
        self.assertEqual(events[-1]['grant']['deadline_ns'],'2000000000')

    def test_no_commit_confirmation_cannot_generate_even_when_stationary(self):
        data=self.fixture();n,r,e,s,events,clock,helper=data
        r.seed_gate.commit_ack=None
        n._generate_active_set_waypoint=lambda *a:self.fail('unconfirmed generation')
        for i in range(7):self.feed(data,1.+i*.02,0.,0)
        r.process();self.assertEqual(events[-1]['reason'],'waiting_for_commit_confirmation')
        clock.now=2.;r.process()
        self.assertEqual(events[-1]['event'],'generation_expired')
        self.assertEqual(len(s),2)

    def test_ack_starts_new_stability_window_and_requires_tracker_receipt(self):
        data=self.fixture();n,r,e,s,events,clock,helper=data
        r.seed_gate.commit_ack=None
        for i in range(7):self.feed(data,1.+i*.02,0.,0)
        r.seed_gate.observe_commit_ack('request_id=1 confirmed=1 mode_epoch=3 query_stamp_ns=1000000000 deadline_ns=2000000000 last_execution_stamp_ns=777',1.12)
        r.process();self.assertIsNotNone(r.pending)
        r.seed_gate.observe_tracker('active=0 execution_stamp_ns=777',1.12)
        r.process();self.assertIsNotNone(r.pending)
        for i in range(1,7):self.feed(data,1.12+i*.02,0.,0)
        r.process();self.assertEqual(events[-1]['event'],'applied_requires_existing_gates')

    def test_moving_until_deadline_finishes_without_inference_or_new_request(self):
        data=self.fixture();n,r,e,s,events,clock,helper=data
        n._generate_active_set_waypoint=lambda *a:self.fail('expired inference')
        for t in [1.,1.2,1.5,1.9,2.]:self.feed(data,t,t,1);r.process()
        self.assertIsNone(r.pending);self.assertEqual(len(s),2)
        self.assertEqual(events[-1]['event'],'generation_expired')
        self.assertEqual(events[-1]['stage'],'waiting_for_stable_seed')
        self.assertEqual(next(iter(r.families.values())),set())

    def test_paused_ros_clock_still_obeys_wall_deadline(self):
        data=self.fixture();n,r,e,s,events,clock,helper=data
        offered=r.pending['offered_wall']
        with patch('candidate_replacement_runtime.time.monotonic',return_value=offered+2.51):r.process()
        self.assertIsNone(r.pending);self.assertEqual(events[-1]['event'],'generation_expired')

    def test_context_change_while_waiting_aborts(self):
        data=self.fixture();n,r,e,s,events,clock,helper=data
        self.feed(data,1.,0.,1);r.process();n._repair_stack.append(3);r.process()
        self.assertEqual(events[-1]['event'],'context_changed');self.assertIsNone(r.pending)

    def test_joint_callback_can_update_during_inference_and_application_rejects_motion(self):
        import threading
        data=self.fixture();n,r,e,s,events,clock,helper=data
        def generate(*a):
            # Observer uses a different lock from process's inference lock.
            t=threading.Thread(target=lambda:self.feed(data,1.13,.02,1));t.start();t.join(1)
            self.assertFalse(t.is_alive())
            return dict(q_zero=np.zeros(7),per_sensor_hybrid=dict(accepted=True,
                selected_sensor_id=6,selected_sensor_frame='EE_sensor1_tof_link',selected_rank=1,
                selected_q_vis=[.4]*7,attempts=[dict(sensor_id=6)]))
        n._generate_active_set_waypoint=generate
        for i in range(7):self.feed(data,1.+i*.02,0.,0)
        r.process()
        self.assertEqual(events[-1]['event'],'measured_state_changed')
        self.assertAlmostEqual(events[-1]['seed_delta_inf'],.02)
        self.assertIn(6,next(iter(r.families.values())))

if __name__=='__main__':unittest.main()
