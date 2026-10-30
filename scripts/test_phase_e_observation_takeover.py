#!/usr/bin/env python3
"""Production regime callbacks with original recorded PROBE boundary fields."""
from pathlib import Path
import copy, json, unittest
import rospy
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String
import test_phase_e_execution_vbc_hold as base

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'outputs/phase_e_recurrence_analysis_20261016/probe_boundary_fixture.json'
TRACE=ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_002_r1_handoff_schedule_v2_002_20261016/run/projector_traces'

class TakeoverTests(unittest.TestCase):
    def setUp(self):
        self.transport=base.ExecutionHoldTests('test_unbound_summary_cannot_refresh_freshness')
        self.transport.setUp(); self.addCleanup(self.transport.doCleanups)
        self.m=self.transport.m; self.m.repair_completion_gate_enabled=True
        self.m.state=self.m.NORMAL
        self.fixture=json.loads(FIXTURE.read_text())
        self.fields={k:str(v) for k,v in self.fixture['first_new_obligation'].items() if k!='ros_s'}
        self.token=self.fields['active_q_vis_token']
        self.q=json.loads(next(TRACE.glob('*008*.json')).read_text())['q_vis']

    def active(self,value=True):self.m._visibility_waypoint_active_cb(Bool(data=value))
    def acquisition(self,**updates):
        f=dict(self.fields,**{k:str(v) for k,v in updates.items()})
        self.m._observation_acquisition_cb(String(data=' '.join(k+'='+v for k,v in f.items())))
    def waypoint(self,token=None,oid='8'):
        msg=Float64MultiArray(data=list(self.q))
        msg.layout.dim=[MultiArrayDimension(label=token or self.token,size=7,stride=7),
            MultiArrayDimension(label='care_region_v1_'+oid+'_'+'b'*24,size=7,stride=7)]
        self.m._observation_waypoint_cb(msg)
    def demand(self):self.active();self.waypoint();self.acquisition()
    def probe(self):
        m=self.m; b=self.fixture['regime_before'];m.state=m.PROBE_NORMAL
        m.pending_probe_execution_stamp_ns=int(b['pending_probe_execution_stamp_ns'])
        m.pending_probe_candidate_seq=int(b['pending_probe_candidate_seq'])
        m.pending_probe_effective_prefix_s=float(b['pending_probe_effective_prefix_s'])
        m.probe_completed_prefix_streak=int(b['probe_completed_prefix_streak'])
        m.probe_completed_prefixes_required=int(b['probe_completed_prefixes_required'])
    def complete_prefix(self,**updates):
        f={k:str(v) for k,v in self.fixture['tracker_completion'].items() if k!='ros_s'}
        f.update({k:str(v) for k,v in updates.items()})
        self.m._tracker_summary_cb(String(data=' '.join(k+'='+v for k,v in f.items())))

    def test_recorded_target_before_final_prefix(self):
        self.probe();self.demand();self.assertEqual(self.m.state,self.m.PROBE_NORMAL)
        self.complete_prefix();self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertEqual(self.m.observation_takeover_count,1)
        self.assertEqual(self.m.probe_completed_prefixes_required,3)

    def test_recorded_target_after_final_prefix(self):
        self.probe();self.complete_prefix();self.assertEqual(self.m.state,self.m.NORMAL)
        self.demand();self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertEqual(self.m.observation_takeover_token,self.token)
        self.assertEqual(self.m.task_stall_stamp_ns,0)

    def test_all_three_input_orders_enter_once(self):
        import itertools
        for order in itertools.permutations(['active','waypoint','acquisition']):
            self.m.state=self.m.NORMAL
            self.m.visibility_waypoint_active=False
            self.m._observation_waypoint=self.m._observation_demand=None
            n=self.m.observation_takeover_count
            for name in order:getattr(self,name)()
            self.assertEqual(self.m.state,self.m.REPAIR,order)
            self.demand();self.assertEqual(self.m.observation_takeover_count,n+1)

    def test_wrong_execution_does_not_end_single_flight(self):
        self.probe();self.demand()
        self.complete_prefix(execution_stamp_ns=self.m.pending_probe_execution_stamp_ns-1)
        self.assertEqual(self.m.state,self.m.PROBE_NORMAL)
        self.assertEqual(self.m.observation_takeover_count,0)

    def test_short_phase_does_not_end_single_flight(self):
        self.probe();self.demand();self.complete_prefix(phase_s=.001)
        self.assertEqual(self.m.state,self.m.PROBE_NORMAL)

    def test_first_completed_prefix_can_yield_but_does_not_count_three(self):
        self.probe();self.m.probe_completed_prefix_streak=0
        self.demand();self.complete_prefix()
        self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertEqual(self.m.probe_completed_execution_count,1)
        self.assertEqual(self.m.probe_completed_prefixes_required,3)

    def test_no_observation_still_requires_three_completed_prefixes(self):
        self.probe();self.m.probe_completed_prefix_streak=0
        self.complete_prefix();self.assertEqual(self.m.state,self.m.PROBE_NORMAL)
        self.assertEqual(self.m.probe_completed_prefix_streak,1)

    def test_waypoint_active_alone_does_not_enter(self):
        self.active();self.assertEqual(self.m.state,self.m.NORMAL)
        self.waypoint();self.assertEqual(self.m.state,self.m.NORMAL)

    def test_mismatched_snapshot_or_region_id_does_not_enter(self):
        self.active();self.acquisition();self.waypoint(oid='9')
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.waypoint(token='care_obs_v1_9_'+'a'*24,oid='9')
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.waypoint();self.assertEqual(self.m.state,self.m.REPAIR)

    def test_invalid_pose_or_duplicate_fields_does_not_enter(self):
        self.active();self.acquisition()
        self.m._observation_waypoint_cb(Float64MultiArray(data=[float('nan')]*7))
        self.assertEqual(self.m.state,self.m.NORMAL)
        self.m._observation_acquisition_cb(String(data='policy=visibility_acquisition complete=0 complete=1'))
        self.waypoint();self.assertEqual(self.m.state,self.m.NORMAL)

    def test_stale_receipts_wait_for_current_snapshot(self):
        self.active();self.waypoint();self.transport.now+=self.m.input_timeout+.001
        self.acquisition();self.assertEqual(self.m.state,self.m.NORMAL)
        self.waypoint();self.assertEqual(self.m.state,self.m.REPAIR)

    def test_closed_gate_resumes_same_current_demand(self):
        self.m.execution_ready=False;self.demand();self.assertEqual(self.m.state,self.m.NORMAL)
        self.m._execution_ready_cb(Bool(data=True));self.assertEqual(self.m.state,self.m.REPAIR)

    def test_falling_gate_invalidates_cached_target(self):
        self.active();self.waypoint()
        self.m._execution_ready_cb(Bool(data=False))
        self.m._execution_ready_cb(Bool(data=True))
        self.acquisition();self.assertEqual(self.m.state,self.m.NORMAL)

    def test_verifying_probe_waits_for_idle(self):
        self.m.state=self.m.PROBE_NORMAL
        self.m.probe_single_flight_phase='VERIFYING';self.demand()
        self.assertEqual(self.m.state,self.m.PROBE_NORMAL)
        self.m._probe_single_flight_summary_cb(String(data='phase=IDLE execution_stamp_ns=0 reason=done'))
        self.assertEqual(self.m.state,self.m.REPAIR)

    def test_actual_completion_retires_old_snapshot(self):
        self.demand();self.acquisition(complete=1,remaining_obligation_count=0)
        self.m.state=self.m.NORMAL;self.demand();self.assertEqual(self.m.state,self.m.NORMAL)
        self.assertEqual(self.m.observation_takeover_count,1)

    def test_old_completion_does_not_retire_new_pose_ahead_of_its_snapshot(self):
        self.demand();self.m.state=self.m.NORMAL
        new='care_obs_v1_9_'+'a'*24
        self.waypoint(token=new,oid='9')
        self.acquisition(complete=1,remaining_obligation_count=0)
        self.acquisition(active_q_vis_token=new,active_obligation_id=9)
        self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertNotIn(new,self.m._observation_retired_tokens)

    def test_completion_after_prefix_takeover_uses_same_pending_identity(self):
        self.probe();self.demand()
        self.m._repair_completion_cb(Bool(data=False))
        self.complete_prefix();self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertFalse(self.m.repair_completion_armed)
        self.acquisition(complete=1,remaining_obligation_count=0)
        self.assertEqual(self.m.state,self.m.PROBE_NORMAL)
        self.assertEqual(self.m.repair_completion_event_count,1)
        self.assertEqual(self.m.probe_completed_prefix_streak,0)
        self.assertEqual(self.m.probe_completed_prefixes_required,3)

    def test_completion_bool_before_or_after_summary_preserves_one_edge(self):
        for before in [True,False]:
            self.m.state=self.m.NORMAL;self.m._observation_retired_tokens.clear()
            self.demand();n=self.m.repair_completion_event_count
            if before:self.m._repair_completion_cb(Bool(data=True))
            self.acquisition(complete=1,remaining_obligation_count=0)
            self.m._repair_completion_cb(Bool(data=True))
            self.acquisition(complete=1,remaining_obligation_count=0)
            self.assertEqual(self.m.state,self.m.PROBE_NORMAL)
            self.assertEqual(self.m.repair_completion_event_count,n+1)

    def test_complete_without_authenticated_takeover_does_not_release(self):
        self.m.state=self.m.REPAIR;self.m.repair_completion_armed=False
        self.active();self.waypoint()
        self.acquisition(complete=1,remaining_obligation_count=0)
        self.m._repair_completion_cb(Bool(data=True))
        self.assertEqual(self.m.state,self.m.REPAIR)

    def test_old_complete_cannot_release_a_new_pose_ahead_of_summary(self):
        self.demand();new='care_obs_v1_9_'+'a'*24
        self.waypoint(token=new,oid='9')
        self.acquisition(complete=1,remaining_obligation_count=0)
        self.assertEqual(self.m.state,self.m.REPAIR)
        self.assertNotIn(new,self.m._observation_retired_tokens)

    def test_stale_or_unstarted_demand_cannot_supply_completion_edge(self):
        for stale in [True,False]:
            self.m.state=self.m.NORMAL;self.m._observation_retired_tokens.clear();self.demand()
            if stale:self.transport.now+=self.m.input_timeout+.001
            self.acquisition(complete=1,remaining_obligation_count=0,started=1 if stale else 0)
            self.assertEqual(self.m.state,self.m.REPAIR)

    def test_closed_gate_or_different_repair_reason_cannot_release(self):
        for ready in [False,True]:
            self.m.state=self.m.NORMAL;self.m.execution_ready=True
            self.m._observation_retired_tokens.clear();self.demand()
            if ready:self.m.last_transition_reason='execution_vbc_unsafe'
            else:self.m.execution_ready=False
            self.acquisition(complete=1,remaining_obligation_count=0)
            self.assertEqual(self.m.state,self.m.REPAIR)

if __name__=='__main__':unittest.main()
