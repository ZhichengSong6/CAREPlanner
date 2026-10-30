#!/usr/bin/env python3
"""Race, stale evidence and exact grid completeness tests for the shadow observer."""
import copy
from pathlib import Path
import threading
import time
import unittest
import numpy as np
from phase_e_stop_shadow import EvidenceBuffer,PrimitiveEnclosure,ShadowAudit,BLOCKERS
from phase_e_stop_envelope import JOINTS

ROOT=Path(__file__).resolve().parents[1]
CONFIG=dict(command_timeout=.1,max_braking_step=.02,acceleration_limits=[3,3,4,4,6,6,6],velocity_limits=[2,2,2,2,2.5,2.5,2.5])


def metadata():
    return dict(valid=True,reason='conditional_projection_no_future_lease',instance='one',epoch=0,revision=5,
        snapshot_ns=10_010_000_000,map_update_ns=10_000_000_000,frame='base_link',resolution=.05,
        map_min=[-.95,-.95,0.],map_max=[.95,.95,1.15],grid_size=[38,38,23],observation_mode='tof_ray',
        clock_fault=False,future_free_space_guaranteed=False,horizon_s=0.,temporal_decay_time=100.)


class Shape:
    def raster(self,q,travel,meta):
        return dict(points=np.array([[.1,0.,.3]]),grid_indices=[1],inside=True,primitive_bounds_m=[.1],point_set_sha256='test')


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.buffer=EvidenceBuffer(); self.now=10_010_000_000; self.calls=0; self.hook=lambda i,r:None
        self.state=dict(t_ns=10_000_000_000,stamp_ns=9_990_000_000,sequence=1,mode=1,clock_fault=False,
            integral_valid=True,max_update_gap=.001,q=[0.]*7,applied=[.1]*7,measured=[.1]*7,rearms=0)
        self.command=dict(stamp_ns=self.state['stamp_ns'],sequence=1,names=JOINTS,velocity=[.2]*7)
        self.buffer.on_state(self.state); self.buffer.on_command(self.command)
        def query(points,horizon,expected):
            self.calls+=1; r=metadata(); r['horizon_s']=horizon
            n=len(points)
            for k,value in [('inside_map',1),('confidence_now',1.),('confidence_at_horizon',1.),('occupancy',0.),
                            ('last_seen_s',9.),('provenance',1),('model_free_at_horizon',1)]: r[k]=[value]*n
            self.hook(self.calls,r)
            if expected and expected!=(r['instance'],r['epoch'],r['revision']):
                r.update(valid=False,reason='version_mismatch')
            return r
        self.audit=ShadowAudit(self.buffer,Shape(),query,lambda:self.now,CONFIG)

    def run_audit(self,reason=None):
        row=self.audit.run(); self.assertFalse(row['runtime_admission'])
        for item in BLOCKERS: self.assertIn(item,row['reasons'])
        if reason: self.assertIn(reason,row['reasons']); self.assertFalse(row['binding_consistent'])
        return row

    def test_coherent_is_still_not_admission(self):
        r=self.run_audit(); self.assertTrue(r['binding_consistent']); self.assertTrue(r['conditional_map_coverage'])

    def test_exact_missing_and_duplicate_conflict(self):
        self.buffer.commands.clear(); self.run_audit('missing_exact_command_identity')
        self.buffer.on_command(self.command)
        self.buffer.on_command(dict(self.command,velocity=[.3]*7)); self.run_audit('ambiguous_command_identity')

    def test_new_command_during_query(self):
        def hook(i,r):
            if i==2: self.buffer.on_state(dict(self.state,stamp_ns=10_001_000_000,sequence=2))
        self.hook=hook; self.run_audit('controller_command_changed_during_check')

    def test_pending_not_yet_applied(self):
        self.buffer.on_command(dict(self.command,stamp_ns=10_002_000_000,sequence=2))
        self.run_audit('newer_request_observed')

    def test_pending_arrives_during_query(self):
        def hook(i,r):
            if i==2: self.buffer.on_command(dict(self.command,stamp_ns=10_002_000_000,sequence=2))
        self.hook=hook; self.run_audit('newer_request_observed')

    def test_old_replay_does_not_extend_lease(self):
        self.buffer.on_command(dict(self.command,stamp_ns=9_980_000_000,sequence=0))
        self.assertTrue(self.run_audit()['binding_consistent'])

    def test_rearm_changes_contract(self):
        def hook(i,r):
            if i==2: self.buffer.on_state(dict(self.state,rearms=1))
        self.hook=hook; self.run_audit('controller_command_changed_during_check')

    def test_version_change_after_full_query(self):
        def hook(i,r):
            if i==3: r.update(valid=False,reason='version_mismatch',revision=6)
        self.hook=hook; self.run_audit('map_version_changed_during_check')

    def test_restart_instance_at_full_query(self):
        def hook(i,r):
            if i==2: r['instance']='two'
        self.hook=hook; self.run_audit('unexpected_snapshot_instance_or_epoch')

    def test_content_update_before_full_read_uses_new_revision(self):
        def hook(i,r):
            if i>=2: r['revision']=6
        self.hook=hook
        row=self.run_audit()
        self.assertTrue(row['binding_consistent'])
        self.assertEqual(row['metadata']['revision'],5)
        self.assertEqual(row['evidence']['revision'],6)
        self.assertEqual(row['version_recheck']['revision'],6)

    def test_epoch_change_before_full_read_rejected(self):
        def hook(i,r):
            if i==2: r['epoch']=1
        self.hook=hook; self.run_audit('unexpected_snapshot_instance_or_epoch')

    def test_grid_change_before_full_read_rejected(self):
        def hook(i,r):
            if i==2: r['resolution']=.1
        self.hook=hook; self.run_audit('map_geometry_changed')

    def test_content_revision_regression_rejected(self):
        def hook(i,r):
            if i==2: r['revision']=4
        self.hook=hook; self.run_audit('snapshot_revision_regressed')

    def test_stale_start(self):
        self.now+=200_000_000; self.run_audit('stale_or_future_measured_state')

    def test_state_ages_out_while_querying(self):
        def hook(i,r):
            if i==2: self.now+=200_000_000
        self.hook=hook; self.run_audit('state_aged_out_during_check')

    def test_clock_regression_and_integral_invalid(self):
        self.buffer.on_state(dict(self.state,t_ns=9_999_000_000)); self.run_audit('invalid_controller_time')

    def test_incomplete_arrays_and_untrusted_free_mask(self):
        def hook(i,r):
            if i==2: r['confidence_now']=[]
        self.hook=hook; self.run_audit('incomplete_snapshot_arrays')
        self.calls=0
        def low(i,r):
            if i==2: r['confidence_at_horizon']=[.2]
        self.hook=low; self.run_audit('unknown_occupied_or_decayed_points')

    def test_occupied_never_free(self):
        def hook(i,r):
            if i==2: r['occupancy']=[1.]
        self.hook=hook; self.run_audit('unknown_occupied_or_decayed_points')

    def test_future_snapshot_rejected(self):
        def hook(i,r):
            if i==2: r['snapshot_ns']+=1_000_000_000
        self.hook=hook; self.run_audit('future_snapshot_time')

    def test_out_of_map_not_clipped_to_clear(self):
        original=self.audit.geometry.raster
        self.audit.geometry.raster=lambda *a:dict(original(*a),inside=False)
        self.run_audit('enclosure_may_leave_map')

    def test_gap_assumption_violation(self):
        self.buffer.on_state(dict(self.state,max_update_gap=.003)); self.run_audit('observed_update_gap_exceeds_assumption')

    def test_braking_does_not_renew_remaining_lease(self):
        self.buffer.on_state(dict(self.state,mode=2))
        r=self.run_audit(); self.assertLess(r['budget']['command_stop_s'],.04)

    def test_rpc_timeout_is_bounded_and_late_result_unused(self):
        from phase_e_stop_shadow_node import BoundedQuery
        gate=threading.Event(); calls=[]
        def slow(value): calls.append(value); gate.wait(1.); return value
        q=BoundedQuery(slow,.02)
        with self.assertRaisesRegex(RuntimeError,'snapshot_service_timeout'): q('old')
        with self.assertRaisesRegex(RuntimeError,'snapshot_service_busy'): q('queued')
        self.assertEqual(calls,['old']); gate.set(); q.thread.join(1.)
        self.assertEqual(q('new'),'new'); self.assertEqual(calls,['old','new'])


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.geometry=PrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')

    def test_complete_grid_matches_independent_full_scan(self):
        from audit_case001_motion_prior import frame_placements,distance_matrix
        g=self.geometry
        for q,dq in [(np.zeros(7),np.zeros(7)),(np.array([.2,-.2,.25,-.2,.2,-.2,.2]),np.full(7,.02))]:
            actual=g.raster(q,dq,metadata()); full=np.zeros(g.model.nq); full[g.qindices]=q
            poses=frame_placements(g.model,g.data,g.shapes,full)
            distances=distance_matrix(g.points,g.shapes,poses)
            expected=np.flatnonzero((distances[:,g.active]<=.020+(g.radii@dq)[g.active]+1e-12).any(axis=1))
            self.assertEqual(actual['grid_indices'],expected.tolist()); self.assertGreater(len(expected),0)

    def test_resource_limit_rejects_without_truncation(self):
        m=metadata(); m.update(grid_size=[200,200,200],map_max=[20,20,20])
        with self.assertRaisesRegex(ValueError,'resource_limit'): self.geometry.raster([0.]*7,[0.]*7,m)

    def test_large_stop_enclosure_detects_map_exit(self):
        self.assertFalse(self.geometry.raster([0.]*7,[1.]*7,metadata())['inside'])


if __name__=='__main__': unittest.main(verbosity=2)
