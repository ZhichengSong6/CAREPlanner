#!/usr/bin/env python3
"""Current confidence previews alter ranking only, within a finite query cap."""
from pathlib import Path
from types import SimpleNamespace
import math,sys,threading,time,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
from observation_corridor_cost import BoundedCorridorQuery,body_path_probes,snapshot_cost
from observation_task_rejoin import task_rejoin_context,candidate_cost,improves_candidate
from validate_visibility_oracle import URDF,find_chain_joints
from check_visibility_self_occlusion import load_collision_primitives


class CorridorTests(unittest.TestCase):
    def snapshot(self,n):
        return SimpleNamespace(valid=True,clock_fault=False,frame='base_link',instance='map',epoch=1,revision=7,
            resolution=.05,snapshot_stamp=SimpleNamespace(to_nsec=lambda:1000000000),
            inside_map=bytes([1]*n),confidence_now=[1.]*n,occupancy=[0.]*n)

    def test_snapshot_distinguishes_unknown_occupied_outside_and_known(self):
        points=[(float(i),0.,0.) for i in range(4)];s=self.snapshot(4)
        s.confidence_now=[0.,1.,1.,1.];s.occupancy=[0.,1.,0.,0.];s.inside_map=bytes([1,1,0,1])
        d=snapshot_cost(s,points,points[:2],points[2:],1010000000,.6)
        self.assertEqual(d['status'],'valid')
        self.assertEqual((d['approach']['unknown'],d['approach']['occupied']),(1,1))
        self.assertEqual(d['rejoin']['outside'],1)
        self.assertEqual(d['approach']['risk_fraction'],1.)
        self.assertEqual(d['rejoin']['risk_fraction'],.5)

    def test_stale_clock_fault_and_invalid_arrays_cannot_supply_a_discount(self):
        p=[(0.,0.,0.)]
        for attr,value in [('clock_fault',True),('valid',False),('frame','world'),
                           ('confidence_now',[float('nan')]),('inside_map',b''),('resolution',.1)]:
            s=self.snapshot(1);setattr(s,attr,value)
            self.assertNotEqual(snapshot_cost(s,p,p,p,1010000000,.6)['status'],'valid')
        self.assertNotEqual(snapshot_cost(self.snapshot(1),p,p,p,1050000001,.6)['status'],'valid')

    def test_timeout_does_not_queue_or_start_another_optional_rpc(self):
        entered=threading.Event();release=threading.Event();calls=[]
        def slow(request):calls.append(request);entered.set();release.wait(.5);return request
        q=BoundedCorridorQuery(slow,.01);start=time.monotonic()
        try:
            self.assertIsNone(q('first'));self.assertTrue(entered.is_set())
            self.assertIsNone(q('second'));self.assertEqual(calls,['first'])
            self.assertLess(time.monotonic()-start,.2)
        finally:release.set()

    def test_actual_urdf_probes_are_finite_bounded_and_cover_whole_active_body(self):
        robot=URDF.from_xml_file(str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'))
        primitives=load_collision_primitives(robot)
        runtime=SimpleNamespace(primitives=primitives,collision_chains=[find_chain_joints(robot,'base_link',p['link']) for p in primitives])
        p=body_path_probes(runtime,np.zeros(7),np.array([.2,.7,.2,-1.,0.,0.,0.]))
        self.assertTrue(p);self.assertLessEqual(len(p),96)
        self.assertEqual(len(p),len(set(p)))
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertGreater(max(x[2] for x in p)-min(x[2] for x in p),.4)

    def test_unknown_return_proxy_can_outweigh_a_shorter_joint_route(self):
        def preview(risk):return dict(status='valid',approach=dict(risk_fraction=risk),rejoin=dict(risk_fraction=risk),map_identity=['map',1])
        c=task_rejoin_context(np.zeros(7),np.ones(7),np.ones(7),np.ones(7))
        c['corridor_evaluator']=lambda q,ctx:preview(1. if q[0]==0. else 0.)
        pose=np.eye(4);baseline=candidate_cost(np.zeros(7),c,pose,pose)
        candidate=candidate_cost(np.full(7,.25),c,pose,pose)
        self.assertTrue(improves_candidate(candidate,baseline,.02))
        self.assertGreater(candidate['approach_s']+candidate['rejoin_s'],
                           baseline['approach_s']+baseline['rejoin_s'])

    def test_missing_or_foreign_snapshot_is_never_treated_as_a_clear_corridor(self):
        base=dict(approach_s=1.,rejoin_s=1.,total_s=4.,task_pose_error=1.,corridor_used=True,
                  corridor_preview=dict(map_identity=['map',1]))
        missing=dict(base,total_s=1.,corridor_used=False)
        self.assertFalse(improves_candidate(missing,base,0.))
        foreign=dict(base,total_s=1.,corridor_preview=dict(map_identity=['other',1]))
        self.assertFalse(improves_candidate(foreign,base,0.))

    def test_motion_and_extra_compute_use_wall_seconds_when_clock_rate_is_known(self):
        c=task_rejoin_context(np.zeros(7),np.ones(7),np.ones(7),np.ones(7));pose=np.eye(4)
        nominal=candidate_cost(np.full(7,.25),c,pose,pose)
        c['ros_per_wall_s']=.5;slow=candidate_cost(np.full(7,.25),c,pose,pose)
        self.assertAlmostEqual(slow['total_s'],2.*nominal['total_s'])


if __name__=='__main__':unittest.main(verbosity=2)
