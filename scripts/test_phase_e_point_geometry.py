#!/usr/bin/env python3
import unittest
from diagnose_phase_e_point_geometry import primitive_sdf, row_support, coverage_audit
from test_phase_e_repair_conflict_diagnosis import fixture
from types import SimpleNamespace


class GeometryTests(unittest.TestCase):
    def test_box_face_corner_inside(self):
        self.assertAlmostEqual(primitive_sdf('box',[2,0,0],[2,2,2]),1)
        self.assertAlmostEqual(primitive_sdf('box',[2,2,1],[2,2,2]),2**.5)
        self.assertAlmostEqual(primitive_sdf('box',[0,0,0],[2,2,2]),-1)

    def test_cylinder_side_cap_corner_inside(self):
        self.assertAlmostEqual(primitive_sdf('cylinder',[2,0,0],[1,2]),1)
        self.assertAlmostEqual(primitive_sdf('cylinder',[0,0,2],[1,2]),1)
        self.assertAlmostEqual(primitive_sdf('cylinder',[2,0,2],[1,2]),2**.5)
        self.assertAlmostEqual(primitive_sdf('cylinder',[0,0,.5],[1,2]),-.5)

    def test_sphere_and_unsupported(self):
        self.assertAlmostEqual(primitive_sdf('sphere',[0,0,2],[1]),1)
        self.assertAlmostEqual(primitive_sdf('sphere',[0,0,0],[1]),-1)
        with self.assertRaises(ValueError): primitive_sdf('mesh',[0,0,0],[])

    def test_dynamics_can_tighten_box_support(self):
        d,_=fixture(); bounds=list(zip(d['x_l'],d['x_u']))
        self.assertAlmostEqual(row_support(d,0,bounds,False)['max_lhs'],.5)
        self.assertAlmostEqual(row_support(d,0,bounds,True)['max_lhs'],.05)

    def test_missing_and_wrong_type_coverage(self):
        box=type('Box',(),{})()
        robot=SimpleNamespace(links=[SimpleNamespace(name='l',collisions=[
            SimpleNamespace(name='a',geometry=box),SimpleNamespace(name='b',geometry=box)])])
        samples=[dict(link_name='l',samples=[dict(source_collision_index=0,source_type='cylinder')])]
        audit=coverage_audit(robot,samples)
        self.assertEqual([x['name'] for x in audit['missing_primitive_samples']],['b'])
        self.assertEqual([x['name'] for x in audit['source_type_mismatches']],['a'])


if __name__=='__main__': unittest.main(verbosity=2)
