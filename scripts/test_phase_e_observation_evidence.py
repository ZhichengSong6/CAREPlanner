#!/usr/bin/env python3
import math
import unittest
import numpy as np
from phase_e_observation_evidence import (sensor_expiry_s,observation_window,local_samples,
    raster_indices,point_coverage,matched_update)


class ObservationTests(unittest.TestCase):
    def test_time_constant_is_not_valid_memory_duration(self):
        end=sensor_expiry_s(10.,100.)
        self.assertAlmostEqual(end,32.31435513142097)
        self.assertGreater(math.exp(-(end-10.-.001)/100),.8)
        self.assertLess(math.exp(-(end-10.+.001)/100),.8)

    def test_reobservation_deadline_includes_the_complete_stop(self):
        e=dict(snapshot_ns=31000000000,temporal_decay_time=100.,provenance=[1,2],
            last_seen_s=[10.,-1.],confidence_at_horizon=[.801,1.],model_free_at_horizon=[1,1])
        window=observation_window(e,32.)
        self.assertEqual(window['action'],'request_reobservation')
        self.assertAlmostEqual(window['remaining_after_required_stop_s'],.31435513142097)
        self.assertEqual(window['frozen_bootstrap_points'],1)
        self.assertFalse(window['runtime_admission'])

    def test_unknown_and_oracle_never_create_an_observation_window(self):
        for p in [0,4,5]:
            e=dict(snapshot_ns=10000000000,temporal_decay_time=100.,provenance=[p],
                last_seen_s=[-1.],confidence_at_horizon=[1.],model_free_at_horizon=[1])
            self.assertEqual(observation_window(e,11.)['action'],'stop_missing_evidence')

    def test_stale_and_future_observations_are_distinct(self):
        e=dict(snapshot_ns=40000000000,temporal_decay_time=100.,provenance=[1],
            last_seen_s=[10.],confidence_at_horizon=[.7],model_free_at_horizon=[0])
        self.assertEqual(observation_window(e,41.)['action'],'stop_missing_evidence')
        e['last_seen_s']=[41.]
        with self.assertRaisesRegex(ValueError,'future_sensor'):observation_window(e,41.)

    def test_fov_and_raster_are_separate_tests(self):
        m=dict(min_range=.15,max_range=.75,local_resolution=.05,forward_axis='z',
            horizontal_fov_deg=55.,vertical_fov_deg=72.,map_min=[-.95,-.95,0.],
            resolution=.05,grid_size=[38,38,23])
        t=dict(translation=[0.,0.,0.],quaternion_xyzw=[0.,0.,0.,1.])
        samples=local_samples(m);self.assertEqual(len(samples),1681)
        c=point_coverage([[0,0,.3],[0,0,.1],[0,0,.8],[.5,0,.3]],m,t)
        np.testing.assert_array_equal(c['visible'],[1,0,0,0])
        np.testing.assert_array_equal(c['too_near'],[0,1,0,0])
        indices=raster_indices(m,t)
        self.assertIn(19*38*23+19*23+6,indices)

    def test_trace_requires_the_actual_instance_revision_and_update_stamp(self):
        e=dict(instance='map',epoch=0,revision=7,map_update_ns=10,snapshot_ns=11)
        update=dict(e,observation_ns=10)
        self.assertIs(matched_update(e,dict(instance='map'),{(0,7):update}),update)
        for changed in [dict(e,revision=8),dict(e,map_update_ns=9),dict(e,instance='other')]:
            with self.assertRaisesRegex(ValueError,'matching_map_update'):matched_update(changed,dict(instance='map'),{(0,7):update})

if __name__=='__main__':unittest.main()
