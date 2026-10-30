"""Pure regression tests for empty streams, frozen timestamps and render env."""
import unittest
import os
from pathlib import Path
import subprocess
from phase_e_sim_environment import simulation_environment
from wait_for_phase_e_perception import PerceptionReadiness


class ReadinessTests(unittest.TestCase):
    def healthy(self):
        h = PerceptionReadiness(['a','b'])
        for topic in h.topics:
            h.cloud(topic, 1, True, 0.)
            h.cloud(topic, 2, True, .1)
        for i in (1,2):
            h.summary('fusion', f'callbacks_received={i} callbacks_processed={i} '
                      'configured_sensors=2 active_sensors=2 ray_observations=5', i*.1)
            h.summary('map', f'ray_packet_count={i} last_ray_count=5 last_free_cell_count=2', i*.1)
        return h

    def test_real_no_hit_rays_need_no_occupied_points(self):
        self.assertTrue(self.healthy().snapshot(.2)['ready'])

    def test_empty_packets_do_not_pass(self):
        h=self.healthy()
        h.summary('fusion', 'callbacks_received=0 callbacks_processed=0 '
                  'configured_sensors=2 active_sensors=0 ray_observations=0', .3)
        h.summary('map', 'ray_packet_count=999 last_ray_count=0 last_free_cell_count=0', .3)
        self.assertFalse(h.snapshot(.3)['ready'])

    def test_one_missing_camera_fails(self):
        h=self.healthy();h.clouds['b']['advances']=0
        self.assertFalse(h.snapshot(.2)['ready'])

    def test_repeated_cloud_stamp_is_not_fresh(self):
        h=self.healthy()
        h.cloud('a', 2, True, 2.)
        self.assertFalse(h.snapshot(2.,raw_only=True)['ready'])

    def test_malformed_cloud_does_not_advance(self):
        h=PerceptionReadiness(['a'])
        h.cloud('a',1,False,0.);h.cloud('a',2,False,.1)
        self.assertFalse(h.snapshot(.1,True)['ready'])

    def test_stale_or_static_summaries_fail(self):
        h=self.healthy()
        self.assertFalse(h.snapshot(2.)['ready'])
        h=self.healthy();h.summaries['map']['latest']=h.summaries['map']['first']
        self.assertFalse(h.snapshot(.2)['ready'])

    def test_invalid_summary_fails(self):
        h=self.healthy();h.summary('map','ray_packet_count=nan',.3)
        self.assertFalse(h.snapshot(.3)['ready'])

    def test_render_environment_retained_without_diagnostic_overrides(self):
        inherited=dict(HOME='/tmp/example',DISPLAY=':7',XAUTHORITY='/tmp/auth',
                       XDG_RUNTIME_DIR='/tmp/session',PER_SENSOR_MAX_BRANCH_ATTEMPTS='999')
        env=simulation_environment({'PER_SENSOR_MAX_BRANCH_ATTEMPTS':'4'},inherited)
        for k in ('DISPLAY','XAUTHORITY','XDG_RUNTIME_DIR'):
            self.assertEqual(env[k],inherited[k])
        self.assertEqual(env['PER_SENSOR_MAX_BRANCH_ATTEMPTS'],'4')
        self.assertEqual(env['CONDA_SHLVL'],'0')

    def test_missing_display_fails_before_launch(self):
        with self.assertRaisesRegex(RuntimeError,'DISPLAY'):
            simulation_environment({}, {})

    def test_shell_entries_reject_missing_display_without_starting_ros(self):
        repo=Path(__file__).resolve().parents[1]
        env=dict(os.environ,REQUIRE_REAL_TOF_READINESS='true')
        env.pop('DISPLAY',None)
        for name in ('run_phase_c4_4_verified_regime_smoke.sh',
                     'run_phase_e_empty_world_qualification.sh'):
            result=subprocess.run(['bash',str(repo/'scripts'/name)],env=env,cwd=repo,
                                  capture_output=True,text=True,timeout=3)
            self.assertEqual(result.returncode,2)
            self.assertIn('Missing DISPLAY',result.stderr)


if __name__=='__main__': unittest.main(verbosity=2)
