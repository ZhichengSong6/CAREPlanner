"""Recorded receiver failures retain hard admission and local latency metadata."""
import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from test_phase_e_o5_certificate_handoff import O5CertificateTests,rospy
from handoff_schedule import state_variation_ready

class RecordedHandoffTests(O5CertificateTests):
    @classmethod
    def setUpClass(cls):
        cls.recorded=json.loads((ROOT/'outputs/phase_e_failure_mechanism_20261016/recorded_failures.json').read_text())

    def test_recorded_rejections_keep_bound_and_cache_observed_latency(self):
        for f in self.recorded:
            with self.subTest(audit=f['context']['trajectory_stamp_ns']):
                n,e,_=self.gate(f);d=f['diagnostics'];now=int(d['decision_ros_ns']);names=f['context']['joint_names']
                n._latest_measured_stamp_ns=int(d['live_stamp_ns']);n._latest_actuator_stamp_ns=now
                n._latest_measured_q_by_name=dict(zip(names,d['live_q']))
                n._latest_measured_velocity_by_name=dict(zip(names,d['live_velocity']))
                n._latest_actuator_velocity=dict(zip(names,d['actuator_velocity']))
                original=copy.deepcopy(e);ok,detail=self.check(n,e,now)
                self.assertFalse(ok);self.assertEqual(detail['failure_kind'],'geometric_certificate_state_changed')
                self.assertAlmostEqual(detail['motion_bound_m'],d['motion_bound_m'],places=13)
                self.assertEqual(e,original)
                cached=n._handoff_certificate[1]
                self.assertAlmostEqual(cached['handoff_latency_s'],(now-int(e['trajectory_stamp_ns']))*1e-9)
                self.assertNotIn('handoff_latency_s',e)

    def test_recorded_stopped_state_fits_latency_scheduling_hint(self):
        f=self.recorded[-1];e=copy.deepcopy(f['certificate']);e['handoff_latency_s']=.1
        s=json.loads((ROOT/'outputs/phase_e_failure_mechanism_20261016/stopped_state.json').read_text())
        # A frozen pair is only a scheduling control. It does not mint a proof.
        old=(s['stamp'],s['q'],s['v']);new=(s['stamp']+10000000,s['q'],s['v'])
        self.assertTrue(state_variation_ready(old,new,s['actuator'],e,f['context']['acceleration'],2.5))
        self.assertFalse(state_variation_ready(old,new,[.01]*7,e,f['context']['acceleration'],2.5))

def load_tests(loader,tests,pattern):
    return unittest.TestSuite(RecordedHandoffTests(name) for name in [
        'test_recorded_rejections_keep_bound_and_cache_observed_latency',
        'test_recorded_stopped_state_fits_latency_scheduling_hint'])

if __name__=='__main__':unittest.main()
