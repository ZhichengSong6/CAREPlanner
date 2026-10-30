#!/usr/bin/env python3
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import phase_e_observation_readiness as obs
import phase_e_benchmark_arm as arm


class Tests(unittest.TestCase):
    def check(self, fault=None):
        clock = NS(now=0., tick=0)
        recorders = ['/rec_'+str(i) for i in range(len(obs.RECORDED))]
        subs = [(obs.PREFIX+t, [n]) for n, t in zip(recorders, obs.RECORDED)]
        edges = [(n, obs.PREFIX+t, live) for n, t, live in obs.EDGES]
        edges += [(n, t, True) for t, ns in subs for n in ns]
        rospy = NS(get_name=lambda: '/watcher')
        checker = obs.ObservationReadiness(recorders, rospy, 'master')

        def rpc(uri, method, deadline, *args):
            if fault == 'rpc_timeout':
                raise TimeoutError('receiver timed out')
            if method == 'lookupNode':
                return args[-1] + ('_reboot' if fault == 'restart' and clock.tick else '')
            rows = [(i+1, t, live) for i, (n, t, live) in enumerate(edges) if n == uri]
            peer = '/wrong_generator' if fault == 'wrong_peer' else obs.GENERATOR
            if method == 'getBusInfo':
                return [[i, peer, 'i', 'TCPROS', t, fault != 'disconnect'] for i, t, _ in rows]
            counts = []
            for i, t, live in rows:
                count = (1+clock.tick if live else 1)
                if fault == 'zero_q' and t.endswith('/visibility_waypoint_q'):
                    count = 0
                if fault == 'stagnant':
                    count = 1
                if fault == 'silent_recorder' and uri == recorders[0]:
                    count = 0
                counts.append([t, [[i, 100, count, 0, True]]])
            return [[], counts, []]

        with patch.object(obs, 'rpc', rpc), patch.object(obs.time, 'monotonic', lambda: clock.now):
            self.assertFalse(checker.check(20., subs))
            clock.tick = 1
            clock.now = .1 if fault != 'deadline' else 20.
            return checker.check(20., subs), checker.last

    def test_receipt_and_advancing_periodic_counts_pass(self):
        self.assertTrue(self.check()[0])

    def test_no_target_needed_but_q_transport_receipt_is_required(self):
        self.assertFalse(self.check('zero_q')[0])

    def test_registered_silent_recorder_cannot_arm(self):
        self.assertFalse(self.check('silent_recorder')[0])

    def test_stale_counts_cannot_arm(self):
        self.assertFalse(self.check('stagnant')[0])

    def test_peer_identity_disconnect_and_restart_cannot_arm(self):
        for mode in ('wrong_peer', 'disconnect', 'restart', 'rpc_timeout'):
            with self.subTest(mode=mode):
                self.assertFalse(self.check(mode)[0])

    def test_deadline_cannot_be_extended_by_rpc(self):
        self.assertFalse(self.check('deadline')[0])

    def test_missing_required_recorder_fails(self):
        checker = obs.ObservationReadiness([], NS(get_name=lambda: '/watcher'), 'master')
        self.assertFalse(checker.check(obs.time.monotonic()+1, []))
        self.assertIn('missing_observation_recorder', checker.last['reason'])

    def test_arm_does_not_publish_or_extend_deadline_for_missing_observation(self):
        clock, sent = NS(now=0.), []
        class Pub:
            resolved_name = arm.ARM_TOPIC
            def get_num_connections(self): return 1
            def publish(self, msg): sent.append(msg)
        checker = NS(check=lambda *a: False, last={'reason': 'observation_missing'})
        rospy = NS(Publisher=lambda *a, **k: Pub(), get_name=lambda: '/watcher',
                   is_shutdown=lambda: False, get_param=lambda key, default=None:
                   True if key.endswith('/observation_readiness_required') else default)
        master = NS(master_uri='master', getSystemState=lambda: ([], [('/data', ['/rec'])], []))
        with patch.dict(sys.modules, {'rospy': rospy, 'rosgraph': NS(Master=lambda _: master),
                                     'std_msgs.msg': NS(String=object)}), \
             patch.object(obs, 'ObservationReadiness', lambda *a: checker), \
             patch.object(arm.time, 'monotonic', lambda: clock.now), \
             patch.object(arm.time, 'sleep', lambda dt: setattr(clock, 'now', clock.now+dt)):
            with self.assertRaisesRegex(RuntimeError, 'observation_missing'):
                arm.prepare_arm(['/rec'], timeout_s=.2, admission_ping_topic='')
        self.assertLessEqual(clock.now, .2)
        self.assertEqual(sent, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
