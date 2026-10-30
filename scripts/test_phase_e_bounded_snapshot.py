#!/usr/bin/env python3
"""Actual RPC wrapper: wait caps, late discard and one outstanding request."""
import threading
import unittest
from phase_e_stop_shadow_node import BoundedQuery


class BoundedSnapshotTests(unittest.TestCase):
    def test_timed_out_call_blocks_overlap_and_late_answer_is_discarded(self):
        release=threading.Event();calls=[]
        def call(value):
            calls.append(value)
            if value=='late':release.wait(1.)
            return value
        query=BoundedQuery(call,.5)
        try:
            with self.assertRaisesRegex(RuntimeError,'snapshot_service_timeout'):
                query.call_with_timeout(.001,'late')
            with self.assertRaisesRegex(RuntimeError,'busy_after_timeout'):
                query('overlap')
            self.assertEqual(calls,['late'])
        finally:
            release.set();query.thread.join(1.)
        self.assertEqual(query('new'),'new')
        self.assertEqual(calls,['late','new'])

    def test_per_call_cap_cannot_extend_configured_timeout(self):
        release=threading.Event();query=BoundedQuery(lambda:release.wait(1.),.001)
        try:
            with self.assertRaisesRegex(RuntimeError,'snapshot_service_timeout'):
                query.call_with_timeout(.5)
        finally:release.set();query.thread.join(1.)

    def test_zero_negative_and_nonfinite_budget_do_not_start_rpc(self):
        calls=[];query=BoundedQuery(lambda:calls.append(1),1.)
        for value in [0.,-.1,float('nan'),float('inf')]:
            with self.assertRaisesRegex(RuntimeError,'snapshot_service_timeout'):
                query.call_with_timeout(value)
        self.assertEqual(calls,[])

    def test_service_exception_is_explicit(self):
        def call():raise ValueError('fixture_service_failure')
        with self.assertRaisesRegex(RuntimeError,'snapshot_service_error: fixture_service_failure'):
            BoundedQuery(call,.5)()


if __name__=='__main__':unittest.main()
