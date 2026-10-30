#!/usr/bin/env python3
"""Production manager state transitions with a deterministic transport clock."""
import ast
from collections import deque
import math
from pathlib import Path
import re
import threading
from types import SimpleNamespace as NS, MethodType
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py'
NAMES = {'_retire_qp_retry_locked','_pump_qp_retry_locked','_qp_retry_ack_cb','_qp_retry_ready_cb','_task_stall_cb','_execution_ready_cb',
         '_maybe_enter_observation_repair_locked'}


class Tests(unittest.TestCase):
    def setUp(self):
        self.clock = NS(t=0.)
        self.sent = []; self.ready = []
        cls = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef))
        nodes = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in NAMES]
        self.assertEqual(len(nodes),len(NAMES))
        scope = dict(math=math,re=re,String=lambda **kw:NS(**kw),Bool=lambda **kw:NS(**kw),
            time=NS(monotonic=lambda:self.clock.t),rospy=NS(Time=NS(now=lambda:self.clock.t)),
            _tokens=lambda s:dict(w.split('=',1) for w in s.split() if '=' in w),_as_int=lambda v,d=0:int(v) if v else d)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),scope)
        self.m = NS(_lock=threading.RLock(),NORMAL='NORMAL',PROBE_NORMAL='PROBE_NORMAL',state='NORMAL',
            execution_ready=False,task_stall_stamp_ns=0,qp_retry_token='',qp_retry_mode='',qp_retry_last_send=-math.inf,
            qp_retry_retired=deque(maxlen=64),qp_retry_send_count=0,qp_retry_ack_count=0,qp_retry_last_status='none',
            qp_retry_ready_nonce='',qp_retry_ready_time=-math.inf,
            qp_retry_pub=NS(publish=lambda msg:self.sent.append(msg.data)),
            qp_retry_ready_pub=NS(publish=lambda msg:self.ready.append(msg.data)),
            _probe_single_flight_busy_locked=lambda:False,
            task_obstacle_blocked_pending=False,task_infeasible_pending=False,task_uncertified_pending=False,
            candidate_outcome_pending_gate_replay=False)
        for n in NAMES: setattr(self.m,n,MethodType(scope[n],self.m))

    def retry(self, token='one', mode='NORMAL'):
        self.m._task_stall_cb(NS(data=f'status=retry token={token} mode={mode}'))

    def test_gate_and_late_connection_keep_same_ticket(self):
        self.retry(); self.assertFalse(self.sent)
        self.m._execution_ready_cb(NS(data=True))
        for i in range(4):
            self.clock.t += .25; self.m._pump_qp_retry_locked()
        self.assertEqual(self.sent,['one']*5)
        self.assertFalse(self.m.task_uncertified_pending)

    def test_bad_ack_does_not_clear_then_matching_ack_stops_replays(self):
        self.m.execution_ready=True; self.retry()
        for data in ['token=wrong status=accepted','token=one status=garbage']:
            self.m._qp_retry_ack_cb(NS(data=data)); self.assertEqual(self.m.qp_retry_token,'one')
        self.m._qp_retry_ack_cb(NS(data='token=one status=duplicate'))
        self.clock.t=1.; self.retry(); self.m._pump_qp_retry_locked()
        self.assertEqual(self.sent,['one']); self.assertEqual(self.m.qp_retry_ack_count,1)

    def test_newer_ticket_rejects_old_ack(self):
        self.m.execution_ready=True; self.retry('one'); self.retry('two')
        self.m._qp_retry_ack_cb(NS(data='token=one status=accepted'))
        self.assertEqual(self.m.qp_retry_token,'two')
        self.retry('one'); self.assertEqual(self.m.qp_retry_token,'two')

    def test_reset_and_mode_mismatch_do_not_replay(self):
        self.retry(); self.m._task_stall_cb(NS(data='status=reset reason=fresh_reference'))
        self.m.execution_ready=True; self.retry(); self.retry('two','PROBE_NORMAL')
        self.assertFalse(self.sent)

    def test_probe_busy_retains_ticket_without_preempting_flight(self):
        self.m.state='PROBE_NORMAL'; self.m.execution_ready=True
        self.m._probe_single_flight_busy_locked=lambda:True
        self.retry('one','PROBE_NORMAL'); self.assertFalse(self.sent)
        self.m._probe_single_flight_busy_locked=lambda:False
        self.m._pump_qp_retry_locked(); self.assertEqual(self.sent,['one'])

    def test_readiness_ack_identity_and_age(self):
        nonce='a'*32; self.m._qp_retry_ready_cb(NS(data=nonce))
        self.m._qp_retry_ack_cb(NS(data='token=ready_'+('b'*32)+' status=ready'))
        self.assertFalse(self.ready)
        self.clock.t=1.; self.m._qp_retry_ack_cb(NS(data='token=ready_'+nonce+' status=ready'))
        self.assertFalse(self.ready)
        self.m._qp_retry_ready_cb(NS(data=nonce))
        self.m._qp_retry_ack_cb(NS(data='token=ready_'+nonce+' status=ready'))
        self.assertEqual(self.ready,[nonce]); self.assertEqual(self.m.qp_retry_ack_count,0)


if __name__=='__main__': unittest.main(verbosity=2)
