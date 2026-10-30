#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import threading
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/egocentric_arm_planner/scripts'))
from rejection_snapshot import RejectionSnapshotSink, failure_signature, ros_value, gcdf_evidence


def record(stamp='100'):
    return dict(raw_candidate_stamp_ns=stamp, audited_trajectory_stamp_ns='200',
        vbc_evidence=dict(trajectory_stamp_ns='200', bundle_seq=4, violations=[dict(
            point=[.1,-.2,.8], sweeping_link='wrist_link2', source_collision_index=0,
            sweep_eval_k=2, latest_see_s=-.2)]),
        audited_trajectory=dict(points=[dict(positions=[.12345678901234567]*7, time_ns='100000000')]))


def context(stamp='100', token='A'):
    return dict(schema=1,raw_candidate_stamp_ns=stamp, observation_token=token,
                mode_epoch=7,target_revision=8,progress_epoch=145)


class SnapshotTests(unittest.TestCase):
    def test_full_queue_rejects_before_expensive_copy(self):
        import queue
        n=RejectionSnapshotSink.__new__(RejectionSnapshotSink)
        n.stop=threading.Event();n.jobs=queue.Queue(maxsize=1);n.jobs.put(None);n.dropped=0
        class NoCopy:
            def __deepcopy__(self,memo):raise AssertionError('unnecessary saturated copy')
        self.assertEqual(n.submit(NoCopy()),'queue_full');self.assertEqual(n.dropped,1)

    def test_gcdf_evidence_full_q_and_source(self):
        t=NS(header=NS(stamp=NS(to_nsec=lambda:200)),points=[
            NS(positions=[.12345678901234567]*7,time_from_start=NS(to_nsec=lambda:300000000))])
        b=NS(header=t.header,source_type=b'\x00\x01',distance=[-.01,-.02],
            original_timestep=[0,0],point_flat=[.1,.2,.3,.4,.5,.6],
            q_linearization_flat=[.2]*14,gradient_flat=list(range(14)))
        e=gcdf_evidence(b,t,0.)
        self.assertEqual([r['source_type'] for r in e['violations']],[0,1])
        self.assertEqual(e['violations'][1]['gradient'],list(range(7,14)))
        self.assertEqual(e['violations'][0]['candidate_q'],t.points[0].positions)
        r=record();r.update(safety_gate='gcdf_unknown',gcdf_evidence=e)
        self.assertNotEqual(failure_signature(r,context()),failure_signature(record(),context()))

    def test_genpy_uint8_bytes_roundtrip(self):
        class Message:
            __slots__=['source_type']
            def __init__(self,v): self.source_type=v
        for value,expected in [(b'',[]),(b'\x00\x01',[0,1]),(bytearray([0,255]),[0,255])]:
            self.assertEqual(json.loads(json.dumps(ros_value(Message(value)))),{'source_type':expected})

    def test_production_callback_snapshot_is_only_exact_rejection(self):
        from test_phase_e_vbc_evidence import methods, tokens
        class Stamp:
            def __init__(self, n): self.n=n
            def to_nsec(self): return self.n
            def to_sec(self): return self.n/1e9
            def __sub__(self, other): return Stamp(self.n-other.n)
        records=[]
        n=methods('src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py',
            {'_global_summary_cb'},dict(copy=copy,json=json,_tokens=tokens,
                trajectory_dict=lambda m: dict(stamp_ns=str(m.header.stamp.to_nsec()),points=m.points),
                _as_bool=lambda v:{'1':True,'0':False}.get(v),
                rospy=NS(Time=NS(now=lambda:Stamp(200)),logwarn=lambda *a:None)))
        n._lock=threading.RLock(); n._selector_cycle_count=0; n._outstanding_dispatch_cycle=-1
        n._outstanding=NS(header=NS(stamp=Stamp(100)),points=[dict(positions=[.1]*7)])
        n._outstanding_sent=Stamp(100); n._outstanding_raw_received=Stamp(90)
        n._outstanding_seq=1; n._outstanding_repair=True; n._outstanding_probe=False
        n._outstanding_view='repair_prefix_brake_hold'
        n._outstanding_diag=dict(raw_candidate_stamp_ns=90,snapshot_raw='RAW',snapshot_handoff='HANDOFF',
            snapshot_final_gcdf_batch='GCDF',snapshot_final_gcdf_trajectory='GCDF_TRAJ')
        n._exact_vbc_stamp_miss_count=n._verification_outcome_count=0
        n._verification_unsafe_count=n._repair_prefix_unsafe_count=0; n._cycle_recovery_active=False
        n._latest_measured_q_by_name={'joint1':0.}
        n._rejection_sink=NS(submit=lambda r:records.append(r) or 'queued')
        n._publish_summary_locked=lambda:None
        n._dispatch_pending_locked=lambda now:(None,None)
        n.verification_pub=n.final_gcdf_query_pub=NS(publish=lambda m:None)
        events=[]; n._make_verification_event=lambda *a,**kw:NS(diag=kw['diag'])
        n.verification_event_pub=NS(publish=events.append)
        def send(stamp):
            evidence=json.dumps(dict(trajectory_stamp_ns=str(stamp),bundle_seq=7,violations=[]),separators=(',',':'))
            n._global_summary_cb(NS(data='trajectory_source=predicted trajectory_stamp_ns=%d has_violation=1 vbc_evidence=%s'%(stamp,evidence)))
        send(101); self.assertFalse(records)
        send(100); self.assertEqual(len(records),1)
        self.assertEqual(records[0]['raw_trajectory'],'RAW')
        self.assertEqual(records[0]['handoff_trajectory'],'HANDOFF')
        self.assertEqual(records[0]['final_gcdf_batch'],'GCDF')
        self.assertEqual(records[0]['audited_trajectory']['stamp_ns'],'100')
        self.assertFalse(records[0]['committed'])
        self.assertEqual(events[0].diag['rejection_snapshot'],'queued')

    def test_manager_hold_feedback_is_diagnostic_not_execution(self):
        from test_phase_e_vbc_evidence import methods, tokens
        n=methods('src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py',
            {'_repair_final_vbc_feedback_cb'},dict(_tokens=tokens,_as_bool=lambda v:{'1':True,'0':False}.get(v)))
        n._lock=threading.RLock(); n.REPAIR=n.state='REPAIR'; n.repair_final_vbc_hold=False
        n._repair_final_vbc_feedback_cb(NS(data='final_vbc_hold=1'))
        self.assertTrue(n.repair_final_vbc_hold)
        self.assertEqual(n.last_transition_reason,'repair_final_vbc_no_progress_hold')
        n._repair_final_vbc_feedback_cb(NS(data='final_vbc_hold=invalid'))
        self.assertTrue(n.repair_final_vbc_hold)
        n._repair_final_vbc_feedback_cb(NS(data='final_vbc_hold=0'))
        self.assertFalse(n.repair_final_vbc_hold)

    def test_exact_join_full_precision_and_copy(self):
        with tempfile.TemporaryDirectory() as root:
            sink=RejectionSnapshotSink(Path(root)/'snap')
            try:
                sink.add_context(json.dumps(context('101','WRONG_NEIGHBOUR')))
                r=record(); sink.submit(r); r['audited_trajectory']['points'].clear()
                sink.add_context(json.dumps(context()))  # independent topic arrives later
                sink.jobs.join()
                saved=json.loads(next(sink.directory.glob('*.json')).read_text())
                self.assertEqual(saved['local_context']['observation_token'],'A')
                self.assertEqual(saved['audited_trajectory']['points'][0]['positions'][0],.12345678901234567)
            finally: sink.close()

    def test_signature_dedup_and_episode_change(self):
        with tempfile.TemporaryDirectory() as root:
            sink=RejectionSnapshotSink(Path(root)/'snap')
            try:
                for stamp in ['100','101','102']:
                    c=context(stamp)
                    if stamp=='102': c['progress_epoch']+=1
                    sink.add_context(json.dumps(c)); sink.submit(record(stamp)); sink.jobs.join()
                self.assertEqual(sink.written,2); self.assertEqual(sink.coalesced,1)
            finally: sink.close()

    def test_missing_identity_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            sink=RejectionSnapshotSink(Path(root)/'snap')
            try:
                sink.add_context(json.dumps(context('101'))); sink.submit(record()); sink.jobs.join()
                saved=json.loads(next(sink.directory.glob('*.json')).read_text())
                self.assertEqual(saved['context_status'],'NOT_RECORDED')
                self.assertIsNone(saved['local_context'])
            finally: sink.close()

    def test_file_and_byte_caps(self):
        with tempfile.TemporaryDirectory() as root:
            sink=RejectionSnapshotSink(Path(root)/'snap',max_files=1)
            try:
                for stamp in ['100','101']:
                    sink.add_context(json.dumps(context(stamp,stamp)))
                    sink.submit(record(stamp)); sink.jobs.join()
                self.assertEqual(sink.written,1); self.assertEqual(sink.dropped,1)
            finally: sink.close()
            sink=RejectionSnapshotSink(Path(root)/'tiny',max_bytes=1)
            try:
                sink.add_context(json.dumps(context())); sink.submit(record()); sink.jobs.join()
                self.assertEqual(sink.written,0); self.assertEqual(sink.dropped,1)
            finally: sink.close()

    def test_existing_directory_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileExistsError): RejectionSnapshotSink(root)

    def test_bad_payload_is_diagnostic_failure_only(self):
        with tempfile.TemporaryDirectory() as root:
            sink=RejectionSnapshotSink(Path(root)/'snap')
            try:
                sink.add_context(json.dumps(context())); sink.submit(dict(raw_candidate_stamp_ns='100'))
                sink.jobs.join(); self.assertEqual(sink.errors,1)
            finally: sink.close()

    def test_signature_does_not_merge_distinct_points(self):
        a=record(); b=copy.deepcopy(a); b['vbc_evidence']['violations'][0]['point'][2]=.75
        self.assertNotEqual(failure_signature(a,context()),failure_signature(b,context()))


if __name__=='__main__': unittest.main()
