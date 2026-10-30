#!/usr/bin/env python3
"""Task identity, asynchronous lineage and real admission callback controls."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.append("/usr/lib/python3/dist-packages")

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/care_visibility_cdf/scripts'))
from task_observation_binding import ObservationTaskOwner, TaskContextReceiver, canonical, validate_binding


def command(request_id='A'):
    return dict(schema=1,action='submit',request_id=request_id,
        target=dict(frame_id='base_link',position=[.3,0.,.5],orientation=[0.,0.,0.,1.]),
        region_min=[-1.]*3,region_max=[1.]*3)


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.now=10_000_000_000
        self.owner=ObservationTaskOwner('test-owner')
        self.first=self.owner.command(command(),self.now)[-1]

    def test_hundred_resends_keep_identity_and_start(self):
        for i in range(100):
            e=self.owner.command(command(),self.now+i*1000000)[-1]
            self.assertEqual(e['candidate_binding'],self.first['candidate_binding'])
            self.assertEqual(e['started_ns'],self.first['started_ns'])

    def test_same_pose_with_new_request_is_new_context(self):
        events=self.owner.command(command('B'),self.now+1)
        self.assertFalse(events[0]['active'])
        self.assertNotEqual(events[1]['candidate_binding'],self.first['candidate_binding'])

    def test_conflicting_same_id_is_rejected_transactionally(self):
        c=command();c['target']['position'][0]+=.1
        with self.assertRaisesRegex(ValueError,'conflict'):self.owner.command(c,self.now+1)
        self.assertEqual(self.owner.current['candidate_binding'],self.first['candidate_binding'])

    def test_cancel_and_late_submit_cannot_reactivate(self):
        c=dict(schema=1,action='cancel',request_id='A',context_id=self.first['context_id'])
        self.assertFalse(self.owner.command(c,self.now+1)[0]['active'])
        with self.assertRaisesRegex(ValueError,'terminated'):self.owner.command(command(),self.now+2)

    def test_delayed_cancel_cannot_close_replacement(self):
        self.owner.command(command('B'),self.now+1)
        with self.assertRaisesRegex(ValueError,'stale'):
            self.owner.command(dict(schema=1,action='cancel',request_id='A',context_id=self.first['context_id']),self.now+2)
        self.assertTrue(self.owner.current['active'])

    def test_expired_owner_does_not_renew_generation(self):
        e=self.owner.tick(self.now+900000000)[0]
        self.assertFalse(e['active']);self.assertEqual(self.owner.tick(self.now+1000000000),[])

    def test_route_replacement_is_cancel_then_new_generation(self):
        c=dict(command(),action='replace_route',context_id=self.first['context_id'])
        c['region_max']=[.8]*3
        events=self.owner.command(c,self.now+1)
        self.assertEqual(len(events),2);self.assertFalse(events[0]['active']);self.assertTrue(events[1]['active'])
        self.assertEqual(events[1]['generation'],self.first['generation']+1)

    def test_invalid_route_replacement_preserves_current_work(self):
        c=dict(command(),action='replace_route',context_id=self.first['context_id'])
        c['region_max']=[-2.]*3
        with self.assertRaises(ValueError):self.owner.command(c,self.now+1)
        self.assertTrue(self.owner.current['active'])

    def test_map_invalidation_cannot_heartbeat_old_context(self):
        e=self.owner.map_invalidated(dict(kind='observation_context_invalidated',
            reason='observation_context_map_changed',observation_context=self.first),self.now+1)
        self.assertFalse(e[0]['active']);self.assertEqual(self.owner.tick(self.now+2),[])

    def test_old_map_event_cannot_close_new_goal(self):
        self.owner.command(command('B'),self.now+1)
        self.assertEqual(self.owner.map_invalidated(dict(kind='observation_context_invalidated',
            reason='observation_context_map_changed',observation_context=self.first),self.now+2),[])

    def test_map_resume_carries_verified_scope_in_new_candidate_binding(self):
        self.owner.map_invalidated(dict(kind='observation_context_invalidated',
            reason='observation_context_map_changed',observation_context=self.first),self.now+1)
        scope=dict(instance='new-native-instance',epoch=2,geometry=['base_link',.05,[-1.]*3,[1.]*3,[40]*3])
        resumed=self.owner.resume_map(dict(schema=1,action='resume_map',request_id='A',
            context_id=self.first['context_id'],fresh_map_scope=scope),self.now+2)[0]
        self.assertEqual(json.loads(resumed['candidate_binding'])['map_scope'],scope)
        self.assertNotEqual(resumed['generation'],self.first['generation'])

    def test_capacity_does_not_discard_replay_protection(self):
        self.owner.capacity=1
        with self.assertRaisesRegex(ValueError,'capacity'):self.owner.command(command('B'),self.now+1)
        self.assertEqual(self.owner.current['goal_request_id'],'A')

    def test_nonfinite_and_empty_quaternion_rejected(self):
        for value in ([0.]*4,[float('nan'),0.,0.,1.]):
            c=command('B');c['target']['orientation']=value
            with self.assertRaises(ValueError):self.owner.command(c,self.now+1)


class ReceiverTests(unittest.TestCase):
    setUp = OwnerTests.setUp
    def test_old_solve_with_new_stamp_still_rejected(self):
        receiver=TaskContextReceiver();receiver.update(self.first,self.now)
        old_binding=self.first['candidate_binding']
        for e in self.owner.command(command('B'),self.now+1):receiver.update(e,self.now+1)
        self.assertFalse(receiver.accepts(old_binding,self.now+2,'base_link'))
        self.assertTrue(receiver.accepts(self.owner.current['candidate_binding'],self.now+2,'base_link'))

    def test_changed_frame_or_payload_cannot_be_guessed(self):
        receiver=TaskContextReceiver();receiver.update(self.first,self.now)
        self.assertFalse(receiver.accepts('',self.now))
        self.assertFalse(receiver.accepts(self.first['candidate_binding'],self.now,'map'))
        e=copy.deepcopy(self.first);e['target']['position'][0]+=.1;e['event_seq']+=1
        with self.assertRaisesRegex(ValueError,'binding'):receiver.update(e,self.now)

    def test_same_epoch_cannot_mutate_goal_even_with_recomputed_envelope(self):
        from task_observation_binding import binding_of
        receiver=TaskContextReceiver();receiver.update(self.first,self.now)
        e=copy.deepcopy(self.first);e['target']['position'][0]+=.1;e['event_seq']+=1
        e['candidate_binding']=binding_of(e)
        with self.assertRaisesRegex(ValueError,'mutated'):receiver.update(e,self.now)

    def test_late_subscriber_keeps_owner_start_across_heartbeats(self):
        receiver=TaskContextReceiver()
        heartbeat=self.owner.tick(self.now+100000000)[0]
        context=receiver.update(heartbeat,self.now+100000000)
        self.assertEqual(context['started_ns'],self.first['started_ns'])
        heartbeat=self.owner.tick(self.now+200000000)[0]
        receiver.update(heartbeat,self.now+200000000)
        self.assertTrue(receiver.accepts(self.first['candidate_binding'],self.now+200000000))

    def test_candidate_first_waits_for_exact_context_with_original_receipt(self):
        import test_phase_e_trajectory_prefix as curves
        receiver=TaskContextReceiver();trajectory=curves.trajectory()
        trajectory.header.stamp=curves.duration(self.now)
        receiver.offer_candidate(trajectory,self.first['candidate_binding'],self.now)
        self.assertIsNone(receiver.candidate(self.now))
        receiver.update(self.owner.tick(self.now+100000000)[0],self.now+100000000)
        candidate=receiver.candidate(self.now+100000000)
        self.assertIsNotNone(candidate);self.assertEqual(candidate[2],self.now)

    def test_pending_duplicate_does_not_extend_age_and_expiry_drops_it(self):
        import test_phase_e_trajectory_prefix as curves
        receiver=TaskContextReceiver();trajectory=curves.trajectory()
        trajectory.header.stamp=curves.duration(self.now)
        receiver.offer_candidate(trajectory,self.first['candidate_binding'],self.now)
        receiver.offer_candidate(trajectory,self.first['candidate_binding'],self.now+500000000)
        self.assertEqual(receiver.pending_candidate['received_ns'],self.now)
        self.assertIsNone(receiver.candidate(self.now+800000001))
        self.assertIsNone(receiver.pending_candidate)

    def test_candidate_before_clock_update_waits_without_rebasing_stamp(self):
        import test_phase_e_trajectory_prefix as curves
        receiver=TaskContextReceiver();receiver.update(self.first,self.now)
        trajectory=curves.trajectory();trajectory.header.stamp=curves.duration(self.now+1000000)
        receiver.offer_candidate(trajectory,self.first['candidate_binding'],self.now)
        self.assertIsNone(receiver.candidate(self.now))
        pending=receiver.candidate(self.now+1000000)
        self.assertIsNotNone(pending);self.assertEqual(pending[2],self.now)

    def test_candidate_before_context_is_rejected_without_implicit_binding(self):
        receiver=TaskContextReceiver()
        self.assertFalse(receiver.accepts(self.first['candidate_binding'],self.now))
        receiver.update(self.first,self.now)
        self.assertTrue(receiver.accepts(self.first['candidate_binding'],self.now))


class AdmissionBindingTests(unittest.TestCase):
    def setUp(self):
        import test_phase_e_candidate_admission as old
        self.old=old.AdmissionTests();self.old.setUp();self.addCleanup(self.old.doCleanups)
        self.g,self.n,self.raw=self.old.g,self.old.n,self.old.raw
        self.g.task_context_enabled=self.n.task_context_enabled=True
        self.n.task_stop_pub=old.Pub()
        # This fixture enables task mode after construction; provide its new
        # task-event publisher as well as the existing tokenized-stop publisher.
        self.n.task_verification_pub=old.Pub()
        self.owner=ObservationTaskOwner('task-admission')
        e=self.owner.command(command(),self.old_module_time().to_nsec())[-1]
        self.first=e;self.binding=e['candidate_binding']
        for node in (self.g,self.n):node._task_context_cb(old.String(data=canonical(e)))

    def start(self):
        self.g._candidate_cb(self.raw,'',self.binding);self.old.pump()

    def test_bound_candidate_passes_original_both_gates(self):
        self.start();self.old.certify()
        self.assertEqual(len(self.n.committed_pub.messages),1)

    def test_bare_candidate_cannot_bypass_binding(self):
        self.g._candidate_cb(self.raw);self.assertFalse(self.g.transfer_pub.messages)

    def test_task_identity_is_in_payload_sha(self):
        from candidate_admission import payload_sha256
        self.assertNotEqual(payload_sha256(self.raw,'',self.binding),payload_sha256(self.raw,'','different'))

    def test_retarget_during_audit_prevents_old_commit(self):
        self.start()
        events=self.owner.command(command('B'),self.old_module_time().to_nsec()+1)
        self.old.advance(.001)
        for event in events:
            for node in (self.g,self.n):node._task_context_cb(self.old_module_string(event))
        self.assertIsNone(self.n._gcdf_outstanding)
        self.assertFalse(self.n.committed_pub.messages)

    def old_module_string(self,event):
        from std_msgs.msg import String
        return String(data=canonical(event))

    def test_delayed_old_planned_candidate_is_rejected(self):
        from egocentric_arm_planner.msg import CandidateTransfer
        self.old.advance(.001)
        for e in self.owner.command(command('B'),self.old_module_time().to_nsec()):
            self.g._task_context_cb(self.old_module_string(e));self.n._task_context_cb(self.old_module_string(e))
        self.raw.header.stamp=self.old_module_time()
        self.g._planned_candidate_cb(CandidateTransfer(action='planned',trajectory=self.raw,task_context=self.binding))
        self.assertFalse(self.g.transfer_pub.messages)

    def old_module_time(self):
        import rospy
        return rospy.Time.from_sec(self.old.now)

    def test_binding_tamper_rejected_at_receiver(self):
        self.g._candidate_cb(self.raw,'',self.binding)
        message=self.g.transfer_pub.messages.pop(0);message.task_context+=' '
        self.n._transfer_cb(message)
        self.assertFalse(self.n.final_gcdf_query_pub.messages)

    def test_cancel_committed_task_requests_exact_tracker_token_stop(self):
        self.start();self.old.certify()
        token=self.n.committed_pub.messages[-1].header.stamp.to_nsec()
        self.old.advance(.001)
        event=self.owner.command(dict(schema=1,action='cancel',request_id='A',
            context_id=self.first['context_id']),self.old_module_time().to_nsec())[0]
        self.n._task_context_cb(self.old_module_string(event))
        self.assertEqual(self.n.task_stop_pub.messages[-1].data,'execution_stamp_ns='+str(token))
        self.assertIsNone(self.n._committed_master)

    def test_gate_buffers_explicit_next_task_candidate_until_context(self):
        from egocentric_arm_planner.msg import CandidateTransfer
        self.old.advance(.001)
        events=self.owner.command(command('B'),self.old_module_time().to_nsec())
        self.raw.header.stamp=self.old_module_time()
        self.g._planned_candidate_cb(CandidateTransfer(action='planned',trajectory=self.raw,
            task_context=events[-1]['candidate_binding']))
        self.assertFalse(self.g.transfer_pub.messages)
        for event in events:self.g._task_context_cb(self.old_module_string(event))
        self.assertEqual(self.g.transfer_pub.messages[-1].task_context,events[-1]['candidate_binding'])


class ProducerBindingTests(unittest.TestCase):
    def fixture(self):
        import test_phase_e_trajectory_prefix as curves
        old=curves.ObserverTests();observer=old.fixture()
        observer.context_enabled=observer.require_task_binding=True
        observer.task_receiver=TaskContextReceiver();observer.observation_context=observer.task_receiver.context
        owner=ObservationTaskOwner('task-producer');event=owner.command(command(),old.f.now)[0]
        observer.offer_context(event)
        trajectory=curves.trajectory();trajectory.header.stamp=curves.duration(old.f.now)
        observer.offer(trajectory,event['candidate_binding'])
        return observer,old,owner,event,trajectory

    def test_actual_report_has_immutable_task_lineage(self):
        observer,old,owner,event,trajectory=self.fixture();observer.tick()
        self.assertEqual(len(old.sent),1)
        self.assertEqual(old.sent[0]['task_context'],event['candidate_binding'])
        self.assertFalse(old.sent[0]['execution_eligible'])

    def test_old_goal_result_with_new_stamp_cannot_be_offered(self):
        import test_phase_e_trajectory_prefix as curves
        observer,old,owner,event,trajectory=self.fixture()
        old.f.now+=1000000
        for e in owner.command(command('B'),old.f.now):observer.offer_context(e)
        trajectory.header.stamp=curves.duration(old.f.now)
        with self.assertRaisesRegex(ValueError,'mismatch'):
            observer.offer(trajectory,event['candidate_binding'])

    def test_goal_replacement_during_map_query_cannot_publish_old_demand(self):
        observer,old,owner,event,trajectory=self.fixture();query=observer.query;changed=[]
        def replace(*args):
            if not changed:
                for e in owner.command(command('B'),old.f.now):observer.offer_context(e)
                changed.append(True)
            return query(*args)
        observer.query=replace;observer.tick();self.assertFalse(old.sent)

    def test_same_goal_geometry_replacement_preserves_bound_demand(self):
        observer,old,owner,event,trajectory=self.fixture();query=observer.query;changed=[]
        def replace(*args):
            if not changed:
                updated=copy.deepcopy(trajectory);updated.points[1].positions[0]=.02
                observer.offer(updated,event['candidate_binding']);changed.append(True)
            return query(*args)
        observer.query=replace;observer.tick()
        self.assertEqual(len(old.sent),1);self.assertFalse(old.sent[0]['candidate_current_at_publish'])
        self.assertEqual(old.sent[0]['task_context'],event['candidate_binding'])


class DemandTimingTests(unittest.TestCase):
    def test_later_new_cell_has_own_deadline_and_old_cell_does_not_slide(self):
        import test_phase_e_observation_context as old
        from stopping_observation_context import ContextObservationDemandQueue
        queue=ContextObservationDemandQueue();report=old.report();context=report['observation_context']
        queue.update(report,context,10.)
        first=copy.deepcopy(report['cells'][0]);old_deadline=first['observation_deadline_s']
        later=copy.deepcopy(report);later['snapshot_ns']=20_000_000_000
        newer=copy.deepcopy(first);newer['grid_index']+=1;newer['point'][0]+=.05
        for key in ('first_use_s','required_until_s','observation_deadline_s','latest_start_s'):
            newer[key]+=20.
        shifted=copy.deepcopy(first)
        for key in ('first_use_s','required_until_s','observation_deadline_s','latest_start_s'):
            shifted[key]+=10.
        later['cells']=[shifted,newer]
        cells={c['grid_index']:c for c in queue.update(later,context,20.)}
        self.assertEqual(cells[first['grid_index']]['observation_deadline_s'],old_deadline)
        self.assertEqual(cells[newer['grid_index']]['observation_deadline_s'],old_deadline+20.)


class SchedulerBindingTests(unittest.TestCase):
    def fixture(self):
        import test_phase_e_observation_context as old
        import test_phase_e_trajectory_prefix as curves
        node,clock=old.SchedulerTests().fixture()
        node._require_task_binding=True;node._stopping_task_receiver=TaskContextReceiver()
        node._stopping_context=node._stopping_task_receiver.context
        owner=ObservationTaskOwner('task-scheduler');event=owner.command(command(),round(clock[0]*1e9))[0]
        node._stopping_context_callback(type('Message',(),dict(data=canonical(event)))())
        node.trajectory.header.stamp=curves.duration(round(clock[0]*1e9))
        node._stopping_task_receiver.offer_candidate(node.trajectory,event['candidate_binding'],round(clock[0]*1e9))
        node._process_context_observation_demands.__func__.__globals__['copy']=copy
        namespace=node._process_context_observation_demands.__func__.__globals__
        namespace['rospy'].Time=lambda sec,ns:type('Stamp',(),dict(to_sec=lambda self:(sec*1e9+ns)*1e-9))()
        namespace['rospy'].Time.now=lambda:type('Now',(),dict(to_sec=lambda self:clock[0],to_nsec=lambda self:round(clock[0]*1e9)))()
        report=old.report();report.update(observation_context=node._stopping_context.snapshot(round(clock[0]*1e9)),
                                        task_context=event['candidate_binding'])
        return node,clock,report,event

    def test_bound_mode_uses_bound_candidate_instead_of_unlabelled_latest(self):
        node,clock,report,event=self.fixture()
        def wrong_latest():raise AssertionError('unlabelled candidate consulted')
        node._preferred_trajectory_locked=wrong_latest
        node._stopping_prefix_pending=report;node._process_context_observation_demands()
        self.assertTrue(node._obligations)

    def test_missing_report_binding_cannot_create_schedule(self):
        node,clock,report,event=self.fixture();report.pop('task_context')
        node._stopping_prefix_pending=report;node._process_context_observation_demands()
        self.assertFalse(node._obligations)

    def test_map_frame_must_match_task_region_frame(self):
        node,clock,report,event=self.fixture();report['map_geometry'][0]='other_frame'
        node._stopping_prefix_pending=report;node._process_context_observation_demands()
        self.assertFalse(node._obligations)

    def test_second_map_change_before_first_report_closes_resumed_context(self):
        from task_observation_binding import binding_of
        node,clock,report,event=self.fixture()
        context=node._stopping_context.current
        context['map_scope']=dict(instance='expected-resumed-map',epoch=0,geometry=report['map_geometry'])
        context['candidate_binding']=binding_of(context)
        report.update(task_context=context['candidate_binding'],observation_context=copy.deepcopy(context))
        node._stopping_prefix_pending=report;node._process_context_observation_demands()
        self.assertTrue(node._stopping_context.closed)
        self.assertFalse(node._obligations)

    def test_old_bound_seed_cannot_generate_for_new_task(self):
        node,clock,report,event=self.fixture()
        node._stopping_task_receiver.pending_candidate['binding']='old task binding'
        node._stopping_prefix_pending=report;node._process_context_observation_demands()
        self.assertFalse(node._obligations)


if __name__=='__main__':unittest.main()
