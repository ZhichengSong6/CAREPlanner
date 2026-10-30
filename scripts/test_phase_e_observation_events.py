#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
import ast
import threading
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from observation_event_trace import observation_token, append_event
from analyze_phase_e_observation_events import exact_raw_map, analyze, string_rows


class EventTests(unittest.TestCase):
    def test_simulated_clock_preamble_keeps_real_records_and_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'execution.csv'
            path.write_text('WARNING: no messages received and simulated time is active.\n'
                            'Is /clock being published?\n%time,field.data\n'
                            '500000000,vbc success=1 has_violation=0\n'
                            '2000000000,vbc success=1 has_violation=0 xyz=1,2,3\n')
            rows=string_rows(path,dict(start_ros_s=1,end_ros_s=5))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['success'],'1')
            self.assertEqual(rows[0]['xyz'],'1,2,3')
            self.assertEqual(rows[0]['ros_s'],2.)

    def test_clock_warning_without_messages_is_empty_but_bad_header_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'optional.csv'
            preamble='WARNING: no messages received and simulated time is active.\nIs /clock being published?\n'
            path.write_text(preamble)
            window=dict(start_ros_s=1,end_ros_s=5)
            self.assertEqual(string_rows(path,window),[])
            path.write_text(preamble+'wrong,header\n')
            with self.assertRaises(ValueError):string_rows(path,window)

    def test_silent_ros_string_recorder_is_empty_not_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'optional.csv'
            window=dict(start_ros_s=1,end_ros_s=5)
            self.assertEqual(string_rows(path,window),[])
            path.touch()
            self.assertEqual(string_rows(path,window),[])
            path.write_text('%time,field.data\n')
            self.assertEqual(string_rows(path,window),[])
            path.write_text('wrong,header\n')
            with self.assertRaises(ValueError): string_rows(path,window)
            path.write_text('\n')
            with self.assertRaises(ValueError): string_rows(path,window)

    def test_missing_recorder_cannot_arm_and_timeout_is_bounded(self):
        import phase_e_benchmark_arm as arm
        clock=SimpleNamespace(t=0.)
        registered=['/unrelated']
        pub=SimpleNamespace(get_num_connections=lambda:1)
        modules={
            'rospy':SimpleNamespace(Publisher=lambda *a,**kw:pub,get_name=lambda:'/test',is_shutdown=lambda:False),
            'rosgraph':SimpleNamespace(Master=lambda name:SimpleNamespace(
                getSystemState=lambda:([],[('/topic',registered)],[]))),
            'std_msgs.msg':SimpleNamespace(String=object)}
        with patch.dict(sys.modules,modules), patch.object(arm.time,'monotonic',lambda:clock.t), \
             patch.object(arm.time,'sleep',lambda dt:setattr(clock,'t',clock.t+dt)):
            with self.assertRaisesRegex(RuntimeError,'missing_recorders'):
                arm.prepare_arm(['/expected_recorder'],timeout_s=.2)
            self.assertLessEqual(clock.t,.25)
            registered[:]=['/expected_recorder']
            self.assertIs(arm.prepare_arm(['/expected_recorder'],timeout_s=.2),pub)

    def test_arm_refresh_same_publisher_without_sending_token(self):
        import phase_e_benchmark_arm as arm
        clock=SimpleNamespace(t=0.,connected=False)
        calls=[]
        pub=SimpleNamespace(resolved_name=arm.ARM_TOPIC,
            get_num_connections=lambda:int(clock.connected))
        def refresh(*args):
            calls.append(args);clock.connected=True
        master=SimpleNamespace(getSystemState=lambda:(
            [(arm.ARM_TOPIC,['/test'])],
            [('/data',['/recorder']),(arm.ARM_TOPIC,['/trial'])],[]),registerPublisher=refresh)
        modules={'rospy':SimpleNamespace(Publisher=lambda *a,**kw:pub,
            get_name=lambda:'/test',get_node_uri=lambda:'http://127.0.0.1:12345/',is_shutdown=lambda:False),
            'rosgraph':SimpleNamespace(Master=lambda name:master),
            'std_msgs.msg':SimpleNamespace(String=type('String',(),{'_type':'std_msgs/String'}))}
        with patch.dict(sys.modules,modules),patch.object(arm.time,'monotonic',lambda:clock.t), \
             patch.object(arm.time,'sleep',lambda dt:setattr(clock,'t',clock.t+dt)):
            self.assertIs(arm.prepare_arm(['/recorder'],timeout_s=.2),pub)
        self.assertEqual(calls,[(arm.ARM_TOPIC,'std_msgs/String','http://127.0.0.1:12345/')])
        self.assertLessEqual(clock.t,.2)

    def test_arm_refresh_cannot_bypass_connection_or_missing_recorders(self):
        import phase_e_benchmark_arm as arm
        for recorder in ['/expected','/missing']:
            with self.subTest(recorder=recorder):
                clock=SimpleNamespace(t=0.);calls=[]
                pub=SimpleNamespace(resolved_name=arm.ARM_TOPIC,get_num_connections=lambda:0)
                master=SimpleNamespace(getSystemState=lambda:([],
                    [('/data',['/expected']),(arm.ARM_TOPIC,['/trial'])],[]),
                    registerPublisher=lambda *args:calls.append(args))
                modules={'rospy':SimpleNamespace(Publisher=lambda *a,**kw:pub,
                    get_name=lambda:'/test',get_node_uri=lambda:'http://127.0.0.1:12345/',is_shutdown=lambda:False),
                    'rosgraph':SimpleNamespace(Master=lambda name:master),
                    'std_msgs.msg':SimpleNamespace(String=type('String',(),{'_type':'std_msgs/String'}))}
                with patch.dict(sys.modules,modules),patch.object(arm.time,'monotonic',lambda:clock.t), \
                     patch.object(arm.time,'sleep',lambda dt:setattr(clock,'t',clock.t+dt)):
                    with self.assertRaisesRegex(RuntimeError,'connections=0'):
                        arm.prepare_arm([recorder],timeout_s=.2)
                self.assertEqual(len(calls),int(recorder=='/expected'))
                self.assertLessEqual(clock.t,.25)

    def test_end_to_end_identity_and_seen_are_independent(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory); (run/'projector_traces').mkdir()
            (run/'goal_stop_status.json').write_text(json.dumps(dict(
                benchmark_window=dict(start_ros_s=1,end_ros_s=5))))
            event=dict(stage='generated',ros_s=2,observation_token='target',
                       per_sensor_hybrid=dict(enabled=True,accepted=False,compute_ms=12))
            seen=dict(stage='seen_query',ros_s=4,observation_token='target',actual_seen=True,q_distance_inf=.5)
            (run/'projector_traces/observation_events.jsonl').write_text(json.dumps(event)+'\n'+json.dumps(seen)+'\n')
            for filename,values in {
                'local_planner_summary.csv':[(3.5,'observation_token=other event=scp_failed plan_seq=8'),
                                            (4.,'observation_token=target event=scp_failed plan_seq=9')],
                'verification_outcome.csv':[(3,'seq=7 committed=1 execution_stamp_ns=3000000000 raw_candidate_stamp_ns=2000000000')],
                'tracker_summary.csv':[(3.1,'seq=7 execution_stamp_ns=3000000001 complete=1'),
                                       (3.2,'seq=2 execution_stamp_ns=3000000000 complete=0')]
            }.items():
                with (run/filename).open('w') as f:
                    writer=csv.writer(f); writer.writerow(['%time','field.data'])
                    writer.writerows([(int(t*1e9),s) for t,s in values])
            report=analyze(run,'[OBS_CANDIDATE] observation_token=target raw_candidate_stamp_ns=2000000000')
            row=report['observations'][0]
            self.assertEqual(row['tracker_records'],1)
            self.assertFalse(row['tracker_complete'])
            self.assertEqual(row['last_stage'],'ACTUAL_SEEN')
            self.assertEqual(row['measured_q_distance_min_inf'],.5)
            self.assertEqual(report['hybrid_latency_ms']['all_rejected']['count'],1)
            self.assertEqual(row['last_local_planning_event']['plan_seq'], '9')
            self.assertEqual(len(row['local_planning_events']), 1)

    def test_exact_join_never_uses_nearest_timestamp(self):
        log='[OBS_CANDIDATE] observation_token=first raw_candidate_stamp_ns=100\n'
        log+='[OBS_CANDIDATE] observation_token=second raw_candidate_stamp_ns=101\n'
        log+='[OBS_CANDIDATE] observation_token=ambiguous raw_candidate_stamp_ns=100\n'
        ids=exact_raw_map(log)
        self.assertEqual(ids['100'],{'first','ambiguous'})
        self.assertEqual(ids['101'],{'second'})
        self.assertNotIn('102',ids)

    def test_broker_nonce_and_existing_prior_gates(self):
        repo=Path(__file__).resolve().parents[1]
        tree=ast.parse((repo/'src/egocentric_arm_planner/scripts/phase_b2_controlled_trial_node.py').read_text())
        names={'_benchmark_arm_callback','_goal_timer_callback'}
        methods=[n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
                 if isinstance(n,ast.FunctionDef) and n.name in names]
        clock=SimpleNamespace(to_sec=lambda:1.,__sub__=lambda self,x:self)
        class Stamp:
            def __sub__(self,other): return self
            def to_sec(self): return 1.
        scope={'rospy':SimpleNamespace(Time=SimpleNamespace(now=lambda:Stamp()),
                    loginfo=lambda *a:None,logwarn_throttle=lambda *a:None,logwarn=lambda *a:None)}
        cls=ast.ClassDef(name='Broker',bases=[],keywords=[],body=methods,decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'broker','exec'),scope)
        b=scope['Broker'](); b._lock=threading.Lock(); b._goal_sent=False
        b._benchmark_arm_token='new'; b._benchmark_armed=False; b._initial_prior_ready=False
        b._startup_time=Stamp(); b.goal_delay=.5
        sent=[]; b.goal_pub=SimpleNamespace(get_num_connections=lambda:1,publish=sent.append)
        b._goal_message=lambda:'goal'; b.goal_x=b.goal_y=b.goal_z=0.
        b._benchmark_arm_callback(SimpleNamespace(data='stale'))
        b._goal_timer_callback(None); self.assertFalse(sent)
        b._benchmark_arm_callback(SimpleNamespace(data='new'))
        b._goal_timer_callback(None); self.assertFalse(sent)
        b._initial_prior_ready=True; b._goal_timer_callback(None)
        b._goal_timer_callback(None); self.assertEqual(sent,['goal'])

    def test_identity_refresh_and_order(self):
        ob=dict(id=1,generation_event_id='generation_a',q_vis=[0.]*7,points=[[1,2,3],[4,5,6]])
        token=observation_token(ob)
        self.assertEqual(token,observation_token(dict(ob,points=ob['points'][::-1])))
        for change in [dict(id=2),dict(generation_event_id='generation_b'),
                       dict(q_vis=[1e-12]+[0.]*6),dict(points=[[1,2,3]])]:
            self.assertNotEqual(token,observation_token(dict(ob,**change)))

    def test_append_preserves_all_rejected_and_error(self):
        with tempfile.TemporaryDirectory() as out:
            for result in ['accepted','all_rejected','runtime_error']:
                append_event(out,'generation',10,result=result,compute_ms=12)
            rows=[json.loads(line) for line in (Path(out)/'observation_events.jsonl').read_text().splitlines()]
            self.assertEqual([r['result'] for r in rows],['accepted','all_rejected','runtime_error'])
            self.assertTrue(all(r['compute_ms']==12 and r['monotonic_s']>0 for r in rows))


if __name__=='__main__': unittest.main(verbosity=2)
