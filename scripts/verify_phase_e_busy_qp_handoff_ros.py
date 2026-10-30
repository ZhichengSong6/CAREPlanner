#!/usr/bin/env python3
"""Controlled busy/denial ordering with real native planner and PIQP.

Only the delivery of an already-denied dependency grant is delayed. Production
Python receives actual native triggers. The granted-expiry mode first produces
two real PIQP candidates, rejects their exact raw identities as controlled path
counterexamples, then delays Python cleanup after the native reservation expiry.
No model, actuator, tracker or Gazebo; synthetic CDF and stationary joints.
"""
from pathlib import Path

source = Path(__file__).with_name('test_phase_e_repair_observation_state.py').read_text()
prefix = source[:source.index('        # Happy path:')]
prefix = prefix.replace("args = ap.parse_args()", "ap.add_argument('--before', action='store_true')\n    ap.add_argument('--granted-expiry', action='store_true')\n    ap.add_argument('--foreign-requests', action='store_true')\n    ap.add_argument('--runtime-source')\n    args = ap.parse_args()")
prefix = prefix.replace('11339', '11672')
prefix = prefix.replace("lp['cdf_wait_timeout'] = 5.0", "lp['cdf_wait_timeout'] = 5.0\n                lp['candidate_replacement_enabled'] = True\n                lp.setdefault('terminal_retention', {})['enabled'] = False\n                lp['scp']['max_iterations'] = 1")
prefix = prefix.replace("if name == 'vbc_handoff':", "if name == 'qp_busy':")
prefix = prefix.replace("data=[1., 7., 1., .2, .3, .4]", "data=[1., 1., 1., .1, .1, .3]")
prefix = prefix.replace("self.target('care_obs_v1_1_deadbeef', .4)", "self.target(runtime_token, .4)")
body = r'''
        import importlib.util
        import sys
        import threading
        from types import SimpleNamespace as NS
        sys.path.insert(0, str(repo/'src/care_visibility_cdf/scripts'))
        from observation_identity import observation_token, observation_region_token
        relative = 'src/care_visibility_cdf/scripts/candidate_replacement_runtime.py'
        path = (repo/'outputs/phase_e_five_point_repair_20261013/before'/relative
                if args.before else repo/relative)
        if args.runtime_source: path=Path(args.runtime_source).resolve()
        spec = importlib.util.spec_from_file_location('busy_runtime_subject', path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        report['runtime_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        report['before'] = args.before
        ob = dict(id=1, points=np.array([[.1,.1,.3]]), q_vis=np.array([.4]+[0.]*6))
        parent = dict(id=2, points=np.array([[.2,.1,.3]]), q_vis=np.zeros(7))
        runtime_token = observation_token(ob)
        node = NS(_repair_stack=[2,1], _obligations=[parent,ob],
            _schedule_publish_lock=threading.RLock(), _obligation_lock=threading.RLock(),
            _trace_published_target=dict(observation_token=runtime_token,
                region_token=observation_region_token(ob), q_vis=ob['q_vis']))
        triggers, grants, sent, runtime_events = [], [], [], []
        pub = rospy.Publisher('/care_planner/local_planner/candidate_replacement_request', String, queue_size=10)
        def send(text):
            sent.append(text); pub.publish(String(data=text))
        runtime = mod.CandidateReplacement(node, lambda:rospy.Time.now().to_sec(), send,
            lambda event, **kw:runtime_events.append(dict(event=event, **kw)))
        def trigger(msg):
            triggers.append(msg.data); runtime.offer_final_vbc(msg.data)
        subs = [rospy.Subscriber('/care_planner/local_planner/candidate_replacement_trigger', String, trigger),
                rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant', String, lambda m:grants.append(m.data))]
        def fields(text): return dict(w.split('=',1) for w in text.split() if '=' in w)
        f = Fixture('qp_busy')
        def refresh(_, fixture=f):
            fixture.measured.header.stamp=rospy.Time.now(); fixture.pubs['joint_states'].publish(fixture.measured)
        timer = rospy.Timer(rospy.Duration(.03), refresh)
        wait(lambda:pub.get_num_connections()>0 and all(s.get_num_connections() for s in subs), 'replacement transport')
        if args.granted_expiry:
            responded=0
            for count in (1,2):
                deadline=time.monotonic()+8
                while len(f.candidates)<count and time.monotonic()<deadline:
                    if len(f.queries)>responded:
                        f.send(f.batch(responded,d=.5,g=0.));responded+=1
                    time.sleep(.01)
                assert len(f.candidates)==count,(count,len(f.candidates),f.events[-3:])
                raw=f.candidates[-1].header.stamp.to_nsec()
                f.pubs['verification_outcome_topic'].publish(String(data=
                    'result=protocol_invalid safety_gate=vbc_protocol committed=0 '
                    'raw_candidate_stamp_ns={} audited_trajectory_stamp_ns={} '
                    'vbc_protocol_reason=geometric_restore_path_counterexample'.format(raw,raw+1)))
                if count==1:wait(lambda:len(f.queries)>responded,'query after first path rejection')
            wait(lambda:len(triggers)==1 and len(grants)==1,'granted path-counterexample reservation')
            first=fields(grants[0]);assert first['granted']=='1' and first['attempts']=='2',first
            assert runtime.pending['reason']=='geometric_restore_path_counterexample'
            runtime.receive(grants[0])
            before=len(f.queries)
            wait(lambda:any('event=candidate_replacement_expired ' in e for e in f.events),
                 'native old reservation expiry')
            wait(lambda:len(f.queries)>before,'fresh query after path expiry')
            f.send(f.batch(len(f.queries)-1,d=-1.,g=0.))
            wait(lambda:len(triggers)==2,'new genuine hard-QP failure')
            wait(lambda:any(e['event'] in ('qp_trigger_deferred','trigger_ignored') for e in runtime_events),
                 'Python busy-path trigger handling')
            if args.foreign_requests:
                trigger=fields(triggers[-1])
                for key,value in [('query_stamp_ns',first['query_stamp_ns']),
                                  ('observation_token','care_obs_v1_foreign'),
                                  ('obligation_id','999'),('mode_epoch',str(int(trigger['mode_epoch'])+1))]:
                    runtime.serial+=1;request=dict(action='reserve',request_id=str(runtime.serial),
                        reason='repair_qp_failure',mode_epoch=trigger['mode_epoch'],
                        query_stamp_ns=trigger['query_stamp_ns'],observation_token=trigger['observation_token'],
                        obligation_id='1',trigger_raw_candidate_stamp_ns='0')
                    request[key]=value;n=len(grants)
                    pub.publish(String(data=' '.join(k+'='+v for k,v in request.items())))
                    wait(lambda:len(grants)>n,'foreign request denied')
                    denied=fields(grants[-1]);assert denied['granted']=='0' and denied['attempts']=='2',denied
            runtime.process()
            if args.before:
                assert len(sent)==2 and runtime.pending is None
                assert any(e.get('reason')=='reservation_already_pending' for e in runtime_events)
                report['checks'].append('BEFORE: granted path reservation expires natively, Python loses the newer real-QP trigger before its own cleanup')
            else:
                wait(lambda:any(fields(g)['request_id']==str(runtime.serial) and fields(g)['granted']=='1'
                    for g in grants[1:]),'deferred replacement granted after old path expiry')
                grant=next(fields(g) for g in grants[1:] if fields(g)['request_id']==str(runtime.serial))
                assert grant['attempts']=='3' and grant['segments']==first['segments'],grant
                assert fields(sent[-1])['query_stamp_ns']==fields(triggers[-1])['query_stamp_ns']
                assert len(sent)==3 and runtime.waiting_qp_trigger is None
                # A delayed finish for the old granted request cannot release the new one.
                pub.publish(String(data='action=finish request_id=1 new_token=none generation_result=generation_expired'))
                runtime.receive(grants[0]);assert runtime.grant is None
                runtime.offer_final_vbc(triggers[-1]);assert len(sent)==3
                report['checks'].append('AFTER: actual path trigger and old native grant2, expiry, fresh hard QP and deferred trigger -> exactly one native grant3; late old finish/grant and duplicate trigger harmless')
                if args.foreign_requests:report['checks'].append('wrong query/token/owner/epoch denials preserve the newer pending trigger and do not spend attempts')
            assert len(f.candidates)==2
            assert sum('event=sparse_qp_started ' in e for e in f.events)==3
        else:
            # Deliberately obsolete context guarantees a denial; retain that
            # denial in transport until the genuine current QP trigger arrives.
            runtime.offer(dict(mode_epoch=0, query_stamp_ns=f.queries[-1].header.stamp.to_nsec()-1,
                query_ros_s=rospy.Time.now().to_sec(), observation_token=runtime_token,
                points=parent['points'].tolist()))
            wait(lambda:len(grants)==1, 'native stale dependency denial')
            denied=fields(grants[0]); assert denied['granted']=='0', denied
            assert runtime.pending is not None
            f.send(f.batch(len(f.queries)-1, d=-1., g=0.))
            wait(lambda:len(triggers)==1, 'real PIQP failure trigger')
            wait(lambda:any(e['event'] in ('qp_trigger_deferred','trigger_ignored') for e in runtime_events), 'Python trigger delivery')
            runtime.receive(grants[0]); runtime.process()
            if args.before:
                time.sleep(1.35)
                assert len(sent)==1 and len(grants)==1 and runtime.pending is None
                assert any(e.get('reason')=='reservation_already_pending' for e in runtime_events)
                f.pubs['replan_request_topic'].publish(Bool(data=True)); time.sleep(.08)
                report['checks'].append('old production loses actual QP trigger while denied observation request is pending')
            else:
                wait(lambda:len(grants)==2, 'deferred QP native reservation')
                grant=fields(grants[1]); assert grant['granted']=='1', grant
                # The genuine QP failure first records its initial direction;
                # reserveReplacementAfterVbc then charges the next direction.
                assert int(denied['attempts'])==0 and int(grant['attempts'])==2
                assert grant['segments']==denied['segments']
                assert fields(sent[1])['query_stamp_ns']==fields(triggers[0])['query_stamp_ns']
                assert runtime.waiting_qp_trigger is None
                runtime.offer_final_vbc(triggers[0]); assert len(sent)==2
                report['checks'].append('same trigger survives denial, native accepts exact query/token and charges exactly one existing attempt; duplicate grants nothing')
            assert not f.candidates
            assert sum('event=sparse_qp_started ' in e for e in f.events)==1
        assert not any('arm_group_velocity_controller/command' in topic for topic,_ in rospy.get_published_topics())
        timer.shutdown(); f.save()
        report.update(triggers=triggers, grants=grants, sent=sent, runtime_events=runtime_events)
        report['checks'].append('protocol fixture only: synthetic map rows and negative audit verdicts, real callbacks/PIQP; no tracker, model generation or actuator')
        report['status']='PASS'
        print(json.dumps(dict(status=report['status'], checks=report['checks']),indent=2))
'''
footer = source[source.index('    finally:\n        (out/') :]
exec(compile(prefix+body+footer, str(Path(__file__).resolve()), 'exec'), globals())
