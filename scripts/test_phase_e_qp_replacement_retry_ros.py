#!/usr/bin/env python3
"""Real planner/PIQP on private ROS: margin rollback -> authenticated next candidate.
Reuses the existing transport fixture; no Gazebo, tracker, actuator, or model.
"""
from pathlib import Path

source=Path(__file__).with_name('test_phase_e_repair_observation_state.py').read_text()
prefix=source[:source.index('        # Happy path:')]
prefix=prefix.replace("uri = 'http://127.0.0.1:11339'", "uri = 'http://127.0.0.1:11463'").replace("['roscore', '-p', '11339']", "['roscore', '-p', '11463']")
prefix=prefix.replace("lp['cdf_wait_timeout'] = 5.0", "lp['cdf_wait_timeout'] = 5.0\n                lp.setdefault('terminal_retention', {})['enabled'] = False\n                lp['candidate_replacement_enabled'] = True\n                lp['scp']['max_iterations'] = 1")
prefix=prefix.replace("if name == 'vbc_handoff':", "if name.startswith('qp_'):").replace("data=[1., 7., 1., .2, .3, .4]", "data=[1., 4., 1., .1, .1, .3]")
body=r'''
        triggers,grants=[],[]
        subs=[rospy.Subscriber('/care_planner/local_planner/candidate_replacement_trigger',String,lambda m:triggers.append(m.data)),
              rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant',String,lambda m:grants.append(m.data))]
        request_pub=rospy.Publisher('/care_planner/local_planner/candidate_replacement_request',String,queue_size=10)
        def fields(s):return dict(w.split('=',1) for w in s.split() if '=' in w)
        def event_count(f,name):return sum('event='+name+' ' in e for e in f.events)
        def refresh_joint(f):
            f.measured.header.stamp=rospy.Time.now();f.pubs['joint_states'].publish(f.measured)
        def reserve(f,rid):
            trigger=fields(triggers[-1]);before=len(grants)
            request_pub.publish(String(data='action=reserve request_id='+str(rid)+
                ' reason=repair_qp_failure mode_epoch='+trigger['mode_epoch']+
                ' query_stamp_ns='+trigger['query_stamp_ns']+
                ' observation_token='+trigger['observation_token']+
                ' obligation_id=4 trigger_raw_candidate_stamp_ns=0'))
            wait(lambda:len(grants)>before,'grant '+str(rid))
            g=fields(grants[-1]);assert g['granted']=='1',g
            return g
        def finish(rid,result='no_geometric_candidate',token='none'):
            request_pub.publish(String(data='action=finish request_id='+str(rid)+
                ' new_token='+token+' generation_result='+result))
        f=Fixture('qp_retry');timer=rospy.Timer(rospy.Duration(.03),lambda _,fixture=f:refresh_joint(fixture))
        wait(lambda:request_pub.get_num_connections()>0,'request transport')
        f.send(f.batch(len(f.queries)-1,d=-1.,g=0.))
        wait(lambda:len(triggers)==1,'initial genuine QP failure')
        first_qp_count=event_count(f,'sparse_qp_started');assert first_qp_count==1 and not f.candidates
        g=reserve(f,1);assert g['attempts']=='2'
        before=len(f.queries);finish(999);time.sleep(.06);assert len(f.queries)==before
        for rid in range(1,4):
            before=len(f.queries);nt=len(triggers);old_stamp=f.queries[-1].header.stamp
            finish(rid)
            wait(lambda:len(f.queries)>before,'fresh context query after rejected generation')
            assert f.queries[-1].header.stamp>old_stamp
            assert event_count(f,'sparse_qp_started')==first_qp_count and not f.candidates
            finish(rid);time.sleep(.04);assert len(f.queries)==before+1
            # A stale response cannot grant the next branch.
            f.send(f.batch(before-1,d=-1.,g=0.));time.sleep(.04)
            assert len(triggers)==nt
            f.send(f.batch(before,d=-1.,g=0.))
            wait(lambda:len(triggers)>nt,'next trigger after fresh context')
            assert event_count(f,'sparse_qp_started')==first_qp_count and not f.candidates
            g=reserve(f,rid+1);assert int(g['attempts'])==rid+2
        before=len(f.queries);finish(4)
        wait(lambda:event_count(f,'repair_qp_candidate_exhausted_hold')>0,'budget exhaustion reported')
        for _ in range(3):f.pubs['replan_request_topic'].publish(Bool(data=True));time.sleep(.08)
        assert len(f.queries)==before and len(triggers)==4 and not f.candidates
        assert event_count(f,'sparse_qp_started')==1
        timer.shutdown();f.save()
        report['checks'].append('recorded CASE000 transition: attempts 2 -> 3 -> 4 -> 5 via fresh queries; zero repeated hard QPs/candidates; duplicate finish and stale reply ignored; exhaustion holds')
        def failed_fixture(name):
            fixture=Fixture('qp_'+name)
            timer=rospy.Timer(rospy.Duration(.03),lambda _,f=fixture:refresh_joint(f))
            wait(lambda:request_pub.get_num_connections()>0 and
                 all(s.get_num_connections()>0 for s in subs),'replacement transport '+name)
            nt=len(triggers)
            fixture.send(fixture.batch(len(fixture.queries)-1,d=-1.,g=0.))
            wait(lambda:len(triggers)>nt,'failure '+name)
            reserve(fixture,1)
            return fixture,timer
        f,timer=failed_fixture('retry_success')
        before=len(f.queries);nt=len(triggers);finish(1,'candidate_not_distinct')
        wait(lambda:len(f.queries)>before,'duplicate candidate refresh')
        f.send(f.batch(before,d=-1.,g=0.));wait(lambda:len(triggers)>nt,'success branch trigger')
        assert event_count(f,'sparse_qp_started')==1 and not f.candidates
        reserve(f,2);before=len(f.queries)
        f.target('care_obs_v1_4_next_sensor',.25)
        finish(2,'applied_requires_existing_gates','care_obs_v1_4_next_sensor')
        wait(lambda:len(f.queries)>before,'new target hard query')
        assert not f.candidates
        f.send(f.batch(before,d=.5,g=0.));wait(lambda:len(f.candidates)==1,'new target hard candidate')
        assert event_count(f,'sparse_qp_started')==2
        timer.shutdown();f.save()
        report['checks'].append('next distinct candidate still requires a new hard QP and fresh CDF; duplicate candidate rejection can also continue within budget')
        for result in ['generation_error','stale_measured_state','context_changed']:
            f,timer=failed_fixture(result)
            before=len(f.queries);nt=len(triggers);finish(1,result)
            wait(lambda:event_count(f,'candidate_replacement_rejected')>0,result+' held')
            for _ in range(3):f.pubs['replan_request_topic'].publish(Bool(data=True));time.sleep(.06)
            assert len(f.queries)==before and len(triggers)==nt and not f.candidates
            timer.shutdown();f.save()
        report['checks'].append('errors, stale measured generation and changed context do not authorize automatic retry')
        f,timer=failed_fixture('stale_finish')
        before=len(f.queries);nt=len(triggers)
        wait(lambda:len(f.queries)==before+1,'expired reservation refreshes without finish')
        finish(1)
        time.sleep(.1)
        assert len(f.queries)==before+1 and len(triggers)==nt and not f.candidates
        timer.shutdown();f.save()
        report['checks'].append('expiry owns one fresh query; stale finish cannot reopen generation or bypass its response')
        f,timer=failed_fixture('changed_target')
        f.target('care_obs_v1_4_changed_before_finish',.2);time.sleep(.08);finish(1)
        time.sleep(.1)
        assert event_count(f,'repair_qp_replacement_refresh_scheduled')==0
        timer.shutdown();f.save()
        report['checks'].append('changed observation token cancels the old retry context')
        f,timer=failed_fixture('moving_refresh')
        before=len(f.queries);nt=len(triggers);finish(1)
        wait(lambda:len(f.queries)>before,'motion refresh query')
        f.measured.position=[.02]+[0.]*6;refresh_joint(f);time.sleep(.08)
        f.send(f.batch(before,d=-1.,g=0.))
        wait(lambda:event_count(f,'repair_qp_replacement_refresh_invalid_hold')>0,'motion invalidates refresh')
        assert len(triggers)==nt and event_count(f,'sparse_qp_started')==1 and not f.candidates
        timer.shutdown();f.save()
        report['checks'].append('measured movement during refresh invalidates authorization without a repeated hard QP')
        f,timer=failed_fixture('missing_refresh')
        before=len(f.queries);nt=len(triggers);finish(1)
        wait(lambda:len(f.queries)>before,'missing response query')
        wait(lambda:event_count(f,'aborted_cdf_wait_timeout')>0,'missing refresh fails closed')
        time.sleep(.1)
        assert len(f.queries)==before+1 and len(triggers)==nt and not f.candidates
        assert event_count(f,'sparse_qp_started')==1
        timer.shutdown();f.save()
        report['checks'].append('missing refresh response times out once, with no generation or repeated QP')
        assert not any('arm_group_velocity_controller/command' in topic for topic,_ in rospy.get_published_topics())
        report['status']='PASS'
        print(json.dumps(dict(status=report['status'],checks=report['checks']),indent=2))
'''
footer=source[source.index('    finally:\n        (out/') :]
exec(compile(prefix+body+footer,str(Path(__file__).resolve()),'exec'),globals())
