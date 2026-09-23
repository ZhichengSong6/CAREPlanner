#!/usr/bin/env python3
"""Real planner/PIQP on private ROS: margin rollback -> authenticated next candidate.
Reuses the existing transport fixture; no Gazebo, tracker, actuator, or model.
"""
from pathlib import Path

source=Path(__file__).with_name('test_phase_e_repair_observation_state.py').read_text()
prefix=source[:source.index('        # Happy path:')]
prefix=prefix.replace("uri = 'http://127.0.0.1:11339'", "uri = 'http://127.0.0.1:11461'").replace("['roscore', '-p', '11339']", "['roscore', '-p', '11461']")
prefix=prefix.replace("lp['cdf_wait_timeout'] = 5.0", "lp['cdf_wait_timeout'] = 5.0\n                lp['candidate_replacement_enabled'] = True\n                lp['scp']['max_iterations'] = 1")
prefix=prefix.replace("if name == 'vbc_handoff':", "if name.startswith('qp_'):").replace("data=[1., 7., 1., .2, .3, .4]", "data=[1., 4., 1., .1, .1, .3]")
body=r'''
        triggers,grants=[],[]
        subs=[rospy.Subscriber('/care_planner/local_planner/candidate_replacement_trigger',String,lambda m:triggers.append(m.data)),
              rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant',String,lambda m:grants.append(m.data))]
        request_pub=rospy.Publisher('/care_planner/local_planner/candidate_replacement_request',String,queue_size=10)
        f=Fixture('qp_sensor')
        wait(lambda:request_pub.get_num_connections()>0,'replacement transport')
        f.send(f.batch(len(f.queries)-1,d=.5,g=0.))
        wait(lambda:len(f.candidates)==1,'first hard candidate')
        raw=f.candidates[-1].header.stamp.to_nsec()
        f.pubs['verification_outcome_topic'].publish(String(data=
            'result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns='+str(raw)+
            ' audited_trajectory_stamp_ns='+str(raw+1)+
            ' vbc_evidence={"violations":[{"point":[0.1,0.1,0.3],"sweeping_link":"wrist_link3","source_collision_name":"sensor","primitive_signed_distance_m":-0.001}]}'))
        wait(lambda:len(f.queries)>=2,'query after margin increase')
        # Raised .20 is impossible for this zero-gradient row; previous .05 is feasible.
        f.send(f.batch(len(f.queries)-1,d=.15,g=0.))
        wait(lambda:bool(triggers),'first QP failure triggers replacement')
        wait(lambda:any('event=repair_qp_candidate_replacement_wait ' in e for e in f.events),'failure event')
        assert len(f.candidates)==1
        log=out/'qp_sensor.log'
        wait(lambda:'point_margin_rollback' in log.read_text(),'proven margin rollback')
        assert 'from=0.2 to=0.05' in log.read_text()
        trigger=dict(w.split('=',1) for w in triggers[-1].split())
        assert trigger['reason']=='repair_qp_failure' and trigger['raw_candidate_stamp_ns']=='0'
        def reserve(rid, token=None):
            request_pub.publish(String(data='action=reserve request_id='+str(rid)+
                ' reason=repair_qp_failure mode_epoch='+trigger['mode_epoch']+
                ' query_stamp_ns='+trigger['query_stamp_ns']+
                ' observation_token='+(token or trigger['observation_token'])+
                ' obligation_id=4 trigger_raw_candidate_stamp_ns=0'))
        # Reproduce the active-set callback interval seen in exp87. QP
        # replacement keeps the exact token/query handshake, but its bounded
        # reservation window must survive more than the old 0.5 seconds.
        time.sleep(.6)
        reserve(1)
        wait(lambda:bool(grants),'QP replacement grant')
        assert 'granted=1' in grants[-1],grants
        # A fresh sensor candidate supplies a distinct token and q_vis.
        f.target('care_obs_v1_4_next_sensor',.25)
        request_pub.publish(String(data='action=finish request_id=1 new_token=care_obs_v1_4_next_sensor'))
        before=len(f.queries)
        wait(lambda:len(f.queries)>before,'new sensor is planned')
        f.send(f.batch(len(f.queries)-1,d=.15,g=0.))
        wait(lambda:len(f.candidates)==2,'rolled-back margin still hard, new target solves')
        report['checks'].append('actual PIQP: raised .20 fails, old .05 hard solve succeeds -> point rollback -> authenticated replacement -> fresh hard candidate')
        # Same-point VBC feedback cannot reapply the proven failed increase.
        raw=f.candidates[-1].header.stamp.to_nsec();before=len(f.queries)
        f.pubs['verification_outcome_topic'].publish(String(data=
            'result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns='+str(raw)+
            ' audited_trajectory_stamp_ns='+str(raw+1)+
            ' vbc_evidence={"violations":[{"point":[0.1,0.1,0.3],"sweeping_link":"wrist_link3","source_collision_name":"sensor","primitive_signed_distance_m":-0.001}]}'))
        wait(lambda:len(f.queries)>before,'new VBC feedback query')
        wait(lambda:'previous_margin=0.05 cdf_margin=0.05' in log.read_text(),'failed raise cannot recur')
        f.send(f.batch(len(f.queries)-1,d=-1.,g=0.))
        wait(lambda:len(triggers)==2,'base constraint failure goes to another candidate')
        trigger=dict(w.split('=',1) for w in triggers[-1].split())
        assert len(f.candidates)==2
        # Deny malformed identity, then ensure no same-input retries or commits.
        reserve(2,'care_obs_v1_wrong')
        wait(lambda:len(grants)>=2,'wrong token denied')
        assert 'granted=0' in grants[-1]
        time.sleep(.8);count=len(f.queries)
        for _ in range(3): f.pubs['replan_request_topic'].publish(Bool(data=True));time.sleep(.05)
        assert len(f.queries)==count and len(f.candidates)==2
        assert not any('event=repair_qp_exhausted ' in e for e in f.events)
        report['checks'].append('rejected margin increase is latched; intrinsic hard-QP failure routes to replacement without five repeats; wrong-token denial cannot reopen same failed candidate')
        f.save()
        f=Fixture('qp_uncausal')
        f.send(f.batch(len(f.queries)-1,d=.5,g=0.))
        wait(lambda:len(f.candidates)==1,'uncausal baseline candidate')
        raw=f.candidates[-1].header.stamp.to_nsec();before=len(f.queries)
        f.pubs['verification_outcome_topic'].publish(String(data=
            'result=unsafe safety_gate=vbc committed=0 raw_candidate_stamp_ns='+str(raw)+
            ' audited_trajectory_stamp_ns='+str(raw+1)+
            ' vbc_evidence={"violations":[{"point":[0.1,0.1,0.3],"sweeping_link":"wrist_link3","source_collision_name":"sensor","primitive_signed_distance_m":-0.001}]}'))
        wait(lambda:len(f.queries)>before,'uncausal raised margin query')
        f.send(f.batch(len(f.queries)-1,d=-1.,g=0.))
        wait(lambda:len(triggers)==3,'uncausal failure replacement')
        assert 'point_margin_rollback' not in (out/'qp_uncausal.log').read_text()
        assert len(f.candidates)==1
        report['checks'].append('both raised and previous hard margins fail: no unproven margin reduction or candidate publication')
        f.save()
        assert not any('arm_group_velocity_controller/command' in topic for topic,_ in rospy.get_published_topics())
        report['checks'].append('no actuator publisher; shadow comparison never publishes a candidate')
        report['status']='PASS'
        print(json.dumps(dict(status=report['status'],checks=report['checks']),indent=2))
'''
footer=source[source.index('    finally:\n        (out/') :]
exec(compile(prefix+body+footer,str(Path(__file__).resolve()),'exec'),globals())
