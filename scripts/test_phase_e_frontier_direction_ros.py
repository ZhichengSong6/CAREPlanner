#!/usr/bin/env python3
"""Real C++ planner/PIQP: forward improvement, blocking and hard infeasibility.

Synthetic state and CDF on an isolated ROS master, no actuator or Gazebo.
"""
from pathlib import Path

source = Path(__file__).with_name('test_phase_e_repair_observation_state.py').read_text()
prefix = source[:source.index('        # Happy path:')]
prefix = prefix.replace('11339', '11465')
prefix = prefix.replace("lp['cdf_wait_timeout'] = 5.0", "lp['cdf_wait_timeout'] = 5.0\n                lp['candidate_replacement_enabled'] = True\n                lp['scp']['max_iterations'] = 1")
prefix = prefix.replace("if name == 'vbc_handoff':", "if name.startswith('direction_'):")
prefix = prefix.replace('data=[1., 7., 1., .2, .3, .4]', 'data=[1., 4., 1., .1, .1, .3]')
body = r'''
        triggers, grants = [], []
        subs = [rospy.Subscriber('/care_planner/local_planner/candidate_replacement_trigger', String, lambda m:triggers.append(m.data)),
                rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant', String, lambda m:grants.append(m.data))]
        request_pub = rospy.Publisher('/care_planner/local_planner/candidate_replacement_request', String, queue_size=10)
        def fields(s): return dict(w.split('=',1) for w in s.split() if '=' in w)
        def refresh(f):
            f.measured.header.stamp=rospy.Time.now(); f.pubs['joint_states'].publish(f.measured)
        def activate(name):
            f=Fixture('direction_'+name)
            timer=rospy.Timer(rospy.Duration(.03),lambda _,fixture=f:refresh(fixture))
            wait(lambda:request_pub.get_num_connections()>0 and all(s.get_num_connections()>0 for s in subs),'replacement transport')
            before=len(triggers)
            f.send(f.batch(len(f.queries)-1,d=-1.,g=0.))
            wait(lambda:len(triggers)>before,'initial actual hard-QP failure')
            t=fields(triggers[-1]); before=len(grants)
            request_pub.publish(String(data='action=reserve request_id=1 reason=repair_qp_failure mode_epoch='+t['mode_epoch']+
                ' query_stamp_ns='+t['query_stamp_ns']+' observation_token='+t['observation_token']+
                ' obligation_id=4 trigger_raw_candidate_stamp_ns=0'))
            wait(lambda:len(grants)>before,'activation grant')
            assert fields(grants[-1])['granted']=='1' and fields(grants[-1])['attempts']=='2'
            f.target('care_obs_v1_4_direction_test',.25)
            f.pubs['visibility_frontier_topic'].publish(Float64MultiArray(data=[1.,1.,0.,.05]+[0.]*6+[1.,1.]+[0.]*5))
            time.sleep(.08); before=len(f.queries)
            request_pub.publish(String(data='action=finish request_id=1 new_token=care_obs_v1_4_direction_test'))
            wait(lambda:len(f.queries)>before,'new candidate query')
            return f,timer
        scenarios=[('improving',.051,1.,0),('deficit_improving',.04,1.,0),
                   ('blocking',.051,-1.,1),('noncrossing',.054,-.01,0),
                   ('conflicting',.051,-1.,1),('infeasible_improving',-1.,1.,0)]
        for name,d,g,normals in scenarios:
            f,timer=activate(name)
            b=f.batch(len(f.queries)-1,d=d,g=g,k=2)
            if name=='conflicting':
                b.num_pairs=2; b.original_timestep=[2,2]; b.source_type=[0,0]
                b.point_flat=[.1,.1,.3]*2; b.distance=[.051,.051]
                b.gradient_flat=[-1.]+[0.]*6+[1.]+[0.]*6
                b.q_linearization_flat=list(b.q_linearization_flat)*2
            before=len(f.events); f.send(b)
            wait(lambda:any('event=scp_solved ' in e or 'event=scp_failed ' in e for e in f.events[before:]),'hard solve '+name)
            e=fields(next(e for e in f.events[before:] if 'event=scp_solved ' in e or 'event=scp_failed ' in e))
            report['fixtures'][f.name]=dict(events=f.events,queries=len(f.queries),candidates=len(f.candidates),solve=e)
            assert int(e['recovery_frontier_normals'])==normals,(name,e)
            if name=='infeasible_improving':
                assert e['event']=='scp_failed' and not f.candidates
                report['checks'].append(name+': forward objective cannot bypass infeasible hard row/velocity bounds')
            else:
                assert e['event']=='scp_solved',e
                wait(lambda:bool(f.candidates),'hard candidate '+name)
                q2=f.candidates[-1].points[2].positions[0]
                if normals==0: assert q2>.02,(name,q2)
                else: assert q2<=.0010001,(name,q2)
                if name=='conflicting': assert q2>=-.0010001
                report['checks'].append(name+': normals='+str(normals)+', q2_joint1='+str(q2))
            if name=='blocking':
                # Retained detour must lose ownership as soon as forward improves.
                before=len(f.queries)
                f.pubs['replan_request_topic'].publish(Bool(data=True))
                wait(lambda:len(f.queries)>before,'cached detour requery')
                before_events=len(f.events);before_candidates=len(f.candidates)
                f.send(f.batch(len(f.queries)-1,d=.051,g=1.,k=2))
                wait(lambda:len(f.candidates)>before_candidates,'rejoin after cached detour')
                e=fields(next(e for e in f.events[before_events:] if 'event=scp_solved ' in e))
                assert e['recovery_frontier_normals']=='0' and f.candidates[-1].points[2].positions[0]>.02
                report['checks'].append('cached detour is discarded when forward no longer blocked; new hard QP still required')
            timer.shutdown(); f.save()
        assert not any('arm_group_velocity_controller/command' in topic for topic,_ in rospy.get_published_topics())
        report['checks'].append('no actuator publisher; this fixture only publishes raw candidates after actual PIQP')
        report['status']='PASS'
        print(json.dumps(dict(status=report['status'],checks=report['checks']),indent=2))
'''
footer = source[source.index('    finally:\n        (out/'):]
exec(compile(prefix + body + footer, str(Path(__file__).resolve()), 'exec'), globals())
