#!/usr/bin/env python3
"""Real planner handshake/QP; synthetic VBC evidence, no execution certificate."""
from pathlib import Path
source=Path(__file__).with_name('verify_phase_e_vbc_failure_memory_ros.py').read_text()
prefix=source[:source.index('        results=[]')].replace('11619','11629')
body=r'''
        grants=[]
        request=rospy.Publisher('/care_planner/local_planner/candidate_replacement_request',String,queue_size=10)
        subs.append(rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant',String,lambda m:grants.append(m.data)))
        wait(lambda:request.get_num_connections() and subs[-1].get_num_connections(),'reserve channel')
        def fields(s):return dict(w.split('=',1) for w in s.split() if '=' in w)
        rid=0;results=[]
        def send(t,action,**extra):
            nonlocal rid
            rid+=1
            values=dict(action=action,request_id=rid,reason='final_vbc_repeat',mode_epoch=t['mode_epoch'],
                query_stamp_ns=t['query_stamp_ns'],observation_token=t['observation_token'],
                obligation_id=t['obligation_id'],trigger_raw_candidate_stamp_ns=t['raw_candidate_stamp_ns'])
            values.update(extra);text=' '.join(str(k)+'='+str(v) for k,v in values.items())
            request.publish(String(data=text));return text
        active.publish(Bool(data=True));repair.publish(Bool(data=True));time.sleep(.1)
        for oid,label in enumerate(['replace','decline','generation_failure'],31):
            old=f'care_obs_v1_{oid}_old';new=f'care_obs_v1_{oid}_new'
            ps=[[.2,0.,.5]];pose=[.3,0,0,0,0,0,0]
            base=len(candidates);marker=len(triggers)
            publish_target(old,oid,ps,pose)
            rp.publish(JointTrajectory(header=Header(stamp=rospy.Time.now()),joint_names=joints,
                points=[JointTrajectoryPoint(positions=[0.]*7,time_from_start=rospy.Duration(0)),
                        JointTrajectoryPoint(positions=pose,time_from_start=rospy.Duration(2))]))
            first=candidate(old,base);base=len(candidates);reject(first)
            second=candidate(old,base);reject(second)
            wait(lambda:len(triggers)>marker,'external-point trigger')
            t=fields(triggers[-1]);assert t['obligation_id']==str(oid)
            assert '.05' in t['repeated_points'] or '0.050' in t['repeated_points']
            marker_s=len(summaries)
            if label=='decline':
                # Wrong raw cannot acknowledge the current wait or spend budget.
                send(t,'decline',trigger_raw_candidate_stamp_ns='1',decline_reason='no_untried_sensor_or_family_capacity')
                time.sleep(.03)
                assert not any('event=candidate_replacement_unavailable ' in s for s in summaries[marker_s:])
                sent=send(t,'decline',decline_reason='no_untried_sensor_or_family_capacity')
                wait(lambda:any('event=candidate_replacement_unavailable ' in s for s in summaries[marker_s:]),'explicit decline')
                event=fields(next(s for s in summaries[marker_s:] if 'event=candidate_replacement_unavailable ' in s))
                assert event['candidate_replacement_trigger_pending']=='0' and event['safe_stall_attempts']=='0'
                request.publish(String(data=sent));time.sleep(.03)
                assert sum('event=candidate_replacement_unavailable ' in s for s in summaries[marker_s:])==1
            else:
                marker_g=len(grants);sent=send(t,'reserve')
                wait(lambda:len(grants)>marker_g,'reserve grant')
                g=fields(grants[-1]);assert g['granted']=='1' and g['attempts']=='2'
                request.publish(String(data=sent));time.sleep(.03)
                assert len(grants)==marker_g+1,'duplicate charged twice'
                if label=='replace':
                    base=len(candidates);pose[0]=.32;publish_target(new,oid,ps,pose)
                    request.publish(String(data=f'action=finish request_id={rid} new_token={new} generation_result=applied_requires_existing_gates'))
                    fresh=candidate(new,base)
                    wait(lambda:any('event=candidate_published ' in s and 'observation_token='+new+' ' in s for s in summaries[marker_s:]),'new hard candidate')
                    event=fields(next(s for s in summaries[marker_s:] if 'event=candidate_published ' in s and 'observation_token='+new+' ' in s))
                    assert event['safe_stall_attempts']=='2' and int(event['cdf_rows'])>0
                    assert fresh.header.stamp!=second.header.stamp
                else:
                    request.publish(String(data=f'action=finish request_id={rid} new_token=none generation_result=no_geometric_candidate'))
                    wait(lambda:any('event=candidate_replacement_rejected ' in s for s in summaries[marker_s:]),'generation failure finish')
                    event=fields(next(s for s in summaries[marker_s:] if 'event=candidate_replacement_rejected ' in s))
                    assert event['candidate_replacement_pending']=='0' and event['safe_stall_attempts']=='2'
            results.append(dict(scenario=label,event=event))
        assert not any('arm_group_velocity_controller/command' in t for t,_ in rospy.get_published_topics())
        result=dict(status='PASS',checks=results,planner_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            scope='Real production handshake and hard QP; VBC outcomes synthetic, no actuator.')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
footer=source[source.index('    finally:\n        stop.set()'):]
exec(compile(prefix+body+footer,str(Path(__file__).resolve()),'exec'),globals())
