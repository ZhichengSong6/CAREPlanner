#!/usr/bin/env python3
"""Real planner expiry/refresh transport; no actuator, simulator or learned model."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
source=Path(__file__).with_name('test_phase_e_repair_observation_state.py').read_text()
prefix=source[:source.index('        # Happy path:')]
prefix=prefix.replace('11339','11464')
prefix=prefix.replace("lp['cdf_wait_timeout'] = 5.0", "lp['cdf_wait_timeout'] = 5.0\n                lp.setdefault('terminal_retention', {})['enabled'] = False\n                lp['candidate_replacement_enabled'] = True\n                lp['scp']['max_iterations'] = 1")
prefix=prefix.replace("if name == 'vbc_handoff':", "if name.startswith('qp_'):").replace('data=[1., 7., 1., .2, .3, .4]','data=[1., 4., 1., .1, .1, .3]')
if os.environ.get('CARE_REPLACEMENT_BINARY'):
    prefix=prefix.replace("binary = repo/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'", "binary = Path(os.environ['CARE_REPLACEMENT_BINARY'])")
body=r'''
        triggers,grants=[],[]
        subs=[rospy.Subscriber('/care_planner/local_planner/candidate_replacement_trigger',String,lambda m:triggers.append(m.data)),
              rospy.Subscriber('/care_planner/local_planner/candidate_replacement_grant',String,lambda m:grants.append(m.data))]
        request_pub=rospy.Publisher('/care_planner/local_planner/candidate_replacement_request',String,queue_size=10)
        def fields(text):return dict(w.split('=',1) for w in text.split() if '=' in w)
        def count(f,name):return sum('event='+name+' ' in e for e in f.events)
        def refresh(f):
            f.measured.header.stamp=rospy.Time.now();f.pubs['joint_states'].publish(f.measured)
        def reserve(f,rid):
            t=fields(triggers[-1]);before=len(grants)
            request_pub.publish(String(data='action=reserve request_id={} reason=repair_qp_failure mode_epoch={} query_stamp_ns={} observation_token={} obligation_id=4 trigger_raw_candidate_stamp_ns=0'.format(rid,t['mode_epoch'],t['query_stamp_ns'],t['observation_token'])))
            wait(lambda:len(grants)>before,'grant')
            result=fields(grants[-1]);assert result['granted']=='1',result
            return result
        def finish(rid,result='no_geometric_candidate',token='none'):
            request_pub.publish(String(data='action=finish request_id={} new_token={} generation_result={}'.format(rid,token,result)))
        f=Fixture('qp_expiry');timer=rospy.Timer(rospy.Duration(.03),lambda _:refresh(f))
        wait(lambda:request_pub.get_num_connections()>0,'request transport')
        f.send(f.batch(len(f.queries)-1,d=-1.,g=0.))
        wait(lambda:len(triggers)==1,'genuine hard failure')
        assert count(f,'sparse_qp_started')==1 and not f.candidates
        baseline=os.environ.get('CARE_EXPECT_EXPIRED_HOLD')=='1'
        for rid in ([1] if baseline else range(1,5)):
            grant=reserve(f,rid);assert int(grant['attempts'])==rid+1
            before=len(f.queries);nt=len(triggers)
            if baseline:
                time.sleep(2.15);finish(rid);time.sleep(.1)
                assert len(f.queries)==before and len(triggers)==nt and not f.candidates
                report['checks'].append('BEFORE: recorded failure reproduced; expiry and late negative finish leave unused attempts without a fresh query')
                break
            if rid==4:
                wait(lambda:count(f,'repair_qp_candidate_exhausted_hold')>0,'finite exhaustion')
                finish(rid);finish(rid,'applied_requires_existing_gates','care_obs_v1_4_late')
                time.sleep(.1);assert len(f.queries)==before and len(triggers)==nt
                break
            wait(lambda:len(f.queries)==before+1,'expiry schedules one fresh query without a finish')
            assert f.queries[-1].header.stamp>f.queries[-2].header.stamp
            assert count(f,'sparse_qp_started')==1 and not f.candidates
            finish(rid);finish(rid);finish(rid,'applied_requires_existing_gates','care_obs_v1_4_late')
            time.sleep(.08);assert len(f.queries)==before+1 and len(triggers)==nt
            f.send(f.batch(before-1,d=-1.,g=0.));time.sleep(.04);assert len(triggers)==nt
            f.send(f.batch(before,d=-1.,g=0.));wait(lambda:len(triggers)==nt+1,'fresh query authorizes next reservation')
        assert count(f,'sparse_qp_started')==1 and not f.candidates
        assert not any('arm_group_velocity_controller/command' in t for t,_ in rospy.get_published_topics())
        timer.shutdown();f.save()
        if not baseline:report['checks'].append('AFTER: expiry alone refreshes attempts2->3->4->5; old/duplicate/late positive finishes and stale batches cannot authorize a head; one hard solve, no candidate/actuator, original finite exhaustion')
        report['status']='PASS';print(json.dumps(dict(status=report['status'],checks=report['checks'])))
'''
footer=source[source.index('    finally:\n        (out/'):]
exec(compile(prefix+body+footer,str(Path(__file__).resolve()),'exec'),globals())
