#!/usr/bin/env python3
"""Integration tests of the built C++ planner, its callbacks and real PIQP.

Synthetic joint states/CDF batches on a private ROS master; no actuator,
tracker, Gazebo or CUDA. This intentionally replaces the old Python model test.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xmlrpc.client
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    uri = 'http://127.0.0.1:11339'
    try:
        xmlrpc.client.ServerProxy(uri).getPid('/repair_test_probe')
    except OSError:
        pass
    else:
        raise RuntimeError('ROS port 11339 already occupied')
    os.environ.update(ROS_MASTER_URI=uri, ROS_HOSTNAME='localhost',
                      ROS_LOG_DIR=str(out/'ros_logs'), ROS_HOME=str(out/'ros_home'))
    children, handles = [], []
    report = dict(status='FAIL', checks=[], fixtures={})
    binary = repo/'devel/lib/egocentric_arm_planner/local_sparse_scp_planner_node'
    report['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()

    def start(argv, name):
        f = (out/name).open('x'); handles.append(f)
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p)
        return p

    def wait(pred, label, timeout=8):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            if pred(): return
            time.sleep(.02)
        raise AssertionError('timeout: '+label)

    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGINT)
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=5)

    try:
        start(['roscore', '-p', '11339'], 'roscore.log')
        def ready():
            try: return xmlrpc.client.ServerProxy(uri).getPid('/repair_test_probe')[0] == 1
            except OSError: return False
        wait(ready, 'private ROS master')
        import rospy
        import yaml
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, String, Float64MultiArray, MultiArrayDimension
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from care_collision_cdf.msg import CollisionCDFWitnessRequest, CollisionCDFWitnessResponse
        from care_collision_cdf.msg import CollisionCDFRejectionFeedback
        rospy.init_node('repair_production_test', disable_signals=True)
        cfg = yaml.safe_load((repo/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        joints = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']

        class Fixture:
            def __init__(self, name):
                self.name = name
                config = copy.deepcopy(cfg)
                lp = config['local_planner']
                keys = ['joint_states', 'reference_trajectory', 'cdf_batch_topic',
                        'query_trajectory_topic', 'candidate_trajectory_topic', 'summary_topic',
                        'recovery_topic', 'single_waypoint_q_topic', 'single_waypoint_active_topic',
                        'visibility_frontier_topic', 'waypoint_schedule_topic', 'replan_request_topic',
                        'verification_outcome_topic', 'vbc_obligation_points_topic']
                for k in keys: lp[k] = '/repair_test/'+name+'/'+k
                # Only transport wait is extended for test-driven responses.
                lp['cdf_wait_timeout'] = 5.0
                lp['rejection_snapshots_enabled'] = name == 'final_vbc'
                rospy.set_param('/repair_test_'+name, config)
                self.events, self.queries, self.candidates, self.requests = [], [], [], []
                self.contexts = []
                self.subs = [
                    rospy.Subscriber(lp['summary_topic'], String, lambda m:self.events.append(m.data)),
                    rospy.Subscriber(lp['query_trajectory_topic'], JointTrajectory, self.queries.append),
                    rospy.Subscriber(lp['candidate_trajectory_topic'], JointTrajectory, self.candidates.append)]
                self.subs.append(rospy.Subscriber(lp['candidate_trajectory_topic']+'/audit_context',
                    String, lambda m:self.contexts.append(json.loads(m.data))))
                self.subs.append(rospy.Subscriber(lp['cdf_batch_topic']+'/witness_request',
                    CollisionCDFWitnessRequest, self.requests.append))
                self.subs.append(rospy.Subscriber(lp['summary_topic']+'/witness',String,lambda m:self.events.append(m.data)))
                self.response_pub=rospy.Publisher(lp['cdf_batch_topic']+'/witness_response',
                    CollisionCDFWitnessResponse,queue_size=10)
                self.pubs = {}
                self.feedback_pub = rospy.Publisher(lp['candidate_trajectory_topic']+'/gcdf_rejection_feedback',
                    CollisionCDFRejectionFeedback, queue_size=10)
                for key, typ in [('joint_states',JointState), ('reference_trajectory',JointTrajectory),
                                 ('cdf_batch_topic',Batch), ('recovery_topic',Bool),
                                 ('single_waypoint_q_topic',Float64MultiArray), ('single_waypoint_active_topic',Bool),
                                 ('replan_request_topic',Bool), ('visibility_frontier_topic',Float64MultiArray),
                                 ('verification_outcome_topic',String),
                                 ('vbc_obligation_points_topic',Float64MultiArray)]:
                    self.pubs[key] = rospy.Publisher(lp[key],typ,queue_size=10)
                self.proc = start([str(binary),'__name:=repair_test_'+name], name+'.log')
                wait(lambda: all(p.get_num_connections() for p in self.pubs.values()) and
                     self.response_pub.get_num_connections() and self.feedback_pub.get_num_connections(), name+' subscriptions')
                self.measured = JointState(name=joints,position=[0.]*7,velocity=[0.]*7)
                for _ in range(3):
                    self.measured.header.stamp = rospy.Time.now()
                    self.pubs['joint_states'].publish(self.measured); time.sleep(.03)
                self.target('care_obs_v1_1_deadbeef', .4)
                self.pubs['single_waypoint_active_topic'].publish(Bool(data=True))
                self.pubs['visibility_frontier_topic'].publish(Float64MultiArray(data=[1.,1.,1.,.3]+[0.]*6))
                self.pubs['recovery_topic'].publish(Bool(data=True))
                time.sleep(.1)
                ref = JointTrajectory(joint_names=joints)
                ref.header.stamp = rospy.Time.now()
                for t,q in [(0.,[0.]*7),(2.,[.4]+[0.]*6)]:
                    ref.points.append(JointTrajectoryPoint(positions=q,time_from_start=rospy.Duration(t)))
                self.pubs['reference_trajectory'].publish(ref)
                if name == 'vbc_handoff':
                    # Current stack-top visibility target. The planner keeps a
                    # short near-term CDF guard while queued schedule entries
                    # remain out of the CDF witness set.
                    self.pubs['vbc_obligation_points_topic'].publish(
                        Float64MultiArray(data=[1., 7., 1., .2, .3, .4]))
                    time.sleep(.05)
                self.query(0)

            def target(self, token, value):
                msg = Float64MultiArray(data=[value]+[0.]*6)
                msg.layout.dim = [MultiArrayDimension(label=token,size=7,stride=7)]
                self.pubs['single_waypoint_q_topic'].publish(msg)

            def query(self, i):
                wait(lambda: len(self.queries)>i, self.name+' query '+str(i))
                wait(lambda:any(r.header.stamp==self.queries[i].header.stamp for r in self.requests),self.name+' witness request')
                return self.queries[i]

            def batch(self, i, d=0., g=1., k=1):
                query = self.query(i)
                b = Batch(header=query.header,dof=7,num_pairs=1,original_timestep=[k],source_type=[0],
                          point_flat=[.1,.1,.3],distance=[d],gradient_flat=[g]+[0.]*6,
                          q_linearization_flat=list(query.points[k].positions))
                return b

            def send(self,b):
                req=self.requests[-1]
                if req.original_timestep:
                    self.response_pub.publish(CollisionCDFWitnessResponse(request=req,batch=b,
                        status=['evaluated_unknown']*len(req.original_timestep)))
                else:
                    self.pubs['cdf_batch_topic'].publish(b)

            def failed_observation(self):
                self.send(self.batch(0,d=-1.,g=0.))  # genuine infeasible hard GCDF row
                self.query(1)
                wait(lambda:any('event=scp_failed ' in e for e in self.events), self.name+' failure')
                e = next(e for e in self.events if 'event=scp_failed ' in e)
                assert 'repair_phase=observation' in e and 'unknown_cdf_rows=1' in e, e
                assert not self.candidates
                wait(lambda:any('event=plan_started ' in e and 'repair_phase=retreat' in e for e in self.events),self.name+' retreat')
                assert all(abs(x)<1e-12 for p in self.queries[1].points for x in p.positions)

            def save(self):
                report['fixtures'][self.name] = dict(events=self.events,queries=len(self.queries),
                    candidates=len(self.candidates), audit_contexts=self.contexts)
                stop(self.proc)

        # Happy path: a boundary UNKNOWN row remains selected in the hard QP.
        f = Fixture('fresh'); f.failed_observation()
        f.send(f.batch(1,d=.05)); f.query(2)
        wait(lambda:bool(f.candidates), 'safe retreat candidate')
        e = next(e for e in f.events if 'event=candidate_published ' in e)
        assert 'repair_phase=retreat' in e and 'repair_witness_fresh=1' in e and 'unknown_cdf_rows=1' in e, e
        assert 'visibility_objective_step=-1' in e and 'frontier_active=0' in e and 'observation_token=none' in e, e
        assert max(abs(x) for p in f.candidates[0].points for x in p.positions)<.01
        # Third query belongs to a new observation plan, not mid-retreat SCP.
        wait(lambda:len([e for e in f.events if 'event=plan_started ' in e and 'repair_phase=observation' in e])==2,'restored observation')
        f.send(f.batch(2,d=-1.,g=0.)); f.query(3)
        e = [e for e in f.events if 'event=scp_failed ' in e][-1]
        assert 'visibility_objective_step=-1' not in e, e
        report['checks'].append('real hard-QP failure -> measured retreat without q_vis/frontier -> fresh selected UNKNOWN hard success -> new observation plan')
        f.save()

        # Missing/invalid evidence must be re-evaluated each query and exhaust
        # the existing five-failure budget; no candidate can escape.
        for kind in ['missing','wrong_q','nan_gradient','malformed','hard_failure']:
            f = Fixture(kind); f.failed_observation()
            for i in range(1,5):
                b = f.batch(i)
                if kind=='missing': b=Batch(header=b.header,dof=7)
                if kind=='wrong_q': b.q_linearization_flat[6]+=.1
                if kind=='nan_gradient': b.gradient_flat[6]=float('nan')
                if kind=='malformed': b.gradient_flat=[]
                if kind=='hard_failure': b.distance=[-1.]; b.gradient_flat=[0.]*7
                f.send(b)
            wait(lambda:any('event=repair_qp_exhausted ' in e for e in f.events),kind+' bounded exhaustion')
            for _ in range(3): f.pubs['replan_request_topic'].publish(Bool(data=True)); time.sleep(.06)
            assert len(f.queries)==5 and not f.candidates, (kind,len(f.queries),len(f.candidates))
            if kind!='hard_failure':
                assert any('status=repair_witness_missing_or_invalid' in e for e in f.events)
                reason=dict(missing='missing_point',wrong_q='q_linearization_mismatch',
                            nan_gradient='nonfinite_gradient',malformed='batch_dimension_error')[kind]
                assert any('event=repair_witness_check' in e and 'reason='+reason in e and 'query_stamp_ns=' in e for e in f.events), (kind,f.events)
            report['checks'].append(kind+': no candidates, bounded five failures, external pulses cannot reopen')
            f.save()

        # A persistent witness owns one spatial identity. A fresh row for the
        # same point at another current knot refreshes it and enters that knot's
        # hard QP with the current q linearization.
        f=Fixture('point_scope_step'); f.failed_observation()
        f.send(f.batch(1,d=.05,k=2)); f.query(2)
        wait(lambda:bool(f.candidates),'point-scoped witness at another knot')
        e=next(e for e in f.events if 'event=candidate_published ' in e)
        assert 'repair_witness_count=1' in e and 'repair_witness_fresh=1' in e, e
        assert not any('reason=wrong_timestep' in e for e in f.events)
        report['checks'].append('same point at another fresh trajectory knot refreshes one persistent identity')
        f.save()

        # Timestamp mismatch must not release retreat; matching fresh batch can.
        f=Fixture('stale'); f.failed_observation()
        b=f.batch(1); b.header=f.queries[0].header; f.send(b)
        time.sleep(.15)
        assert len(f.queries)==2 and not f.candidates
        f.send(f.batch(1,d=.05)); f.query(2)
        wait(lambda:bool(f.candidates),'fresh after stale')
        assert any('stamp_miss=1' in e for e in f.events)
        report['checks'].append('stale query stamp ignored; matching batch can subsequently release retreat')
        f.save()

        # A successful INTERNAL SCP iteration cannot restore observation. Its
        # fresh evidence also cannot authorize the next iterate's missing row.
        f=Fixture('fresh_then_missing'); f.failed_observation()
        f.send(f.batch(1,d=-.03,g=1.)); f.query(2)
        wait(lambda:any('event=scp_solved ' in e for e in f.events),'internal hard success')
        e=next(e for e in f.events if 'event=scp_solved ' in e)
        assert 'repair_phase=retreat' in e and 'plan_seq=2 ' in e, e
        assert not f.candidates
        assert len([e for e in f.events if 'event=plan_started ' in e])==2
        for i in range(2,6):
            f.send(Batch(header=f.query(i).header,dof=7))
        wait(lambda:any('event=repair_qp_exhausted ' in e for e in f.events),'freshness not latched')
        assert len(f.queries)==6 and not f.candidates
        report['checks'].append('multi-iterate retreat stays retreat; previously fresh evidence cannot cover a later missing batch')
        f.save()

        # Empty source_type is the legacy UNKNOWN interface. A persistent
        # witness remains a hard row even when its fresh distance is clear.
        f=Fixture('legacy_screened'); b=f.batch(0,d=-1.,g=0.); b.source_type=[]; f.send(b); f.query(1)
        b=f.batch(1,d=1.,g=0.); b.source_type=[]; f.send(b); f.query(2)
        wait(lambda:bool(f.candidates),'legacy screened fresh witness')
        e=next(e for e in f.events if 'event=candidate_published ' in e)
        assert 'repair_witness_count=1' in e and 'cdf_rows=1' in e and 'screened_safe=0' in e and 'repair_phase=retreat' in e, e
        report['checks'].append('legacy empty source_type refreshes the retained hard witness without safe-row screening')
        f.save()

        # Reproduce the exp83 capacity shape: many UNKNOWN rows for the same
        # spatial points must consume one slot per point, while the next query
        # still expands every retained point across all 20 current knots.
        f=Fixture('point_compression')
        q=f.query(0)
        points=[[.1+.01*u,.1,.3] for u in range(7)]
        rows=[(p,k) for p in points for k in range(1,6)]
        b=Batch(header=q.header,dof=7,num_pairs=len(rows),
            original_timestep=[k for _,k in rows],source_type=[0]*len(rows),
            point_flat=[v for p,_ in rows for v in p],distance=[-1.]*len(rows),
            gradient_flat=[0.]*(len(rows)*7),
            q_linearization_flat=[v for _,k in rows for v in q.points[k].positions])
        f.send(b); q=f.query(1)
        wait(lambda:any('event=scp_failed ' in e for e in f.events),'compressed observation failure')
        req=next(r for r in f.requests if r.header.stamp==q.header.stamp)
        assert len(req.original_timestep)==7*20
        assert sorted(set(tuple(req.point_flat[i:i+3]) for i in range(0,len(req.point_flat),3))) == \
            sorted(tuple(p) for p in points)
        assert not any('reason=witness_capacity_exceeded' in e for e in f.events)
        b=Batch(header=q.header,dof=7,num_pairs=len(points),original_timestep=[2]*len(points),
            source_type=[0]*len(points),point_flat=[v for p in points for v in p],
            distance=[.05]*len(points),gradient_flat=[v for _ in points for v in [1.]+[0.]*6],
            q_linearization_flat=list(q.points[2].positions)*len(points))
        f.send(b); f.query(2)
        wait(lambda:bool(f.candidates),'compressed witness retreat candidate')
        e=next(e for e in f.events if 'event=candidate_published ' in e)
        assert 'repair_witness_count=7' in e and 'repair_witness_fresh=1' in e, e
        report['checks'].append('35 timestep rows compress to 7 point identities; all 140 fresh-query rows remain requested')
        f.save()

        f=Fixture('overflow'); b=f.batch(0,d=-1.,g=0.)
        b.num_pairs=33; b.original_timestep=[1]*33; b.source_type=[0]*33
        b.point_flat=[v for i in range(33) for v in [.1+i*.001,.1,.3]]
        b.distance=[-1.]*33; b.gradient_flat=[0.]*(33*7)
        b.q_linearization_flat=list(f.queries[0].points[1].positions)*33
        f.send(b); f.query(1)
        for i in range(1,5): f.send(f.batch(i,d=1.,g=0.))
        wait(lambda:any('event=repair_qp_exhausted ' in e for e in f.events),'overflow exhausts')
        assert not f.candidates
        assert any('repair_witness_count=32' in e for e in f.events)
        report['checks'].append('33 unique witness points still exceed capacity and fail closed without eviction')
        f.save()

        # Target replacement invalidates pending old work and witness scope.
        f=Fixture('target'); f.failed_observation()
        f.target('care_obs_v1_2_deadbeef',.5); time.sleep(.12)
        f.send(f.batch(1,d=-1.,g=0.)); f.query(2)
        wait(lambda:any('event=plan_started ' in e and 'observation_token=care_obs_v1_2_deadbeef' in e for e in f.events),'new target')
        e=[e for e in f.events if 'event=plan_started ' in e][-1]
        assert 'repair_witness_count=0' in e and 'repair_phase=observation' in e and not f.candidates, e
        report['checks'].append('new target resets phase/witness scope and rejects pending old target result')
        f.save()

        # The blocker-aware scheduler publishes only its current stack-top VBC
        # obligation. The planner must carry that point over the full repair
        # horizon, without requiring a local-GCDF witness first.
        f = Fixture('vbc_handoff')
        req = f.requests[-1]
        target_horizon=cfg['local_planner']['cdf']['visibility_obligation_horizon_steps']
        assert list(req.original_timestep) == list(range(1,target_horizon+1))
        assert len(req.point_flat) == 3 * target_horizon
        for k in range(target_horizon):
            assert np.allclose(req.point_flat[3*k:3*k+3], [.2, .3, .4])
        report['checks'].append('active visibility target is guarded over its configured horizon; confirmed GCDF witnesses remain separate')
        f.save()

        f=Fixture('reply_identity'); f.failed_observation()
        req=f.requests[-1]; b=f.batch(1)
        assert list(req.original_timestep)==list(range(1,21))
        assert len(req.q_flat)==7*20
        assert list(req.q_flat)==[x for k in range(1,21) for x in f.queries[1].points[k].positions]
        f.pubs['cdf_batch_topic'].publish(b); time.sleep(.1)
        assert len(f.queries)==2 and not f.candidates, 'plain batch bypassed explicit reply'
        wrong=copy.deepcopy(req); wrong.plan_sequence+=1
        f.response_pub.publish(CollisionCDFWitnessResponse(request=wrong,batch=b,status=['resolved_free']*len(wrong.original_timestep)))
        time.sleep(.1)
        assert len(f.queries)==2 and not f.candidates
        assert any('reason=response_identity_mismatch' in e for e in f.events)
        # A matching atomic response from the selector explicitly resolves the
        # map voxel; the ordinary empty batch alone did not authorize this.
        empty=Batch(header=b.header,dof=7)
        f.response_pub.publish(CollisionCDFWitnessResponse(request=req,batch=empty,map_stamp=rospy.Time.now(),status=['resolved_free']*len(req.original_timestep)))
        f.query(2); wait(lambda:bool(f.candidates),'explicit free disposition')
        e=next(e for e in f.events if 'event=candidate_published ' in e)
        assert 'repair_witness_count=0' in e and 'repair_phase=retreat' in e, e
        report['checks'].append('request carries current raw q; plain/stale-plan reply cannot release witness; matching explicit free reply can')
        f.save()
        # Real PIQP repeatedly solves; final executable rejection is a different
        # liveness event. No manager/Gazebo/actuator is needed for this callback test.
        for kind in ['gcdf_feedback', 'gcdf_feedback_overflow', 'gcdf_feedback_stale']:
            f=Fixture(kind)
            responded=0
            deadline=time.monotonic()+8
            while not f.candidates and time.monotonic()<deadline:
                if len(f.queries)>responded:
                    f.send(Batch(header=f.query(responded).header,dof=7)); responded+=1
                time.sleep(.02)
            assert len(f.candidates)==1
            c=f.candidates[0]
            feedback=CollisionCDFRejectionFeedback(header=copy.deepcopy(c.header),
                raw_candidate_stamp=c.header.stamp,point_flat=[.1,.1,.3])
            feedback.header.stamp=rospy.Time.now()
            before=len(f.queries)
            bad=copy.deepcopy(feedback); bad.raw_candidate_stamp=rospy.Time(1)
            f.feedback_pub.publish(bad)
            time.sleep(.08); assert len(f.queries)==before
            if kind=='gcdf_feedback_stale':
                f.measured.position[0]=.02
                f.pubs['joint_states'].publish(f.measured);time.sleep(.1)
                f.feedback_pub.publish(feedback);time.sleep(.1)
                assert not any('event=repair_final_gcdf_witness_requery ' in e for e in f.events)
                report['checks'].append('final GCDF feedback from old measured-progress episode ignored')
                f.save();continue
            feedback.overflow=kind=='gcdf_feedback_overflow'
            f.feedback_pub.publish(feedback)
            q=f.query(before)
            req=next(r for r in f.requests if r.header.stamp==q.header.stamp)
            assert list(req.original_timestep)==list(range(1,21))
            assert list(req.q_flat)==[x for k in range(1,21) for x in q.points[k].positions]
            f.feedback_pub.publish(feedback);time.sleep(.08)
            assert len(f.queries)==before+1, 'duplicate feedback started another plan'
            # A plain empty batch cannot conceal a broad-phase omitted point.
            f.pubs['cdf_batch_topic'].publish(Batch(header=q.header,dof=7));time.sleep(.08)
            assert len(f.candidates)==1 and len(f.queries)==before+1
            if kind=='gcdf_feedback':
                # Same point is negative even outside natural proximity. Its
                # fresh explicit row enters the hard QP at the new q.
                n_req=len(req.original_timestep)
                b=Batch(header=q.header,dof=7,num_pairs=n_req,original_timestep=req.original_timestep,
                    source_type=[0]*n_req,point_flat=req.point_flat,q_linearization_flat=req.q_flat,
                    distance=[-.07]*n_req,gradient_flat=[0.]*(n_req*7))
                f.response_pub.publish(CollisionCDFWitnessResponse(request=req,batch=b,
                    status=['evaluated_unknown']*n_req))
                q=f.query(before+1)
                assert len(f.candidates)==1
                assert any('unknown_cdf_rows=20' in e and 'event=scp_failed ' in e for e in f.events)
                req=next(r for r in f.requests if r.header.stamp==q.header.stamp)
                f.response_pub.publish(CollisionCDFWitnessResponse(request=req,
                    batch=Batch(header=q.header,dof=7),status=['resolved_free']*len(req.original_timestep)))
                wait(lambda:len(f.candidates)==2,'fresh explicit free after feedback')
                report['checks'].append('final GCDF point -> full 20-knot obligation request at fresh q; duplicate/plain empty ignored; negative rows enter hard QP; explicit free retires')
            else:
                for i in range(5):
                    q=f.query(before+i)
                    req=next(r for r in f.requests if r.header.stamp==q.header.stamp)
                    f.response_pub.publish(CollisionCDFWitnessResponse(request=req,
                        batch=Batch(header=q.header,dof=7),status=['resolved_free']*len(req.original_timestep))) if req.original_timestep else f.send(Batch(header=q.header,dof=7))
                wait(lambda:any('event=repair_qp_exhausted ' in e for e in f.events),'feedback overflow hold')
                assert len(f.candidates)==1
                report['checks'].append('final feedback capacity overflow cannot be cleared by partial free replies; bounded hold, no new candidate')
            f.save()
        f=Fixture('final_vbc')
        responded = 0
        def solve_to_candidate(count):
            nonlocal responded
            deadline=time.monotonic()+8
            while len(f.candidates)<count and time.monotonic()<deadline:
                if len(f.queries)>responded:
                    q=f.query(responded)
                    f.send(Batch(header=q.header,dof=7))
                    responded+=1
                time.sleep(.02)
            assert len(f.candidates)==count, (len(f.candidates), count)
            wait(lambda:len(f.contexts)>=count,'exact candidate audit context')

        def verdict(raw, **overrides):
            fields=dict(result='unsafe',safety_gate='vbc',committed='0',
                raw_candidate_stamp_ns=str(raw),audited_trajectory_stamp_ns=str(raw+1))
            fields.update(overrides)
            f.pubs['verification_outcome_topic'].publish(String(data=' '.join(k+'='+v for k,v in fields.items())))

        solve_to_candidate(1)
        raw=f.candidates[-1].header.stamp.to_nsec()
        for overrides in [dict(result='safe',committed='1'),dict(safety_gate='gcdf'),
                          dict(raw_candidate_stamp_ns='1'),dict(raw_candidate_stamp_ns=str(raw)+'bad')]:
            verdict(raw,**overrides)
        time.sleep(.15)
        assert not any('event=repair_final_vbc_' in e for e in f.events)
        for n in range(1,6):
            solve_to_candidate(n)
            raw=f.candidates[-1].header.stamp.to_nsec()
            ctx=f.contexts[-1]
            assert ctx['raw_candidate_stamp_ns']==str(raw)
            assert ctx['query_trajectory']['stamp_ns']==ctx['local_gcdf_batch']['stamp_ns']
            assert len(ctx['query_trajectory']['points'])==21
            verdict(raw)
            expected='repair_final_vbc_no_progress_hold' if n==5 else 'repair_final_vbc_retry'
            wait(lambda:any('event='+expected+' ' in e and 'final_vbc_failures='+str(n)+' ' in e for e in f.events),'final VBC count '+str(n))
            before=len([e for e in f.events if 'event=repair_final_vbc_' in e])
            verdict(raw); verdict(raw); time.sleep(.06)
            assert len([e for e in f.events if 'event=repair_final_vbc_' in e])==before
        count=len(f.queries)
        for _ in range(3): f.pubs['replan_request_topic'].publish(Bool(data=True)); time.sleep(.08)
        assert len(f.queries)==count and len(f.candidates)==5
        f.measured.position[0]=.02
        f.pubs['joint_states'].publish(f.measured)
        solve_to_candidate(6)
        assert any('event=plan_started ' in e and 'final_vbc_failures=0 ' in e for e in f.events)
        verdict(raw); time.sleep(.06)  # old transaction cannot poison resumed motion episode
        report['checks'].append('real QP success + five distinct final VBC rejections -> bounded hold; duplicate/unknown/malformed/non-VBC ignored; measured motion resumes; exact query/candidate context')
        f.save()
        assert not any('arm_group_velocity_controller/command' in topic for topic,_ in rospy.get_published_topics())
        report['checks'].append('no actuator publisher')
        report['status']='PASS'
        print(json.dumps(dict(status=report['status'],checks=report['checks']),indent=2))
    finally:
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        for p in reversed(children): stop(p)
        for f in handles: f.close()


if __name__ == '__main__': main()
