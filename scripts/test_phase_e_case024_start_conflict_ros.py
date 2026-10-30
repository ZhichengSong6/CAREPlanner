#!/usr/bin/env python3
"""Real planner/PIQP: a new hard witness cannot burn sensor attempts.

Uses a private ROS master with synthetic CDF evidence. No actuator or Gazebo.
"""
from pathlib import Path

source = Path(__file__).with_name('test_phase_e_repair_observation_state.py').read_text()
prefix = source[:source.index('        # Happy path:')]
prefix = prefix.replace('11339', '11467')
prefix = prefix.replace("lp['cdf_wait_timeout'] = 5.0",
    "lp['cdf_wait_timeout'] = 5.0\n                lp['candidate_replacement_enabled'] = True")
prefix = prefix.replace("if name == 'vbc_handoff':",
    "if name in ('vbc_handoff', 'case024_start_conflict'):")
body = r'''
        triggers = []
        trigger_sub = rospy.Subscriber(
            '/care_planner/local_planner/candidate_replacement_trigger',
            String, lambda m: triggers.append(m.data))
        f = Fixture('case024_start_conflict')
        responded = 0
        deadline = time.monotonic() + 8
        while not f.candidates and time.monotonic() < deadline:
            if len(f.queries) > responded:
                query = f.query(responded)
                req = next(r for r in f.requests if r.header.stamp == query.header.stamp)
                n = len(req.original_timestep)
                safe = Batch(header=query.header, dof=7, num_pairs=n,
                    original_timestep=req.original_timestep, source_type=[0]*n,
                    point_flat=req.point_flat, q_linearization_flat=req.q_flat,
                    distance=[1.]*n, gradient_flat=[0.]*(7*n))
                f.response_pub.publish(CollisionCDFWitnessResponse(
                    request=req, batch=safe, status=['evaluated_unknown']*n))
                responded += 1
            time.sleep(.02)
        assert len(f.candidates) == 1, 'need a real candidate for exact feedback identity'
        c = f.candidates[0]
        before = len(f.queries)
        feedback = CollisionCDFRejectionFeedback(
            header=copy.deepcopy(c.header), raw_candidate_stamp=c.header.stamp,
            point_flat=[.15, 0., .30])
        feedback.header.stamp = rospy.Time.now()
        f.feedback_pub.publish(feedback)
        q = f.query(before)
        req = next(r for r in f.requests if r.header.stamp == q.header.stamp)
        low_steps = [req.original_timestep[i]
            for i in range(len(req.original_timestep))
            if np.allclose(req.point_flat[3*i:3*i+3], [.15, 0., .30])]
        assert low_steps == list(range(1, 21))
        case = json.loads((repo/'outputs/phase_e_case024_qp_conflicts_20261009/report.json').read_text())['cases'][0]
        g = case['rows'][0]['g']
        n = len(req.original_timestep)
        distances, gradients = [], []
        for i in range(n):
            point = req.point_flat[3*i:3*i+3]
            if np.allclose(point, [.15, 0., .30]):
                distances.append(-.226); gradients.extend(g)
            else:
                distances.append(1.); gradients.extend([0.]*7)
        b = Batch(header=q.header, dof=7, num_pairs=n,
            original_timestep=req.original_timestep, source_type=[0]*n,
            point_flat=req.point_flat, q_linearization_flat=req.q_flat,
            distance=distances, gradient_flat=gradients)
        before_triggers = len(triggers)
        f.response_pub.publish(CollisionCDFWitnessResponse(
            request=req, batch=b, status=['evaluated_unknown']*n))
        wait(lambda: any('event=repair_witness_start_infeasible_hold ' in e for e in f.events),
             'proven start conflict hold')
        hold = next(e for e in f.events if 'event=repair_witness_start_infeasible_hold ' in e)
        assert 'reachable_conflicting_repair_witness_rows=' in hold
        assert int(hold.split('reachable_conflicting_repair_witness_rows=')[1].split()[0]) > 0
        assert len(triggers) == before_triggers and len(f.candidates) == 1
        assert 'repair_witness_count=1' in hold
        assert 'vbc_obligation_id=7' in hold and 'safe_stall_attempts=0' in hold
        assert 'candidate_replacement_trigger_count=0' in hold
        held_queries = len(f.queries)
        f.pubs['replan_request_topic'].publish(Bool(data=True))
        time.sleep(.1)
        assert len(f.queries) == held_queries, 'same-state replan reopened failed witness'
        f.measured.position[0] = .02
        f.measured.header.stamp = rospy.Time.now()
        f.pubs['joint_states'].publish(f.measured)
        f.query(held_queries)
        next_req = next(r for r in f.requests
            if r.header.stamp == f.queries[held_queries].header.stamp)
        assert any(np.allclose(next_req.point_flat[3*i:3*i+3], [.15, 0., .30])
                   for i in range(len(next_req.original_timestep)))
        assert len(f.candidates) == 1
        report['checks'].append('real hard QP held on proven new witness; no replacement trigger or new raw candidate; same-state retry blocked; measured progress re-queries retained witness')
        f.save()
        assert not any('arm_group_velocity_controller/command' in topic
                       for topic,_ in rospy.get_published_topics())
        report['checks'].append('no actuator publisher')
        report['status'] = 'PASS'
        print(json.dumps(dict(status=report['status'], checks=report['checks']), indent=2))
'''
footer = source[source.index('    finally:\n        (out/'):]
exec(compile(prefix + body + footer, str(Path(__file__).resolve()), 'exec'), globals())
