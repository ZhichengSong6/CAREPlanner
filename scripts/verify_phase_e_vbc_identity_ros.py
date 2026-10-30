#!/usr/bin/env python3
"""Frozen/current complete C++ selector over TCPROS; controlled known map, no robot."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'wrist_joint1', 'wrist_joint2', 'wrist_joint3']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', required=True, type=Path)
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    frozen = ROOT / 'outputs/phase_e_vbc_identity_fix_20261012/before'
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    os.environ.update(ROS_MASTER_URI='http://127.0.0.1:'+str(port), ROS_HOSTNAME='127.0.0.1',
                      ROS_HOME=str(out/'ros_home'), ROS_LOG_DIR=str(out/'ros_logs'))
    children = []; streams = []
    report = dict(status='FAIL', checks=[], qualification='Actual full frozen/current C++ selectors, ROS serialization and network, controlled all-known map. No physical robot, motion, trained CDF, or task-success claim.')
    def start(argv, name):
        stream = (out/(name+'.log')).open('x'); streams.append(stream)
        p = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(p); return p
    def wait(pred, limit=8):
        end = time.monotonic()+limit
        while time.monotonic() < end:
            if pred():
                return
            time.sleep(.005)
        raise AssertionError('private ROS condition timed out')
    def tokens(s):
        return dict(re.findall(r'(\w+)=([^\s]+)', s))
    try:
        start(['roscore', '-p', str(port)], 'master')
        def ready():
            try:
                return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/identity_fixture')[0] == 1
            except OSError:
                return False
        wait(ready)
        import rospy, yaml
        from std_msgs.msg import String
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from care_confidence_map.srv import QueryConfidence, QueryConfidenceResponse
        rospy.init_node('vbc_identity_driver', disable_signals=True)
        helpers = {}
        for version, base in [('old', frozen), ('fixed', ROOT)]:
            spec = importlib.util.spec_from_file_location('context_'+version, base/'src/egocentric_arm_planner/scripts/initial_shell_recovery.py')
            helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper); helpers[version] = helper
        summaries = {'old': [], 'fixed': []}; protocol = []; wire = {'old': [], 'fixed': []}
        known = rospy.Service('/identity/query', QueryConfidence, lambda r: QueryConfidenceResponse([1.]*len(r.points), [0.]*len(r.points), [1]*len(r.points)))
        pubs = {}; contexts = {}; subs = []
        for version, base in [('old', frozen), ('fixed', ROOT)]:
            prefix = '/identity/'+version
            cfg = yaml.safe_load((ROOT/'src/care_confidence_map/config/trajectory_risk.yaml').read_text())['trajectory_vbc']
            cfg.update(geometry_backend='primitive', robot_urdf_file=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
                primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                base_frame='base_link', input_trajectory_topic=prefix+'/task', predicted_trajectory_topic=prefix+'/candidate',
                prefer_predicted_trajectory=True, predicted_trajectory_timeout=.5, event_driven_eval=True, eval_rate=20.,
                predicted_periodic_refresh_rate=20., swept_volume_margin_m=.02, continuous_motion_bound_enabled=True,
                ignored_risk_links=['base_link','link1'], confidence_query_service='/identity/query',
                initial_shell_recovery_enabled=True, initial_shell_execution_margin_m=.01)
            rospy.set_param('/identity_'+version, {'trajectory_vbc': cfg})
            subs.append(rospy.Subscriber(prefix+'/summary', String, lambda m,v=version:summaries[v].append(tokens(m.data)), queue_size=50))
            subs.append(rospy.Subscriber(prefix+'/candidate', JointTrajectory, lambda m,v=version:wire[v].append(copy.deepcopy(m)), queue_size=50))
            pubs[version] = rospy.Publisher(prefix+'/candidate', JointTrajectory, queue_size=1, tcp_nodelay=True)
            contexts[version] = rospy.Publisher(prefix+'/candidate/initial_shell_context', String, queue_size=8, tcp_nodelay=True)
            start([str(base/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node'),
                '__name:=identity_'+version, '/care_planner/trajectory_risk/vbc_summary:='+prefix+'/summary'], version)
        subs.append(rospy.Subscriber('/identity/fixed/candidate/protocol_status', String, lambda m:protocol.append(tokens(m.data)), queue_size=50))
        wait(lambda: all(pubs[v].get_num_connections() >= 2 and contexts[v].get_num_connections() for v in pubs))
        time.sleep(.05)
        seq = {'old': 0, 'fixed': 0}
        def prepare(version, business, mutate=None):
            t = JointTrajectory(); t.header.frame_id = 'base_link'; t.header.stamp = rospy.Time.now(); t.header.seq = business
            t.joint_names = NAMES
            for s,q in [(0., [0.]*7), (.4, [.001]*7)]:
                t.points.append(JointTrajectoryPoint(positions=q, time_from_start=rospy.Duration(s)))
            raw = helpers[version].context(t,t.header.stamp.to_nsec()+500000000,False,[3.,3.,4.,4.,6.,6.,6.],.05,20)
            if mutate:
                raw = mutate(raw)
            return t, raw
        def send(version, t, raw, reverse=False):
            if reverse:
                pubs[version].publish(t); seq[version] += 1
                wait(lambda:any(x.get('trajectory_stamp_ns') == str(t.header.stamp.to_nsec()) and x.get('state') == 'waiting' for x in protocol), .2)
                contexts[version].publish(String(data=raw))
            else:
                contexts[version].publish(String(data=raw)); pubs[version].publish(t); seq[version] += 1
            return t.header.stamp.to_nsec()
        def result(version, stamp):
            return [x for x in summaries[version] if x.get('trajectory_stamp_ns') == str(stamp) and x.get('trajectory_source') == 'predicted']
        def accepted(label, business, reverse=False):
            t,raw = prepare('fixed', business); stamp = send('fixed', t, raw, reverse)
            wait(lambda:result('fixed', stamp), .25)
            row = result('fixed', stamp)[0]
            assert row['has_violation'] == '0' and row['context_request_sequence'] == str(business), row
            wait(lambda:any(m.header.stamp.to_nsec() == stamp for m in wire['fixed']), .1)
            actual = next(m for m in wire['fixed'] if m.header.stamp.to_nsec() == stamp)
            report['checks'].append(dict(label=label, result='safe_on_known_map', business_seq=business,
                wire_seq=actual.header.seq, stamp_ns=stamp, context=json.loads(raw), summary=row))
            return t, raw, stamp
        # Same preceding skip on both real publishers: no message was sent for
        # business seq 1; the first wire publication is seq 1, business seq 2.
        old,raw = prepare('old',2); stamp_old = send('old',old,raw)
        fixed,raw_fixed,stamp_fixed = accepted('upstream_skip_business2_wire1',2)
        time.sleep(.26)
        assert not result('old',stamp_old), result('old',stamp_old)
        assert wire['old'][0].header.seq == 1 and wire['fixed'][0].header.seq == 1
        report['checks'].append(dict(label='frozen_code_same_skip_has_no_certificate', result='no_certificate', business_seq=2, wire_seq=1))
        accepted('multiple_skips_business57_wire2',57)
        t,raw,stamp = accepted('trajectory_arrives_before_context',58,reverse=True)
        n = len(result('fixed',stamp)); contexts['fixed'].publish(String(data=raw)); time.sleep(.01)
        assert not any(x.get('trajectory_stamp_ns') == str(stamp) and x.get('state') == 'invalid' for x in protocol)
        report['checks'].append(dict(label='duplicate_identical_context_is_idempotent', result='valid'))
        # A foreign or late context must not decide the following trajectory.
        contexts['fixed'].publish(String(data=raw_fixed))
        accepted('old_context_does_not_decide_new_audit',59)
        def mutation(key, value):
            def change(raw):
                d=json.loads(raw);d[key]=value(d) if callable(value) else value
                return json.dumps(d,separators=(',',':'))
            return change
        def invalid(label, change, expected):
            t,raw = prepare('fixed', 100+len(report['checks']), change); stamp=send('fixed',t,raw)
            wait(lambda:any(x.get('trajectory_stamp_ns') == str(stamp) and x.get('state') == 'invalid' for x in protocol), .2)
            row=next(x for x in protocol if x.get('trajectory_stamp_ns') == str(stamp) and x.get('state') == 'invalid')
            assert row['reason'] == expected,row
            time.sleep(.01); assert not result('fixed', stamp)
            report['checks'].append(dict(label=label, result='protocol_invalid', status=row))
        def changed_q(d):
            q=copy.deepcopy(d['q']);q[1][0] += .01;return q
        invalid('changed_q_rejected',mutation('q',changed_q),'context_q_mismatch')
        invalid('changed_frame_rejected',mutation('frame','foreign'),'context_frame_mismatch')
        invalid('changed_times_rejected',mutation('times_s',[0.,.3]),'context_time_mismatch')
        invalid('expired_context_rejected',mutation('deadline_ns',lambda d:str(int(d['trajectory_stamp_ns'])-1)),'context_deadline_invalid')
        invalid('duplicate_identity_key_rejected',lambda raw:raw[:-1]+',"request_sequence":999}', 'missing_or_duplicate_field')
        invalid('unsupported_old_schema_rejected',mutation('schema',1),'unsupported_schema')
        # Conflicting content for one stamp remains invalid after replaying the
        # original context; feed both before that trajectory enters evaluation.
        t,raw = prepare('fixed',200)
        contexts['fixed'].publish(String(data=raw));contexts['fixed'].publish(String(data=mutation('frame','foreign')(raw)))
        contexts['fixed'].publish(String(data=raw));time.sleep(.02)
        pubs['fixed'].publish(t);seq['fixed'] += 1;stamp=t.header.stamp.to_nsec()
        wait(lambda:any(x.get('trajectory_stamp_ns') == str(stamp) and x.get('reason') == 'conflicting_context' for x in protocol),.2)
        assert not result('fixed',stamp)
        report['checks'].append(dict(label='conflicting_identity_cannot_be_revived_by_duplicate', result='protocol_invalid'))
        # A restart resets the ROS publisher sequence. Content/stamp identity
        # still works without learning or adjusting this topic counter.
        pubs['fixed'].unregister()
        pubs['fixed'] = rospy.Publisher('/identity/fixed/candidate',JointTrajectory,queue_size=1,tcp_nodelay=True)
        wait(lambda:pubs['fixed'].get_num_connections() >= 2)
        _,_,stamp=accepted('publisher_restart_wire_counter_reset',300)
        actual=next(m for m in wire['fixed'] if m.header.stamp.to_nsec()==stamp)
        assert actual.header.seq == 1, actual.header.seq
        report['status']='PASS'
        report['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'src/care_confidence_map/src/trajectory_vbc_selector_temporal_cluster_node.cpp',
            ROOT/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node',
            frozen/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node']}
    finally:
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        for stream in streams:stream.close()
        report['owned_remaining']=[p.pid for p in children if p.poll() is None]
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(report['checks']),owned_remaining=report['owned_remaining'])))


if __name__ == '__main__':
    main()
