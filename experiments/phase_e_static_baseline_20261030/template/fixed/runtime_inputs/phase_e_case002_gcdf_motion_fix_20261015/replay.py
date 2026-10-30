"""Exact production construction on a recorded raw/audit identity pair."""
from pathlib import Path
import argparse, csv, hashlib, json, sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src/egocentric_arm_planner/scripts')]
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import optimized_trajectory_continuity_node as module
from evaluate_phase_d_run import JOINTS, load_fk_helper, fk_T, pose_err, quat_rot
from urdf_parser_py.urdf import URDF

def trajectories(path):
    result = {}
    with path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 11 or not row[2].isdigit(): continue
            assert (len(row)-11)%22 == 0
            msg = JointTrajectory(joint_names=row[4:11])
            msg.header.stamp = rospy.Time(0, int(row[2]))
            msg.header.frame_id = row[3]
            for offset in range(11, len(row), 22):
                msg.points.append(JointTrajectoryPoint(
                    positions=[float(x) for x in row[offset:offset+7]],
                    velocities=[float(x) for x in row[offset+7:offset+14]],
                    accelerations=[float(x) for x in row[offset+14:offset+21]],
                    time_from_start=rospy.Duration(0, int(row[offset+21]))))
            result[row[2]] = msg
    return result

def build(raw, anchor, normal=False):
    n = module.OptimizedTrajectoryContinuityNode.__new__(module.OptimizedTrajectoryContinuityNode)
    n.repair_execution_prefix_s = .25
    n.repair_acceleration_limits = [3.,3.,4.,4.,6.,6.,6.]
    n.repair_brake_dt_s = .05
    n.repair_brake_max_steps = 20
    n.repair_hold_s = .1
    n._latest_measured_stamp_ns = n._latest_actuator_stamp_ns = 10000000000
    n._latest_measured_q_by_name = dict(zip(raw.joint_names, anchor.positions))
    n._latest_measured_velocity_by_name = dict(zip(raw.joint_names, anchor.velocities))
    # Controlled matching applied velocity; recorded audit q/v is the exact
    # construction anchor. This does not reproduce live transport scheduling.
    n._latest_actuator_velocity = dict(n._latest_measured_velocity_by_name)
    with patch.object(rospy.Time, 'now', return_value=rospy.Time(10)):
        if normal and hasattr(n, '_normal_with_goal_stop'):
            curve = n._normal_with_goal_stop(raw)
        else:
            curve = n._repair_prefix_with_braking_tail(raw, raw.points[-1].time_from_start.to_sec(), consistent_motion=True)
    assert curve is not None, n._last_prefix_build_reason
    for a,b in zip(curve.points, curve.points[1:]):
        dt = (b.time_from_start-a.time_from_start).to_sec()
        assert dt > 0
        assert all(abs(y-x-.5*dt*(v+w)) < 1e-9 for x,y,v,w in zip(a.positions,b.positions,a.velocities,b.velocities))
        assert all(abs(w-v) <= limit*dt+1e-9 for v,w,limit in zip(a.velocities,b.velocities,n.repair_acceleration_limits))
    assert curve.points[-1].velocities == [0.]*7
    return curve, n

def main():
    ap = argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--normal',action='store_true');args=ap.parse_args()
    run = ROOT/'outputs/c5_5_vbc_gcdf_regime/phase_e_goal_002_r1_failure_fixes_v4_full30_20261014/run'
    pairs = json.loads((ROOT/'outputs/phase_e_v4_failure_review_20261014/case002_normal_endpoint_comparison.json').read_text())['records']
    raws = trajectories(run/'raw_candidate_trajectory.csv');audits = trajectories(run/'audited_candidate_trajectory.csv')
    summary = json.loads((ROOT/'outputs/phase_e_r1_mainline_30/r1_failure_fixes_v4_full30_20261014/case_summaries/phase_e_goal_002.json').read_text())
    helper=load_fk_helper(str(ROOT));robot=URDF.from_xml_file(str(ROOT/'src/arm_description/urdf/Arm.urdf'))
    chain=helper.find_chain_joints(robot,'base_link','EE_link');goal=np.array(summary['goal_position']);rot=quat_rot(summary['goal_orientation_xyzw'])
    def error(p):
        a,b=pose_err(fk_T(helper,chain,dict(zip(JOINTS,p.positions))),goal,rot)
        return dict(position_error_m=float(a),orientation_error_rad=float(b))
    report=dict(scope='Production construction with recorded exact raw and audited initial q/v; controlled matching applied velocity. No live scheduler or historical full-map replay.', records=[])
    for pair in pairs[:2]:
        raw=raws[pair['raw']['stamp']];expected=audits[pair['audited']['stamp']]
        curve,n=build(raw,expected.points[0],args.normal)
        old_exact=len(curve.points)==len(expected.points) and all(max(abs(x-y) for x,y in zip(a.positions+a.velocities,b.positions+b.velocities))<1e-9 for a,b in zip(curve.points,expected.points))
        if not args.normal: assert old_exact
        prefix=min(curve.points,key=lambda p:abs(p.time_from_start.to_sec()-1.))
        item=dict(raw_stamp=pair['raw']['stamp'],audit_stamp=pair['audited']['stamp'],execution_stamp=pair['committed']['stamp'],outcome_ros_s=pair['outcome_ros_s'],baseline_exact=old_exact,
            raw_endpoint=error(raw.points[-1]),constructed_prefix_endpoint=error(prefix),constructed_stop=error(curve.points[-1]),
            ramp_shift_rad=[b-a for a,b in zip(raw.points[-1].positions,prefix.positions)],brake_shift_rad=[b-a for a,b in zip(prefix.positions,curve.points[-1].positions)],
            duration_s=curve.points[-1].time_from_start.to_sec(),brake_s=n._last_repair_brake_duration_s,
            knots=[dict(t=p.time_from_start.to_sec(),q=list(p.positions),v=list(p.velocities),a=list(p.accelerations)) for p in curve.points])
        report['records'].append(item)
        print(json.dumps({k:v for k,v in item.items() if k!='knots'}))
    report['inputs']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'raw_candidate_trajectory.csv',run/'audited_candidate_trajectory.csv']}
    (OUT/(args.label+'.json')).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
