#!/usr/bin/env python3
"""Real task owner and C++ reference/MPC lineage, with controlled CDF replies.

No actuator publisher. Empty CDF replies isolate asynchronous task ownership;
they are not geometric certificates or task success measurements.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import threading
import time
import traceback
import tarfile
import xmlrpc.client
sys.path.append('/usr/lib/python3/dist-packages')
import yaml
from phase_e_owned_processes import identity, remember_descendants, reap_owned
from verify_phase_e_leased_response_gazebo import protected

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--planner-prefix',type=Path,required=True)
    parser.add_argument('--port',type=int,default=12170)
    parser.add_argument('--with-observation',action='store_true')
    args=parser.parse_args()
    assert 12000 <= args.port <= 12998
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    before=protected()
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',args.port))!=0
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[];handles=[];owned={};done=threading.Event();thread=None
    events=[];references=[];candidates=[];summaries=[];queries=[];held=[];task_commands=[]
    reports=[];schedules=[];states=[];tf_trace=[];q=[0.]*7;tf_thread=None
    result=dict(status='FAIL',scope=__doc__,runtime_admission=False,production_issuer_enabled=False,
                new_formal_windows=0,checks=[])
    def start(argv,name):
        f=(out/(name+'.log')).open('x');handles.append(f)
        p=subprocess.Popen(list(map(str,argv)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);owned[p.pid]=identity(p.pid);return p
    def wait(check,label,seconds=15):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if check():return
            if any(p.poll() is not None for p in children):raise AssertionError('child exited: '+label)
            time.sleep(.01)
        raise AssertionError('timeout: '+label)
    # Freeze every runtime source and binary before starting processes.
    source_paths=set(json.loads((out.parent/'before_source_sha256.json').read_text()))
    source_paths.update(str(f.relative_to(ROOT)) for f in ROOT.glob('scripts/*task_context*'))
    source_paths.update(['src/care_visibility_cdf/scripts/task_observation_binding.py',
        'src/egocentric_arm_planner/scripts/observation_task_owner_node.py',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/task_observation_context.hpp'])
    source_paths.update(str(f.relative_to(ROOT)) for f in (args.planner_prefix.resolve()/'lib/egocentric_arm_planner').glob('*_planner_node'))
    binary_files=[f for f in (args.planner_prefix.resolve()/'lib/egocentric_arm_planner').glob('*_planner_node') if f.is_file()]
    dynamic_paths=set()
    for binary_file in binary_files:
        listing=subprocess.check_output(['ldd',str(binary_file)],text=True)
        for line in listing.splitlines():
            for token in line.split():
                if token.startswith(str(ROOT)) and Path(token).is_file():
                    dynamic_paths.add(str(Path(token).resolve().relative_to(ROOT)))
    source_paths.update(dynamic_paths)
    (out/'dynamic_project_objects.json').write_text(json.dumps(sorted(dynamic_paths),indent=2)+'\n')
    (out/'runtime_environment.json').write_text(json.dumps(dict(executable=sys.executable,**{
        key:os.environ.get(key,'') for key in ('CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','PYTHONPATH','ROS_PACKAGE_PATH','ROS_MASTER_URI')}),indent=2)+'\n')
    if args.with_observation:
        source_paths.update(str(f.relative_to(ROOT)) for f in (out.parent/'visibility_install').rglob('*.py'))
        source_paths.update(['outputs/phase_e_observation_ledger_20261020/native/confidence_map_node',
            'outputs/phase_e_trajectory_observation_20261020/native_final/leased_controller_host',
            'outputs/phase_e_trajectory_observation_20261020/native_final/libcare_velocity_controller.so',
            'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt'])
    input_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(source_paths) if (ROOT/name).is_file()}
    with tarfile.open(out/'input_sources.tar.gz','w:gz',dereference=True) as archive:
        for name in input_hashes:archive.add(ROOT/name,arcname=name,recursive=False)
    (out/'input_sha256.json').write_text(json.dumps(input_hashes,indent=2)+'\n')
    try:
        start(['roscore' ,'-p',args.port],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/task_context_probe')[0]==1
            except OSError:return False
        wait(ready,'master')
        import rospy
        from std_msgs.msg import String,Header,Bool
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory
        from egocentric_arm_planner.msg import CandidateTransfer
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch
        from verify_phase_e_observation_memory_ros import RobotFrames,JOINTS
        from stopping_trajectory_prefix import trajectory_record
        from task_observation_binding import canonical
        rospy.set_param('/use_sim_time',args.with_observation)
        rospy.init_node('task_context_probe',disable_signals=True)
        if args.with_observation:
            import tf2_ros
            from geometry_msgs.msg import TransformStamped
            from std_msgs.msg import Float64MultiArray
            from care_velocity_controller.msg import CommandStatus
            from std_srvs.srv import Trigger
            from phase_e_leased_stop_evidence import status_record
            map_config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml').read_text())
            robot=RobotFrames(map_config);broadcaster=tf2_ros.TransformBroadcaster()
            controller_config=yaml.safe_load((ROOT/'src/arm_description/config/controllers_watchdog.yaml').read_text())['care_arm']['arm_group_velocity_controller']
            controller_config['require_execution_lease']=True;rospy.set_param('/lease_test/controller',controller_config)
            rospy.Subscriber('/lease_test/controller/state',CommandStatus,lambda m:states.append(status_record(m)),queue_size=100)
            native=ROOT/'outputs/phase_e_trajectory_observation_20261020/native_final'
            start([native/'leased_controller_host',native/'libcare_velocity_controller.so'],'controller')
            wait(lambda:states and rospy.Time.now().to_nsec()>10_000_000_000,'real waiting controller')
            def sensor_frames():
                while not done.wait(.01):
                    stamp=rospy.Time.now();current=list(q);messages=[]
                    for row in robot.transforms(current):
                        m=TransformStamped();m.header.frame_id='base_link';m.child_frame_id=row['frame'];m.header.stamp=stamp
                        m.transform.translation.x,m.transform.translation.y,m.transform.translation.z=row['translation']
                        v=row['quaternion_xyzw'];m.transform.rotation.x,m.transform.rotation.y,m.transform.rotation.z,m.transform.rotation.w=v
                        messages.append(m)
                    broadcaster.sendTransform(messages);tf_trace.append(dict(stamp_ns=stamp.to_nsec(),q=current))
            tf_thread=threading.Thread(target=sensor_frames,daemon=True);tf_thread.start()
            map_config['confidence_map'].update(query_service='/task_map/query',observation_trace_file=str(out/'observations.jsonl'))
            map_config['current_body_prior'].update(geometry_backend='primitive',primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                refresh_service='/task_map/refresh',deactivate_service='/task_map/deactivate')
            native_map_config=copy.deepcopy(map_config)
            rospy.set_param('/task_map',map_config)
            map_process=start([ROOT/'outputs/phase_e_observation_ledger_20261020/native/confidence_map_node','__name:=task_map'],'map')
            rospy.wait_for_service('/task_map/refresh',timeout=15.)
            assert rospy.ServiceProxy('/task_map/refresh',Trigger)().success
            rospy.set_param('/observation_task_owner/snapshot_service','/task_map/query_snapshot')
            visibility=out.parent/'visibility_install'
            params=dict(checkpoint=str(ROOT/'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt'),
                urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),device='cpu',enable_oracle_diagnostics=False,
                output_root=str(out/'scheduler_data'),region_schedule_mode='accumulated_multi_deadline',
                enable_stopping_observation_demands=True,enable_observation_context=True,require_task_binding=True,
                task_candidate_topic='/care_planner/local_planner/candidate_trajectory/planned',
                predicted_trajectory_topic='/care_planner/local_planner/candidate_trajectory',predicted_trajectory_timeout=1.,
                trajectory_topic='/task_unused_reference',confidence_query_service='/task_map/query',
                joint_state_topic='/care_arm/joint_states',candidate_vbc_summary_topic='/task_unused_verdict',
                schedule_topic='/task/schedule',schedule_summary_topic='/task/schedule_summary',rate=20.,max_obligations=12)
            rospy.set_param('/task_scheduler',params)
            start([sys.executable,visibility/'lib/care_visibility_cdf/vbc_deadline_waypoint_online_node.py',
                '__name:=task_scheduler','/task_scheduler/stopping_observation_demands:=/care_planner/active_sensing/stopping_observation_demands'],'scheduler')
            rospy.Subscriber('/task/schedule',Float64MultiArray,lambda m:schedules.append(dict(ns=rospy.Time.now().to_nsec(),data=list(m.data))),queue_size=100)
            wait(lambda:'ONLINE WARMUP READY' in (out/'scheduler.log').read_text(),'scheduler warmup',60.)
            oc=dict(enable_observation_context=True,require_task_binding=True,
                context_topic='/care_planner/active_sensing/observation_context',
                trajectory_topic='/care_planner/local_planner/candidate_trajectory',
                output_topic='/care_planner/active_sensing/stopping_observation_demands',
                state_topic='/lease_test/controller/state',lease_topic='/lease_test/controller/execution_lease',
                query_service='/task_map/query_snapshot',horizon_s=2.,admission_wait_s=.8,input_max_age_s=.8,
                wall_budget_s=.5,lease_s=.1,update_gap_s=.001,max_braking_step_s=.02,
                target_abs_velocity=[.06,.03,.15,.08,.4,.4,.4],
                observation_timing=dict(motion_s=2.,acquisition_s=.1,verification_s=.05,stop_reserve_s=.2))
            (out/'observer_config.json').write_text(json.dumps(oc,indent=2)+'\n')
            rospy.Subscriber(oc['output_topic'],String,lambda m:reports.append(json.loads(m.data)),queue_size=100)
            start([sys.executable,ROOT/'scripts/phase_e_trajectory_observer_node.py','--diagnostic-private-master',
                '--config',out/'observer_config.json','--output-dir',out/'observer'],'observer')
        config=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        config['enable_observation_context']=True
        config['urdf_path']=str(ROOT/'src/arm_description/urdf/Arm.urdf')
        config['local_planner']['terminal_retention']['enabled']=False
        config['local_planner']['primitive_observation_guard']=dict(enabled=False)
        rospy.set_param('/robot_description',(ROOT/'src/arm_description/urdf/Arm.urdf').read_text())
        rospy.set_param('/task_reference',config);rospy.set_param('/task_mpc',config)
        prefix=args.planner_prefix.resolve()/'lib/egocentric_arm_planner'
        request=rospy.Publisher('/care_planner/task_request',String,queue_size=32)
        joint=rospy.Publisher(config['local_planner']['joint_states'],JointState,queue_size=2)
        # The nominal generator uses the top-level configured joint state topic.
        nominal_joint=rospy.Publisher(config.get('topics',{}).get('joint_states','/care_arm/joint_states'),JointState,queue_size=2)
        batch=rospy.Publisher(config['local_planner']['cdf_batch_topic'],Batch,queue_size=16)
        replan=rospy.Publisher('/care_planner/local_planner/replan_request',Bool,queue_size=4)
        repair=rospy.Publisher('/care_planner/execution/predicted_vbc_recovery_triggered',Bool,queue_size=4,latch=True)
        refresh=rospy.Publisher('/care_planner/local_planner/normal_task_replan_request',Bool,queue_size=4)
        control=dict(hold=False)
        def reply(msg):batch.publish(Batch(header=msg.header,num_pairs=0,dof=7))
        def query(msg):
            queries.append(dict(stamp_ns=msg.header.stamp.to_nsec(),received_ns=rospy.Time.now().to_nsec()))
            if control['hold']:held.append(copy.deepcopy(msg))
            else:reply(msg)
        def candidate(msg):
            candidates.append(dict(received_ns=rospy.Time.now().to_nsec(),binding=msg.task_context,
                trajectory=trajectory_record(msg.trajectory)))
        rospy.Subscriber('/care_planner/task_request',String,lambda m:task_commands.append(
            dict(received_ns=rospy.Time.now().to_nsec(),command=json.loads(m.data))),queue_size=256)
        rospy.Subscriber('/care_planner/active_sensing/observation_context',String,
                         lambda m:events.append(dict(received_ns=rospy.Time.now().to_nsec(),event=json.loads(m.data))),queue_size=128)
        rospy.Subscriber('/care_planner/task_trajectory/bound',CandidateTransfer,
                         lambda m:references.append(dict(binding=m.task_context,trajectory=trajectory_record(m.trajectory))),queue_size=32)
        rospy.Subscriber(config['local_planner']['candidate_trajectory_topic']+'/planned',CandidateTransfer,candidate,queue_size=64)
        rospy.Subscriber(config['local_planner']['query_trajectory_topic'],JointTrajectory,query,queue_size=64)
        rospy.Subscriber(config['local_planner']['summary_topic'],String,lambda m:summaries.append(m.data),queue_size=128)
        start(['/usr/bin/python3',ROOT/'src/egocentric_arm_planner/scripts/observation_task_owner_node.py'],'owner')
        start([prefix/'receding_horizon_planner_node','__name:=task_reference'],'reference')
        start([prefix/'local_sparse_scp_planner_node','__name:=task_mpc'],'mpc')
        def joints():
            while not done.wait(.01):
                m=JointState(header=Header(stamp=rospy.Time.now()),name=JOINTS,position=list(q),velocity=[0.]*7)
                joint.publish(m);nominal_joint.publish(m)
        thread=threading.Thread(target=joints,daemon=True);thread.start()
        wait(lambda:request.get_num_connections()>=2 and joint.get_num_connections() and batch.get_num_connections(),'owner plus diagnostic request subscribers')
        map_config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_case001_motion_prior.yaml').read_text())
        goal_robot=RobotFrames(map_config)
        def submit(label,goal_q):
            # Compute an actually reachable EE target from independent FK.
            full=goal_robot.data
            import numpy as np,pinocchio as pin
            values=np.zeros(goal_robot.model.nq);values[goal_robot.indices]=goal_q
            pin.forwardKinematics(goal_robot.model,full,values);pin.updateFramePlacements(goal_robot.model,full)
            pose=full.oMf[goal_robot.model.getFrameId('base_link')].inverse()*full.oMf[goal_robot.model.getFrameId('EE_link')]
            c=dict(schema=1,action='submit',request_id=label,target=dict(frame_id='base_link',
                position=pose.translation.tolist(),orientation=pin.Quaternion(pose.rotation).coeffs().tolist()),
                region_min=[-.95,-.95,0.],region_max=[.95,.95,1.15])
            request.publish(String(data=canonical(c)));return c
        loaded={}
        for child in children:
            paths=set()
            try:
                for line in Path('/proc/'+str(child.pid)+'/maps').read_text().splitlines():
                    pieces=line.split()
                    if len(pieces)>=6 and pieces[-1].startswith(str(ROOT)) and Path(pieces[-1]).is_file():
                        paths.add(pieces[-1])
            except FileNotFoundError:pass
            loaded[str(child.pid)]={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(paths)}
        (out/'loaded_project_objects.json').write_text(json.dumps(loaded,indent=2)+'\n')
        a=submit('A',[0.,-.28,0.,0.,0.,0.,0.])
        wait(lambda:bool(candidates),'A MPC candidate')
        binding_a=candidates[-1]['binding'];first=copy.deepcopy(events[-1]['event'])
        assert json.loads(binding_a)['goal_request_id']=='A'
        geometry=[]
        for step in range(4):
            q[1]=-.015*step
            before_count=len(candidates);replan.publish(Bool(data=True))
            wait(lambda:len(candidates)>before_count,'changed-geometry MPC candidate')
            geometry.append(candidates[-1]['trajectory'])
        motion_shapes={canonical(dict(joint_names=t['joint_names'],points=t['points'])) for t in geometry}
        assert len(motion_shapes)>1
        assert all(c['binding']==binding_a for c in candidates)
        result['checks'].append('same_goal_real_mpc_changes_geometry_with_stable_binding')
        before_count=len(references)
        for _ in range(100):request.publish(String(data=canonical(a)));time.sleep(.002)
        time.sleep(.2)
        assert len(references)==before_count
        assert all(e['event']['candidate_binding']==binding_a for e in events)
        result['checks'].append('hundred_duplicate_requests_do_not_restart_nominal_or_context')
        if args.with_observation:
            # Keep replacing actual MPC geometry while the sensor remains at
            # its declared kinematic pose. No fabricated SAFE verdict clears work.
            for step in range(8):
                replan.publish(Bool(data=True));time.sleep(.15)
            wait(lambda:any(r.get('cells') for r in reports),'real-map prefix report')
            assert all(r.get('task_context')==binding_a for r in reports if 'cells' in r)
            wait(lambda:any(s['data'] for s in schedules),'bound learned observation schedule',10.)
            result['checks'].append('real_map_and_learned_scheduler_consume_bound_real_mpc_geometry')
            result['observation']=dict(reports=len(reports),nonempty_schedules=sum(bool(s['data']) for s in schedules),
                pending_peak=max(sum(c['action']!='covered' for c in r['cells']) for r in reports if 'cells' in r),
                controller_always_waiting=all(s['mode']==0 and s['applied']==[0.]*7 for s in states))
            assert result['observation']['controller_always_waiting']

        # A measured-state reference refresh is distinct from a new goal.
        reference_marker=len(references);refresh.publish(Bool(data=True))
        wait(lambda:len(references)>reference_marker,'same-task nominal refresh')
        assert references[-1]['binding']==binding_a
        result['checks'].append('explicit_same_task_reference_refresh_keeps_context')
        if args.with_observation:
            # Entering the existing recovery mode must retain the very work
            # that requested sensing, even while the solver requests new CDF.
            previous_ids={int(schedules[-1]['data'][j]) for j in range(0,len(schedules[-1]['data']),16)}
            repair.publish(Bool(data=True));time.sleep(.2)
            assert events[-1]['event']['candidate_binding']==binding_a
            assert schedules[-1]['data']
            assert previous_ids & {int(schedules[-1]['data'][j]) for j in range(0,len(schedules[-1]['data']),16)}
            repair.publish(Bool(data=False));replan.publish(Bool(data=True));time.sleep(.15)
            result['checks'].append('normal_to_repair_keeps_goal_context_and_pending_observation_ids')
            # Route abandonment is an explicit cancellation, not completion.
            old_report=copy.deepcopy(next(r for r in reversed(reports) if r.get('cells')))
            prior_context=events[-1]['event']['context_id']
            request.publish(String(data=canonical(dict(a,action='replace_route',context_id=prior_context))))
            wait(lambda:events[-1]['event']['context_id']!=prior_context,'explicit route generation')
            binding_a=events[-1]['event']['candidate_binding']
            wait(lambda:any(r.get('task_context')==binding_a for r in reports),'new-route report')
            delayed=rospy.Publisher('/care_planner/active_sensing/stopping_observation_demands',String,queue_size=8)
            delayed.publish(String(data=canonical(old_report)))
            wait(lambda:bool(schedules[-1]['data']),'new-route schedule')
            result['checks'].append('explicit_route_abandonment_replaces_context_without_claiming_observation')
            # Restart the actual map. The owner closes the old context and
            # verifies a fresh native snapshot before allowing a new generation.
            old_report=copy.deepcopy(next(r for r in reversed(reports) if r.get('task_context')==binding_a))
            os.kill(map_process.pid,signal.SIGINT);map_process.wait(timeout=5.)
            children.remove(map_process)
            native_map_config['confidence_map']['observation_trace_file']=str(out/'observations_restarted.jsonl')
            rospy.set_param('/task_map',native_map_config)
            map_process=start([ROOT/'outputs/phase_e_observation_ledger_20261020/native/confidence_map_node',
                               '__name:=task_map'],'map_restarted')
            wait(lambda:any(e['event'].get('reason')=='map_invalidated' for e in events),'owner map invalidation')
            wait(lambda:schedules and not schedules[-1]['data'],'map invalidation clears schedule')
            closed=copy.deepcopy(events[-1]['event'])
            delayed.publish(String(data=canonical(old_report)));time.sleep(.1)
            assert not schedules[-1]['data']
            rospy.wait_for_service('/task_map/refresh',timeout=15.)
            assert rospy.ServiceProxy('/task_map/refresh',Trigger)().success
            request.publish(String(data=canonical(dict(schema=1,action='resume_map',request_id='A',
                context_id=closed['context_id']))))
            wait(lambda:events[-1]['event']['active'] and events[-1]['event']['context_id']!=closed['context_id'],
                 'verified fresh-map generation')
            binding_a=events[-1]['event']['candidate_binding']
            wait(lambda:any(r.get('task_context')==binding_a for r in reports),'fresh-map candidate report')
            result['checks'].append('actual_map_restart_closes_owner_and_verified_snapshot_starts_new_generation')
        # Hold the real solver's CDF transaction across an actual goal replacement.
        control['hold']=True;replan.publish(Bool(data=True));wait(lambda:bool(held),'inflight A query')
        marker=len(candidates);submit('B',[.12,-.18,0.,0.,0.,0.,0.])
        wait(lambda:any(e['event']['goal_request_id']=='B' and e['event']['active'] for e in events),'B context')
        b=next(e['event'] for e in reversed(events) if e['event']['goal_request_id']=='B')
        time.sleep(.05);old=list(held);held.clear();control['hold']=False
        for query_message in old:reply(query_message)
        wait(lambda:any(json.loads(c['binding'])['goal_request_id']=='B' for c in candidates[marker:]),'B candidate')
        assert all(json.loads(c['binding'])['goal_request_id']=='B' for c in candidates[marker:])
        result['checks'].append('inflight_A_query_reply_cannot_publish_into_B')
        marker=len(candidates)
        request.publish(String(data=canonical(dict(schema=1,action='cancel',request_id='B',context_id=b['context_id']))))
        wait(lambda:events[-1]['event']['active'] is False,'cancel event')
        time.sleep(.15);marker=len(candidates)
        for query_message in old:reply(query_message)
        replan.publish(Bool(data=True));time.sleep(.3)
        assert len(candidates)==marker
        result['checks'].append('explicit_cancel_late_queries_and_replan_do_not_revive_task')
        result.update(status='PASS',real_mpc_candidates=len(candidates),distinct_motion_shapes=len(motion_shapes),
                      source='actual owner + actual nominal IK/reference + actual sparse SCP; empty controlled CDF replies')
        (out/'planner_config.json').write_text(json.dumps(config,indent=2)+'\n')
    except Exception as exc:
        result.update(error=str(exc),traceback=traceback.format_exc())
    finally:
        done.set()
        if thread:thread.join(timeout=2)
        if tf_thread:tf_thread.join(timeout=2)
        remember_descendants({p.pid for p in children if p.poll() is None},owned)
        result['cleanup']=reap_owned(owned)
        for p in children:
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:pass
        for f in handles:f.close()
        result['inputs_unchanged']=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in input_hashes.items())
        result['protected_identity_equal']=protected()==before
        with socket.socket() as s:result['port_released']=s.connect_ex(('127.0.0.1',args.port))!=0
        for name,value in [('events',events),('references',references),('candidates',candidates),('summaries',summaries),('queries',queries),('reports',reports),('schedules',schedules),('states',states),('tf_trace',tf_trace),('task_commands',task_commands)]:
            (out/(name+'.json')).write_text(json.dumps(value)+'\n')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['status']=='PASS' and not result['cleanup']['remaining'] and result['protected_identity_equal'] and result['port_released']


if __name__=='__main__':main()
