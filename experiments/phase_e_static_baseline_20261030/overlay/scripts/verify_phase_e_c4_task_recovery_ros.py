#!/usr/bin/env python3
"""Automatic task-bound C4 recovery with real reference/MPC/map/learned acquisition.

No actuator publisher. Controlled empty batches and constant-positive witness
replies isolate ownership/protocols; they are not geometric certificates or case scores.
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
    parser.add_argument('--with-observation',action='store_true',default=True)
    parser.add_argument('--actual-cdf',action='store_true')
    parser.add_argument('--per-sensor',action='store_true')
    parser.add_argument('--with-verification',action='store_true')
    parser.add_argument('--with-handoff',action='store_true',help='Actual prepared tracker ZERO-only handshake with synthetic zero leases')
    parser.add_argument('--require-bound-policy',action='store_true',help='Require measured, acknowledged frontier/guidance on actual strict native candidates')
    parser.add_argument('--schedule-mode',choices=['accumulated_multi_deadline','blocker_aware_acquisition'],default='blocker_aware_acquisition')
    args=parser.parse_args()
    assert 12000 <= args.port <= 12998
    assert not args.with_verification or args.actual_cdf
    assert not args.with_handoff or args.with_verification
    assert not args.require_bound_policy or (args.with_handoff and args.per_sensor)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    cdf_socket=out/'cdf.sock'
    if len(str(cdf_socket).encode())>100:cdf_socket=out.parent/('cdf_'+str(args.port)+'.sock')
    if args.actual_cdf:assert not cdf_socket.exists(),'CDF socket already owned' 
    before=protected()
    with socket.socket() as s:assert s.connect_ex(('127.0.0.1',args.port))!=0
    uri='http://127.0.0.1:'+str(args.port)
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[];handles=[];owned={};done=threading.Event();thread=None
    events=[];references=[];candidates=[];summaries=[];queries=[];held=[];task_commands=[]
    snapshots=[];regime_controls=[];regime_summaries=[];injections=[];witness_requests=[];witness_replies=[];held_witnesses=[];actual_batches=[];actual_witnesses=[];selector_summaries=[];verification_events=[];committed=[];vbc_summaries=[];gate_summaries=[];continuity_summaries=[];prepared=[];preparation_status=[];final_batches=[];final_queries=[];verification_queries=[]
    reports=[];schedules=[];states=[];tf_trace=[];q=[0.]*7;tf_thread=None
    handoff_status=[];handoff_shadow=[];handoff_desired=[];handoff_limited=[];handoff_leases=[];native_policy_events=[]
    result=dict(status='FAIL',scope=__doc__,runtime_admission=False,production_issuer_enabled=False,
                new_formal_windows=0,checks=[])
    if args.with_handoff:
        result['scope']='Actual R1/CUDA CDF/final GCDF/VBC and installed tracker/limiter/plugin zero application; prescribed joints/TF and synthetic zero-only leases. No qualified issuer, nonzero motion, Gazebo recovery or formal case score.'
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
    source_paths.update(str(f.relative_to(ROOT)) for f in ROOT.glob('scripts/*task*context*'))
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
    if args.per_sensor:
        source_paths.add('src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt')
    if args.actual_cdf:
        actual_files=['devel/lib/care_collision_cdf/cpp_forbidden_voxel_gpu_shadow_node',
            'devel/lib/care_confidence_map/gcdf_primitive_anchor_node',
            'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py',
            'src/care_collision_cdf/scripts/collision_cdf_model.py',
            'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt',
            'src/care_collision_cdf/config/cdf_training_contract.json']
        if args.with_verification:actual_files.append('devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node')
        source_paths.update(actual_files)
        source_paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'src/care_collision_cdf/scripts').glob('*.py'))
        for binary_name in actual_files[:2]+(actual_files[-1:] if args.with_verification else []):
            for token in subprocess.check_output(['ldd',str(ROOT/binary_name)],text=True).split():
                if token.startswith(str(ROOT)) and Path(token).is_file():
                    dynamic_paths.add(str(Path(token).resolve().relative_to(ROOT)))
    source_paths.update(dynamic_paths)
    if args.with_handoff:
        source_paths.update(['src/egocentric_arm_planner/scripts/execution_handoff.py',
            'src/egocentric_arm_planner/scripts/execution_preparation_handoff_node.py',
            'src/egocentric_arm_planner/include/egocentric_arm_planner/prepared_execution_state.hpp'])
        source_paths.update(str(p.relative_to(ROOT)) for p in
            (args.planner_prefix.resolve()/'lib/egocentric_arm_planner').glob('*.py'))
        tracker_binary=args.planner_prefix.resolve()/'lib/egocentric_arm_planner/trajectory_execution_manager_node'
        source_paths.add(str(tracker_binary.relative_to(ROOT)))
        for token in subprocess.check_output(['ldd',str(tracker_binary)],text=True).split():
            if token.startswith(str(ROOT)) and Path(token).is_file():source_paths.add(str(Path(token).resolve().relative_to(ROOT)))
    source_paths.update([str(Path(__file__).resolve().relative_to(ROOT)), 'scripts/test_phase_e_task_recovery.py', 'src/care_visibility_cdf/scripts/task_recovery_events.py','src/egocentric_arm_planner/scripts/execution_preparation.py'])
    source_paths.update(['src/care_visibility_cdf/scripts/recovery_policy.py',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/task_recovery_policy.hpp',
        'scripts/test_phase_e_recovery_policy.py','scripts/test_phase_e_recovery_policy_native.cpp'])
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
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch, CollisionCDFWitnessRequest, CollisionCDFWitnessResponse
        from verify_phase_e_observation_memory_ros import RobotFrames,JOINTS
        from stopping_trajectory_prefix import trajectory_record
        from task_observation_binding import canonical
        from candidate_admission import payload_sha256
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
                per_sensor_hybrid_enabled=args.per_sensor,
                per_sensor_checkpoint=str(ROOT/'src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt'),
                per_sensor_allow_nominal_proposals=True,per_sensor_high_witness_priority_enabled=True,
                per_sensor_high_witness_z_min=.85,task_rejoin_limits_namespace='/task_mpc/mpc',
                joint_velocity_limits=controller_config['velocity_limits'],
                joint_acceleration_limits=controller_config['acceleration_limits'],
                output_root=str(out/'scheduler_data'),region_schedule_mode=args.schedule_mode,
                enable_stopping_observation_demands=True,enable_observation_context=True,require_task_binding=True,
                task_candidate_topic='/care_planner/local_planner/candidate_trajectory/planned',
                predicted_trajectory_topic='/care_planner/local_planner/candidate_trajectory',predicted_trajectory_timeout=1.,
                trajectory_topic='/task_unused_reference',confidence_query_service='/task_map/query',
                joint_state_topic='/care_arm/joint_states',candidate_vbc_summary_topic='/task_unused_verdict',
                schedule_topic='/task/schedule',schedule_summary_topic='/task/schedule_summary',rate=20.,max_obligations=12)
            rospy.set_param('/task_scheduler',params)
            if args.require_bound_policy:
                rospy.set_param('/task_scheduler/rolling_visibility_guidance_mode','drive')
                rospy.set_param('/task_scheduler/vbc_gated_frontier_step_enabled',True)
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
        config['require_bound_regime']=True
        if args.require_bound_policy:
            config['local_planner']['rolling_visibility_guidance_enabled']=True
            config['local_planner']['require_bound_recovery_policy']=True
        config['urdf_path']=str(ROOT/'src/arm_description/urdf/Arm.urdf')
        if not args.actual_cdf:
            config['local_planner']['terminal_retention']['enabled']=False
            config['local_planner']['primitive_observation_guard']=dict(enabled=False)
        else:
            config['local_planner']['primitive_observation_guard']=dict(enabled=True,
                fk_urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
                primitive_urdf=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                base_frame='base_link',ignored_links=['base_link','link1'],margin_m=.020)

        rospy.set_param('/robot_description',(ROOT/'src/arm_description/urdf/Arm.urdf').read_text())
        rospy.set_param('/task_reference',config);rospy.set_param('/task_mpc',config)
        prefix=args.planner_prefix.resolve()/'lib/egocentric_arm_planner'
        request=rospy.Publisher('/care_planner/task_request',String,queue_size=32)
        joint=rospy.Publisher(config['local_planner']['joint_states'],JointState,queue_size=2)
        # The nominal generator uses the top-level configured joint state topic.
        nominal_joint=rospy.Publisher(config.get('topics',{}).get('joint_states','/care_arm/joint_states'),JointState,queue_size=2)
        batch=rospy.Publisher(config['local_planner']['cdf_batch_topic'],Batch,queue_size=16)
        witness_pub=rospy.Publisher(config['local_planner']['cdf_batch_topic']+'/witness_response',CollisionCDFWitnessResponse,queue_size=16)
        replan=rospy.Publisher('/care_planner/local_planner/replan_request',Bool,queue_size=4)
        repair=rospy.Publisher('/care_planner/execution/predicted_vbc_recovery_triggered',Bool,queue_size=4,latch=True)
        refresh=rospy.Publisher('/care_planner/local_planner/normal_task_replan_request',Bool,queue_size=4)
        control=dict(hold=False)
        def reply(msg):
            if not args.actual_cdf:batch.publish(Batch(header=msg.header,num_pairs=0,dof=7))
        def query(msg):
            queries.append(dict(stamp_ns=msg.header.stamp.to_nsec(),received_ns=rospy.Time.now().to_nsec()))
            if control['hold']:held.append(copy.deepcopy(msg))
            else:reply(msg)
        def witness_reply(request_message):
            count=len(request_message.original_timestep)
            # Explicit synthetic rows, never a claim that UNKNOWN became free.
            value=CollisionCDFWitnessResponse(request=request_message,map_stamp=rospy.Time.now(),
                status=['evaluated_unknown']*count,
                batch=Batch(header=request_message.header,num_pairs=count,dof=7,
                    original_timestep=list(request_message.original_timestep),source_type=[0]*count,
                    point_flat=list(request_message.point_flat),q_linearization_flat=list(request_message.q_flat),
                    distance=[1.]*count,gradient_flat=[0.]*(count*7),approx_body_clearance_m=[1.]*count))
            witness_pub.publish(value)
            witness_replies.append(dict(stamp_ns=value.request.header.stamp.to_nsec(),pairs=count,
                                        scope='synthetic_constant_positive_protocol_rows'))
        def witness(request_message):
            witness_requests.append(dict(stamp_ns=request_message.header.stamp.to_nsec(),pairs=len(request_message.original_timestep)))
            if control['hold']:held_witnesses.append(copy.deepcopy(request_message))
            elif not args.actual_cdf:witness_reply(request_message)
        rospy.Subscriber(config['local_planner']['cdf_batch_topic']+'/witness_request',
                         CollisionCDFWitnessRequest,witness,queue_size=32)
        def candidate(msg):
            candidates.append(dict(received_ns=rospy.Time.now().to_nsec(),binding=msg.task_context,
                regime=msg.task_regime,prefix_context=msg.planned_prefix_context,trajectory=trajectory_record(msg.trajectory),
                payload_sha256=payload_sha256(msg.trajectory,msg.planned_prefix_context,msg.task_context,msg.task_regime)))
        if args.actual_cdf:
            checkpoint=ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'
            worker=start([sys.executable,ROOT/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py',
                '--checkpoint',checkpoint,'--activation','gelu','--device','cuda',
                '--socket',cdf_socket,'--max-pairs','8000'],'cdf_worker')
            wait(lambda:(cdf_socket).exists(),'actual CDF CUDA worker',60.)
            anchor_topic='/task_actual_cdf/anchors'
            rospy.set_param('/task_actual_anchors',dict(
                robot_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                base_frame='base_link',input_trajectory_topic=config['local_planner']['query_trajectory_topic'],
                output_topic=anchor_topic,body_inflation_m=.015,max_eval_timesteps=50,
                ignored_risk_links=['base_link','link1']))
            rospy.set_param('/task_actual_selector',dict(anchor_topic=anchor_topic,
                geometry_backend='primitive',map_topic='/care_planner/confidence_map/points',
                constraint_batch_topic=config['local_planner']['cdf_batch_topic'],
                summary_topic='/task_actual_cdf/summary',output_jsonl=str(out/'selector.jsonl'),
                gpu_socket=str(cdf_socket),rate=50.,process_on_input_callback=True,
                local_witness_requests_required=True,confidence_threshold=.8,
                map_resolution=.05,proximity_margin=.025,max_pairs_per_step=250,
                signed_zero_band=.05,max_pairs=8000))
            def batch_record(message):
                fields=['num_pairs','dof','original_timestep','source_type','point_flat',
                    'q_linearization_flat','distance','gradient_flat','approx_body_clearance_m',
                    'pair_selection_ms','ipc_roundtrip_ms','gpu_worker_total_ms','online_pipeline_ms']
                row=dict(stamp_ns=message.header.stamp.to_nsec(),received_ns=rospy.Time.now().to_nsec())
                for field in fields:
                    value=getattr(message,field)
                    row[field]=list(value) if not isinstance(value,(int,float)) else value
                return row
            rospy.Subscriber(config['local_planner']['cdf_batch_topic'],Batch,
                lambda m:actual_batches.append(batch_record(m)),queue_size=64)
            rospy.Subscriber(config['local_planner']['cdf_batch_topic']+'/witness_response',CollisionCDFWitnessResponse,
                lambda m:actual_witnesses.append(dict(request_stamp_ns=m.request.header.stamp.to_nsec(),
                    map_stamp_ns=m.map_stamp.to_nsec(),plan_sequence=m.request.plan_sequence,
                    mode_epoch=m.request.mode_epoch,status=list(m.status),batch=batch_record(m.batch))),queue_size=64)
            rospy.Subscriber('/task_actual_cdf/summary',String,lambda m:selector_summaries.append(m.data),queue_size=128)
            if args.with_verification:
                rospy.Subscriber('/care_planner/final_gcdf/constraint_batch',Batch,lambda m:final_batches.append(batch_record(m)),queue_size=64)
                rospy.Subscriber('/care_planner/final_gcdf/query_trajectory',JointTrajectory,lambda m:final_queries.append(trajectory_record(m)),queue_size=64)
                rospy.Subscriber('/care_planner/optimized_trajectory',JointTrajectory,lambda m:verification_queries.append(trajectory_record(m)),queue_size=64)
            start([ROOT/'devel/lib/care_confidence_map/gcdf_primitive_anchor_node','__name:=task_actual_anchors'],'anchors')
            start([ROOT/'devel/lib/care_collision_cdf/cpp_forbidden_voxel_gpu_shadow_node','__name:=task_actual_selector'],'selector')
        rospy.Subscriber('/care_planner/task_request',String,lambda m:task_commands.append(
            dict(received_ns=rospy.Time.now().to_nsec(),command=json.loads(m.data))),queue_size=256)
        rospy.Subscriber('/care_planner/active_sensing/observation_context',String,
                         lambda m:events.append(dict(received_ns=rospy.Time.now().to_nsec(),event=json.loads(m.data))),queue_size=128)
        rospy.Subscriber('/care_planner/task_trajectory/bound',CandidateTransfer,
                         lambda m:references.append(dict(binding=m.task_context,trajectory=trajectory_record(m.trajectory))),queue_size=32)
        rospy.Subscriber(config['local_planner']['candidate_trajectory_topic']+'/planned',CandidateTransfer,candidate,queue_size=64)
        rospy.Subscriber(config['local_planner']['query_trajectory_topic'],JointTrajectory,query,queue_size=64)
        rospy.Subscriber(config['local_planner']['summary_topic'],String,lambda m:summaries.append(m.data),queue_size=128)
        rospy.Subscriber('/care_planner/local_planner/task_event',String,
            lambda m:native_policy_events.append(json.loads(m.data)),queue_size=128)
        if args.with_verification:
            raw_topic=config['local_planner']['candidate_trajectory_topic']
            if args.with_handoff:
                from care_velocity_controller.msg import ExecutionLease,LeasedVelocityCommand
                from phase_e_leased_stop_evidence import command_record
                zero_publisher=rospy.Publisher('/lease_test/controller/execution_lease',ExecutionLease,queue_size=10)
                issued_preparations=set()
                def handoff_state(message):
                    value=json.loads(message.data);handoff_status.append(value)
                    key=(value.get('preparation_session'),value.get('preparation_sequence'))
                    if (value.get('pending') and value['state']=='waiting_zero_authorization' and
                            key not in issued_preparations and zero_publisher.get_num_connections()>=2):
                        now=rospy.Time.now();until=min(now.to_nsec()+80000000,value['expires_ns'])
                        if until-now.to_nsec()<10000000:return
                        grant=ExecutionLease();grant.controller_session=value['controller_session']
                        grant.execution_stamp_ns=value['execution_stamp_ns'];grant.lease_sequence=len(issued_preparations)+1
                        grant.issued_at=now;grant.valid_until=rospy.Time(0,until);grant.target_range_enforced=True
                        grant.target_abs_velocity=[0.]*7;grant.certificate_sha256='b'*64
                        issued_preparations.add(key);zero_publisher.publish(grant)
                        handoff_leases.append(dict(preparation_session=key[0],preparation_sequence=key[1],
                            token=grant.execution_stamp_ns,issued_ns=now.to_nsec(),until_ns=until,synthetic_fixture=True))
                rospy.Subscriber('/care_planner/execution/tracker_preparation_status',String,handoff_state,queue_size=100)
                rospy.Subscriber('/care_planner/execution/handoff_status',String,lambda m:handoff_shadow.append(json.loads(m.data)),queue_size=100)
                rospy.Subscriber('/handoff_actual/desired',LeasedVelocityCommand,lambda m:handoff_desired.append(command_record(m)),queue_size=100)
                rospy.Subscriber('/lease_test/controller/command_leased',LeasedVelocityCommand,lambda m:handoff_limited.append(command_record(m)),queue_size=100)
                tracker_config=copy.deepcopy(config)
                tracker_config['execution'].update(require_command_lease=True,require_prepared_execution=True,
                    primitive_tracking_guard=False,joint_states='/lease_test/joints',
                    controller_state_topic='/lease_test/controller/state',lease_topic='/lease_test/controller/execution_lease',
                    leased_command_topic='/handoff_actual/desired')
                rospy.set_param('/task_prepared_tracker',tracker_config)
                start([args.planner_prefix.resolve()/'lib/egocentric_arm_planner/trajectory_execution_manager_node',
                    '__name:=task_prepared_tracker'],'prepared_tracker')
                shadow_config=copy.deepcopy(config);shadow_config['controller_state_topic']='/lease_test/controller/state'
                rospy.set_param('/task_handoff_shadow',shadow_config)
                start(['/usr/bin/python3',args.planner_prefix.resolve()/'lib/egocentric_arm_planner/execution_preparation_handoff_node.py',
                    '__name:=task_handoff_shadow'],'handoff_shadow')
                rospy.set_param('/task_handoff_limiter',dict(joint_names=config['joint_names'],require_execution_lease=True,
                    leased_input_topic='/handoff_actual/desired',leased_output_topic='/lease_test/controller/command_leased',
                    controller_state_topic='/lease_test/controller/state',vbc_safety_hold_topic='/handoff_actual/vbc',
                    external_safety_hold_topic='/handoff_actual/gcdf',mpc=config['mpc']))
                start(['/usr/bin/python3',args.planner_prefix.resolve()/'lib/egocentric_arm_planner/joint_velocity_rate_limiter_node.py',
                    '__name:=task_handoff_limiter'],'handoff_limiter')
            gated_topic=raw_topic+'_execution_gated'
            rospy.set_param('/task_candidate_gate',dict(enable_observation_context=True,
                require_bound_regime=True,candidate_admission_enabled=True,
                input_topic=raw_topic,output_topic=gated_topic))
            continuity_params=copy.deepcopy(config)
            continuity_params.update(enable_observation_context=True,require_bound_regime=True,
                candidate_admission_enabled=True,input_topic=gated_topic,
                controller_state_topic='/lease_test/controller/state',
                output_topic='/care_planner/optimized_trajectory',
                committed_output_topic='/care_planner/committed_trajectory',
                global_summary_topic='/care_planner/candidate_vbc/summary',
                final_gcdf_enabled=True,final_gcdf_safety_margin=0.,final_gcdf_timeout_s=.5,
                gcdf_rejection_feedback_topic=raw_topic+'/gcdf_rejection_feedback',
                repair_prefix_verification_enabled=True,repair_execution_prefix_s=.15,
                probe_execution_prefix_s=.15,repair_brake_dt_s=.05,repair_hold_s=.10,
                execution_audit_enabled=True,continuation_enabled=False,
                normal_commit_start_mismatch_max=.03,repair_commit_start_mismatch_max=.03,
                probe_commit_start_mismatch_max=.03,probe_start_rebase_enabled=True,
                probe_start_joint_state_max_age_s=.05)
            rospy.set_param('/task_candidate_continuity',continuity_params)
            rospy.set_param('/task_final_anchors',dict(
                robot_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                base_frame='base_link',input_trajectory_topic='/care_planner/final_gcdf/query_trajectory',
                output_topic='/care_planner/final_gcdf/body_sweep_anchors',body_inflation_m=.015,
                max_eval_timesteps=50,ignored_risk_links=['base_link','link1']))
            vbc_params=yaml.safe_load((ROOT/'src/care_confidence_map/config/trajectory_risk.yaml').read_text())
            vbc_params.setdefault('trajectory_vbc',{}).update(dict(geometry_backend='primitive',
                robot_urdf_file=str(ROOT/'src/arm_description/urdf/Arm.urdf'),
                primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
                base_frame='base_link',input_trajectory_topic='/care_planner/optimized_trajectory',
                predicted_trajectory_topic='/care_planner/optimized_trajectory',prefer_predicted_trajectory=True,
                predicted_trajectory_timeout=.2,event_driven_eval=True,tcp_nodelay=True,
                confidence_query_persistent=True,confidence_query_service='/task_map/query',
                min_visibility_before_sweep_margin_s=.3,swept_volume_margin_m=.020,
                continuous_motion_bound_enabled=True,ignored_risk_links=['base_link','link1']))
            rospy.set_param('/task_exact_vbc',vbc_params)
            rospy.Subscriber('/care_planner/verification_outcome/task_context',String,
                lambda m:verification_events.append(json.loads(m.data)),queue_size=128)
            rospy.Subscriber('/care_planner/execution/prepared_trajectory',CandidateTransfer,
                lambda m:prepared.append(dict(action=m.action,session=m.session,sequence=m.sequence,
                    payload_sha256=m.payload_sha256,task_context=m.task_context,task_regime=m.task_regime,
                    received_ns=rospy.Time.now().to_nsec(),remaining_budget_s=m.remaining_budget_s,
                    metadata=m.planned_prefix_context,trajectory=trajectory_record(m.trajectory))),queue_size=32)
            rospy.Subscriber('/care_planner/execution/preparation_status',String,
                lambda m:preparation_status.append(json.loads(m.data)),queue_size=128)
            rospy.Subscriber('/care_planner/committed_trajectory',JointTrajectory,
                lambda m:committed.append(trajectory_record(m)),queue_size=64)
            rospy.Subscriber('/care_planner/candidate_vbc/summary',String,
                lambda m:vbc_summaries.append(m.data),queue_size=128)
            rospy.Subscriber('/care_planner/execution/probe_single_flight_summary',String,
                lambda m:gate_summaries.append(m.data),queue_size=128)
            rospy.Subscriber('/care_planner/optimized_trajectory_summary',String,
                lambda m:continuity_summaries.append(m.data),queue_size=128)
            start(['/usr/bin/python3',(prefix/'probe_single_flight_gate_node.py' if args.require_bound_policy else ROOT/'src/egocentric_arm_planner/scripts/probe_single_flight_gate_node.py'),
                '__name:=task_candidate_gate'],'candidate_gate')
            start(['/usr/bin/python3',(prefix/'optimized_trajectory_continuity_node.py' if args.require_bound_policy else ROOT/'src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py'),
                '__name:=task_candidate_continuity'],'candidate_continuity')
            start([ROOT/'devel/lib/care_confidence_map/gcdf_primitive_anchor_node','__name:=task_final_anchors'],'final_anchors')
            start([ROOT/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node',
                '__name:=task_exact_vbc','/care_planner/trajectory_risk/vbc_summary:=/care_planner/candidate_vbc/summary'],'exact_vbc')
            (out/'verification_config.json').write_text(json.dumps(dict(continuity=continuity_params,vbc=vbc_params),indent=2)+'\n')
        rospy.set_param('/task_regime',dict(enable_observation_context=True,repair_completion_gate_enabled=True))
        rospy.Subscriber('/care_planner/active_sensing/task_recovery_snapshot',String,
                         lambda m:snapshots.append(dict(received_ns=rospy.Time.now().to_nsec(),event=json.loads(m.data))),queue_size=128)
        rospy.Subscriber('/care_planner/c4_4/task_control',String,
                         lambda m:regime_controls.append(dict(received_ns=rospy.Time.now().to_nsec(),event=json.loads(m.data))),queue_size=128)
        rospy.Subscriber('/care_planner/c4_4/regime_summary',String,
                         lambda m:regime_summaries.append(m.data),queue_size=128)
        ready_pub=rospy.Publisher('/care_planner/execution/ready',Bool,queue_size=1,latch=True)
        # Readiness is a controlled planning input here; the actual plugin
        # remains WAITING. This does not authorize a motor command.
        ready_pub.publish(Bool(data=True))
        regime_process=start(['/usr/bin/python3',(prefix/'c4_4_verified_regime_manager_node.py' if args.require_bound_policy else ROOT/'src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py'),'__name:=task_regime'],'regime')
        start(['/usr/bin/python3',(prefix/'observation_task_owner_node.py' if args.require_bound_policy else ROOT/'src/egocentric_arm_planner/scripts/observation_task_owner_node.py')],'owner')
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
        def active(label):
            matches=[e['event'] for e in events if e['event']['active'] and e['event']['goal_request_id']==label]
            return matches[-1] if matches else None
        def mode(binding):
            matches=[e['event'] for e in regime_controls if e['event']['task_context']==binding]
            return matches[-1]['payload']['mode'] if matches else None
        def check(name,value):
            assert value,name
            result['checks'].append(name)
        if args.actual_cdf:
            for label,goal in [('actual_A',[0.,-.28,0.,0.,0.,0.,0.]),
                               ('actual_B',[.02,-.24,0.,0.,0.,0.,0.])]:
                submit(label,goal)
                wait(lambda:active(label) is not None,'actual CDF task context')
                wait(lambda:bool(queries),'actual native model query')
                start_ns=rospy.Time.now().to_nsec()
                wait(lambda:rospy.Time.now().to_nsec()-start_ns>=(6000000000 if args.require_bound_policy else 3000000000),'actual CDF bounded observation',20.)
            check('actual_cuda_cdf_returns_rows_or_explicit_native_disposition',bool(actual_batches or actual_witnesses))
            if args.with_verification:
                check('actual_candidate_admission_reaches_final_geometry_verification',
                      bool(verification_events) or any('final_gcdf_query_count=0' not in text and 'final_gcdf_query_count=' in text for text in continuity_summaries))
            check('no_synthetic_witness_reply_in_actual_model_run',not witness_replies)
            if args.with_verification:
                check('geometry_verification_cannot_bypass_execution_preparation',not committed)
            if args.with_handoff:
                check('actual_model_preparation_reaches_actual_controller_zero_application',any(s['zero_applied'] for s in handoff_status))
                check('unqualified_preparation_never_activates_motion',bool(states) and all(not any(s['applied']) for s in states) and
                    handoff_desired and all(not any(s['velocity']) for s in handoff_desired+handoff_limited) and
                    all(not s['execution_authorized'] and s['phase_origin_ns']==0 for s in handoff_status))
                result.update(actual_zero_handshake=True,synthetic_zero_lease_count=len(handoff_leases),
                    zero_applied_tokens=sorted({s['execution_stamp_ns'] for s in handoff_status if s['zero_applied']}))
            if args.require_bound_policy:
                policies=[r['event']['payload']['selection'] for r in snapshots
                    if r['event']['payload'].get('selection',{}).get('policy_content')]
                lineage=[r for r in native_policy_events if 'event=recovery_policy_candidate ' in r['payload'].get('text','') and
                         'recovery_policy_sha256=none' not in r['payload']['text']]
                check('measured_policy_snapshots_reach_strict_pipeline',bool(policies))
                check('actual_strict_native_candidates_bind_recovery_policy_and_target',bool(lineage))
                import re
                raw_ids={int(dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)',e['payload']['text']))['raw_candidate_stamp_ns']) for e in lineage}
                policy_candidates=[c for c in candidates if c['trajectory']['stamp_ns'] in raw_ids]
                policy_preparations=[p for p in prepared if any(
                    json.loads(p['metadata'])['source_payload_sha256']==c['payload_sha256'] and
                    p['task_context']==c['binding'] and p['task_regime']==c['regime'] for c in policy_candidates)]
                check('same_policy_candidate_passes_actual_final_geometry',bool(policy_preparations))
                check('same_policy_preparation_gets_actual_zero_application',any(
                    s['zero_applied'] and s['preparation_sha256']==p['payload_sha256']
                    for s in handoff_status for p in policy_preparations))
                result['bound_policy_snapshots']=len(policies)
                result['native_policy_candidates']=len(lineage)
                result['bound_policy_preparations']=len(policy_preparations)
            else:
                check('actual_model_run_keeps_actuator_waiting',bool(states) and all(s['mode']==0 and not any(s['applied']) for s in states))
            result.update(status='PASS_ACTUAL_MODEL_DIAGNOSTIC',real_mpc_candidates=len(candidates),
                actual_c4=True,actuator_motion=False,synthetic_cdf_replies=False,per_sensor_r1=args.per_sensor,
                actual_model_batches=len(actual_batches),actual_model_witness_responses=len(actual_witnesses),
                geometric_planner_overrides=False,formal_case_windows=0,
                full_candidate_verification=args.with_verification,verification_events=len(verification_events),
                candidate_commits=len(committed),geometry_verified_preparations=len(prepared),
                execution_preparation_enabled=args.with_verification,
                source='actual CUDA CDF/primitive guard/terminal retention with prescribed joints; local planning diagnostic only')
            (out/'planner_config.json').write_text(json.dumps(config,indent=2)+'\n')
            return
        a=submit('A',[0.,-.28,0.,0.,0.,0.,0.])
        wait(lambda:active('A') is not None and bool(candidates),'A bound MPC candidate',25.)
        context_a=copy.deepcopy(active('A'));binding_a=context_a['candidate_binding']
        wait(lambda:any(e['event']['task_context']==binding_a and e['event']['payload']['target'] is not None for e in snapshots),
             'actual learned task snapshot',30.)
        wait(lambda:mode(binding_a)=='REPAIR','automatic C4 acquisition takeover',20.)
        wait(lambda:any(c['binding']==binding_a and c['regime'].endswith(':REPAIR') for c in candidates),
             'actual MPC candidate captures automatic REPAIR',25.)
        check('actual_learned_snapshot_drives_automatic_c4_and_bound_native_repair',True)
        a_snapshot=copy.deepcopy(next(e['event'] for e in snapshots if e['event']['task_context']==binding_a and e['event']['payload']['target'] is not None))
        references_before=len(references)
        for _ in range(100):request.publish(String(data=canonical(a)));time.sleep(.002)
        time.sleep(.15)
        check('hundred_same_task_submits_preserve_context_and_reference',len(references)==references_before and active('A')['candidate_binding']==binding_a)
        raw_complete=rospy.Publisher('/care_planner/active_sensing/visibility_acquisition_complete',Bool,queue_size=1,latch=True)
        raw_probe=rospy.Publisher('/care_planner/c4_4/probe_active',Bool,queue_size=1,latch=True)
        raw_complete.publish(Bool(data=True));raw_probe.publish(Bool(data=True));repair.publish(Bool(data=False))
        time.sleep(.2)
        check('raw_completion_and_mode_bools_cannot_override_task_bound_recovery',mode(binding_a)=='REPAIR')
        # Hold actual A2 CDF requests across a real replacement, then release
        # their original headers after B has produced a valid native candidate.
        control['hold']=True
        submit('A2',[.02,-.24,0.,0.,0.,0.,0.])
        wait(lambda:active('A2') is not None and bool(held),'held A2 native query',25.)
        control['hold']=False
        submit('B',[-.02,-.24,0.,0.,0.,0.,0.])
        wait(lambda:active('B') is not None,'B context')
        context_b=copy.deepcopy(active('B'));binding_b=context_b['candidate_binding']
        wait(lambda:any(c['binding']==binding_b for c in candidates),'B actual native candidate',25.)
        before_release=len(candidates)
        for query_message in held:reply(query_message)
        for request_message in held_witnesses:witness_reply(request_message)
        time.sleep(.25)
        check('late_old_native_query_results_cannot_publish_into_replacement',
              all(c['binding']==binding_b for c in candidates[before_release:]))
        # Deliberately restamp an old-task completion so rejection depends on
        # its immutable task identity rather than only its transport age.
        replay=copy.deepcopy(a_snapshot);replay['event_seq']+=1000000
        replay['stamp_ns']=rospy.Time.now().to_nsec();replay['valid_until_ns']=replay['stamp_ns']+250000000
        replay['payload'].update(complete=True,pending_count=0,target=None)
        delayed=rospy.Publisher('/care_planner/active_sensing/task_recovery_snapshot',String,queue_size=2)
        wait(lambda:delayed.get_num_connections()>=3,'old snapshot injection subscribers')
        delayed.publish(String(data=canonical(replay)));injections.append(dict(kind='restamped_old_task_completion',event=replay))
        time.sleep(.15)
        check('fresh_transport_stamp_cannot_relabel_old_task_completion',mode(binding_b)!='PROBE_NORMAL')
        # Actual owner cancellation clears demand/goal state; it is never a
        # successful observation completion.
        old_completions=sum(e['event']['payload']['complete'] for e in snapshots if e['event']['task_context']==binding_b)
        request.publish(String(data=canonical(dict(schema=1,action='cancel',request_id='B',context_id=context_b['context_id']))))
        wait(lambda:any(not e['event']['active'] and e['event']['goal_request_id']=='B' for e in events),'B cancelled')
        time.sleep(.25)
        after_cancel=len(candidates);time.sleep(.15)
        check('cancellation_stops_new_planning_without_observation_success',len(candidates)==after_cancel and
              old_completions==sum(e['event']['payload']['complete'] for e in snapshots if e['event']['task_context']==binding_b))
        # Pause only this run's C4 process, shorter than the owner heartbeat
        # lifetime but longer than the 250 ms bound control lifetime.
        submit('C',[.01,-.25,0.,0.,0.,0.,0.])
        wait(lambda:active('C') is not None,'C context')
        binding_c=active('C')['candidate_binding']
        wait(lambda:any(c['binding']==binding_c for c in candidates),'C actual candidate',25.)
        keys_before={c['regime'] for c in candidates if c['binding']==binding_c}
        pause_start=rospy.Time.now().to_nsec()
        os.killpg(regime_process.pid,signal.SIGSTOP)
        try:
            wait(lambda:rospy.Time.now().to_nsec()-pause_start>=310000000,'C4 control expiration')
            stopped_count=len(candidates)
            wait(lambda:rospy.Time.now().to_nsec()-pause_start>=430000000,'expired control observation window')
            check('expired_bound_control_prevents_new_native_candidates',len(candidates)==stopped_count)
        finally:os.killpg(regime_process.pid,signal.SIGCONT)
        wait(lambda:any(c['binding']==binding_c and c['regime'] not in keys_before for c in candidates),
             'fresh control generation after C4 resume',25.)
        check('c4_resume_uses_new_regime_revision_instead_of_old_control',True)
        context_c=active('C')
        request.publish(String(data=canonical(dict(schema=1,action='cancel',request_id='C',context_id=context_c['context_id']))))
        wait(lambda:any(not e['event']['active'] and e['event']['goal_request_id']=='C' for e in events),'C cancelled')
        check('actual_controller_remains_waiting_with_zero_applied_output',bool(states) and all(s['mode']==0 and not any(s['applied']) for s in states))
        deadlines={}
        for row in snapshots:
            event=row['event'];target=event['payload']['target']
            if target is None:continue
            key=(event['task_context'],target['obligation_id'])
            check_value=target['deadline_ns']
            assert key not in deadlines or check_value<=deadlines[key],'task deadline postponed'
            deadlines[key]=check_value
        check('actual_snapshot_obligation_deadlines_never_postponed',bool(deadlines))
        check('every_actual_native_candidate_carries_task_and_regime',all(c['binding'] and c['regime'] for c in candidates))
        result.update(status='PASS',real_mpc_candidates=len(candidates),schedule_mode=args.schedule_mode,
            actual_c4=True,actuator_motion=False,synthetic_cdf_replies=True,per_sensor_r1=args.per_sensor,synthetic_witness_reply_count=len(witness_replies),
            source='actual task owner + C4 + nominal IK/reference + SparseSCP + native ideal_fov map + learned acquisition; prescribed joints and controlled empty/constant-positive witness CDF replies')
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
        for name,value in [('events',events),('references',references),('candidates',candidates),('summaries',summaries),('queries',queries),('reports',reports),('schedules',schedules),('states',states),('tf_trace',tf_trace),('task_commands',task_commands),('snapshots',snapshots),('regime_controls',regime_controls),('regime_summaries',regime_summaries),('injections',injections),('witness_requests',witness_requests),('witness_replies',witness_replies),('actual_batches',actual_batches),('actual_witnesses',actual_witnesses),('selector_summaries',selector_summaries),('verification_events',verification_events),('committed',committed),('vbc_summaries',vbc_summaries),('gate_summaries',gate_summaries),('continuity_summaries',continuity_summaries),('prepared',prepared),('preparation_status',preparation_status),('final_batches',final_batches),('final_queries',final_queries),('verification_queries',verification_queries)]:
            (out/(name+'.json')).write_text(json.dumps(value)+'\n')
        if args.with_handoff:
            for name,value in dict(handoff_status=handoff_status,handoff_shadow=handoff_shadow,
                handoff_desired=handoff_desired,handoff_limited=handoff_limited,handoff_leases=handoff_leases).items():
                (out/(name+'.json')).write_text(json.dumps(value)+'\n')
        (out/'native_policy_events.json').write_text(json.dumps(native_policy_events)+'\n')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if args.actual_cdf:
            assert not result['cleanup']['remaining'] and result['protected_identity_equal'] and result['port_released']
            assert result['status']=='PASS_ACTUAL_MODEL_DIAGNOSTIC',result.get('error')
    print(json.dumps(result,indent=2))
    assert result['status'] in ('PASS','PASS_ACTUAL_MODEL_DIAGNOSTIC') and not result['cleanup']['remaining'] and result['protected_identity_equal'] and result['port_released']


if __name__=='__main__':main()
