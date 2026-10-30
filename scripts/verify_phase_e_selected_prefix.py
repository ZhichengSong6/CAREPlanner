#!/usr/bin/env python3
"""Frozen/current PIQP and full continuous selector; real bound TCPROS pipeline."""
import argparse,copy,hashlib,json,os,re,shlex,signal,socket,subprocess,sys,time,xmlrpc.client
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'outputs/phase_e_selected_prefix_fix_20261012'
NAMES=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--compile-only',action='store_true');ap.add_argument('--reuse-compiled',action='store_true');args=ap.parse_args()
    out=args.output_dir.resolve()
    if not args.reuse_compiled:out.mkdir(parents=True,exist_ok=False)
    cpp=Path('src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp');hpp=Path('src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp')
    obj=ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir/src'
    flags=(obj.parent/'flags.make').read_text();inc=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1]);defs=shlex.split(re.search(r'^CXX_DEFINES = (.*)$',flags,re.M)[1])
    link=shlex.split((obj.parent/'link.txt').read_text());libs=link[link.index('-o')+2:]
    if not args.reuse_compiled:
        with (out/'compile.log').open('x') as log:
            for v in ['before','after']:
                folder=out/v;folder.mkdir();source=EVIDENCE/'before' if v=='before' else ROOT
                header=(source/hpp).read_text();(folder/'planner_diagnostic_access.hpp').write_text(header.replace('private:','public:'))
                if v=='before':
                    shim=folder/'include/egocentric_arm_planner';shim.mkdir(parents=True);(shim/hpp.name).write_text(header)
                    object_file=folder/'planner.o'
                    subprocess.run(['c++','-std=c++14','-O3',*defs,'-I'+str(folder/'include'),*inc,'-c',str(source/cpp),'-o',str(object_file)],stdout=log,stderr=subprocess.STDOUT,check=True)
                else:object_file=obj/'local_sparse_scp_planner.cpp.o'
                subprocess.run(['c++','-std=c++14','-O2','-DCARE_NEW='+str(int(v=='after')),*inc,'-I'+str(folder),str(ROOT/'scripts/test_phase_e_selected_prefix.cpp'),str(object_file),str(obj/'primitive_observation_guard.cpp.o'),'-o',str(folder/'test'),*libs],stdout=log,stderr=subprocess.STDOUT,check=True)
        paths=[ROOT/cpp,ROOT/hpp,ROOT/'scripts/test_phase_e_selected_prefix.cpp',obj/'local_sparse_scp_planner.cpp.o',out/'before/test',out/'after/test']
        (out/'compiled_sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    if args.compile_only:print('COMPILE_PASS');return
    for p,d in json.loads((out/'compiled_sha256.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d,'compiled input changed'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    os.environ.update(ROS_MASTER_URI=f'http://127.0.0.1:{port}',ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[];streams=[];report=dict(status='FAIL',qualification='Saved intermediate CASE001 q/u and four witnesses; controlled80CDF rows and complete confidence map. Actual frozen/current PIQP, full continuous VBC and TCPROS admission. No robot, trained-CDF replay or historical full-QP reconstruction.')
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def start(command,name):
        stream=(out/(name+'.log')).open('x');streams.append(stream)
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True);children.append(p);return p
    def wait(pred,limit=10):
        end=time.monotonic()+limit
        while time.monotonic()<end:
            if pred():return
            time.sleep(.01)
        raise AssertionError('private ROS condition timed out')
    def tokens(raw):return dict(re.findall(r'(\w+)=([^\s]+)',raw))
    try:
        start(['roscore','-p',str(port)],'master')
        def ready():
            try:return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/prefix_test')[0]==1
            except OSError:return False
        wait(ready)
        import rospy,yaml
        from std_msgs.msg import Bool,String,Header
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
        from egocentric_arm_planner.msg import CandidateTransfer,RepairPrefixSelection
        from care_collision_cdf.msg import CollisionCDFConstraintBatch
        from care_confidence_map.srv import QueryConfidence,QueryConfidenceResponse
        sys.path.insert(0,str(ROOT/'src/egocentric_arm_planner/scripts'))
        from initial_shell_recovery import context
        from optimized_trajectory_continuity_node import OptimizedTrajectoryContinuityNode
        rospy.init_node('prefix_test_driver',disable_signals=True)
        arm=[];sub_arm=rospy.Subscriber('/care_planner/task_start',Bool,lambda m:arm.append(m.data))
        cfg=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
        cfg['local_planner']['primitive_observation_guard']=dict(enabled=True,fk_urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),primitive_urdf=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),base_frame='base_link',ignored_links=['base_link','link1'],margin_m=.02)
        cfg['local_planner']['repair_executable_view']=dict(enabled=True,prefix_s=.25,cycle_enabled=True,cycle_prefixes_s=[.15,.1,.05],brake_dt_s=.05,max_brake_steps=20,acceleration_limits=[3.,3.,4.,4.,6.,6.,6.])
        rospy.set_param('/selected_prefix_test',cfg)
        fixtures=ROOT/'outputs/phase_e_case001_geometry_analysis_20261012/fixtures.json'
        for v in ['before','after']:
            p=start([str(out/v/'test'),str(fixtures),str(out/v/'results.json')],v);assert p.wait(timeout=30)==0,v
            report[v]=json.loads((out/v/'results.json').read_text())
        old,new=report['before'],report['after']
        assert old['proposal']['solved'] and not old['proposal']['executable_valid']
        assert new['proposal']['solved'] and new['proposal']['executable_valid']
        assert old['proposal']['cdf_rows']==new['proposal']['cdf_rows']==80
        assert old['proposal']['max_slack']==new['proposal']['max_slack']==0.
        assert new['proposal']['geometry_rows']<old['proposal']['geometry_rows']
        assert not new['hostile']['solved'] and new['hostile']['cdf_rows']==80
        assert new['identity_controls']
        assert not old['fixed']['solved'] and new['fixed']['solved'] and new['fixed']['executable_valid']
        mode=['known'];unknown=[x['point'] for x in json.loads(fixtures.read_text())[0]['points']]
        def confidence(req):
            low=[mode[0]=='unknown' and any(max(abs(a-b) for a,b in zip([p.x,p.y,p.z],q))<1e-5 for q in unknown) for p in req.points]
            return QueryConfidenceResponse([0. if v else 1. for v in low],[0.]*len(low),[1]*len(low))
        srv=rospy.Service('/prefix/query',QueryConfidence,confidence)
        vcfg=yaml.safe_load((ROOT/'src/care_confidence_map/config/trajectory_risk.yaml').read_text())['trajectory_vbc']
        vcfg.update(geometry_backend='primitive',robot_urdf_file=str(ROOT/'src/arm_description/urdf/Arm.urdf'),primitive_urdf_file=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),base_frame='base_link',input_trajectory_topic='/prefix/task',predicted_trajectory_topic='/prefix/candidate',prefer_predicted_trajectory=True,predicted_trajectory_timeout=.5,event_driven_eval=True,eval_rate=20.,predicted_periodic_refresh_rate=20.,swept_volume_margin_m=.02,continuous_motion_bound_enabled=True,ignored_risk_links=['base_link','link1'],confidence_query_service='/prefix/query',initial_shell_recovery_enabled=True,initial_shell_execution_margin_m=.01)
        rospy.set_param('/prefix_selector',{'trajectory_vbc':vcfg})
        summaries=[];protocol=[];subs=[rospy.Subscriber('/prefix/summary',String,lambda m:summaries.append(tokens(m.data)),queue_size=50),rospy.Subscriber('/prefix/candidate/protocol_status',String,lambda m:protocol.append(tokens(m.data)),queue_size=50)]
        start([str(ROOT/'devel/lib/care_confidence_map/trajectory_vbc_selector_temporal_cluster_node'),'__name:=prefix_selector','/care_planner/trajectory_risk/vbc_summary:=/prefix/summary'],'selector')
        pub=rospy.Publisher('/prefix/candidate',JointTrajectory,queue_size=1);ctx=rospy.Publisher('/prefix/candidate/initial_shell_context',String,queue_size=8)
        wait(lambda:pub.get_num_connections() and ctx.get_num_connections())
        constructor=OptimizedTrajectoryContinuityNode.__new__(OptimizedTrajectoryContinuityNode)
        constructor.repair_acceleration_limits=[3.,3.,4.,4.,6.,6.,6.];constructor.repair_brake_dt_s=.05;constructor.repair_brake_max_steps=20;constructor.repair_hold_s=.1;constructor.repair_execution_prefix_s=.25
        def raw_from(q,u):
            return JointTrajectory(joint_names=NAMES,points=[JointTrajectoryPoint(positions=pose,velocities=u[k] if k<len(u) else [0.]*7,time_from_start=rospy.Duration(k*.05)) for k,pose in enumerate(q)])
        def evaluate(label,raw,prefix,unknown_map=False,mutation=None):
            mode[0]='unknown' if unknown_map else 'known'
            candidate=constructor._repair_prefix_with_braking_tail(raw,prefix)
            candidate.header.frame_id='base_link';candidate.header.stamp=rospy.Time.now()
            contract=dict(schema=1,session='a'*32,revision=1,prefix_s=prefix,acceleration=[3.,3.,4.,4.,6.,6.,6.],brake_dt_s=.05,brake_steps=20)
            data=json.loads(context(candidate,candidate.header.stamp.to_nsec()+500000000,False,contract['acceleration'],.05,20,request_session='fixed',request_sequence=17,planned_prefix=contract))
            if mutation:mutation(data)
            ctx.publish(String(data=json.dumps(data,separators=(',',':'))));pub.publish(candidate);stamp=str(candidate.header.stamp.to_nsec())
            if mutation:
                wait(lambda:any(r.get('trajectory_stamp_ns')==stamp and r.get('state')=='invalid' for r in protocol),.4)
                assert not any(r.get('trajectory_stamp_ns')==stamp for r in summaries)
                row=dict(label=label,decision='no_certificate')
            else:
                wait(lambda:any(r.get('trajectory_stamp_ns')==stamp for r in summaries),.45)
                row=dict(next(r for r in summaries if r.get('trajectory_stamp_ns')==stamp));row['label']=label
                assert row['context_prefix_session']=='a'*32 and row['context_prefix_revision']=='1' and float(row['context_prefix_s'])==prefix
            report.setdefault('selector',[]).append(row);return row
        raw=raw_from(new['proposal']['q'],new['proposal']['u'])
        known=evaluate('native_selected_long_known_map',raw,.25);assert known['has_violation']=='0'
        recorded=next(f for f in json.loads(fixtures.read_text()) if f['name']=='plan77_input3')
        recorded_raw=raw_from(recorded['q'],recorded['u'])
        evaluate('recorded_long_unknown_witnesses',recorded_raw,.25,True)
        short=evaluate('recorded_short_unknown_witnesses',recorded_raw,.05,True);assert short['has_violation']=='1',short
        evaluate('wrong_prefix_acceleration',raw,.25,mutation=lambda d:d['planned_prefix']['acceleration'].__setitem__(0,1.))
        evaluate('wrong_prefix_boot',raw,.25,mutation=lambda d:d['planned_prefix'].__setitem__('session','invalid'))
        # Actual sender/receiver processes exercise the new serialized message
        # through offer/deliver, fresh GCDF, full VBC and commit.
        mode[0]='known'
        rospy.set_param('/prefix_gate',dict(input_topic='/prefix/raw',output_topic='/prefix/in',candidate_admission_enabled=True,repair_prefix_binding_enabled=True))
        rospy.set_param('/prefix_receiver',dict(input_topic='/prefix/in',candidate_admission_enabled=True,repair_prefix_binding_enabled=True,final_gcdf_enabled=True,final_gcdf_safety_margin=.01,initial_shell_recovery_enabled=True,repair_prefix_verification_enabled=True,repair_projection_consistent_dispatch_enabled=True,repair_execution_prefix_s=.25,cycle_recovery_enabled=True,output_topic='/prefix/candidate',global_summary_topic='/prefix/summary',committed_output_topic='/prefix/committed',final_gcdf_query_topic='/prefix/gcdf/query',final_gcdf_batch_topic='/prefix/gcdf/batch',smooth_handoff_rebase_enabled=True))
        for script,name in [('probe_single_flight_gate_node.py','gate'),('optimized_trajectory_continuity_node.py','receiver')]:start(['/usr/bin/python3',str(ROOT/'src/egocentric_arm_planner/scripts'/script),'__name:=prefix_'+name],name)
        selections=[];committed=[];queries=[]
        subs += [rospy.Subscriber('/care_planner/repair_prefix_selection',RepairPrefixSelection,lambda m:selections.append(m),queue_size=20),rospy.Subscriber('/prefix/committed',JointTrajectory,lambda m:committed.append(m)),rospy.Subscriber('/prefix/gcdf/query',JointTrajectory,lambda m:queries.append(m),queue_size=20)]
        planned=rospy.Publisher('/prefix/raw/planned',CandidateTransfer,queue_size=1);repair=rospy.Publisher('/care_planner/execution/predicted_vbc_recovery_triggered',Bool,queue_size=1,latch=True)
        joint=rospy.Publisher('/care_arm/joint_states',JointState,queue_size=1);gcdf=rospy.Publisher('/prefix/gcdf/batch',CollisionCDFConstraintBatch,queue_size=1)
        timer=rospy.Timer(rospy.Duration(.005),lambda _:joint.publish(JointState(header=Header(stamp=rospy.Time.now()),name=NAMES,position=list(raw.points[0].positions),velocity=[0.]*7)))
        repair.publish(Bool(data=True));wait(lambda:selections and planned.get_num_connections() and repair.get_num_connections()>=2 and joint.get_num_connections() and gcdf.get_num_connections());time.sleep(.1)
        from repair_prefix_contract import make
        selection=selections[-1];raw.header.frame_id='base_link';raw.header.stamp=rospy.Time.now()
        bound=make(selection.session,selection.revision,selection.prefix_s,selection.acceleration,selection.brake_dt_s,selection.brake_steps)
        planned.publish(CandidateTransfer(action='planned',trajectory=raw,planned_prefix_context=bound));wait(lambda:queries,.4)
        assert not committed
        gcdf.publish(CollisionCDFConstraintBatch(header=Header(stamp=queries[-1].header.stamp)))
        wait(lambda:committed,.5)
        assert len(queries)==len(committed)==1 and not arm
        report.update(status='PASS',arm_messages=len(arm),wire_commits=len(committed),wire_gcdf_queries=len(queries),wire_prefix=json.loads(bound))
        timer.shutdown()
    finally:
        remaining=[]
        for p in reversed(children):
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
            if p.poll() is None:remaining.append(p.pid)
        for stream in streams:stream.close()
        report['cleanup_remaining']=remaining;save('report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('before','after','selector')},indent=2))
if __name__=='__main__':main()
