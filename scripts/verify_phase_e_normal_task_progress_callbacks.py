"""Actual compiled planner callback chain; controlled evidence, no actuator."""
from pathlib import Path
import argparse,hashlib,json,os,re,shlex,signal,socket,subprocess,time,xmlrpc.client
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--compile-only',action='store_true')
ap.add_argument('--reuse-compiled',action='store_true')
args=ap.parse_args();out=args.output_dir.resolve()
obj=ROOT/'build/egocentric_arm_planner/CMakeFiles/local_sparse_scp_planner_node.dir'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not args.reuse_compiled:
    out.mkdir(parents=True,exist_ok=False)
    header=ROOT/'src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp'
    (out/'planner_diagnostic_access.hpp').write_text(header.read_text().replace('private:','public:'))
    flags=(obj/'flags.make').read_text();inc=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1])
    link=shlex.split((obj/'link.txt').read_text());libraries=link[link.index('-o')+2:]
    sources=[ROOT/'scripts/test_phase_e_normal_task_progress_callbacks.cpp',obj/'src/local_sparse_scp_planner.cpp.o',obj/'src/primitive_observation_guard.cpp.o']
    with (out/'compile.log').open('x') as log:
        subprocess.run(['c++','-std=c++14','-O2',*inc,'-I'+str(out),*map(str,sources),'-o',str(out/'test'),*libraries],stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'compiled_sha256.json').write_text(json.dumps({str(p):sha(p) for p in [header,*sources,out/'test']},indent=2)+'\n')
if args.compile_only:
    print('COMPILE_PASS');raise SystemExit
assert all(sha(Path(p))==v for p,v in json.loads((out/'compiled_sha256.json').read_text()).items())
port=11962
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
env=dict(os.environ,ROS_MASTER_URI=f'http://127.0.0.1:{port}',ROS_IP='127.0.0.1',ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
processes=[];streams=[];report=dict(status='FAIL')
def start(command,name):
    f=(out/name).open('x');streams.append(f);p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);processes.append(p);return p
try:
    master=start(['roscore','-p',str(port)],'master.log');deadline=time.monotonic()+15
    while True:
        try:
            if xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/task_progress_driver')[0]==1:break
        except OSError:pass
        assert master.poll() is None and time.monotonic()<deadline
        time.sleep(.05)
    os.environ.update(env)
    import rospy,yaml
    rospy.init_node('task_progress_driver',disable_signals=True)
    cfg=yaml.safe_load((ROOT/'src/egocentric_arm_planner/config/planner_c5_5_vbc_gcdf_regime.yaml').read_text())
    cfg['local_planner']['primitive_observation_guard']=dict(enabled=True,
        fk_urdf=str(ROOT/'src/arm_description/urdf/Arm.urdf'),primitive_urdf=str(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'),
        base_frame='base_link',ignored_links=['base_link','link1'],margin_m=.02)
    rospy.set_param('/task_progress_callback_test',cfg)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    p=start([str(out/'test'),str(out/'result.json')],'test.log');assert p.wait(timeout=45)==0
    report.update(json.loads((out/'result.json').read_text()),qualification='Controlled blocked-point evidence, full-horizon verdict and simulated measured state through actual candidate-publication/goal/reference/joint/verification/tracker callbacks with real URDF FK. No learned model, full-map proof, Gazebo or actuator.')
finally:
    remaining=[]
    for p in reversed(processes):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGINT)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
        if p.poll() is None:remaining.append(p.pid)
    for f in streams:f.close()
    report['cleanup_remaining']=remaining
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
