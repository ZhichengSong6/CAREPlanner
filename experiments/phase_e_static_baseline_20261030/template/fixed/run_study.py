"""Run a finite baseline qualification, then the frozen12-window scene plan."""
from pathlib import Path
import argparse,hashlib,json,os,signal,socket,subprocess,sys,time,traceback,zipfile
O=Path(__file__).resolve().parent;RT=O/'runtime';T=RT/'outputs/cdf_model_ab'
active_model_binding=O/'old_model_binding.json'
sys.path.insert(0,str(RT/'scripts'))
from phase_e_owned_processes import identity,remember_descendants,reap_owned

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def save(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');temp.replace(p)

def fingerprint():
    files=[]
    suffixes={'.inc','.cpp','.hpp','.h','.py','.yaml','.yml','.launch','.sh','.urdf','.xacro','.world','.xml','.json','.msg','.srv'}
    for folder in ('src','scripts'):
        files += [p for p in (RT/folder).rglob('*') if p.is_file() and p.suffix in suffixes and '__pycache__' not in p.parts]
    files += [p for p in (RT/'devel/lib').rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    names=['plan.json','scenes.json','bootstrap_gate.json']
    names += ['demo_configuration.json'] if (O/'demo_configuration.json').is_file() else ['mapping_gate.json','guard_qualification.json','detour_route.json']
    files += list(T.glob('*.py'))+list(O.glob('*.py'))+[O/n for n in names]
    files += [p for p in (O/'scenes').glob('*.world')]
    binding=json.loads(active_model_binding.read_text())
    files += [active_model_binding,Path(binding['checkpoint']),RT/'src/care_visibility_cdf/checkpoints/hierarchical9_r1_scratch50k/final.pt']
    files += [RT/'src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt']
    return {str(p.relative_to(O)):sha(p) for p in sorted(set(files))}

def freeze_check():
    expected=json.loads((O/'frozen_sha256.json').read_text());actual=fingerprint()
    changed=[p for p in expected.keys()|actual.keys() if expected.get(p)!=actual.get(p)]
    assert not changed,changed

def free(port):
    with socket.socket() as s:return s.connect_ex(('127.0.0.1',port))!=0

def environment(item):
    env=json.loads(subprocess.check_output(['bash','--noprofile','--norc','-c',
        'source '+str(RT/'devel/setup.bash')+' && /usr/bin/python3 -c \'import os,json;print(json.dumps(dict(os.environ)))\''],text=True))
    plan=json.loads((O/'plan.json').read_text());scene=json.loads((O/'scenes.json').read_text())['configurations'][item['configuration']]
    env.update(REPO=str(RT),CARE_CDF_MODEL_BINDING=str(O/'old_model_binding.json'),PYTHONDONTWRITEBYTECODE='1',
        ROS_IP='127.0.0.1',ROS_HOSTNAME='127.0.0.1',ROS_HOME=str(T/(item['tag']+'_ros_home')),ROS_LOG_DIR=str(T/(item['tag']+'_ros_logs')),
        GAZEBO_MODEL_DATABASE_URI='',WORLD_FILE=scene['world'],
        GOAL_OBSERVE_TERMINAL_STOP='true',CARE_POSE_TRANSFORM_REUSE_ENABLED='false',
        VBC_SWEPT_VOLUME_MARGIN_M='0.020',EXECUTION_VBC_SWEPT_VOLUME_MARGIN_M='0.010',PRIMITIVE_TRACKING_GUARD='true',
        ACTUATOR_COMMAND_WATCHDOG='false',CDF_SELECTOR_ENABLED='false',SUMMARY_STATE_CLOCK_COHERENT='true',
        FIXED_START_RECOVERY_ENABLED='false',FIXED_START_REBASE_REAUDIT_ENABLED='false',
        CARE_COMPLETION_RANKING_ENABLED='false',CARE_COMPLETION_PRUNE_ENABLED='false',CARE_COMPLETION_PROFILE_FILE=str(O/'calibration_final/profile.json'),
        CARE_PENDING_RELEVANCE_ENABLED='true',CARE_OBSERVATION_PROGRESS_ENABLED='true',CARE_KNOWN_GROUND_ENABLED='true',
        CARE_GROUND_BACKEND='native',CARE_KNOWN_GROUND_FILE=str(RT/'src/egocentric_arm_planner/config/known_ground_geometry.json'),
        CARE_OBSERVATION_TRACE_FILE=str(T/(item['tag']+'_observation.jsonl')),CARE_BODY_TRACE_FILE=str(T/(item['tag']+'_body.jsonl')),
        CARE_TASK_WINDOW_FILE=str(T/(item['tag']+'_window.json')),CARE_STATIC_SCENE_JSON=str(O/'scenes.json'),
        CARE_STATIC_SCENE_ID=item['configuration'],CARE_STATIC_SCENE_TRACE=str(T/(item['tag']+'_scene.jsonl')))
    env.update(ROS_MASTER_URI='http://127.0.0.1:12900',GAZEBO_MASTER_URI='http://127.0.0.1:12901',
        CARE_SCENE_GUARD_PREFIX=str(T/(item['tag']+'_guard')),CARE_BOOTSTRAP_GATE=str(O/'bootstrap_gate.json'))
    env['CARE_METRIC_CURVE_ENABLED']='false'
    env['CARE_OCCUPIED_VOLUME_QP_ENABLED']='false'
    env['CARE_VBC_WHOLE_VOXEL']='false'
    env['CARE_RESEARCH_PROFILE']='nominal_plane_point_v1'
    env['CARE_RESEARCH_OPPORTUNISTIC_PROBE']='false'
    env['CARE_RESEARCH_SOFT_UNKNOWN']='false'
    env['CARE_RESEARCH_VERIFY_TIMEOUT_S']='0.75'
    env['PREDICTION_TIMEOUT']='1.0'
    env['FINAL_GCDF_SAFETY_MARGIN']='0.010'
    env['CARE_ANALYTIC_GROUND_FILE']=str(RT/'src/egocentric_arm_planner/config/known_ground_geometry.json')
    env['CARE_ANALYTIC_GROUND_APPLY']='true'
    env['CARE_METRIC_CURVE_TRACE_FILE']=str(T/(item['tag']+'_metric_curve.jsonl'))
    env['CARE_CAPTURE_ALL_FINAL_BATCHES']='true'
    env['CARE_REJECTION_EVIDENCE_FILE']=str(T/(item['tag']+'_rejections.jsonl'))
    env.pop('CARE_QP_CAPTURE_DIR',None)
    return env

def execute(item):
    assert item['role'] in ('development','qualification','formal')
    assert not (O.parent/'artifact_sha256.json').exists(),'Study sealed'
    freeze_check();assert all(free(p) for p in (12900,12901))
    env=environment(item);owned={};p=None;result=dict(item)
    print('START',item['role'],item['tag'],flush=True)
    try:
        with (T/(item['tag']+'_runner.log')).open('x') as log:
            p=subprocess.Popen(['/usr/bin/python3','-B',str(T/'run_case.py'),'--batch-id',item['batch'],'--case-ids',item['case']],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owned[p.pid]=identity(p.pid);deadline=time.monotonic()+480
            guard_path=Path(env['CARE_SCENE_GUARD_PREFIX']+'_state.json')
            trip_path=Path(env['CARE_SCENE_GUARD_PREFIX']+'_trip.json')
            stop_status=RT/'outputs/c5_5_vbc_gcdf_regime'/('phase_e_goal_'+item['case']+'_'+item['batch'])/'run/goal_stop_status.json'
            abort_since=None;last_descendants=0;guard_seen=False
            while p.poll() is None:
                now=time.monotonic()
                if now-last_descendants>.5:remember_descendants([p.pid],owned);last_descendants=now
                if guard_seen and not guard_path.exists() and not stop_status.exists():
                    result['guard_failure']='Guard state disappeared'
                    try:subprocess.run(['rosservice','call','/gazebo/pause_physics'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=1)
                    except Exception:pass
                    raise RuntimeError(result['guard_failure'])
                if guard_path.exists():
                    guard_seen=True
                    guard=json.loads(guard_path.read_text())
                    if not stop_status.exists() and (now-guard['heartbeat_monotonic_s']>1 or guard.get('finished')):
                        result['guard_failure']='Guard died or heartbeat became stale'
                        try:subprocess.run(['rosservice','call','/gazebo/pause_physics'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=1)
                        except Exception:pass
                        raise RuntimeError(result['guard_failure'])
                    if trip_path.exists():
                        if abort_since is None:abort_since=now;result['guard_abort']=json.loads(trip_path.read_text())
                        if now-abort_since>1 and not guard.get('pause_confirmed'):
                            raise RuntimeError('Contact guard pause unconfirmed: reap owned simulation')
                        if now-abort_since>45:raise RuntimeError('Contact abort archival deadline: reap owned simulation')
                if time.monotonic()>deadline:raise TimeoutError('Single window infrastructure ceiling')
                time.sleep(.05)
            result['runner_rc']=p.returncode
    finally:
        cleanup=reap_owned(owned);result['cleanup']=cleanup
        prefix=env['CARE_SCENE_GUARD_PREFIX']
        if Path(prefix+'_state.json').exists():result['contact_guard']=json.loads(Path(prefix+'_state.json').read_text())
        if Path(prefix+'_trip.json').exists():result['guard_abort']=json.loads(Path(prefix+'_trip.json').read_text())
        save(T/(item['tag']+'_attempt.json'),result)
    if result.get('guard_abort'):
        result.update(status='ABORTED_BY_SCENE_GUARD',valid_window=False,task_success=False,strict_success=False)
        save(T/(item['tag']+'_aborted.json'),result)
        raise RuntimeError('Scene guard stopped this attempt and all later windows: '+json.dumps(result['guard_abort']))
    assert result['runner_rc']==0 and not cleanup['remaining'],result
    summary=RT/'outputs/phase_e_r1_mainline_30'/item['batch']/('case_summaries/phase_e_goal_'+item['case']+'.json')
    s=json.loads(summary.read_text());result.update(task_success=s['task_success'],overall_safe=s['overall_safe'],
        valid_window=s['benchmark_window']['valid'],benchmark_window=s['benchmark_window'],
        final_position_error_m=s['final_position_error_m'],final_orientation_error_rad=s['final_orientation_error_rad'])
    for script,prefix in [('review.py','review'),('analyze.py','analysis'),('cdf_review.py','cdf_review')]:
        with (T/(item['tag']+'_'+prefix+'.log')).open('x') as log:
            r=subprocess.run(['/usr/bin/python3','-B',str(T/script),'--batch',item['batch'],'--case',item['case'],'--tag',item['tag']],cwd=RT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            result[prefix+'_rc']=r.returncode
    review=T/('formal_review_'+item['tag']+'.json')
    if review.exists():
        r=json.loads(review.read_text());result.update(strict_success=r['strict_formal_stopped_hold_demonstrated'],execution_vbc=r['execution_vbc_through_stop'],commits=r['original_commits'],review_issues=r['issues'])
    scene_trace=T/(item['tag']+'_scene.jsonl');assert scene_trace.is_file()
    events=[json.loads(l) for l in scene_trace.read_text().splitlines()];assert events[0]['event']=='started'
    result['contact_events']=[r for r in events if r['event']=='contact']
    result['scene_monitor_finished']=bool(events and events[-1]['event']=='finished')
    result['scene_trace_sha256']=sha(scene_trace)
    archive=RT/f"CAREPlanner_C5_RESULT_phase_e_goal_{item['case']}_{item['batch']}.zip"
    with zipfile.ZipFile(archive) as z:result['archive_crc_valid']=z.testzip() is None
    result['archive_sha256']=sha(archive)
    freeze_check();assert all(free(p) for p in (12900,12901))
    save(T/(item['tag']+'_result.json'),result)
    if (result.get('review_rc')==result.get('cdf_review_rc')==0 and result.get('analysis_rc')==1 and
            not result['task_success'] and not result['overall_safe'] and result['commits']==0 and
            result['execution_vbc']['records']==0):
        recovery=subprocess.run(['/usr/bin/python3','-B',str(O.parent/'review_stationary_failure.py'),
            '--tag',item['tag'],'--batch',item['batch'],'--case',item['case']],cwd=RT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (T/(item['tag']+'_stationary_review.log')).write_text(recovery.stdout)
        result['stationary_failure_review_rc']=recovery.returncode
        assert recovery.returncode==0,result
    else:
        assert all(result.get(k)==0 for k in ('review_rc','analysis_rc','cdf_review_rc')),result
    assert result['valid_window'] and result['archive_crc_valid'] and result['scene_monitor_finished'],result
    assert not result['contact_events'],'Physical box contact; preserve attempt and stop'
    guard=result['contact_guard']
    assert guard['armed'] and guard['finished'] and not guard['tripped'],guard
    result['configured_audits_passed']=bool(result['overall_safe'] and not result['execution_vbc']['unsafe'])
    print('END',item['tag'],json.dumps({k:result.get(k) for k in ('task_success','strict_success','overall_safe','commits','execution_vbc')}),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qualification',action='store_true');ap.add_argument('--formal',action='store_true');a=ap.parse_args();assert a.qualification!=a.formal
    plan=json.loads((O/'plan.json').read_text())
    if a.qualification:
        assert not (O/'qualification_attempt.json').exists()
        save(O/'qualification_attempt.json',dict(start_monotonic_ns=time.monotonic_ns(),item=plan['qualification']))
        try:
            r=execute(plan['qualification']);passed=all(r.get(k) for k in ('task_success','strict_success','overall_safe','valid_window')) and not r['execution_vbc']['unsafe']
            save(O/'qualification_result.json',dict(passed=passed,result=r,formal_windows=0))
            print('QUALIFICATION',passed,flush=True)
        except BaseException as exc:
            save(O/'qualification_error.json',dict(error=repr(exc)));raise
    else:
        assert json.loads((O/'qualification_result.json').read_text())['passed']
        assert not (O/'formal_state.json').exists()
        rows=[]
        try:
            for item in [r for r in plan['runs'] if r['role']=='formal']:
                save(O/'formal_state.json',dict(status='RUNNING',completed=len(rows),current=item))
                rows.append(execute(item));save(O/'formal_results_partial.json',rows)
            save(O/'formal_results.json',rows);save(O/'formal_state.json',dict(status='COMPLETE',completed=len(rows)))
        except BaseException as exc:
            save(O/'formal_state.json',dict(status='STOPPED',completed=len(rows),stopped_attempt=item,remaining_not_launched=sum(r['role']=='formal' for r in plan['runs'])-len(rows)-1,error=repr(exc)));raise

if __name__=='__main__':main()
