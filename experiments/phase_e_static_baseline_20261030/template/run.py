"""Recorded windows for scoped recovery feasibility, with the same hard-policy control."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,shutil,traceback
O=Path(__file__).resolve().parent
def save(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['old','new'],required=True);p.add_argument('--case',default='007');p.add_argument('--configuration',choices=['baseline','nonblocking_box','detour_box','blocked_goal'],default='detour_box')
    p.add_argument('--tag',required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--policy',choices=['feasible','progress','task_first','executable_task','continuation','prefix_objective'],default='prefix_objective')
    p.add_argument('--discovery',choices=['immediate','terminal'],default='terminal')
    p.add_argument('--refinement',choices=['inherit','legacy','aligned'],default='aligned')
    p.add_argument('--handoff-retry',choices=['inherit','enabled','disabled'],default='inherit')
    p.add_argument('--reuse',choices=['enabled','disabled'],default='enabled')
    p.add_argument('--ranking',choices=['legacy','task_pose'],default='task_pose')
    p.add_argument('--view-selection',choices=['current','native','aligned_shadow','aligned_select'],default='current')
    p.add_argument('--activation-refresh', choices=['off','shadow','select'], default='off')
    p.add_argument('--task-corridor',choices=['legacy','reference_pair'],default='legacy')
    p.add_argument('--observation-release',choices=['legacy','matched'],default='legacy')
    p.add_argument('--repair-goal-stopping',choices=['off','on'],default='off')
    p.add_argument('--normal-goal-prefix',choices=['off','on'],default='off')
    p.add_argument('--normal-goal-retention',choices=['off','on'],default='off')
    p.add_argument('--observation-guidance',choices=['off','unseen'],default='off')
    p.add_argument('--observation-task-retreat',choices=['off','on'],default='off')
    p.add_argument('--normal-prefix-objective',choices=['averaged','selected'],default='averaged')
    p.add_argument('--normal-prefix-refinement',choices=['off','on'],default='off')
    p.add_argument('--normal-prefix-guards',choices=['off','on'],default='off')
    p.add_argument('--brake-prediction',choices=['off','shadow','enforce'],default='off')
    p.add_argument('--unknown-vbc',choices=['enforce','shadow'],default='enforce')
    p.add_argument('--brake-dynamics',choices=['ideal50','command10'],default='ideal50')
    p.add_argument('--capture-qp',action='store_true',help='Save exact failed production QPs for same-input replay')
    p.add_argument('--role',choices=['development','formal'],default='development')
    p.add_argument('--repeat',type=int,default=0);p.add_argument('--run-seconds-wall',type=int,choices=[60,120],default=60);a=p.parse_args()
    assert a.tag.replace('_','').isalnum() and len(a.tag)<40
    group=O/'fixed';spec=importlib.util.spec_from_file_location('occlusion_study',group/'run_study.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    inherited=m.environment
    def environment(item):
        env=inherited(item);env['CARE_ROLLING_TASK_OBSERVATION']='true';env.update(FIXED_START_RECOVERY_ENABLED='true',FIXED_START_REBASE_REAUDIT_ENABLED='true',
            CARE_OCCLUSION_REROUTE_ENABLED='true',SAFETY_MARGIN='0.30',CARE_RESEARCH_LEAD_MARGIN_S='0.30',
            CARE_OCCLUSION_TRACE_FILE=str(m.T/(item['tag']+'_occlusion.jsonl')),
            CARE_REPAIR_UNKNOWN_SOFT='true', CARE_REPLACEMENT_EXECUTION_V2=str(a.policy!='feasible').lower(),
            CARE_OBSERVATION_ACTION_PROGRESS=str(a.policy in ('progress','task_first','executable_task','continuation','prefix_objective')).lower(),
            CARE_TASK_FIRST_RESEARCH=str(a.policy in ('task_first','executable_task','continuation','prefix_objective')).lower(),
            CARE_EXECUTABLE_TASK_RESEARCH=str(a.policy in ('executable_task','continuation','prefix_objective')).lower(),
            CARE_CONTINUATION_RESEARCH=str(a.policy in ('continuation','prefix_objective')).lower(),
            CARE_TERMINAL_CANDIDATE_DISCOVERY=str(a.discovery=='terminal').lower(),
            CARE_DISCOVERY_TRACE_FILE=str(m.T/(item['tag']+'_discovery.jsonl')),
            CARE_PREFIX_TASK_OBJECTIVE=str(a.policy=='prefix_objective').lower(),
            CARE_RECEIVER_HANDOFF_RETRY=str(a.handoff_retry=='enabled' or (a.handoff_retry=='inherit' and a.policy!='prefix_objective')).lower(),
            CARE_CERTIFICATE_REFINEMENT_POLICY=a.refinement,
            CARE_CONTINUATION_TRACE_FILE=str(m.T/(item['tag']+'_continuation.jsonl')))
        if a.capture_qp:
            folder=O/'evidence'/item['tag'];folder.mkdir(parents=True,exist_ok=True)
            env['CARE_QP_CAPTURE_DIR']=str(folder)
        env['CARE_CDF_MODEL_BINDING']=str(group/('new_model_binding.json' if a.model=='new' else 'old_model_binding.json'))
        env['CARE_NATIVE_VIEW_SELECTION']=str(a.view_selection!='current').lower()
        env['CARE_NATIVE_VIEW_MODE']=a.view_selection if a.view_selection.startswith('aligned_') else 'legacy'
        env['CARE_ACTIVATION_REFRESH']=a.activation_refresh
        env['CARE_TASK_CORRIDOR']=a.task_corridor
        env['CARE_MATCHED_OBSERVATION_RELEASE']=str(a.observation_release=='matched').lower()
        env['CARE_REPAIR_GOAL_STOPPING']=str(a.repair_goal_stopping=='on').lower()
        env['CARE_NORMAL_GOAL_PREFIX']=str(a.normal_goal_prefix=='on').lower()
        env['CARE_VBC_UNKNOWN_POLICY']=a.unknown_vbc
        env['CARE_BRAKE_PREDICTION']=a.brake_prediction
        env['CARE_BRAKE_DYNAMICS']=a.brake_dynamics
        env['CARE_NORMAL_PREFIX_REFINEMENT']=str(a.normal_prefix_refinement=='on').lower()
        env['CARE_NORMAL_PREFIX_GUARDS']=str(a.normal_prefix_guards=='on').lower()
        env['CARE_NORMAL_REFINEMENT_TRACE_FILE']=str(m.T/(item['tag']+'_normal_refinement.jsonl'))
        env['CARE_NORMAL_PREFIX_SELECTED']=str(a.normal_prefix_objective=='selected').lower()
        env['CARE_OBSERVATION_TASK_RETREAT']=str(a.observation_task_retreat=='on').lower()
        env['CARE_NORMAL_GOAL_RETENTION']=str(a.normal_goal_retention=='on').lower()
        env['ROLLING_VISIBILITY_GUIDANCE_MODE']='drive' if a.observation_guidance=='unseen' else 'off'
        env['CARE_COMPLETION_PRUNE_ENABLED']='true'
        env['CARE_TASK_POSE_RANKING']=str(a.ranking=='task_pose').lower()
        env['CARE_OBSERVATION_REUSE']=str(a.reuse=='enabled').lower()
        return env
    m.environment=environment
    item=dict(model=a.model,configuration=a.configuration,tag=a.tag,batch='live_'+a.tag,case=a.case,role=a.role,repeat=a.repeat,policy=a.policy,refinement=a.refinement,handoff_retry=a.handoff_retry,discovery=a.discovery,reuse=a.reuse,ranking=a.ranking,view_selection=a.view_selection,activation_refresh=a.activation_refresh,task_corridor=a.task_corridor,observation_release=a.observation_release,repair_goal_stopping=a.repair_goal_stopping)
    item['unknown_vbc']=a.unknown_vbc
    item['brake_prediction']=a.brake_prediction
    item['brake_dynamics']=a.brake_dynamics
    item['normal_prefix_refinement']=a.normal_prefix_refinement
    item['normal_prefix_guards']=a.normal_prefix_guards
    item['normal_prefix_objective']=a.normal_prefix_objective
    item['observation_task_retreat']=a.observation_task_retreat
    item['normal_goal_prefix']=a.normal_goal_prefix
    item['normal_goal_retention']=a.normal_goal_retention
    item['observation_guidance']=a.observation_guidance
    item['task_wall_budget_s']=a.run_seconds_wall
    env=m.environment(item)
    m.active_model_binding=Path(env['CARE_CDF_MODEL_BINDING'])
    if a.preflight:
        import subprocess
        command=['/usr/bin/python3','-B',str(m.T/'run_case.py'),'--batch-id',item['batch'],'--case-ids',a.case,'--run-seconds-wall',str(a.run_seconds_wall),'--dry-run']
        subprocess.run(command,cwd=m.RT,env=env,check=True)
        print('PREFLIGHT PASS',flush=True);return
    assert not (O/'artifact_sha256.json').exists(),'Study sealed'
    receipt=O/'runs'/(a.tag+'.json');assert not receipt.exists()
    plan=json.loads((group/'plan.json').read_text());plan['runs'].append(item);save(group/'plan.json',plan)
    version=O/'versions'/a.tag;version.mkdir()
    fingerprint=m.fingerprint();save(group/'frozen_sha256.json',fingerprint);save(version/'runtime_sha256.json',fingerprint)
    for rel in ['src/egocentric_arm_planner/scripts/candidate_admission.py',
        'src/egocentric_arm_planner/scripts/probe_single_flight_gate_node.py',
        'src/care_visibility_cdf/scripts/observation_result_store.py',
        'src/care_visibility_cdf/scripts/occlusion_reroute_runtime.py',
        'src/care_visibility_cdf/scripts/observation_replacement_selection.py',
        'src/care_visibility_cdf/scripts/observation_task_rejoin.py',
        'src/care_visibility_cdf/scripts/observation_research_trace.py',
        'src/egocentric_arm_planner/scripts/known_ground.py',
        'src/egocentric_arm_planner/scripts/handoff_schedule.py',
        'src/care_visibility_cdf/scripts/vbc_deadline_waypoint_rolling_impl.py',
        'src/care_visibility_cdf/scripts/vbc_visibility_acquisition_impl.py',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/observation_action_progress.hpp',
        'src/care_visibility_cdf/scripts/replacement_execution_runtime.py',
        'src/care_visibility_cdf/scripts/candidate_replacement_runtime.py',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/safe_frontier_recovery.hpp',
        'src/care_visibility_cdf/scripts/occupied_occlusion.py','src/care_visibility_cdf/scripts/occlusion_reroute_runtime.py',
        'src/care_visibility_cdf/scripts/vbc_blocker_aware_acquisition_impl.py','src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py',
        'src/care_visibility_cdf/scripts/rolling_visibility_guidance.py',
        'src/egocentric_arm_planner/scripts/c4_4_verified_regime_manager_node.py',
        'src/egocentric_arm_planner/scripts/observation_release_identity.py',
        'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/local_sparse_scp_planner.hpp',
        'src/egocentric_arm_planner/scripts/optimized_trajectory_continuity_node.py',
        'src/egocentric_arm_planner/scripts/task_continuation.py',
        'src/egocentric_arm_planner/scripts/normal_goal_prefix.py',
        'src/egocentric_arm_planner/scripts/candidate_discovery.py',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/executable_task_objective.hpp',
        'src/egocentric_arm_planner/include/egocentric_arm_planner/repair_executable_view.hpp',
        'src/care_confidence_map/src/trajectory_vbc_selector_temporal_cluster_node.cpp',
        'src/egocentric_arm_planner/scripts/initial_shell_recovery.py',
        'src/egocentric_arm_planner/scripts/vbc_execution_reference_gate_node.py',
        'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch',
        'src/egocentric_arm_planner/launch/phase1_planner.launch']:
        if not (m.RT/rel).exists():continue  # Historical source has no later optional diagnostic file.
        dest=version/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(m.RT/rel,dest)
    record=dict(item,mode='fixed',status='RUNNING',started_utc=datetime.now(timezone.utc).isoformat(),
        development_not_scored=a.role=='development',actual_environment={k:v for k,v in env.items() if k.startswith(('CARE_','VBC_','FIXED_START','ROLLING_'))},
        runtime_manifest=str(version/'runtime_sha256.json'))
    save(receipt,record)
    try:record.update(status='COMPLETE',result=m.execute(item))
    except BaseException as exc:
        record.update(status='ERROR_OR_CONTACT_ABORT',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat();save(receipt,record)
    print(json.dumps(dict(tag=a.tag,task=record['result']['task_success'],strict=record['result']['strict_success'])),flush=True)
if __name__=='__main__':main()
