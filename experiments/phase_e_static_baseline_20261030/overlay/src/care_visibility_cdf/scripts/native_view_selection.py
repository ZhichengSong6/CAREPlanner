"""Bounded native prefix comparison; every chosen proposal still needs live audits."""
import copy
import json
import threading
import time
import uuid
import numpy as np


class NativeViewClient:
    def __init__(self, timeout_s=.45):
        import rospy
        from std_msgs.msg import String
        self.String=String;self.timeout_s=timeout_s
        self.lock=threading.Lock();self.pending={}
        self.pub=rospy.Publisher('/care_planner/view_preview/request',String,queue_size=1)
        self.sub=rospy.Subscriber('/care_planner/view_preview/response',String,self.receive,queue_size=2)

    def receive(self,msg):
        try:result=json.loads(msg.data)
        except (ValueError,TypeError):return
        with self.lock:
            item=self.pending.get(result.get('id'))
            if item is not None:item[1].append(result);item[0].set()

    def __call__(self,candidates,points):
        if self.pub.get_num_connections()==0:return dict(status='unavailable',reason='no_native_receiver')
        request=dict(id=uuid.uuid4().hex,candidates=candidates,points=np.asarray(points).tolist())
        event=threading.Event();result=[]
        with self.lock:
            if self.pending:return dict(status='unavailable',reason='native_busy')
            self.pending[request['id']]=(event,result)
        try:
            self.pub.publish(self.String(data=json.dumps(request)))
            return result[0] if event.wait(self.timeout_s) and result else dict(status='unavailable',reason='native_timeout')
        finally:
            with self.lock:self.pending.pop(request['id'],None)


def candidates_from_hybrid(runtime,hybrid):
    result=[]
    def add(sid,q,mask):
        if any(r['sensor_id']==int(sid) for r in result):return
        result.append(dict(sensor_id=int(sid),q=list(q),mask=list(mask)))
    add(hybrid['selected_sensor_id'],hybrid['selected_q_vis'],hybrid['selected_joint_mask'])
    for other in hybrid.get('native_alternatives',[]):
        if other.get('accepted'):add(other['selected_sensor_id'],other['selected_q_vis'],other['selected_joint_mask'])
    for row in hybrid.get('attempts',[]):
        if row.get('geometry',{}).get('accepted'):
            sid=int(row['sensor_id']);add(sid,row['q_candidate'],runtime.sensor_masks_np[sid].tolist())
    return result[:2]


def prefix_metrics(runtime,points,candidate,result,response,context):
    from validate_visibility_oracle import fk_transform
    from check_visibility_self_occlusion import q_row_to_map
    from evaluate_direct_vs_projection_ascent import DEFAULT_JOINT_NAMES
    if not result.get('solved') or not result.get('primitive_valid') or not result.get('samples'):
        return dict(valid=False,status=result.get('status'))
    q0=np.asarray(response['q_start']);sid=candidate['sensor_id'];target=np.asarray(candidate['q']);mask=np.asarray(candidate['mask'])
    rows=result['samples'];qs=[q0]+[np.asarray(r['q']) for r in rows]
    # Prefix and brake samples are actual native trajectory samples, not q0->qvis interpolation.
    visible=[];margins=[]
    for q in qs:
        from validate_visibility_oracle import sensor_margin
        pose=fk_transform(runtime.sensor_chains[sid],q_row_to_map(DEFAULT_JOINT_NAMES,q))
        nominal=[float(sensor_margin(point,pose,runtime.nom_hfov,runtime.nom_vfov,runtime.nom_z_min,runtime.nom_z_max)[0]) for point in points]
        margins.append(min(nominal))
        # An outside-FOV point cannot be visible, regardless of self occlusion.
        # Avoid expensive primitive raycasts unless at least one point is in FOV.
        if max(nominal)<0.:visible.append(0.)
        else:
            geometry=runtime._candidate_geometry(points,q,sid)
            visible.append(np.mean([r['nominal_margin']>=0. and not r['primitive_self_occluded'] for r in geometry['per_point']]))
    final=qs[-1]
    gap0=float(np.linalg.norm(mask*(target-q0)));gap1=float(np.linalg.norm(mask*(target-final)))
    goal=fk_transform(runtime.task_ee_chain,q_row_to_map(DEFAULT_JOINT_NAMES,context['q_task']))
    def task_cost(q):
        pose=fk_transform(runtime.task_ee_chain,q_row_to_map(DEFAULT_JOINT_NAMES,q))
        angle=np.arccos(np.clip((np.trace(goal[:3,:3].T@pose[:3,:3])-1.)*.5,-1.,1.))
        return float(np.linalg.norm(pose[:3,3]-goal[:3,3])/.20+angle)
    vision_gain=max(visible)-visible[0]
    margin_gain=max(margins)-margins[0]
    gap_gain=gap0-gap1;task_gain=task_cost(q0)-task_cost(final)
    # Soft gains; temporary task retreat and initially invisible prefixes remain eligible.
    score=4.*vision_gain+np.clip(margin_gain/.02,-2.,2.)+.5*np.clip(gap_gain/.1,-2.,2.)+.25*task_gain
    return dict(valid=True,score=float(score),predicted_visible_gain=float(vision_gain),
        predicted_visible_max=float(max(visible)),nominal_margin_gain_m=float(margin_gain),
        view_gap_gain_rad=gap_gain,task_cost_gain=task_gain,final_q=final.tolist(),samples=len(rows),
        qualification='FOV/self-LOS prediction and local native constraints; no external-LOS or full-map execution proof')


def select_native(runtime,points,hybrid,context,client):
    if not hybrid.get('accepted') or context is None:return hybrid
    selected=hybrid;tic=time.monotonic();diagnostic=dict(selected_by='original_fallback',motion_authorized=False)
    try:
        candidates=candidates_from_hybrid(runtime,hybrid)
        if len(candidates)<2:
            diagnostic.update(status='no_second_accepted_candidate',candidate_count=len(candidates));return selected
        response=client(candidates,points);diagnostic.update(response=response,candidates=candidates)
        if response.get('status')!='valid':return selected
        metrics=[prefix_metrics(runtime,points,c,r,response,context) for c,r in zip(candidates,response['candidate_results'])]
        if len(metrics)!=len(candidates):return selected
        evaluator=context.get('prefix_corridor_evaluator')
        corridors=evaluator(response) if evaluator is not None else None
        if corridors is not None and len(corridors)==len(metrics):
            diagnostic['common_prefix_map_preview']=corridors
            for metric,corridor in zip(metrics,corridors):
                if metric['valid']:
                    metric['score']-=.5*corridor['approach']['risk_fraction']
                    metric['new_unknown_risk_fraction']=corridor['approach']['risk_fraction']
        diagnostic['metrics']=metrics
        viable=[i for i,m in enumerate(metrics) if m['valid']]
        if not viable:return selected
        index=max(viable,key=lambda i:metrics[i]['score'])
        if index!=0 and (not metrics[0]['valid'] or metrics[index]['score']>metrics[0]['score']+1e-6):
            from evaluate_direct_vs_projection_ascent import DEFAULT_SENSOR_FRAMES
            row=candidates[index];selected=copy.copy(hybrid)
            selected.update(selected_sensor_id=row['sensor_id'],selected_q_vis=row['q'],selected_joint_mask=row['mask'],
                selected_sensor_frame=DEFAULT_SENSOR_FRAMES[row['sensor_id']],
                selected_rank=hybrid.get('ranking',[]).index(row['sensor_id'])+1 if row['sensor_id'] in hybrid.get('ranking',[]) else -1)
            diagnostic['selected_by']='native_prefix_progress'
        diagnostic['selected_sensor_id']=selected['selected_sensor_id']
    except Exception as exc:diagnostic['error']=str(exc)
    finally:
        diagnostic['elapsed_s']=time.monotonic()-tic
        selected['native_prefix_selection']=diagnostic
    return selected
