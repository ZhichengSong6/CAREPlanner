"""Online measured-map occlusion, deferred sensing, and finite route proposals.

The broker owns only /task_trajectory (a reference). It never publishes an
execution command, a confidence update, or an acquisition-success message.
"""
import copy,json,math,os,threading,time
from pathlib import Path
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from occupied_occlusion import first_blocker,RouteProposer,sample_polyline,quaternion_matrix
from observation_identity import observation_token,observation_region_token,restore_observation_target_state
from observation_result_store import ObservationResultStore
from evaluate_direct_vs_projection_ascent import DEFAULT_JOINT_NAMES
from validate_visibility_oracle import fk_transform,sensor_margin


def point_key(points):
    return tuple(sorted(tuple(float(x) for x in row) for row in np.round(np.asarray(points).reshape(-1,3),5)))


class OcclusionReroute:
    def __init__(self,node):
        self.node=node;self.lock=threading.RLock();self.occupied=np.empty((0,3));self.map_stamp=0.
        self.deferred={};self.goal=None;self.nominal=None;self.route=None;self.route_index=1
        self.generation=0;self.route_generation=0;self.goal_revision=0;self.pending=False;self.busy=False
        self.last_attempt=-math.inf;self.last_publish=-math.inf;self.last_route_event=-math.inf
        self.regime='NORMAL';self.proposer=None;self.reason='';self.raw_candidates={}
        self.screened_maps=set()
        self.route_cutoff_ns=0
        self.task_first_research=os.environ.get('CARE_TASK_FIRST_RESEARCH')=='true'
        self.task_candidate_cutoff_ns=0
        self.reuse_enabled=os.environ.get('CARE_OBSERVATION_REUSE')=='true'
        self.results=ObservationResultStore()
        self.relevance_first_seen={};self.relevance_excluded=set();self.last_relevance_check=-math.inf
        self.sensing_retry_keys=set();self.sensing_first_seen={}
        self.terminal_reference_retained=False
        self.event_pub=rospy.Publisher('/care_planner/active_sensing/occlusion_reroute',String,queue_size=8)
        self.reference_pub=rospy.Publisher('/care_planner/task_trajectory',JointTrajectory,queue_size=1,latch=True)
        self.trace_path=Path(os.environ['CARE_OCCLUSION_TRACE_FILE']);self.trace_path.parent.mkdir(parents=True,exist_ok=True)
        self.trace_lock=threading.Lock()
        self.subs=[
            rospy.Subscriber('/care_planner/confidence_map/points',PointCloud2,self.map_cb,queue_size=1,buff_size=8*1024*1024),
            rospy.Subscriber('/care_planner/nominal_task_trajectory',JointTrajectory,self.nominal_cb,queue_size=1),
            rospy.Subscriber('/care_planner/ee_target_pose',PoseStamped,self.goal_cb,queue_size=1),
            rospy.Subscriber('/care_planner/verification_outcome',String,self.verdict_cb,queue_size=32),
            rospy.Subscriber('/care_planner/local_planner/candidate_trajectory',JointTrajectory,self.raw_cb,queue_size=8),
            rospy.Subscriber('/care_planner/c4_4/regime_summary',String,self.regime_cb,queue_size=1)]
        self.timer=rospy.Timer(rospy.Duration(.25),self.tick)
        self.trace('enabled',changes_map=False,changes_execution_gate=False)

    def trace(self,event,**data):
        row=dict(event=event,ros_s=rospy.Time.now().to_sec(),wall_s=time.time(),goal_revision=self.goal_revision,**data)
        with self.trace_lock:
            with self.trace_path.open('a') as f:f.write(json.dumps(row,allow_nan=True,separators=(',',':'))+'\n')

    def map_cb(self,msg):
        if msg.header.frame_id!='base_link' or msg.height!=1:return
        fields={f.name:f for f in msg.fields};names=['x','y','z','occupancy','hit_count','hit_sensor_mask']
        if any(n not in fields for n in names):return
        formats={7:'f4',6:'u4'};byte='>' if msg.is_bigendian else '<'
        try:
            dtype=np.dtype(dict(names=names,formats=[byte+formats[fields[n].datatype] for n in names],
                offsets=[fields[n].offset for n in names],itemsize=msg.point_step))
            a=np.frombuffer(msg.data,dtype=dtype,count=msg.width)
            mask=(a['occupancy']>.5)&(a['hit_count']>0)&(a['hit_sensor_mask']>0)
            occupied=np.column_stack([a[n][mask] for n in names[:3]]).astype(float)
            if not np.isfinite(occupied).all():return
        except (ValueError,KeyError,TypeError):return
        with self.lock:self.occupied=occupied;self.map_stamp=msg.header.stamp.to_sec()

    def snapshot(self):
        with self.lock:
            age=rospy.Time.now().to_sec()-self.map_stamp
            return (self.occupied.copy(),self.map_stamp) if self.map_stamp>0 and 0<=age<=.5 else (None,self.map_stamp)

    def goal_cb(self,msg):
        p=msg.pose.position;q=msg.pose.orientation
        key=(msg.header.frame_id,p.x,p.y,p.z,q.x,q.y,q.z,q.w)
        with self.lock:
            if self.goal==key:return
            self.goal=key;self.goal_revision+=1;self.route=None;self.deferred.clear()
            self.results.clear(self.goal_revision)
            self.nominal=None
            self.pending=False;self.raw_candidates.clear()
            self.screened_maps.clear()
            self.route_cutoff_ns=0
            self.task_candidate_cutoff_ns=0
            self.relevance_first_seen.clear();self.relevance_excluded.clear()
            self.sensing_retry_keys.clear();self.sensing_first_seen.clear()
            self.terminal_reference_retained=False
        self.trace('goal_changed')

    def nominal_cb(self,msg):
        if msg.header.frame_id!='base_link' or list(msg.joint_names)!=list(DEFAULT_JOINT_NAMES) or not msg.points:return
        if len(msg.points[-1].positions)!=7:return
        with self.lock:
            self.nominal=copy.deepcopy(msg);active=self.route is not None
        if not active:self.reference_pub.publish(msg)

    def raw_cb(self,msg):
        with self.lock:
            self.raw_candidates[msg.header.stamp.to_nsec()]=(self.goal_revision,rospy.Time.now().to_sec())
            while len(self.raw_candidates)>64:self.raw_candidates.pop(next(iter(self.raw_candidates)))

    def regime_cb(self,msg):
        f=dict(w.split('=',1) for w in msg.data.split() if '=' in w)
        with self.lock:self.regime=f.get('state',self.regime)

    def verdict_cb(self,msg):
        f=dict(w.split('=',1) for w in msg.data.split() if '=' in w)
        if (self.task_first_research and f.get('result')=='safe' and f.get('committed')=='1' and
                f.get('verification_view') in ('full_horizon','probe_prefix_brake_hold')):
            try:raw=int(f['raw_candidate_stamp_ns']);audit=int(f['audited_trajectory_stamp_ns'])
            except (KeyError,ValueError):return
            with self.lock:
                record=self.raw_candidates.get(raw)
                if record is None or record[0]!=self.goal_revision or audit<=0:return
                self.task_candidate_cutoff_ns=max(self.task_candidate_cutoff_ns,audit)
            self.trace('task_prefix_supersedes_candidate_work',raw_candidate_stamp_ns=str(raw),
                cutoff_ns=str(audit),execution_stamp_ns=f.get('execution_stamp_ns'),
                actual_observation_completed=False,map_changed=False)
            return
        if f.get('result')!='unsafe' or f.get('safety_gate') not in ('gcdf_occupied','gcdf_mixed'):return
        try:raw=int(f['raw_candidate_stamp_ns'])
        except (KeyError,ValueError):return
        with self.lock:
            record=self.raw_candidates.pop(raw,None)
            if record is None or record[0]!=self.goal_revision or not 0<=rospy.Time.now().to_sec()-record[1]<=2:return
            self.pending=True;self.reason='final_occupied_rejection'
        self.trace('occupied_rejection_requested_route',raw_candidate_stamp_ns=str(raw),
            audited_stamp_ns=f.get('audited_trajectory_stamp_ns'),occupied_source_preserved=True)

    def candidate_check(self,points,q,sensor_id):
        occupied,stamp=self.snapshot()
        if occupied is None:return dict(available=False,blocked=False,reason='no_current_map')
        runtime=self.node._per_sensor_runtime
        tf=fk_transform(runtime.sensor_chains[int(sensor_id)],dict(zip(DEFAULT_JOINT_NAMES,q)))
        hits=[dict(point=np.asarray(p).tolist(),**hit) for p in np.asarray(points).reshape(-1,3)
              for hit in [first_blocker(tf[:3,3],p,occupied)] if hit is not None]
        return dict(available=True,blocked=bool(hits),map_stamp_s=stamp,sensor_id=int(sensor_id),
            sensor_origin=tf[:3,3].tolist(),hits=hits)

    def process_obligations(self):
        # Called in the existing schedule worker, never in a map callback.
        if self.snapshot()[0] is None and not (self.task_first_research and self.task_candidate_cutoff_ns):return
        with self.node._obligation_lock:obs=[dict(o) for o in self.node._obligations]
        now=time.monotonic();check_relevance=now-self.last_relevance_check>=1.
        if check_relevance:self.last_relevance_check=now
        for ob in obs:
            if ob.get('obligation_bundle_source')!='candidate':continue
            with self.node._obligation_lock:
                if not any(observation_token(o)==observation_token(ob) for o in self.node._obligations):continue
            # The actual published pose can differ from a stored individual
            # pose after the shared solver. Check that exact current view.
            published=getattr(self.node,'_trace_published_target',None)
            view=dict(ob)
            if published is not None and published.get('region_token')==observation_region_token(ob):
                restore_observation_target_state(view,published)
            token=observation_token(ob);key=point_key(ob['points'])
            first=self.relevance_first_seen.setdefault(token,now)
            measured=getattr(self.node,'_trace_measured',(None,math.nan))
            relevance=None
            # A dependency is needed to observe its parent, even if the task
            # route misses it. Retiring it alone just recreates the same child.
            is_dependency=int(ob['id']) in getattr(self.node,'_dependency_edges',{})
            if (not is_dependency and check_relevance and self.proposer is not None and self.route is not None and
                    now-first>=2. and key not in self.relevance_excluded and measured[0] is not None and
                    0<=rospy.Time.now().to_sec()-measured[1]<=.2):
                knots=np.vstack([measured[0],self.route[self.route_index:]])
                relevance=self.proposer.point_clearance_bound(knots,ob['points'])
            source=int(ob.get('obligation_candidate_stamp_ns',0))
            task_superseded=self.task_first_research and 0<source<=self.task_candidate_cutoff_ns
            obsolete=((self.route is not None and 0<source<=self.route_cutoff_ns or task_superseded) and
                    ob.get('obligation_bundle_source')=='candidate')
            if obsolete or (relevance is not None and relevance>=.020):
                retired_ids={int(ob['id'])}
                edges=getattr(self.node,'_dependency_edges',{})
                for _ in range(len(edges)):
                    retired_ids.update(child for child,parent in edges.items() if parent in retired_ids)
                with self.node._obligation_lock:
                    removed=[o for o in self.node._obligations if int(o['id']) in retired_ids and
                             o.get('obligation_bundle_source')=='candidate']
                    self.node._obligations=[o for o in self.node._obligations if not any(o is x for x in removed)]
                with self.lock:
                    for old in removed:
                        old_view=dict(old)
                        if published is not None and published.get('region_token')==observation_region_token(old):
                            restore_observation_target_state(old_view,published)
                        self.save_result(old_view,'task_or_route_preemption')
                        self.deferred[point_key(old['points'])]=dict(obligation_id=int(old['id']),token=observation_token(old),
                            published_token=observation_token(old_view),
                            points=np.asarray(old['points']).tolist(),reason='alternative_reference',
                            source_cutoff_ns=self.task_candidate_cutoff_ns if task_superseded else rospy.Time.now().to_nsec(),evidence=[])
                self.trace('observation_deferred_for_route',token=token,points=np.asarray(ob['points']).tolist(),
                    source_stamp_ns=str(source),generation=self.generation,actual_observation_completed=False,
                    point_clearance_bound_m=relevance,reason='task_prefix_superseded' if task_superseded else
                    ('superseded_candidate' if obsolete else 'alternative_reference_avoids_region'))
                continue
            sid=int(view.get('per_sensor_selected_sensor_id',-1))
            if sid<0:
                # Scalar fallback can only be deferred if every FOV-admissible
                # sensor at that candidate pose is blocked by measured support.
                checks=[];qmap=dict(zip(DEFAULT_JOINT_NAMES,view['q_vis']))
                rt=self.node._per_sensor_runtime
                for sensor in range(len(rt.sensor_chains)):
                    tf=fk_transform(rt.sensor_chains[sensor],qmap)
                    if all(sensor_margin(p,tf,rt.nom_hfov,rt.nom_vfov,rt.nom_z_min,rt.nom_z_max)[0]>=0 for p in ob['points']):
                        checks.append(self.candidate_check(ob['points'],view['q_vis'],sensor))
                blocked=bool(checks) and all(c['blocked'] for c in checks)
                evidence=checks
            else:
                check=self.candidate_check(ob['points'],view['q_vis'],sid)
                blocked=check['blocked'];evidence=[check]
            if not blocked:continue
            token=observation_token(ob);key=point_key(ob['points'])
            with self.node._obligation_lock:
                current=next((o for o in self.node._obligations if int(o['id'])==int(ob['id'])),None)
                if current is None or observation_token(current)!=token:continue
                self.node._obligations=[o for o in self.node._obligations if int(o['id'])!=int(ob['id'])]
            with self.lock:
                new=key not in self.deferred
                self.save_result(view,'occupied_occlusion')
                self.deferred[key]=dict(obligation_id=int(ob['id']),token=token,points=np.asarray(ob['points']).tolist(),evidence=evidence)
                self.deferred[key]['published_token']=observation_token(view)
                if new or self.route is None:self.pending=True;self.reason='external_occupied_occlusion'
            self.trace('observation_deferred',obligation_id=int(ob['id']),token=token,points=np.asarray(ob['points']).tolist(),
                evidence=evidence,actual_observation_completed=False,map_changed=False)

    def save_result(self,ob,reason,revision=None):
        if not self.reuse_enabled:return
        revision=self.goal_revision if revision is None else revision
        with self.lock:saved=self.results.save(ob,revision,reason)
        if saved:
            self.trace('observation_result_saved',obligation_id=int(ob['id']),reason=reason,
                points=np.asarray(ob['points']).tolist(),actual_observation_completed=False)

    def superseded(self,source,identity):
        if not self.reuse_enabled or source!='candidate':return False
        stamp=int((identity or {}).get('trajectory_stamp_ns',0))
        with self.lock:
            return 0<stamp<=max(self.task_candidate_cutoff_ns if self.task_first_research else 0,
                               self.route_cutoff_ns if self.route is not None else 0)

    def restore_result(self,region,source,identity,sweep,trajectory_source):
        if not self.reuse_enabled or self.superseded(source,identity):return None
        with self.lock:revision=self.goal_revision
        # Nearby geometry is only a lookup hint. The original physical FOV/LOS
        # or scalar compatibility check must accept every current point.
        with self.lock:candidates=self.results.candidates(region,revision,self.node.region_match_distance_m)
        for saved in candidates:
            ob=saved['obligation']
            # An unmatched distant region must not reuse an ID whose current
            # owner is still live. Leave that region to normal generation.
            with self.node._obligation_lock:
                if any(int(live['id'])==int(ob['id']) for live in self.node._obligations):continue
            if not self.node._match_candidate_compatible(ob,region):continue
            now=rospy.Time.now().to_sec()
            measured=self.node._latest_measured_q
            if measured is None:return None
            ob.update(points=np.asarray(region['points']).copy(),keys=tuple(region['keys']),
                centroid=np.asarray(region['centroid']).copy(),
                obligation_bundle_source=source,
                obligation_candidate_stamp_ns=int((identity or {}).get('trajectory_stamp_ns',0)),
                last_seen_ros_s=now,trajectory_source=trajectory_source,
                discovered_sweep_time_s=float(sweep),deadline_abs_s=now+float(sweep),
                min_hit_time_from_measured_rest_s=self.node._rest_min_hit_time(measured,ob['q_vis']),
                reused_result_count=int(ob.get('reused_result_count',0))+1)
            with self.lock:
                if revision!=self.goal_revision:return None
                self.deferred.pop(point_key(region['points']),None)
            self.trace('observation_result_reused',obligation_id=int(ob['id']),
                original_generation_event_id=ob.get('generation_event_id'),
                original_points=saved['obligation']['q_vis_source_points'].tolist(),
                points=np.asarray(region['points']).tolist(),q_vis=np.asarray(ob['q_vis']).tolist(),
                measured_seed=np.asarray(measured).tolist(),source=source,
                source_stamp_ns=str(ob['obligation_candidate_stamp_ns']),
                actual_observation_completed=False)
            return ob
        return None

    def suppress_region(self,region,source,identity):
        # Execution blockers always stay in the normal recovery path. Candidate
        # work may be deferred, but the UNKNOWN map and VBC remain untouched.
        if source!='candidate':return False
        key=point_key(region['points'])
        source_stamp=int((identity or {}).get('trajectory_stamp_ns',0))
        if self.task_first_research and 0<source_stamp<=self.task_candidate_cutoff_ns:
            return True
        if self.route is not None and 0<source_stamp<=self.route_cutoff_ns:
            return True  # Superseded discovery; new candidates still use full VBC.
        with self.lock:record=self.deferred.get(key)
        if record is None:return False
        if record.get('reason')=='alternative_reference':
            # A new candidate that still sweeps this point must create a new
            # observation. Only superseded candidate work is excluded here.
            source_stamp=int((identity or {}).get('trajectory_stamp_ns',0))
            if not source_stamp or source_stamp>record['source_cutoff_ns']:
                with self.lock:
                    self.deferred.pop(key,None);self.relevance_excluded.add(key)
                self.trace('observation_reactivated',points=list(key),reason='new_candidate_requires_region')
                return False
            return True
        occupied,stamp=self.snapshot()
        if occupied is None:return False
        # Recheck the view that actually failed; a changed obstacle map can
        # reactivate the observation instead of leaving a permanent blacklist.
        still_blocked=all(any(first_blocker(e['sensor_origin'],p,occupied) is not None
                             for p in region['points']) for e in record['evidence'])
        if not still_blocked:
            with self.lock:self.deferred.pop(key,None)
            self.trace('observation_reactivated',points=list(key),map_stamp_s=stamp)
            return False
        return True

    def tick(self,_event):
        try:self._tick()
        except Exception as exc:
            self.trace('runtime_error',error=repr(exc));rospy.logerr_throttle(1.,'[occlusion_reroute] %s',exc)

    def _tick(self):
        now=time.monotonic()
        measured=getattr(self.node,'_trace_measured',(None,math.nan))
        if measured[0] is None or not 0<=rospy.Time.now().to_sec()-measured[1]<=.2:return
        occupied,stamp=self.snapshot()
        with self.node._obligation_lock:current_obligations=[dict(o) for o in self.node._obligations]
        with self.lock:
            if self.nominal is not None and occupied is not None and len(occupied) and self.route is None:
                signature=point_key(occupied)
                if signature not in self.screened_maps and not self.busy:
                    self.screened_maps.add(signature);self.pending=True;self.reason='nominal_occupied_screen'
            # A bounded alternative to repeatedly observing prerequisites of
            # prerequisites. These remain UNKNOWN; only choose a new reference
            # if its primitive sweep avoids every current candidate obligation.
            candidates=[o for o in current_obligations if o.get('obligation_bundle_source')=='candidate']
            # Preserve a found route while it acquires information. A timer on
            # UNKNOWN alone is not evidence for a different route; it can return
            # the same suffix and discard its ongoing observation work.
            if (not self.task_first_research and self.route is not None and
                    self.regime=='REPAIR' and candidates and not self.busy):
                key=point_key([p for o in candidates for p in o['points']])
                first=self.sensing_first_seen.setdefault(key,now)
                if now-first>=4. and key not in self.sensing_retry_keys and len(self.sensing_retry_keys)<3:
                    self.sensing_retry_keys.add(key);self.pending=True;self.reason='sensing_route_dependency'
            if self.pending and not self.busy and self.nominal is not None and occupied is not None and now-self.last_attempt>=2.:
                self.busy=True;self.pending=False;self.last_attempt=now;self.generation+=1
                args=(np.asarray(measured[0]).copy(),np.asarray(self.nominal.points[-1].positions),occupied,
                    [p for ob in self.deferred.values() if ob.get('evidence') for p in ob['points']],self.goal_revision,self.generation,self.reason,
                    list(self.goal[1:]) if self.goal is not None else None,current_obligations)
                threading.Thread(target=self.build_route,args=args,daemon=True).start()
            active=self.route is not None
        with self.node._obligation_lock:observing=bool(self.node._obligations)
        if active and now-self.last_publish>=.75 and (self.regime!='REPAIR' or not observing):
            self.publish_route(np.asarray(measured[0]))

    def build_route(self,start,goal,occupied,avoid,revision,generation,reason,goal_pose,current_obligations):
        try:
            if self.proposer is None:
                runtime=Path(__file__).resolve().parents[3]
                self.proposer=RouteProposer(runtime,self.node.urdf_path,self.node.device)
            lower=self.node.q_min.detach().cpu().numpy().reshape(7)
            upper=self.node.q_max.detach().cpu().numpy().reshape(7)
            if reason=='nominal_occupied_screen':
                distance=float(self.proposer.distances(sample_polyline([start,goal]),occupied).min())
                self.trace('nominal_occupied_screen',cdf_min=distance,generation=generation)
                if distance>=.010:return
            unknown_avoid=([p for o in current_obligations if o.get('obligation_bundle_source')=='candidate'
                            for p in o['points']] if reason=='sensing_route_dependency' else [])
            route,diag=self.proposer.propose(start,goal,lower,upper,occupied,avoid,generation,
                goal_pose=goal_pose,unknown_avoid=unknown_avoid,
                wall_budget=1.5 if self.task_first_research else 4.)
            with self.lock:
                if revision!=self.goal_revision:return
                if route is not None:
                    self.route=route;self.route_index=1;self.route_generation=generation
                    self.route_cutoff_ns=rospy.Time.now().to_nsec()
            if route is not None:
                tokens={observation_token(o) for o in current_obligations}
                with self.node._obligation_lock:
                    removed=[o for o in self.node._obligations if observation_token(o) in tokens and
                             o.get('obligation_bundle_source')=='candidate']
                    self.node._obligations=[o for o in self.node._obligations if not any(o is old for old in removed)]
                with self.lock:
                    for ob in removed:
                        key=point_key(ob['points']);token=observation_token(ob)
                        view=dict(ob);published=getattr(self.node,'_trace_published_target',None)
                        if published is not None and published.get('region_token')==observation_region_token(ob):
                            restore_observation_target_state(view,published)
                        self.save_result(view,'alternative_route')
                        if key not in self.deferred:
                            self.deferred[key]=dict(obligation_id=int(ob['id']),token=token,
                                published_token=observation_token(view),
                                points=np.asarray(ob['points']).tolist(),reason='alternative_reference',
                                source_cutoff_ns=self.route_cutoff_ns,evidence=[])
                        self.trace('observation_deferred_for_route',token=token,points=np.asarray(ob['points']).tolist(),
                            generation=generation,actual_observation_completed=False)
            self.trace('route_proposal',generation=generation,reason=reason,start=start.tolist(),goal=goal.tolist(),
                occupied=occupied.tolist(),avoid=avoid,route=None if route is None else route.tolist(),diagnostics=diag,
                proposal_only=True)
            if route is not None:rospy.logwarn('[occlusion_reroute] route=%d waypoints=%d checked=%d',generation,len(route),diag['checked'])
        except Exception as exc:
            self.trace('route_error',error=repr(exc));rospy.logerr('[occlusion_reroute] route failed: %s',exc)
        finally:
            with self.lock:self.busy=False

    def publish_route(self,measured):
        with self.node._obligation_lock:active=bool(self.node._obligations)
        with self.lock:
            if self.route is None:return
            while self.route_index<len(self.route)-1 and np.max(np.abs(measured-self.route[self.route_index]))<.12:
                self.route_index+=1
            if (os.environ.get('CARE_REPAIR_UNKNOWN_SOFT','false')=='true' and not active and
                    self.regime=='NORMAL' and self.route_index==len(self.route)-1 and
                    self.proposer is not None and self.goal is not None and self.goal[0]=='base_link'):
                pose,_=self.proposer.pose_jacobian(measured)
                position=float(np.linalg.norm(pose[:3,3]-np.asarray(self.goal[1:4])))
                desired=quaternion_matrix(self.goal[4:])
                angle=math.acos(float(np.clip((np.trace(desired.T@pose[:3,:3])-1.)*.5,-1.,1.)))
                if position<=.01 and angle<=.10:
                    # Repeated reference stamps cancel terminal retention before
                    # the certified tail can finish. Keep the reference, not a
                    # motion command or a success claim; live audits remain on.
                    if not self.terminal_reference_retained:
                        self.trace('terminal_reference_retained',position_error_m=position,angle_error_rad=angle)
                    self.terminal_reference_retained=True;self.last_publish=time.monotonic()
                    return
            self.terminal_reference_retained=False
            knots=np.vstack([measured,self.route[self.route_index:]])
            generation=self.route_generation;revision=self.goal_revision
        # Publish a receding reference from actual measured progress. Pausing
        # for sensing never lets a wall-clock reference skip the detour point.
        msg=JointTrajectory();msg.header.frame_id='base_link';msg.header.stamp=rospy.Time.now()
        msg.joint_names=list(DEFAULT_JOINT_NAMES);elapsed=0.
        for k,q in enumerate(knots):
            if k:elapsed+=max(.15,float(np.max(np.abs(q-knots[k-1])))/.35)
            p=JointTrajectoryPoint();p.positions=q.tolist();p.velocities=[0.]*7;p.time_from_start=rospy.Duration(elapsed);msg.points.append(p)
        self.reference_pub.publish(msg);self.last_publish=time.monotonic()
        with self.node._obligation_lock:active=bool(self.node._obligations)
        if not active and time.monotonic()-self.last_route_event>=.75:
            event=dict(schema=1,action='try_alternative_reference',goal_revision=revision,route_generation=generation,
                reference_stamp_ns=str(msg.header.stamp.to_nsec()),deferred_tokens=list({t for v in self.deferred.values()
                    for t in (v['token'],v.get('published_token',v['token']))}),
                actual_observation_completed=False,goal_key=self.goal)
            self.event_pub.publish(String(data=json.dumps(event,separators=(',',':'))))
            self.last_route_event=time.monotonic()
        self.trace('route_reference',generation=generation,reference_stamp_ns=str(msg.header.stamp.to_nsec()),
            waypoint_index=self.route_index,active_observations=active,knots=knots.tolist())


def attach_occlusion_reroute(node):
    if os.environ.get('CARE_OCCLUSION_REROUTE_ENABLED','false').lower()!='true':return None
    return OcclusionReroute(node)
