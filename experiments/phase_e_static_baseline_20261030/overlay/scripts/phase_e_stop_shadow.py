#!/usr/bin/env python3
"""Read-only conditional stop enclosure audit. Never grants actuator admission."""
import copy
import hashlib
import math
import threading
from collections import OrderedDict
import numpy as np
from phase_e_stop_envelope import stopping_budget, primitive_radii, JOINTS

BLOCKERS=['missing_physical_response_bound','update_gap_bound_unqualified',
          'command_channel_not_fenced','missing_future_environment_lease']


def identity(command):
    return (command['stamp_ns'],command['sequence'])


def finite7(values):
    return len(values)==7 and all(math.isfinite(v) for v in values)


class EvidenceBuffer:
    def __init__(self,capacity=4096):
        self.lock=threading.Lock(); self.capacity=capacity
        self.state=None; self.commands=OrderedDict(); self.serial=0; self.regressions=0

    def on_state(self,state):
        with self.lock:
            if self.state and state['t_ns']<self.state['t_ns']: self.regressions+=1
            self.state=copy.deepcopy(state)

    def on_command(self,command):
        with self.lock:
            self.serial+=1; key=identity(command); previous=self.commands.get(key)
            conflict=bool(previous and (previous['conflict'] or previous['command']!=command))
            self.commands[key]=dict(command=copy.deepcopy(command),serial=self.serial,conflict=conflict)
            self.commands.move_to_end(key)
            while len(self.commands)>self.capacity: self.commands.popitem(last=False)

    def capture(self,base=None):
        with self.lock:
            state=copy.deepcopy(self.state)
            key=identity(base or state) if (base or state) else None
            entry=copy.deepcopy(self.commands.get(key))
            # Include already received but not yet acknowledged commands, not
            # just packets arriving after the query starts. Older replays cannot
            # renew this core's strictly increasing generation-time lease.
            pending=[list(k) for k in self.commands if key and k[0]>key[0]]
            return dict(state=state,entry=entry,pending=pending,serial=self.serial,regressions=self.regressions)


def version(s): return (s['instance'],s['epoch'],s['revision'])


def geometry_key(s):
    return (s['frame'],s['resolution'],tuple(s['map_min']),tuple(s['map_max']),tuple(s['grid_size']))


class PrimitiveEnclosure:
    """Reuse the audited Pinocchio FK, primitive SDF and global joint radii."""
    def __init__(self,urdf,max_grid_cells=250000):
        import pinocchio as pin
        import xml.etree.ElementTree as ET
        from audit_case001_motion_prior import load_shapes
        self.pin=pin; self.model=pin.buildModelFromUrdf(str(urdf)); self.data=self.model.createData()
        self.shapes=load_shapes(ET.parse(urdf).getroot())
        self.active=[i for i,s in enumerate(self.shapes) if s['link'] not in ('base_link','link1')]
        self.radii=primitive_radii(urdf,self.shapes)
        self.qindices=[self.model.joints[self.model.getJointId(j)].idx_q for j in JOINTS]
        self.key=None; self.max_grid_cells=max_grid_cells

    def raster(self,q,travel,meta,margin=.020):
        from scipy.spatial import cKDTree
        from audit_case001_motion_prior import frame_placements,collision_sdf
        key=geometry_key(meta)
        if meta['frame']!='base_link': raise ValueError('unsupported_map_frame')
        low=np.array(meta['map_min']); high=np.array(meta['map_max']); dims=np.array(meta['grid_size'])
        resolution=meta['resolution']
        if (not np.isfinite(np.r_[low,high,resolution]).all() or resolution<=0 or
            np.any(high<=low) or np.any(dims<=0) or not np.all(dims==dims.astype(int))):
            raise ValueError('invalid_grid_geometry')
        if math.prod(map(int,dims))>self.max_grid_cells: raise ValueError('grid_resource_limit_no_truncation')
        if np.any(low+(dims-1)*resolution>high+1e-10): raise ValueError('inconsistent_grid_geometry')
        if key!=self.key:
            axes=[lo+resolution*np.arange(int(n)) for lo,n in zip(low,dims)]
            self.points=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
            self.tree=cKDTree(self.points); self.key=key
        full=np.zeros(self.model.nq); full[self.qindices]=q
        placements=frame_placements(self.model,self.data,self.shapes,full)
        bounds=self.radii@travel; mask=np.zeros(len(self.points),dtype=bool); inside=True
        for i in self.active:
            s,p=self.shapes[i],placements[i]; inflation=margin+float(bounds[i])
            nearby=np.asarray(self.tree.query_ball_point(p.translation,s['circumradius']+inflation+1e-9),dtype=int)
            if nearby.size: mask[nearby] |= collision_sdf(self.points[nearby],s,p)<=inflation+1e-12
            rotation=np.asarray(p.rotation)
            if s['kind']=='box': extent=np.abs(rotation)@s['half_size']
            elif s['kind']=='cylinder': extent=s['radius']*np.sqrt(rotation[:,0]**2+rotation[:,1]**2)+s['half_length']*np.abs(rotation[:,2])
            else: extent=np.full(3,s['radius'])
            extent+=inflation
            inside=inside and bool(np.all(p.translation-extent>=low) and np.all(p.translation+extent<=high))
        ids=np.flatnonzero(mask); points=self.points[ids]
        return dict(points=points,grid_indices=ids.tolist(),inside=inside,primitive_bounds_m=bounds.tolist(),
                    point_set_sha256=hashlib.sha256(np.asarray(points,dtype='<f8').tobytes()).hexdigest())


class ShadowAudit:
    def __init__(self,buffer,geometry,query,clock_ns,config,max_state_age=.1,assumed_gap=.001):
        self.buffer=buffer; self.geometry=geometry; self.query=query; self.clock_ns=clock_ns
        self.config=config; self.max_state_age=max_state_age; self.assumed_gap=assumed_gap

    def run(self):
        row=dict(protocol='content_snapshot_then_recheck_v2',runtime_admission=False,physical_bound_certified=False,reasons=list(BLOCKERS),
                 binding_consistent=False,conditional_map_coverage=False,scope='conditional command-domain grid-center enclosure')
        issues=[]
        def reject(reason):
            issues.append(reason)
        try:
            meta=self.query([],0.,None)
            row['metadata']=meta
            if not meta['valid']: raise ValueError('metadata_'+meta['reason'])
            captured=self.buffer.capture(); state=captured['state']; row['state']=state
            if not state: raise ValueError('missing_controller_state')
            row['initial_pending_newer_request_ids']=captured['pending']
            if captured['pending']: reject('newer_request_observed')
            row['evaluation_start_ns']=self.clock_ns()
            if state['mode'] not in (1,2,3): raise ValueError('controller_not_tracking_or_stopping')
            if state['clock_fault'] or not state['integral_valid'] or captured['regressions']:
                raise ValueError('invalid_controller_time')
            if not all(finite7(state[k]) for k in ('q','applied','measured')): raise ValueError('invalid_controller_vectors')
            age=(row['evaluation_start_ns']-state['t_ns'])/1e9
            if age<0 or age>self.max_state_age: raise ValueError('stale_or_future_measured_state')
            if not math.isfinite(state['max_update_gap']) or state['max_update_gap']>self.assumed_gap+1e-9:
                raise ValueError('observed_update_gap_exceeds_assumption')
            entry=captured['entry']
            if entry is None: raise ValueError('missing_exact_command_identity')
            if entry['conflict']: raise ValueError('ambiguous_command_identity')
            cmd=entry['command']; row['exact_command']=cmd
            if cmd['names']!=JOINTS or not finite7(cmd['velocity']): raise ValueError('invalid_command_payload')
            if cmd['stamp_ns']<=0 or cmd['stamp_ns']>state['t_ns']: raise ValueError('invalid_command_time')
            command_age=(row['evaluation_start_ns']-cmd['stamp_ns'])/1e9
            target=cmd['velocity'] if state['mode']==1 else [0.]*7
            if state['mode'] in (2,3): command_age=max(command_age,self.config['command_timeout'])
            budget=stopping_budget(state['applied'],target,command_age,age,self.assumed_gap,
                self.config['acceleration_limits'],self.config['velocity_limits'],self.config['command_timeout'],self.config['max_braking_step'])
            row['budget']=dict(joint_travel_rad=budget.command_travel_rad.tolist(),command_stop_s=budget.command_stop_s,
                               measured_age_s=age,assumed_update_gap_s=self.assumed_gap,
                               required_until_ns=row['evaluation_start_ns']+int(math.ceil(budget.command_stop_s*1e9)))
            footprint=self.geometry.raster(state['q'],budget.command_travel_rad,meta)
            row['enclosure']={k:v for k,v in footprint.items() if k!='points'}
            if not footprint['inside']: reject('enclosure_may_leave_map')
            points=footprint['points']; n=len(points)
            if not n: raise ValueError('empty_enclosure')
            # Geometry does not depend on the confidence content revision.
            # Read the entire point set at ONE current content version, then
            # bind the recheck to that version. Never reuse an earlier answer.
            evidence=self.query(points,budget.command_stop_s,None); row['evidence']=evidence
            if not evidence['valid']: raise ValueError('snapshot_'+evidence['reason'])
            if version(evidence)[:2]!=version(meta)[:2]: raise ValueError('unexpected_snapshot_instance_or_epoch')
            if evidence['revision']<meta['revision']: raise ValueError('snapshot_revision_regressed')
            if geometry_key(evidence)!=geometry_key(meta): raise ValueError('map_geometry_changed')
            array_fields=('inside_map','confidence_now','confidence_at_horizon','occupancy','last_seen_s','provenance','model_free_at_horizon')
            if any(len(evidence[k])!=n for k in array_fields): raise ValueError('incomplete_snapshot_arrays')
            if any(not all(math.isfinite(v) for v in evidence[k]) for k in array_fields): raise ValueError('nonfinite_snapshot_evidence')
            if evidence['clock_fault'] or evidence['snapshot_ns']<meta['snapshot_ns']: raise ValueError('invalid_snapshot_time')
            if abs(evidence['horizon_s']-budget.command_stop_s)>1e-9: raise ValueError('snapshot_horizon_mismatch')
            if evidence['snapshot_ns']+int(round(evidence['horizon_s']*1e9))<row['budget']['required_until_ns']-1:
                raise ValueError('evidence_horizon_too_short')
            inside=np.array(evidence['inside_map'])==1
            free=np.array(evidence['model_free_at_horizon'])==1
            occupancy=np.array(evidence['occupancy']); confidence=np.array(evidence['confidence_at_horizon'])
            if np.any((occupancy<0)|(occupancy>1)|(confidence<0)|(confidence>1)):
                raise ValueError('invalid_snapshot_values')
            # Check numbers independently of the server's convenience mask.
            free &= (occupancy<=.5)&(confidence>=.8)&inside
            row['evidence_counts']=dict(points=n,inside=int(inside.sum()),model_free=int(free.sum()),
                occupied=int((occupancy>.5).sum()),sensor=int(np.sum(np.array(evidence['provenance'])&1>0)),
                bootstrap=int(np.sum(np.array(evidence['provenance'])&2>0)),oracle=int(np.sum(np.array(evidence['provenance'])&4>0)))
            if not inside.all(): reject('query_points_outside_map')
            if not free.all(): reject('unknown_occupied_or_decayed_points')
            checked=self.query([],0.,version(evidence)); row['version_recheck']=checked
            if not checked['valid'] or version(checked)!=version(evidence): reject('map_version_changed_during_check')
            if geometry_key(checked)!=geometry_key(evidence): reject('map_geometry_changed')
            if checked['snapshot_ns']<evidence['snapshot_ns']: reject('snapshot_time_regressed')
            row['conditional_map_coverage']=bool(not issues)
        except (ValueError,RuntimeError,KeyError,TypeError,OverflowError) as exc:
            reject(str(exc))
        finally:
            if row.get('state'):
                end=self.buffer.capture(row['state']); latest=end['state']; row['end_state']=latest
                row['evaluation_end_ns']=self.clock_ns()
                row['pending_newer_request_ids']=end['pending']
                if end['pending']: reject('newer_request_observed')
                if end['entry'] and end['entry']['conflict']: reject('ambiguous_command_identity')
                if latest and (identity(latest)!=identity(row['state']) or latest['rearms']!=row['state']['rearms']):
                    reject('controller_command_changed_during_check')
                if latest and (latest['clock_fault'] or not latest['integral_valid'] or end['regressions']):
                    reject('invalid_controller_time')
                age=(row['evaluation_end_ns']-row['state']['t_ns'])/1e9
                row['end_measured_age_s']=age
                if age<0 or age>self.max_state_age: reject('state_aged_out_during_check')
                for snapshot_key in ('metadata','evidence','version_recheck'):
                    snapshot=row.get(snapshot_key)
                    if snapshot and (snapshot['snapshot_ns']>row['evaluation_end_ns'] or snapshot['map_update_ns']>snapshot['snapshot_ns']):
                        reject('future_snapshot_time')
                if row.get('evaluation_start_ns',row['evaluation_end_ns'])>row['evaluation_end_ns']:
                    reject('client_clock_regressed')
                if 'captured' in locals() and end['serial']-captured['serial']>=self.buffer.capacity:
                    reject('command_history_overflow')
            row['binding_consistent']=not issues and 'version_recheck' in row
            # Point coverage alone is kept separate from command/time binding.
            row['reasons']+=list(dict.fromkeys(issues))
        return row
