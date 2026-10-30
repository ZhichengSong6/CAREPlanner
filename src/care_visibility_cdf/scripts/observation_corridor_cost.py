"""Bounded current-map path preview for proposal ranking, with no safety authority."""
import math
import threading
import numpy as np
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, fk_transform


class BoundedCorridorQuery:
    """At most one optional RPC; a slow server cannot block generation."""
    def __init__(self, query, timeout_s=.02):
        self.query=query;self.timeout_s=timeout_s
        self.lock=threading.Lock();self.busy=False

    def __call__(self, request):
        with self.lock:
            if self.busy:return None
            self.busy=True
        done=threading.Event();result=[]
        def run():
            try:result.append(self.query(request))
            except Exception:pass
            finally:
                with self.lock:self.busy=False
                done.set()
        threading.Thread(target=run,daemon=True).start()
        return result[0] if done.wait(self.timeout_s) and result else None


def body_path_probes(runtime, q_start, q_end, resolution=.05, limit=96):
    """Four joint-path samples, active URDF solids, bounded voxel probes.

    These samples estimate new observation work; they do not certify a swept
    path, collision avoidance, or visibility. Each leg has an equal probe cap.
    """
    cells=set()
    for q in np.linspace(np.asarray(q_start),np.asarray(q_end),4):
        qmap=dict(zip(DEFAULT_JOINT_NAMES,q));frames={}
        for primitive,chain in zip(runtime.primitives,runtime.collision_chains):
            link=primitive['link']
            if link in ('base_link','link1'):continue
            if link not in frames:frames[link]=fk_transform(chain,qmap)
            transform=frames[link]@primitive['origin'];g=primitive['geometry']
            half=(np.asarray(g.size)*.5 if primitive['type']=='box' else
                  np.asarray([g.radius,g.radius,g.length*.5]) if primitive['type']=='cylinder' else
                  np.full(3,g.radius))
            local=[np.zeros(3)]
            for axis in range(3):
                direction=np.zeros(3);direction[axis]=half[axis]
                local.extend([direction,-direction])
            for point in local:
                p=transform[:3,:3]@point+transform[:3,3]
                if not np.all(np.isfinite(p)):raise ValueError('nonfinite path preview')
                cells.add(tuple(np.rint(p/resolution).astype(int)))
    ordered=sorted(cells)
    if len(ordered)>limit:
        ordered=[ordered[i] for i in np.linspace(0,len(ordered)-1,limit,dtype=int)]
    return [tuple(float(x)*resolution for x in cell) for cell in ordered]


def approach_visibility_order(runtime, points, q_start, q_end, sensor_id):
    """Rank an available straight approach against one needing a detour.

    A sampled path is a proposal hint only. A missing clear straight path does
    not reject the first accepted fallback or prove the absence of a detour.
    The exact runtime confidence and complete execution gates still decide.
    """
    if not getattr(runtime,'primitives',None) or not getattr(runtime,'collision_chains',None):
        return dict(status='unavailable')
    points=np.asarray(points,dtype=float).reshape(-1,3)
    start=np.asarray(q_start,dtype=float).reshape(7);end=np.asarray(q_end,dtype=float).reshape(7)
    if not len(points) or not np.all(np.isfinite(points)) or not np.all(np.isfinite([start,end])):
        return dict(status='unavailable')
    seen=np.zeros(len(points),dtype=bool);first_seen=[None]*len(points);first_entry=[None]*len(points)
    minimum=np.full(len(points),float('inf'))
    for k,q in enumerate(np.linspace(start,end,17)):
        qmap=dict(zip(DEFAULT_JOINT_NAMES,q));frames={};distance=np.full(len(points),float('inf'))
        for primitive,chain in zip(runtime.primitives,runtime.collision_chains):
            link=primitive['link']
            if link in ('base_link','link1'):continue
            if link not in frames:frames[link]=fk_transform(chain,qmap)
            transform=frames[link]@primitive['origin'];g=primitive['geometry']
            local=(points-transform[:3,3])@transform[:3,:3]
            if primitive['type']=='sphere':d=np.linalg.norm(local,axis=1)-g.radius
            else:
                gaps=(np.abs(local)-np.asarray(g.size)*.5 if primitive['type']=='box' else
                      np.stack([np.linalg.norm(local[:,:2],axis=1)-g.radius,np.abs(local[:,2])-g.length*.5],axis=1))
                d=np.linalg.norm(np.maximum(gaps,0.),axis=1)+np.minimum(np.max(gaps,axis=1),0.)
            distance=np.minimum(distance,d)
        minimum=np.minimum(minimum,distance)
        for j in range(len(points)):
            if not seen[j] and distance[j]<.01 and first_entry[j] is None:first_entry[j]=k
        view=runtime._candidate_geometry(points,q,sensor_id)
        for j,p in enumerate(view['per_point']):
            if p['nominal_margin']>=0. and not p['primitive_self_occluded']:
                seen[j]=True
                if first_seen[j] is None:first_seen[j]=k
    if not np.all(np.isfinite(minimum)):return dict(status='unavailable')
    return dict(status='valid',samples=17,requires_detour=any(x is not None for x in first_entry),
        first_nominal_visible_sample=first_seen,first_unseen_below_execution_floor_sample=first_entry,
        minimum_body_distance_m=minimum.tolist(),execution_floor_m=.01)


def snapshot_cost(response, points, approach, rejoin, now_ns, threshold):
    if response is None:return dict(status='unavailable')
    try:
        n=len(points);stamp=response.snapshot_stamp.to_nsec()
        inside=(np.frombuffer(response.inside_map,dtype=np.uint8) if
            isinstance(response.inside_map,(bytes,bytearray)) else np.asarray(response.inside_map))
        confidence=np.asarray(response.confidence_now,dtype=float);occupancy=np.asarray(response.occupancy,dtype=float)
        if (not response.valid or response.clock_fault or response.frame!='base_link' or
                not response.instance or not 0<=now_ns-stamp<=50000000 or
                not math.isclose(float(response.resolution),.05,rel_tol=0.,abs_tol=1e-12) or
                any(len(x)!=n for x in (inside,confidence,occupancy)) or
                not np.all(np.isfinite(confidence)) or not np.all(np.isfinite(occupancy)) or
                np.any((confidence<0)|(confidence>1)|(occupancy<0)|(occupancy>1))):
            return dict(status='invalid_or_stale')
        inside=inside.astype(bool);unknown=inside&(occupancy<=.5)&(confidence<threshold)
        occupied=inside&(occupancy>.5);risk=unknown|occupied|~inside
        index={p:i for i,p in enumerate(points)}
        def leg(probes):
            ids=[index[p] for p in probes]
            return dict(probes=len(ids),unknown=int(np.sum(unknown[ids])),
                        occupied=int(np.sum(occupied[ids])),outside=int(np.sum(~inside[ids])),
                        risk_fraction=float(np.mean(risk[ids])) if ids else 0.)
        return dict(status='valid',approach=leg(approach),rejoin=leg(rejoin),
            map_identity=[response.instance,int(response.epoch)],map_revision=int(response.revision),
            snapshot_stamp_ns=stamp,query_points=n)
    except (AttributeError,KeyError,ValueError,TypeError,OverflowError):
        return dict(status='invalid_or_stale')
