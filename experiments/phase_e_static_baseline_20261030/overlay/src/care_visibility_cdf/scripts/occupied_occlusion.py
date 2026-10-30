"""External line-of-sight proposals from measured occupied voxel supports.

Occlusion is scoped to (map snapshot, sensor pose, point). It never updates
confidence/occupancy and never certifies a trajectory or global invisibility.
"""
import numpy as np
import os


def quaternion_matrix(quaternion):
    q=np.asarray(quaternion,dtype=float);length=np.linalg.norm(q)
    if q.shape!=(4,) or not np.isfinite(q).all() or length<1e-12:raise ValueError('invalid target quaternion')
    x,y,z,w=q/length
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def rotation_vector(rotation):
    r=np.asarray(rotation,dtype=float)
    angle=float(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))
    skew=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    if angle<1e-7:return skew/2
    if np.pi-angle<1e-5:
        _,vectors=np.linalg.eigh((r+r.T+2*np.eye(3))/4);axis=vectors[:,-1]
        if axis@skew<0:axis=-axis
        return angle*axis
    return angle/(2*np.sin(angle))*skew


def first_blocker(origin, target, centers, resolution=.05, endpoint_gap=.002):
    origin=np.asarray(origin,dtype=float).reshape(3)
    target=np.asarray(target,dtype=float).reshape(3)
    centers=np.asarray(centers,dtype=float).reshape(-1,3)
    if not (np.isfinite(origin).all() and np.isfinite(target).all() and
            np.isfinite(centers).all() and np.isfinite(resolution) and resolution>0):
        raise ValueError('invalid occlusion geometry')
    ray=target-origin;length=float(np.linalg.norm(ray))
    if length<=endpoint_gap or not len(centers):return None
    lo=centers-resolution/2;hi=centers+resolution/2
    enter=np.zeros(len(centers));leave=np.ones(len(centers));valid=np.ones(len(centers),bool)
    for j in range(3):
        if abs(ray[j])<1e-12:
            valid&=(origin[j]>=lo[:,j])&(origin[j]<=hi[:,j])
        else:
            a=(lo[:,j]-origin[j])/ray[j];b=(hi[:,j]-origin[j])/ray[j]
            enter=np.maximum(enter,np.minimum(a,b));leave=np.minimum(leave,np.maximum(a,b))
    valid&=(enter<=leave)&(enter*length<length-endpoint_gap)
    indices=np.flatnonzero(valid)
    if not len(indices):return None
    i=int(indices[np.argmin(enter[indices])])
    return dict(center=centers[i].tolist(),entry_m=float(enter[i]*length),target_m=length)


def sample_polyline(knots, step=.08):
    knots=np.asarray(knots,dtype=float).reshape(-1,7)
    if len(knots)<2 or not np.isfinite(knots).all() or step<=0:
        raise ValueError('invalid route')
    rows=[knots[0]]
    for a,b in zip(knots[:-1],knots[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/step)))
        rows.extend(a+(b-a)*t/n for t in range(1,n+1))
    return np.asarray(rows)


def route_candidates(start, goal, lower, upper, generation=0):
    """Finite deterministic joint-space proposals; no scene truth or saved path."""
    start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
    lower=np.asarray(lower);upper=np.asarray(upper)
    if not all(x.shape==(7,) and np.isfinite(x).all() for x in (start,goal,lower,upper)):
        raise ValueError('invalid route bounds')
    candidates=[np.asarray([start,goal])]
    # Vary the arm configuration rather than making a Cartesian EE-only detour.
    directions=[]
    for j in range(4):
        for sign in (-1.,1.):
            d=np.zeros(7);d[j]=sign;directions.append(d)
    for a,b in [(0,1),(0,3),(1,2),(1,3),(2,3)]:
        for sa,sb in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
            d=np.zeros(7);d[a]=sa;d[b]=sb;directions.append(d)
    for amplitude in (.35,.70,1.05):
        for d in directions:
            mid=np.clip((start+goal)/2+amplitude*d,lower,upper)
            candidates.append(np.asarray([start,mid,goal]))
    # Reproducible additional combinations, independent of the test geometry.
    rng=np.random.default_rng(1701+int(generation))
    for _ in range(32):
        mid=np.clip((start+goal)/2+rng.normal(0,.55,7)*[1,1,1,1,.4,.4,.4],lower,upper)
        candidates.append(np.asarray([start,mid,goal]))
    return sorted(candidates,key=lambda p:float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1))))


class RouteProposer:
    """Sampled CDF/ground filtering only; original downstream gates stay active."""
    def __init__(self, runtime, urdf, device):
        import sys,json
        from pathlib import Path
        runtime=Path(runtime)
        for rel in ('src/care_collision_cdf/scripts','src/egocentric_arm_planner/scripts',
                    'src/care_visibility_cdf/scripts','src/care_confidence_map/scripts'):
            path=str(runtime/rel)
            if path not in sys.path:sys.path.append(path)
        from collision_cdf_model import CollisionCDF
        from known_ground import KnownGround
        binding=json.loads((runtime.parent/'old_model_binding.json').read_text())
        self.cdf=CollisionCDF(binding['checkpoint'],device=str(device),activation=binding['activation'])
        self.ground=KnownGround(str(urdf),str(runtime/'src/egocentric_arm_planner/config/known_ground_geometry.json'),backend='native')
        self.ee_chain=[];name='EE_link'
        while name!='base_link':
            data=self.ground.frames[name];self.ee_chain.append((name,data));name=data[0]
        self.ee_chain.reverse();self.ik_cache={}
        from primitive_point_geometry import load_primitives
        self.point_primitives=load_primitives(runtime/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
        self.point_weights={}
        for link,primitives in self.point_primitives.items():
            for primitive in primitives:
                dimensions=primitive.dimensions
                extent=(dimensions[0] if primitive.kind=='sphere' else
                    np.hypot(dimensions[0],dimensions[1]/2) if primitive.kind=='cylinder' else
                    np.linalg.norm(np.asarray(dimensions)/2))
                radius=float(np.linalg.norm(primitive.center)+extent);weights=np.zeros(7);name=link
                while name!='base_link':
                    parent,origin,kind,axis,index=self.ground.frames[name]
                    if kind in ('revolute','continuous'):weights[index]=radius
                    radius+=float(np.linalg.norm(origin[:3,3]));name=parent
                self.point_weights[(link,primitive.collision_index)]=weights

    def point_clearance_bound(self,knots,points):
        """Lower bound for this joint-linear proposal, not an executable curve.

        Exact primitive point distance at interval centers minus a rigid-motion
        displacement bound. Original point model; no full-voxel claim.
        """
        q=sample_polyline(knots,.06);points=np.asarray(points).reshape(-1,3);minimum=np.inf
        for a,b in zip(q[:-1],q[1:]):
            mid=(a+b)/2;radius=np.abs(a-b)/2;cache={'base_link':np.eye(4)}
            def pose(name):
                if name not in cache:
                    parent,origin,kind,axis,index=self.ground.frames[name];tf=pose(parent)@origin
                    if kind in ('revolute','continuous'):
                        outer,skew=self.ground.rotations[name];c=np.cos(mid[index]);s=np.sin(mid[index])
                        tf[:3,:3]=tf[:3,:3]@(c*np.eye(3)+(1-c)*outer+s*skew)
                    cache[name]=tf
                return cache[name]
            for link,primitives in self.point_primitives.items():
                tf=pose(link);local=(points-tf[:3,3])@tf[:3,:3]
                for primitive in primitives:
                    bound=float(self.point_weights[(link,primitive.collision_index)]@radius)
                    value=min(primitive.distance(point) for point in local)-bound
                    minimum=min(minimum,value)
                    if minimum<.020:return minimum
        return float(minimum)

    def pose_jacobian(self,q):
        tf=np.eye(4);axes={};origins={}
        for name,(_,origin,kind,axis,index) in self.ee_chain:
            tf=tf@origin
            if kind in ('revolute','continuous'):
                axes[index]=tf[:3,:3]@(axis/np.linalg.norm(axis));origins[index]=tf[:3,3].copy()
                outer,skew=self.ground.rotations[name];c=np.cos(q[index]);s=np.sin(q[index])
                tf[:3,:3]=tf[:3,:3]@(c*np.eye(3)+(1-c)*outer+s*skew)
            elif kind=='prismatic':raise ValueError('unsupported prismatic IK')
        jac=np.zeros((6,7))
        for index,axis in axes.items():
            jac[:3,index]=np.cross(axis,tf[:3,3]-origins[index]);jac[3:,index]=axis
        return tf,jac

    def goal_options(self,start,nominal,pose,lower,upper,points):
        target=np.asarray(pose,dtype=float);key=tuple(np.round(target,8))
        if key not in self.ik_cache:
            desired=quaternion_matrix(target[3:])
            rng=np.random.default_rng(4201);solutions=[np.asarray(nominal)]
            seeds=[nominal,start]+[np.clip(nominal+rng.normal(0,1.,7),lower,upper) for _ in range(24)]
            for seed in seeds:
                q=np.asarray(seed).copy()
                for _ in range(100):
                    tf,jac=self.pose_jacobian(q)
                    dp=target[:3]-tf[:3,3];dr=rotation_vector(desired@tf[:3,:3].T)
                    if np.linalg.norm(dp)<.004 and np.linalg.norm(dr)<.04:
                        if all(np.max(np.abs(q-old))>.10 for old in solutions):solutions.append(q.copy())
                        break
                    error=np.r_[dp,.3*dr];jac[3:]*=.3
                    step=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-4,error)
                    step*=min(1.,.25/max(1e-12,np.max(np.abs(step))))
                    q=np.clip(q+step,lower,upper)
            self.ik_cache[key]=solutions
        solutions=self.ik_cache[key]
        distances=self.distances(solutions,points)
        valid=[q for q,d in zip(solutions,distances) if d>=.010 and self.ground.clearance(q)[0]>=.020-1e-10]
        return sorted(valid,key=lambda q:float(np.linalg.norm(q-start))),dict(
            ik_solutions=len(solutions),collision_filtered_goal_options=len(valid),goal_cdf_values=distances.tolist())

    def distances(self, q, points):
        import torch
        q=np.asarray(q,dtype=float).reshape(-1,7);points=np.asarray(points,dtype=float).reshape(-1,3)
        if not len(points):return np.full(len(q),np.inf)
        # Small chunks avoid monopolizing the GPU used by the real planner.
        result=[]
        with torch.no_grad():
            pt=torch.as_tensor(points,device=self.cdf.device,dtype=torch.float32)
            count=max(1,2048//len(points))
            for off in range(0,len(q),count):
                qt=torch.as_tensor(q[off:off+count],device=self.cdf.device,dtype=torch.float32)
                inputs=torch.cat((pt[None].expand(len(qt),-1,-1),qt[:,None].expand(-1,len(pt),-1)),dim=2)
                result.extend(self.cdf.model(inputs.reshape(-1,10)).reshape(len(qt),len(pt)).min(dim=1).values.cpu().tolist())
        return np.asarray(result)

    def propose(self,start,goal,lower,upper,occupied,avoid,generation=0,wall_budget=4.0,goal_pose=None,unknown_avoid=()):
        import time
        begin=time.monotonic()
        research=os.environ.get('CARE_TASK_FIRST_RESEARCH')=='true'
        observation_route=None;observation_diag=None
        if len(unknown_avoid) and not research:
            initial=self.point_clearance_bound([start,start],unknown_avoid)
            if initial<.020:
                return None,dict(checked=0,reason='unknown_support_at_current_body',
                    current_point_clearance_m=initial,unknown_avoid_count=len(unknown_avoid),
                    elapsed_s=time.monotonic()-begin)
        points=np.unique(np.round(np.vstack([np.asarray(occupied).reshape(-1,3),np.asarray(avoid).reshape(-1,3)]),6),axis=0)
        endpoint={};goals=[goal]
        if goal_pose is not None:goals,endpoint=self.goal_options(start,goal,goal_pose,lower,upper,points)
        candidates=[]
        for option in goals[:8]:candidates.extend(route_candidates(start,option,lower,upper,generation))
        candidates.sort(key=lambda p:float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1))))
        checked=0;best=-float('inf');minimum_ground=float('inf');unknown_rejected=0;budget_exhausted=False
        for knots in candidates:
            if time.monotonic()-begin>wall_budget:
                budget_exhausted=True;break
            q=sample_polyline(knots);checked+=1
            d=self.distances(q,points);minimum=float(np.min(d));best=max(best,minimum)
            if minimum<.010:continue
            ground=min(self.ground.clearance(row)[0] for row in q)
            minimum_ground=ground
            if ground+1e-10<.020:continue
            if len(unknown_avoid) and self.point_clearance_bound(knots,unknown_avoid)<.020:
                unknown_rejected+=1
                if research and observation_route is None:
                    observation_route=knots
                    observation_diag=dict(cdf_min=minimum,ground_min=ground,**endpoint,
                        sampled_poses=len(q),requires_observation=True)
                continue
            return knots,dict(checked=checked,cdf_min=minimum,ground_min=ground,**endpoint,
                sampled_poses=len(q),elapsed_s=time.monotonic()-begin,points=len(points),
                unknown_avoid_count=len(unknown_avoid),unknown_rejected=unknown_rejected)
        if observation_route is not None:
            return observation_route,dict(observation_diag,checked=checked,elapsed_s=time.monotonic()-begin,
                points=len(points),unknown_avoid_count=len(unknown_avoid),unknown_rejected=unknown_rejected,
                reason='bounded_route_with_observation',proposal_only=True)
        return None,dict(checked=checked,best_cdf_min=best,last_ground_min=minimum_ground,**endpoint,
            elapsed_s=time.monotonic()-begin,points=len(points),unknown_avoid_count=len(unknown_avoid),
            unknown_rejected=unknown_rejected,
            reason='proposal_time_budget_exhausted' if budget_exhausted else 'finite_candidate_pool_exhausted')
