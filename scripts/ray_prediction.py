"""Finite primitive background prediction; no environment truth or FREE writes."""
import numpy as np
from validate_visibility_oracle import fk_transform, DEFAULT_JOINT_NAMES

def body_hits(origin, directions, primitives, transforms):
    best=np.full(len(directions),np.inf)
    for p,T in zip(primitives,transforms):
        o=(origin-T[:3,3])@T[:3,:3];d=directions@T[:3,:3];g=p['geometry']
        if p['type']=='box':
            half=np.asarray(g.size)/2
            if np.all(abs(o)<=half):continue
            with np.errstate(divide='ignore',invalid='ignore'):
                a=(-half-o)/d;b=(half-o)/d
            low=np.max(np.minimum(a,b),axis=1);high=np.min(np.maximum(a,b),axis=1)
            hit=np.where((high>=low)&(low>1e-5),low,np.inf)
        elif p['type']=='sphere':
            if np.linalg.norm(o)<=g.radius:continue
            b=d@o;c=o@o-g.radius*g.radius;disc=b*b-c
            hit=np.where(disc>=0,-b-np.sqrt(np.maximum(disc,0)),np.inf)
            hit[hit<=1e-5]=np.inf
        else:
            if np.linalg.norm(o[:2])<=g.radius and abs(o[2])<=g.length/2:continue
            a=np.sum(d[:,:2]**2,axis=1);b=d[:,:2]@o[:2];c=o[:2]@o[:2]-g.radius*g.radius
            disc=b*b-a*c
            with np.errstate(divide='ignore',invalid='ignore'):
                t=(-b-np.sqrt(np.maximum(disc,0)))/a
            z=o[2]+t*d[:,2]
            hit=np.where((disc>=0)&(t>1e-5)&(abs(z)<=g.length/2),t,np.inf)
            for cap in [-g.length/2,g.length/2]:
                with np.errstate(divide='ignore',invalid='ignore'):t=(cap-o[2])/d[:,2]
                xy=o[:2]+t[:,None]*d[:,:2]
                hit=np.minimum(hit,np.where((t>1e-5)&(np.sum(xy*xy,axis=1)<=g.radius*g.radius),t,np.inf))
        best=np.minimum(best,hit)
    return best

def voxel_hits(origin,directions,centers):
    if not len(centers):return np.full(len(directions),np.inf)
    centers=np.asarray(centers)
    with np.errstate(divide='ignore',invalid='ignore'):
        a=(centers[None,:,:]-.025-origin)/directions[:,None,:]
        b=(centers[None,:,:]+.025-origin)/directions[:,None,:]
    low=np.max(np.minimum(a,b),axis=2);high=np.min(np.maximum(a,b),axis=2)
    return np.min(np.where((high>=low)&(low>1e-5),low,np.inf),axis=1)

def predict(node,q,points,occupied,allow_unknown=False,*,sensor_ids=None,
            measured_surfaces=None,deadline=float('inf'),grid_indices=None):
    if grid_indices is None:
        from path_seed_geometry import indices as grid_indices
    from time import monotonic
    qm=dict(zip(DEFAULT_JOINT_NAMES,q))
    transforms=[fk_transform(c,qm)@p['origin'] for p,c in zip(node.primitives,node.chains)]
    w,h,stride=240,335,8
    # Gazebo pinhole uses one focal length from horizontal FOV and square pixels.
    # Finite cloud coordinates are authoritative; nonfinite fusion reconstruction
    # uses a different fy and is not used for this finite-return prediction.
    focal=w/(2*np.tan(np.deg2rad(55)/2));cx=(w-1)/2;cy=(h-1)/2
    u,v=np.meshgrid(np.arange(0,w,stride),np.arange(0,h,stride))
    camera=np.column_stack(((u.ravel()-cx)/focal,(v.ravel()-cy)/focal,np.ones(u.size)))
    camera/=np.linalg.norm(camera,axis=1)[:,None]
    target_ids=set(grid_indices(np.asarray(points)).tolist());best=None
    for sid,chain in enumerate(node.sensors):
        if monotonic() >= deadline:break
        if sensor_ids is not None and sid not in sensor_ids:continue
        if allow_unknown and hasattr(node,"observation_key") and node.context_failures.get(node.observation_key(q,np.asarray(points),sid),0)>0:continue
        T=fk_transform(chain,qm);origin=T[:3,3];all_directions=camera@T[:3,:3].T
        possible=np.zeros(len(camera),bool)
        for point in points:
            rel=point-origin;along=all_directions@rel
            possible|=(along>0)&(np.linalg.norm(rel-along[:,None]*all_directions,axis=1)<.06)
        subset=np.flatnonzero(possible)
        if not len(subset):continue
        directions=all_directions[subset];optical=camera[subset]
        body_depths=body_hits(origin,directions,node.primitives,transforms)
        environment_depths=voxel_hits(origin,directions,occupied)
        surface_depths=np.full(len(directions),np.inf)
        if measured_surfaces is not None and sid in measured_surfaces:
            surface_depths=voxel_hits(origin,directions,measured_surfaces[sid])
        finite_depths=np.minimum(body_depths,surface_depths)
        finite_supported=np.isfinite(finite_depths)&(finite_depths<=environment_depths+.025)
        depths=np.minimum(body_depths,environment_depths)
        depths=np.where(finite_supported,np.minimum(depths,surface_depths),depths)
        opportunity=(~np.isfinite(depths))|(depths*optical[:,2]>.75)
        if allow_unknown:depths=np.where(opportunity,.75/optical[:,2],depths)
        axial=depths*optical[:,2]
        valid=np.isfinite(depths)&(axial>=.15)&(axial<=.75)
        if not allow_unknown:valid&=finite_supported
        starts=.15/optical[:,2];ends=depths-.025
        valid&=(ends-starts>=.05)
        # Both first known-body and measured-environment surfaces are finite
        # return hypotheses. Only actual measured segments update the map.
        rays=np.flatnonzero(valid)
        hits=set();finite_hits=set();target_rays=[]
        for ray in rays:
            distance=starts[ray]+np.arange(0,max(0,ends[ray]-starts[ray]),.025)
            xyz=origin+distance[:,None]*directions[ray]
            # Match actual sensor-only map: sample d<free_end, no appended endpoint.
            acquired=target_ids.intersection(grid_indices(xyz).tolist())
            if acquired:
                hits.update(acquired)
                finite=bool(not opportunity[ray] and finite_supported[ray])
                if finite:finite_hits.update(acquired)
                if opportunity[ray]:background='unmodeled_return_opportunity'
                elif finite and surface_depths[ray]<body_depths[ray]:background='recent_measured_surface'
                elif finite:background='known_body'
                else:background='measured_occupied_voxel'
                target_rays.append(dict(pixel=[int(u.ravel()[subset[ray]]),int(v.ravel()[subset[ray]])],
                    target_ids=sorted(acquired),body_return_m=None if opportunity[ray] else float(depths[ray]),
                    background=background,measured_return_expected=finite,
                    valid_start_m=float(starts[ray]),valid_end_m=float(ends[ray])))
        if hits and (best is None or (len(finite_hits),len(hits))>(len(best['finite_ids']),best['count'])):
            best=dict(count=len(hits),sensor=sid,rays=target_rays[:8],target_ids=sorted(hits),finite_ids=sorted(finite_hits))
    return best
