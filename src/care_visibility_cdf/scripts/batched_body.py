"""Batch the existing primitive-center membership calculation over joint poses."""
import numpy as np

def _primitive_distances(node, poses, points, tree=None, dense=False, search_margin=0.):
    q=np.asarray(poses);n=len(q);frames={'base_link':(np.broadcast_to(np.eye(3),(n,3,3)),np.zeros((n,3)))}
    kin=node.task_model.kinematics
    def frame(name):
        if name not in frames:
            parent,origin,kind,axis,index=kin.frames[name];R,t=frame(parent)
            t=t+np.einsum('nij,j->ni',R,origin[:3,3]);R=R@origin[:3,:3]
            if kind in ('revolute','continuous'):
                outer,skew=kin.rotations[name];c=np.cos(q[:,index])[:,None,None];s=np.sin(q[:,index])[:,None,None]
                R=R@(c*np.eye(3)+(1-c)*outer+s*skew)
            frames[name]=(R,t)
        return frames[name]
    for p in node.primitives:
        R,t=frame(p['link']);origin=p['origin'];t=t+np.einsum('nij,j->ni',R,origin[:3,3]);R=R@origin[:3,:3];g=p['geometry'];kind=p['type']
        radius=np.linalg.norm(np.asarray(g.size)*.5) if kind=='box' else np.hypot(g.radius,g.length*.5) if kind=='cylinder' else g.radius
        if dense:
            ids = np.arange(len(points))
        elif tree is not None:
            near = tree.query_ball_point(t, radius + search_margin + 1e-8)
            ids = np.unique(np.concatenate([np.asarray(x, dtype=int) for x in near]))
        else:
            bound = radius + search_margin + 1e-8
            lower = t.min(axis=0) - bound
            upper = t.max(axis=0) + bound
            ids = np.flatnonzero(np.all((points >= lower) & (points <= upper), axis=1))
        if not len(ids):continue
        local=(points[ids][None,:,:]-t[:,None,:]) @ R
        if kind=='box':
            d=np.abs(local)-np.asarray(g.size)*.5;dist=np.linalg.norm(np.maximum(d,0.),axis=2)+np.minimum(np.max(d,axis=2),0.)
        elif kind=='cylinder':
            d=np.stack([np.linalg.norm(local[:,:,:2],axis=2)-g.radius,np.abs(local[:,:,2])-g.length*.5],axis=2);dist=np.linalg.norm(np.maximum(d,0.),axis=2)+np.minimum(np.max(d,axis=2),0.)
        else:dist=np.linalg.norm(local,axis=2)-g.radius
        yield ids, dist


def memberships(node, poses, grid=None):
    if grid is None:
        import path_seed_geometry as geometry
        points, tree = geometry.COORDS, geometry.TREE
    else:
        points, tree = grid, None
    result = [set() for _ in poses]
    for ids, distances in _primitive_distances(node, poses, points, tree):
        for members, row in zip(result, distances):
            members.update(ids[row <= 1e-8].tolist())
    return [np.asarray(sorted(ids),dtype=int) for ids in result]


def point_clearances(node, poses, points):
    """Minimum signed center clearance to the same primitives at each pose."""
    result = np.full((len(poses), len(points)), np.inf)
    for _, distances in _primitive_distances(node, poses, points, dense=True):
        np.minimum(result, distances, out=result)
    return result
