"""Independent sampled whole-body entry check using raw ToF integration.

Uses URDF solids and every recorded measured joint state. No initial environment FREE is accepted; initial robot-occupied geometry
is the only start-volume assumption; continuous intervals are not certified here.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/care_visibility_cdf/scripts'))
from validate_visibility_oracle import DEFAULT_JOINT_NAMES, URDF, find_chain_joints, fk_transform
from check_visibility_self_occlusion import load_collision_primitives

LO=np.array([-.95,-.95,0.]);RES=.05;SHAPE=np.floor((np.array([.95,.95,1.15])-LO)/RES).astype(int)+1;N=int(np.prod(SHAPE))
COORDS=np.stack(np.meshgrid(*[LO[j]+RES*np.arange(SHAPE[j]) for j in range(3)],indexing='ij'),axis=-1).reshape(-1,3)
TREE=cKDTree(COORDS)
def array(msg):
 types={1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',8:'<f8'}
 dt=np.dtype(dict(names=[f.name for f in msg.fields],formats=[types[f.datatype] for f in msg.fields],offsets=[f.offset for f in msg.fields],itemsize=msg.point_step))
 assert not msg.is_bigendian and msg.row_step==msg.width*msg.point_step
 return np.frombuffer(msg.data,dtype=dt,count=msg.width*msg.height)
def indices(points):
 grid=np.floor((points-LO)/RES+.5).astype(np.int64);valid=np.all((grid>=0)&(grid<SHAPE),axis=1)
 g=grid[valid];return g[:,0]*SHAPE[1]*SHAPE[2]+g[:,1]*SHAPE[2]+g[:,2]
def cells(runtime,q,margin):
 frames={};qmap=dict(zip(DEFAULT_JOINT_NAMES,q));all_ids=[]
 for p,chain in zip(runtime.primitives,runtime.chains):
  if p['link'] not in frames:frames[p['link']]=fk_transform(chain,qmap)
  T=frames[p['link']]@p['origin'];g=p['geometry']
  radius=(np.linalg.norm(np.asarray(g.size)*.5) if p['type']=='box' else np.hypot(g.radius,g.length*.5) if p['type']=='cylinder' else g.radius)+margin
  ids=np.asarray(TREE.query_ball_point(T[:3,3],radius+1e-8),dtype=int)
  if not len(ids):continue
  local=(COORDS[ids]-T[:3,3])@T[:3,:3]
  if p['type']=='box':
   d=np.abs(local)-np.asarray(g.size)*.5;dist=np.linalg.norm(np.maximum(d,0.),axis=1)+np.minimum(np.max(d,axis=1),0.)
  elif p['type']=='cylinder':
   d=np.stack([np.linalg.norm(local[:,:2],axis=1)-g.radius,np.abs(local[:,2])-g.length*.5],axis=1);dist=np.linalg.norm(np.maximum(d,0.),axis=1)+np.minimum(np.max(d,axis=1),0.)
  else:dist=np.linalg.norm(local,axis=1)-g.radius
  all_ids.extend(ids[dist<=margin+1e-8])
 return np.unique(all_ids).astype(int)
def integrate(a):
 origin=np.stack([a['origin_'+s] for s in 'xyz'],axis=1).astype(float);end=np.stack([a[s] for s in 'xyz'],axis=1).astype(float)
 delta=end-origin;length=np.linalg.norm(delta,axis=1)
 valid=np.isfinite(origin).all(axis=1)&np.isfinite(end).all(axis=1)&(length>=.05)&(length<=1.5)&(a['sensor_id']>=0)&(a['sensor_id']<8)
 origin=origin[valid];end=end[valid];length=length[valid];hit=a['hit'][valid]!=0;unit=(end-origin)/length[:,None]
 free_end=np.where(hit,np.maximum(0.,length-.025),length);mask=np.zeros(N,dtype=bool)
 for d in np.arange(0.,1.50001,.025):
  active=d<free_end
  if active.any():mask[indices(origin[active]+unit[active]*d)]=True
 mask[indices(end[~hit])]=True
 occupied=indices(end[hit]);mask[occupied]=False
 return mask,occupied
