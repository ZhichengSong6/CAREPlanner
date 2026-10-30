#!/usr/bin/env python3
"""Audit current CDF inputs/gradients against recorded rows; no training claims."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/care_collision_cdf/scripts'))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 import torch,numpy as np,pinocchio as pin
 from collision_cdf_model import CollisionCDF,_looks_like_state_dict
 torch.set_num_threads(1)
 path=ROOT/'src/care_collision_cdf/config/cdf_training_contract.json';contract=json.loads(path.read_text());checkpoint=ROOT/contract['checkpoint']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(checkpoint)==contract['checkpoint_sha256']
 net=CollisionCDF(str(checkpoint),device='cpu',activation=contract['runtime_input']['activation'])
 payload=torch.load(checkpoint,map_location='cpu');metadata=[] if _looks_like_state_dict(payload) else [str(k) for k,v in payload.items() if not _looks_like_state_dict(v)]
 urdfs=[ROOT/'src/arm_description/urdf'/f for f in ['Arm.urdf','Arm_with_self_filter_collision.urdf']]
 models=[pin.buildModelFromUrdf(str(p)) for p in urdfs];names=contract['runtime_input']['joint_order']
 assert all(list(m.names)[1:]==names for m in models)
 old3=ROOT/'outputs/phase_e_case003_geometry_20261012/report.json';d3=json.loads(old3.read_text())
 old14=ROOT/'outputs/phase_e_case010_014_execution_validation_20261012/case014_cdf_analysis.json';d14=json.loads(old14.read_text())
 inputs=[dict(case='003',point=d3['point'],q=next(s['q'] for s in d3['snapshots'] if s['pose']=='cdf_q_linearization'),logged=d3['logged_cdf'])]
 inputs += [dict(case='014',point=r['point'],q=d14['terminal_q0'],logged=r['logged_distance_model_units']) for r in d14['terminal_rows']]
 results=[]
 for row in inputs:
  p=torch.tensor([row['point']],dtype=torch.float32);q=torch.tensor([row['q']],dtype=torch.float32)
  d,g=net.pair_distance_and_gradient(p,q);fd=[]
  with torch.no_grad():
   for j in range(7):
    plus,minus=q.clone(),q.clone();plus[0,j]+=.001;minus[0,j]-=.001
    fd.append((net.model(torch.cat((p,plus),1)).item()-net.model(torch.cat((p,minus),1)).item())/.002)
  gradient_error=float(np.max(np.abs(g.numpy()[0]-fd)));distance_error=abs(d.item()-row['logged'])
  assert gradient_error<.002 and distance_error<1e-5,(row,gradient_error,distance_error)
  datas=[m.createData() for m in models]
  for m,data in zip(models,datas):pin.framesForwardKinematics(m,data,np.array(row['q']))
  links=[frame.name for frame in models[1].frames if models[0].existFrame(frame.name)]
  fk_error=max(float(np.max(np.abs(datas[0].oMf[models[0].getFrameId(link)].homogeneous-datas[1].oMf[models[1].getFrameId(link)].homogeneous))) for link in links)
  assert fk_error<1e-10
  results.append(dict(row,replayed_distance=d.item(),distance_error=distance_error,gradient_7d=g.numpy()[0].tolist(),finite_difference_7d=fd,max_gradient_error=gradient_error,current_fk_error=fk_error))
 report=dict(status='RUNTIME_CHECKS_PASS_TRAINING_CONTRACT_UNRESOLVED',contract=contract,architecture=net.architecture,checkpoint_metadata_keys=metadata,rows=results,sources={str(p.relative_to(ROOT)):sha(p) for p in [path,checkpoint,old3,old14,*urdfs,Path(__file__)]},limitations=['No original training geometry/labels/normalization manifest exists in these inputs.','5 mm has not been converted into a neural error bound or a runtime safety margin.','Matching recorded signed CDF and its derivatives does not prove model accuracy against training ground truth.'])
 with a.output.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(dict(status=report['status'],rows=len(results),max_gradient_error=max(r['max_gradient_error'] for r in results))))
if __name__=='__main__':main()
