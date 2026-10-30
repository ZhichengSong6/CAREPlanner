#!/usr/bin/env python3
"""Independently check saved stall-QP evidence, or requery its nonlinear CDF."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def matrix(m):
    a=np.zeros(m['shape'])
    for i,j,v in m['coo']: a[int(i),int(j)]+=v
    return a


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-dir',type=Path,required=True)
    ap.add_argument('--cdf',action='store_true')
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--frontier-dir',type=Path)
    args=ap.parse_args();out=args.evidence_dir
    read=lambda name:json.loads((out/name).read_text())
    report=read('report.json');rows=read('selected_rows.json')
    result={}
    if args.cdf:
        import torch
        sys.path.insert(0,str(ROOT/'src/care_collision_cdf/scripts'))
        from collision_cdf_model import CollisionCDF
        torch.set_num_threads(2)
        model=CollisionCDF(str(ROOT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'),device=args.device,activation='gelu')
        points=torch.tensor([json.loads(r['point']) for r in rows],dtype=torch.float32)
        qlin=torch.tensor([json.loads(r['q_linearization']) for r in rows],dtype=torch.float32)
        actual,grad=model.pair_distance_and_gradient(points,qlin)
        result['device']=args.device
        result['recorded_distance_requery_error_inf']=float(np.max(np.abs(actual.detach().cpu().numpy()-[float(r['distance']) for r in rows])))
        result['recorded_gradient_requery_error_inf']=float(np.max(np.abs(grad.detach().cpu().numpy()-[json.loads(r['gradient']) for r in rows])))
        print('requery baseline',json.dumps(result),flush=True)
        # CPU/GPU FP32 accumulation is not bit-identical. Require agreement
        # within the production QP absolute tolerance, and report actual error.
        assert result['recorded_distance_requery_error_inf']<1e-5
        assert result['recorded_gradient_requery_error_inf']<2e-5
        result['variants']={}
        variants=([(f'attempt_{i}',None) for i in range(1,5)] if args.frontier_dir else
                  [('actual',None),('witness_cap_012',.12),('witness_cap_002',.02),('stronger_frontier',None)])
        for name,cap in variants:
            solution=json.loads(((args.frontier_dir or out)/(name+'_solution.json')).read_text())
            x=solution['x']
            q=np.array(x[:140]).reshape(20,7)
            distance,_=model.pair_distance_and_gradient(points,torch.tensor(np.array([q[int(r['timestep'])-1] for r in rows]),dtype=torch.float32))
            dv=distance.detach().cpu().numpy()
            margins=np.array([r['effective_margin'] if cap is None else min(cap,r['effective_margin']) for r in rows])
            result['variants'][name]=dict(min_nonlinear_cdf=float(dv.min()),
                min_nonlinear_margin_residual=float(np.min(dv-margins)),
                violated_rows_over_1e5=int(np.sum(dv-margins < -1e-5)))
        filename='nonlinear_cdf_check.json'
    else:
        for name in ('actual','witness_cap_012','witness_cap_002','stronger_frontier'):
            p=read(name+'_problem.json');s=read(name+'_solution.json');x=np.array(s['x'])
            P,A,G=[matrix(p[k]) for k in ('P','A','G')]
            station=P@x+p['c']+A.T@np.array(s['y'])+G.T@(np.array(s['z_u'])-s['z_l'])+np.array(s['z_bu'])-s['z_bl']
            primal=max(np.max(np.abs(A@x-p['b'])),np.max(np.array(p['h_l'])-G@x),np.max(G@x-p['h_u']),np.max(np.array(p['x_l'])-x),np.max(x-p['x_u']))
            assert primal<1e-8
            assert np.max(np.abs(station))<1e-3
            result[name]=dict(primal_residual=float(primal),stationarity_inf=float(np.max(np.abs(station))))
        active=report['active_rows']
        assert len(active)==1 and active[0]['k']==2
        assert np.max(np.abs(np.array(active[0]['point'])-[-.15,0,.9]))<1e-6
        assert abs(active[0]['margin']-.22)<1e-12
        g=np.array(active[0]['gradient']);delta=np.array(report['frontier_step'])
        result['frontier_vs_blocker_gradient_cosine']=float(delta@g/(np.linalg.norm(delta)*np.linalg.norm(g)))
        # Exercise the actual generated ROS byte-array wire representation.
        from care_confidence_map.srv import QueryConfidenceResponse
        transport=[]
        for flags in ([1,1,1,1],[0,0,0,0],[1,0,1,0]):
            source=QueryConfidenceResponse(confidence=[0.]*4,current_visibility=[0.]*4,inside_map=flags)
            buf=io.BytesIO();source.serialize(buf)
            decoded=QueryConfidenceResponse().deserialize(buf.getvalue())
            wrong=np.asarray(decoded.inside_map,dtype=bool)
            correct=np.frombuffer(decoded.inside_map,dtype=np.uint8).astype(bool)
            assert wrong.shape==() and np.count_nonzero(wrong)==1
            assert correct.tolist()==list(map(bool,flags))
            transport.append(dict(sent=flags,decoded_type=type(decoded.inside_map).__name__,
                current_inside_count=int(np.count_nonzero(wrong)),actual_inside_count=int(np.count_nonzero(correct))))
        result['inside_map_transport_reproduction']=transport
        filename='independent_checks.json'
    with ((args.frontier_dir or out)/filename).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
