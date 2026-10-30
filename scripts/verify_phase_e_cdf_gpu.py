#!/usr/bin/env python3
"""Six real GPU CQ01 requests against the same-device original model loader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--study',type=Path,required=True)
    a=p.parse_args(); study=a.study.resolve(); rt=study/'runtime'; out=study/'gpu_verification'
    out.mkdir(exist_ok=False)
    sys.path[:0]=[str(rt/'scripts'),str(rt/'src/care_collision_cdf/scripts')]
    from collision_cdf_model import CollisionCDF
    from cdf_input_audit import domain_summary, sha_file
    from analyze_cdf_input_audit import analyze
    from verify_cdf_worker import verify
    def save(n,v):
        with (out/n).open('x') as f: json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
    binding=json.loads((study/'old_model_binding.json').read_text())
    assert binding['model']=='old' and binding['activation']=='gelu'
    assert sha_file(binding['checkpoint'])==binding['checkpoint_sha256']
    assert torch.cuda.is_available(), 'Real CUDA required'
    save('environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
         binding=binding,started_wall_s=time.time(),pid=os.getpid()))
    queries=json.loads((study.parents[1]/'outputs/phase_e_cdf_qualification_20261023/bundle/queries.json').read_text())
    frozen=np.array([q['point_m']+q['q_rad'] for q in queries],dtype=np.float32)
    maximum=np.resize(frozen,(8000,10)).copy();maximum[-1,2]=1.05
    sock=Path('/tmp')/f'care_cdf_gpu_check_{os.getpid()}.sock';assert not sock.exists()
    args=[sys.executable,'-B',str(rt/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py'),
        '--checkpoint',binding['checkpoint'],'--activation','gelu','--checkpoint-key','latest',
        '--expected-checkpoint-sha256',binding['checkpoint_sha256'],'--device','cuda',
        '--socket',str(sock),'--model-attestation',str(out/'attestation.json'),'--domain-audit',str(out/'audit.jsonl')]
    records=[]; proc=None; log=(out/'worker.log').open('x'); t0=time.monotonic()
    try:
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        while not sock.exists():
            if proc.poll() is not None: raise RuntimeError('Worker exited: '+str(proc.returncode))
            if time.monotonic()-t0>20: raise TimeoutError('GPU readiness >20s')
            time.sleep(.025)
        readiness=verify(out/'attestation.json',out/'audit.jsonl',binding['checkpoint_sha256'],'gelu',sock)
        readiness['startup_wall_s']=time.monotonic()-t0;save('readiness.json',readiness)
        model=json.loads((out/'attestation.json').read_text());assert model['selected_checkpoint']=='iteration:39990'
        direct=CollisionCDF(binding['checkpoint'],device='cuda',activation='gelu',checkpoint_key='latest')
        req=struct.Struct('<4sI');resp=struct.Struct('<4sI4d')
        def recv(conn,n):
            data=bytearray()
            while len(data)<n:
                b=conn.recv(n-len(data))
                if not b:raise EOFError('Partial GPU response')
                data.extend(b)
            return data
        with socket.socket(socket.AF_UNIX) as conn:
            conn.settimeout(30);conn.connect(str(sock))
            for repeat in (1,2,3):
                for label,rows in [('frozen1878',frozen),('maximum8000',maximum)]:
                    x=torch.tensor(rows,device='cuda');torch.cuda.synchronize();start=time.perf_counter()
                    d,g=direct.pair_distance_and_gradient(x[:,:3].contiguous(),x[:,3:].contiguous())
                    torch.cuda.synchronize();direct_ms=(time.perf_counter()-start)*1000
                    d=d.detach().cpu().numpy();g=g.detach().cpu().numpy()
                    payload=rows.astype('<f4').tobytes();start=time.perf_counter();conn.sendall(req.pack(b'CQ01',len(rows))+payload)
                    magic,count,*timing=resp.unpack(recv(conn,resp.size));assert magic==b'CR01' and count==len(rows)
                    got_d=np.frombuffer(recv(conn,count*4),dtype='<f4').copy()
                    got_g=np.frombuffer(recv(conn,count*7*4),dtype='<f4').reshape(-1,7).copy()
                    roundtrip=(time.perf_counter()-start)*1000
                    np.testing.assert_allclose(got_d,d,atol=1e-6,rtol=1e-5)
                    np.testing.assert_allclose(got_g,g,atol=1e-6,rtol=1e-5)
                    records.append(dict(repeat=repeat,batch=label,pairs=count,payload_sha256=hashlib.sha256(payload).hexdigest(),
                        distance_max_error=float(np.max(np.abs(got_d-d))),gradient_max_error=float(np.max(np.abs(got_g-g))),
                        direct_gpu_ms=direct_ms,worker_timing_ms=timing,ipc_roundtrip_ms=roundtrip))
        proc.send_signal(signal.SIGTERM);proc.wait(timeout=10);assert proc.returncode==0
        report=analyze(out/'audit.jsonl',out/'attestation.json')
        assert report['requests']==6 and report['pairs']==29634
        assert report['outside_tolerance_rows']==3*sum(domain_summary(x)['outside_tolerance_rows'] for x in (frozen,maximum))
        received=[r for r in map(json.loads,(out/'audit.jsonl').read_text().splitlines()) if r['event']=='request']
        assert [r['payload_sha256'] for r in received]==[r['payload_sha256'] for r in records]
        save('report.json',dict(status='PASS_REAL_GPU_SAME_DEVICE_COMPARISON',comparisons=records,domain=report,
             attestation_sha256=sha_file(out/'attestation.json'),readiness=readiness,
             limits='Six correctness requests with injected OOD probes; not closed-loop timing or real-task domain evidence.'))
        print('GPU_VERIFICATION_PASS',json.dumps(report),flush=True)
    except BaseException as exc:
        save('failure.json',dict(error=repr(exc),comparisons=records,traceback=traceback.format_exc()));raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close()

if __name__=='__main__':main()
