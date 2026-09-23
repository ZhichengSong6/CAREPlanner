"""Compare standalone R1 against the pinned original training definition.

No --checkpoint: random-weight architecture check only, NOT a trained-model test.
With --checkpoint: SHA-verified R1 value/q-gradient comparison, no ROS/Gazebo.
Both modes retain the training reference only as a test dependency, not runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from r1_visibility_model import (
    PrivateTailCDF, HierarchicalSensorView, load_r1_checkpoint,
    R1_PARAMETERS, R1_SHA256, SOURCE_COMMIT,
)
import reference_model_r012 as original


def value_gradient(model, x, q, output_column):
    q = q.detach().clone().requires_grad_(True)
    inputs = torch.cat((x, q.expand(len(x), -1)), dim=-1)
    value = model(inputs)[:, output_column].min()
    grad = torch.autograd.grad(value, q)[0]
    return value.detach(), grad.detach()


def run_checks(checkpoint_path=None, device='cpu'):
    torch.set_num_threads(2)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref_path = Path(original.__file__)
    raw = ref_path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob != '0b9dab1986114d4b95b33bb0fa01fd8e0c68e53b':
        raise RuntimeError('Original reference file has changed')

    checkpoint = None
    if checkpoint_path is not None:
        adapter, checkpoint = load_r1_checkpoint(checkpoint_path, device)
        full = adapter.full_model
    cases = []
    for seed in (0, 41, 260921):
        torch.manual_seed(seed)
        ref = original.build_model('R1')
        if checkpoint is None:
            full = PrivateTailCDF()
            full.load_state_dict(ref.state_dict(), strict=True)
            full = full.to(device).eval().requires_grad_(False)
            adapter = HierarchicalSensorView(full).eval()
        else:
            ref.load_state_dict(checkpoint['model_state'], strict=True)
        ref = ref.to(device).eval().requires_grad_(False)
        if full.parameter_count() != R1_PARAMETERS:
            raise RuntimeError('Parameter count mismatch')
        reference_shapes = {k: list(v.shape) for k, v in ref.state_dict().items()}
        new_shapes = {k: list(v.shape) for k, v in full.state_dict().items()}
        if reference_shapes != new_shapes:
            raise RuntimeError('State dictionary keys/shapes mismatch')

        gen = torch.Generator(device='cpu').manual_seed(seed + 177)
        x = (torch.rand((17, 3), generator=gen)*.6 + torch.tensor([-.3,-.3,.1])).to(device)
        q = (torch.rand((1, 7), generator=gen)*2 - 1).to(device)
        with torch.no_grad():
            inp = torch.cat((x,q.expand(len(x),-1)), dim=-1)
            a, b, eight = ref(inp), full(inp), adapter(inp)
            if tuple(a.shape) != (17,9) or tuple(eight.shape) != (17,8):
                raise RuntimeError('Output shape mismatch')
            value_error = (a-b).abs().max().item()
            view_error = (a[:,1:9]-eight).abs().max().item()
            fast_error = (eight-full.forward_sensors(inp)).abs().max().item()
        grad_error = 0.0
        grads = {}
        for sensor in range(8):
            vr, gr = value_gradient(ref,x,q,sensor+1)
            va, ga = value_gradient(adapter,x,q,sensor)
            if not torch.isfinite(ga).all():
                raise RuntimeError('Non-finite input gradient')
            grad_error = max(grad_error,(gr-ga).abs().max().item())
            grads['S'+str(sensor)] = {'value_abs_error': (vr-va).abs().item(),
                'gradient_max_abs_error':(gr-ga).abs().max().item(),
                'gradient_norm':ga.norm().item()}
        with torch.no_grad():
            q2=q.clone(); q2[:,0]+=.07
            shifted=adapter(torch.cat((x,q2.expand(len(x),-1)),dim=-1))
            recompute_delta=(shifted-eight).abs().max().item()
        if max(value_error,view_error,fast_error)>2e-6 or grad_error>3e-6:
            raise RuntimeError('Value/gradient equivalence failed')
        if recompute_delta<=1e-8 or any(p.requires_grad for p in full.parameters()):
            raise RuntimeError('q recomputation or parameter freeze check failed')
        cases.append({'seed':seed,'value_max_abs_error':value_error,
            'sensor_view_max_abs_error':view_error,'gradient_max_abs_error':grad_error,
            'q_recompute_max_delta':recompute_delta,'sensors':grads})
    return {'status':'PASS',
        'test_kind':'TRAINED_R1_EQUIVALENCE' if checkpoint else 'RANDOM_WEIGHT_ARCHITECTURE_ONLY',
        'trained_checkpoint_test':'PASS' if checkpoint else 'NOT_RUN',
        'checkpoint_sha256':R1_SHA256 if checkpoint else None,
        'device':str(device),'torch':torch.__version__,
        'source_commit':SOURCE_COMMIT,'reference_git_blob_sha':blob,
        'parameter_count':R1_PARAMETERS,'state_dict_tensor_count':len(reference_shapes),
        'cases':cases,
        'limits':'No FOV/LOS, ROS/Gazebo, VBC/GCDF, planner, or execution qualification.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    report=run_checks(a.checkpoint,a.device)
    text=json.dumps(report,indent=2)
    if a.output:
        out=Path(a.output).expanduser()
        out.parent.mkdir(parents=True,exist_ok=True)
        # Do not overwrite a previous result.
        with out.open('x',encoding='utf-8') as f:
            f.write(text+'\n')
    print(text)


if __name__=='__main__':
    main()
