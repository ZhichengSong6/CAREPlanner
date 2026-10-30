#!/usr/bin/env python3
"""Fit an explicitly selected empty floor patch in base coordinates.

Residual statistics describe the recorded sample, never a future error bound.
This command writes a reviewable calibration proposal; activation is separate.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def fit_plane(points,threshold_m,minimum_inlier_fraction=.8,seed=20261027):
    points=np.asarray(points,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)<100 or not np.isfinite(points).all() or
        not np.isfinite(threshold_m) or threshold_m<=0):raise ValueError('invalid_calibration_points_or_threshold')
    rng=np.random.RandomState(seed);order=rng.permutation(len(points));train=points[order[::2]];holdout=points[order[1::2]]
    best=None
    for _ in range(256):
        a,b,c=train[rng.choice(len(train),3,replace=False)];normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-8:continue
        normal/=length;offset=-normal@a;mask=np.abs(train@normal+offset)<=threshold_m
        if best is None or int(mask.sum())>int(best.sum()):best=mask
    if best is None or best.mean()<minimum_inlier_fraction:raise ValueError('insufficient_plane_support')
    for _ in range(3):
        patch=train[best];center=patch.mean(axis=0);_,singular,v=np.linalg.svd(patch-center,full_matrices=False)
        if len(singular)<3 or singular[1]/np.sqrt(len(patch))<.02:raise ValueError('degenerate_floor_patch')
        normal=v[-1];normal*=1. if normal[2]>=0 else -1.;offset=float(-normal@center)
        best=np.abs(train@normal+offset)<=threshold_m
    residual=np.abs(holdout@normal+offset);inliers=residual<=threshold_m
    if best.mean()<minimum_inlier_fraction or inliers.mean()<minimum_inlier_fraction:raise ValueError('holdout_plane_support_failed')
    if normal[2]<np.cos(np.deg2rad(30.)):raise ValueError('plane_outside_configured_floor_orientation')
    return dict(normal=normal.tolist(),offset_m=offset,train_points=len(train),holdout_points=len(holdout),
        train_inlier_fraction=float(best.mean()),holdout_inlier_fraction=float(inliers.mean()),
        holdout_inlier_max_residual_m=float(residual[inliers].max()),holdout_all_max_residual_m=float(residual.max()),
        residuals_are_error_bound=False,threshold_m=threshold_m)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--points',type=Path,required=True)
    parser.add_argument('--threshold-m',type=float,required=True);parser.add_argument('--calibration-id',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--selected-empty-patch',action='store_true',required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(str(args.output))
    with np.load(args.points,allow_pickle=False) as data:
        if str(data['frame'])!='base_link':raise ValueError('calibration_points_must_be_in_base_frame')
        fit=fit_plane(data['points'],args.threshold_m)
    result=dict(schema=1,status='CALIBRATION_PROPOSAL',frame='base_link',calibration_id=args.calibration_id,
        selected_empty_patch=True,points_sha256=hashlib.sha256(args.points.read_bytes()).hexdigest(),fit=fit,
        plane_uncertainty_m=None,registration_bound_m=None,association_enabled=False,
        activation_requirements='Independent installation/registration uncertainty contract and source-time validation; sample residuals alone do not activate ground substitution.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
