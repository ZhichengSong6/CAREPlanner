"""Same-point model comparison + fresh FOV/normal/FD checks (no training)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import json_safe, write_json


def learned(model, x, qs, sensor):
    q = qs.detach().clone().float().requires_grad_(True)
    xx = x.float().reshape(1, 3).expand(len(q), -1)
    with torch.enable_grad():
        y = model(torch.cat((xx, q), 1))[:, sensor+1]
        grad = torch.autograd.grad(y.sum(), q)[0]
    return y.detach(), grad.detach()


def geometry_check(oracle, x, q0, qp, qm, normal, sensor, mask, lo, hi, stored, cfg):
    flags = []
    with torch.enable_grad():
        q = q0.clone().detach().reshape(1, 7).requires_grad_(True)
        h = oracle.planes(x, q, sensor)[0]
        face = int(h.detach().argmin())
        jac = torch.autograd.grad(h[face], q)[0][0].detach()
    h0 = h.detach()
    sorted_h = h0.sort().values
    gap = float(sorted_h[1]-sorted_h[0])
    active_jac = jac*mask
    norm = float(active_jac.norm())
    recomputed_n = active_jac / max(norm, 1e-30)
    cos = float(F.cosine_similarity(recomputed_n[None], normal[None], dim=1, eps=1e-12)[0])
    with torch.no_grad():
        hp = oracle.planes(x, qp[None], sensor)[0]
        hm = oracle.planes(x, qm[None], sensor)[0]
    g = np.asarray([float(h0.min()), float(hp.min()), float(hm.min())])
    margin_error = float(np.max(np.abs(g - np.asarray(stored))))
    tolerance = cfg["margin_tolerance_m"]
    if abs(g[0]) > tolerance:
        flags.append("ZERO_NOT_ON_ANALYTIC_BOUNDARY")
    if g[1] <= 0 or g[2] >= 0:
        flags.append("PAIRED_ANALYTIC_SIGNS_WRONG")
    if int(hp.argmin()) != face or int(hm.argmin()) != face:
        flags.append("ACTIVE_FACE_CHANGED")
    if margin_error > tolerance:
        flags.append("STORED_RECOMPUTED_MARGIN_MISMATCH")
    if norm < 1e-8 or cos < cfg["normal_cosine_min"]:
        flags.append("STORED_NORMAL_MISMATCH")
    inactive_norm = float((jac*(1-mask)).norm())
    if inactive_norm > cfg["inactive_jacobian_tolerance_m_per_rad"]:
        flags.append("ORACLE_MASK_MISMATCH")
    inactive_move = float(torch.max(torch.abs((qp-qm)*(1-mask))))
    if inactive_move > cfg["inactive_joint_tolerance_rad"]:
        flags.append("INACTIVE_JOINT_MOVED")
    in_limits = bool(((torch.stack((q0, qp, qm)) >= lo-1e-6) &
                      (torch.stack((q0, qp, qm)) <= hi+1e-6)).all())
    if not in_limits:
        flags.append("SOURCE_Q_OUT_OF_LIMITS")
    fds = []
    # Fixed active FACE derivatives, not nondifferentiable min across switching faces.
    for eps in cfg["finite_difference_steps_rad"]:
        directions = torch.eye(7, dtype=q0.dtype, device=q0.device)*eps
        with torch.no_grad():
            hplus = oracle.planes(x, q0[None]+directions, sensor)[:, face]
            hminus = oracle.planes(x, q0[None]-directions, sensor)[:, face]
        fd = (hplus-hminus)/(2*eps)
        err = float((fd-jac).abs().max())
        threshold = cfg["finite_difference_jacobian_atol"] + cfg["finite_difference_jacobian_rtol"]*float(jac.abs().max())
        fds.append({"epsilon_rad": eps, "max_abs_error_m_per_rad": err,
                    "within_tolerance": err <= threshold})
    if not all(v["within_tolerance"] for v in fds):
        flags.append("ORACLE_JACOBIAN_FD_REVIEW")
    try:
        upstream = oracle.verify_against_upstream(x, torch.stack((q0, qp, qm)))
    except RuntimeError as exc:
        upstream = {"status": "FAIL", "error": str(exc)}
        flags.append("UPSTREAM_FOV_DISAGREES")
    return {"flags": flags, "recomputed_g_m": g, "active_face": face,
            "plane_gap_m": gap, "normal_cosine": cos, "jacobian_norm_m_per_rad": norm,
            "inactive_jacobian_norm": inactive_norm, "inactive_movement_rad": inactive_move,
            "margin_max_abs_difference_m": margin_error, "finite_difference": fds,
            "upstream": upstream}, recomputed_n


def point_metrics(model, oracle, x, qs, target_normal, sensor, radius, mask, lo, hi, cfg):
    # qs order is zero, plus, minus. No labels are reused for generated training data.
    y, grad = learned(model, x, qs, sensor)
    yp, ym = y[1], y[2]
    slope = (yp-ym)/(2*radius)
    n0 = float(grad[0].norm())
    active_g = grad[2]*mask
    gnorm = float(active_g.norm())
    finite = bool(torch.isfinite(y).all() and torch.isfinite(grad).all())
    result = {
        "nonfinite": not finite, "f_zero": float(y[0]), "f_plus": float(yp), "f_minus": float(ym),
        "pair_slope": float(slope), "pair_slope_abs_error": float(abs(slope-1)),
        "pair_both_sign_correct": bool(yp > 0 and ym < 0),
        "side_mean_squared_error_rad2": float(((yp-radius)**2+(ym+radius)**2)/2),
        "midpoint_bias": float((yp+ym)/2), "boundary_grad_norm": n0,
        "boundary_cosine": float(F.cosine_similarity(grad[0:1], target_normal[None], eps=1e-8)[0]),
        "boundary_inactive_grad_norm": float((grad[0]*(1-mask)).norm()),
        "outside_grad_norm": gnorm,
    }
    # A fixed-length direction diagnostic, NOT another solver or a tuning sweep.
    qminus = qs[2]
    before = oracle.value(x, qminus, sensor)
    if not finite or gnorm <= cfg["small_gradient_norm"]:
        result["direction_probe"] = {"status": "NONFINITE" if not finite else "SMALL_GRADIENT",
                                     "before_g_m": before, "after_g_m": None, "delta_g_m": None}
    else:
        raw = qminus + cfg["diagnostic_ascent_step_rad"]*active_g/gnorm
        moved = torch.maximum(torch.minimum(raw, hi), lo)
        after = oracle.value(x, moved, sensor)
        result["direction_probe"] = {"status": "EVALUATED", "before_g_m": before,
            "after_g_m": after, "delta_g_m": after-before, "fov_pass": after >= 0,
            "clamped": not torch.equal(raw, moved), "step_norm_rad": float((moved-qminus).norm()),
            "note": "Generated diagnostic probe only; never written back as training data."}
    return result


def activation_audit(model, arrays, ids, device):
    """Neuron activity on the selected points only; never claim globally 'dead' ReLUs."""
    result = {}
    for sensor in range(8):
        chosen = ids[arrays["sensor"][ids] == sensor]
        if not len(chosen):
            continue
        x = np.repeat(arrays["x"][chosen], 3, axis=0)
        q = np.stack((arrays["q_zero"][chosen], arrays["q_plus"][chosen], arrays["q_minus"][chosen]), 1).reshape(-1, 7)
        inp = torch.tensor(np.concatenate((x, q), 1), device=device)
        entries, handles = {}, []
        for name, module in model.named_modules():
            relevant = name.startswith("early.") or name.startswith(f"sensor_tails.{sensor}.") or name.startswith(f"sensor_heads.{sensor}.")
            if isinstance(module, torch.nn.ReLU) and relevant:
                def hook(_module, _inp, output, name=name):
                    a = output.detach()
                    pos = a > 0
                    entries[name] = {"evaluated_elements": a.numel(), "positive_elements": int(pos.sum()),
                                     "positive_per_unit": pos.sum(0).cpu().tolist(), "rows": len(a)}
                handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad():
                model(inp)
        finally:
            for h in handles:
                h.remove()
        result[f"S{sensor}"] = entries
    return result


def run_boundary(out, arrays, ids, models, oracle, masks, lo, hi, cfg, rank):
    file = Path(out) / f"boundary.rank{rank}.jsonl"
    with file.open("x") as handle:
        for count, i in enumerate(ids):
            s = int(arrays["sensor"][i])
            radius = cfg["radii_rad"][int(arrays["radius_id"][i])]
            device = lo.device
            x = torch.tensor(arrays["x"][i], device=device)
            qs = torch.tensor(np.stack([arrays[k][i] for k in ("q_zero", "q_plus", "q_minus")]), device=device)
            normal = torch.tensor(arrays["normal"][i], device=device)
            audit, _ = geometry_check(oracle, x, *qs, normal, s, masks[s], lo, hi, arrays["stored_g_m"][i], cfg)
            row = {"case_id": int(i), "split": int(arrays["split"][i]), "sensor": s,
                   "radius_id": int(arrays["radius_id"][i]), "x_index": int(arrays["x_index"][i]),
                   "source_slot": int(arrays["source_slot"][i]), "shard": int(arrays["shard"][i]),
                   "geometry": audit, "models": {}}
            for name, model in models.items():
                row["models"][name] = point_metrics(model, oracle, x, qs, normal, s, radius, masks[s], lo, hi, cfg)
            handle.write(json.dumps(json_safe(row), allow_nan=False)+"\n")
            if (count+1) % 128 == 0:
                handle.flush()
                print(f"[boundary rank{rank}] {count+1}/{len(ids)} pairs", flush=True)
    activation = {name: activation_audit(model, arrays, ids, lo.device) for name, model in models.items()}
    write_json(Path(out) / f"activations.rank{rank}.json", activation)
