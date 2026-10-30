#!/usr/bin/env python3
"""Build development-only interval supervision from preserved FOV brackets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/care_visibility_cdf/scripts'))
from visibility_boundary_supervision import boundary_intervals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=ROOT/'outputs/phase_e_candidate_network_20261019')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    files = [args.source/'boundary_generalization.json', args.source/'label_subset.npz',
             args.source/'training_label_equivalence.json', args.source/'training_helper_provenance.json']
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    rows = json.loads(files[0].read_text())['rows']
    dataset, statistics = [], []
    with np.load(files[1]) as source:
        lookup = {int(x): k for k, x in enumerate(source['indices'])}
        for row in rows:
            index, sid = row['index'], row['sensor_id']
            assert np.array_equal(source['x'][lookup[index]], source['all_x'][index])
            qminus, qplus, root = [np.asarray(row[k], dtype=float) for k in ('minus_q', 'plus_q', 'root_q')]
            for q in (qminus, qplus, root):
                assert np.all(q >= source['q_min']-1e-8) and np.all(q <= source['q_max']+1e-8)
            bound = boundary_intervals(qminus, qplus, root, row['minus_margin_m'],
                                       row['plus_margin_m'], source['sensor_chain_masks'][sid])
            statistics.append(dict(index=index, sensor_id=sid,
                bracket_width_rad=bound['bracket_width_rad'], original_label_jump_rad=row['target_jump_rad'],
                old_jump_exceeds_interval_pair_max=row['target_jump_rad'] > 2.*bound['bracket_width_rad']+1e-8))
            for name, q in [('minus', qminus), ('plus', qplus), ('root', root)]:
                lo, hi = bound[name]
                dataset.append(dict(point_index=index, sensor_id=sid,
                    x=source['x'][lookup[index]].tolist(), q=q.tolist(), lower_rad=lo, upper_rad=hi,
                    origin=name, development_split='validation' if index % 3 == 0 else 'train',
                    root_residual_m=row['boundary_residual_m'], bracket_margins_m=[row['minus_margin_m'], row['plus_margin_m']]))
    (args.output_dir/'interval_samples.json').write_text(json.dumps(dataset, indent=2)+'\n')
    report = dict(status='PASS_INTERVAL_CONSTRUCTION', bracket_probes=len(rows), samples=len(dataset),
        distinct_spatial_points=len({r['point_index'] for r in dataset}),
        distinct_point_head_pairs=len({(r['point_index'], r['sensor_id']) for r in dataset}),
        independence_note='Repeated brackets and three samples per bracket are correlated; they are not independent statistical samples.',
        exact_global_distance_labels=False, checkpoint_changed=False, training_run=False,
        scope='conservative analytic FOV field only, active-joint radians; not nominal FOV, LOS, map evidence or motion safety',
        proof='Opposite continuous FOV signs imply a boundary on the segment; the larger endpoint distance bounds query-to-segment-boundary distance.',
        qualification='Uses preserved independently checked numerical sign brackets; no claim about physical sensing or global closest-boundary solutions.',
        split='all samples/heads at the same spatial point stay in one development split',
        original_label_pairs_exceeding_interval_bound=sum(x['old_jump_exceeds_interval_pair_max'] for x in statistics),
        provenance=provenance, rows=statistics)
    (args.output_dir/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','provenance']}), flush=True)


if __name__ == '__main__':
    main()
