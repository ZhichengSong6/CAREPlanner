"""P3 identity: P2 objective plus refreshed, actual-FOV neighborhood sign constraints."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.util

# P2's protocol intentionally remains a different module from this file.
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
P2_DIR = REPO / 'experiments/hierarchical9_p2_no_boundary_eikonal_v1'
_spec = importlib.util.spec_from_file_location('care_p2_protocol', P2_DIR / 'protocol.py')
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)
old = p2.old
FORMAT = 'care_h9_p3_neighborhood_sign_v1'
BASE_COMMIT = 'a84ef5329a919c44663c8cf995aef8218906d241'


@dataclass(frozen=True)
class NeighborhoodConfig:
    radius_min: float = .005
    radius_max: float = .05
    margin_per_rad: float = .25
    loss_weight: float = 5.0
    ambiguous_g_m: float = 1e-5
    microbatch: int = 256
    rng_tag: int = 93017


NEIGHBOR = asdict(NeighborhoodConfig())


def fingerprints():
    return {str(f.relative_to(REPO)): old.sha256(f) for f in sorted(HERE.iterdir())
            if f.suffix in ('.py', '.sh', '.sbatch')}


def load_references(root):
    root = Path(root).resolve()
    cache, c0, c1, hashes = p2.load_references(root)
    c2, h2 = p2.load_saved(root / 'P2/final.pt')
    p2.assert_p2(c2, c0, cache.identity, hashes, require_pilot=True)
    return cache, {'P0': c0, 'P1': c1, 'P2': c2}, {**hashes, 'P2': h2}


def make_args(control, output, mode):
    args = p2.make_args(control, output, mode)
    args.arm = 'P3'
    return args


def assert_p3(cp, control, cache_identity, references, require_pilot):
    mode = cp.get('mode')
    if mode not in ('smoke', 'pilot') or (require_pilot and mode != 'pilot'):
        raise ValueError('Wrong P3 mode; smoke is not a matched 2000-update comparison')
    steps = 2 if mode == 'smoke' else 2000
    expected = dict(format=FORMAT, arm='P3', completed=True, mode=mode,
        pilot_updates=steps, step=steps, total_updates=50000+steps, parent_updates=50000,
        parent_sha256=old.V1_SHA, out_dim=9, frozen_parameters=0,
        initialization='V1_weights_only_fresh_Adam',
        optimizer_policy='fresh_Adam_constant_lr_no_scheduler_no_clipping',
        boundary_weights=p2.WEIGHTS, neighborhood_config=NEIGHBOR,
        cache_manifest_sha256=cache_identity, reference_sha256=references,
        source_sha256=control['source_sha256'], p2_source_sha256=p2.fingerprints(),
        p3_source_sha256=fingerprints(), architecture=control['architecture'],
        output_layout=control['output_layout'])
    for key, value in expected.items():
        if cp.get(key) != value:
            raise ValueError('P3 identity/config mismatch: ' + key)
    if cp.get('args') != vars(make_args(control, Path(cp['args']['output']), mode)):
        raise ValueError('P3 old scientific settings changed')
    streams = cp.get('sample_stream_sha256_by_rank', [])
    extra = cp.get('neighborhood_stream_sha256_by_rank', [])
    if len(streams) != 4 or len(extra) != 4:
        raise ValueError('Missing rank-specific streams')
    for item in extra:
        if set(item) != {'proposal', 'labeled'} or any(len(v) != 64 for v in item.values()):
            raise ValueError('Invalid neighborhood stream digests')
    expected_match = 'MATCH' if mode == 'pilot' else 'NOT_COMPARABLE_2_VS_2000'
    if cp.get('pair_sample_streams') != expected_match:
        raise ValueError('Invalid global/anchor stream comparison claim')
    if mode == 'pilot' and streams != control['sample_stream_sha256_by_rank']:
        raise ValueError('P3 original uniform/anchor streams differ from P0/P1/P2')


def check_smoke(root, control, cache_identity, references):
    import json
    root = Path(root)
    cp, digest = p2.load_saved(root / 'P3_smoke/final.pt')
    assert_p3(cp, control, cache_identity, references, require_pilot=False)
    manifest = json.loads((root / 'evaluation_p3_smoke/manifest.json').read_text())
    if (cp['mode'] != 'smoke' or manifest.get('status') != 'COMPLETE'
            or manifest.get('mode') != 'smoke' or manifest.get('pilot_updates') != 2
            or manifest.get('models', {}).get('P3_smoke') != digest
            or manifest.get('p3_source_sha256') != fingerprints()):
        raise ValueError('Complete the current-source P3 smoke including evaluation first')
