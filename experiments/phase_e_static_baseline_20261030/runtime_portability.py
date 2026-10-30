"""Adapt launcher paths and environment names; leave planning algorithms intact."""
import hashlib


def apply(runtime):
    changes = []
    paths = (
        'scripts/run_and_pack_c5_5_vbc_gcdf_regime.sh',
        'scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh',
        'scripts/run_phase_c4_4_verified_regime_smoke.sh',
        'scripts/run_phase_e_case001_r1_mainline.sh',
    )
    for relative in paths:
        path = runtime / relative
        original = path.read_bytes()
        text = original.decode()
        text = text.replace('REPO="${REPO:-/home/zhicheng/Project/CAREPlanner}"',
                            'REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"')
        text = text.replace('if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then',
                            'if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then\n  :\nelif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then')
        text = text.replace('export GPU_ENV=viscdf GPU_DEVICE=cuda NCDF_ENV=ncdf_l4c NCDF_DEVICE=cpu',
                            'export GPU_ENV="${GPU_ENV:-viscdf}" GPU_DEVICE=cuda NCDF_ENV="${NCDF_ENV:-ncdf_l4c}" NCDF_DEVICE=cpu')
        text = text.replace('${STOP_SHADOW_PYTHON:-/home/zhicheng/miniconda3/envs/viscdf/bin/python}',
                            '${STOP_SHADOW_PYTHON:-python3}')
        data = text.encode()
        if data != original:
            path.write_bytes(data)
            changes.append(dict(path=relative, original_sha256=hashlib.sha256(original).hexdigest(),
                                prepared_sha256=hashlib.sha256(data).hexdigest()))
    return changes
