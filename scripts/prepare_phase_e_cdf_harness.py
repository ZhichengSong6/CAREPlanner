#!/usr/bin/env python3
"""Prepare a separate, attested model-comparison runtime; never launch ROS."""
import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT/'outputs/phase_e_cdf_domain_prereg_20261023_v5'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    out = ap.parse_args().output_dir.resolve()
    for p in (out, *out.parents):
        if (p/'artifact_sha256.json').exists():
            raise ValueError('Output under a sealed study')
    out.mkdir(parents=True, exist_ok=False)
    lock = json.loads((PREREG/'runtime_lock.json').read_text())
    parent = ROOT/lock['source']
    for name, digest in lock['files'].items():
        assert sha(parent/name) == digest, name
    save(out/'parent_runtime_lock.json', lock)
    save(out/'root_before.json', {str(p.relative_to(ROOT)): sha(p)
        for base in ('src', 'scripts', 'devel/lib') for p in (ROOT/base).rglob('*')
        if p.is_file() and not p.is_symlink() and
        (p.suffix in ('.py', '.cpp', '.hpp', '.h', '.sh', '.launch', '.yaml', '.urdf', '.pt', '.so')
         or (p.read_bytes()[:4] == b'\x7fELF' if base == 'devel/lib' else False))})
    for name in ('NEXT_SESSION.md', 'outputs/README.md'):
        shutil.copy2(ROOT/name, out/(Path(name).stem+'_before.md'))
    rt = out/'runtime'
    rt.mkdir()
    for folder in ('src', 'scripts', 'devel'):
        subprocess.run(['cp', '-a', '--reflink=auto', str(parent/folder), str(rt/folder)], check=True)
        print('COPIED', folder, flush=True)
    for folder in ('phase_e_goal_sampling', 'phase_e_visibility_order_20261017', 'phase_e_case002_gcdf_motion_fix_20261015'):
        dest = rt/'outputs'/folder
        dest.parent.mkdir(exist_ok=True)
        subprocess.run(['cp', '-a', '--reflink=auto', str(parent/'outputs'/folder), str(dest)], check=True)
    shutil.copytree(parent.parent/'calibration_final', out/'calibration_final')
    relocated = []
    for base, dirs, files in os.walk(rt, followlinks=False):
        for name in dirs+files:
            p = Path(base)/name
            if p.is_symlink():
                old = os.readlink(p)
                new = old.replace(str(parent), str(rt))
                if old != new:
                    p.unlink(); p.symlink_to(new); relocated.append(str(p.relative_to(rt)))
            elif p.is_file() and p.stat().st_size < 16*1024**2:
                old = p.read_bytes()
                if b'\0' not in old:
                    new = old.replace(str(parent).encode(), str(rt).encode())
                    if old != new:
                        p.write_bytes(new); relocated.append(str(p.relative_to(rt)))
    save(out/'relocated_paths.json', relocated)
    changed = []

    def edit(name, old, new):
        p = rt/name
        text = p.read_text()
        assert text.count(old) == 1, (name, old[:80], text.count(old))
        p.write_text(text.replace(old, new))
        if name not in changed:
            changed.append(name)

    scripts = 'src/care_collision_cdf/scripts/'
    worker = scripts+'collision_cdf_gpu_worker.py'
    edit(worker, 'from collision_cdf_model import CollisionCDF  # noqa: E402',
         'from collision_cdf_model import CollisionCDF  # noqa: E402\nfrom cdf_input_audit import InputAudit, attest_model, sha_file')
    edit(worker, 'p.add_argument("--activation", choices=("gelu", "relu"), default="gelu")',
         '''p.add_argument("--activation", choices=("gelu", "relu"), required=True)
    p.add_argument("--expected-checkpoint-sha256", required=True)
    p.add_argument("--model-attestation", required=True)
    p.add_argument("--domain-audit", required=True)
    p.add_argument("--diagnostic-cpu", action="store_true")''')
    edit(worker, 'if not str(args.device).startswith("cuda"):',
         'if not str(args.device).startswith("cuda") and not (args.device == "cpu" and args.diagnostic_cpu):')
    edit(worker, 'if not torch.cuda.is_available():',
         'if str(args.device).startswith("cuda") and not torch.cuda.is_available():')
    edit(worker, '        self.stop = False', '        self.stop = False\n        self.active_connection = None')
    edit(worker, '        server.settimeout(0.5)', '        server.settimeout(0.05)')
    edit(worker, '                with conn:', '                self.active_connection = conn\n                with conn:')
    edit(worker, '    def request_stop(self, *_args):\n        self.stop = True', '''    def request_stop(self, *_args):
        self.stop = True
        if self.active_connection is not None:
            try:
                self.active_connection.shutdown(socket.SHUT_RD)
            except OSError:
                pass''')
    edit(worker, '        self.cdf = CollisionCDF(', '''        for path in (args.model_attestation, args.domain_audit):
            resolved = Path(path).resolve()
            if resolved.exists():
                raise ValueError("Refusing to overwrite audit artifact: " + str(resolved))
            if any((p / "artifact_sha256.json").exists() for p in resolved.parents):
                raise ValueError("Refusing to write inside a sealed study")
        before_sha = sha_file(args.checkpoint)
        if before_sha != args.expected_checkpoint_sha256:
            raise ValueError("Checkpoint SHA256 differs from declaration")
        self.cdf = CollisionCDF(''')
    edit(worker, '        b = max(1, min(int(args.warmup_pairs), int(args.max_pairs)))', '''        attest_model(args, self.cdf, before_sha, __file__, SCRIPT_DIR / "collision_cdf_model.py")
        b = max(1, min(int(args.warmup_pairs), int(args.max_pairs)))''')
    p = rt/worker
    text = p.read_text().replace('torch.cuda.synchronize(self.device)', 'self._synchronize()')
    text = text.replace('f"gpu={torch.cuda.get_device_name(self.device)} "',
                        'f"device={self.device} name={torch.cuda.get_device_name(self.device) if self.device.type == \'cuda\' else \'CPU_DIAGNOSTIC\'} "')
    p.write_text(text)
    edit(worker, '        print(\n            "[C5.2g GPU worker] READY "',
         '        self.audit = InputAudit(args.domain_audit, args.model_attestation)\n        print(\n            "[C5.2g GPU worker] READY "')
    edit(worker, '    def _handle_request(self, conn, pair_count, payload):', '''    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _handle_request(self, conn, pair_count, payload):''')
    edit(worker, '        points_cpu = torch.from_numpy(',
         '        input_audit_ms = self.audit.begin(rows, payload)\n        points_cpu = torch.from_numpy(')
    edit(worker, '        header = RESP_HEADER.pack(', '''        self.audit.finish(distance, gradient, pair_count, input_audit_ms, worker_total_ms)
        worker_total_ms = (time.perf_counter() - worker_t0) * 1000.0
        header = RESP_HEADER.pack(''')
    edit(worker, '                        payload = recv_exact(\n', '''                        if not 0 < pair_count <= self.args.max_pairs:
                            raise ValueError("Invalid pair count before receiving payload")
                        payload = recv_exact(
''')
    edit(worker, '                        self._handle_request(\n                            conn, int(pair_count), payload\n                        )', '''                        try:
                            self._handle_request(conn, int(pair_count), payload)
                        except Exception as exc:
                            self.audit.error(exc)
                            raise''')
    edit(worker, '    worker.serve()', '''    try:
        worker.serve()
    except Exception as exc:
        if worker.audit.failed == 0:
            worker.audit.error(exc)
        raise
    finally:
        worker.audit.close()''')
    shutil.copy2(ROOT/'scripts/cdf_input_audit.py', rt/scripts/'cdf_input_audit.py')
    for name in ('cdf_worker_launch.py', 'verify_cdf_worker.py'):
        shutil.copy2(ROOT/'scripts'/name, rt/'scripts'/name)
    mainline = 'scripts/run_phase_e_case001_r1_mainline.sh'
    edit(mainline, 'export CHECKPOINT="${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt"',
         '''export CHECKPOINT="${CHECKPOINT:?explicit model checkpoint required}"
export CDF_ACTIVATION="${CDF_ACTIVATION:?explicit activation required}"
export CDF_EXPECTED_SHA256="${CDF_EXPECTED_SHA256:?explicit checkpoint SHA256 required}"
export CDF_CHECKPOINT_KEY="${CDF_CHECKPOINT_KEY:-latest}"''')
    local = 'scripts/run_and_pack_phase_c5_4_local_sparse_scp.sh'
    edit(local, 'CHECKPOINT="${CHECKPOINT:-${REPO}/src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt}"',
         '''export CHECKPOINT="${CHECKPOINT:?explicit model checkpoint required}"
export CDF_ACTIVATION="${CDF_ACTIVATION:?explicit activation required}"
export CDF_EXPECTED_SHA256="${CDF_EXPECTED_SHA256:?explicit checkpoint SHA256 required}"
export CDF_CHECKPOINT_KEY="${CDF_CHECKPOINT_KEY:-latest}"''')
    edit(local, 'SUMMARY_JSON="${ROOT_OUT}/c5_4_local_sparse_scp_summary.json"', '''SUMMARY_JSON="${ROOT_OUT}/c5_4_local_sparse_scp_summary.json"
export CDF_MODEL_ATTESTATION="${ROOT_OUT}/cdf_model_attestation.json"
export CDF_INPUT_AUDIT="${ROOT_OUT}/cdf_input_domain.jsonl"
export GPU_DEVICE GPU_SOCKET''')
    start = (rt/local).read_text().index('  exec python -u src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py')
    end = (rt/local).read_text().index('\n" >"${ROOT_LOG}/gpu_worker.log"', start)
    original = (rt/local).read_text()[start:end]
    edit(local, original, '  exec python -u scripts/cdf_worker_launch.py')
    edit(local, 'PIDS+=("$!")', 'CDF_GPU_WORKER_PID="$!"\nPIDS+=("$!")')
    edit(local, '    kill -TERM -- "-${pid}" 2>/dev/null || true\n    sleep 0.20', '''    kill -TERM -- "-${pid}" 2>/dev/null || true
    if [[ "${CDF_GPU_WORKER_PID:-}" == "${pid}" ]]; then
      for _ in $(seq 1 200); do
        if ! kill -0 "${pid}" 2>/dev/null; then break; fi
        sleep 0.01
      done
    else
      sleep 0.20
    fi''')
    edit(local, '# C5.4-specific ROS diagnostics are recorded by the common runner itself.', '''python3 scripts/verify_cdf_worker.py \\
  --attestation "${CDF_MODEL_ATTESTATION}" --audit "${CDF_INPUT_AUDIT}" \\
  --expected-sha "${CDF_EXPECTED_SHA256}" --activation "${CDF_ACTIVATION}" --socket "${GPU_SOCKET}" \\
  > "${ROOT_OUT}/cdf_worker_readiness.json"

# C5.4-specific ROS diagnostics are recorded by the common runner itself.''')
    edit(local, 'cdf_activation=gelu', '''cdf_activation=${CDF_ACTIVATION}
cdf_checkpoint_sha256=${CDF_EXPECTED_SHA256}
cdf_checkpoint_key=${CDF_CHECKPOINT_KEY}
cdf_model_attestation=${CDF_MODEL_ATTESTATION}
cdf_input_audit=${CDF_INPUT_AUDIT}''')
    edit(local, '  # Keep full key logs. Raw high-rate CSVs', '''  for f in cdf_model_attestation.json cdf_input_domain.jsonl cdf_worker_readiness.json; do
    if [[ -f "${ROOT_OUT}/${f}" ]]; then cp -f "${ROOT_OUT}/${f}" "${stage}/root/"; fi
  done

  # Keep full key logs. Raw high-rate CSVs''')
    # Stage only a bound single-case runner, never the old twelve-window supervisor.
    target = rt/'outputs/cdf_model_ab'
    target.mkdir()
    runner = parent/'outputs/rebase_task_ab/run_case.py'
    text = runner.read_text().replace('GCDF = REPO / "src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt"', '''MODEL_BINDING_PATH = Path(os.environ["CARE_CDF_MODEL_BINDING"]).resolve()
MODEL_BINDING = json.loads(MODEL_BINDING_PATH.read_text())
GCDF = Path(MODEL_BINDING["checkpoint"]).resolve()
if MODEL_BINDING["activation"] not in ("gelu", "relu"):
    raise ValueError("Explicit actual activation required")''')
    text = text.replace('GCDF_SHA = "41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70"',
                        'GCDF_SHA = MODEL_BINDING["checkpoint_sha256"]')
    text = text.replace('original_case_sha256=CASE_SHA, r1_sha256=R1_SHA, gcdf_sha256=GCDF_SHA,',
                        'original_case_sha256=CASE_SHA, r1_sha256=R1_SHA, gcdf_sha256=GCDF_SHA,\n        model_activation=MODEL_BINDING["activation"], model_binding_sha256=sha(MODEL_BINDING_PATH),')
    text = text.replace('            env.update(CASE_FILE=', '''            env.update(CHECKPOINT=str(GCDF), CDF_ACTIVATION=MODEL_BINDING["activation"],
                       CDF_EXPECTED_SHA256=GCDF_SHA, CDF_CHECKPOINT_KEY=str(MODEL_BINDING.get("checkpoint_key", "latest")),
                       SUMMARY_STATE_CLOCK_COHERENT="true", FIXED_START_RECOVERY_ENABLED="false",
                       FIXED_START_REBASE_REAUDIT_ENABLED="false")
            env.update(CASE_FILE=''')
    text = text.replace('    frozen[str(R1.relative_to(REPO))] = R1_SHA',
                        '    frozen[str(R1.relative_to(REPO))] = R1_SHA\n    frozen[str(GCDF)] = sha(GCDF)\n    frozen[str(MODEL_BINDING_PATH)] = sha(MODEL_BINDING_PATH)')
    text = text.replace('            actual[str(R1.relative_to(REPO))] = sha(R1)',
                        '            actual[str(R1.relative_to(REPO))] = sha(R1)\n            actual[str(GCDF)] = sha(GCDF)\n            actual[str(MODEL_BINDING_PATH)] = sha(MODEL_BINDING_PATH)')
    # Online orchestration requires a later, concrete preflight; this preparation
    # exposes only dry-run validation of the frozen single-case runner.
    text = text.replace('    frozen = fingerprint()',
                        '    raise RuntimeError("Task launch awaits qualified model and comparison preflight; use --dry-run")\n    frozen = fingerprint()')
    (target/'run_case.py').write_text(text)
    binding = dict(checkpoint=str(rt/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'),
                   checkpoint_sha256='41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70',
                   activation='gelu', checkpoint_key='latest', model='old', diagnostic_only=True)
    save(out/'old_model_binding.json', binding)
    patch = []
    for name in changed:
        patch.extend(difflib.unified_diff((parent/name).read_text().splitlines(True),
                     (rt/name).read_text().splitlines(True), fromfile='parent/'+name, tofile='candidate/'+name))
    (out/'implementation.patch').write_text(''.join(patch))
    for name in changed:
        p = rt/name
        if p.suffix == '.py': ast.parse(p.read_text(), filename=name)
        if p.suffix == '.sh': subprocess.run(['bash', '-n', str(p)], check=True)
    ast.parse((target/'run_case.py').read_text())
    # Verify that every inherited file either relocates text paths, or belongs
    # to the small, explicit implementation change set. ELFs never relocate.
    for name, digest in lock['files'].items():
        old, new = parent/name, rt/name
        assert sha(old) == digest
        a, b = old.read_bytes(), new.read_bytes()
        if b'\0' not in b:
            b = b.replace(str(rt).encode(), str(parent).encode())
        if name not in changed: assert a == b, name
    save(out/'preparation.json', dict(status='ISOLATED_HARNESS_PREPARED_NOT_LAUNCHED',
         changed_inherited_files=changed, root_algorithm_promoted=False, inherited_files=len(lock['files']),
         copied_runtime=str(rt), new_model=None, diagnostic_cpu_supported=True,
         domain_recording='Every complete CQ01 request, before inference, plus result/closure accounting; no point filtering.',
         channel_limit='CQ01 has no channel id; all active local/final queries share this worker.'))
    shutil.copy2(__file__, out/Path(__file__).name)
    print('HARNESS_PREPARED', out, flush=True)


if __name__ == '__main__':
    main()
