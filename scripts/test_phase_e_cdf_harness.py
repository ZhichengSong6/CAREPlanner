#!/usr/bin/env python3
"""Exercise the real isolated worker over CQ01, exclusively on one CPU thread."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser()
P.add_argument('--runtime', type=Path, required=True)
P.add_argument('--results-dir', type=Path, required=True)
ARGS, REST = P.parse_known_args()
RT = ARGS.runtime.resolve()
OUT = ARGS.results_dir.resolve()
OUT.mkdir(exist_ok=False)
sys.path.insert(0, str(RT/'src/care_collision_cdf/scripts'))
from collision_cdf_model import CollisionCDF, MLPRegression
from cdf_input_audit import domain_summary, InputAudit, sha_file
from cdf_worker_launch import command
from analyze_cdf_input_audit import analyze

torch.set_num_threads(1)
MODEL = RT/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'
MODEL_SHA = '41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70'
FIXTURE = OUT/"relu $(touch SHOULD_NOT_EXIST) ' checkpoint.pt"
torch.manual_seed(20261023)
tiny = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[8, 4], act_fn=torch.nn.ReLU, nerf=True)
torch.save({'1': tiny.state_dict(), '2': {k: v+.01 for k, v in tiny.state_dict().items()}}, FIXTURE)
FIXTURE_SHA = sha_file(FIXTURE)
REQ = struct.Struct('<4sI')
RESP = struct.Struct('<4sI4d')


def receive(conn, n):
    data = bytearray()
    while len(data) < n:
        more = conn.recv(n-len(data))
        if not more:
            raise EOFError('Worker stopped before complete response')
        data.extend(more)
    return data


class RunningWorker:
    def __init__(self, name, model=MODEL, digest=MODEL_SHA, activation='gelu', key='latest', extra=()):
        self.path = OUT/name
        self.path.mkdir()
        self.socket = Path(tempfile.gettempdir())/('care_cdf_'+str(os.getpid())+'_'+name+'.sock')
        assert not self.socket.exists()
        self.log = (self.path/'worker.log').open('x')
        argv = [sys.executable, '-B', str(RT/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py'),
                '--checkpoint', str(model), '--expected-checkpoint-sha256', digest,
                '--activation', activation, '--checkpoint-key', key,
                '--model-attestation', str(self.path/'attestation.json'),
                '--domain-audit', str(self.path/'audit.jsonl'), '--socket', str(self.socket),
                '--device', 'cpu', '--diagnostic-cpu', '--warmup-pairs', '8', *extra]
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                              PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)

    def ready(self):
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            if self.socket.exists(): return
            if self.proc.poll() is not None:
                raise RuntimeError((self.path/'worker.log').read_text())
            time.sleep(.02)
        raise TimeoutError('Worker startup timeout')

    def close(self):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()
            raise
        finally:
            self.log.close()


class Controls(unittest.TestCase):
    def test_01_all_rows_and_float_boundary(self):
        x = np.zeros((8000, 10), dtype=np.float32)
        x[:, 2] = .5
        x[0, 0] = .8
        x[-1, 2] = 1.05
        before = x.tobytes()
        r = domain_summary(x)
        self.assertEqual(r['pairs'], 8000)
        self.assertEqual(r['outside_tolerance_rows'], 1)
        self.assertEqual(r['strict_outside_rows'], 2)
        self.assertEqual(r['outside_examples'][0]['row'], 7999)
        self.assertEqual(x.tobytes(), before)

    def test_02_nonfinite_and_bounded_examples(self):
        x = np.zeros((100, 10), dtype=np.float32)
        x[:, 2] = 1.05
        x[0, 0], x[1, 4] = np.nan, np.inf
        r = domain_summary(x)
        self.assertEqual(r['nonfinite_xyz_rows'], 1)
        self.assertEqual(r['nonfinite_q_rows'], 1)
        self.assertEqual(r['outside_tolerance_rows'], 99)
        self.assertEqual(len(r['outside_examples']), 8)
        json.dumps(r, allow_nan=False)

    def test_03_launcher_uses_safe_argv(self):
        env = dict(CHECKPOINT=str(FIXTURE), CDF_ACTIVATION='relu', CDF_EXPECTED_SHA256=FIXTURE_SHA,
                   CDF_MODEL_ATTESTATION=str(OUT/'unused_attestation.json'), CDF_INPUT_AUDIT=str(OUT/'unused_audit.jsonl'),
                   GPU_SOCKET='/tmp/unused_cdf_fixture.sock', GPU_DEVICE='cuda')
        argv = command(env, RT)
        self.assertEqual(argv[argv.index('--checkpoint')+1], str(FIXTURE))
        result = subprocess.run([sys.executable, '-B', str(RT/'scripts/cdf_worker_launch.py'), '--print-command'],
                                env=dict(os.environ, **env), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), argv)
        self.assertFalse((ROOT/'SHOULD_NOT_EXIST').exists())
        with self.assertRaises(ValueError): command(dict(env, CDF_ACTIVATION='unknown'), RT)
        with self.assertRaises(ValueError): command(dict(env, CDF_EXPECTED_SHA256='bad'), RT)
        with self.assertRaises(ValueError): command(dict(env, CDF_ACTIVATION=''), RT)

    def test_04_missing_footer_and_binding_mismatch(self):
        p = OUT/'reader_controls'; p.mkdir()
        att = p/'att.json'; att.write_text(json.dumps(dict(pid=os.getpid())))
        a = InputAudit(p/'audit.jsonl', att)
        rows = np.zeros((2, 10), dtype=np.float32)
        a.begin(rows, rows.tobytes())
        a.finish(np.ones(2), np.ones((2, 7)), 2, .1, .2)
        with self.assertRaises(AssertionError): analyze(p/'audit.jsonl', att)
        a.close()
        # Header cannot be accepted against different loaded-model evidence.
        att.write_text(json.dumps(dict(pid=os.getpid(), changed=True)))
        with self.assertRaises(AssertionError): analyze(p/'audit.jsonl', att)

    def test_05_wrong_hash_rejected_before_inference(self):
        w = RunningWorker('wrong_hash', FIXTURE, '0'*64, 'relu')
        try:
            w.proc.wait(timeout=30)
            self.assertNotEqual(w.proc.returncode, 0)
            self.assertFalse((w.path/'attestation.json').exists())
            self.assertFalse((w.path/'audit.jsonl').exists())
            self.assertFalse(w.socket.exists())
        finally: w.close()

    def test_06_relu_and_iteration_are_actual(self):
        w = RunningWorker('relu_fixture', FIXTURE, FIXTURE_SHA, 'relu', '1')
        try:
            w.ready()
            att = json.loads((w.path/'attestation.json').read_text())
            self.assertEqual(att['activation'], 'relu')
            self.assertEqual(att['selected_checkpoint'], 'iteration:1')
            self.assertEqual(att['checkpoint_sha256'], FIXTURE_SHA)
            self.assertTrue(att['diagnostic_cpu'])
            rows = np.zeros((12, 10), dtype=np.float32); rows[:, 2] = 1.05
            direct = CollisionCDF(str(FIXTURE), device='cpu', activation='relu', checkpoint_key='1')
            with socket.socket(socket.AF_UNIX) as conn:
                conn.settimeout(30); conn.connect(str(w.socket))
                for _ in range(3): self.exchange(conn, rows, direct)
        finally: w.close()
        result = analyze(w.path/'audit.jsonl', w.path/'attestation.json')
        self.assertEqual(result['pairs'], 36)
        self.assertEqual(result['outside_tolerance_rows'], 36)
        self.assertEqual(result['status'], 'COMPLETE_WITH_DOMAIN_EXCEEDANCES')

    def exchange(self, conn, rows, baseline):
        tensor = torch.tensor(rows, dtype=torch.float32)
        t0 = time.perf_counter()
        d, g = baseline.pair_distance_and_gradient(tensor[:, :3], tensor[:, 3:])
        direct_ms = (time.perf_counter()-t0)*1000
        payload = rows.astype('<f4').tobytes()
        t0 = time.perf_counter()
        conn.sendall(REQ.pack(b'CQ01', len(rows))+payload)
        magic, count, *timing = RESP.unpack(receive(conn, RESP.size))
        got_d = np.frombuffer(receive(conn, count*4), dtype='<f4').copy()
        got_g = np.frombuffer(receive(conn, count*7*4), dtype='<f4').reshape(-1, 7).copy()
        roundtrip = (time.perf_counter()-t0)*1000
        self.assertEqual((magic, count), (b'CR01', len(rows)))
        np.testing.assert_array_equal(got_d, d.detach().numpy())
        np.testing.assert_array_equal(got_g, g.detach().numpy())
        self.assertTrue(all(t >= 0 for t in timing))
        return dict(pairs=count, payload_sha256=hashlib.sha256(payload).hexdigest(),
                    distance_max_error=float(np.max(np.abs(got_d-d.detach().numpy()))),
                    gradient_max_error=float(np.max(np.abs(got_g-g.detach().numpy()))),
                    direct_cpu_ms=direct_ms, worker_timing_ms=timing, ipc_roundtrip_ms=roundtrip)

    def test_07_old_model_full_payload_three_repetitions(self):
        bundle = json.loads((ROOT/'outputs/phase_e_cdf_qualification_20261023/bundle/queries.json').read_text())
        frozen = np.array([q['point_m']+q['q_rad'] for q in bundle], dtype=np.float32)
        maximum = np.resize(frozen, (8000, 10)).copy(); maximum[-1, 2] = 1.05
        direct = CollisionCDF(str(MODEL), device='cpu', activation='gelu')
        w = RunningWorker('old_model_ipc')
        records = []
        try:
            w.ready()
            att = json.loads((w.path/'attestation.json').read_text())
            self.assertEqual(att['checkpoint_sha256'], MODEL_SHA)
            self.assertEqual(att['selected_checkpoint'], 'iteration:39990')
            self.assertEqual(att['activation'], 'gelu')
            with socket.socket(socket.AF_UNIX) as conn:
                conn.settimeout(45); conn.connect(str(w.socket))
                for repeat in range(1, 4):
                    for label, rows in [('frozen1878_including_6_deliberate_oob', frozen), ('maximum8000_last_row_oob', maximum)]:
                        records.append(dict(repeat=repeat, batch=label, **self.exchange(conn, rows, direct)))
        finally: w.close()
        report = analyze(w.path/'audit.jsonl', w.path/'attestation.json')
        self.assertEqual(report['requests'], 6)
        self.assertEqual(report['pairs'], 3*(1878+8000))
        expected_oob = 3*sum(domain_summary(x)['outside_tolerance_rows'] for x in (frozen, maximum))
        self.assertEqual(report['outside_tolerance_rows'], expected_oob)
        requests = [r for r in map(json.loads, (w.path/'audit.jsonl').read_text().splitlines()) if r['event'] == 'request']
        self.assertEqual([r['payload_sha256'] for r in requests], [r['payload_sha256'] for r in records])
        (w.path/'report.json').write_text(json.dumps(dict(domain=report, comparisons=records), indent=2)+'\n')

    def test_08_online_readiness_rejects_cpu_diagnostic(self):
        spec = importlib.util.spec_from_file_location('fixture_ready', RT/'scripts/verify_cdf_worker.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        w = RunningWorker('readiness_control', FIXTURE, FIXTURE_SHA, 'relu')
        try:
            w.ready()
            with self.assertRaises(AssertionError):
                module.verify(w.path/'attestation.json', w.path/'audit.jsonl', FIXTURE_SHA, 'relu', w.socket)
        finally: w.close()

    def test_09_malformed_count_never_reaches_model(self):
        w = RunningWorker('bad_count', FIXTURE, FIXTURE_SHA, 'relu')
        try:
            w.ready()
            with socket.socket(socket.AF_UNIX) as conn:
                conn.settimeout(10); conn.connect(str(w.socket)); conn.sendall(REQ.pack(b'CQ01', 8001))
            w.proc.wait(timeout=10)
            self.assertNotEqual(w.proc.returncode, 0)
        finally: w.close()
        events = list(map(json.loads, (w.path/'audit.jsonl').read_text().splitlines()))
        self.assertEqual([r['event'] for r in events], ['session', 'error', 'close'])
        self.assertEqual(events[-1]['started_pairs'], 0)
        self.assertEqual(events[-1]['failed_requests'], 1)
        with self.assertRaises(ValueError): analyze(w.path/'audit.jsonl', w.path/'attestation.json')

    def test_10_binding_dry_run_has_no_task_launch(self):
        binding = RT.parent/'old_model_binding.json'
        cmd = ['/usr/bin/python3', '-B', str(RT/'outputs/cdf_model_ab/run_case.py'),
               '--dry-run', '--case-ids', '008,029,025,026', '--batch-id', 'cdf_harness_dryrun_only']
        r = subprocess.run(cmd, cwd=RT, env=dict(os.environ, CARE_CDF_MODEL_BINDING=str(binding),
                          PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=30)
        (OUT/'bound_case_dryrun.log').write_text(r.stdout+r.stderr)
        self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        self.assertEqual(json.loads(r.stdout)['case_count'], 4)
        self.assertFalse((RT/'outputs/phase_e_r1_mainline_30/cdf_harness_dryrun_only').exists())

    def test_11_shutdown_with_connected_idle_client(self):
        w = RunningWorker('shutdown_idle', FIXTURE, FIXTURE_SHA, 'relu')
        conn = socket.socket(socket.AF_UNIX)
        try:
            w.ready(); conn.connect(str(w.socket))
            # A real request proves accept() has installed the live connection.
            rows = np.zeros((1, 10), dtype=np.float32)
            direct = CollisionCDF(str(FIXTURE), device='cpu', activation='relu')
            self.exchange(conn, rows, direct)
            t0 = time.perf_counter()
            w.close()
            elapsed = time.perf_counter()-t0
            self.assertLess(elapsed, 2.)
            self.assertEqual(w.proc.returncode, 0)
            self.assertEqual(analyze(w.path/'audit.jsonl', w.path/'attestation.json')['requests'], 1)
            (w.path/'shutdown.json').write_text(json.dumps(dict(elapsed_s=elapsed, exit_code=w.proc.returncode))+'\n')
        finally:
            conn.close()
            if not w.log.closed: w.close()

    def test_12_exclusive_log_and_corruption_rejection(self):
        p = OUT/'exclusive_control'; p.mkdir()
        att = p/'att.json'; att.write_text(json.dumps(dict(pid=os.getpid(), checkpoint_sha256='0'*64, activation='gelu')))
        a = InputAudit(p/'audit.jsonl', att)
        with self.assertRaises(FileExistsError): InputAudit(p/'audit.jsonl', att)
        x = np.zeros((1, 10), dtype=np.float32)
        a.begin(x, x.tobytes()); a.finish(np.ones(1), np.ones((1, 7)), 1, .1, .2); a.close()
        rows = list(map(json.loads, (p/'audit.jsonl').read_text().splitlines()))
        rows[1]['pairs'] = 2
        corrupted = p/'corrupted.jsonl'; corrupted.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(AssertionError): analyze(corrupted, att)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *REST], verbosity=2)
