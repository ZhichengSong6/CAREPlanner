#!/usr/bin/env python3
"""Local synthetic tests only: no network, actual checkpoints or robot execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "v1_pull", Path(__file__).with_name("pull_hierarchical9_v1_artifacts.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class PullTests(unittest.TestCase):
    def test_pinned_hashes(self):
        self.assertEqual(len(M.MODELS), 3)
        for value in M.MODELS.values():
            self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_verified_install_and_same_file(self):
        with tempfile.TemporaryDirectory() as root:
            source, dest = Path(root) / "part", Path(root) / "final.pt"
            source.write_bytes(b"known-model")
            expected = M.digest(source)
            M.install_verified(source, dest, expected)
            M.install_verified(source, dest, expected)
            self.assertEqual(dest.read_bytes(), b"known-model")

    def test_mismatch_does_not_publish(self):
        with tempfile.TemporaryDirectory() as root:
            source, dest = Path(root) / "part", Path(root) / "final.pt"
            source.write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):
                M.install_verified(source, dest, "0" * 64)
            self.assertFalse(dest.exists())

    def test_never_overwrites_existing_different(self):
        with tempfile.TemporaryDirectory() as root:
            source, dest = Path(root) / "part", Path(root) / "final.pt"
            source.write_bytes(b"new")
            dest.write_bytes(b"preserve-user-file")
            with self.assertRaises(RuntimeError):
                M.install_verified(source, dest, M.digest(source))
            self.assertEqual(dest.read_bytes(), b"preserve-user-file")

    def test_symlinks_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            repo.mkdir()
            outside = Path(root) / "outside"
            outside.mkdir()
            (repo / "link").symlink_to(outside, target_is_directory=True)
            for rel in ("../outside/file", "link/final.pt", "link"):
                with self.assertRaises(RuntimeError):
                    M.local_target(repo, rel)

    def test_existing_verified_fetch_is_offline(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            (repo / "final.pt").write_bytes(b"already-here")
            with mock.patch.object(M.subprocess, "run") as run:
                M.fetch_one(repo, M.HOST, M.REMOTE_REPO, "final.pt", M.digest(repo / "final.pt"))
                run.assert_not_called()

    def test_fake_transfer_and_source_identity_check(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            data = b"synthetic-download"
            def fake_scp(cmd, **kwargs):
                self.assertEqual(cmd[0], "scp")
                Path(cmd[-1]).write_bytes(data)
                return subprocess.CompletedProcess(cmd, 0)
            with mock.patch.object(M.subprocess, "run", side_effect=fake_scp):
                M.fetch_one(repo, M.HOST, M.REMOTE_REPO, "models/final.pt",
                            hashlib.sha256(data).hexdigest())
            self.assertEqual((repo / "models/final.pt").read_bytes(), data)
            self.assertEqual(list((repo / "models").glob("*.part")), [])
        wrong = dict(M.MODELS)
        wrong[next(iter(wrong))] = "0" * 64
        with mock.patch.object(M.subprocess, "run", return_value=
                               subprocess.CompletedProcess([], 0, json.dumps(wrong))):
            with self.assertRaises(RuntimeError):
                M.remote_hashes(M.HOST, M.REMOTE_REPO, list(M.MODELS))

    def test_offline_manifest_and_report_guards(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            models = {rel: hashlib.sha256(rel.encode()).hexdigest() for rel in M.MODELS}
            for rel in models:
                target = repo / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(rel.encode())
            report = repo / M.EVAL
            report.mkdir(parents=True)
            manifest = {"checkpoints": {name: {"step": 50000, "sha256": models[rel]}
                        for name, rel in zip(("hierarchical", "old_scalar", "old8"), models)}}
            (report / "manifest.json").write_text(json.dumps(manifest))
            (report / "comparison.json").write_text(json.dumps(
                {"field": {"hierarchical_sensor_max": {"count": 163840}}}))
            (report / "summary.md").write_text("synthetic")
            (repo / M.REPORTS[-1]).write_text("{}")
            with mock.patch.object(M, "MODELS", models):
                M.check_local(repo, list(models) + M.REPORTS, models)
                manifest["checkpoints"]["hierarchical"]["step"] = 2
                (report / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(RuntimeError):
                    M.check_local(repo, list(models) + M.REPORTS, models)


if __name__ == "__main__":
    unittest.main(verbosity=2)
