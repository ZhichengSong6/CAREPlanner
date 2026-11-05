"""CPU-only tests. No real Slurm, GPU, network or server files are used."""
import argparse
from contextlib import nullcontext, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location('grid25_slurm_tasks_tested', HERE/'slurm_tasks.py')
g = importlib.util.module_from_spec(s); s.loader.exec_module(g)


def reply(stdout='', returncode=0, stderr=''):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.repo = self.root/'repo'; self.repo.mkdir()
        self.b = types.SimpleNamespace(STATE='two_node_v1', SOFT_SECONDS=84600,
            read=lambda p: json.loads(Path(p).read_text()), write=write, sha=digest,
            idle_locks=lambda p: nullcontext(), lock=lambda p: nullcontext(),
            empty_queue=Mock(), recorded_jobs=Mock(), probe_shard_locks=Mock(),
            scan_shards=Mock(return_value=(list(range(190)), 12160, [])),
            wait_file=Mock(), shared_probe=Mock())
        self.engine = types.SimpleNamespace(choose_segment=Mock(return_value=0),
            terminate=Mock(), engine_integrity=Mock(), stage_exists=Mock(return_value=True))
        self.m = {'source': str(self.root/'engine'), 'segments': [[0, 1], [2, 3]]}
        self.c = types.SimpleNamespace(nshards=lambda cfg: 2707)
        self.job = {'cfg': {}}
        self.b.load_engine = Mock(return_value=(self.engine, self.m, self.c, self.job))
        (self.root/'formal').mkdir(); (self.root/'formal/job.json').write_text('{}')
        (self.root/'fast_segments_v1').mkdir(); (self.root/'fast_segments_v1/manifest.json').write_text('{}')
        write(self.root/'two_node_v1/recovery_latest.json', {
            'base_job_sha256': digest(self.root/'formal/job.json'),
            'manifest_sha256': digest(self.root/'fast_segments_v1/manifest.json')})
        self.args = argparse.Namespace(root=self.root, repo=self.repo, nodes='auto', nodelist='',
                                       account='', qos='', partition='GPU', mode='next')

    def submit_context(self, responses):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(g, 'freeze', return_value=(self.root/'frozen', {})))
        stack.enter_context(patch.object(g, 'association_hint', return_value=(None, {})))
        self.called = stack.enter_context(patch.object(g, 'call', side_effect=responses))
        stack.enter_context(patch.object(g, 'status'))
        stack.enter_context(patch.dict(os.environ, VIS_PYTHON=sys.executable, SBATCH_NTASKS='999'))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_auto_and_fixed_node_modes(self):
        self.assertEqual(g.resources('auto'), '1-2')
        self.assertEqual(g.resources('auto', cap=1), '1')
        self.assertEqual(g.resources('1'), '1')
        self.assertEqual(g.resources('2'), '2')

    def test_auto_does_not_force_both_named_nodes(self):
        with self.assertRaises(RuntimeError): g.resources('auto', '3090node2,3090node3')
        self.assertEqual(g.resources('2', '3090node2,3090node3'), '2')
        with self.assertRaises(RuntimeError): g.resources('1', '4090node1')
        with self.assertRaises(RuntimeError): g.resources('2', '3090node2,3090node2')

    def test_site_example_layout_with_explicit_gpu_reservation(self):
        text = g.batch_text(self.root, self.repo, self.root, '1-2', '', '', '', 'GPU')
        for line in ('#SBATCH --nodes=1-2', '#SBATCH --ntasks-per-node=4',
                     '#SBATCH --cpus-per-task=16', '#SBATCH --gres=gpu:3090:4'):
            self.assertIn(line, text)
        self.assertNotIn('--ntasks=', text)
        self.assertNotIn('--exclusive', text)
        self.assertNotIn('--nodelist', text)
        p = self.root/'batch.sh'; p.write_text(text)
        subprocess.run(['bash', '-n', str(p)], check=True)

    def test_unlimited_batch_requests_no_time_limit_and_no_end_signal(self):
        batch = g.batch_text(self.root, self.repo, self.root, '1', '', '', '', 'GPU',
                             unlimited=True)
        self.assertIn('#SBATCH --time=0', batch)
        self.assertNotIn('#SBATCH --signal=', batch)
        self.assertIn('#SBATCH --nodes=1', batch)
        self.assertIn('#SBATCH --gres=gpu:3090:4', batch)
        self.assertIn('#SBATCH --no-requeue', batch)
        path = self.root/'unlimited.sbatch'; path.write_text(batch)
        subprocess.run(['bash', '-n', str(path)], check=True)
        normal = g.batch_text(self.root, self.repo, self.root, '1', '', '', '', 'GPU')
        self.assertIn('#SBATCH --time=1-00:00:00', normal)
        self.assertIn('#SBATCH --signal=B:USR1@1800', normal)

    def test_unlimited_submission_preserves_shards_and_freezes_mode(self):
        self.args.unlimited = True
        with self.submit_context([reply('estimated'), reply('899')]):
            g.submit(self.b, self.args)
        self.assertEqual(self.called.call_count, 2)
        record = self.b.read(self.root/g.STATE/'latest_submission.json')
        self.assertEqual(record['job_id'], 899)
        spec = self.b.read(record['spec'])
        self.assertIs(spec['unlimited'], True)
        self.assertIn('#SBATCH --time=0', Path(record['spec']).parent.joinpath('batch.sbatch').read_text())
        self.assertEqual((self.root/'formal/job.json').read_text(), '{}')

    def test_unlimited_batch_runs_past_old_soft_deadline_to_completion(self):
        self.b.SOFT_SECONDS = -1  # Simulate an already-expired 23.5h window.
        self.engine.choose_segment.side_effect = [0, None, None]
        proc = Mock(); proc.wait.return_value = 0
        with patch.object(g, 'load_spec', return_value=({'requested_nodes': '1', 'unlimited': True}, self.root)) as _:
            with patch.object(g.subprocess, 'Popen', return_value=proc) as popen:
                with patch.dict(os.environ, SLURM_JOB_ID='899', SLURM_JOB_NUM_NODES='1'):
                    with redirect_stdout(io.StringIO()):
                        g.batch(self.b, self.root/'spec')
        self.assertEqual(popen.call_count, 1)
        report = self.b.read(self.root/g.STATE/'runs/899/result.json')
        self.assertEqual(report['status'], 'READY_TO_MERGE')
        allocation = self.b.read(self.root/g.STATE/'runs/899/allocation.json')
        self.assertIs(allocation['unlimited'], True)
        self.assertEqual(allocation['deadline'], g.UNLIMITED_DEADLINE_UNIX)
        step = self.b.read(self.root/g.STATE/'runs/899/segment_000/step.json')
        self.assertEqual(step['deadline'], g.UNLIMITED_DEADLINE_UNIX)

    def test_unlimited_stops_if_segment_makes_zero_progress(self):
        self.engine.choose_segment.return_value = 0
        self.engine.stage_exists.return_value = False
        proc = Mock(); proc.wait.return_value = 0
        with patch.object(g, 'load_spec', return_value=({'requested_nodes': '1', 'unlimited': True}, self.root)):
            with patch.object(g.subprocess, 'Popen', return_value=proc) as popen:
                with patch.dict(os.environ, SLURM_JOB_ID='900', SLURM_JOB_NUM_NODES='1'):
                    with redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(RuntimeError, 'without new published shards'):
                            g.batch(self.b, self.root/'spec')
        self.assertEqual(popen.call_count, 1)

    def test_unlimited_retries_incomplete_segment_only_with_progress(self):
        self.engine.choose_segment.side_effect = [0, 0, None, None]
        self.engine.stage_exists.side_effect = [False, False, True, False,
                                               True, False, True, True]
        proc = Mock(); proc.wait.return_value = 0
        with patch.object(g, 'load_spec', return_value=({'requested_nodes': '1', 'unlimited': True}, self.root)):
            with patch.object(g.subprocess, 'Popen', return_value=proc) as popen:
                with patch.dict(os.environ, SLURM_JOB_ID='901', SLURM_JOB_NUM_NODES='1'):
                    with redirect_stdout(io.StringIO()):
                        g.batch(self.b, self.root/'spec')
        self.assertEqual(popen.call_count, 2)
        self.assertTrue((self.root/g.STATE/'runs/901/segment_000_retry_001').is_dir())
        self.assertEqual(self.b.read(self.root/g.STATE/'runs/901/result.json')['status'], 'READY_TO_MERGE')

    def test_no_shell_injection_in_resource_fields(self):
        with self.assertRaises(RuntimeError):
            g.batch_text(self.root, self.repo, self.root, '2', '', 'account\n#SBATCH --nodes=99', '', 'GPU')

    def test_dynamic_srun_four_or_eight_tasks(self):
        for n in (1, 2):
            cmd = g.step_command(self.root, self.root/'spec', self.root/'step', n)
            self.assertIn('--nodes='+str(n), cmd)
            self.assertIn('--ntasks='+str(n*4), cmd)
            self.assertIn('--ntasks-per-node=4', cmd)
            self.assertIn('--gres=gpu:3090:4', cmd)
            self.assertIn('--kill-on-bad-exit=1', cmd)
            self.assertIn('task', cmd)

    def test_gpu_selection_uses_allocated_tokens_not_physical_guesses(self):
        self.assertEqual(g.gpu_token('GPU-a,GPU-b,GPU-c,GPU-d', 2), 'GPU-c')
        self.assertEqual(g.gpu_token('3,5,6,7', 0), '3')
        self.assertEqual(g.gpu_token('GPU-bound', 3), 'GPU-bound')
        for bad in ('', '0,1', '0,0,1,2'):
            with self.assertRaises(RuntimeError): g.gpu_token(bad, 0)

    def identities(self, nodes):
        return [dict(rank=i, node=i//4, local=i%4, host='host'+str(i//4), gpu_uuid='gpu'+str(i))
                for i in range(4*nodes)]

    def test_identity_four_and_eight_workers(self):
        for n in (1, 2): g.validate_identities(self.identities(n), n)

    def test_duplicate_gpu_is_rejected_before_collection(self):
        rows = self.identities(2); rows[1]['gpu_uuid'] = rows[0]['gpu_uuid']
        with self.assertRaises(RuntimeError): g.validate_identities(rows, 2)

    def test_same_physical_host_for_two_nodes_is_rejected(self):
        rows = self.identities(2)
        for r in rows: r['host'] = 'same'
        with self.assertRaises(RuntimeError): g.validate_identities(rows, 2)

    def test_visible_association_node_limit_caps_auto(self):
        with patch.object(g, 'call', return_value=reply('cluster|lab|GPU|cpu=256,node=1|\n')):
            cap, evidence = g.association_hint('', 'GPU')
        self.assertEqual(cap, 1); self.assertTrue(evidence['conservative_single_node'])

    def test_other_partition_limit_is_not_used(self):
        with patch.object(g, 'call', return_value=reply('cluster|lab|CPU|node=1|\n')):
            self.assertIsNone(g.association_hint('', 'GPU')[0])

    def test_unavailable_policy_does_not_claim_unlimited(self):
        with patch.object(g, 'call', side_effect=FileNotFoundError('sacctmgr unavailable')):
            cap, evidence = g.association_hint('', 'GPU')
        self.assertIsNone(cap); self.assertIn('warning', evidence)

    def test_zero_node_association_stops(self):
        with patch.object(g, 'call', return_value=reply('cluster|lab|GPU|node=0|\n')):
            with self.assertRaises(RuntimeError): g.association_hint('', 'GPU')

    def test_draft_is_zero_scheduler_calls(self):
        self.args.mode = 'draft'; self.args.nodes = '1'
        with self.submit_context([]): g.submit(self.b, self.args)
        self.called.assert_not_called()
        self.assertFalse((self.root/g.STATE/'latest_submission.json').exists())

    def test_exactly_one_real_submission_and_existing_points_kept(self):
        with self.submit_context([reply('predicted'), reply('777\n')]):
            g.submit(self.b, self.args)
        commands = [x.args[0] for x in self.called.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertIn('--test-only', commands[0]); self.assertNotIn('--test-only', commands[1])
        self.assertNotIn('SBATCH_NTASKS', self.called.call_args_list[1].kwargs['env'])
        rec = self.b.read(self.root/g.STATE/'latest_submission.json')
        self.assertEqual(rec['job_id'], 777)
        self.b.scan_shards.assert_called_once()
        self.assertEqual((self.root/'formal/job.json').read_text(), '{}')

    def test_testonly_failure_means_no_real_submission(self):
        with self.submit_context([reply(returncode=1, stderr='limit')]):
            with self.assertRaises(RuntimeError): g.submit(self.b, self.args)
        self.assertEqual(self.called.call_count, 1)

    def test_actual_failure_is_not_retried(self):
        with self.submit_context([reply('ok'), reply(returncode=1, stderr='denied')]):
            with self.assertRaises(RuntimeError): g.submit(self.b, self.args)
        self.assertEqual(self.called.call_count, 2)

    def test_ambiguous_reply_is_not_retried(self):
        with self.submit_context([reply('ok'), reply('unparseable')]):
            with self.assertRaises(RuntimeError): g.submit(self.b, self.args)
        self.assertEqual(self.called.call_count, 2)

    def test_busy_queue_stops_before_submission(self):
        self.b.empty_queue.side_effect = RuntimeError('busy')
        with self.submit_context([]):
            with self.assertRaises(RuntimeError): g.submit(self.b, self.args)
        self.called.assert_not_called()

    def test_unpublished_leftovers_stop_before_submission(self):
        self.b.scan_shards.return_value = ([1], 64, ['unfinished'])
        with self.submit_context([]):
            with self.assertRaises(RuntimeError): g.submit(self.b, self.args)
        self.called.assert_not_called()

    def test_complete_collection_does_not_submit(self):
        self.engine.choose_segment.return_value = None
        with self.submit_context([]): g.submit(self.b, self.args)
        self.called.assert_not_called()

    def test_old_and_new_history_both_checked(self):
        observed = []
        self.b.recorded_jobs.side_effect = lambda _: observed.append(self.b.STATE)
        g.history(self.b, self.root)
        self.assertEqual(observed, ['two_node_v1', 'slurm_tasks_v2'])
        self.assertEqual(self.b.STATE, 'two_node_v1')

    def test_fields_parser_preserves_workdir_spaces(self):
        got = g.fields('JobId=7 JobState=PENDING Reason=AssocMaxNodePerJobLimit WorkDir=/a b StdOut=/a b/log')
        self.assertEqual(got['WorkDir'], '/a b')
        self.assertEqual(got['JobState'], 'PENDING')

    def prepare_cancel(self, state='PENDING', reason='AssocMaxNodePerJobLimit', owner=None):
        import pwd
        user = owner or pwd.getpwuid(os.getuid()).pw_name
        write(self.root/'two_node_v1/latest_submission.json', {'job_id': 18851})
        return ('JobId=18851 UserId='+user+'(1000) JobName=g25_two JobState='+state+' Reason='+reason+
                ' WorkDir='+str(self.repo)+' StdOut='+str(self.root/'logs/two_18851.out'))

    def test_cancel_requires_matching_pending_identity_and_uses_state_filter(self):
        info = self.prepare_cancel()
        with patch.object(g, 'call', side_effect=[reply('18851|PENDING|AssocMaxNodePerJobLimit'), reply(info), reply()]) as c:
            with redirect_stdout(io.StringIO()): g.cancel_blocked(self.b, self.root, self.repo)
        self.assertEqual(c.call_count, 3)
        self.assertEqual(c.call_args.args[0][0], 'scancel')
        self.assertIn('--state=PENDING', c.call_args.args[0])

    def test_cancel_never_cancels_running_job(self):
        info = self.prepare_cancel('RUNNING', 'None')
        with patch.object(g, 'call', side_effect=[reply('18851|RUNNING|node1'), reply(info)]) as c:
            with self.assertRaises(RuntimeError): g.cancel_blocked(self.b, self.root, self.repo)
        self.assertEqual(c.call_count, 2)

    def test_cancel_wrong_owner_is_refused(self):
        info = self.prepare_cancel(owner='someone_else')
        with patch.object(g, 'call', side_effect=[reply('18851|PENDING|AssocMaxNodePerJobLimit'), reply(info)]) as c:
            with self.assertRaises(RuntimeError): g.cancel_blocked(self.b, self.root, self.repo)
        self.assertEqual(c.call_count, 2)

    def test_finished_record_does_not_require_scontrol_or_scancel(self):
        self.prepare_cancel()
        with patch.object(g, 'call', return_value=reply()) as c:
            g.cancel_blocked(self.b, self.root, self.repo)
        self.assertEqual(c.call_count, 1)

    def test_two_segments_stay_in_one_job_for_one_and_two_nodes(self):
        for nodes in (1, 2):
            with self.subTest(nodes=nodes):
                self.engine.choose_segment.side_effect = [0, 1, None, None]
                proc = Mock(); proc.wait.return_value = 0
                with patch.object(g, 'load_spec', return_value=({'requested_nodes': '1-2'}, self.root)), \
                     patch.object(g.subprocess, 'Popen', return_value=proc) as popen, \
                     patch.dict(os.environ, SLURM_JOB_ID=str(nodes), SLURM_JOB_NUM_NODES=str(nodes)), \
                     redirect_stdout(io.StringIO()):
                    g.batch(self.b, self.root/'spec')
                self.assertEqual(popen.call_count, 2)
                for invocation in popen.call_args_list:
                    self.assertEqual(invocation.args[0][0], 'srun')
                    self.assertIn('--ntasks='+str(4*nodes), invocation.args[0])
                record = self.b.read(self.root/g.STATE/'runs'/str(nodes)/'result.json')
                self.assertEqual(record['status'], 'READY_TO_MERGE')

    def test_failed_step_never_starts_second_segment(self):
        proc = Mock(); proc.wait.return_value = 1
        with patch.object(g, 'load_spec', return_value=({'requested_nodes': '1'}, self.root)), \
             patch.object(g.subprocess, 'Popen', return_value=proc) as popen, \
             patch.dict(os.environ, SLURM_JOB_ID='9', SLURM_JOB_NUM_NODES='1'), redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError): g.batch(self.b, self.root/'spec')
        self.assertEqual(popen.call_count, 1)

    def test_task_executes_one_original_worker_with_local_rank(self):
        step = self.root/'step'; step.mkdir()
        write(step/'step.json', dict(nodes=1, segment=0, stop_file=str(self.root/'STOP'), deadline=9999999999))
        for row in self.identities(1): write(step/f'identity_{row["rank"]}.json', row)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True,
            device_count=lambda: 1, get_device_properties=lambda _: types.SimpleNamespace(name='RTX 3090', uuid='gpu2')))
        with patch.object(g, 'load_spec', return_value=({}, self.root)), \
             patch.dict(sys.modules, torch=fake_torch), patch.object(g.socket, 'gethostname', return_value='host0'), \
             patch.dict(os.environ, SLURM_PROCID='2', SLURM_LOCALID='2', SLURM_NODEID='0', CUDA_VISIBLE_DEVICES='0,1,2,3'), \
             patch.object(g.os, 'execv') as execute, redirect_stdout(io.StringIO()):
            g.task(self.b, self.root/'spec', step)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2')
        execute.assert_called_once()
        command = execute.call_args.args[1]
        self.assertIn(str(self.root/'engine/run.py'), command)
        self.assertEqual(command[command.index('--rank')+1], '2')
        self.assertTrue((step/'task_2/request.json').exists())

    def test_base_source_is_pinned(self):
        (self.root/'launch.py').write_text('unreviewed code')
        with patch.object(g, 'HERE', self.root):
            with self.assertRaises(RuntimeError): g.base()


if __name__ == '__main__': unittest.main()
