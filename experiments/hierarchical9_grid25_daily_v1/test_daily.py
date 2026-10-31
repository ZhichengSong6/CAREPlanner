"""CPU/mock-scheduler tests; no cluster, GPU, or training data access."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('daily_under_test', HERE / 'launch.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


class FakeEngine:
    def __init__(self, complete=()):
        self.done = set(complete)
        self.integrities = []
        self.terminate_calls = []
    def choose_segment(self, m, c, job):
        for i, ids in enumerate(m['segments']):
            if any(s not in self.done for s in ids):
                return i
        return None
    def stage_exists(self, c, job, s):
        return s in self.done
    def engine_integrity(self, c, job, ids, m):
        self.integrities.append(list(ids))
    def terminate(self, processes):
        self.terminate_calls.append(processes)


class FakeProcess:
    def __init__(self, rc=0):
        self.returncode = rc
    def poll(self):
        return self.returncode


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_resource_request_single_node_four_3090_24h(self):
        cmd = d.sbatch_command(self.root, self.root, HERE, '3090node2')
        for flag in ('--nodes=1','--ntasks=1','--gres=gpu:3090:4',
                     '--time=1-00:00:00','--signal=B:USR1@1800','--no-requeue'):
            self.assertIn(flag, cmd)
        self.assertEqual(d.SOFT_SECONDS, 84600)
        self.assertFalse(any(x.startswith(('--array', '--dependency')) for x in cmd))
        self.assertIn('--nodelist=3090node1', d.sbatch_command(self.root,self.root,HERE,'3090node1'))
        for node in ('3090node3','4090node2','3090node1,3090node2'):
            with self.assertRaises(RuntimeError):
                d.sbatch_command(self.root,self.root,HERE,node)

    def test_queue_busy_or_unreadable_fails_closed(self):
        for rc, out in ((0, '18777 RUNNING g25_fast'), (1, '')):
            with mock.patch.object(d.subprocess,'run',return_value=types.SimpleNamespace(returncode=rc, stdout=out)):
                with self.assertRaises(RuntimeError):
                    d.empty_queue()

    def test_previous_daily_and_old_jobs_require_terminal(self):
        for folder, jid in (('fast_segments_v1',18772),('daily_v1',18790)):
            d.write(self.root/folder/'latest_submission.json',dict(job_id=jid))
        def reply(cmd, **kw):
            jid = cmd[cmd.index('-j')+1]
            return types.SimpleNamespace(returncode=0,stdout=f'{jid}|COMPLETED|\n')
        with mock.patch.object(d.subprocess,'run',side_effect=reply) as call:
            d.previous_done(self.root)
            self.assertEqual(call.call_count,2)
        with mock.patch.object(d.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='18772|RUNNING|\n')):
            with self.assertRaises(RuntimeError): d.previous_done(self.root)

    def test_write_never_overwrites_without_permission(self):
        p = self.root/'a.json'; d.write(p,dict(a=1))
        with self.assertRaises(FileExistsError): d.write(p,dict(a=2))
        self.assertEqual(d.read(p),dict(a=1))
        d.write(p,dict(a=2),replace=True)
        self.assertEqual(d.read(p),dict(a=2))

    def test_shared_entry_lock_excludes_nested_submit(self):
        with d.submission_lock(self.root):
            with self.assertRaises(RuntimeError):
                with d.submission_lock(self.root): pass

    def test_source_freeze_tamper_and_existing_data_unchanged(self):
        p=self.root/'formal/shards/00001/data.npz';p.parent.mkdir(parents=True);p.write_bytes(b'keep')
        out=d.freeze(self.root);self.assertEqual(d.freeze(self.root),out)
        self.assertEqual(p.read_bytes(),b'keep')
        self.assertEqual(d.read(out/'launcher_manifest.json')['files'],d.own_hashes(HERE))
        (out/'launch.py').write_text('bad')
        with self.assertRaises(RuntimeError): d.freeze(self.root)

    def submission_patches(self, replies, done=False):
        e=FakeEngine((0,) if done else ()); m=dict(root=str(self.root),source=str(HERE),segments=[[0]])
        manifest=self.root/'fast_segments_v1/manifest.json';d.write(manifest,m)
        patches=mock.patch.multiple(d,empty_queue=mock.DEFAULT,previous_done=mock.DEFAULT,load_engine=mock.DEFAULT,freeze=mock.DEFAULT)
        patched=patches.start(); self.addCleanup(patches.stop)
        patched['load_engine'].return_value=(e,m,None,{})
        patched['freeze'].return_value=HERE
        call=mock.patch.object(d.subprocess,'run',side_effect=replies).start(); self.addCleanup(mock.patch.stopall)
        return call,patched

    def test_submit_only_one_real_sbatch_preserves_old_records(self):
        old=self.root/'fast_segments_v1/latest_submission.json';d.write(old,dict(job_id=18772))
        before=old.read_bytes()
        replies=[types.SimpleNamespace(returncode=0,stdout='allowed',stderr=''),
                 types.SimpleNamespace(returncode=0,stdout='18800;cluster\n',stderr='')]
        call,_=self.submission_patches(replies)
        d.submit(self.root,self.root,'3090node2')
        self.assertEqual(call.call_count,2)
        self.assertIn('--test-only',call.call_args_list[0].args[0])
        self.assertNotIn('--test-only',call.call_args_list[1].args[0])
        self.assertEqual(d.read(self.root/'daily_v1/latest_submission.json')['job_id'],18800)
        self.assertEqual(old.read_bytes(),before)
        specp=Path(call.call_args_list[1].kwargs['env']['G25D_SPEC'])
        self.assertEqual(d.read(specp)['soft_seconds'],84600)
        self.assertNotIn('deadline_unix',d.read(specp))

    def test_failed_resource_test_never_submits(self):
        call,_=self.submission_patches([types.SimpleNamespace(returncode=1,stdout='',stderr='denied')])
        with self.assertRaises(RuntimeError): d.submit(self.root,self.root,'3090node2')
        self.assertEqual(call.call_count,1)
        self.assertFalse((self.root/'daily_v1/latest_submission.json').exists())

    def test_real_failure_or_ambiguous_reply_never_retries(self):
        for rc, out in ((1,''),(0,'unknown')):
            with self.subTest(rc=rc):
                sub=Path(tempfile.mkdtemp(dir=self.root))
                m=dict(root=str(sub),source=str(HERE),segments=[[0]])
                d.write(sub/'fast_segments_v1/manifest.json',m)
                with mock.patch.object(d,'empty_queue'),mock.patch.object(d,'previous_done'),\
                     mock.patch.object(d,'load_engine',return_value=(FakeEngine(),m,None,{})),\
                     mock.patch.object(d,'freeze',return_value=HERE),\
                     mock.patch.object(d.subprocess,'run',side_effect=[
                        types.SimpleNamespace(returncode=0,stdout='',stderr=''),
                        types.SimpleNamespace(returncode=rc,stdout=out,stderr='error')]) as call:
                    with self.assertRaises(RuntimeError):d.submit(sub,sub,'3090node2')
                    self.assertEqual(call.call_count,2)
                    self.assertEqual(len(list((sub/'daily_v1/submissions').glob('*_reply.json'))),1)

    def test_all_done_submits_nothing_and_does_not_merge(self):
        call,_=self.submission_patches([],done=True)
        d.submit(self.root,self.root,'3090node2')
        self.assertEqual(call.call_count,0)

    def test_worker_is_unchanged_frozen_executable(self):
        m=dict(source='/data/frozen_fast')
        cmd=d.worker_command(m,Path('/manifest'),2,3,Path('/request'))
        self.assertEqual(cmd[:4],[sys.executable,'-u','/data/frozen_fast/run.py','worker'])
        self.assertEqual(cmd[-6:],['--segment','2','--rank','3','--request','/request'])

    def test_launch_uses_common_deadline_four_gpu_ids_and_unique_paths(self):
        e=FakeEngine((4,5));m=dict(root=str(self.root),source='/frozen',segments=[[4,5]])
        folder=self.root/'segment_000';stop=self.root/'STOP'
        with mock.patch.dict(os.environ,SLURM_JOB_ID='91'),\
             mock.patch.object(d.subprocess,'Popen',side_effect=lambda *a,**kw:FakeProcess()) as pop:
            self.assertTrue(d.launch_segment(e,m,None,{},folder,0,123456,stop,['0','1','2','3']))
        self.assertEqual(pop.call_count,4)
        self.assertEqual([x.kwargs['env']['CUDA_VISIBLE_DEVICES'] for x in pop.call_args_list],['0','1','2','3'])
        request=d.read(folder/'request.json')
        self.assertEqual(request['deadline_unix'],123456)
        self.assertEqual(request['stop_file'],str(stop))
        self.assertEqual(d.read(folder/'result.json')['status'],'SEGMENT_COMPLETE')
        self.assertEqual(e.integrities,[[4,5]])

    def test_worker_failure_does_not_report_success(self):
        e=FakeEngine();m=dict(root=str(self.root),source='/frozen',segments=[[0]])
        with mock.patch.object(d.subprocess,'Popen',side_effect=lambda *a,**kw:FakeProcess(1)):
            with self.assertRaises(RuntimeError):
                d.launch_segment(e,m,None,{},self.root/'s',0,100,self.root/'STOP',['0','1','2','3'])
        self.assertFalse((self.root/'s/result.json').exists())
        self.assertEqual(len(e.terminate_calls),1)

    def test_one_allocation_advances_multiple_segments_and_skips_existing(self):
        e=FakeEngine((0,1));m=dict(segments=[[0,1],[2,3],[4,5]])
        before=self.root/'old_data';before.write_bytes(b'original')
        seen=[]
        def launch(engine,m,c,j,folder,seg,deadline,stop,visible):
            seen.append(seg);engine.done.update(m['segments'][seg]);return True
        with mock.patch.object(d,'launch_segment',side_effect=launch),\
             mock.patch.object(d.time,'time',return_value=0),\
             mock.patch.dict(os.environ,SLURM_JOB_ID='99'):
            result=d.run_window(e,m,None,{},self.root,['0','1','2','3'],100,self.root/'STOP')
        self.assertEqual(seen,[1,2]);self.assertEqual(result['status'],'READY_TO_MERGE')
        self.assertEqual(before.read_bytes(),b'original')

    def test_deadline_can_pause_inside_segment_without_losing_shards(self):
        e=FakeEngine();m=dict(segments=[[0,1],[2,3]])
        def partial(engine,m,c,j,folder,seg,deadline,stop,visible):
            engine.done.add(0);stop.touch();return False
        with mock.patch.object(d,'launch_segment',side_effect=partial) as call,\
             mock.patch.object(d.time,'time',return_value=0),\
             mock.patch.dict(os.environ,SLURM_JOB_ID='99'):
            result=d.run_window(e,m,None,{},self.root,['0','1','2','3'],100,self.root/'STOP')
        self.assertEqual(call.call_count,1);self.assertEqual(e.done,{0})
        self.assertEqual(result['status'],'WINDOW_PAUSED');self.assertEqual(e.choose_segment(m,None,{}),0)

    def test_expired_window_launches_no_more_workers(self):
        e=FakeEngine();m=dict(segments=[[0]])
        with mock.patch.object(d,'launch_segment') as call,\
             mock.patch.object(d.time,'time',return_value=101),\
             mock.patch.dict(os.environ,SLURM_JOB_ID='99'):
            result=d.run_window(e,m,None,{},self.root,['0','1','2','3'],100,self.root/'STOP')
        call.assert_not_called();self.assertEqual(result['status'],'WINDOW_PAUSED')

    def test_unexpected_partial_segment_is_not_silently_retried(self):
        e=FakeEngine();m=dict(segments=[[0]])
        with mock.patch.object(d,'launch_segment',return_value=False) as call,\
             mock.patch.object(d.time,'time',return_value=0),\
             mock.patch.dict(os.environ,SLURM_JOB_ID='99'):
            with self.assertRaises(RuntimeError):
                d.run_window(e,m,None,{},self.root,['0','1','2','3'],100,self.root/'STOP')
        self.assertEqual(call.call_count,1)

    def test_batch_clock_starts_at_execution_and_signal_stops_new_work(self):
        frozen = d.freeze(self.root)
        manifest = self.root/'fast_segments_v1/manifest.json'
        d.write(manifest, dict(dummy=True))
        request = self.root/'submit.json'
        d.write(request, dict(root=str(self.root), source=str(frozen),
                soft_seconds=d.SOFT_SECONDS, launcher_hashes=d.own_hashes(frozen),
                manifest_sha256=d.sha(manifest)))
        handlers = {}
        def install(sig, handler):
            old = handlers.get(sig, signal.SIG_DFL); handlers[sig] = handler; return old
        def run(e,m,c,j,folder,visible,deadline,stop):
            self.assertEqual(deadline, 1000+d.SOFT_SECONDS)
            self.assertEqual(visible, ['0','1','2','3'])
            self.assertFalse(stop.exists())
            handlers[signal.SIGUSR1]()
            self.assertTrue(stop.exists())
            return dict(status='WINDOW_PAUSED')
        with mock.patch.object(d,'HERE',frozen), \
             mock.patch.object(d,'load_engine',return_value=(FakeEngine(),{},None,{})), \
             mock.patch.object(d,'run_window',side_effect=run), \
             mock.patch.object(d.time,'time',return_value=1000), \
             mock.patch.object(d.signal,'signal',side_effect=install), \
             mock.patch.dict(os.environ,SLURM_JOB_ID='123',SLURM_JOB_NUM_NODES='1',CUDA_VISIBLE_DEVICES='0,1,2,3'):
            self.assertEqual(d.batch(request)['status'],'WINDOW_PAUSED')
        self.assertEqual(d.read(self.root/'daily_v1/runs/123/window.json')['start_unix'],1000)
        self.assertEqual(handlers[signal.SIGUSR1],signal.SIG_DFL)

    def test_syntax_and_spool_script_uses_absolute_source(self):
        subprocess.run(['bash','-n',str(HERE/'run.sh')],check=True)
        compile((HERE/'launch.py').read_bytes(),str(HERE/'launch.py'),'exec')
        text=(HERE/'run.sh').read_text()
        self.assertLess(text.index('if [[ "$MODE" == _batch ]]'),text.index('HERE='))
        self.assertIn('"$G25D_SOURCE/launch.py" batch --spec "$G25D_SPEC"',text)
        self.assertIn('NODE="${NODE:-3090node2}"',text)
        self.assertNotIn('scancel',text)


if __name__ == '__main__':
    unittest.main()
