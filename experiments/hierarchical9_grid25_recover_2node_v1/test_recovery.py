import contextlib
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import launch as L


def shared_process(folder, rank, token):
    L.shared_probe(folder, rank, token)


def held_process(path, ready, release):
    with L.lock(path):
        ready.set(); release.wait(10)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        (self.root/'formal/shards').mkdir(parents=True)
        (self.root/'formal/locks').mkdir()
        (self.root/'fast_segments_v1').mkdir()
        L.write(self.root/'formal/job.json', {'cfg': {}, 'out': str(self.root/'formal')})
        L.write(self.root/'fast_segments_v1/manifest.json', {})

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, bad=False):
        c = NS(nshards=lambda cfg: 3, stage=lambda job, sid, check=False: {'points':64})
        def integrity(c, job, ids, m):
            if bad: raise RuntimeError('Corrupt published data')
            for sid in ids:
                p = self.root/'formal/shards'/f'{sid:05d}'
                r = L.read(p/'complete.json')
                L.require(L.sha(p/'data.npz') == r['sha'], 'Corrupt published data')
        engine = NS(engine_integrity=integrity, choose_segment=lambda *a: 0)
        return engine, {'source': str(self.root/'frozen'), 'segments': [[0,1,2]]}, c, {'cfg':{}, 'out':str(self.root/'formal')}

    def completed(self, sid=0):
        p=self.root/'formal/shards'/f'{sid:05d}'; p.mkdir()
        (p/'data.npz').write_bytes(b'published data bytes')
        L.write(p/'complete.json', {'sha':L.sha(p/'data.npz')})
        return p

    def test_exact_one_allocation_two_nodes_four_gpus_each(self):
        cmd=L.sbatch_command(self.root, self.root, self.root, '3090node1,3090node2')
        for arg in ('--nodes=2','--ntasks=2','--ntasks-per-node=1','--gres=gpu:3090:4',
                    '--exclusive','--no-requeue','--time=1-00:00:00','--signal=B:USR1@1800'):
            self.assertIn(arg,cmd)
        self.assertEqual(cmd.count('sbatch'),1)
        self.assertFalse(any('--array' in x or '--dependency' in x for x in cmd))
        self.assertNotIn('--gres=gpu:3090:8',cmd)

    def test_nodes_must_be_two_distinct_3090s(self):
        for text in ('3090node1','3090node1,3090node1','4090node1,3090node2','3090node[1-2]', '3090node2,3090node3,3090node1'):
            with self.assertRaises(RuntimeError):L.nodes_arg(text)
        self.assertEqual(L.nodes_arg('3090node2,3090node1'), ['3090node2','3090node1'])

    def test_single_srun_places_one_coordinator_per_node(self):
        cmd=L.srun_command(self.root/'spec.json',self.root/'step')
        self.assertEqual(cmd[0],'srun')
        for arg in ('--nodes=2','--ntasks=2','--ntasks-per-node=1','--kill-on-bad-exit=1','--gres=gpu:3090:4'):
            self.assertIn(arg,cmd)

    def test_busy_and_unreadable_queue_stop(self):
        for rc,stdout in ((0,'99 RUNNING g25_day\n'),(1,'')):
            with patch.object(L.subprocess,'run',return_value=NS(returncode=rc,stdout=stdout,stderr='err')):
                with self.assertRaises(RuntimeError):L.empty_queue()

    def test_accounting_missing_is_recorded_not_completed(self):
        L.write(self.root/'daily_v1/latest_submission.json',{'job_id':7})
        with patch.object(L.subprocess,'run',return_value=NS(returncode=0,stdout='',stderr='')):
            h=L.recorded_jobs(self.root)
        self.assertEqual(h[0]['states'],[])
        self.assertIn('UNAVAILABLE',h[0]['note'])

    def test_nonterminal_accounting_blocks_recovery(self):
        L.write(self.root/'daily_v1/latest_submission.json',{'job_id':7})
        with patch.object(L.subprocess,'run',return_value=NS(returncode=0,stdout='7|RUNNING|\n',stderr='')):
            with self.assertRaises(RuntimeError):L.recorded_jobs(self.root)

    def test_completed_shards_survive_cleanup_byte_identical(self):
        p=self.completed(); before={x.name:L.sha(x) for x in p.iterdir()}
        t=self.root/'formal/shards/.00001.orphan'; t.mkdir(); (t/'data.npz').write_bytes(b'incomplete')
        e,m,c,j=self.fixture(); good,pts,candidates=L.scan_shards(self.root,e,m,c,j)
        moved=L.quarantine(self.root,candidates)
        self.assertEqual((good,pts),([0],64)); self.assertEqual(len(moved),1)
        self.assertFalse(t.exists()); self.assertTrue(Path(moved[0]['to_path']).exists())
        self.assertEqual(before,{x.name:L.sha(x) for x in p.iterdir()})

    def test_corrupt_published_shard_stops_before_move(self):
        p=self.completed(); (p/'data.npz').write_bytes(b'bad')
        t=self.root/'formal/shards/.00001.orphan';t.mkdir()
        with self.assertRaises(RuntimeError):L.scan_shards(self.root,*self.fixture())
        self.assertTrue(t.exists());self.assertTrue(p.exists())

    def test_unpublished_canonical_and_merge_are_quarantined(self):
        (self.root/'formal/shards/00001').mkdir()
        (self.root/'formal/.dataset.orphan').mkdir()
        good,pts,candidates=L.scan_shards(self.root,*self.fixture())
        self.assertEqual(len(L.quarantine(self.root,candidates)),2)
        self.assertEqual(good,[])

    def test_unknown_files_and_provenance_logs_not_removed(self):
        u=self.root/'formal/shards/notes.txt';u.write_text('keep')
        g=self.root/'daily_v1/runs/42/gate_rank0.json';L.write(g,{'status':'PASS'})
        l=self.root/'formal/locks/shard_00000.lock';l.touch()
        _,_,cand=L.scan_shards(self.root,*self.fixture());L.quarantine(self.root,cand)
        self.assertTrue(u.exists());self.assertTrue(g.exists());self.assertTrue(l.exists())

    def test_symlink_cleanup_is_rejected(self):
        (self.root/'formal/shards/.00001.orphan').symlink_to(self.root/'formal',target_is_directory=True)
        with self.assertRaises(RuntimeError):L.scan_shards(self.root,*self.fixture())

    def test_external_target_cannot_be_quarantined(self):
        outside=self.root/'outside';outside.mkdir()
        with self.assertRaises(RuntimeError):L.quarantine(self.root,[(outside,1,'bad')])
        self.assertTrue(outside.exists())

    def test_recovery_has_no_cancellation_or_deletion(self):
        self.completed();t=self.root/'formal/shards/.00001.orphan';t.mkdir()
        with patch.object(L,'empty_queue'),patch.object(L,'recorded_jobs',return_value=[]),patch.object(L,'load_engine',return_value=self.fixture()):
            L.recover(self.root)
        report=L.read(self.root/L.STATE/'recovery_latest.json')
        self.assertEqual(report['preserved_shards'],1)
        self.assertEqual(report['preserved_points'],64)
        self.assertEqual(len(report['quarantined']),1)
        self.assertTrue((self.root/'formal/job.json').exists())

    def test_live_shard_lock_blocks_probe(self):
        ctx=mp.get_context('spawn');ready=ctx.Event();release=ctx.Event()
        p=ctx.Process(target=held_process,args=(self.root/'formal/locks/shard_00000.lock',ready,release));p.start()
        try:
            self.assertTrue(ready.wait(3))
            with self.assertRaises(RuntimeError):L.probe_shard_locks(self.root,3)
        finally:release.set();p.join(3);p.terminate() if p.is_alive() else None
        self.assertEqual(p.exitcode,0)

    def test_real_two_process_shared_lock_handshake(self):
        ctx=mp.get_context('spawn');folder=self.root/'handshake'
        ps=[ctx.Process(target=shared_process,args=(folder,r,'job1:segment2')) for r in range(2)]
        for p in ps:p.start()
        try:
            for p in ps:p.join(6)
            self.assertTrue(all(p.exitcode==0 for p in ps))
            self.assertEqual(L.read(folder/'done.json')['token'],'job1:segment2')
        finally:
            for p in ps:
                if p.is_alive():p.terminate();p.join()

    def test_global_log_names_and_local_worker_ranks(self):
        req=self.root/'request.json';L.write(req,{'segment':4})
        logs=[]
        for node in range(2):
            for rank in range(4):
                cmd,env,p=L.worker_spec({'source':'/original/frozen'},self.root,self.root/'step',node,rank,str(rank),req)
                self.assertEqual(cmd[cmd.index('--rank')+1],str(rank))
                self.assertIn('/original/frozen/run.py',cmd)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],str(rank));logs.append(str(p))
        self.assertEqual(len(set(logs)),8)
        self.assertTrue(any('worker_7.log' in p for p in logs))

    def submit_fixture(self):
        L.write(self.root/L.STATE/'recovery_latest.json',{
            'base_job_sha256':L.sha(self.root/'formal/job.json'),
            'manifest_sha256':L.sha(self.root/'fast_segments_v1/manifest.json')})
        stack=contextlib.ExitStack()
        for name in ('empty_queue','recorded_jobs','probe_shard_locks'):
            stack.enter_context(patch.object(L,name))
        stack.enter_context(patch.object(L,'load_engine',return_value=self.fixture()))
        stack.enter_context(patch.object(L,'scan_shards',return_value=([],0,[])))
        stack.enter_context(patch.object(L,'freeze',return_value=L.HERE))
        return stack

    def test_draft_calls_no_scheduler(self):
        with self.submit_fixture(),patch.object(L.subprocess,'run') as run:
            L.submit(self.root,self.root,'3090node1,3090node2',True)
            run.assert_not_called()
        self.assertFalse((self.root/L.STATE/'latest_submission.json').exists())

    def test_exactly_one_real_sbatch_and_one_record(self):
        def fake(cmd,**kw):return NS(returncode=0,stdout='allowed\n' if '--test-only' in cmd else '888;cluster\n',stderr='')
        with self.submit_fixture(),patch.object(L.subprocess,'run',side_effect=fake) as run:
            L.submit(self.root,self.root,'3090node1,3090node2')
        actual=[c.args[0] for c in run.call_args_list if '--test-only' not in c.args[0]]
        self.assertEqual(len(actual),1);self.assertEqual(actual[0][0],'sbatch')
        self.assertEqual(L.read(self.root/L.STATE/'latest_submission.json')['job_id'],888)

    def test_testonly_rejection_no_actual_job(self):
        with self.submit_fixture(),patch.object(L.subprocess,'run',return_value=NS(returncode=1,stdout='',stderr='limit')) as run:
            with self.assertRaises(RuntimeError):L.submit(self.root,self.root,'3090node1,3090node2')
            self.assertEqual(run.call_count,1)
        self.assertFalse((self.root/L.STATE/'latest_submission.json').exists())

    def test_actual_rejection_and_ambiguous_response_not_retried(self):
        for rc,out in ((1,''),(0,'unrecognized')):
            calls=[NS(returncode=0,stdout='OK',stderr=''),NS(returncode=rc,stdout=out,stderr='failed')]
            with self.submit_fixture(),patch.object(L.subprocess,'run',side_effect=calls) as run:
                with self.assertRaises(RuntimeError):L.submit(self.root,self.root,'3090node1,3090node2')
                self.assertEqual(run.call_count,2)
        self.assertEqual(len(list((self.root/L.STATE/'submissions').glob('*_reply.json'))),2)

    def test_freeze_and_corruption_check(self):
        dest=L.freeze(self.root); self.assertEqual(L.own_hashes(dest),L.own_hashes())
        (dest/'launch.py').write_text('tampered')
        with self.assertRaises(RuntimeError):L.freeze(self.root)

    def test_failed_srun_no_next_segment(self):
        e,m,c,j=self.fixture(); e.terminate=lambda p:None
        p=NS(wait=lambda:1)
        folder=self.root/'window';folder.mkdir()
        with patch.object(L.subprocess,'Popen',return_value=p) as pop:
            with self.assertRaises(RuntimeError):L.window(self.root/'spec',folder,time.time()+100,folder/'STOP',e,m,c,j)
        self.assertEqual(pop.call_count,1)
        self.assertFalse((folder/'result.json').exists())

    def test_cross_segment_same_deadline_no_new_sbatch(self):
        e,m,c,j=self.fixture();m['segments']=[[0],[1]]
        seq=iter((0,1,None,None));e.choose_segment=lambda *a:next(seq)
        e.stage_exists=lambda *a:True;e.engine_integrity=lambda *a:None;e.terminate=lambda p:None
        folder=self.root/'window';folder.mkdir();deadline=time.time()+100
        with patch.dict(os.environ,{'SLURM_JOB_ID':'99'}),patch.object(L.subprocess,'Popen',return_value=NS(wait=lambda:0)) as pop:
            L.window(self.root/'spec',folder,deadline,folder/'STOP',e,m,c,j)
        self.assertEqual(pop.call_count,2)
        self.assertTrue(all(c.args[0][0]=='srun' for c in pop.call_args_list))
        self.assertEqual({L.read(p)['deadline_unix'] for p in folder.glob('segment_*/step.json')},{deadline})
        self.assertEqual(L.read(folder/'result.json')['status'],'READY_TO_MERGE')

    def test_expired_window_does_not_launch(self):
        e,m,c,j=self.fixture(); folder=self.root/'window';folder.mkdir()
        with patch.dict(os.environ,{'SLURM_JOB_ID':'99'}),patch.object(L.subprocess,'Popen') as pop:
            L.window(self.root/'spec',folder,time.time()-1,folder/'STOP',e,m,c,j)
            pop.assert_not_called()
        self.assertEqual(L.read(folder/'result.json')['status'],'WINDOW_PAUSED')

    def test_shell_syntax_and_spool_uses_absolute_snapshot(self):
        shell=L.HERE/'run.sh'
        subprocess.run(['bash','-n',str(shell)],check=True)
        spool=self.root/'spool_script';spool.write_bytes(shell.read_bytes())
        env=dict(os.environ,VIS_PYTHON='/bin/echo',G25R_SOURCE='/frozen/code',G25R_SPEC='/frozen/spec.json',
                 G25R_SPEC_SHA='sha',SLURM_JOB_ID='99')
        p=subprocess.run(['bash',str(spool),'_batch'],env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0)
        self.assertIn('/frozen/code/launch.py batch --spec /frozen/spec.json',p.stdout)


if __name__=='__main__':unittest.main()
