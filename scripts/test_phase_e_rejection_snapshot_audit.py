#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import tempfile
import unittest
from audit_phase_e_rejection_snapshots import validate_record, audit_snapshots


def fixture():
    def trajectory(stamp):
        return dict(stamp_ns=stamp,joint_names=['j'+str(i) for i in range(7)],
            points=[dict(time_ns='0',positions=[0.]*7)])
    return dict(result='unsafe',safety_gate='vbc',committed=False,
        raw_candidate_stamp_ns='10',audited_trajectory_stamp_ns='30',context_status='exact_raw_stamp',
        local_context=dict(raw_candidate_stamp_ns='10',query_trajectory=trajectory('9'),local_gcdf_batch=dict(stamp_ns='9')),
        raw_trajectory=trajectory('10'),handoff_trajectory=trajectory('10'),
        final_gcdf_trajectory=trajectory('20'),audited_trajectory=trajectory('30'),
        final_gcdf_batch=dict(header=dict(stamp='20')),vbc_evidence=dict(trajectory_stamp_ns='30',violations=[dict(point=[.1,.2,.3])]))


class AuditTests(unittest.TestCase):
    def test_gcdf_record_and_tampering(self):
        d=fixture();d['safety_gate']='gcdf_unknown'
        d['final_gcdf_trajectory']['stamp_ns']='30'
        d['final_gcdf_batch']=dict(header=dict(stamp='30'),point_flat=[.1,.2,.3],
            distance=[-.02],source_type=[0],q_linearization_flat=[0.]*7,gradient_flat=[1.]*7)
        d['gcdf_evidence']=dict(trajectory_stamp_ns='30',batch_stamp_ns='30',safety_margin=0.,
            violations=[dict(pair=0,timestep=0,point=[.1,.2,.3],distance=-.02,source_type=0,
                q_linearization=[0.]*7,gradient=[1.]*7,candidate_q=[0.]*7,time_ns='0')])
        self.assertEqual(validate_record(d),('10','30'))
        d['gcdf_evidence']['violations'][0]['candidate_q'][6]=.1
        with self.assertRaises(AssertionError):validate_record(d)

    def test_complete_record(self):
        self.assertEqual(validate_record(fixture()),('10','30'))

    def test_bad_id_or_changed_geometry_fails(self):
        for mutate in [lambda d:d.update(context_status='NOT_RECORDED'),
                       lambda d:d['local_context'].update(raw_candidate_stamp_ns='11'),
                       lambda d:d['audited_trajectory']['points'][0]['positions'].__setitem__(0,.01)]:
            d=fixture();mutate(d)
            with self.assertRaises(AssertionError):validate_record(d)

    def test_queued_without_file_is_failure_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'verification_outcome.csv').write_text('%time,field.data\n2000000000,result=unsafe safety_gate=vbc raw_candidate_stamp_ns=10 audited_trajectory_stamp_ns=30 rejection_snapshot=queued\n')
            audit=audit_snapshots(p,dict(start_ros_s=1,end_ros_s=3))
            self.assertEqual(audit['status'],'FAIL')
            self.assertEqual(audit['final_vbc_rejections'],1)

    def test_no_trigger_is_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(audit_snapshots(directory,dict(start_ros_s=1,end_ros_s=3))['status'],'NOT_RUN')


if __name__=='__main__':unittest.main()
