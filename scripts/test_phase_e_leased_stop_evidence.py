#!/usr/bin/env python3
"""Conditional fixtures only; no physical qualification is manufactured here."""
import copy,hashlib,json,unittest
import numpy as np
from phase_e_leased_stop_evidence import evaluate,query_complete_stopping_snapshot,JOINTS

class BindingTests(unittest.TestCase):
    def setUp(self):
        self.now=10000000000
        identity=dict(controller_session=71,execution_stamp_ns=9000000000,lease_sequence=3,
            source_sequence=4,valid_until_ns=10090000000,certificate_sha256='a'*64,generated_ns=9999000000)
        self.command=dict(identity,joint_names=JOINTS,velocity=[.3]*7,stop=False)
        self.state=dict(identity,t_ns=self.now,mode=1,clock_fault=False,integral_valid=True,lease_required=True,
            q=[0.]*7,measured=[.2]*7,applied=[.2]*7,max_update_gap=.001,
            velocity_limits=[2.]*7,acceleration_limits=[3.,3.,4.,4.,6.,6.,6.],max_braking_step_s=.02)
        self.args=dict(now_ns=self.now,state=self.state,command=self.command,
            velocity_limits=[2.]*7,acceleration_limits=[3.,3.,4.,4.,6.,6.,6.])
        limits=dict(velocity=self.args['velocity_limits'],acceleration=self.args['acceleration_limits'])
        digest=hashlib.sha256(json.dumps(limits,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        base=dict(limits_sha256=digest,runtime_sha256='fixture_only',evidence_sha256='b'*64)
        self.timing=dict(base,kind='enforced_ros_sim_time_bounds',next_decision_s=.01,transport_bound_s=.002,update_gap_s=.001)
        self.response=dict(base,kind='qualified_full_horizon_response_bound',error_rad=[.001]*7,settle_s=.01)

    def test_exact_ack_binding_does_not_invent_missing_bounds(self):
        r=evaluate(**self.args)
        self.assertTrue(r['binding_consistent']);self.assertFalse(r['qualified'])
        self.assertIn('missing_physical_response_bound',r['reasons'])
        self.assertIn('missing_qualified_time_contract',r['reasons'])
        self.assertIn('missing_complete_stopping_snapshot',r['reasons'])
        self.assertEqual(r['budgets'],{});self.assertFalse(r['runtime_admission'])

    def test_each_receipt_identity_field_must_match(self):
        for field in ('controller_session','execution_stamp_ns','lease_sequence','source_sequence','valid_until_ns','generated_ns','certificate_sha256'):
            with self.subTest(field=field):
                command=copy.deepcopy(self.command)
                command[field]=command[field]+1 if isinstance(command[field],int) else 'c'*64
                args=dict(self.args,command=command)
                self.assertFalse(evaluate(**args)['binding_consistent'])

    def test_age_expiry_fault_and_unbounded_pending_grant_reject(self):
        for update in [dict(t_ns=self.now-51000000),dict(clock_fault=True),dict(lease_required=False),
                       dict(mode=2),dict(applied=[float('nan')]*7)]:
            args=dict(self.args,state=dict(self.state,**update))
            self.assertFalse(evaluate(**args)['binding_consistent'])
        pending=[dict(controller_session=71,valid_until_ns=self.state['valid_until_ns']+1)]
        self.assertIn('outstanding_authorization_not_enclosed',evaluate(**self.args,outstanding_leases=pending)['reasons'])

    def test_observed_trial_maximum_cannot_be_used_as_qualified_bound(self):
        observed=dict(self.response,kind='observed_trial_maximum')
        result=evaluate(**self.args,timing=self.timing,response=observed,runtime_sha256='fixture_only')
        self.assertIn('missing_physical_response_bound',result['reasons'])

    def test_controller_acceleration_mismatch_cannot_underestimate_stopping(self):
        state=dict(self.state,acceleration_limits=[1.]*7)
        result=evaluate(**dict(self.args,state=state))
        self.assertIn('controller_limits_mismatch',result['reasons'])
        self.assertFalse(result['binding_consistent'])

    def test_next_cycle_and_physical_terms_survive_fixture_qualification(self):
        r=evaluate(**self.args,timing=self.timing,response=self.response,runtime_sha256='fixture_only')
        self.assertTrue(r['binding_consistent']);self.assertFalse(r['qualified'])
        a=r['budgets']['withdraw_now'];b=r['budgets']['continue_then_stop']
        self.assertTrue(np.all(np.array(b['physical_travel_rad'])>a['physical_travel_rad']))
        self.assertTrue(np.all(np.array(a['physical_travel_rad'])>a['command_travel_rad']))
        self.assertFalse(r['runtime_admission'])

    def test_actual_braking_step_changes_the_budget(self):
        args=dict(self.args,timing=dict(self.timing,update_gap_s=.01),response=self.response,runtime_sha256='fixture_only')
        ordinary=evaluate(**args)
        args['state']=dict(self.state,max_braking_step_s=.002)
        limited=evaluate(**args)
        self.assertGreater(limited['budgets']['withdraw_now']['command_stop_s'],ordinary['budgets']['withdraw_now']['command_stop_s'])

    def test_acknowledged_target_range_reduces_budget_and_must_match(self):
        cap=[.3,0.,0.,0.,0.,0.,0.]
        self.state.update(target_range_enforced=True,target_abs_velocity=cap)
        self.command.update(target_range_enforced=True,target_abs_velocity=cap,velocity=[.3,0.,0.,0.,0.,0.,0.])
        args=dict(self.args,timing=self.timing,response=self.response,runtime_sha256='fixture_only')
        bounded=evaluate(**args)
        self.assertTrue(bounded['binding_consistent'])
        self.assertGreater(bounded['budgets']['withdraw_now']['command_travel_rad'][1],0.)
        self.assertLess(bounded['budgets']['withdraw_now']['command_travel_rad'][1],.1)
        self.command['target_abs_velocity']=[.3]*7
        self.assertIn('controller_command_target_range_mismatch',evaluate(**args)['reasons'])

    def test_earlier_but_wider_outstanding_range_cannot_be_ignored(self):
        self.state.update(target_range_enforced=True,target_abs_velocity=[.3]*7)
        self.command.update(target_range_enforced=True,target_abs_velocity=[.3]*7)
        pending=[dict(controller_session=71,valid_until_ns=self.now+1000000,target_range_enforced=False)]
        result=evaluate(**self.args,outstanding_leases=pending)
        self.assertIn('outstanding_target_range_not_enclosed',result['reasons'])

class MapTests(unittest.TestCase):
    def setUp(self):
        self.calls=[];self.defect=None
        self.budgets={k:dict(physical_travel_rad=[d]*7,required_until_s=11.) for k,d in [('withdraw_now',.1),('continue_then_stop',.2)]}
        class Geometry:
            def raster(inner,q,travel,meta,margin):
                ids=[0,1] if travel[0]<.15 else [1,2]
                return dict(grid_indices=ids,points=np.asarray([[i*.05,0.,0.] for i in ids]),inside=True)
        self.geometry=Geometry()

    def query(self,points,horizon,expected):
        self.calls.append(dict(points=copy.deepcopy(points),horizon=horizon,expected=expected))
        n=len(points)
        row=dict(valid=True,instance='map',epoch=0,revision=3,frame='base_link',resolution=.05,
            map_min=[0.,0.,0.],map_max=[1.,1.,1.],grid_size=[20,20,20],snapshot_ns=10000000000,
            clock_fault=False,horizon_s=horizon,inside_map=[1]*n,confidence_at_horizon=[.95]*n,
            occupancy=[0.]*n,provenance=[1]*n,model_free_at_horizon=[1]*n)
        if n and self.defect=='unknown':row['confidence_at_horizon'][1]=0.
        if n and self.defect=='incomplete':row['inside_map'].pop()
        if n and self.defect=='oracle':row['provenance'][1]=4
        if expected and self.defect=='changed':row['revision']=4
        return row

    def run_query(self):
        return query_complete_stopping_snapshot(budgets=self.budgets,q=[0.]*7,geometry=self.geometry,
            query=self.query,now_ns=lambda:10000000000)

    def test_queries_full_union_and_rechecks_one_content_version(self):
        result=self.run_query()
        self.assertEqual(result['grid_indices'],[0,1,2]);self.assertEqual(result['points_queried'],3)
        self.assertEqual(len(self.calls),3);self.assertEqual(self.calls[2]['expected'],('map',0,3))
        self.assertTrue(result['all_centers_free']);self.assertTrue(result['stable_version'])

    def test_full_point_evidence_can_be_saved_for_later_analysis(self):
        saved=[]
        result=query_complete_stopping_snapshot(budgets=self.budgets,q=[0.]*7,geometry=self.geometry,
            query=self.query,now_ns=lambda:10000000000,evidence_sink=saved.append)
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0]['enclosure_sha256'],result['enclosure_sha256'])
        self.assertEqual(saved[0]['grid_indices'],[0,1,2])
        for key in ['inside_map','confidence_at_horizon','occupancy','provenance','model_free_at_horizon']:
            self.assertEqual(len(saved[0]['evidence'][key]),3)

    def test_proven_nested_geometry_builds_larger_box_once_for_both_modes(self):
        calls=[]
        class NestedGeometry:
            nested_joint_box_coverage=True
            def raster(inner,q,travel,meta,margin):
                calls.append(list(travel))
                ids=[0,1] if travel[0]<.15 else [0,1,2]
                return dict(grid_indices=ids,points=np.asarray([[i*.05,0.,0.] for i in ids]),inside=True)
        result=query_complete_stopping_snapshot(budgets=self.budgets,q=[0.]*7,geometry=NestedGeometry(),
            query=self.query,now_ns=lambda:10000000000)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['grid_indices'],[0,1,2])
        self.assertEqual(result['bound_modes'],['withdraw_now','continue_then_stop'])
        self.assertFalse(result['future_environment_qualified']);self.assertFalse(result['runtime_admission'])

    def test_unknown_or_test_oracle_cannot_certify_all_centers_free(self):
        for defect in ['unknown','oracle']:
            self.defect=defect;self.assertFalse(self.run_query()['all_centers_free'])

    def test_changed_or_incomplete_content_is_not_accepted(self):
        self.defect='changed';self.assertFalse(self.run_query()['stable_version'])
        self.defect='incomplete'
        with self.assertRaisesRegex(ValueError,'incomplete_stopping_snapshot'):self.run_query()

    def test_command_only_enclosure_cannot_substitute_for_mechanical_enclosure(self):
        self.budgets['withdraw_now'].pop('physical_travel_rad')
        with self.assertRaisesRegex(ValueError,'missing_physical_enclosure'):self.run_query()

    def test_explicit_command_only_map_diagnostic_is_identified(self):
        budgets={k:dict(command_travel_rad=v['physical_travel_rad'],required_until_s=11.)
                 for k,v in self.budgets.items()}
        result=query_complete_stopping_snapshot(budgets=budgets,q=[0.]*7,geometry=self.geometry,
            query=self.query,now_ns=lambda:10000000000,enclosure_field='command_travel_rad')
        self.assertEqual(result['enclosure_kind'],'command_only')
        self.assertFalse(result['runtime_admission'])

if __name__=='__main__':unittest.main()
