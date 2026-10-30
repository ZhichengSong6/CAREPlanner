"""Offline launch expansion only: no master, CUDA, Gazebo, or robot nodes."""
import ast
import os
from pathlib import Path
import subprocess
import unittest
import roslaunch

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT/'src/egocentric_arm_planner/launch/phaseC4_4_verified_regime_planner.launch'
SENTINEL = '/tmp/CARE_MISSING_legacy_samples.yaml'
FULL = ['use_local_sparse_scp:=true', 'final_executable_gcdf_enabled:=true',
        'execution_gcdf_audit_enabled:=true', 'cdf_shadow_vbc_audit_enabled:=true',
        'body_samples_file:='+SENTINEL]


def expand(path, args):
    config = roslaunch.config.ROSLaunchConfig()
    # XmlLoader is sufficient to expand includes/conditions/parameters; unlike
    # load_config_default it need not enumerate host network interfaces.
    roslaunch.xmlloader.XmlLoader().load(str(path), config, argv=args, verbose=False)
    return config


def samples(config):
    return {k: v.value for k, v in config.params.items() if k.endswith('/body_samples_file')}


class CleanupTests(unittest.TestCase):
    def test_mainline_default_has_no_legacy_parameters_or_nodes(self):
        config = expand(MAIN, FULL)
        self.assertEqual(samples(config), {})
        self.assertEqual(sum(n.type == 'gcdf_primitive_anchor_node' for n in config.nodes), 3)
        self.assertFalse(any(n.type in ['body_sample_model_node', 'body_sample_marker_publisher.py'] for n in config.nodes))

    def test_samples_rollback_preserves_custom_file(self):
        values = samples(expand(MAIN, FULL+['geometry_backend:=samples']))
        self.assertEqual(len(values), 9)
        self.assertEqual(set(values.values()), {SENTINEL})

    def test_each_mixed_branch_gets_only_its_legacy_parameters(self):
        expected = {'vbc_geometry_backend': 3, 'gcdf_geometry_backend': 3,
                    'body_prior_geometry_backend': 1, 'diagnostic_geometry_backend': 2}
        for branch, count in expected.items():
            with self.subTest(branch=branch):
                values = samples(expand(MAIN, FULL+[branch+':=samples']))
                self.assertEqual(len(values), count)
                self.assertEqual(set(values.values()), {SENTINEL})

    def test_leaf_launches_no_yaml_and_legacy_default(self):
        for name in ['confidence_map', 'trajectory_risk', 'trajectory_vbc_selector',
                     'predicted_vbc_auditor', 'executed_vbc_logger', 'body_observability', 'body_sample_model']:
            path = ROOT/'src/care_confidence_map/launch'/(name+'.launch')
            backend = 'body_prior_geometry_backend' if name == 'confidence_map' else 'geometry_backend'
            with self.subTest(launch=name):
                self.assertEqual(samples(expand(path, [backend+':=primitive', 'body_samples_file:='+SENTINEL])), {})
                values = samples(expand(path, [backend+':=samples']))
                self.assertTrue(values)
                self.assertTrue(all(Path(v).is_file() for v in values.values()))

    def test_shell_only_forwards_yaml_to_legacy_consumers(self):
        script = ROOT/'scripts/run_phase_c4_4_verified_regime_smoke.sh'
        prefix = script.read_text().split('LOCAL_SCP_CANDIDATE_TOPIC=')[0]
        keys = ['GEOMETRY_BACKEND', 'GCDF_GEOMETRY_BACKEND', 'BODY_PRIOR_GEOMETRY_BACKEND',
                'DIAGNOSTIC_GEOMETRY_BACKEND', 'VBC_GEOMETRY_BACKEND', 'BODY_SAMPLES_FILE']
        clean = {k:v for k,v in os.environ.items() if k not in keys}
        for override, expected in [({}, 0), ({'GEOMETRY_BACKEND':'samples'}, 1),
                                   ({'VBC_GEOMETRY_BACKEND':'samples'}, 1)]:
            env = dict(clean, BODY_SAMPLES_FILE=SENTINEL, **override)
            result = subprocess.run(['bash'], input=prefix+'\nprintf "%s" "${#BODY_SAMPLES_LAUNCH_ARGS[@]}"\n',
                                    env=env, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, str(expected))

    def test_yaml_import_is_legacy_only(self):
        path = ROOT/'src/care_confidence_map/scripts/executed_vbc_logger_node.py'
        tree = ast.parse(path.read_text())
        top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(n, ast.Import) and any(a.name == 'yaml' for a in n.names) for n in top_imports))
        loader = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_load_samples')
        self.assertTrue(any(isinstance(n, ast.Import) and any(a.name == 'yaml' for a in n.names) for n in ast.walk(loader)))


if __name__ == '__main__': unittest.main()
