#!/usr/bin/env python3
"""Read-only install preflight. Never launches ROS nodes or actuator owners.

Reports dependency/asset provenance; a PASS is not a performance or robot release.
Run after sourcing ONLY this install space (which extends /opt/ros/noetic).
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


PACKAGES = ('arm_description', 'arm_model', 'arm_trajectory', 'care_confidence_map',
            'care_collision_cdf', 'care_visibility_cdf', 'egocentric_arm_planner')


def is_elf(path):
    with path.open('rb') as handle:
        return handle.read(4) == b'\x7fELF'


def inspect(prefix):
    import rospkg
    report = dict(status='FAIL', checks={}, prefix=str(prefix), failure_stage='package resolution')
    try:
        packages = {p: Path(rospkg.RosPack().get_path(p)).resolve() for p in PACKAGES}
        assert all(path == prefix/'share'/p for p, path in packages.items()), packages
        report['checks']['installed_package_resolution'] = 'PASS'
        report['failure_stage'] = 'artifact provenance'
        files = [p for p in prefix.rglob('*') if p.is_file()]
        assert files, 'empty installation'
        assert all(prefix in p.resolve().parents for p in files), 'external artifact symlink'
        forbidden = [str(p.relative_to(prefix)) for p in files if
                     p.suffix in {'.pt', '.pth', '.npz', '.bag', '.zip', '.jsonl'} or
                     'checkpoints' in p.relative_to(prefix).parts]
        assert not forbidden, forbidden
        report['checks']['no_bundled_models_datasets_or_logs'] = 'PASS'
        report['sha256'] = {str(p.relative_to(prefix)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in files if p.suffix not in {'.pyc', '.pyo'}}
        report['failure_stage'] = 'dynamic linker'
        elf = [p for p in files if os.access(p, os.X_OK) and is_elf(p)]
        assert elf, 'missing runtime binaries'
        dependencies = {}
        for p in elf:
            proc = subprocess.run(['ldd', str(p)], text=True, capture_output=True)
            assert proc.returncode == 0 and 'not found' not in proc.stdout, (p, proc.stdout, proc.stderr)
            # No workspace devel/build RPATH leakage in the actual resolved dependencies.
            assert '/devel/' not in proc.stdout and '/build/' not in proc.stdout, (p, proc.stdout)
            dependencies[str(p.relative_to(prefix))] = proc.stdout.splitlines()
        report['dependencies'] = dependencies
        report['checks']['all_ELF_dependencies_resolve_without_devel_or_build'] = 'PASS'
        for path, lines in dependencies.items():
            assert not any('libbody_sample_model.so' in line for line in lines), (path, lines)
        report['checks']['no_eager_legacy_body_library_dependency'] = 'PASS'
        report['failure_stage'] = 'visibility runtime helper closure'
        scripts = packages['care_visibility_cdf']/'scripts'
        modules = {p.stem for p in scripts.glob('*.py')}
        required = {'vbc_deadline_waypoint_online_node', 'per_sensor_visibility_runtime',
                    'hierarchical_visibility_cdf_model', 'bounded_visibility_recovery',
                    'check_visibility_self_occlusion', 'evaluate_direct_vs_projection_ascent'}
        assert required <= modules, sorted(required-modules)
        # Syntax-check without executing scripts or creating __pycache__ in deployment.
        for p in scripts.glob('*.py'):
            ast.parse(p.read_text(), filename=str(p))
        report['checks']['visibility_runtime_assets_and_syntax'] = 'PASS'
        report['failure_stage'] = 'production launch resolution'
        import roslaunch.config
        launch = packages['egocentric_arm_planner']/'launch/phaseC4_4_verified_regime_planner.launch'
        cfg = roslaunch.config.load_config_default([(str(launch), [
            'use_local_sparse_scp:=true', 'final_executable_gcdf_enabled:=true',
            'execution_gcdf_audit_enabled:=true', 'cdf_shadow_vbc_audit_enabled:=true'])], None, verbose=False)
        for n in cfg.nodes:
            if n.package in PACKAGES:
                path = prefix/'lib'/n.package/n.type
                assert path.is_file() and os.access(path, os.X_OK), str(path)
        assert all(p.value == 'primitive' for k, p in cfg.params.items() if k.endswith('/geometry_backend'))
        assert not any(k.endswith('/body_samples_file') for k in cfg.params), 'legacy YAML parameter in primitive graph'
        report['checks']['primitive_graph_has_no_body_sample_parameters'] = 'PASS'
        sample_alias = packages['care_confidence_map']/'config/body_samples.yaml'
        sample_canonical = packages['care_confidence_map']/'legacy/config/body_samples.yaml'
        assert sample_alias.is_symlink() and sample_alias.resolve(strict=True) == sample_canonical.resolve(strict=True)
        for name in ('generate_body_samples.py', 'body_sample_marker_publisher.py'):
            script = prefix/'lib/care_confidence_map'/name
            assert script.is_file() and not script.is_symlink() and os.access(script, os.X_OK), script
        report['checks']['legacy_assets_isolated_with_compatible_entrypoints'] = 'PASS'
        assert cfg.params['/optimized_trajectory_continuity/tcp_nodelay'].value is True
        assert cfg.params['/confidence_map_node/confidence_map/query_dedicated_queue'].value is True
        rollback = roslaunch.config.load_config_default([(str(launch), ['confidence_query_dedicated_queue:=false'])], None, verbose=False)
        assert rollback.params['/confidence_map_node/confidence_map/query_dedicated_queue'].value is False
        report['checks']['resolved_primitive_graph_executables'] = 'PASS'
        report['status'] = 'PASS'
        report['failure_stage'] = None
    except Exception:
        report['error'] = traceback.format_exc()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-prefix', type=Path, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    args = parser.parse_args()
    # Reserve output before inspection: never overwrite an earlier failed acceptance.
    with args.output_file.open('x') as handle:
        report = inspect(args.install_prefix.resolve())
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'sha256', 'dependencies'}}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
