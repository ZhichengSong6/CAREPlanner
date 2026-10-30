#!/usr/bin/env python3
"""Import installed visibility runtime with viscdf Python; no ROS init/model load."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-prefix', type=Path, required=True)
    parser.add_argument('--output-file', type=Path, required=True)
    args = parser.parse_args()
    prefix = args.install_prefix.resolve()
    with args.output_file.open('x') as handle:
        result = dict(status='FAIL', scope=__doc__)
        try:
            sys.path.insert(0, str(prefix/'lib/care_visibility_cdf'))
            sys.path.append('/usr/lib/python3/dist-packages')
            names = ['vbc_deadline_waypoint_online_node', 'vbc_deadline_waypoint_rolling_node',
                     'per_sensor_visibility_runtime', 'hierarchical_visibility_cdf_model']
            result['paths'] = {n: importlib.import_module(n).__file__ for n in names}
            assert all(prefix in Path(p).resolve().parents for p in result['paths'].values()), result['paths']
            result['status'] = 'PASS'
        except Exception:
            result['error'] = traceback.format_exc()
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
