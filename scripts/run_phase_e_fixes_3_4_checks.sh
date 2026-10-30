#!/usr/bin/env bash
set -eo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
source devel/setup.bash
set -u
g++ -std=c++14 -O1 -I/usr/include/eigen3 -Isrc/egocentric_arm_planner/include -Isrc/arm_model/include scripts/test_phase_e_fixes_3_4.cpp -o /tmp/care_fix34_test
/tmp/care_fix34_test
g++ -std=c++14 -O1 -I/usr/include/eigen3 -Isrc/egocentric_arm_planner/include scripts/test_phase_e_repair_no_progress.cpp -o /tmp/care_repair_policy_test
/tmp/care_repair_policy_test
care_piqp_include="$(python3 scripts/prepare_piqp.py)"
g++ -std=c++14 -O1 -I/usr/include/eigen3 -I"${care_piqp_include}" scripts/replay_phase_e_qp.cpp -o /tmp/care_replay_qp
/usr/bin/python3 scripts/test_phase_e_fixes_3_4.py -v
/usr/bin/python3 scripts/test_phase_e_fixes_1_2.py -v
/usr/bin/python3 scripts/test_phase_e_reference_retry_ros.py --fix34 --output-dir "${1:?new output directory required}"
