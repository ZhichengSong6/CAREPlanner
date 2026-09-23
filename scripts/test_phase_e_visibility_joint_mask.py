#!/usr/bin/env python3
"""Compile the production q_vis objective block and verify joint masking."""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

PREFIX = r'''
#include "egocentric_arm_planner/visibility_objective_step.hpp"
#include <Eigen/Core>
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>
namespace ros { struct Time { static Time now() { return {}; } double toSec() const { return 10.; } }; }
using egocentric_arm_planner::visibilityObjectiveStep;
using V = Eigen::VectorXd;
struct Waypoint { bool terminal_objective=true; double deadline_abs_s=11.; V q; V joint_mask; };
struct Frontier { bool active=false; double qvis_weight_scale=1.; };
std::vector<std::tuple<int,double,double>> objective(const V& mask) {
  const int dof_=7, num_intervals_=20;
  const double dt_=.05, visibility_waypoint_weight_=3000.;
  const bool repair_mode=true;
  Frontier frontier;
  Waypoint waypoint; waypoint.q=V::Zero(7); waypoint.joint_mask=mask;
  std::vector<Waypoint> schedule{waypoint};
  std::vector<std::tuple<int,double,double>> terms;
  struct Result { int visibility_objective_step=-1; } out;
  auto qIndex=[](int k,int j) { return 7*k+j; };
  auto addQuadraticTarget=[&](int i,double w,double q) {
    if(w>0.) terms.emplace_back(i,w,q);
  };
'''

SUFFIX = r'''
  return terms;
}
int main() {
  V s45(7); s45 << 1.,1.,1.,1.,0.,0.,0.;
  const auto masked=objective(s45);
  assert(masked.size()==4);
  for(const auto& term:masked) {
    assert(std::get<0>(term)>=140 && std::get<0>(term)<=143);
    assert(std::get<1>(term)==3000.);
  }
  const auto legacy=objective(V::Ones(7));
  assert(legacy.size()==7);
  std::cout << "PASS S4/S5 constrain four upstream joints; legacy constrains seven\n";
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    planner = (ROOT / 'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp').read_text()
    start = planner.index('  const double qvis_weight =')
    end = planner.index('  if (repair_mode &&\n      frontier.active &&', start)
    source = output / 'production_visibility_joint_mask.cpp'
    source.write_text(PREFIX + planner[start:end] + SUFFIX)
    binary = output / 'production_visibility_joint_mask'
    build = subprocess.run([
        'g++', '-std=c++14', '-O0', '-I/usr/include/eigen3',
        '-I' + str(ROOT / 'src/egocentric_arm_planner/include'),
        str(source), '-o', str(binary),
    ], capture_output=True, text=True)
    (output / 'build.log').write_text(build.stdout + build.stderr)
    if build.returncode:
        raise RuntimeError(build.stderr)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    (output / 'result.txt').write_text(run.stdout + run.stderr)
    print(run.stdout + run.stderr, end='')
    run.check_returncode()


if __name__ == '__main__':
    main()
