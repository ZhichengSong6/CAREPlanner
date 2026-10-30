#!/usr/bin/env python3
"""Compile the unchanged production frontier objective block in a small fixture."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREFIX = r'''
#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include <cassert>
#include <tuple>
#include <iostream>
using namespace egocentric_arm_planner;
namespace care_collision_cdf {
struct CollisionCDFConstraintBatch { enum { SOURCE_UNKNOWN=0, SOURCE_OCCUPIED=1 };
  std::vector<int> source_type; std::vector<double> point_flat; };
}
using V=Eigen::VectorXd;
struct Row { int pair=0,k=2; double d=.22,safety_margin=.22; V g=V::Unit(7,0),qlin=V::Zero(7); };
struct Result { int recovery_normals=0,recovery_tight_normals=0; bool recovery_rejoining=false;
  V recovery_frontier; std::vector<Eigen::Vector3d> observation_dependency_points;
  std::vector<std::tuple<int,double,double>> terms; };
Result objective(int source, double d=.22, double desired=-.03, int attempt=1, bool cached=false) {
  const bool repair_mode=true;
  const int dof_=7, num_intervals_=20, visibility_frontier_horizon_step_=2;
  const double visibility_waypoint_weight_=4000., trust_radius=.2;
  struct Frontier { bool active=true; V q=V::Zero(7),joint_mask=V::Ones(7),recovery_target;
    int recovery_attempt=1; double frontier_weight_scale=10.; } frontier;
  frontier.q[0]=desired; frontier.recovery_attempt=attempt;
  if(cached) frontier.recovery_target=V::Constant(7,.012);
  Eigen::MatrixXd q_bar=Eigen::MatrixXd::Zero(7,21);
  std::vector<Row> selected(1); selected[0].d=d;
  care_collision_cdf::CollisionCDFConstraintBatch batch;
  if(source>=0) batch.source_type={source};
  batch.point_flat={-.15,0.,.9};
  Result out;
  auto finiteVector=[](const V& q) { return q.allFinite(); };
  auto qIndex=[](int k,int j) { return 7*k+j; };
  auto addQuadraticTarget=[&](int i,double w,double q) { out.terms.emplace_back(i,w,q); };
'''
SUFFIX = r'''
  return out;
}
int main() {
  auto unknown=objective(0);
  assert(unknown.observation_dependency_points.size()==1);
  assert(unknown.recovery_normals==1 && !unknown.recovery_rejoining);
  assert(unknown.terms.size()==19*7); // k=2..20, including executable prefix/tail
  for(const auto& t:unknown.terms) {
    assert(std::get<0>(t)>=14 && std::get<0>(t)<147);
    assert(std::get<1>(t)==40000.);
    assert(std::get<2>(t)==unknown.recovery_frontier[std::get<0>(t)%7]);
  }
  assert(objective(1).observation_dependency_points.empty()); // OCCUPIED != sensing
  assert(objective(-1).observation_dependency_points.empty()); // missing provenance
  assert(objective(0,.22,.03).observation_dependency_points.empty()); // not blocking goal
  auto fresh=objective(0,.5,-.03,1,true);
  assert(fresh.recovery_rejoining && fresh.observation_dependency_points.empty());
  assert(std::abs(fresh.recovery_frontier[0]+.03)<1e-12); // disregard old tangent
  auto cached=objective(0,.22,-.03,1,true);
  assert((cached.recovery_frontier-V::Constant(7,.012)).norm()<1e-12);
  auto ordinary=objective(0,.22,-.03,0);
  assert(ordinary.terms.size()==7 && ordinary.observation_dependency_points.size()==1);
  assert(ordinary.recovery_normals==0 && ordinary.recovery_tight_normals==0);
  assert(objective(1,.22,-.03,0).observation_dependency_points.empty());
  assert(objective(-1,.22,-.03,0).observation_dependency_points.empty());
  assert(objective(0,.22,.03,0).observation_dependency_points.empty());
  std::cout << "PASS production UNKNOWN routing, OCCUPIED/missing-source exclusion, tail tracking and local rejoin\n";
}
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    text = (ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp').read_text()
    start = text.index('  if (repair_mode &&\n      frontier.active &&')
    end = text.index('  for (int k = 0; k < num_intervals_; ++k)', start)
    source = out/'production_dependency.cpp'
    source.write_text(PREFIX+text[start:end]+SUFFIX)
    binary = out/'production_dependency'
    build = subprocess.run(['g++', '-std=c++14', '-O0', '-I/usr/include/eigen3',
        '-I'+str(ROOT/'src/egocentric_arm_planner/include'), str(source), '-o', str(binary)],
        capture_output=True, text=True)
    (out/'build.log').write_text(build.stdout+build.stderr)
    if build.returncode:
        raise RuntimeError(build.stderr)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    (out/'result.txt').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr, end='')
    result.check_returncode()


if __name__ == '__main__':
    main()
