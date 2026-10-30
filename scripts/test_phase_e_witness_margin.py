#!/usr/bin/env python3
"""Compile and exercise production row selection and QP lower-bound assembly.

Only surrounding message/config fixtures are substituted. The classifier,
witness lookup, screening and margin selection are extracted unmodified from
solveSparseSubproblem; no ROS graph, CUDA or simulated robot is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'src/egocentric_arm_planner/src/local_sparse_scp_planner.cpp'

PREFIX = r'''
#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace care_collision_cdf {
struct CollisionCDFConstraintBatch {
  enum { SOURCE_UNKNOWN=0, SOURCE_OCCUPIED=1 };
  std::vector<double> point_flat, distance, gradient_flat, q_linearization_flat;
  std::vector<int> original_timestep, source_type;
};
}
using Batch=care_collision_cdf::CollisionCDFConstraintBatch;
struct Witness {
  Eigen::Vector3d point;
  int timestep;
  double safety_margin;
  bool vbc_feedback;
  double previous_margin=-1., margin_ceiling=.30;
};
using RepairWitness=Witness;
struct Stats {
  int skipped_step0_rows=0, skipped_horizon_rows=0, skipped_safety_horizon_rows=0;
  int screened_safe_rows=0, selected_obligation_cdf_rows=0;
  int selected_occupied_cdf_rows=0, selected_unknown_cdf_rows=0;
  double min_distance=std::numeric_limits<double>::infinity(), qlin_error_inf=0;
  std::vector<int> selected_unknown_pair_indices;
};
struct Answer { Stats stats; std::vector<double> margins, lower; };
Answer select(const Batch& batch, const std::vector<Eigen::Vector3d>& obligation_points,
              const std::vector<Witness>& repair_unknown_witnesses_, double obs=.02,
              bool repair_mode=true, bool probe_mode=false) {
  const bool previous_vbc_margins=false;
  const int dof_=7, num_intervals_=20, cdf_constraint_horizon_steps_=5;
  const int n_pairs=batch.distance.size();
  const double cdf_safety_margin_=.01, trust_radius=.2;
  const bool cdf_safe_row_screening_=true;
  const std::atomic<double> visibility_obligation_cdf_margin_effective_{obs};
  Eigen::MatrixXd q_bar=Eigen::MatrixXd::Zero(7,21);
  Stats out;
  const auto& witnesses = repair_unknown_witnesses_;
'''

SUFFIX = r'''
  Answer answer; answer.stats=out;
  Eigen::VectorXd h_l=Eigen::VectorXd::Zero(selected.size());
  for (int row=0; row<int(selected.size()); ++row) {
    const auto& row_data=selected[row];
    /*ASSEMBLY*/
    answer.margins.push_back(row_data.safety_margin);
    answer.lower.push_back(h_l[row]);
  }
  return answer;
}
void require(bool ok,const char* name) { if(!ok) throw std::runtime_error(name); }
void equal(double a,double b,const char* name) { require(std::abs(a-b)<1e-12,name); }
int main() {
  try {
    const Eigen::Vector3d point(.15,0,.95), other(-.15,0,.7);
    Batch b; b.point_flat={.15,0,.95}; b.distance={.06};
    b.gradient_flat={1,0,0,0,0,0,0}; b.q_linearization_flat={0,0,0,0,0,0,0};
    b.original_timestep={6}; b.source_type={Batch::SOURCE_UNKNOWN};
    auto run=[&](std::vector<Eigen::Vector3d> obs,std::vector<Witness> witnesses,double margin=.02) {
      return select(b,obs,witnesses,margin);
    };
    auto a=run({point},{{point,1,.12,true}});
    require(a.margins.size()==1,"overlapping witness was dropped");
    equal(a.margins[0],.12,"regression: witness .12 overwritten by observation .02");
    equal(a.lower[0],.06,"actual QP lower bound must require q >= .06");
    equal(run({point},{{point,1,.12,true}},.32).margins[0],.32,"stronger observation preserved");
    equal(run({},{{point,1,.12,true}}).margins[0],.12,"witness-only margin");
    equal(run({point},{},.02).margins[0],.02,"observation-only margin");
    equal(run({point},{{point,1,.12,true},{point,1,.22,true},{other,1,.9,true}}).margins[0],.22,
          "strongest same-point witness, no cross-point leakage");
    equal(run({point},{{point,6,.12,false}}).margins[0],.12,"ordinary witness anchor timestep");
    equal(run({point},{{point,5,.12,false}}).margins[0],.12,"ordinary witness is point-scoped");
    b.original_timestep[0]=20;
    equal(run({point},{{point,1,.12,true}}).margins[0],.12,"VBC witness spans full horizon");
    equal(run({},{{point,1,.12,false}}).margins[0],.12,"ordinary witness spans full horizon");
    require(run({},{}).margins.empty(),"ordinary row beyond prefix must still be skipped");
    b.original_timestep[0]=2;
    equal(run({},{}).margins[0],.01,"ordinary row base margin unchanged");
    b.distance[0]=1.;
    require(run({},{}).margins.empty(),"ordinary safe row screening unchanged");
    require(run({point},{{point,1,.12,true}}).margins.size()==1,"persistent row must not be screened");
    b.distance[0]=.06;b.source_type[0]=Batch::SOURCE_OCCUPIED;
    a=run({point},{{point,1,.12,true}});
    equal(a.margins[0],.12,"occupied witness margin");
    require(a.stats.selected_occupied_cdf_rows==1,"occupied classification unchanged");
    a=select(b,{point},{{point,1,.12,true}},.02,false,true);
    equal(a.margins[0],.12,"PROBE overlapping identity");
    b.original_timestep[0]=0;
    require(run({point},{{point,1,.12,true}}).margins.empty(),"step zero remains excluded");
    b.original_timestep[0]=2;b.gradient_flat[0]=std::numeric_limits<double>::quiet_NaN();
    require(run({point},{{point,1,.12,true}}).margins.empty(),"nonfinite input remains excluded");
    std::cout << "PASS 17 production row-selection/assembly checks\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    source=SOURCE.read_text()
    method=source[source.index('LocalSparseSCPPlanner::solveSparseSubproblem('):]
    start=method.index('  struct SelectedRow {')
    end=method.index('  out.selected_cdf_rows =',start)
    selection=method[start:end]
    start=method.index('    h_l[row] =\n        row_data.safety_margin')
    end=method.index('    h_u[row] =',start)
    assembly=method[start:end]
    generated=PREFIX+selection+SUFFIX.replace('/*ASSEMBLY*/',assembly)
    with (out/'production_rows.cpp').open('x') as f:f.write(generated)
    build=subprocess.run(['g++','-std=c++14','-O0','-I/usr/include/eigen3',
                          str(out/'production_rows.cpp'),'-o',str(out/'production_rows')],
                         capture_output=True,text=True)
    with (out/'build.log').open('x') as f:f.write(build.stdout+build.stderr)
    if build.returncode:raise RuntimeError(build.stderr)
    test=subprocess.run([str(out/'production_rows')],capture_output=True,text=True)
    result=dict(status='PASS' if test.returncode==0 else 'FAIL',returncode=test.returncode,
                output=test.stdout+test.stderr,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                selection_sha256=hashlib.sha256(selection.encode()).hexdigest(),
                test_scope='Unmodified production row selection and actual h_l assembly; no full planner/ROS run.')
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(result['output'],end='')
    raise SystemExit(test.returncode)


if __name__=='__main__':main()
