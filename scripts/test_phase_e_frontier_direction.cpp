#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <cassert>
#include <iostream>
#include <limits>
using namespace egocentric_arm_planner;
using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& t) {
  Eigen::VectorXd v(t.size()); int i=0;
  for(const auto& p:t) v[i++]=std::stod(p.second.data());
  return v;
}
int main(int argc,char** argv) {
  assert(!recoveryForwardBlocked(.001,.05)); // tight but improves
  assert(!recoveryForwardBlocked(-.1,.02)); // improves yet still deficient: hard QP decides
  assert(recoveryForwardBlocked(.001,-.05)); // genuinely crosses boundary
  assert(!recoveryForwardBlocked(.004,-.001)); // decreases but remains feasible
  assert(recoveryForwardBlocked(-.001,0.)); // deficient and no improvement
  assert(!recoveryForwardBlocked(.001,0.)); // feasible neutral direction
  assert(recoveryForwardBlocked(-.001,1e-10)); // numerical noise is not improvement
  assert(!recoveryForwardBlocked(0.,0.));
  assert(recoveryForwardBlocked(std::numeric_limits<double>::quiet_NaN(),.1));
  assert(recoveryForwardBlocked(.001,std::numeric_limits<double>::infinity()));
  // A conflicting row is still a blocker even when another row improves.
  const Eigen::VectorXd q=Eigen::VectorXd::Zero(7), forward=.05*Eigen::VectorXd::Unit(7,0);
  std::vector<Eigen::VectorXd> blockers;
  for(int sign:{-1,1}) {
    const Eigen::VectorXd g=sign*Eigen::VectorXd::Unit(7,0);
    if(recoveryForwardBlocked(.001,g.dot(forward))) blockers.push_back(g);
  }
  assert(blockers.size()==1);
  const auto detour=boundedTangentTarget(q,forward,blockers,2,.03);
  assert(detour.size()==7 && std::abs(detour[0])<1e-12);
  int records=0, alternatives=0, rows=0;
  if(argc==2) {
    ptree root; boost::property_tree::read_json(argv[1],root);
    for(const auto& record:root.get_child("records")) {
      const auto& r=record.second; const auto q0=vec(r.get_child("q0"));
      for(const auto& candidate:r.get_child("compatible_forward_targets")) {
        const auto target=vec(candidate.second); int blocked=0;
        for(const auto& row:r.get_child("k2_rows")) {
          const auto& w=row.second;
          const double at_start=w.get<double>("distance")+
              vec(w.get_child("gradient")).dot(q0-vec(w.get_child("q_linearization")));
          const double along=vec(w.get_child("gradient")).dot(target-q0);
          blocked+=recoveryForwardBlocked(at_start-w.get<double>("margin"),along); ++rows;
        }
        assert(blocked==0); ++alternatives;
      }
      ++records;
    }
    assert(records==52);
  }
  std::cout<<"{\"status\":\"PASS\",\"recorded_solves\":"<<records
      <<",\"compatible_forward_inputs\":"<<alternatives<<",\"row_checks\":"<<rows<<"}\n";
}
