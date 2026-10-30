#include "egocentric_arm_planner/qp_reachable_conflict.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

using boost::property_tree::ptree;
using egocentric_arm_planner::reachableCdfUpperBound;

Eigen::VectorXd values(const ptree& a) {
  Eigen::VectorXd v(a.size());
  int j = 0;
  for (const auto& item : a) v[j++] = item.second.get_value<double>();
  return v;
}

int main(int argc, char** argv) {
  assert(argc == 2);
  ptree report;
  boost::property_tree::read_json(argv[1], report);
  Eigen::VectorXd qmin(7), qmax(7), velocity(7);
  qmin << -3.14, -2.30, -3.14, -2.65, -3.14, -3.14, -1.20;
  qmax << 3.14, 2.30, 3.14, 2.65, 3.14, 3.14, 1.20;
  velocity << 2., 2., 2., 2., 2.5, 2.5, 2.5;
  int cases = 0, rows = 0, conflicts = 0, low_conflicts = 0;
  for (const auto& c : report.get_child("cases")) {
    const Eigen::VectorXd q0 = values(c.second.get_child("q0"));
    int case_conflicts = 0;
    for (const auto& item : c.second.get_child("rows")) {
      const auto& r = item.second;
      const Eigen::VectorXd g = values(r.get_child("g"));
      const Eigen::VectorXd qlin = values(r.get_child("qlin"));
      const int k = r.get<int>("k");
      const double upper = reachableCdfUpperBound(
          r.get<double>("d"), g, qlin, q0, q0, qmin, qmax,
          velocity, .01, .15, .05, k);
      assert(std::isfinite(upper));
      assert(std::abs(upper-r.get<double>("reachable_best")) < 1e-9);
      const bool conflict = r.get<double>("margin")-upper > 1e-8;
      const Eigen::VectorXd point = values(r.get_child("point"));
      const bool low = (point-Eigen::Vector3d(.15, 0., .30)).lpNorm<Eigen::Infinity>() < 1e-5;
      if (conflict) {
        ++conflicts; ++case_conflicts;
        assert(low);
        ++low_conflicts;
      }
      ++rows;
    }
    assert(case_conflicts == 20);
    ++cases;
  }
  assert(cases == 3 && rows == 180 && conflicts == 60 && low_conflicts == 60);
  // A feasible row must remain distinguishable from an impossible one.
  const Eigen::VectorXd z = Eigen::VectorXd::Zero(7);
  const Eigen::VectorXd g = Eigen::VectorXd::Unit(7, 0);
  assert(reachableCdfUpperBound(.05, g, z, z, z, qmin, qmax,
                               velocity, .01, .15, .05, 1) >= .05);
  assert(reachableCdfUpperBound(-.226, g, z, z, z, qmin, qmax,
                               velocity, .01, .15, .05, 1) < .05);
  std::cout << "PASS cases=" << cases << " rows=" << rows
            << " proven_conflicts=" << conflicts << '\n';
}
