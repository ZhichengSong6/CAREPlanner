// Offline-only replay using the repository's exact vendored PIQP headers.
#include <piqp/piqp.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <iomanip>
using boost::property_tree::ptree;
Eigen::VectorXd vector(const ptree& tree) {
  Eigen::VectorXd v(tree.size()); int i=0;
  for (const auto& child:tree) v[i++]=std::stod(child.second.data());
  return v;
}
Eigen::SparseMatrix<double> matrix(const ptree& tree) {
  const auto shape=vector(tree.get_child("shape"));
  Eigen::SparseMatrix<double> a(shape[0],shape[1]);
  std::vector<Eigen::Triplet<double>> triplets;
  for (const auto& child:tree.get_child("coo")) {
    const auto row=vector(child.second);
    triplets.emplace_back(row[0],row[1],row[2]);
  }
  a.setFromTriplets(triplets.begin(),triplets.end()); a.makeCompressed(); return a;
}
int main(int argc,char** argv) {
  if (argc!=2) { std::cerr<<"usage: replay_phase_e_qp SNAPSHOT.json\n"; return 2; }
  try {
    ptree t; boost::property_tree::read_json(argv[1],t);
    piqp::SparseSolver<double> solver;
    auto& s=solver.settings();
    s.max_iter=t.get<int>("max_iter");
    s.eps_abs=t.get<double>("eps_abs"); s.eps_rel=t.get<double>("eps_rel");
    s.kkt_solver=piqp::KKTSolver::sparse_ldlt; s.compute_timings=true;
    const auto P=matrix(t.get_child("P")), A=matrix(t.get_child("A")), G=matrix(t.get_child("G"));
    const auto c=vector(t.get_child("c")), b=vector(t.get_child("b"));
    const auto hl=vector(t.get_child("h_l")), hu=vector(t.get_child("h_u"));
    const auto xl=vector(t.get_child("x_l")), xu=vector(t.get_child("x_u"));
    solver.setup(P,c,A,b,G,hl,hu,xl,xu);
    const auto status=solver.solve(); const auto& info=solver.result().info;
    const bool same=t.get<std::string>("status")==piqp::status_to_string(status);
    std::cout<<std::setprecision(17)<<"{\"status\":\""<<piqp::status_to_string(status)
      <<"\",\"same_status\":"<<(same?"true":"false")
      <<",\"iterations\":"<<info.iter<<",\"primal_residual\":"<<info.primal_res
      <<",\"dual_residual\":"<<info.dual_res<<"}\n";
    return same ? 0 : 1;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
