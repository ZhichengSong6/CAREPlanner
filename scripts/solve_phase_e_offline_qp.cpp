// Offline JSON/stdin QP solver. Uses the same vendored PIQP as production.
#include <piqp/piqp.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <iomanip>
#include <iostream>

using boost::property_tree::ptree;
Eigen::VectorXd vec(const ptree& t) {
  Eigen::VectorXd v(t.size()); int i = 0;
  for (const auto& x : t) v[i++] = std::stod(x.second.data());
  return v;
}
Eigen::SparseMatrix<double> mat(const ptree& t) {
  const auto shape = vec(t.get_child("shape"));
  Eigen::SparseMatrix<double> m(shape[0], shape[1]);
  std::vector<Eigen::Triplet<double>> entries;
  for (const auto& x : t.get_child("coo")) {
    const auto a = vec(x.second); entries.emplace_back(a[0], a[1], a[2]);
  }
  m.setFromTriplets(entries.begin(), entries.end()); m.makeCompressed(); return m;
}
int main() {
  try {
    ptree t; boost::property_tree::read_json(std::cin, t);
    piqp::SparseSolver<double> solver;
    auto& s = solver.settings();
    s.max_iter = t.get<int>("max_iter");
    s.eps_abs = t.get<double>("eps_abs"); s.eps_rel = t.get<double>("eps_rel");
    s.kkt_solver = piqp::KKTSolver::sparse_ldlt;
    solver.setup(mat(t.get_child("P")), vec(t.get_child("c")),
                 mat(t.get_child("A")), vec(t.get_child("b")),
                 mat(t.get_child("G")), vec(t.get_child("h_l")), vec(t.get_child("h_u")),
                 vec(t.get_child("x_l")), vec(t.get_child("x_u")));
    const auto status = solver.solve(); const auto& r = solver.result();
    std::cout << std::setprecision(17) << "{\"status\":\"" << piqp::status_to_string(status)
              << "\",\"iterations\":" << r.info.iter
              << ",\"primal_residual\":" << r.info.primal_res
              << ",\"dual_residual\":" << r.info.dual_res << ",\"x\":[";
    if (status == piqp::PIQP_SOLVED) {
      for (int i = 0; i < r.x.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << r.x[i];
      }
    }
    std::cout << ']';
    if (status == piqp::PIQP_SOLVED && t.get<bool>("include_duals", false)) {
      auto emit = [&](const char* name, const Eigen::VectorXd& v) {
        std::cout << ",\"" << name << "\":[";
        for (int i = 0; i < v.size(); ++i) {
          if (i) std::cout << ',';
          std::cout << v[i];
        }
        std::cout << ']';
      };
      emit("y", r.y); emit("z_l", r.z_l); emit("z_u", r.z_u);
      emit("z_bl", r.z_bl); emit("z_bu", r.z_bu);
    }
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
