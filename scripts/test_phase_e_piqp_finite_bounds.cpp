// Regression: PIQP must not interpret unused finite-bound index slots as rows.
#include <piqp/piqp.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

struct FiniteBoundProbe : piqp::SparseSolver<double> {
  // Model allocator history without changing any initialized problem input.
  void fillUnusedIndices(bool upper) {
    auto& data = this->m_data;
    auto& indices = upper ? data.h_u_idx : data.h_l_idx;
    const auto count = upper ? data.n_h_u : data.n_h_l;
    for (Eigen::Index i=count;i<data.m;++i) indices[i]=i;
  }
};

int main() {
  try {
    for (bool upper : {true,false}) for (bool reused : {false,true}) {
      Eigen::SparseMatrix<double> P(2,2),A(0,2),G(3,2); P.setIdentity();
      std::vector<Eigen::Triplet<double>> entries={{0,0,1},{1,1,1},{2,0,1},{2,1,1}};
      G.setFromTriplets(entries.begin(),entries.end());
      Eigen::VectorXd c=Eigen::VectorXd::Zero(2),b(0),hl=Eigen::VectorXd::Constant(3,-1),
          hu=Eigen::VectorXd::Constant(3,1),xl=Eigen::VectorXd::Constant(2,-2),xu=Eigen::VectorXd::Constant(2,2);
      FiniteBoundProbe solver;solver.settings().max_iter=300;
      if(reused)solver.setup(P,c,A,b,G,hl,hu,xl,xu);
      if(upper)hu.tail(2).setConstant(std::numeric_limits<double>::infinity());
      else hl.tail(2).setConstant(-std::numeric_limits<double>::infinity());
      solver.setup(P,c,A,b,G,hl,hu,xl,xu);
      if(!reused)solver.fillUnusedIndices(upper);
      const auto status=solver.solve(); const auto& result=solver.result();
      if(status!=piqp::PIQP_SOLVED || !std::isfinite(result.info.primal_res) ||
          !std::isfinite(result.info.dual_res) || !std::isfinite(result.info.duality_gap) ||
          result.x.lpNorm<Eigen::Infinity>()>1e-6)
        throw std::runtime_error("unused finite-bound indices changed a feasible solve");
      std::cout << "upper="<<upper<<" reused="<<reused<<" iter="<<result.info.iter<<" PASS\n";
    }
    // An actually inconsistent pair of rows remains rejected.
    Eigen::SparseMatrix<double> P(1,1),A(0,1),G(2,1);P.setIdentity();
    G.insert(0,0)=1.;G.insert(1,0)=1.;G.makeCompressed();
    Eigen::VectorXd c=Eigen::VectorXd::Zero(1),b(0),hl(2),hu(2),xl(1),xu(1);
    hl<<1.,-1e30;hu<<1e30,0.;xl<<-2.;xu<<2.;
    piqp::SparseSolver<double> solver;solver.settings().max_iter=300;solver.setup(P,c,A,b,G,hl,hu,xl,xu);
    if(solver.solve()==piqp::PIQP_SOLVED)throw std::runtime_error("infeasible control accepted");
    std::cout << "infeasible control PASS\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
