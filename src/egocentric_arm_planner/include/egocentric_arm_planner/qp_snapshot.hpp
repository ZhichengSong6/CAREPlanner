#pragma once
#include <Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <ostream>

namespace egocentric_arm_planner {
inline void jsonNumber(std::ostream& out, double v) {
  if (std::isfinite(v)) out << std::setprecision(17) << v;
  else out << (std::isnan(v) ? "\"nan\"" : v > 0 ? "\"inf\"" : "\"-inf\"");
}
template<class Vector> void jsonVector(std::ostream& out, const Vector& v) {
  out << '[';
  for (std::size_t i = 0; i < static_cast<std::size_t>(v.size()); ++i) {
    if (i) out << ',';
    jsonNumber(out, v[i]);
  }
  out << ']';
}
inline void jsonSparse(std::ostream& out, const Eigen::SparseMatrix<double>& a) {
  out << "{\"shape\":[" << a.rows() << ',' << a.cols() << "],\"coo\":[";
  bool first = true;
  for (int k = 0; k < a.outerSize(); ++k)
    for (Eigen::SparseMatrix<double>::InnerIterator it(a,k); it; ++it) {
      if (!first) out << ',';
      first = false;
      out << '[' << it.row() << ',' << it.col() << ',';
      jsonNumber(out, it.value()); out << ']';
    }
  out << "]}";
}
}  // namespace egocentric_arm_planner
