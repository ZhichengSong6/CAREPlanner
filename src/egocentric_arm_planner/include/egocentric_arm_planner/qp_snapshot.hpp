#pragma once
#include <Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <string>

namespace egocentric_arm_planner {
inline void jsonString(std::ostream& out, const std::string& value) {
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c < 0x20) {
      const char* hex = "0123456789abcdef";
      out << "\\u00" << hex[c >> 4] << hex[c & 15];
    } else out << static_cast<char>(c);
  }
  out << '"';
}
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
template<class Matrix> void jsonTrajectory(std::ostream& out, const Matrix& m) {
  out << '[';
  for (int k = 0; k < m.cols(); ++k) {
    if (k) out << ',';
    jsonVector(out, m.col(k));
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
  out << "],\"compressed\":" << (a.isCompressed() ? "true" : "false")
      << ",\"outer\":[";
  for (int i=0;i<=a.outerSize();++i) { if(i)out << ','; out << a.outerIndexPtr()[i]; }
  out << "],\"inner\":[";
  for (int i=0;i<a.nonZeros();++i) { if(i)out << ','; out << a.innerIndexPtr()[i]; }
  out << "],\"values\":[";
  for (int i=0;i<a.nonZeros();++i) { if(i)out << ','; jsonNumber(out,a.valuePtr()[i]); }
  out << "]}";
}
}  // namespace egocentric_arm_planner
