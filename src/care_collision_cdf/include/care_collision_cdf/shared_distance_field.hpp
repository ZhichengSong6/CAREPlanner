#pragma once

// Reader for export_distance_field_casadi.py's common CDF/RDF CasADi ABI.
// No Python, PyTorch, or CasADi runtime linkage is required. Use -ldl.
#include <array>
#include <cmath>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace care_collision_cdf {

class SharedDistanceField {
 public:
  struct Result { double distance; std::array<double, 7> gradient; };
  SharedDistanceField(const std::string& library, const std::string& expected_units) {
    handle_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) throw std::runtime_error(std::string("Cannot load distance library: ")+dlerror());
    try {
      if (symbol<int (*)()>("care_distance_abi_version")() != 1)
        throw std::runtime_error("Unsupported distance ABI version");
      units_ = symbol<const char* (*)()>("care_distance_output_units")();
      backend_ = symbol<const char* (*)()>("care_distance_backend")();
      checkpoint_sha_ = symbol<const char* (*)()>("care_distance_checkpoint_sha256")();
      if (units_ != expected_units || (backend_ != "cdf" && backend_ != "rdf"))
        throw std::runtime_error("Distance backend/units mismatch: "+backend_+" / "+units_);
      if (symbol<Integer (*)()>("care_distance_n_in")() != 1 ||
          symbol<Integer (*)()>("care_distance_n_out")() != 2)
        throw std::runtime_error("Distance input/output count mismatch");
      const auto si = symbol<const Integer* (*)(Integer)>("care_distance_sparsity_in");
      const auto so = symbol<const Integer* (*)(Integer)>("care_distance_sparsity_out");
      checkDense(si(0), 1, 10); checkDense(so(0), 1, 1); checkDense(so(1), 1, 7);
      if (symbol<int (*)(Integer*, Integer*, Integer*, Integer*)>("care_distance_work")(&na_, &nr_, &ni_, &nw_) ||
          na_ < 1 || nr_ < 2 || ni_ < 0 || nw_ < 0 || nw_ > 100000000)
        throw std::runtime_error("Invalid CasADi workspace sizes");
      function_ = symbol<Function>("care_distance");
    } catch (...) { dlclose(handle_); handle_ = nullptr; throw; }
  }
  ~SharedDistanceField() { if (handle_) dlclose(handle_); }
  SharedDistanceField(const SharedDistanceField&) = delete;
  SharedDistanceField& operator=(const SharedDistanceField&) = delete;
  const std::string& units() const { return units_; }
  const std::string& backend() const { return backend_; }
  const std::string& checkpointSha256() const { return checkpoint_sha_; }

  // Local workspace makes concurrent evaluations safe for generated pure-C
  // functions; there is no shared model state or implicit global backend.
  std::vector<Result> evaluate(const std::vector<std::array<double, 10>>& pairs) const {
    std::vector<const double*> args(na_, nullptr);
    std::vector<double*> res(nr_, nullptr);
    std::vector<Integer> iw(ni_ > 0 ? ni_ : 1);
    std::vector<double> w(nw_ > 0 ? nw_ : 1);
    std::vector<Result> values(pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
      for (double x : pairs[i]) if (!std::isfinite(x))
        throw std::invalid_argument("Nonfinite distance-field input");
      args[0] = pairs[i].data(); res[0] = &values[i].distance; res[1] = values[i].gradient.data();
      if (function_(args.data(), res.data(), iw.data(), w.data(), 0))
        throw std::runtime_error("Distance-field evaluation failed");
      if (!std::isfinite(values[i].distance)) throw std::runtime_error("Nonfinite distance");
      for (double g : values[i].gradient) if (!std::isfinite(g))
        throw std::runtime_error("Nonfinite distance gradient");
    }
    return values;
  }

 private:
  using Integer = long long;
  using Function = int (*)(const double**, double**, Integer*, double*, int);
  template <class T> T symbol(const char* name) {
    dlerror(); auto pointer = dlsym(handle_, name); const char* error = dlerror();
    if (error) throw std::runtime_error(std::string("Missing distance symbol: ")+name);
    return reinterpret_cast<T>(pointer);
  }
  static void checkDense(const Integer* p, Integer rows, Integer cols) {
    if (!p || p[0] != rows || p[1] != cols)
      throw std::runtime_error("Distance sparsity shape mismatch");
    // CasADi's compact dense sparsity encoding has a 1 in position 2.
    if (p[2] == 1) return;
    if (p[2] != 0 || p[2+cols] != rows*cols)
      throw std::runtime_error("Distance outputs must be dense");
    for (Integer c = 0; c <= cols; ++c) if (p[2+c] != rows*c)
      throw std::runtime_error("Invalid distance sparsity columns");
    for (Integer i = 0; i < rows*cols; ++i) if (p[3+cols+i] != i%rows)
      throw std::runtime_error("Invalid distance sparsity rows");
  }
  void* handle_ = nullptr;
  Function function_ = nullptr;
  Integer na_ = 0, nr_ = 0, ni_ = 0, nw_ = 0;
  std::string units_, backend_, checkpoint_sha_;
};
}  // namespace care_collision_cdf
