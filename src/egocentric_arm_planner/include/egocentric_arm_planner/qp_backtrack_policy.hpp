#pragma once

#include <string>

namespace egocentric_arm_planner {

// A late hard-QP failure can be caused by the CDF point set changing after an
// accepted SCP iterate.  This policy permits one bounded re-query from the
// previous iterate.  It never treats a soft diagnostic solve as executable and
// never changes the hard-QP safety constraints.
struct QPBacktrackPolicy {
  int attempts = 0;
  int max_attempts = 1;

  void reset() { attempts = 0; }

  bool used() const { return attempts > 0; }

  static bool isLateHardFailure(const std::string& status) {
    return status.find("primal infeasible") != std::string::npos ||
           status.find("max iterations") != std::string::npos ||
           status.find("maximum iterations") != std::string::npos;
  }

  bool consume(bool normal_mode,
               bool has_hard_iterate,
               int selected_unknown_cdf_rows,
               const std::string& status) {
    if (!normal_mode || !has_hard_iterate ||
        selected_unknown_cdf_rows <= 0 || attempts >= max_attempts ||
        !isLateHardFailure(status)) {
      return false;
    }
    ++attempts;
    return true;
  }
};

}  // namespace egocentric_arm_planner
