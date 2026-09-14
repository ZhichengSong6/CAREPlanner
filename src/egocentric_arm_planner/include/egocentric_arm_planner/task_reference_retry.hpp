#pragma once
#include <cstdint>

namespace egocentric_arm_planner {
// Pure state machine: steady-clock seconds supplied by the caller. Retries
// rebuild only the nominal reference, never an actuator command.
struct TaskReferenceRetry {
  std::uint64_t id = 0;
  int attempts = 0;
  int max_attempts = 3;
  double retry_delay_s = 0.5;
  double timeout_s = 5.0;
  double deadline = 0.0;
  double next_attempt = 0.0;
  bool active = false;
  void start(double now) {
    ++id; attempts = 0; active = true;
    deadline = now + timeout_s; next_attempt = now;
  }
  bool expired(double now) const { return active && now >= deadline; }
  bool due(double now) const {
    return active && !expired(now) && attempts < max_attempts && now >= next_attempt;
  }
  bool claim(double now) {
    if (!due(now)) return false;
    ++attempts; next_attempt = deadline; return true;
  }
  bool fail(std::uint64_t request, double now) {
    if (request != id || !active) return false;
    active = attempts < max_attempts && now < deadline;
    next_attempt = now + retry_delay_s;
    return active;
  }
  bool succeed(std::uint64_t request, double now) {
    if (request != id || !active || expired(now)) return false;
    active = false; return true;
  }
};

// Correlate cross-topic reference/status delivery without relying on ROS's
// header.seq (which publishers are allowed to rewrite).
struct TaskReferenceReceipt {
  std::uint64_t id = 0, min_stamp = 0, last_reference_stamp = 0;
  bool pending = false, failed = false;
  bool status(std::uint64_t request, std::uint64_t stamp, bool exhausted) {
    if (request < id) return false;
    const bool newer = request > id;
    id = request; min_stamp = stamp;
    if (last_reference_stamp != 0 && last_reference_stamp >= stamp && (!failed || newer)) {
      failed = false; pending = false; return true;
    }
    failed = exhausted; pending = !exhausted; return true;
  }
  bool reference(std::uint64_t stamp) {
    if (stamp < min_stamp) return false;
    if (failed) { last_reference_stamp = stamp; return false; }
    last_reference_stamp = stamp; pending = false; return true;
  }
};
}  // namespace egocentric_arm_planner
