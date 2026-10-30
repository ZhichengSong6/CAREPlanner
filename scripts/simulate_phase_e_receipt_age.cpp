// Independent schedule replay of the production fixed-lease policy.
// No budget formula, ROS, or actuator response model is used in this program.
#include <care_velocity_controller/leased_velocity_core.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

using Controller = care_velocity_controller::LeasedVelocityCore;
template <class T> void array(const T& values) {
  std::cout << '[';
  for (size_t j = 0; j < values.size(); ++j) {
    if (j) std::cout << ',';
    std::cout << values[j];
  }
  std::cout << ']';
}

int main(int argc,char** argv) {
  const bool bounded=argc==2 && std::string(argv[1])=="--bounded-targets";
  if(argc>1 && !bounded)throw std::invalid_argument("unknown replay mode");
  std::cout << std::setprecision(17);
  std::mt19937 random(190640);
  std::uniform_real_distribution<double> uniform(-1., 1.);
  uint64_t id = 0;
  for (uint64_t gap : {1000000, 5000000, 20000000, 50000000})
    for (double max_step : {.002, .02, .04})
      for (uint64_t age : {0, 1000000, 5000000, 20000000, 40000000, 50000000})
        for (uint64_t duration : {60000000, 80000000, 100000000})
          for (int pattern = 0; pattern < 5; ++pattern)
            for (uint64_t extra : {0, 10000000}) {
              Controller controller;
              controller.core.max_step = max_step;
              controller.start(1000000000, 71);
              Controller::Joints measured{}, initial_target{};
              const double fraction = std::array<double, 4>{{0., .2, .75, 1.}}[id % 4];
              for (size_t j = 0; j < 7; ++j)
                initial_target[j] = fraction * 2. * ((id + j) % 2 ? -1. : 1.);
              Controller::Lease grant;
              uint64_t sequence = 0, now = 1000000000;
              auto issue = [&](uint64_t end) {
                grant.valid = true;
                grant.session = 71;
                grant.token = 123;
                ++grant.sequence;
                ++grant.ingress;
                grant.issued_ns = now;
                grant.until_ns = end;
                grant.certificate = std::string(64, 'a');
              };
              auto update = [&](uint64_t elapsed, const Controller::Joints& target) {
                Controller::Command command;
                command.valid = true;
                command.lease = grant;
                command.generated_ns = now;
                command.source_sequence = ++sequence;
                command.ingress = sequence;
                command.velocity = target;
                measured = controller.core.output;
                controller.update(now, elapsed * 1e-9, grant, command, 0, false, measured);
              };
              // Reach an actual policy state; do not inject a synthetic output.
              for (int step = 1; step <= 2000; ++step) {
                now += 1000000;
                issue(now + 100000000);
                update(1000000, initial_target);
              }
              const uint64_t anchor = now;
              auto anchor_target=initial_target;
              if(bounded) {
                grant.target_range_enforced=true;
                for(size_t j=0;j<7;++j) {
                  const double cap=std::array<double,4>{{0.,.3,1.,2.}}[(id+j)%4];
                  grant.target_abs_velocity[j]=cap;
                  anchor_target[j]=std::max(-cap,std::min(cap,initial_target[j]));
                }
              }
              issue(anchor + duration);
              // The application acknowledgement is at this same update.
              now += 1;
              update(1, anchor_target);
              if(controller.active.sequence!=grant.sequence)throw std::runtime_error("range was not applied");
              const uint64_t measured_at = now;
              const auto applied = controller.core.output;
              Controller::Joints travel{};
              const uint64_t final_end = anchor + duration + extra;
              uint64_t ticks = 0;
              bool extended = extra == 0;
              while (controller.core.mode != Controller::Core::LATCHED && ticks < 30000) {
                ++ticks;
                uint64_t elapsed = gap;
                if (id % 2)
                  elapsed = std::max<uint64_t>(1, uint64_t(gap * (.1 + .9 * std::abs(uniform(random)))));
                // Alternate exact-deadline and just-after-deadline detection.
                if (id % 3 == 0 && now < final_end && now + elapsed > final_end)
                  elapsed = final_end - now + (id % 2);
                for (size_t j = 0; j < 7; ++j)
                  travel[j] += std::abs(controller.core.output[j]) * elapsed * 1e-9;
                now += elapsed;
                if (!extended && now < anchor + duration && final_end - now <= 100000000) {
                  issue(final_end);
                  extended = true;
                }
                Controller::Joints target{};
                for (size_t j = 0; j < 7; ++j) {
                  const double sign = applied[j] < 0. ? -1. : 1.;
                  if (pattern == 0) target[j] = sign * 2.;
                  if (pattern == 1) target[j] = -sign * 2.;
                  if (pattern == 2) target[j] = (ticks % 2 ? -1. : 1.) * 2.;
                  if (pattern == 3) target[j] = uniform(random) * 2.;
                  if(bounded)target[j]=std::max(-grant.target_abs_velocity[j],std::min(grant.target_abs_velocity[j],target[j]));
                }
                update(elapsed, target);
              }
              if (!extended || controller.core.mode != Controller::Core::LATCHED ||
                  controller.core.clock_fault || controller.rejected_leases)
                throw std::runtime_error("invalid or unfinished fixture");
              std::cout << "{\"id\":" << id++ << ",\"measured_ns\":" << measured_at
                        << ",\"now_ns\":" << measured_at + age
                        << ",\"lease_end_ns\":" << anchor + duration
                        << ",\"extra_ns\":" << extra << ",\"gap_ns\":" << gap
                        << ",\"max_step_s\":" << max_step << ",\"pattern\":" << pattern
                        << ",\"applied\":";
              array(applied);
              std::cout << ",\"target_range_enforced\":" << (bounded?"true":"false") << ",\"target_abs_velocity\":";
              array(grant.target_abs_velocity);
              std::cout << ",\"absolute_command_travel\":";
              array(travel);
              std::cout << ",\"stopped_ns\":" << now << "}\n";
            }
}
