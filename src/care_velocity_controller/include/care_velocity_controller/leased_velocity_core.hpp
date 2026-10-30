#pragma once
#include <care_velocity_controller/expiring_velocity_core.hpp>
#include <string>

namespace care_velocity_controller {
// Fixed authorization deadlines are independent of producer timestamps. A new
// lease is staged; it only replaces the active lease when its command applies.
// This is a protocol fence, NOT a stopping-envelope certificate validator.
class LeasedVelocityCore {
 public:
  using Core = ExpiringVelocityCore;
  using Joints = Core::Joints;
  struct Lease {
    uint64_t session=0, token=0, sequence=0, issued_ns=0, until_ns=0, ingress=0;
    std::string certificate;
    bool target_range_enforced=false;
    Joints target_abs_velocity{};
    bool valid=false;
  };
  struct Command {
    Lease lease;
    uint64_t generated_ns=0, source_sequence=0, ingress=0;
    Joints velocity{};
    bool valid=false, stop=false;
  };
  Core core;
  Lease active;
  uint64_t session=0, retired_through=0, rejected_leases=0, source_sequence=0;
  uint64_t max_lease_ns=100000000, source_timeout_ns=100000000;

  LeasedVelocityCore() { core.velocity_limits.fill(2.); }
  void start(uint64_t now_ns, uint64_t new_session) {
    if (!new_session || !max_lease_ns || !source_timeout_ns)
      throw std::invalid_argument("invalid lease session/timing");
    session=new_session; started_=last_now_=now_ns;
    retired_through=rejected_leases=source_sequence=0;
    grant_ingress_=command_ingress_=grant_sequence_=last_source_sequence_=0;
    active=Lease(); pending_=Lease(); rearm_lease_=Lease(); forwarded_=Core::Command();
    core.start(now_ns*1e-9);
  }
  void update(uint64_t now_ns, double period, const Lease& grant,
              const Command& command, uint64_t revoked_through,
              bool rearm, const Joints& measured) {
    const double now=now_ns*1e-9;
    const bool clock_bad=now_ns<last_now_ || !std::isfinite(period) || period<0.;
    last_now_=now_ns;
    if (clock_bad) { core.clock_fault=true; retire(now); }
    // Check OLD authorization and source timeout BEFORE processing replacements.
    if (active.token && core.mode==Core::TRACKING &&
        (now_ns>=active.until_ns || now_ns<accepted_generated_ ||
         now_ns-accepted_generated_>=source_timeout_ns)) retire(now);
    if (revoked_through) {
      retired_through=std::max(retired_through,revoked_through);
      if (active.token && active.token<=retired_through) retire(now);
      if (pending_.token<=retired_through) pending_=Lease();
    }
    if (grant.ingress && grant.ingress!=grant_ingress_) {
      grant_ingress_=grant.ingress;
      if (leaseValid(grant,now_ns) && grant.sequence>grant_sequence_ &&
          (!active.token || grant.token>=active.token)) {
        pending_=grant; grant_sequence_=grant.sequence;
      } else { ++rejected_leases; }
    }
    if (command.ingress && command.ingress!=command_ingress_) {
      command_ingress_=command.ingress;
      const Lease* selected=nullptr;
      if (same(command.lease,pending_)) selected=&pending_;
      else if (same(command.lease,active)) selected=&active;
      bool valid=selected && command.valid && !core.clock_fault &&
          command.source_sequence>last_source_sequence_ &&
          command.generated_ns>started_ && command.generated_ns<=now_ns &&
          now_ns-command.generated_ns<source_timeout_ns;
      for (size_t j=0;j<7;++j)
        valid=valid && std::isfinite(command.velocity[j]) &&
            std::abs(command.velocity[j])<=core.velocity_limits[j] &&
            (!selected || !selected->target_range_enforced ||
             std::abs(command.velocity[j])<=selected->target_abs_velocity[j]);
      if (valid && command.stop && active.token==command.lease.token) {
        last_source_sequence_=command.source_sequence; retire(now);
      } else if (valid && !command.stop && leaseValid(*selected,now_ns) &&
                 command.generated_ns>=selected->issued_ns) {
        last_source_sequence_=command.source_sequence;
        forwarded_.valid=true; forwarded_.stamp=command.generated_ns*1e-9;
        forwarded_.sequence=static_cast<uint32_t>(command.source_sequence);
        forwarded_.ingress=command.ingress; forwarded_.velocity=command.velocity;
        rearm_lease_=*selected; rearm_source_=command.source_sequence;
        rearm_generated_=command.generated_ns;
      } else { ++core.rejected; }
    }
    const auto prior_successes=core.rearm_successes;
    const bool rearm_valid=leaseValid(rearm_lease_,now_ns) &&
        rearm_lease_.token>retired_through;
    if (rearm && !rearm_valid) ++core.rearm_rejections;
    core.update(now,clock_bad ? -1. : period,forwarded_,rearm && rearm_valid,measured);
    if (core.mode==Core::TRACKING && forwarded_.valid &&
        (core.command_ingress==forwarded_.ingress || core.rearm_successes>prior_successes)) {
      active=rearm_lease_; source_sequence=rearm_source_; accepted_generated_=rearm_generated_;
    }
    if (active.token && (core.mode==Core::BRAKING || core.mode==Core::LATCHED))
      retired_through=std::max(retired_through,active.token);
  }
  static bool same(const Lease& a,const Lease& b) {
    return a.token && a.session==b.session && a.token==b.token &&
        a.sequence==b.sequence && a.until_ns==b.until_ns && a.certificate==b.certificate &&
        a.target_range_enforced==b.target_range_enforced && a.target_abs_velocity==b.target_abs_velocity;
  }
  uint64_t generationNs() const { return accepted_generated_; }
 private:
  uint64_t started_=0,last_now_=0,grant_ingress_=0,command_ingress_=0;
  uint64_t grant_sequence_=0,last_source_sequence_=0,accepted_generated_=0;
  uint64_t rearm_source_=0,rearm_generated_=0;
  Lease pending_,rearm_lease_;
  Core::Command forwarded_;
  bool leaseValid(const Lease& l,uint64_t now) const {
    for (size_t j=0;j<7;++j)
      if (!std::isfinite(l.target_abs_velocity[j]) || l.target_abs_velocity[j]<0. ||
          l.target_abs_velocity[j]>core.velocity_limits[j]) return false;
    return !core.clock_fault && l.valid && l.session==session &&
        l.token>retired_through && l.sequence && l.issued_ns>started_ &&
        l.issued_ns<=now && l.until_ns>now && l.until_ns>l.issued_ns &&
        l.until_ns-l.issued_ns<=max_lease_ns && l.certificate.size()==64 &&
        l.certificate.find_first_not_of("0123456789abcdef")==std::string::npos;
  }
  void retire(double now) {
    if (active.token) retired_through=std::max(retired_through,active.token);
    core.requestStop(now);
  }
};
}  // namespace care_velocity_controller
