// Thin C ABI fixture around the ACTUAL production core. Ideal held output only.
#include <care_velocity_controller/leased_velocity_core.hpp>
#include <cstdint>
using care_velocity_controller::LeasedVelocityCore;
struct FixtureCore {
  LeasedVelocityCore value;
  LeasedVelocityCore::Joints q{};
  uint64_t last=0;
};
extern "C" {
void* ledger_core_create(uint64_t start,uint64_t session,const double* q) {
  auto* f=new FixtureCore;
  for(size_t j=0;j<7;++j)f->q[j]=q[j];
  f->value.start(start,session);f->last=start;return f;
}
void ledger_core_destroy(void* raw) {delete static_cast<FixtureCore*>(raw);}
void ledger_core_step(void* raw,uint64_t now,const uint64_t* packet,
                      const double* cap,const double* target,const char* certificate,
                      unsigned flags,uint64_t revoked) {
  auto& f=*static_cast<FixtureCore*>(raw);
  const double dt=(static_cast<double>(now)-static_cast<double>(f.last))*1e-9;
  for(size_t j=0;j<7;++j)f.q[j]+=f.value.core.output[j]*dt;
  LeasedVelocityCore::Lease g;
  g.session=packet[0];g.token=packet[1];g.sequence=packet[2];
  g.issued_ns=packet[3];g.until_ns=packet[4];g.certificate=certificate;
  g.ingress=(flags&1)?packet[2]:0;g.valid=flags&1;g.target_range_enforced=true;
  for(size_t j=0;j<7;++j)g.target_abs_velocity[j]=cap[j];
  LeasedVelocityCore::Command c;
  c.lease=g;c.generated_ns=packet[5];c.source_sequence=packet[6];
  c.ingress=(flags&2)?packet[6]:0;c.valid=flags&2;c.stop=flags&8;
  for(size_t j=0;j<7;++j)c.velocity[j]=target[j];
  f.value.update(now,dt,g,c,revoked,flags&4,f.value.core.output);f.last=now;
}
void ledger_core_read(void* raw,uint64_t* ints,double* values) {
  auto& f=*static_cast<FixtureCore*>(raw);const auto& g=f.value.active;
  ints[0]=f.last;ints[1]=f.value.session;ints[2]=f.value.core.mode;
  ints[3]=g.token;ints[4]=g.sequence;ints[5]=g.until_ns;
  ints[6]=f.value.core.clock_fault;ints[7]=f.value.core.rejected;
  ints[8]=f.value.rejected_leases;ints[9]=f.value.core.rearm_successes;
  for(size_t j=0;j<7;++j) {
    values[j]=f.value.core.output[j];values[j+7]=f.q[j];
    values[j+14]=g.target_abs_velocity[j];
  }
}
}
