#include <care_velocity_controller/leased_velocity_core.hpp>
#include <gtest/gtest.h>
#include <limits>

using Policy=care_velocity_controller::LeasedVelocityCore;
class LeaseTest : public ::testing::Test {
 protected:
  Policy p;
  Policy::Lease grant;
  Policy::Command command;
  Policy::Joints measured{};
  uint64_t now=1000000000,seq=0,ingress=0,revoke=0;
  void SetUp() override { p.start(now,71); }
  void tick(uint64_t ns=1000000,bool rearm=false) {
    now+=ns;p.update(now,ns*1e-9,grant,command,revoke,rearm,measured);
  }
  void lease(uint64_t token=100,uint64_t duration=100000000) {
    grant.valid=true;grant.session=71;grant.token=token;grant.sequence++;
    grant.issued_ns=now+1000000;grant.until_ns=grant.issued_ns+duration;
    grant.certificate=std::string(64,'a');grant.ingress++;
  }
  void send(double velocity=.6,bool stop=false) {
    command.lease=grant;command.generated_ns=now+1000000;command.source_sequence=++seq;
    command.ingress=++ingress;command.valid=true;command.stop=stop;command.velocity.fill(velocity);
  }
  void startMotion() { lease();send();tick();ASSERT_EQ(p.core.mode,Policy::Core::TRACKING); }
  void settle() { for(int i=0;i<1000;++i)tick();ASSERT_EQ(p.core.mode,Policy::Core::LATCHED); }
};
TEST_F(LeaseTest, FreshCommandsCannotExtendFixedDeadline) {
  startMotion();const auto until=grant.until_ns;
  while(now<until) {send();tick();}
  EXPECT_NE(p.core.mode,Policy::Core::TRACKING);
  EXPECT_EQ(p.active.until_ns,until);EXPECT_EQ(p.retired_through,100u);
  settle();EXPECT_DOUBLE_EQ(p.core.output[0],0.);
}
TEST_F(LeaseTest, OriginalGenerationTimeCannotBeRefreshedByReceipt) {
  startMotion();command.ingress++;command.source_sequence++;command.generated_ns=now-200000000;
  tick();EXPECT_EQ(p.core.rejected,1u);EXPECT_EQ(p.source_sequence,1u);
}
TEST_F(LeaseTest, UnknownLeaseAndWrongSessionNeverStart) {
  lease();send();command.lease.sequence++;tick();EXPECT_EQ(p.core.mode,Policy::Core::WAITING);
  lease();grant.session=72;send();tick();EXPECT_EQ(p.core.mode,Policy::Core::WAITING);
}
TEST_F(LeaseTest, EachIdentityFieldAndReplayIsChecked) {
  startMotion();const auto accepted=p.source_sequence;
  for(int i=0;i<6;++i) {
    send();
    if(i==0)command.lease.token++;
    if(i==1)command.lease.certificate[0]='b';
    if(i==2)command.lease.until_ns++;
    if(i==3)command.source_sequence=accepted;
    if(i==4)command.generated_ns=now+2000000;
    if(i==5)command.velocity[2]=2.5;
    tick();EXPECT_EQ(p.source_sequence,accepted);
  }
  EXPECT_EQ(p.core.rejected,6u);
}
TEST_F(LeaseTest, EarlierPendingLeaseDoesNotReplaceAppliedDeadline) {
  startMotion();const auto old=p.active;
  lease(101,1000000);tick();tick(3000000);
  EXPECT_EQ(p.active.token,old.token);EXPECT_EQ(p.active.until_ns,old.until_ns);
}
TEST_F(LeaseTest, RenewOnlyBeforeExpiryAndOnlyOnMatchingCommand) {
  startMotion();const auto old=p.active.until_ns;
  lease();tick();EXPECT_EQ(p.active.until_ns,old);
  send();tick();EXPECT_GT(p.active.until_ns,old);
  const auto end=p.active.until_ns;
  now=end;lease();send();tick();EXPECT_NE(p.core.mode,Policy::Core::TRACKING);
  EXPECT_EQ(p.retired_through,100u);
}
TEST_F(LeaseTest, ExpiryAtNanosecondBoundaryPrecedesFreshGrant) {
  startMotion();now=p.active.until_ns-1;
  lease(101);send();tick(1);
  EXPECT_NE(p.core.mode,Policy::Core::TRACKING);EXPECT_EQ(p.active.token,100u);
}
TEST_F(LeaseTest, NewLeaseWithDuplicateGenerationCannotClaimApplication) {
  startMotion();const auto old=p.active;const auto generation=p.generationNs();
  lease();send();command.generated_ns=generation;tick();
  EXPECT_EQ(p.active.sequence,old.sequence);EXPECT_EQ(p.active.until_ns,old.until_ns);
  EXPECT_EQ(p.source_sequence,1u);
}
TEST_F(LeaseTest, StickyRevocationSurvivesReplacementCommand) {
  startMotion();revoke=100;lease(101);send();tick();
  EXPECT_NE(p.core.mode,Policy::Core::TRACKING);EXPECT_EQ(p.active.token,100u);
}
TEST_F(LeaseTest, TerminalStopRetiresTokenAndRetainsApplicationIdentity) {
  startMotion();send(0.,true);tick();
  EXPECT_EQ(p.active.token,100u);EXPECT_EQ(p.retired_through,100u);
  lease();send();tick();EXPECT_NE(p.core.mode,Policy::Core::TRACKING);
  EXPECT_GT(p.rejected_leases,0u);
}
TEST_F(LeaseTest, RearmRequiresNewTokenPostStopZeroAndExplicitRequest) {
  startMotion();settle();
  lease();send(0.);tick(1000000,true);EXPECT_EQ(p.core.rearm_successes,0u);
  lease(101);send(.6);tick(1000000,true);EXPECT_EQ(p.core.rearm_successes,0u);
  lease(101);send(0.);tick();EXPECT_EQ(p.core.mode,Policy::Core::LATCHED);
  tick(1000000,true);EXPECT_EQ(p.core.mode,Policy::Core::TRACKING);
  EXPECT_EQ(p.active.token,101u);EXPECT_EQ(p.core.rearm_successes,1u);
  EXPECT_DOUBLE_EQ(p.core.output[0],0.);
  send(.6);tick();EXPECT_GT(p.core.output[0],0.);
}
TEST_F(LeaseTest, RestartRejectsPreviousSession) {
  startMotion();p.start(now,72);send();tick();
  EXPECT_EQ(p.core.mode,Policy::Core::WAITING);EXPECT_DOUBLE_EQ(p.core.output[0],0.);
}
TEST_F(LeaseTest, ClockRollbackLatchesFault) {
  startMotion();now-=10000000;tick();EXPECT_TRUE(p.core.clock_fault);
  settle();lease(101);send(0.);tick(1000000,true);EXPECT_EQ(p.core.rearm_successes,0u);
}
TEST_F(LeaseTest, BoundTargetsRejectDormantAxisMotionAndForgedRange) {
  lease();grant.target_range_enforced=true;grant.target_abs_velocity[0]=.4;
  send(0.);command.velocity[0]=.3;tick();
  ASSERT_EQ(p.core.mode,Policy::Core::TRACKING);
  const auto accepted=p.source_sequence;
  send(0.);command.velocity[1]=.01;tick();
  EXPECT_EQ(p.source_sequence,accepted);
  send(0.);command.lease.target_abs_velocity[1]=.1;command.velocity[1]=.01;tick();
  EXPECT_EQ(p.source_sequence,accepted);
  send(0.);command.lease.target_range_enforced=false;tick();
  EXPECT_EQ(p.source_sequence,accepted);
  EXPECT_DOUBLE_EQ(p.core.output[1],0.);
}
TEST_F(LeaseTest, MalformedRangesNeverBecomePendingRights) {
  for(double cap : {-.1,2.1,std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::infinity()}) {
    lease();grant.target_range_enforced=true;grant.target_abs_velocity[0]=cap;
    send(0.);tick();EXPECT_EQ(p.core.mode,Policy::Core::WAITING);
  }
  EXPECT_EQ(p.rejected_leases,4u);
}
TEST_F(LeaseTest, SmallerRangeRetainsOutputAndAppliedGrantUntilMatchingCommand) {
  startMotion();for(int i=0;i<20;++i){send();tick();}
  const auto old=p.active;const double output=p.core.output[0];
  lease();grant.target_range_enforced=true;grant.target_abs_velocity.fill(0.);
  tick();EXPECT_EQ(p.active.sequence,old.sequence);
  send(0.);const double before=p.core.output[0];tick();
  EXPECT_TRUE(p.active.target_range_enforced);EXPECT_DOUBLE_EQ(p.active.target_abs_velocity[0],0.);
  EXPECT_GT(p.core.output[0],0.);EXPECT_GT(output,0.);
  EXPECT_NEAR(before-p.core.output[0],.003,1e-12);
}
TEST_F(LeaseTest, ZeroRangeAllowsFirstZeroAndExplicitPostStopRearm) {
  lease();grant.target_range_enforced=true;send(0.);tick();
  ASSERT_EQ(p.core.mode,Policy::Core::TRACKING);settle();
  lease(101);send(0.);tick();tick(1000000,true);
  EXPECT_EQ(p.core.mode,Policy::Core::TRACKING);EXPECT_EQ(p.active.token,101u);
  send(.1);tick();EXPECT_DOUBLE_EQ(p.core.output[0],0.);
}
TEST_F(LeaseTest, BoundGrantStillExpiresWithFreshLegalTargets) {
  lease();grant.target_range_enforced=true;grant.target_abs_velocity.fill(.4);
  send(.4);tick();ASSERT_EQ(p.core.mode,Policy::Core::TRACKING);
  while(now<grant.until_ns){send(.4);tick();}
  EXPECT_NE(p.core.mode,Policy::Core::TRACKING);settle();
  EXPECT_EQ(p.retired_through,100u);
}
int main(int argc,char** argv) {::testing::InitGoogleTest(&argc,argv);return RUN_ALL_TESTS();}
