#include <care_velocity_controller/expiring_velocity_core.hpp>
#include <gtest/gtest.h>
#include <limits>

using Core=care_velocity_controller::ExpiringVelocityCore;
int main(int argc,char** argv) { ::testing::InitGoogleTest(&argc,argv); return RUN_ALL_TESTS(); }
class Watchdog : public ::testing::Test {
 protected:
  Core c;
  Core::Command command;
  Core::Joints measured{};
  double now=1.;
  void SetUp() override { c.start(now); }
  void tick(double dt=.001,bool rearm=false) { now+=dt; c.update(now,dt,command,rearm,measured); }
  void send(double speed,double age=0.) {
    command.valid=true; command.stamp=now-age; command.velocity.fill(speed); ++command.ingress;
  }
  void moving() {
    for(int i=0;i<400;++i) { now+=.001; send(.6); c.update(now,.001,command,false,measured); }
    ASSERT_NEAR(c.output[0],.6,1e-12);
  }
  void expire() { for(int i=0;i<400;++i) tick(); ASSERT_EQ(c.mode,Core::LATCHED); }
};

TEST_F(Watchdog, StopsWithoutAnyProducerCallbacksAndLatches) {
  moving(); double stamp=command.stamp, previous=.6;
  for(int i=0;i<400;++i) {
    tick(); EXPECT_LE(c.output[0],previous+1e-12);
    EXPECT_LE(std::abs(c.output[0]-previous),.003+1e-12); previous=c.output[0];
  }
  EXPECT_EQ(c.mode,Core::LATCHED); EXPECT_DOUBLE_EQ(c.output[0],0.);
  EXPECT_LE(c.stopped_stamp-stamp,.1+.6/3.+.003);
}
TEST_F(Watchdog, LateFreshReplacementCannotConcealExpiredLease) {
  moving(); now+=.101; send(.6); c.update(now,.101,command,false,measured);
  EXPECT_EQ(c.mode,Core::BRAKING); EXPECT_LT(c.output[0],.6);
}
TEST_F(Watchdog, ReplayedFutureMalformedAndNonfiniteDoNotRenew) {
  moving(); const double active=c.command_stamp;
  for(int k=0;k<5;++k) {
    tick(); send(.6);
    if(k==0) command.stamp=active;
    if(k==1) command.stamp=now+1.;
    if(k==2) command.valid=false;
    if(k==3) command.velocity[2]=std::numeric_limits<double>::quiet_NaN();
    if(k==4) command.velocity[0]=2.1;
    tick(); EXPECT_DOUBLE_EQ(c.command_stamp,active);
  }
  EXPECT_EQ(c.rejected,5u); expire();
}
TEST_F(Watchdog, ReceiptOfOldPacketCannotStartMotion) {
  now+=1.; send(.6,.2); tick(); EXPECT_EQ(c.mode,Core::WAITING);
  EXPECT_DOUBLE_EQ(c.output[0],0.); EXPECT_EQ(c.rejected,1u);
}
TEST_F(Watchdog, FreshNonzeroCannotResumeAfterFault) {
  moving(); expire();
  for(int i=0;i<200;++i) { send(.6); tick(); EXPECT_EQ(c.mode,Core::LATCHED); }
  tick(.001,true); EXPECT_EQ(c.rearm_successes,0u); EXPECT_EQ(c.rearm_rejections,1u);
}
TEST_F(Watchdog, RearmRequiresFreshPostStopZeroAndMeasuredRest) {
  moving(); expire(); tick(.001,true); EXPECT_EQ(c.rearm_rejections,1u);
  send(0.); measured[0]=.01; tick(.001,true); EXPECT_EQ(c.rearm_rejections,2u);
  measured.fill(0.); send(0.); tick(.001,true);
  EXPECT_EQ(c.rearm_successes,1u); EXPECT_EQ(c.mode,Core::TRACKING);
  const double rearmed=now;
  command.velocity.fill(.6); command.stamp=rearmed-.0005; ++command.ingress; tick();
  EXPECT_DOUBLE_EQ(c.output[0],0.);
  send(.6); tick(); EXPECT_GT(c.output[0],0.);
}
TEST_F(Watchdog, RearmWhileBrakingIsNotQueuedUntilLater) {
  moving(); for(int i=0;i<110;++i) tick(); ASSERT_EQ(c.mode,Core::BRAKING);
  send(0.); tick(.001,true); EXPECT_EQ(c.rearm_rejections,1u);
  expire(); EXPECT_EQ(c.rearm_successes,0u);
}
TEST_F(Watchdog, ClockRollbackBrakesAndRequiresRestart) {
  moving(); now-=.5; c.update(now,.001,command,false,measured);
  EXPECT_TRUE(c.clock_fault); expire(); send(0.); tick(.001,true);
  EXPECT_EQ(c.mode,Core::LATCHED); EXPECT_EQ(c.rearm_successes,0u);
}
TEST_F(Watchdog, SevenAxesBothSignsAndSchedulingGapsRespectLimits) {
  for(double sign : {-1.,1.}) for(double gap : {.001,.01,.05}) {
    now=1.; c.start(now); command=Core::Command();
    for(int i=0;i<3000;++i) {
      now+=.001; send(0.);
      for(size_t j=0;j<7;++j) command.velocity[j]=sign*c.velocity_limits[j];
      c.update(now,.001,command,false,measured);
    }
    auto previous=c.output;
    for(int i=0;i<3000;++i) {
      tick(gap);
      for(size_t j=0;j<7;++j) {
        EXPECT_LE(std::abs(c.output[j]),std::abs(previous[j])+1e-12);
        EXPECT_LE(std::abs(c.output[j]-previous[j]),c.acceleration_limits[j]*gap+1e-11);
      }
      previous=c.output;
    }
    EXPECT_EQ(c.mode,Core::LATCHED);
  }
}
