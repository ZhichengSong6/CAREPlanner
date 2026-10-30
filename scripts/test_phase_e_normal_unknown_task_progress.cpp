#include "egocentric_arm_planner/normal_unknown_task_progress.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <limits>
using Guard=egocentric_arm_planner::NormalUnknownTaskProgress;
int main(int argc,char**) {
  if (argc>1) {
    Guard g;char kind;std::cout<<std::setprecision(12);
    while (std::cin>>kind) {
      unsigned long long raw,query,execution;std::string context;double t,p,a;int eligible;
      if (kind=='R') {std::cin>>raw>>query>>context;g.remember(raw,query,context,true);}
      if (kind=='C') {std::cin>>raw>>execution>>context;g.certify(raw,execution,context);}
      if (kind=='O') {
        std::cin>>t>>execution>>eligible>>context>>p>>a;
        if (context=="none")context.clear();
        if (g.observe(t,execution,p,a,context,eligible))
          std::cout<<t<<' '<<g.execution_stamps<<' '<<g.last.query<<'\n';
      }
    }
    return 0;
  }
  int passed=0;
  for (int mode=0;mode<14;++mode) {
    Guard g;bool fired=false;
    for (int i=0;i<65;++i) {
      double t=10.+i*.1;
      const auto raw=static_cast<unsigned long long>((10.+(i/8)*.8)*1e9);
      const auto execution=mode==5?10010000000ULL:raw+10000000ULL;
      if (i%8==0) {
        g.remember(raw,raw-1,"A",mode!=1);
        if (mode!=2)g.certify(raw,execution,mode==3?"wrong":"A");
      }
      double p=.5,a=.8;
      if(mode==4)p-=i*.004;
      if(mode==6){p=.005;a=1.4-i*.015;}
      if(mode==7){p=.01;a=.1;}
      if(mode==8)p=std::numeric_limits<double>::quiet_NaN();
      if(mode==9)p=i%10<5?.5:.52;
      if(mode==10)t+=3.;
      if(mode==11 && i%15==0)g.resetWindow("safety_hold");
      if(mode==12 && i%15==0)g.reset("goal_changed");
      const std::string context=mode==13?std::to_string(i/15):"A";
      fired |= g.observe(t+.06,execution,p,a,context,mode!=11 || i%15!=0);
    }
    const bool expected=mode==0 || mode==9;
    assert(fired==expected);++passed;
  }
  Guard g;assert(g.remember(10000000000ULL,9,"A",true));
  assert(!g.remember(10000000000ULL,10,"A",true));
  assert(g.certify(10000000000ULL,10010000000ULL,"A"));
  assert(!g.certify(10000000000ULL,10020000000ULL,"A"));
  assert(g.certified.size()==1);++passed;
  assert(!g.observe(10.1,10010000000ULL,.5,.8,"A",true));
  assert(!g.observe(12.01,10010000000ULL,.5,.8,"A",true));
  assert(g.reason=="execution_evidence_missing_or_stale" && g.first_time_s<0.);++passed;
  std::cout<<"PASS "<<passed<<" production task-progress controls\n";
}
