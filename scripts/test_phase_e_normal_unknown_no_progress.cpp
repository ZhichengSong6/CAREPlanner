#include "egocentric_arm_planner/normal_unknown_no_progress.hpp"
#include <cassert>
#include <iostream>
using egocentric_arm_planner::NormalUnknownNoProgress;
int main(int argc, char**) {
  if (argc > 1) {
    // Recorded-event replay: R raw query span qualified; C raw execution;
    // O wall/ROS seconds execution eligible q[7]. Same production policy.
    NormalUnknownNoProgress g;
    char kind;
    while (std::cin >> kind) {
      unsigned long long a,b; double t,span; int eligible;
      if (kind=='R') { std::cin>>a>>b>>span>>eligible; g.remember(a,b,"recorded",span,eligible); }
      if (kind=='C') { std::cin>>a>>b; g.certify(a,b,"recorded"); }
      if (kind=='O') {
        Eigen::VectorXd q(7); std::cin>>t>>a>>eligible;
        for(int j=0;j<7;++j) std::cin>>q[j];
        if(g.observe(t,a,q,"recorded",eligible)) std::cout<<t<<' '<<g.progress.execution_stamps<<'\n';
      }
    }
    return 0;
  }
  Eigen::VectorXd q=Eigen::VectorXd::Zero(7);
  for (int mode=0;mode<9;++mode) {
    NormalUnknownNoProgress g; bool fired=false;
    for(int i=0;i<60;++i) {
      double t=10.+i*.1;
      auto raw=static_cast<unsigned long long>((10.+(i/8)*.8)*1e9);
      auto exec=raw+10000000ULL;
      if(i%8==0) {
        g.remember(raw,raw-1,"A",mode==2?.02:.001,mode!=1);
        if(mode!=3) g.certify(raw,exec,mode==4?"B":"A");
      }
      q[0]=mode==5?i*.002:0.;
      bool eligible=mode!=6;
      const auto context=mode==7?std::to_string(i):"A";
      fired |= g.observe(t+.06,mode==8?99:exec,q,context,eligible);
    }
    assert(fired==(mode==0));
  }
  NormalUnknownNoProgress g;
  g.remember(10000000000ULL,1,"A",.001,true);
  g.certify(10000000000ULL,10010000000ULL,"A");
  g.certify(10000000000ULL,10020000000ULL,"A");
  assert(g.certified.size()==1);
  assert(!g.observe(10.1,10010000000ULL,q,"A",true));
  assert(!g.observe(13.,10010000000ULL,q,"A",true));
  assert(g.progress.execution_stamps==0);
  g.reset(); assert(g.pending.empty()&&g.certified.empty());
  std::cout << "PASS: positive, missing UNKNOWN, large candidate, uncertified, wrong context, motion, ineligible, changed context, wrong execution, duplicate, stale, reset\n";
}
