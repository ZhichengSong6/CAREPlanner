#include "egocentric_arm_planner/vbc_failure_direction_memory.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
using M=egocentric_arm_planner::VbcFailureDirectionMemory;
using Q=Eigen::VectorXd;
M::Context context() {
  M::Context c; c.enabled=true;c.mode=2;c.reference=1;c.owner=11;c.token="care_obs_v1_11_old";
  c.target=Q::Constant(7,.3);c.mask=Q::Ones(7);c.points={{.05,-.25,.8},{.05,-.25,.85}};return c;
}
M::Points failures={{.05,-.25,.8},{.1,-.25,.8}};
std::uint64_t ns(double t) {return std::llround(t*1e9);}
struct Fixture {
  M m;M::Context old=context(),grown=old;Q q=Q::Zero(7);
  Fixture() {
    grown.token="care_obs_v1_11_new";grown.points.push_back({0.,-.25,.75});
    m.observeMeasured(q,1.);
    assert(m.remember(ns(1.01),ns(1.),old));
    assert(m.reject(ns(1.01),ns(1.02),ns(1.02),ns(1.),old,failures,q,1.03).repeated.empty());
  }
  void tick() {m.observeMeasured(q,1.1);}
  M::Match reject(M::Context c) {
    tick();m.remember(ns(1.11),ns(1.1),c);
    return m.reject(ns(1.11),ns(1.12),ns(1.12),ns(1.1),c,failures,q,1.13);
  }
};
int main(int argc,char** argv) {
  if (argc>1 && std::string(argv[1])=="--replay") {
    M m;auto c=context();Q q=Q::Zero(7);std::string line;
    while(std::getline(std::cin,line)) {
      std::istringstream s(line);char kind;s>>kind;
      if(kind=='C') {
        int count;s>>c.token>>count;c.points.clear();
        for(int i=0;i<count;++i) {Eigen::Vector3d p;s>>p[0]>>p[1]>>p[2];c.points.push_back(p);}
        for(int j=0;j<7;++j)s>>c.target[j];
        for(int j=0;j<7;++j)s>>c.mask[j];
        m.observeContext(c);
      } else if(kind=='M') {double t;s>>t;for(int j=0;j<7;++j)s>>q[j];m.observeMeasured(q,t);}
      else if(kind=='R') {std::uint64_t raw,query;s>>raw>>query;assert(m.remember(raw,query,c));}
      else if(kind=='F') {
        double now;std::uint64_t raw,audited,query;int n;s>>now>>raw>>audited>>query>>n;
        M::Points p;for(int i=0;i<n;++i){Eigen::Vector3d v;s>>v[0]>>v[1]>>v[2];p.push_back(v);}
        const auto match=m.reject(raw,audited,audited,query,c,p,q,now);
        if(!match.repeated.empty())std::cout<<raw<<" "<<match.previous_raw<<" "<<match.repeated.size()<<"\n";
      } else assert(false);
      assert(!s.fail());
    }
    return 0;
  }
  {Fixture f;auto r=f.reject(f.grown);assert(r.repeated.size()==2 && r.previous_raw==ns(1.01));
    assert(f.m.reject(ns(1.11),ns(1.12),ns(1.12),ns(1.1),f.grown,failures,f.q,1.14).repeated.empty());}
  // Both topic delivery orders preserve history until a current candidate fails.
  for(bool token_first:{false,true}) {
    Fixture f;auto intermediate=f.old;
    if(token_first)intermediate.token=f.grown.token;else intermediate.points=f.grown.points;
    f.m.observeContext(intermediate);f.m.observeContext(f.grown);
    assert(f.reject(f.grown).repeated.size()==2);
  }
  for(int test=0;test<9;++test) {
    Fixture f;auto c=f.grown;
    if(test==0)c.target[0]+=1e-8; // exact target, even below legacy 1e-5 change threshold
    if(test==1)c.mask[0]=0;
    if(test==2)++c.mode;
    if(test==3)++c.reference;
    if(test==4){c.owner=12;c.token="care_obs_v1_12_new";}
    if(test==5)c.points=f.old.points; // token-only change
    if(test==6)c.points.erase(c.points.begin()); // non-superset
    if(test==7)c.enabled=false;
    if(test==8)c.token="care_obs_v1_12_new"; // mismatched owner token
    assert(f.reject(c).repeated.empty());
  }
  {Fixture f;f.q[0]=.005;assert(f.reject(f.grown).repeated.size()==2);}
  {Fixture f;f.q[0]=.005001;assert(f.reject(f.grown).repeated.empty());}
  {Fixture f;Q away=f.q;away[0]=.01;f.m.observeMeasured(away,1.05);assert(f.reject(f.grown).repeated.empty());}
  {Fixture f;auto changed=f.old;changed.target[0]+=.1;f.m.observeContext(changed);f.m.observeContext(f.old);assert(f.reject(f.grown).repeated.empty());}
  {Fixture f;f.m.observeContext(f.grown);f.m.observeContext(f.old);assert(f.reject(f.grown).repeated.empty());}
  {Fixture f;f.m.accepted(ns(1.01),f.old);assert(f.reject(f.grown).repeated.size()==2);} // consumed rejection duplicate
  {Fixture f;f.tick();f.m.remember(ns(1.1),ns(1.09),f.old);f.m.accepted(ns(1.1),f.old);assert(f.reject(f.grown).repeated.empty());}
  {Fixture f;f.tick();f.m.remember(ns(1.11),ns(1.1),f.old);
    assert(f.m.reject(ns(1.11),ns(1.12),ns(1.12),ns(1.1),f.grown,failures,f.q,1.13).repeated.empty());
    assert(f.m.history_raw==ns(1.01));} // old identity cannot write new history
  for(int test=0;test<5;++test) {
    Fixture f;f.tick();f.m.remember(ns(1.11),ns(1.1),f.grown);
    auto points=failures;if(test==4)points={{.9,.9,.9}};
    assert(f.m.reject(ns(1.11),ns(1.12),ns(test==0?1.121:1.12),ns(test==1?1.101:1.1),
        f.grown,points,f.q,test==2?1.7:test==3?1.4:1.13).repeated.empty());
  }
  {Fixture f;for(int i=1;i<23;++i)f.m.observeMeasured(f.q,1.+i*.1);
    f.m.remember(ns(3.21),ns(3.2),f.grown);
    assert(f.m.reject(ns(3.21),ns(3.22),ns(3.22),ns(3.2),f.grown,failures,f.q,3.23).repeated.empty());}
  {Fixture f;f.m.observeMeasured(f.q,1.4);assert(f.reject(f.grown).repeated.empty());}
  {Fixture f;auto c=f.grown;c.points.assign(257,Eigen::Vector3d::Zero());assert(f.reject(c).repeated.empty());}
  {Fixture f;for(int i=0;i<100;++i)f.m.remember(ns(2.+i*.01),ns(1.),f.old);assert(f.m.pending.size()==32);}
  std::cout<<"PASS growth, asynchronous topic orders, identity/freshness, motion excursion, resets, duplicates and bounds\n";
}
