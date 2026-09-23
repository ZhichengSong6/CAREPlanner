#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
using namespace egocentric_arm_planner;
using V = Eigen::VectorXd;

int main(int argc, char** argv) {
  if (argc == 24 && std::string(argv[1]) == "--target") {
    V q(7), target(7), normal(7);
    for (int j=0;j<7;++j) { q[j]=std::stod(argv[3+j]); target[j]=std::stod(argv[10+j]); normal[j]=std::stod(argv[17+j]); }
    const V result=boundedTangentTarget(q,target,{normal},std::stoi(argv[2]),.03);
    std::cout << std::setprecision(17) << '[';
    for(int j=0;j<result.size();++j) std::cout << (j?",":"") << result[j];
    std::cout << "]\n"; return 0;
  }
  V zero=V::Zero(7), q=zero;
  SafeFrontierRecovery s; s.select("A",1);
  assert(!s.certify(10,11) && !s.observe(11,1.,zero)); // no raw/commit evidence
  auto sample=[&](unsigned long long raw,double t,double span,const V& measured) {
    s.remember(raw,span); assert(s.certify(raw,raw+1));
    return s.observe(raw+1,t,measured);
  };
  for(int i=0;i<5;++i) assert(!sample(100+10*i,1.+i*.25,.0001,zero));
  assert(sample(150,2.25,.0001,zero)); assert(s.stage()==1);
  assert(!s.observe(151,3.,zero)); // exact execution deduplication
  assert(!s.certify(100,1000)); // consumed raw cannot certify again
  assert(!s.reject(9999)); // unrelated rejection has no authority
  s.remember(200,.02); assert(!s.reject(200));
  s.remember(210,.02); assert(s.reject(210)); assert(s.stage()==2);
  s.remember(220,.001); s.select("B",1); assert(!s.certify(220,221));
  assert(s.stage()==0 && s.attempts==0); // identity change invalidates pending
  s.remember(230,.001); s.select("B",2); assert(!s.certify(230,231));
  for(int i=0;i<10;++i) assert(!sample(300+i*10,3.+i*.01,.0001,zero));
  assert(s.stage()==0); // count alone, before minimum time, is not stagnation
  assert(sample(410,4.,.05,zero)); assert(s.stage()==1); // raw span cannot exempt real stagnation
  s.select("moving",2);
  for(int i=0;i<10;++i) { q[0]=i*.002; assert(!sample(500+i*10,5.+i*.25,.0001,q)); }
  assert(s.stage()==0); // real measured movement prevents false positive
  s.select("C",3);
  for(int stage=1;stage<=6;++stage) {
    for(int i=0;i<6;++i) sample(1000+stage*100+i*10,stage*2.+i*.25,.0001,zero);
    assert(s.attempts==stage);
  }
  assert(s.blocked());
  for(int i=0;i<10;++i) sample(2000+i*10,20.+i,.0001,zero);
  assert(s.attempts==6); // finite attempts, successful holds do not replenish
  s.select("D",4); s.advance();
  sample(3000,30.,.04,zero); q=zero;q[0]=.02;
  assert(sample(3010,30.2,.04,q)); assert(s.active && s.attempts==1 && s.segments==1);
  assert(s.target.size()==0); // regenerate local target, retain recovery
  s.remember(3012,.02); s.invalidateMotion();
  assert(!s.certify(3012,3013) && s.attempts==1 && s.segments==1);
  s.select("child",4); s.select("D",4);
  assert(s.active && s.attempts==1 && s.segments==1); // preemption is not a reset
  sample(3020,30.3,.04,q);
  for(int i=0;i<5;++i) {
    q[0]+=.02; assert(sample(3030+i*10,30.4+i,.04,q));
  }
  assert(s.active && s.attempts==2 && s.segments==0); // finite continuation
  s.following_frontier = true;
  q[0]+=.02; assert(sample(3090,39.,.04,q));
  assert(s.active && s.attempts==2 && s.segments==0); // rejoin is not completion or a reset
  while(!s.blocked()) s.advance();
  q[0]+=.02; sample(3100,40.,.04,q);
  assert(s.blocked()); // motion cannot reopen exhausted recovery
  s.select("E",5); s.remember(4000,.001); assert(s.certify(4000,4001));
  q[0]=std::numeric_limits<double>::quiet_NaN();
  assert(!s.observe(4001,31.,q)); assert(s.tiny_commits==0);
  for(int i=0;i<100;++i) s.remember(5000+i,.001);
  assert(s.pending.size()==64);
  for(int i=0;i<100;++i) s.select("region"+std::to_string(i),5);
  assert(s.budgets.size()<=64 && s.blocked());
  const Eigen::Vector3d p(.1,.2,.3);
  const std::vector<double> points{.1,.2,.3,.1,.2,.3};
  assert(resolvedFreePoint({"resolved_free","resolved_free"},points,p));
  assert(!resolvedFreePoint({"resolved_free","evaluated_unknown"},points,p));
  assert(!resolvedFreePoint({"resolved_free","evaluated_occupied"},points,p));
  assert(!resolvedFreePoint({"resolved_free","service_error"},points,p));
  assert(!resolvedFreePoint({"resolved_free"},points,p));
  assert(!resolvedFreePoint({}, {}, p));
  assert(!resolvedFreePoint({"resolved_free","resolved_free"},points,Eigen::Vector3d::Zero()));

  V normal(7); normal << .07582505,.97494304,.05187582,.48323214,.01985239,.00074879,.02837613;
  const V goal=-.05*normal/normal.lpNorm<Eigen::Infinity>();
  std::vector<V> targets;
  for(int stage=1;stage<=5;++stage) {
    V t=boundedTangentTarget(zero,goal,{normal},stage,.03);
    assert(t.size()==7 && t.allFinite());
    assert(std::abs(t.lpNorm<Eigen::Infinity>()-.03)<1e-10);
    assert(std::abs(normal.dot(t))<1e-8);
    for(const auto& old:targets) assert((t-old).norm()>.001);
    targets.push_back(t);
  }
  assert((targets[0]+targets[1]).norm()<1e-8);
  std::vector<V> full;
  for(int i=0;i<7;++i) full.push_back(V::Unit(7,i));
  assert(boundedTangentTarget(zero,goal,full,1,.03).size()==0);
  assert(boundedTangentTarget(zero,goal,{},1,.03).size()==0);
  assert(boundedTangentTarget(zero,goal,{normal},6,.03).size()==0);
  std::cout << "PASS safe frontier identity, progress, bounded attempts and tangent tests\n";
}
