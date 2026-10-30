#include "egocentric_arm_planner/safe_frontier_recovery.hpp"
#include "egocentric_arm_planner/repair_candidate_failure.hpp"
#include <cassert>
#include <iostream>
using egocentric_arm_planner::SafeFrontierRecovery;
using egocentric_arm_planner::RepairCandidateFailure;
int main() {
  RepairCandidateFailure failed;
  failed.record("same_obligation_target", "care_obs_v1_old", 3, 7);
  assert(failed.blocks("same_obligation_target", "care_obs_v1_old", 3, 7));
  assert(!failed.blocks("same_obligation_target", "care_obs_v1_new", 3, 7));
  assert(!failed.blocks("other_target", "care_obs_v1_old", 3, 7));
  assert(!failed.blocks("same_obligation_target", "care_obs_v1_old", 4, 7));
  assert(!failed.blocks("same_obligation_target", "care_obs_v1_old", 3, 8));
  SafeFrontierRecovery bootstrap;
  bootstrap.select("obligation:bootstrap",3);
  assert(!bootstrap.reserveReplacement());
  assert(bootstrap.reserveReplacementAfterVbc(false));
  assert(bootstrap.active && bootstrap.attempts==2);
  SafeFrontierRecovery charged;
  charged.select("obligation:charged",3);
  charged.advance(); charged.remember(7,.03); charged.remember(8,.03);
  assert(!charged.reject(7));
  assert(charged.reject(8) && charged.attempts==2);
  assert(charged.reserveReplacementAfterVbc(true));
  assert(charged.attempts==2); // reject already charged direction two
  SafeFrontierRecovery r;
  r.select("obligation:1",3);
  r.advance(); r.segments=3;
  r.remember(11,.03);
  r.rejected=1;
  assert(r.reserveReplacement());
  assert(r.attempts==2 && r.segments==3 && r.rejected==0 && !r.certify(11,12));
  r.invalidateMotion(); // new token, old grant lost, or generation failure
  assert(r.attempts==2 && r.segments==3);
  r.select("obligation:5",3);r.select("obligation:1",3);
  assert(r.attempts==2 && r.segments==3);
  assert(r.reserveReplacement()); assert(r.reserveReplacement());
  assert(r.reserveReplacement());
  assert(r.attempts==5 && r.final_stage_replacement_used && !r.reserveReplacement());
  r.advance();assert(r.blocked() && !r.reserveReplacement());
  SafeFrontierRecovery stall_final;
  stall_final.select("fifth_group_after_stall",3);
  for(int i=0;i<5;++i) stall_final.advance();
  assert(stall_final.attempts==5 && !stall_final.final_stage_replacement_used);
  assert(stall_final.reserveReplacement());
  assert(stall_final.attempts==5 && stall_final.final_stage_replacement_used);
  assert(!stall_final.reserveReplacement());
  stall_final.select("other",3);stall_final.select("fifth_group_after_stall",3);
  assert(stall_final.final_stage_replacement_used && !stall_final.reserveReplacement());
  r.select("obligation:1",4);r.advance();r.segments=6;
  assert(!r.reserveReplacement());
  const int attempts_before_owner = r.attempts;
  const int segments_before_owner = r.segments;
  assert(r.authorizeOwnerPromotion());
  assert(r.attempts==attempts_before_owner && r.segments==segments_before_owner);
  r.attempts=SafeFrontierRecovery::max_attempts+1;
  assert(!r.authorizeOwnerPromotion());
  std::cout << "replacement ledger PASS\n";
}
