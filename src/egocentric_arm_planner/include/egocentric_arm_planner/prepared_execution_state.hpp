#pragma once
#include "egocentric_arm_planner/task_observation_context.hpp"
#include <egocentric_arm_planner/CandidateTransfer.h>
#include <care_velocity_controller/CommandStatus.h>
#include <care_velocity_controller/ExecutionLease.h>
#include <care_velocity_controller/LeasedVelocityCommand.h>
#include <openssl/sha.h>
#include <ros/serialization.h>
#include <chrono>

namespace egocentric_arm_planner {
// Stages immutable motion and confirms ZERO application only. No activation
// method or nonzero output exists until stopping qualification is connected.
class PreparedExecutionState {
 public:
  TaskObservationContext context;
  TaskPlanningEventReceiver events;
  std::string regime,reason="waiting_preparation";
  uint64_t control_until=0,revision=0;
  bool ready=false,pending=false,zero_applied=false,clock_fault=false;
  CandidateTransfer preparation;
  care_velocity_controller::CommandStatus state;
  care_velocity_controller::ExecutionLease grant;
  uint64_t first_source=0,last_source=0,applied_source=0,applied_ns=0;

  static std::string digest(const CandidateTransfer& value) {
    auto raw=value.trajectory;raw.header.seq=0;
    std::string bytes(ros::serialization::serializationLength(raw),'\0');
    ros::serialization::OStream stream(reinterpret_cast<uint8_t*>(&bytes[0]),bytes.size());
    ros::serialization::serialize(stream,raw);
    for(const auto& field:std::vector<std::pair<std::string,std::string>>{
        {"repair_prefix_context",value.planned_prefix_context},
        {"task_observation_context",value.task_context},{"task_regime",value.task_regime}}) {
      if(!field.second.empty()) {bytes.push_back('\0');bytes+=field.first;bytes.push_back('\0');bytes+=field.second;}
    }
    unsigned char output[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size(),output);
    std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(auto c:output)out<<std::setw(2)<<static_cast<unsigned>(c);
    return out.str();
  }
  void cancel(const std::string& why) {
    pending=false;zero_applied=false;grant=care_velocity_controller::ExecutionLease();reason=why;
  }
  void updateContext(const std::string& raw,const ros::Time& now) {
    const auto before=context.binding;
    context.update(raw,now);
    if(before!=context.binding || !context.current(context.binding,now)) {
      cancel("task_context_changed");regime.clear();ready=false;control_until=revision=0;control_source.clear();
    }
  }
  void updateControl(const std::string& raw,const ros::Time& now) {
    const auto e=events.parse(raw,context,now,"regime_control");const auto& p=e.payload;
    const auto mode=p.get<std::string>("mode");const auto rev=p.get<uint64_t>("mode_revision");
    taskControlRecoveryTarget(p);
    if((mode!="NORMAL" && mode!="REPAIR" && mode!="PROBE_NORMAL") ||
       !rev || rev>=9007199254740992ULL ||
       (e.source==control_source && rev<revision))throw std::runtime_error("invalid_prepared_control");
    const bool next_ready=p.get<bool>("ready");
    p.get<bool>("clear");p.get<bool>("force_bootstrap");
    if(p.get<uint64_t>("replan_seq")>=9007199254740992ULL ||
       p.get<uint64_t>("reference_refresh_seq")>=9007199254740992ULL ||
       p.get<std::string>("qp_retry").size()>256)throw std::runtime_error("invalid_prepared_control");
    const auto identity=e.source+":"+std::to_string(rev)+":"+mode;
    if(e.source==control_source && rev==revision && !regime.empty() && identity!=regime)
      throw std::runtime_error("mutated_prepared_control");
    events.remember(e);
    if(identity!=regime || !next_ready)cancel("task_regime_changed");
    regime=identity;ready=next_ready;control_until=e.until;revision=rev;control_source=e.source;
  }
  bool fresh(const ros::Time& now) {
    const auto ns=now.toNSec();
    if(!ns || (last_now && ns<last_now)) {clock_fault=true;cancel("prepared_clock_reversed");}
    last_now=std::max(last_now,ns);
    if(clock_fault)return false;
    if(!context.current(context.binding,now) || !ready || ns>=control_until) {
      if(pending)cancel("prepared_context_or_control_expired");
      return false;
    }
    if(pending && (ns>=expires || std::chrono::steady_clock::now()>=wall_until)) {
      cancel("execution_preparation_expired");return false;
    }
    return true;
  }
  bool receiptFresh(const ros::Time& now) const {
    return state.controller_session && !state.header.stamp.isZero() && state.header.stamp<=now &&
        (now-state.header.stamp).toSec()<=.05 && state.execution_lease_required && !state.clock_fault;
  }
  bool atRest() const {
    for(size_t j=0;j<7;++j)
      if(!std::isfinite(state.applied_velocity[j]) || state.applied_velocity[j]!=0. ||
         !std::isfinite(state.measured_velocity[j]) || std::abs(state.measured_velocity[j])>.001)return false;
    return true;
  }
  void observe(const care_velocity_controller::CommandStatus& value,const ros::Time& now,
               const std::vector<double>& velocity,const std::vector<double>& acceleration) {
    if(!value.controller_session || !value.execution_lease_required || value.header.stamp.isZero() ||
       value.header.stamp>now || (now-value.header.stamp).toSec()>.05 || velocity.size()!=7 || acceleration.size()!=7)return;
    for(size_t j=0;j<7;++j)
      if(!std::isfinite(value.applied_velocity[j]) || !std::isfinite(value.measured_velocity[j]) ||
         !std::isfinite(value.measured_position[j]) || value.velocity_limits[j]!=velocity[j] ||
         value.acceleration_limits[j]!=acceleration[j])return;
    if(value.controller_session==state.controller_session && value.header.stamp<=state.header.stamp)return;
    if(value.controller_session!=state.controller_session) {
      if(retired_sessions.count(value.controller_session) || retired_sessions.size()>=16)return;
      if(state.controller_session)retired_sessions.insert(state.controller_session);
      cancel("controller_session_changed");last_lease_sequence=0;
    }
    state=value;
    if(!pending)return;
    if(value.clock_fault || value.mode>=care_velocity_controller::CommandStatus::BRAKING) {
      cancel("controller_stopping_or_faulted");return;
    }
    if(!fresh(now))return;
    if(grant.lease_sequence && value.mode==care_velocity_controller::CommandStatus::TRACKING &&
       value.execution_stamp_ns==preparation.trajectory.header.stamp.toNSec() &&
       value.lease_sequence==grant.lease_sequence && value.lease_valid_until==grant.valid_until &&
       value.certificate_sha256==grant.certificate_sha256 && value.target_range_enforced &&
       value.target_abs_velocity==grant.target_abs_velocity && first_source &&
       value.source_sequence>=first_source && value.source_sequence<=last_source && atRest()) {
      if(!zero_applied) {applied_source=value.source_sequence;applied_ns=value.header.stamp.toNSec();}
      zero_applied=true;reason="zero_applied_waiting_stopping_qualification";
    }
  }
  bool offer(const CandidateTransfer& value,const ros::Time& now) {
    if(pending && fresh(now) && value.session==preparation.session && value.sequence==preparation.sequence) {
      if(value.payload_sha256!=preparation.payload_sha256 || value.payload_sha256!=digest(value))
        throw std::runtime_error("prepared_payload_mutated");
      return false;
    }
    if(!fresh(now) || !receiptFresh(now) || !atRest() || state.mode!=care_velocity_controller::CommandStatus::WAITING)
      throw std::runtime_error("preparation_requires_fresh_waiting_rest");
    if(value.action!="execution_prepare" || value.trajectory.points.empty() || value.trajectory.points.size()>257 ||
       !std::regex_match(value.session,std::regex("[A-Za-z0-9_.:-]{1,128}")) || !value.sequence ||
       value.task_context!=context.binding || value.task_regime!=regime ||
       value.trajectory.header.frame_id!=context.target.header.frame_id ||
       !value.trajectory.header.stamp.toNSec() || value.trajectory.header.stamp.toNSec()<=state.retired_execution_through_ns ||
       value.planned_prefix_context.size()>32768 || value.payload_sha256!=digest(value))
      throw std::runtime_error("invalid_prepared_identity_or_digest");
    boost::property_tree::ptree meta;std::istringstream input(value.planned_prefix_context);read_json(input,meta);
    std::set<std::string> keys;for(const auto& field:meta)keys.insert(field.first);
    const auto verified=meta.get<uint64_t>("verified_ns"),until=meta.get<uint64_t>("expires_ns");
    if(meta.size()!=9 || keys.size()!=9 || meta.get<int>("schema")!=1 ||
       meta.get<std::string>("phase")!="execution_prepare" || meta.get<std::string>("execution_qualification")!="UNMET" ||
       !meta.get<uint64_t>("verification_seq") || !meta.get<uint64_t>("source_sequence") ||
       !std::regex_match(meta.get<std::string>("source_session"),std::regex("[A-Za-z0-9_.:-]{1,128}")) ||
       !std::regex_match(meta.get<std::string>("source_payload_sha256"),std::regex("[0-9a-f]{64}")) ||
       verified<context.started || verified>now.toNSec() || until<=now.toNSec() || until<=verified ||
       until-verified>250000000ULL)
      throw std::runtime_error("invalid_prepared_metadata_or_deadline");
    if(pending && value.session==preparation.session && value.sequence==preparation.sequence) {
      if(value.payload_sha256!=preparation.payload_sha256)throw std::runtime_error("prepared_payload_mutated");
      return false;
    }
    const auto found=source_sequences.find(value.session);
    if((found!=source_sequences.end() && value.sequence<=found->second) ||
       (found==source_sequences.end() && source_sequences.size()>=16))
      throw std::runtime_error("retired_preparation_source_or_capacity");
    const auto local_until=std::min({until,control_until,context.until});
    if(local_until<=now.toNSec())throw std::runtime_error("expired_local_preparation_window");
    source_sequences[value.session]=value.sequence;preparation=value;expires=local_until;verified_ns=verified;
    wall_until=std::chrono::steady_clock::now()+std::chrono::nanoseconds(local_until-now.toNSec());
    pending=true;zero_applied=false;first_source=last_source=applied_source=applied_ns=0;
    grant=care_velocity_controller::ExecutionLease();reason="waiting_zero_authorization";return true;
  }
  bool lease(const care_velocity_controller::ExecutionLease& value,const ros::Time& now) {
    if(!pending || !fresh(now) || !receiptFresh(now))return false;
    if(value.revoke && value.controller_session==state.controller_session &&
       value.execution_stamp_ns==preparation.trajectory.header.stamp.toNSec()) {cancel("preparation_revoked");return false;}
    if(value.revoke || value.controller_session!=state.controller_session ||
       value.execution_stamp_ns!=preparation.trajectory.header.stamp.toNSec() ||
       value.execution_stamp_ns<=state.retired_execution_through_ns || !value.target_range_enforced ||
       !value.lease_sequence || value.lease_sequence<=last_lease_sequence || value.issued_at>now ||
       value.issued_at.toNSec()<verified_ns || value.valid_until<=now || value.valid_until.toNSec()>expires ||
       value.valid_until<=value.issued_at || (value.valid_until-value.issued_at).toSec()>.1+1e-12 ||
       !std::regex_match(value.certificate_sha256,std::regex("[0-9a-f]{64}")))return false;
    for(double v:value.target_abs_velocity)if(v!=0.)return false;
    if(grant.lease_sequence)return false; // One first-zero grant; no implicit renewal.
    grant=value;last_lease_sequence=value.lease_sequence;reason="waiting_zero_application";return true;
  }
  bool zero(const ros::Time& now,uint64_t sequence,const std::vector<std::string>& names,
            care_velocity_controller::LeasedVelocityCommand& value) {
    if(!pending || !fresh(now) || !receiptFresh(now) || !atRest() || !grant.lease_sequence || grant.valid_until<=now)return false;
    value.header.stamp=now;value.source_sequence=sequence;value.joint_names=names;
    value.controller_session=grant.controller_session;value.execution_stamp_ns=grant.execution_stamp_ns;
    value.lease_sequence=grant.lease_sequence;value.valid_until=grant.valid_until;
    value.certificate_sha256=grant.certificate_sha256;value.target_range_enforced=true;
    value.target_abs_velocity=grant.target_abs_velocity;value.velocity.assign(7,0.);
    if(!first_source)first_source=sequence;
    last_source=sequence;
    return true;
  }
  std::string status() const {
    std::ostringstream out;out<<"{\"state\":"<<taskJsonQuote(reason)<<",\"pending\":"<<(pending?"true":"false")
        <<",\"zero_applied\":"<<(zero_applied?"true":"false")
        <<",\"execution_authorized\":false,\"activation_available\":false,\"phase_origin_ns\":0,\"phase_s\":0"
        <<",\"execution_stamp_ns\":"<<(pending?preparation.trajectory.header.stamp.toNSec():0)
        <<",\"preparation_sha256\":"<<taskJsonQuote(pending?preparation.payload_sha256:"")
        <<",\"preparation_session\":"<<taskJsonQuote(pending?preparation.session:"")
        <<",\"preparation_sequence\":"<<(pending?preparation.sequence:0)
        <<",\"expires_ns\":"<<(pending?expires:0)
        <<",\"task_regime\":"<<taskJsonQuote(regime)
        <<",\"controller_session\":"<<state.controller_session<<",\"zero_application_ns\":"<<applied_ns
        <<",\"zero_application_source_sequence\":"<<applied_source<<"}";return out.str();
  }
 private:
  std::string control_source;
  uint64_t last_now=0,expires=0,verified_ns=0,last_lease_sequence=0;
  std::map<std::string,uint64_t> source_sequences;
  std::set<uint64_t> retired_sessions;
  std::chrono::steady_clock::time_point wall_until;
};
} // namespace egocentric_arm_planner
