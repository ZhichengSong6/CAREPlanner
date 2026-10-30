#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace egocentric_arm_planner {

struct PrimitiveObservationPoint {
  Eigen::Vector3d point = Eigen::Vector3d::Zero();
  std::string sweeping_link, collision_name;
  int collision_index = -1;
  double sweep_time_s = 0.;
  std::uint64_t audited_stamp = 0;
  // A path-floor counterexample is a point guard, not a collision witness.
  // It uses the nearest primitive. Fresh UNKNOWN points may use the
  // existing execution-floor certificate with a soft planning-margin target.
  bool ordinary_path_guard = false;
  double proposal_reserve_m = 0.;
};

struct HandoffResidualFailure {
  PrimitiveObservationPoint point;
  bool requires_reconstruction=false;
  bool initial_state_changed=false;
  double required_residual_m=0.,certified_residual_m=0.;
};
bool parseHandoffResidualFailure(const std::string& evidence,
    std::uint64_t raw_stamp,std::uint64_t audited_stamp,std::uint64_t mode_epoch,
    const std::string& observation_token,double ordinary_floor,HandoffResidualFailure* failure);

// Strict, identity-bound input for proposal geometry. Malformed evidence never
// removes a CDF row or grants permission to execute.
bool parsePrimitiveObservationEvidence(const std::string& evidence,
    std::uint64_t audited_stamp, std::vector<PrimitiveObservationPoint>* points);
bool parseOrdinaryPathCounterexample(const std::string& evidence,
    std::uint64_t audited_stamp, double ordinary_floor, PrimitiveObservationPoint* point);
// Scheduling evidence only; exhaustion does not establish a distance violation.
bool parseProofResourceFailure(const std::string& evidence,
    std::uint64_t audited_stamp, std::uint64_t mode_epoch,
    const std::string& observation_token, double ordinary_floor,
    PrimitiveObservationPoint* point);

class PrimitiveObservationGuard {
 public:
  struct Row {
    int timestep = 0;
    Eigen::Vector3d point;
    double distance_m = 0.;
    Eigen::VectorXd gradient;
    Eigen::VectorXd q_linearization;
    std::string link, collision_name;
    bool known_ground = false;
    bool occupied_volume = false;
    double required_margin_m = 0.;
  };
  PrimitiveObservationGuard();
  ~PrimitiveObservationGuard();
  std::unique_ptr<PrimitiveObservationGuard> cloneForPreview() const;
  bool initialize(const std::string& fk_urdf, const std::string& primitive_urdf,
      const std::string& base_frame, const std::vector<std::string>& joint_names,
      const std::vector<std::string>& ignored_links, std::string* error);
  bool linearize(const Eigen::MatrixXd& q,
      const std::vector<PrimitiveObservationPoint>& points,
      std::vector<Row>* rows, std::string* error,
      bool nearest_only_proposal = false) const;
  // Objective-only pose check with private FK scratch, independent of the
  // worker's mutable collision linearization data and certificate FK ledger.
  bool framePose(const Eigen::VectorXd& q,const std::string& frame,
      Eigen::Isometry3d* pose) const;
  bool linearizeOccupiedVoxels(const Eigen::MatrixXd& q,const std::vector<Eigen::Vector3d>& centers,
      double resolution,double physical,double tracking,std::vector<Row>* rows,std::string* error,double budget_s=.03) const;
  bool knownGroundEnabled() const;
  double knownGroundMargin() const;
  bool linearizeKnownGround(const Eigen::MatrixXd& q,std::vector<Row>* rows,
      std::string* error) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace egocentric_arm_planner
