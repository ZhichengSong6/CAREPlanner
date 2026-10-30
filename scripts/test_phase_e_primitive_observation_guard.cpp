// Real primitive FK/SDF and Jacobian regression, without ROS graph or robot.
#include "egocentric_arm_planner/primitive_observation_guard.hpp"
#include <care_confidence_map/vbc_primitive_model.hpp>
#include <iostream>
#include <iomanip>
#include <stdexcept>

using namespace egocentric_arm_planner;
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
  try {
    require(argc == 3, "FK and primitive URDF paths required");
    const std::vector<std::string> joints{"joint1","joint2","joint3","joint4",
        "wrist_joint1","wrist_joint2","wrist_joint3"};
    PrimitiveObservationGuard guard;
    std::string error;
    require(guard.initialize(argv[1],argv[2],"base_link",joints,{"base_link","link1"},&error),error);
    care_confidence_map::TrajectoryRiskEvaluator fk;
    care_confidence_map::VbcPrimitiveModel model;
    require(fk.initializeKinematics(argv[1],"base_link",&error),error);
    require(model.load(argv[2],{"base_link","link1"},&error),error);
    const std::string prefix = "{\"schema\":1,\"trajectory_stamp_ns\":\"26193000000\","
        "\"geometry_backend\":\"primitive\",\"spatial_semantics\":\"voxel_centers_piecewise_linear_enclosure\","
        "\"violations\":[{\"point\":[-0.15,-0.20,0.85],\"sweeping_link\":\"wrist_link3\","
        "\"source_collision_name\":\"wrist_link3_sensor1_block\",\"source_collision_index\":1,";
    // The production collision index is resolved below from the immutable URDF.
    int collision = -1;
    for (const auto& p : model.primitives()) if (p.collision_name == "wrist_link3_sensor1_block")
      collision = p.collision_index;
    require(collision >= 0, "Missing recorded CASE026 primitive");
    std::vector<PrimitiveObservationPoint> points;
    require(parsePrimitiveObservationEvidence(prefix+"\"sweep_s\":0.0333333333333333}]}",26193000000,&points),"valid evidence");
    points.front().collision_index = collision;
    std::vector<PrimitiveObservationPoint> rejected;
    require(!parsePrimitiveObservationEvidence(prefix+"\"sweep_s\":0.03}]}",26193000001,&rejected),"foreign identity accepted");
    require(!parsePrimitiveObservationEvidence(prefix+"\"sweep_s\":-1}]}",26193000000,&rejected),"negative entry accepted");
    require(!parsePrimitiveObservationEvidence(prefix+"\"sweep_s\":0,\"sweep_s\":1}]}",26193000000,&rejected),"duplicate entry accepted");
    require(!parsePrimitiveObservationEvidence(prefix+"\"sweep_s\":0}]}junk",26193000000,&rejected),"trailing junk accepted");

    Eigen::MatrixXd q(7,3);
    q.col(0) << -.3766,.2381,-.7043,-.6656,.4890,-.8407,-.5084;
    q.col(1) << -.1262889825,.5830495024,-.5395028537,-.4452299737,.2022769275,-.1999110316,-.6663374306;
    q.col(2) << .23,-.19,.31,-.43,.22,-.28,.17;
    std::vector<PrimitiveObservationGuard::Row> rows;
    require(guard.linearize(q,points,&rows,&error),error);
    double max_gradient_error = 0., max_distance_error = 0.;
    int comparisons = 0;
    for (const auto& row : rows) {
      auto sdf = [&](const Eigen::VectorXd& pose) {
        std::vector<care_confidence_map::VbcPrimitiveFrame> frames;
        require(model.computeTrajectory(fk,{pose},&frames,&error),error);
        for (const auto& p : frames.front().primitives)
          if (p.link_name == row.link && p.collision_name == row.collision_name)
            return p.signedDistance(row.point);
        throw std::runtime_error("Missing linearized body");
      };
      max_distance_error = std::max(max_distance_error,std::abs(sdf(q.col(row.timestep))-row.distance_m));
      for (int j = 0; j < 7; ++j) {
        Eigen::VectorXd plus = q.col(row.timestep), minus = plus;
        constexpr double h = 1e-6;
        plus[j] += h; minus[j] -= h;
        const double numeric = (sdf(plus)-sdf(minus))/(2*h);
        max_gradient_error = std::max(max_gradient_error,std::abs(numeric-row.gradient[j]));
        ++comparisons;
      }
    }
    require(max_distance_error < 1e-10,"Production SDF disagrees");
    require(max_gradient_error < 2e-6,"Joint-space distance Jacobian disagrees with independent finite difference");
    auto malformed = points; malformed.front().collision_name = "foreign_body";
    require(!guard.linearize(q,malformed,&rows,&error),"Foreign primitive accepted");
    require(rows.empty(),"Partial geometry leaked on failure");
    require(!guard.linearize(q,std::vector<PrimitiveObservationPoint>(33,points.front()),&rows,&error),"Point budget exceeded");
    q(0,0) = std::numeric_limits<double>::quiet_NaN();
    require(!guard.linearize(q,points,&rows,&error),"Nonfinite configuration accepted");
    std::cout << std::setprecision(17) << "{\"status\":\"PASS\",\"gradient_comparisons\":" << comparisons
        << ",\"max_gradient_error\":" << max_gradient_error
        << ",\"max_distance_error_m\":" << max_distance_error << "}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
