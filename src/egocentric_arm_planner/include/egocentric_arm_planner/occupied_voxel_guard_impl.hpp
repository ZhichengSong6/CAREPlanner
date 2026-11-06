// Included inside namespace egocentric_arm_planner after Impl is defined.
bool PrimitiveObservationGuard::linearizeOccupiedVoxels(const Eigen::MatrixXd& q,
    const std::vector<Eigen::Vector3d>& centers,double resolution,double physical,double tracking,
    std::vector<Row>* rows,std::string* error,double budget_s) const {
  if(!rows)return false;rows->clear();
  const auto started=std::chrono::steady_clock::now();
  const auto timed_out=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()>budget_s;};
  try {
    const auto& p=*impl_;
    if(!p.ready || q.rows()!=int(p.q_indices.size()) || q.cols()<1 || !q.allFinite() || centers.size()>32 ||
        resolution!=.05 || physical!=.02 || tracking!=.02 || !std::isfinite(budget_s) || budget_s<=0.)
      throw std::runtime_error("invalid_occupied_volume_proposal");
    hpp::fcl::Box cube(resolution,resolution,resolution);hpp::fcl::DistanceRequest request(true);request.gjk_tolerance=1e-9;
    pinocchio::Data data(p.model);const auto& local=p.voxel_primitives.primitives();
    constexpr double numerical_reserve=1e-6;
    for(int k=0;k<q.cols();++k) {
      if(timed_out())throw std::runtime_error("occupied_volume_proposal_wall_budget");
      Eigen::VectorXd full=pinocchio::neutral(p.model);
      for(int j=0;j<q.rows();++j)full[p.q_indices[j]]=q(j,k);
      pinocchio::computeJointJacobians(p.model,data,full);pinocchio::updateFramePlacements(p.model,data);
      const auto base_inverse=data.oMf[p.base_id].inverse();
      std::vector<Eigen::Isometry3d> poses;std::vector<Eigen::Vector3d> origins;
      std::vector<Eigen::MatrixXd> jacobians;
      for(std::size_t i=0;i<local.size();++i) {
        const auto frame=base_inverse*data.oMf[p.voxel_frame_ids[i]];
        Eigen::Isometry3d pose=Eigen::Isometry3d::Identity();
        pose.linear()=frame.rotation()*local[i].rotation;pose.translation()=frame.translation()+frame.rotation()*local[i].center;
        poses.push_back(pose);origins.push_back(frame.translation());
        Eigen::Matrix<double,6,Eigen::Dynamic> J(6,p.model.nv);J.setZero();
        pinocchio::getFrameJacobian(p.model,data,p.voxel_frame_ids[i],pinocchio::LOCAL_WORLD_ALIGNED,J);
        Eigen::MatrixXd selected(6,q.rows());
        for(int j=0;j<q.rows();++j) {
          selected.col(j).head<3>()=base_inverse.rotation()*J.col(p.v_indices[j]).head<3>();
          selected.col(j).tail<3>()=base_inverse.rotation()*J.col(p.v_indices[j]).tail<3>();
        }
        jacobians.push_back(selected);
      }
      for(const auto& center:centers) {
        if(!center.allFinite())throw std::runtime_error("nonfinite_occupied_volume_center");
        const hpp::fcl::Transform3f cube_pose(Eigen::Matrix3d::Identity(),center);
        std::size_t nearest=0;double best_slack=std::numeric_limits<double>::infinity(),distance=0.,required=0.;
        hpp::fcl::DistanceResult best_result;
        for(std::size_t i=0;i<local.size();++i) {
          if(timed_out())throw std::runtime_error("occupied_volume_proposal_wall_budget");
          hpp::fcl::DistanceResult result;
          const double d=hpp::fcl::distance(p.voxel_shapes[i].get(),hpp::fcl::Transform3f(poses[i].rotation(),poses[i].translation()),&cube,cube_pose,request,result)-numerical_reserve;
          const double floor=physical+(p.voxel_moving[i]?tracking:0.);
          if(!std::isfinite(d) || d>1e10)throw std::runtime_error("invalid_occupied_volume_distance");
          if(d-floor<best_slack){nearest=i;best_slack=d-floor;distance=d;required=floor;best_result=result;}
        }
        Row row;row.timestep=k;row.point=center;row.q_linearization=q.col(k);row.occupied_volume=true;
        row.distance_m=distance;row.required_margin_m=required;row.link=local[nearest].link_name;
        row.collision_name=local[nearest].collision_name;row.gradient.resize(q.rows());
        const auto& pose=poses[nearest];const auto& J=jacobians[nearest];
        const Eigen::Vector3d difference=best_result.nearest_points[0]-best_result.nearest_points[1];
        if(distance>1e-7 && difference.allFinite() && difference.norm()>1e-9) {
          const Eigen::Vector3d normal=difference.normalized();
          for(int j=0;j<q.rows();++j)row.gradient[j]=normal.dot(J.col(j).head<3>()+
              J.col(j).tail<3>().cross(best_result.nearest_points[0]-origins[nearest]));
        } else {
          // At penetration/nonunique nearest points, differentiate rigid
          // workspace motions. This spends no perturbed joint FK calls.
          constexpr double h=1e-6;
          for(int j=0;j<q.rows();++j) {
            const Eigen::Vector3d omega=J.col(j).tail<3>();
            const Eigen::Vector3d velocity=J.col(j).head<3>()+omega.cross(pose.translation()-origins[nearest]);
            double values[2];int n=0;
            for(const double sign:{-1.,1.}) {
              Eigen::Matrix3d rotation=pose.rotation();
              if(omega.norm()>1e-12)rotation=Eigen::AngleAxisd(sign*h*omega.norm(),omega.normalized()).toRotationMatrix()*rotation;
              hpp::fcl::DistanceResult result;
              values[n++]=hpp::fcl::distance(p.voxel_shapes[nearest].get(),hpp::fcl::Transform3f(rotation,pose.translation()+sign*h*velocity),&cube,cube_pose,request,result);
            }
            row.gradient[j]=(values[1]-values[0])/(2*h);
          }
        }
        if(!row.gradient.allFinite())throw std::runtime_error("invalid_occupied_volume_gradient");
        rows->push_back(std::move(row));
      }
    }
    if(timed_out())throw std::runtime_error("occupied_volume_proposal_wall_budget");
    return true;
  } catch(const std::exception& e) {rows->clear();if(error)*error=e.what();return false;}
}
