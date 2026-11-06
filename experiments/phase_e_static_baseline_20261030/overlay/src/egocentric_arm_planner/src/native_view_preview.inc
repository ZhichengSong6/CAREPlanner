// Optional proposal ranking only. No publication to any motion/audit/owner topic.
void LocalSparseSCPPlanner::viewPreviewCallback(const std_msgs::StringConstPtr& message) {
  if(view_preview_aligned_enabled_) {alignedViewPreview(message);return;}
  std::ostringstream out;std::string id="invalid";
  const auto started=ros::WallTime::now();
  try {
    boost::property_tree::ptree request;std::istringstream input(message->data);
    boost::property_tree::read_json(input,request);id=request.get<std::string>("id");
    const auto& candidates=request.get_child("candidates");
    if(candidates.empty() || candidates.size()>2)throw std::runtime_error("candidate_count");
    ViewPreviewSnapshot snapshot;Eigen::VectorXd measured,velocity(dof_);
    Eigen::MatrixXd task_ref,task_u;care_collision_cdf::CollisionCDFConstraintBatch batch;
    RepairPrefixSelection prefix;ros::Time measured_stamp,batch_received;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if(!view_preview_batch_ || !extractMeasuredQ(latest_joint_state_,measured) ||
          plan_q_ref_.rows()!=dof_)throw std::runtime_error("no_native_context");
      batch=*view_preview_batch_;batch_received=view_preview_batch_received_;
      measured_stamp=latest_joint_state_.header.stamp;
      if((ros::Time::now()-measured_stamp).toSec()>.2 ||
          (ros::Time::now()-batch_received).toSec()>1.)throw std::runtime_error("stale_native_context");
      for(int j=0;j<dof_;++j) {
        auto it=std::find(latest_joint_state_.name.begin(),latest_joint_state_.name.end(),joint_names_[j]);
        const auto k=std::distance(latest_joint_state_.name.begin(),it);
        if(it==latest_joint_state_.name.end() || k>=static_cast<long>(latest_joint_state_.velocity.size()))
          throw std::runtime_error("missing_velocity");
        velocity[j]=latest_joint_state_.velocity[k];
      }
      snapshot.witnesses=repair_unknown_witnesses_;snapshot.shell=initial_shell_recovery_;
      if(restore_path_guard_scope_==std::to_string(mode_epoch_)+":"+latest_observation_token_)
        snapshot.path_guards=restore_path_guard_points_;
      snapshot.goal=terminal_goal_;snapshot.goal_valid=terminal_goal_valid_;
      task_ref=plan_q_ref_;task_u=plan_u_ref_;prefix=latest_prefix_selection_;
    }
    snapshot.guard=primitive_observation_guard_.cloneForPreview();
    for(const auto& point:request.get_child("points")) {
      Eigen::Vector3d p;int j=0;for(const auto& value:point.second){if(j>=3)throw std::runtime_error("point_size");p[j++]=value.second.get_value<double>();}
      if(j!=3 || !p.allFinite())throw std::runtime_error("point_invalid");snapshot.points.push_back(p);
    }
    if(snapshot.points.empty() || !prefixSelectionCompatible(prefix))throw std::runtime_error("missing_points_or_prefix");
    Eigen::MatrixXd q_bar,u_bar;
    if(!buildNormalMotionSeed(measured,velocity,task_ref,q_bar,u_bar))throw std::runtime_error("seed_error");
    // Re-query the identical selected environment rows at the common new seed.
    // The selected-point snapshot is bounded/local, not a complete map proof.
    care_collision_cdf::QueryCollisionCDFPairs query;
    std::vector<int> source;
    for(int i=0;i<batch.num_pairs;++i) {
      int k=batch.original_timestep.at(i);
      if(k<1 || k>num_intervals_)continue;
      geometry_msgs::Point point;point.x=batch.point_flat.at(3*i);point.y=batch.point_flat.at(3*i+1);point.z=batch.point_flat.at(3*i+2);
      query.request.points.push_back(point);source.push_back(i);
      for(int j=0;j<dof_;++j)query.request.q_flat.push_back(q_bar(j,k));
    }
    query.request.num_pairs=source.size();
    auto client=nh_.serviceClient<care_collision_cdf::QueryCollisionCDFPairs>("/care_planner/collision_cdf/query_pairs",false);
    if(!client.call(query) || !query.response.success || query.response.distance.size()!=source.size() ||
        query.response.gradient_flat.size()!=source.size()*dof_)throw std::runtime_error("cdf_query_unavailable");
    const auto original=batch;batch.num_pairs=source.size();batch.original_timestep.clear();batch.source_type.clear();batch.point_flat.clear();
    batch.distance=query.response.distance;batch.gradient_flat=query.response.gradient_flat;batch.q_linearization_flat=query.request.q_flat;
    for(int i:source){batch.original_timestep.push_back(original.original_timestep[i]);batch.source_type.push_back(original.source_type.at(i));
      for(int j=0;j<3;++j)batch.point_flat.push_back(original.point_flat[3*i+j]);}
    out<<"{\"id\":";jsonString(out,id);out<<",\"status\":\"valid\",\"motion_authorized\":false,\"measured_stamp_ns\":\""<<measured_stamp.toNSec()
       <<"\",\"selected_map_query_ns\":\""<<batch.header.stamp.toNSec()<<"\",\"batch_received_ns\":\""<<batch_received.toNSec()<<"\",\"q_start\":";jsonVector(out,measured);
    out<<",\"velocity_start\":";jsonVector(out,velocity);out<<",\"candidate_results\":[";
    int index=0;
    for(const auto& item:candidates) {
      Eigen::VectorXd q(dof_),mask(dof_);int j=0;
      for(const auto& value:item.second.get_child("q")){if(j>=dof_)throw std::runtime_error("q_size");q[j++]=value.second.get_value<double>();}
      if(j!=dof_)throw std::runtime_error("q_size");j=0;
      for(const auto& value:item.second.get_child("mask")){if(j>=dof_)throw std::runtime_error("mask_size");mask[j++]=value.second.get_value<double>();}
      if(j!=dof_ || !q.allFinite() || !mask.allFinite())throw std::runtime_error("candidate_nonfinite");
      DeadlineWaypoint wp;wp.id=-1;wp.terminal_objective=true;wp.q=q;wp.joint_mask=mask;
      FrontierObjective frontier;frontier.active=true;frontier.q=q;frontier.joint_mask=mask;
      frontier.frontier_weight_scale=1.;frontier.qvis_weight_scale=1.;
      const auto result=solveSparseSubproblem(batch,q_bar,u_bar,task_ref,task_u,velocity,{wp},frontier,
          snapshot.points,true,false,trust_region_initial_,cdf_slack_linear_weight_,false,false,1.,false,&prefix,false,&snapshot);
      if(index++)out<<',';out<<"{\"sensor_id\":"<<item.second.get<int>("sensor_id")<<",\"solved\":"<<(result.solved?"true":"false")
        <<",\"primitive_valid\":"<<(result.primitive_executable_valid?"true":"false")<<",\"status\":";jsonString(out,result.status);
      out<<",\"setup_solve_ms\":";jsonNumber(out,result.setup_and_solve_ms);
      out<<",\"prefix_s\":"<<prefix.prefix_s<<",\"samples\":[";
      std::vector<RepairExecutableSample> samples;auto settings=repair_executable_settings_;
      settings.initial_velocity=velocity;settings.prefixes={prefix.prefix_s};std::string error;
      if(result.solved && repairExecutableSamples(result.q,result.u,dt_,settings,&samples,&error)) {
        int sample_index=0;for(const auto& sample:samples){if(sample_index++)out<<',';out<<"{\"t\":"<<sample.time_s<<",\"q\":";jsonVector(out,sample.q);out<<'}';}
      }
      out<<"]}";
    }
    out<<"],\"elapsed_ms\":"<<(ros::WallTime::now()-started).toSec()*1000.<<'}';
  } catch(const std::exception& error) {
    out.str("");out.clear();out<<"{\"id\":";jsonString(out,id);out<<",\"status\":\"unavailable\",\"reason\":";jsonString(out,error.what());out<<'}';
  }
  std_msgs::String result;result.data=out.str();view_preview_pub_.publish(result);
}
