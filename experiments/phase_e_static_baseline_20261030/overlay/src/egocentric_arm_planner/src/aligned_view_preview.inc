// Private, one-shot comparison of two hypothetical observation objectives on an
// immutable local solve context. It grants no execution or map-completion rights.
void LocalSparseSCPPlanner::alignedViewPreview(const std_msgs::StringConstPtr& message) {
  std::unique_lock<std::mutex> request_lock(view_preview_request_mutex_,std::try_to_lock);
  std::ostringstream out;std::string id="invalid";
  try {
    boost::property_tree::ptree request;std::istringstream input(message->data);
    boost::property_tree::read_json(input,request);id=request.get<std::string>("id");
    if(!request_lock.owns_lock())throw std::runtime_error("native_busy");
    const std::string phase=request.get<std::string>("phase");
    const auto now=ros::WallTime::now();
    if(phase=="capture") {
      std::shared_ptr<const ViewLocalContext> context;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if(view_pending_context_ && (now-view_pending_started_).toSec()<.45)
          throw std::runtime_error("snapshot_busy");
        view_pending_context_.reset();view_pending_id_.clear();
        context=view_local_context_;
        if(!context || (now-context->wall).toSec()>.2 ||
            context->epoch!=mode_epoch_ || context->goal_revision!=terminal_goal_revision_ ||
            context->owner!=latest_observation_token_ || context->region!=latest_observation_region_ ||
            context->q_bar.rows()!=dof_ || context->q_bar.cols()!=num_intervals_+1 ||
            context->previous.size()!=dof_ || !prefixSelectionCurrentLocked(context->prefix))
          throw std::runtime_error("no_fresh_matched_context");
        Eigen::VectorXd measured;
        const double age=(ros::Time::now()-latest_joint_state_.header.stamp).toSec();
        if(age<0. || age>.2 || !extractMeasuredQ(latest_joint_state_,measured) ||
            (measured-context->q_bar.col(0)).lpNorm<Eigen::Infinity>()>.05)
          throw std::runtime_error("local_anchor_outdated");
        view_pending_context_=context;view_pending_id_=id;view_pending_started_=now;
      }
      out<<"{\"id\":";jsonString(out,id);out<<",\"status\":\"captured\",\"context_id\":";jsonString(out,id);
      out<<",\"q_start\":";jsonVector(out,context->q_bar.col(0));
      out<<",\"previous_command\":";jsonVector(out,context->previous);
      out<<",\"plan_sequence\":"<<context->plan_sequence<<",\"source_repair\":"<<(context->repair?"true":"false")
         <<",\"source_obligation_points\":"<<context->obligation_points.size()
         <<",\"source_active_points\":"<<context->snapshot.points.size()
         <<",\"source_witnesses\":"<<context->snapshot.witnesses.size()
         <<",\"source_age_s\":"<<(now-context->wall).toSec()<<",\"trust_radius\":"<<context->trust
         <<",\"iteration\":"<<context->iteration<<",\"prefix_s\":"<<context->prefix.prefix_s
         <<",\"motion_authorized\":false}";
    } else if(phase=="evaluate") {
      std::shared_ptr<const ViewLocalContext> context;ros::WallTime started;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!view_pending_context_ || request.get<std::string>("context_id")!=view_pending_id_)
          throw std::runtime_error("unknown_or_consumed_context");
        context=view_pending_context_;started=view_pending_started_;
        view_pending_context_.reset();view_pending_id_.clear(); // Consume even if invalid.
        if((now-started).toSec()>.45 || context->epoch!=mode_epoch_ ||
            context->goal_revision!=terminal_goal_revision_ || context->owner!=latest_observation_token_ ||
            context->region!=latest_observation_region_ || !prefixSelectionCurrentLocked(context->prefix))
          throw std::runtime_error("expired_or_changed_owner");
      }
      auto check_budget=[&]() {
        if((ros::WallTime::now()-started).toSec()>=.45)throw std::runtime_error("total_preview_budget");
      };
      ViewPreviewSnapshot snapshot=context->snapshot;
      snapshot.guard=primitive_observation_guard_.cloneForPreview();
      std::vector<Eigen::Vector3d> points;
      for(const auto& entry:request.get_child("points")) {
        Eigen::Vector3d point;int j=0;
        for(const auto& value:entry.second) {if(j>=3)throw std::runtime_error("point_size");point[j++]=value.second.get_value<double>();}
        if(j!=3 || !point.allFinite())throw std::runtime_error("invalid_points");points.push_back(point);
      }
      if(points.empty() || points.size()>32)throw std::runtime_error("point_count");
      // Preserve current witnesses/guards and append the proposed target points.
      auto obligations=context->obligation_points;
      for(const auto& point:points)if(std::none_of(obligations.begin(),obligations.end(),
          [&](const Eigen::Vector3d& old){return (old-point).lpNorm<Eigen::Infinity>()<=1e-5;}))obligations.push_back(point);
      for(const auto& point:points)if(std::none_of(snapshot.points.begin(),snapshot.points.end(),
          [&](const Eigen::Vector3d& old){return (old-point).lpNorm<Eigen::Infinity>()<=1e-5;}))snapshot.points.push_back(point);
      const auto& candidates=request.get_child("candidates");
      if(candidates.size()!=2)throw std::runtime_error("candidate_count");
      auto vector=[&](const boost::property_tree::ptree& node,const char* key) {
        Eigen::VectorXd value(dof_);int j=0;
        for(const auto& item:node.get_child(key)){if(j>=dof_)throw std::runtime_error("vector_size");value[j++]=item.second.get_value<double>();}
        if(j!=dof_ || !value.allFinite())throw std::runtime_error("invalid_vector");return value;
      };
      out<<"{\"id\":";jsonString(out,id);out<<",\"status\":\"valid\",\"context_id\":";jsonString(out,request.get<std::string>("context_id"));
      out<<",\"motion_authorized\":false,\"q_start\":";jsonVector(out,context->q_bar.col(0));
      out<<",\"velocity_start\":";jsonVector(out,context->previous);
      out<<",\"plan_sequence\":"<<context->plan_sequence<<",\"source_repair\":"<<(context->repair?"true":"false")
         <<",\"selected_map_query_ns\":\""<<context->batch.header.stamp.toNSec()<<"\",\"candidate_results\":[";
      int index=0;
      for(const auto& item:candidates) {
        check_budget();
        const auto q=vector(item.second,"q"),mask=vector(item.second,"mask"),target=vector(item.second,"target"),gradient=vector(item.second,"gradient");
        if(((mask.array()!=0.) && (mask.array()!=1.)).any() || mask.sum()<=0. ||
            (target-context->q_bar.col(0)).lpNorm<Eigen::Infinity>()>.2+1e-9 ||
            ((Eigen::VectorXd::Ones(dof_)-mask).cwiseProduct(gradient)).norm()>1e-9)
          throw std::runtime_error("invalid_rolling_objective");
        DeadlineWaypoint wp;wp.id=-1;wp.terminal_objective=true;wp.q=q;wp.joint_mask=mask;
        const std::vector<DeadlineWaypoint> schedule{wp};
        FrontierObjective frontier;frontier.active=true;frontier.q=target;frontier.joint_mask=mask;
        frontier.visibility_gradient=gradient;frontier.frontier_weight_scale=1.;frontier.qvis_weight_scale=0.;
        ObservationDecisionPolicy policy;
        policy.repair=true;policy.observation_phase=true;policy.frontier_active=true;
        policy.iteration=context->iteration;policy.max_iterations=max_scp_iterations_;
        policy.free_joint_task_enabled=repair_free_joint_task_tracking_scale_>0.;
        policy.schedule_size=1;policy.mask_has_free_joint=(mask.array()==0.).any();
        const auto decision=observationLocalDecision(policy,[&](bool free_joint_task) {
          check_budget();
          return solveSparseSubproblem(context->batch,context->q_bar,context->u_bar,context->q_ref,context->u_ref,
              context->previous,schedule,frontier,obligations,true,false,context->trust,context->slack,
              false,false,context->step_scale,false,&context->prefix,free_joint_task,&snapshot);
        },[&](const SparseSolveResult& primary,const SparseSolveResult& trial) {
          return freeJointTaskImproves(primary,trial,schedule,frontier,context->q_ref);
        });
        check_budget();const auto& result=decision.result;
        if(index++)out<<',';out<<"{\"sensor_id\":"<<item.second.get<int>("sensor_id")
          <<",\"solved\":"<<(result.solved?"true":"false")<<",\"primitive_valid\":"<<(result.primitive_executable_valid?"true":"false")
          <<",\"free_joint_trial_ran\":"<<(decision.free_joint_trial_ran?"true":"false")
          <<",\"free_joint_trial_used\":"<<(decision.free_joint_trial_used?"true":"false")<<",\"status\":";jsonString(out,result.status);
        out<<",\"setup_solve_ms\":";jsonNumber(out,result.setup_and_solve_ms);
        out<<",\"prefix_s\":"<<context->prefix.prefix_s<<",\"samples\":[";
        auto settings=repair_executable_settings_;settings.initial_velocity=context->previous;settings.prefixes={context->prefix.prefix_s};
        std::vector<RepairExecutableSample> samples;std::string error;
        if(result.solved && repairExecutableSamples(result.q,result.u,dt_,settings,&samples,&error)) {
          int count=0;for(const auto& sample:samples){if(count++)out<<',';out<<"{\"t\":"<<sample.time_s<<",\"q\":";jsonVector(out,sample.q);out<<'}';}
        }
        out<<"]}";
      }
      check_budget();
      {std::lock_guard<std::mutex> lock(mutex_);Eigen::VectorXd measured;
        if(context->epoch!=mode_epoch_ || context->goal_revision!=terminal_goal_revision_ ||
            context->owner!=latest_observation_token_ || context->region!=latest_observation_region_ ||
            !prefixSelectionCurrentLocked(context->prefix) || !extractMeasuredQ(latest_joint_state_,measured) ||
            (measured-context->q_bar.col(0)).lpNorm<Eigen::Infinity>()>.05)
          throw std::runtime_error("owner_changed_during_preview");}
      out<<"],\"elapsed_ms\":"<<(ros::WallTime::now()-started).toSec()*1000.
         <<",\"qualification\":\"Frozen local solve with hypothetical ordinary observation objective; no future SCP or execution certificate\"}";
    } else throw std::runtime_error("unknown_phase");
  } catch(const std::exception& error) {
    out.str("");out.clear();out<<"{\"id\":";jsonString(out,id);out<<",\"status\":\"unavailable\",\"reason\":";jsonString(out,error.what());out<<'}';
  }
  std_msgs::String response;response.data=out.str();view_preview_pub_.publish(response);
}
