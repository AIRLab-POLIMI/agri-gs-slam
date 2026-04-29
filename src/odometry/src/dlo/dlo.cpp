#include <dlo/dlo.hpp>
#include <chrono>
#include <iostream>

namespace dlo {

DirectLidarOdometry::DirectLidarOdometry() {
    // Initialize point clouds
    original_scan_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    current_scan_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    current_scan_t_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    
    keyframe_cloud_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    keyframes_cloud_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    submap_cloud_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    
    source_cloud_ = nullptr;
    target_cloud_ = nullptr;

    // Initialize transformation matrices
    T_ = Eigen::Matrix4f::Identity();
    T_s2s_ = Eigen::Matrix4f::Identity();
    T_s2s_prev_ = Eigen::Matrix4f::Identity();

    // Initialize pose and orientation
    pose_ = Eigen::Vector3f::Zero();
    rotq_ = Eigen::Quaternionf::Identity();
    pose_s2s_ = Eigen::Vector3f::Zero();
    rotq_s2s_ = Eigen::Quaternionf::Identity();

    // Initialize motion model
    motion_model_enabled_ = true;
    motion_smoothing_factor_ = 0.8f;
    velocity_ = Eigen::Vector3f::Zero();
    angular_velocity_ = Eigen::Vector3f::Zero();
    prev_timestamp_ = 0.0;
    motion_initialized_ = false;

    // Configure hull objects
    convex_hull_.setDimension(3);
    concave_hull_.setDimension(3);
    concave_hull_.setAlpha(keyframe_thresh_dist_);
    concave_hull_.setKeepInformation(true);

    // Configure GICP
    gicp_s2s_.setMaximumIterations(gicps2s_max_iter_);
    gicp_s2s_.setTransformationEpsilon(gicps2s_transformation_ep_);
    gicp_s2s_.setEuclideanFitnessEpsilon(gicps2s_euclidean_fitness_ep_);
    gicp_s2s_.setMaxCorrespondenceDistance(2.0f);
    gicp_s2s_.setTransformationEpsilon(1e-6);
    gicp_s2s_.setMaximumIterations(50);

    gicp_.setMaximumIterations(gicps2m_max_iter_);
    gicp_.setTransformationEpsilon(gicps2m_transformation_ep_);
    gicp_.setEuclideanFitnessEpsilon(gicps2m_euclidean_fitness_ep_);

    // Configure filters
    crop_.setNegative(true);
    crop_.setMin(Eigen::Vector4f(-crop_size_, -crop_size_, -crop_size_, 1.0));
    crop_.setMax(Eigen::Vector4f(crop_size_, crop_size_, crop_size_, 1.0));

    vf_scan_.setLeafSize(vf_scan_res_, vf_scan_res_, vf_scan_res_);
    vf_submap_.setLeafSize(vf_submap_res_, vf_submap_res_, vf_submap_res_);

    // Initialize metrics
    metrics_.spaciousness.push_back(0.0);
}

DirectLidarOdometry::~DirectLidarOdometry() {}

// PYTHON_BIND: Main tracking function - call this for each new point cloud
OdometryResult DirectLidarOdometry::track(const pcl::PointCloud<PointType>::Ptr& cloud, 
                                         double timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    OdometryResult result;
    result.timestamp = timestamp;
    result.success = false;
    result.is_keyframe = false;
    
    // Check minimum number of points
    if (cloud->points.size() < gicp_min_num_points_) {
        return result;
    }
    
    // Copy input cloud
    *current_scan_ = *cloud;
    
    // Initialize if not done yet
    if (!initialized_) {
        initialize();
        if (!initialized_) {
            return result;
        }
    }
    
    // Update motion model with timestamp
    double dt = timestamp - prev_timestamp_;
    if (motion_model_enabled_ && prev_timestamp_ > 0.0 && dt > 0.0) {
        updateMotionModel(dt);
    }
    prev_timestamp_ = timestamp;
    
    // Preprocess points
    preprocessPoints();
    
    // Compute metrics
    computeMetrics();
    
    // Set adaptive parameters
    if (adaptive_params_use_) {
        setAdaptiveParams();
    }
    
    // Handle first frame - no tracking, just initialize the target and return identity
    if (target_cloud_ == nullptr) {
        initializeInputTarget();
        
        // First frame always has identity transformation
        T_ = Eigen::Matrix4f::Identity();
        T_s2s_ = Eigen::Matrix4f::Identity();
        T_s2s_prev_ = Eigen::Matrix4f::Identity();
        
        // Apply initial pose if provided
        if (initial_pose_use_) {
            pose_ = initial_position_;
            rotq_ = initial_orientation_;
            T_.block(0,3,3,1) = pose_;
            T_.block(0,0,3,3) = rotq_.toRotationMatrix();
            T_s2s_.block(0,3,3,1) = pose_;
            T_s2s_.block(0,0,3,3) = rotq_.toRotationMatrix();
            T_s2s_prev_.block(0,3,3,1) = pose_;
            T_s2s_prev_.block(0,0,3,3) = rotq_.toRotationMatrix();
        } else {
            pose_ = Eigen::Vector3f::Zero();
            rotq_ = Eigen::Quaternionf::Identity();
        }
        
        // Update trajectory with first pose
        trajectory_.push_back(std::make_pair(pose_, rotq_));
        
        result.success = true;
        result.position = {pose_[0], pose_[1], pose_[2]};
        result.orientation = {rotq_.x(), rotq_.y(), rotq_.z(), rotq_.w()};
        result.transformation.resize(16);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result.transformation[i * 4 + j] = T_(i, j);
            }
        }
        result.is_keyframe = true;
        
        // Calculate computation time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double comp_time = duration.count() / 1000000.0;
        metrics_.computation_times.push_back(comp_time);
        result.computation_time = comp_time;
        
        return result;
    }
    
    // Set source frame
    source_cloud_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    source_cloud_ = current_scan_;

    // Set new frame as input source for both gicp objects
    setInputSources();
    
    // Get the next pose via S2S + S2M
    getNextPose();
    
    // Update current keyframe poses and map
    int prev_keyframes = num_keyframes_;
    updateKeyframes();
    bool is_keyframe = (num_keyframes_ > prev_keyframes);
    
    // Update trajectory
    trajectory_.push_back(std::make_pair(pose_, rotq_));
    
    // Calculate computation time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double comp_time = duration.count() / 1000000.0;
    metrics_.computation_times.push_back(comp_time);
    
    // Fill result
    result.success = true;
    result.position = {pose_[0], pose_[1], pose_[2]};
    result.orientation = {rotq_.x(), rotq_.y(), rotq_.z(), rotq_.w()};
    result.transformation.resize(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result.transformation[i * 4 + j] = T_(i, j);
        }
    }
    result.is_keyframe = is_keyframe;
    result.computation_time = comp_time;
    
    return result;
}

void DirectLidarOdometry::updateMotionModel(double dt) {
    if (trajectory_.size() < 2) {
        return;
    }
    
    // Get current and previous poses
    auto current_pose_pair = trajectory_.back();
    auto prev_pose_pair = trajectory_[trajectory_.size() - 2];
    
    Eigen::Vector3f current_pos = current_pose_pair.first;
    Eigen::Vector3f prev_pos = prev_pose_pair.first;
    Eigen::Quaternionf current_rot = current_pose_pair.second;
    Eigen::Quaternionf prev_rot = prev_pose_pair.second;
    
    if (dt <= 0.0 || dt > 1.0) {  // Add upper bound check
        return;
    }
    
    // Compute linear velocity with outlier protection
    Eigen::Vector3f linear_vel = (current_pos - prev_pos) / dt;
    float linear_speed = linear_vel.norm();
    const float max_linear_speed = 50.0f; // m/s
    if (linear_speed > max_linear_speed) {
        linear_vel = linear_vel * (max_linear_speed / linear_speed);
    }
    
    // Compute angular velocity with improved precision for turns
    Eigen::Quaternionf dq = current_rot * prev_rot.inverse();
    dq.normalize();
    
    // Ensure shortest path rotation
    if (dq.w() < 0) {
        dq.coeffs() = -dq.coeffs();
    }
    
    Eigen::Vector3f angular_vel = Eigen::Vector3f::Zero();
    float w = std::abs(dq.w());
    if (w < 1.0f) {  // Avoid numerical issues when w ≈ 1
        float angle = 2.0f * std::acos(std::min(w, 1.0f));
        if (angle > 1e-6) {
            float sin_half_angle = std::sqrt(1.0f - w * w);
            if (sin_half_angle > 1e-6) {
                angular_vel = Eigen::Vector3f(dq.x(), dq.y(), dq.z()) * (angle / (sin_half_angle * dt));
            }
        }
    }
    
    // Clamp angular velocity for robustness
    float angular_speed = angular_vel.norm();
    const float max_angular_speed = 5.0f; // rad/s
    if (angular_speed > max_angular_speed) {
        angular_vel = angular_vel * (max_angular_speed / angular_speed);
    }
    
    // Apply adaptive smoothing based on motion type
    float motion_factor = std::min(1.0f, (linear_speed + angular_speed * 2.0f) / 5.0f); // Higher weight for dynamic motion
    float adaptive_smoothing = motion_smoothing_factor_ * (1.0f - motion_factor * 0.3f); // Reduce smoothing during dynamic motion
    
    // Apply smoothing if motion model is already initialized
    if (motion_initialized_) {
        velocity_ = adaptive_smoothing * velocity_ + (1.0f - adaptive_smoothing) * linear_vel;
        angular_velocity_ = adaptive_smoothing * angular_velocity_ + (1.0f - adaptive_smoothing) * angular_vel;
    } else {
        velocity_ = linear_vel;
        angular_velocity_ = angular_vel;
        motion_initialized_ = true;
    }
}

Eigen::Matrix4f DirectLidarOdometry::predictMotion(double dt) {
    if (!motion_model_enabled_ || !motion_initialized_ || dt <= 0.0 || dt > 1.0) {
        return Eigen::Matrix4f::Identity();
    }
    
    // Scale prediction based on confidence (lower for high angular velocity)
    float angular_speed = angular_velocity_.norm();
    float confidence = std::exp(-angular_speed * 0.5f); // Reduce confidence for high angular motion
    confidence = std::max(0.1f, std::min(1.0f, confidence));
    
    // Predict linear displacement with confidence scaling
    Eigen::Vector3f predicted_translation = velocity_ * dt * confidence;
    
    // Predict angular displacement with improved precision
    Eigen::Vector3f angular_displacement = angular_velocity_ * dt * confidence;
    float angle = angular_displacement.norm();
    
    Eigen::Matrix4f motion_prediction = Eigen::Matrix4f::Identity();
    
    // Set translation
    motion_prediction.block(0, 3, 3, 1) = predicted_translation;
    
    // Set rotation with improved numerical stability
    if (angle > 1e-8) {
        Eigen::Vector3f axis = angular_displacement / angle;
        
        // Use Rodriguez formula for better numerical stability
        Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
        K(0, 1) = -axis(2); K(0, 2) = axis(1);
        K(1, 0) = axis(2);  K(1, 2) = -axis(0);
        K(2, 0) = -axis(1); K(2, 1) = axis(0);
        
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity() + std::sin(angle) * K + (1.0f - std::cos(angle)) * K * K;
        motion_prediction.block(0, 0, 3, 3) = R;
    }
    
    return motion_prediction;
}

void DirectLidarOdometry::getNextPose() {
    // Compute motion prediction for initial guess with current dt
    Eigen::Matrix4f motion_prediction = Eigen::Matrix4f::Identity();
    if (motion_model_enabled_ && motion_initialized_ && prev_timestamp_ > 0.0) {
        // Use a more conservative dt for prediction
        double prediction_dt = std::min(0.1, prev_timestamp_ > 0.0 ? (std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count() / 1000000.0) - prev_timestamp_ : 0.1);
        motion_prediction = predictMotion(prediction_dt);
    }
    
    // FRAME-TO-FRAME PROCEDURE
    pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>);
    
    // Set current scan as source and previous scan as target for S2S
    gicp_s2s_.clearSource();
    gicp_s2s_.clearTarget();
    gicp_s2s_.setInputSource(current_scan_);
    gicp_s2s_.setInputTarget(target_cloud_);
    
    // Check if source and target are properly set
    if (!gicp_s2s_.getInputSource() || !gicp_s2s_.getInputTarget()) {
        std::cerr << "[C++/getNextPose()] Error: GICP source or target not set!" << std::endl;
        return;
    }
    
    // Validate point cloud sizes
    if (current_scan_->points.size() < 100 || target_cloud_->points.size() < 100) {
        std::cerr << "[C++/getNextPose()] Warning: Insufficient points for alignment. Source: " 
                  << current_scan_->points.size() << ", Target: " << target_cloud_->points.size() << std::endl;
    }
    
    bool s2s_success = false;
    Eigen::Matrix4f T_S2S = Eigen::Matrix4f::Identity();
    
    // First attempt with motion prediction
    try {
        gicp_s2s_.align(*aligned, motion_prediction);
        
        if (gicp_s2s_.hasConverged() && gicp_s2s_.getFitnessScore() < 1.0) {
            s2s_success = true;
            T_S2S = gicp_s2s_.getFinalTransformation();
        }
    } catch (const std::exception& e) {
        std::cerr << "[C++/getNextPose()] S2S GICP alignment failed: " << e.what() << std::endl;
    }
    
    // Recovery attempt with identity initialization if first attempt failed
    if (!s2s_success) {
        try {
            gicp_s2s_.align(*aligned, Eigen::Matrix4f::Identity());
            
            if (gicp_s2s_.hasConverged() && gicp_s2s_.getFitnessScore() < 2.0) {
                s2s_success = true;
                T_S2S = gicp_s2s_.getFinalTransformation();
            }
        } catch (const std::exception& e) {
            std::cerr << "[C++/getNextPose()] S2S recovery failed: " << e.what() << std::endl;
        }
    }
    
    // Final fallback if both attempts failed
    if (!s2s_success) {
        std::cerr << "[C++/getNextPose()] Warning: S2S GICP did not converge! Using motion prediction. Fitness: " 
                  << gicp_s2s_.getFitnessScore() << std::endl;
        T_S2S = motion_prediction;
        
        // If motion prediction is also identity, use small incremental movement
        if (T_S2S.isApprox(Eigen::Matrix4f::Identity())) {
        }
    }
    
    // Validate S2S transformation (check for reasonable values)
    Eigen::Vector3f s2s_translation = T_S2S.block(0, 3, 3, 1);
    Eigen::Matrix3f s2s_rotation = T_S2S.block(0, 0, 3, 3);
    
    if (s2s_translation.norm() > keyframe_thresh_dist_ * 5.0f || 
        std::abs(s2s_rotation.determinant() - 1.0f) > 0.1f) {
        std::cerr << "[C++/getNextPose()] Warning: S2S transformation seems unreasonable, using motion prediction" << std::endl;
        T_S2S = motion_prediction;
    }
    
    // Get the global S2S transform
    propagateS2S(T_S2S);
    
    // FRAME-TO-SUBMAP
    getSubmapKeyframes();
    
    if (submap_hasChanged_) {
        // Set the current global submap as the target cloud
        gicp_.clearTarget();
        gicp_.setInputTarget(submap_cloud_);
    }
    
    // Set current scan as source for S2M alignment
    gicp_.clearSource();
    gicp_.setInputSource(current_scan_);
    
    // Align with current submap with global S2S transformation as initial guess
    bool s2m_success = false;
    try {
        gicp_.align(*aligned, T_s2s_);
        
        if (gicp_.hasConverged()) {
            s2m_success = true;
        } else {
            std::cerr << "[C++/getNextPose()] Warning: S2M GICP did not converge! Fitness: " 
                      << gicp_.getFitnessScore() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[C++/getNextPose()] S2M GICP alignment failed: " << e.what() << std::endl;
    }
    
    // Get final transformation in global frame
    if (s2m_success) {
        Eigen::Matrix4f T_candidate = gicp_.getFinalTransformation();
        
        // Validate S2M transformation
        Eigen::Vector3f s2m_translation = T_candidate.block(0, 3, 3, 1);
        Eigen::Matrix3f s2m_rotation = T_candidate.block(0, 0, 3, 3);
        
        if (s2m_translation.norm() < 100.0f && std::abs(s2m_rotation.determinant() - 1.0f) < 0.1f) {
            T_ = T_candidate;
        } else {
            std::cerr << "[C++/getNextPose()] Warning: S2M transformation unreasonable, using S2S result" << std::endl;
            T_ = T_s2s_;
        }
    } else {
        // Use S2S result as fallback
        T_ = T_s2s_;
    }
    
    // Update the S2S transform for next propagation
    T_s2s_prev_ = T_;
    
    // Update next global pose
    propagateS2M();
    
    // Update target cloud for next S2S iteration
    // The current scan becomes the target for the next frame
    target_cloud_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    *target_cloud_ = *current_scan_;
}

void DirectLidarOdometry::updateKeyframes() {
    // Transform point cloud
    transformCurrentScan();
    
    // Calculate difference in pose and rotation to all poses in trajectory
    float closest_d = std::numeric_limits<float>::infinity();
    int closest_idx = 0;
    int keyframes_idx = 0;
    int num_nearby = 0;
    
    for (const auto& k : keyframes_) {
        // Calculate distance between current pose and pose in keyframes
        float delta_d = sqrt(pow(pose_[0] - k.first.first[0], 2) + 
                           pow(pose_[1] - k.first.first[1], 2) + 
                           pow(pose_[2] - k.first.first[2], 2));
        
        // Count the number nearby current pose
        if (delta_d <= keyframe_thresh_dist_ * 1.5) {
            ++num_nearby;
        }
        
        // Store into variable
        if (delta_d < closest_d) {
            closest_d = delta_d;
            closest_idx = keyframes_idx;
        }
        
        keyframes_idx++;
    }
    
    // Get closest pose and corresponding rotation
    Eigen::Vector3f closest_pose = keyframes_[closest_idx].first.first;
    Eigen::Quaternionf closest_pose_r = keyframes_[closest_idx].first.second;
    
    // Calculate distance between current pose and closest pose from above
    float dd = sqrt(pow(pose_[0] - closest_pose[0], 2) + 
                   pow(pose_[1] - closest_pose[1], 2) + 
                   pow(pose_[2] - closest_pose[2], 2));
    
    // Calculate difference in orientation with improved precision
    Eigen::Quaternionf dq = rotq_ * (closest_pose_r.inverse());
    dq.normalize();
    
    // Ensure shortest path for angle calculation
    if (dq.w() < 0) {
        dq.coeffs() = -dq.coeffs();
    }
    
    // More robust angle calculation
    float w_clamped = std::min(std::abs(dq.w()), 1.0f);
    float theta_rad = 2.0f * std::acos(w_clamped);
    float theta_deg = theta_rad * (180.0f / M_PI);
    
    // Update keyframe with enhanced logic for turning scenarios
    bool newKeyframe = false;
    
    // Standard distance or rotation threshold
    if (dd > keyframe_thresh_dist_ || theta_deg > keyframe_thresh_rot_) {
        newKeyframe = true;
    }
    
    // Override: if very close in distance, don't create keyframe unless significant rotation AND sparse area
    if (dd <= keyframe_thresh_dist_ * 0.5f) {
        newKeyframe = false;
    }
    
    // Special case for turns: create keyframe for significant rotation in sparse areas
    if (dd <= keyframe_thresh_dist_ && theta_deg > keyframe_thresh_rot_ * 0.7f && num_nearby <= 1) {
        newKeyframe = true;
    }
    
    // Enhanced turning detection: check angular velocity for sharp turns
    if (motion_initialized_ && angular_velocity_.norm() > 1.0f) { // rad/s threshold for sharp turns
        if (dd > keyframe_thresh_dist_ * 0.5f || theta_deg > keyframe_thresh_rot_ * 0.5f) {
            newKeyframe = true;
        }
    }
    
    // Prevent too frequent keyframes during rapid motion
    static auto last_keyframe_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_since_last_kf = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_keyframe_time);
    
    if (newKeyframe && time_since_last_kf.count() < 100) { // Minimum 100ms between keyframes
        newKeyframe = false;
    }
    
    if (newKeyframe) {
        ++num_keyframes_;
        last_keyframe_time = current_time;
        
        // Voxelization for submap
        if (vf_submap_use_) {
            vf_submap_.setInputCloud(current_scan_t_);
            vf_submap_.filter(*current_scan_t_);
        }
        
        // Update keyframe vector
        keyframes_.push_back(std::make_pair(std::make_pair(pose_, rotq_), current_scan_t_));
        
        // Compute keyframe normals
        *keyframes_cloud_ += *current_scan_t_;
        *keyframe_cloud_ = *current_scan_t_;
        
        gicp_s2s_.setInputSource(keyframe_cloud_);
    }
}

void DirectLidarOdometry::setInputSources() {
    // This function is now simplified since we set sources directly in getNextPose
    // But we keep it for consistency with the existing interface
    
    // Set the input source for the S2S gicp (current scan)
    gicp_s2s_.setInputSource(current_scan_);
    
    // Set input source for S2M gicp (current scan)
    gicp_.setInputSource(current_scan_);
}

// PYTHON_BIND: Get current pose as 7-element vector [x, y, z, qw, qx, qy, qz]
std::vector<double> DirectLidarOdometry::getCurrentPose() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {pose_[0], pose_[1], pose_[2], rotq_.w(), rotq_.x(), rotq_.y(), rotq_.z()};
}

// PYTHON_BIND: Get current transformation matrix as 16-element vector (row-major)
std::vector<double> DirectLidarOdometry::getCurrentTransformation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<double> transform(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transform[i * 4 + j] = T_(i, j);
        }
    }
    return transform;
}

// PYTHON_BIND: Get all trajectory poses as vector of 7-element vectors
std::vector<std::vector<double>> DirectLidarOdometry::getTrajectoryPoses() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::vector<double>> poses;
    for (const auto& pose_pair : trajectory_) {
        std::vector<double> pose = {
            pose_pair.first[0], pose_pair.first[1], pose_pair.first[2],
            pose_pair.second.w(), pose_pair.second.x(), pose_pair.second.y(), pose_pair.second.z()
        };
        poses.push_back(pose);
    }
    return poses;
}

// PYTHON_BIND: Get number of keyframes
int DirectLidarOdometry::getNumKeyframes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return num_keyframes_;
}

// PYTHON_BIND: Reset the odometry system
void DirectLidarOdometry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    initialized_ = false;
    pose_ = Eigen::Vector3f::Zero();
    rotq_ = Eigen::Quaternionf::Identity();
    pose_s2s_ = Eigen::Vector3f::Zero();
    rotq_s2s_ = Eigen::Quaternionf::Identity();
    
    T_ = Eigen::Matrix4f::Identity();
    T_s2s_ = Eigen::Matrix4f::Identity();
    T_s2s_prev_ = Eigen::Matrix4f::Identity();
    
    // Reset motion model
    velocity_ = Eigen::Vector3f::Zero();
    angular_velocity_ = Eigen::Vector3f::Zero();
    prev_timestamp_ = 0.0;
    motion_initialized_ = false;
    
    source_cloud_ = nullptr;
    target_cloud_ = nullptr;
    
    keyframes_.clear();
    keyframe_normals_.clear();
    submap_normals_.clear();
    submap_kf_idx_curr_.clear();
    submap_kf_idx_prev_.clear();
    keyframe_convex_.clear();
    keyframe_concave_.clear();
    
    num_keyframes_ = 0;
    submap_hasChanged_ = true;
    
    trajectory_.clear();
    metrics_.spaciousness.clear();
    metrics_.computation_times.clear();
    metrics_.spaciousness.push_back(0.0);
    
    // Clear point clouds
    keyframe_cloud_->clear();
    keyframes_cloud_->clear();
    submap_cloud_->clear();
}

// PYTHON_BIND: Set initial pose [x, y, z, qw, qx, qy, qz]
void DirectLidarOdometry::setInitialPose(const std::vector<double>& pose_vec) {
    if (pose_vec.size() != 7) {
        std::cerr << "Error: Initial pose must be 7 elements [x, y, z, qw, qx, qy, qz]" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    initial_pose_use_ = true;
    initial_position_ = Eigen::Vector3f(pose_vec[0], pose_vec[1], pose_vec[2]);
    initial_orientation_ = Eigen::Quaternionf(pose_vec[3], pose_vec[4], pose_vec[5], pose_vec[6]);
}

// PYTHON_BIND: Configure basic parameters
void DirectLidarOdometry::configure(double keyframe_dist_thresh, 
    double keyframe_rot_thresh,
    bool enable_adaptive,
    bool use_voxel_filter,
    double voxel_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    keyframe_thresh_dist_ = keyframe_dist_thresh;
    keyframe_thresh_rot_ = keyframe_rot_thresh;
    adaptive_params_use_ = enable_adaptive;
    vf_scan_use_ = use_voxel_filter;
    vf_submap_use_ = use_voxel_filter;
    vf_scan_res_ = voxel_size;
    vf_submap_res_ = voxel_size;
    
    concave_hull_.setAlpha(keyframe_thresh_dist_);
    vf_scan_.setLeafSize(vf_scan_res_, vf_scan_res_, vf_scan_res_);
    vf_submap_.setLeafSize(vf_submap_res_, vf_submap_res_, vf_submap_res_);
}

// PYTHON_BIND: Configure motion model parameters
void DirectLidarOdometry::configureMotionModel(bool enable, double smoothing_factor) {
    std::lock_guard<std::mutex> lock(mutex_);
    motion_model_enabled_ = enable;
    motion_smoothing_factor_ = smoothing_factor;
}

// PYTHON_BIND: Get accumulated map (all keyframes)
pcl::PointCloud<PointType>::Ptr DirectLidarOdometry::getMap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_cloud_;
}

// PYTHON_BIND: Get computation statistics
std::vector<double> DirectLidarOdometry::getComputationTimes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_.computation_times;
}

// Keep existing configuration functions for advanced users
void DirectLidarOdometry::setKeyframeThresholds(double distance_thresh, double rotation_thresh) {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframe_thresh_dist_ = distance_thresh;
    keyframe_thresh_rot_ = rotation_thresh;
    concave_hull_.setAlpha(keyframe_thresh_dist_);
}

void DirectLidarOdometry::setSubmapParameters(int knn, int kcv, int kcc) {
    std::lock_guard<std::mutex> lock(mutex_);
    submap_knn_ = knn;
    submap_kcv_ = kcv;
    submap_kcc_ = kcc;
}

void DirectLidarOdometry::setVoxelFilterParameters(bool use_scan_filter, double scan_res, 
                                                  bool use_submap_filter, double submap_res) {
    std::lock_guard<std::mutex> lock(mutex_);
    vf_scan_use_ = use_scan_filter;
    vf_scan_res_ = scan_res;
    vf_submap_use_ = use_submap_filter;
    vf_submap_res_ = submap_res;
    
    vf_scan_.setLeafSize(vf_scan_res_, vf_scan_res_, vf_scan_res_);
    vf_submap_.setLeafSize(vf_submap_res_, vf_submap_res_, vf_submap_res_);
}

void DirectLidarOdometry::setCropBoxParameters(bool use_crop, double crop_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    crop_use_ = use_crop;
    crop_size_ = crop_size;
    
    crop_.setMin(Eigen::Vector4f(-crop_size_, -crop_size_, -crop_size_, 1.0));
    crop_.setMax(Eigen::Vector4f(crop_size_, crop_size_, crop_size_, 1.0));
}

void DirectLidarOdometry::setGICPParameters(int min_points, int max_iter_s2s, int max_iter_s2m,
                                           double transform_eps, double fitness_eps) {
    std::lock_guard<std::mutex> lock(mutex_);
    gicp_min_num_points_ = min_points;
    gicps2s_max_iter_ = max_iter_s2s;
    gicps2m_max_iter_ = max_iter_s2m;
    gicps2s_transformation_ep_ = transform_eps;
    gicps2m_transformation_ep_ = transform_eps;
    gicps2s_euclidean_fitness_ep_ = fitness_eps;
    gicps2m_euclidean_fitness_ep_ = fitness_eps;
    
    gicp_s2s_.setMaximumIterations(gicps2s_max_iter_);
    gicp_s2s_.setTransformationEpsilon(gicps2s_transformation_ep_);
    gicp_s2s_.setEuclideanFitnessEpsilon(gicps2s_euclidean_fitness_ep_);
    
    gicp_.setMaximumIterations(gicps2m_max_iter_);
    gicp_.setTransformationEpsilon(gicps2m_transformation_ep_);
    gicp_.setEuclideanFitnessEpsilon(gicps2m_euclidean_fitness_ep_);
}

void DirectLidarOdometry::setAdaptiveParameters(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    adaptive_params_use_ = enable;
}

void DirectLidarOdometry::setInitialPose(const Eigen::Vector3f& position, const Eigen::Quaternionf& orientation) {
    std::lock_guard<std::mutex> lock(mutex_);
    initial_pose_use_ = true;
    initial_position_ = position;
    initial_orientation_ = orientation;
}

std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> DirectLidarOdometry::getTrajectory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return trajectory_;
}

pcl::PointCloud<PointType>::Ptr DirectLidarOdometry::getKeyframesCloud() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_cloud_;
}

void DirectLidarOdometry::initialize() {
    if (initial_pose_use_) {
        // Set known position
        pose_ = initial_position_;
        T_.block(0,3,3,1) = pose_;
        T_s2s_.block(0,3,3,1) = pose_;
        T_s2s_prev_.block(0,3,3,1) = pose_;
        
        // Set known orientation
        rotq_ = initial_orientation_;
        T_.block(0,0,3,3) = rotq_.toRotationMatrix();
        T_s2s_.block(0,0,3,3) = rotq_.toRotationMatrix();
        T_s2s_prev_.block(0,0,3,3) = rotq_.toRotationMatrix();
    }
    
    initialized_ = true;
}

void DirectLidarOdometry::preprocessPoints() {
    // Original scan
    *original_scan_ = *current_scan_;
    
    // Remove NaNs
    std::vector<int> idx;
    current_scan_->is_dense = false;
    pcl::removeNaNFromPointCloud(*current_scan_, *current_scan_, idx);
    
    // Deskew point cloud (assuming 10Hz spinning LiDAR)
    deskewPointCloud();
    
    // Crop box filter
    if (crop_use_) {
        crop_.setInputCloud(current_scan_);
        crop_.filter(*current_scan_);
    }
    
    // Voxel grid filter
    if (vf_scan_use_) {
        vf_scan_.setInputCloud(current_scan_);
        vf_scan_.filter(*current_scan_);
    }
}

void DirectLidarOdometry::deskewPointCloud() {
    if (!motion_initialized_ || current_scan_->points.empty()) {
        return;
    }
    
    const double scan_period = 0.1; // 10Hz = 0.1 seconds per scan
    
    // Estimate angular velocity magnitude for deskewing
    float angular_speed = angular_velocity_.norm();
    float linear_speed = velocity_.norm();
    
    // Skip deskewing if motion is very small
    if (angular_speed < 0.01 && linear_speed < 0.1) {
        return;
    }
    
    // Deskew each point based on its relative timestamp within the scan
    for (size_t i = 0; i < current_scan_->points.size(); ++i) {
        PointType& point = current_scan_->points[i];
        
        // Calculate relative time within scan (0 to scan_period)
        // Assuming points are ordered by acquisition time
        double relative_time = (static_cast<double>(i) / current_scan_->points.size()) * scan_period;
        
        // Create transformation for this point's timestamp
        Eigen::Matrix4f point_transform = Eigen::Matrix4f::Identity();
        
        // Apply linear motion compensation
        Eigen::Vector3f linear_displacement = velocity_ * relative_time;
        point_transform.block(0, 3, 3, 1) = -linear_displacement; // Negative to undo motion
        
        // Apply angular motion compensation
        if (angular_speed > 0.01) {
            Eigen::Vector3f angular_displacement = angular_velocity_ * relative_time;
            float angle = angular_displacement.norm();
            
            if (angle > 1e-6) {
                Eigen::Vector3f axis = angular_displacement / angle;
                
                // Use Rodrigues' formula for rotation matrix
                Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
                K(0, 1) = -axis(2); K(0, 2) = axis(1);
                K(1, 0) = axis(2);  K(1, 2) = -axis(0);
                K(2, 0) = -axis(1); K(2, 1) = axis(0);
                
                // Negative angle to undo rotation
                Eigen::Matrix3f R = Eigen::Matrix3f::Identity() - std::sin(angle) * K + (1.0f - std::cos(angle)) * K * K;
                point_transform.block(0, 0, 3, 3) = R;
            }
        }
        
        // Apply transformation to deskew the point
        Eigen::Vector4f point_vec(point.x, point.y, point.z, 1.0f);
        Eigen::Vector4f deskewed_point = point_transform * point_vec;
        
        point.x = deskewed_point(0);
        point.y = deskewed_point(1);
        point.z = deskewed_point(2);
    }
}


void DirectLidarOdometry::initializeInputTarget() {
    // Convert current scan to target cloud for first frame
    target_cloud_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    *target_cloud_ = *current_scan_;
    
    // Set initial target for S2S (will be updated in subsequent frames)
    gicp_s2s_.setInputTarget(target_cloud_);
    
    // Initialize keyframes
    pcl::PointCloud<PointType>::Ptr first_keyframe(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*current_scan_, *first_keyframe, T_);
    
    // Voxelization for submap
    if (vf_submap_use_) {
        vf_submap_.setInputCloud(first_keyframe);
        vf_submap_.filter(*first_keyframe);
    }
    
    // Keep history of keyframes
    keyframes_.push_back(std::make_pair(std::make_pair(pose_, rotq_), first_keyframe));
    *keyframes_cloud_ += *first_keyframe;
    *keyframe_cloud_ = *first_keyframe;
    
    ++num_keyframes_;
}


void DirectLidarOdometry::propagateS2S(const Eigen::Matrix4f& T) {
    T_s2s_ = T_s2s_prev_ * T;
    T_s2s_prev_ = T_s2s_;
    
    pose_s2s_ << T_s2s_(0,3), T_s2s_(1,3), T_s2s_(2,3);
    rotSO3_s2s_ << T_s2s_(0,0), T_s2s_(0,1), T_s2s_(0,2),
                   T_s2s_(1,0), T_s2s_(1,1), T_s2s_(1,2),
                   T_s2s_(2,0), T_s2s_(2,1), T_s2s_(2,2);
    
    Eigen::Quaternionf q(rotSO3_s2s_);
    
    // Normalize quaternion
    double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
    q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
    rotq_s2s_ = q;
}

void DirectLidarOdometry::propagateS2M() {
    pose_ << T_(0,3), T_(1,3), T_(2,3);
    rotSO3_ << T_(0,0), T_(0,1), T_(0,2),
               T_(1,0), T_(1,1), T_(1,2),
               T_(2,0), T_(2,1), T_(2,2);
    
    Eigen::Quaternionf q(rotSO3_);
    
    // Normalize quaternion
    double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
    q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
    rotq_ = q;
}

void DirectLidarOdometry::transformCurrentScan() {
    current_scan_t_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*current_scan_, *current_scan_t_, T_);
}

void DirectLidarOdometry::computeMetrics() {
    computeSpaciousness();
}

void DirectLidarOdometry::computeSpaciousness() {
    // Compute range of points
    std::vector<float> ds;
    
    for (size_t i = 0; i < current_scan_->points.size(); i++) {
        float d = std::sqrt(pow(current_scan_->points[i].x, 2) + 
                           pow(current_scan_->points[i].y, 2) + 
                           pow(current_scan_->points[i].z, 2));
        ds.push_back(d);
    }
    
    if (ds.empty()) return;
    
    // Median
    std::nth_element(ds.begin(), ds.begin() + ds.size()/2, ds.end());
    float median_curr = ds[ds.size()/2];
    static float median_prev = median_curr;
    float median_lpf = 0.95f * median_prev + 0.05f * median_curr;
    median_prev = median_lpf;
    
    // Push
    metrics_.spaciousness.push_back(median_lpf);
}

void DirectLidarOdometry::computeConvexHull() {
    // At least 4 keyframes for convex hull
    if (num_keyframes_ < 4) {
        return;
    }
    
    // Create a pointcloud with points at keyframes
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    
    for (const auto& k : keyframes_) {
        PointType pt;
        pt.x = k.first.first[0];
        pt.y = k.first.first[1];
        pt.z = k.first.first[2];
        cloud->push_back(pt);
    }
    
    // Calculate the convex hull of the point cloud
    convex_hull_.setInputCloud(cloud);
    
    // Get the indices of the keyframes on the convex hull
    pcl::PointCloud<PointType>::Ptr convex_points(new pcl::PointCloud<PointType>);
    convex_hull_.reconstruct(*convex_points);
    
    pcl::PointIndices::Ptr convex_hull_point_idx(new pcl::PointIndices);
    convex_hull_.getHullPointIndices(*convex_hull_point_idx);
    
    keyframe_convex_.clear();
    for (size_t i = 0; i < convex_hull_point_idx->indices.size(); ++i) {
        keyframe_convex_.push_back(convex_hull_point_idx->indices[i]);
    }
}

void DirectLidarOdometry::computeConcaveHull() {
    // At least 5 keyframes for concave hull
    if (num_keyframes_ < 5) {
        return;
    }
    
    // Create a pointcloud with points at keyframes
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    
    for (const auto& k : keyframes_) {
        PointType pt;
        pt.x = k.first.first[0];
        pt.y = k.first.first[1];
        pt.z = k.first.first[2];
        cloud->push_back(pt);
    }
    
    // Calculate the concave hull of the point cloud
    concave_hull_.setInputCloud(cloud);
    
    // Get the indices of the keyframes on the concave hull
    pcl::PointCloud<PointType>::Ptr concave_points(new pcl::PointCloud<PointType>);
    concave_hull_.reconstruct(*concave_points);
    
    pcl::PointIndices::Ptr concave_hull_point_idx(new pcl::PointIndices);
    concave_hull_.getHullPointIndices(*concave_hull_point_idx);
    
    keyframe_concave_.clear();
    for (size_t i = 0; i < concave_hull_point_idx->indices.size(); ++i) {
        keyframe_concave_.push_back(concave_hull_point_idx->indices[i]);
    }
}

void DirectLidarOdometry::setAdaptiveParams() {
    // Set keyframe thresh from spaciousness metric
    if (metrics_.spaciousness.back() > 20.0) {
        keyframe_thresh_dist_ = 10.0;
    } else if (metrics_.spaciousness.back() > 10.0 && metrics_.spaciousness.back() <= 20.0) {
        keyframe_thresh_dist_ = 5.0;
    } else if (metrics_.spaciousness.back() > 5.0 && metrics_.spaciousness.back() <= 10.0) {
        keyframe_thresh_dist_ = 1.0;
    } else if (metrics_.spaciousness.back() <= 5.0) {
        keyframe_thresh_dist_ = 0.5;
    }
    
    // Set concave hull alpha
    concave_hull_.setAlpha(keyframe_thresh_dist_);
}

void DirectLidarOdometry::pushSubmapIndices(const std::vector<float>& dists, int k, 
                                           const std::vector<int>& frames) {
    // Make sure dists is not empty
    if (dists.empty()) return;
    
    // Maintain max heap of at most k elements
    std::priority_queue<float> pq;
    
    for (auto d : dists) {
        if (pq.size() >= static_cast<size_t>(k) && pq.top() > d) {
            pq.push(d);
            pq.pop();
        } else if (pq.size() < static_cast<size_t>(k)) {
            pq.push(d);
        }
    }
    
    // Get the kth smallest element, which should be at the top of the heap
    float kth_element = pq.top();
    
    // Get all elements smaller or equal to the kth smallest element
    for (size_t i = 0; i < dists.size(); ++i) {
        if (dists[i] <= kth_element)
            submap_kf_idx_curr_.push_back(frames[i]);
    }
}

void DirectLidarOdometry::getSubmapKeyframes() {
    // Clear vector of keyframe indices to use for submap
    submap_kf_idx_curr_.clear();
    
    // TOP K NEAREST NEIGHBORS FROM ALL KEYFRAMES
    std::vector<float> ds;
    std::vector<int> keyframe_nn;
    int i = 0;
    Eigen::Vector3f curr_pose = T_s2s_.block(0,3,3,1);
    
    for (const auto& k : keyframes_) {
        float d = sqrt(pow(curr_pose[0] - k.first.first[0], 2) + 
                      pow(curr_pose[1] - k.first.first[1], 2) + 
                      pow(curr_pose[2] - k.first.first[2], 2));
        ds.push_back(d);
        keyframe_nn.push_back(i);
        i++;
    }
    
    // Get indices for top K nearest neighbor keyframe poses
    pushSubmapIndices(ds, submap_knn_, keyframe_nn);
    
    // TOP K NEAREST NEIGHBORS FROM CONVEX HULL
    computeConvexHull();
    
    // Get distances for each keyframe on convex hull
    std::vector<float> convex_ds;
    for (const auto& c : keyframe_convex_) {
        convex_ds.push_back(ds[c]);
    }
    
    // Get indices for top kNN for convex hull
    pushSubmapIndices(convex_ds, submap_kcv_, keyframe_convex_);
    
    // TOP K NEAREST NEIGHBORS FROM CONCAVE HULL
    computeConcaveHull();
    
    // Get distances for each keyframe on concave hull
    std::vector<float> concave_ds;
    for (const auto& c : keyframe_concave_) {
        concave_ds.push_back(ds[c]);
    }
    
    // Get indices for top kNN for concave hull
    pushSubmapIndices(concave_ds, submap_kcc_, keyframe_concave_);
    
    // BUILD SUBMAP
    std::sort(submap_kf_idx_curr_.begin(), submap_kf_idx_curr_.end());
    auto last = std::unique(submap_kf_idx_curr_.begin(), submap_kf_idx_curr_.end());
    submap_kf_idx_curr_.erase(last, submap_kf_idx_curr_.end());
    
    // Sort current and previous submap kf list of indices
    std::sort(submap_kf_idx_curr_.begin(), submap_kf_idx_curr_.end());
    std::sort(submap_kf_idx_prev_.begin(), submap_kf_idx_prev_.end());
    
    // Check if submap has changed from previous iteration
    if (submap_kf_idx_curr_ == submap_kf_idx_prev_) {
        submap_hasChanged_ = false;
    } else {
        submap_hasChanged_ = true;
        
        // Reinitialize submap cloud
        submap_cloud_ = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        submap_normals_.clear();
        
        for (auto k : submap_kf_idx_curr_) {
            // Create current submap cloud
            *submap_cloud_ += *keyframes_[k].second;
        }
        submap_kf_idx_prev_ = submap_kf_idx_curr_;
    }
}

// PYTHON_BIND: Parallel point cloud alignment of source RGB point clouds against target RGB point clouds
std::pair<std::vector<std::vector<double>>, std::vector<pcl::PointCloud<PointType>::Ptr>>
DirectLidarOdometry::refine(const std::vector<pcl::PointCloud<PointType>::Ptr>& source_clouds,
                         const std::vector<pcl::PointCloud<PointType>::Ptr>& target_clouds,
                         int max_iterations,
                         double transformation_epsilon,
                         double euclidean_fitness_epsilon,
                         double max_correspondence_distance) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (source_clouds.empty() || target_clouds.empty() || source_clouds.size() != target_clouds.size()) {
        return {{}, {}};
    }
    
    // Check if keyframe cloud is available
    if (!keyframe_cloud_ || keyframe_cloud_->empty()) {
        return {{}, {}};
    }
    
    // Create merged reference cloud (keyframe + submap)
    pcl::PointCloud<PointType>::Ptr merged_reference_cloud(new pcl::PointCloud<PointType>);
    *merged_reference_cloud = *keyframe_cloud_;
    
    // Add submap cloud if available
    if (submap_cloud_ && !submap_cloud_->empty()) {
        *merged_reference_cloud += *submap_cloud_;
    }
    
    const size_t num_pairs = source_clouds.size();
    std::vector<std::vector<double>> refined_poses(num_pairs);
    std::vector<pcl::PointCloud<PointType>::Ptr> aligned_clouds(num_pairs);
    
    // Pre-allocate results storage
    for (size_t i = 0; i < num_pairs; ++i) {
        aligned_clouds[i] = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        refined_poses[i].resize(16);
    }
    
    // Always use identity as initial guess
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    
    // Use hardware concurrency for optimal thread count
    const size_t hardware_threads = std::thread::hardware_concurrency();
    const size_t num_threads = std::min(num_pairs, hardware_threads > 0 ? hardware_threads : 4);
    
    // Parallel alignment using thread pool pattern
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    std::atomic<size_t> pair_index{0};
    std::atomic<size_t> success_count{0};
    std::atomic<size_t> failure_count{0};
    
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            // Multiscale approach: coarse to fine
            std::vector<float> voxel_scales = {0.2f, 0.1f, 0.05f}; // Coarse to fine scales
            std::vector<int> scale_iterations = {max_iterations/4, max_iterations/4, max_iterations/4}; // Distribute iterations (reserve some for final alignment)
            std::vector<double> scale_correspondence_dist = {
                max_correspondence_distance * 2.0,  // Coarse scale - larger correspondence distance
                max_correspondence_distance * 1.5,  // Medium scale
                max_correspondence_distance         // Fine scale
            };
            
            // Thread-local GICP instances for different stages
            std::vector<nano_gicp::NanoGICP<PointType, PointType>> gicp_to_reference(voxel_scales.size());
            nano_gicp::NanoGICP<PointType, PointType> gicp_source_to_target;
            std::vector<pcl::VoxelGrid<PointType>> voxel_filters(voxel_scales.size());
            
            // Configure GICP instances for reference cloud alignment
            for (size_t scale = 0; scale < voxel_scales.size(); ++scale) {
                gicp_to_reference[scale].setMaximumIterations(scale_iterations[scale]);
                gicp_to_reference[scale].setTransformationEpsilon(transformation_epsilon * (scale + 1));
                gicp_to_reference[scale].setEuclideanFitnessEpsilon(euclidean_fitness_epsilon * (scale + 1));
                gicp_to_reference[scale].setMaxCorrespondenceDistance(scale_correspondence_dist[scale]);
                
                voxel_filters[scale].setLeafSize(voxel_scales[scale], voxel_scales[scale], voxel_scales[scale]);
            }
            
            // Configure GICP for final source-to-target alignment
            gicp_source_to_target.setMaximumIterations(max_iterations/4);
            gicp_source_to_target.setTransformationEpsilon(transformation_epsilon);
            gicp_source_to_target.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);
            gicp_source_to_target.setMaxCorrespondenceDistance(max_correspondence_distance);
            
            size_t i;
            while ((i = pair_index.fetch_add(1)) < num_pairs) {
                try {
                    // Validate input clouds
                    if (!source_clouds[i] || source_clouds[i]->empty() || 
                        !target_clouds[i] || target_clouds[i]->empty()) {
                        // Set identity transformation
                        Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
                        for (int row = 0; row < 4; ++row) {
                            for (int col = 0; col < 4; ++col) {
                                refined_poses[i][row * 4 + col] = identity(row, col);
                            }
                        }
                        if (source_clouds[i] && !source_clouds[i]->empty()) {
                            *aligned_clouds[i] = *source_clouds[i];
                        }
                        failure_count++;
                        continue;
                    }
                    
                    // Phase 1: Align both source and target clouds to merged reference cloud
                    Eigen::Matrix4f source_to_reference_transform = Eigen::Matrix4f::Identity();
                    Eigen::Matrix4f target_to_reference_transform = Eigen::Matrix4f::Identity();
                    bool source_aligned_success = false;
                    bool target_aligned_success = false;
                    
                    // Align source cloud to merged reference cloud
                    for (size_t scale = 0; scale < voxel_scales.size(); ++scale) {
                        try {
                            // Downsample source cloud and merged reference cloud for current scale
                            pcl::PointCloud<PointType>::Ptr downsampled_source(new pcl::PointCloud<PointType>);
                            pcl::PointCloud<PointType>::Ptr downsampled_reference(new pcl::PointCloud<PointType>);
                            
                            voxel_filters[scale].setInputCloud(source_clouds[i]);
                            voxel_filters[scale].filter(*downsampled_source);
                            
                            voxel_filters[scale].setInputCloud(merged_reference_cloud);
                            voxel_filters[scale].filter(*downsampled_reference);
                            
                            if (downsampled_source->size() < 50 || downsampled_reference->size() < 50) {
                                continue;
                            }
                            
                            gicp_to_reference[scale].setInputSource(downsampled_source);
                            gicp_to_reference[scale].setInputTarget(downsampled_reference);
                            
                            pcl::PointCloud<PointType> aligned_source_to_reference;
                            gicp_to_reference[scale].align(aligned_source_to_reference, source_to_reference_transform);
                            
                            if (gicp_to_reference[scale].hasConverged()) {
                                Eigen::Matrix4f candidate_transform = gicp_to_reference[scale].getFinalTransformation();
                                Eigen::Vector3f translation = candidate_transform.block(0, 3, 3, 1);
                                Eigen::Matrix3f rotation = candidate_transform.block(0, 0, 3, 3);
                                
                                float max_translation = (scale == 0) ? 5.0f : (scale == 1) ? 3.0f : 2.0f;
                                if (translation.norm() <= max_translation && 
                                    std::abs(rotation.determinant() - 1.0f) <= 0.3f) {
                                    source_to_reference_transform = candidate_transform;
                                    source_aligned_success = true;
                                    break;
                                }
                            }
                        } catch (const std::exception& e) {}
                    }
                    
                    // Align target cloud to merged reference cloud
                    for (size_t scale = 0; scale < voxel_scales.size(); ++scale) {
                        try {
                            // Downsample target cloud and merged reference cloud for current scale
                            pcl::PointCloud<PointType>::Ptr downsampled_target(new pcl::PointCloud<PointType>);
                            pcl::PointCloud<PointType>::Ptr downsampled_reference(new pcl::PointCloud<PointType>);
                            
                            voxel_filters[scale].setInputCloud(target_clouds[i]);
                            voxel_filters[scale].filter(*downsampled_target);
                            
                            voxel_filters[scale].setInputCloud(merged_reference_cloud);
                            voxel_filters[scale].filter(*downsampled_reference);
                            
                            if (downsampled_target->size() < 50 || downsampled_reference->size() < 50) {
                                continue;
                            }
                            
                            gicp_to_reference[scale].setInputSource(downsampled_target);
                            gicp_to_reference[scale].setInputTarget(downsampled_reference);
                            
                            pcl::PointCloud<PointType> aligned_target_to_reference;
                            gicp_to_reference[scale].align(aligned_target_to_reference, target_to_reference_transform);
                            
                            if (gicp_to_reference[scale].hasConverged()) {
                                Eigen::Matrix4f candidate_transform = gicp_to_reference[scale].getFinalTransformation();
                                Eigen::Vector3f translation = candidate_transform.block(0, 3, 3, 1);
                                Eigen::Matrix3f rotation = candidate_transform.block(0, 0, 3, 3);
                                
                                float max_translation = (scale == 0) ? 5.0f : (scale == 1) ? 3.0f : 2.0f;
                                if (translation.norm() <= max_translation && 
                                    std::abs(rotation.determinant() - 1.0f) <= 0.3f) {
                                    target_to_reference_transform = candidate_transform;
                                    target_aligned_success = true;
                                    break;
                                }
                            }
                        } catch (const std::exception& e) {}
                    }
                    
                    // Phase 2: Apply reference alignments and perform source-to-target alignment
                    // Apply reference transformations to get aligned versions
                    pcl::PointCloud<PointType>::Ptr source_aligned_cloud(new pcl::PointCloud<PointType>);
                    pcl::PointCloud<PointType>::Ptr target_aligned_cloud(new pcl::PointCloud<PointType>);
                    
                    if (source_aligned_success) {
                        pcl::transformPointCloud(*source_clouds[i], *source_aligned_cloud, source_to_reference_transform);
                    } else {
                        *source_aligned_cloud = *source_clouds[i];
                    }
                    
                    if (target_aligned_success) {
                        pcl::transformPointCloud(*target_clouds[i], *target_aligned_cloud, target_to_reference_transform);
                    } else {
                        *target_aligned_cloud = *target_clouds[i];
                    }
                    
                    // Now perform source-to-target alignment in the reference frame
                    bool final_alignment_successful = false;
                    Eigen::Matrix4f source_to_target_transform = Eigen::Matrix4f::Identity();
                    
                    try {
                        // Use finest scale for final alignment
                        pcl::PointCloud<PointType>::Ptr final_source(new pcl::PointCloud<PointType>);
                        pcl::PointCloud<PointType>::Ptr final_target(new pcl::PointCloud<PointType>);
                        
                        voxel_filters[2].setInputCloud(source_aligned_cloud);
                        voxel_filters[2].filter(*final_source);
                        
                        voxel_filters[2].setInputCloud(target_aligned_cloud);
                        voxel_filters[2].filter(*final_target);
                        
                        if (final_source->size() >= 50 && final_target->size() >= 50) {
                            gicp_source_to_target.setInputSource(final_source);
                            gicp_source_to_target.setInputTarget(final_target);
                            
                            pcl::PointCloud<PointType> final_aligned;
                            gicp_source_to_target.align(final_aligned, initial_guess);
                            
                            if (gicp_source_to_target.hasConverged()) {
                                Eigen::Matrix4f candidate_transform = gicp_source_to_target.getFinalTransformation();
                                Eigen::Vector3f translation = candidate_transform.block(0, 3, 3, 1);
                                Eigen::Matrix3f rotation = candidate_transform.block(0, 0, 3, 3);
                                
                                if (translation.norm() <= 2.0f && std::abs(rotation.determinant() - 1.0f) <= 0.2f) {
                                    source_to_target_transform = candidate_transform;
                                    final_alignment_successful = true;
                                }
                            }
                        }
                    } catch (const std::exception& e) {}
                    
                    // Compute final transformation: source -> reference -> target
                    Eigen::Matrix4f final_transform;
                    if (final_alignment_successful && source_aligned_success && target_aligned_success) {
                        // Complete pipeline: source -> reference -> target-in-reference-frame -> final-alignment
                        final_transform = source_to_target_transform * source_to_reference_transform;
                        success_count++;
                    } else if (source_aligned_success && target_aligned_success) {
                        // Use reference alignments only (source aligned to same frame as target)
                        final_transform = target_to_reference_transform.inverse() * source_to_reference_transform;
                        success_count++;
                    } else {
                        // Fallback to identity
                        final_transform = Eigen::Matrix4f::Identity();
                        failure_count++;
                    }
                    
                    // Apply final transformation to original source cloud
                    pcl::transformPointCloud(*source_clouds[i], *aligned_clouds[i], final_transform);
                    
                    // Store results (transformation matrix in row-major format)
                    for (int row = 0; row < 4; ++row) {
                        for (int col = 0; col < 4; ++col) {
                            refined_poses[i][row * 4 + col] = final_transform(row, col);
                        }
                    }
                    
                } catch (const std::exception& e) {
                    std::cout << "[C++/refine()] Pair " << i << ": Exception during alignment: " << e.what() 
                              << " - using identity" << std::endl;
                    // Set identity as fallback
                    for (int row = 0; row < 4; ++row) {
                        for (int col = 0; col < 4; ++col) {
                            refined_poses[i][row * 4 + col] = initial_guess(row, col);
                        }
                    }
                    if (source_clouds[i] && !source_clouds[i]->empty()) {
                        pcl::transformPointCloud(*source_clouds[i], *aligned_clouds[i], initial_guess);
                    }
                    failure_count++;
                } catch (...) {
                    std::cout << "[C++/refine()] Pair " << i << ": Unknown exception during alignment - using identity" << std::endl;
                    // Set identity as fallback
                    for (int row = 0; row < 4; ++row) {
                        for (int col = 0; col < 4; ++col) {
                            refined_poses[i][row * 4 + col] = initial_guess(row, col);
                        }
                    }
                    if (source_clouds[i] && !source_clouds[i]->empty()) {
                        pcl::transformPointCloud(*source_clouds[i], *aligned_clouds[i], initial_guess);
                    }
                    failure_count++;
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    return {refined_poses, aligned_clouds};
}

// PYTHON_BIND: Fast density-aware normal computation for point clouds
std::vector<std::vector<double>> DirectLidarOdometry::computeNormals(const pcl::PointCloud<PointType>::Ptr& cloud,
    int k_neighbors,
    double radius) {
    if (!cloud || cloud->empty()) {
        return {};
    }
    
    const size_t num_points = cloud->points.size();
    std::vector<std::vector<double>> normals(num_points, std::vector<double>(3, 0.0));
    
    // Use hardware concurrency for optimal thread count
    const size_t hardware_threads = std::thread::hardware_concurrency();
    const size_t num_threads = std::min(num_points, hardware_threads > 0 ? hardware_threads : 4);
    
    // Fast KdTree for neighbor search
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    tree->setInputCloud(cloud);
    
    // Parallel normal computation
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    std::atomic<size_t> point_index{0};
    
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            std::vector<int> indices;
            std::vector<float> distances;
            
            size_t i;
            while ((i = point_index.fetch_add(1)) < num_points) {
                try {
                    const PointType& point = cloud->points[i];
                    
                    // Skip invalid points
                    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                        continue;
                    }
                    
                    indices.clear();
                    distances.clear();
                    
                    // Density-aware: adapt k/radius based on local density
                    int local_k = k_neighbors;
                    double local_radius = radius;
                    // Estimate local density by counting points in a small radius
                    int density_neighbors = tree->radiusSearch(point, radius * 0.5, indices, distances);
                    if (density_neighbors > 20) {
                        // High density: use smaller radius or more neighbors
                        local_radius = radius * 0.7;
                        local_k = std::max(8, k_neighbors / 2);
                    } else if (density_neighbors < 5) {
                        // Low density: use larger radius or more neighbors
                        local_radius = radius * 1.5;
                        local_k = std::max(k_neighbors, 15);
                    }
                    
                    indices.clear();
                    distances.clear();
                    int found_neighbors = 0;
                    if (local_radius > 0.0) {
                        found_neighbors = tree->radiusSearch(point, local_radius, indices, distances);
                    } else {
                        found_neighbors = tree->nearestKSearch(point, local_k, indices, distances);
                    }
                    
                    if (found_neighbors < 3) {
                        continue; // Need at least 3 points for normal estimation
                    }
                    
                    // Compute covariance matrix
                    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
                    for (int idx : indices) {
                        const PointType& neighbor = cloud->points[idx];
                        centroid += Eigen::Vector3f(neighbor.x, neighbor.y, neighbor.z);
                    }
                    centroid /= static_cast<float>(indices.size());
                    
                    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
                    for (int idx : indices) {
                        const PointType& neighbor = cloud->points[idx];
                        Eigen::Vector3f diff(neighbor.x - centroid.x(), 
                                           neighbor.y - centroid.y(), 
                                           neighbor.z - centroid.z());
                        covariance += diff * diff.transpose();
                    }
                    covariance /= static_cast<float>(indices.size() - 1);
                    
                    // Compute eigenvalues and eigenvectors
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
                    if (solver.info() != Eigen::Success) {
                        continue;
                    }
                    
                    // Normal is the eigenvector with smallest eigenvalue
                    Eigen::Vector3f normal = solver.eigenvectors().col(0);
                    
                    // Ensure consistent orientation (towards viewpoint at origin)
                    if (normal.dot(Eigen::Vector3f(point.x, point.y, point.z)) > 0) {
                        normal = -normal;
                    }
                    
                    // Normalize
                    normal.normalize();
                    
                    // Store result
                    normals[i][0] = normal.x();
                    normals[i][1] = normal.y();
                    normals[i][2] = normal.z();
                    
                } catch (...) {
                    // Skip point on any error
                    continue;
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    return normals;
}

// PYTHON_BIND: Fast multiscale robust point cloud registration
std::tuple<std::vector<double>, double, bool> DirectLidarOdometry::alignPointClouds(
    const pcl::PointCloud<PointType>::Ptr& source_cloud,
    const pcl::PointCloud<PointType>::Ptr& target_cloud,
    const std::vector<double>& initial_guess_vec,
    int max_iterations,
    double transformation_epsilon,
    double euclidean_fitness_epsilon,
    double max_correspondence_distance) {
    
    // Initialize return values
    std::vector<double> transform_matrix(16);
    double fitness_score = std::numeric_limits<double>::max();
    bool has_converged = false;
    
    // Set identity as default
    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transform_matrix[i * 4 + j] = identity(i, j);
        }
    }
    
    // Validate input clouds
    if (!source_cloud || source_cloud->empty() || 
        !target_cloud || target_cloud->empty()) {
        return {transform_matrix, fitness_score, has_converged};
    }

    // Convert initial guess vector to Eigen::Matrix4f
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    if (initial_guess_vec.size() == 16) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                initial_guess(i, j) = initial_guess_vec[i * 4 + j];
            }
        }
    }
    
    try {
        // Multiscale approach: coarse to fine
        std::vector<float> voxel_scales = {0.2f, 0.1f, 0.05f};
        std::vector<int> scale_iterations = {max_iterations/4, max_iterations/4, max_iterations/4};
        std::vector<double> scale_correspondence_dist = {
            max_correspondence_distance * 2.0,  // Coarse scale
            max_correspondence_distance * 1.5,  // Medium scale
            max_correspondence_distance         // Fine scale
        };
        
        // Initialize GICP instances and voxel filters
        std::vector<nano_gicp::NanoGICP<PointType, PointType>> gicp_instances(voxel_scales.size());
        std::vector<pcl::VoxelGrid<PointType>> voxel_filters(voxel_scales.size());
        
        // Configure GICP instances for each scale
        for (size_t scale = 0; scale < voxel_scales.size(); ++scale) {
            gicp_instances[scale].setMaximumIterations(scale_iterations[scale]);
            gicp_instances[scale].setTransformationEpsilon(transformation_epsilon * (scale + 1));
            gicp_instances[scale].setEuclideanFitnessEpsilon(euclidean_fitness_epsilon * (scale + 1));
            gicp_instances[scale].setMaxCorrespondenceDistance(scale_correspondence_dist[scale]);
            
            voxel_filters[scale].setLeafSize(voxel_scales[scale], voxel_scales[scale], voxel_scales[scale]);
        }
        
        Eigen::Matrix4f current_transform = initial_guess;
        
        // Multiscale alignment from coarse to fine
        for (size_t scale = 0; scale < voxel_scales.size(); ++scale) {
            // Downsample clouds for current scale
            pcl::PointCloud<PointType>::Ptr downsampled_source(new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr downsampled_target(new pcl::PointCloud<PointType>);
            
            voxel_filters[scale].setInputCloud(source_cloud);
            voxel_filters[scale].filter(*downsampled_source);
            
            voxel_filters[scale].setInputCloud(target_cloud);
            voxel_filters[scale].filter(*downsampled_target);
            
            // Check minimum point requirements
            if (downsampled_source->size() < 50 || downsampled_target->size() < 50) {
                continue;
            }
            
            // Set up GICP
            gicp_instances[scale].setInputSource(downsampled_source);
            gicp_instances[scale].setInputTarget(downsampled_target);
            
            // Perform alignment
            pcl::PointCloud<PointType> aligned_cloud;
            gicp_instances[scale].align(aligned_cloud, current_transform);
            
            // Check convergence and validate transformation
            if (gicp_instances[scale].hasConverged()) {
                Eigen::Matrix4f candidate_transform = gicp_instances[scale].getFinalTransformation();
                Eigen::Vector3f translation = candidate_transform.block(0, 3, 3, 1);
                Eigen::Matrix3f rotation = candidate_transform.block(0, 0, 3, 3);
                
                // Validate transformation bounds based on scale
                float max_translation = (scale == 0) ? 5.0f : (scale == 1) ? 3.0f : 2.0f;
                if (translation.norm() <= max_translation && 
                    std::abs(rotation.determinant() - 1.0f) <= 0.3f) {
                    current_transform = candidate_transform;
                    has_converged = true;
                    fitness_score = gicp_instances[scale].getFitnessScore();
                }
            }
        }
        
        // Final refinement with original resolution if we have a good initial alignment
        if (has_converged) {
            nano_gicp::NanoGICP<PointType, PointType> final_gicp;
            final_gicp.setMaximumIterations(max_iterations/4);
            final_gicp.setTransformationEpsilon(transformation_epsilon);
            final_gicp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);
            final_gicp.setMaxCorrespondenceDistance(max_correspondence_distance);
            
            final_gicp.setInputSource(source_cloud);
            final_gicp.setInputTarget(target_cloud);
            
            pcl::PointCloud<PointType> final_aligned;
            final_gicp.align(final_aligned, current_transform);
            
            if (final_gicp.hasConverged()) {
                Eigen::Matrix4f final_transform = final_gicp.getFinalTransformation();
                Eigen::Vector3f translation = final_transform.block(0, 3, 3, 1);
                Eigen::Matrix3f rotation = final_transform.block(0, 0, 3, 3);
                
                // Final validation
                if (translation.norm() <= 10.0f && 
                    std::abs(rotation.determinant() - 1.0f) <= 0.2f) {
                    current_transform = final_transform;
                    fitness_score = final_gicp.getFitnessScore();
                }
            }
        }
        
        // Convert final transformation to row-major format
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                transform_matrix[row * 4 + col] = current_transform(row, col);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[C++/alignPointClouds()] Exception during alignment: " << e.what() << std::endl;
        has_converged = false;
        fitness_score = std::numeric_limits<double>::max();
    } catch (...) {
        std::cerr << "[C++/alignPointClouds()] Unknown exception during alignment" << std::endl;
        has_converged = false;
        fitness_score = std::numeric_limits<double>::max();
    }
    
    return {transform_matrix, fitness_score, has_converged};
}

} // namespace dlo