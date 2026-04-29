// dlo.hpp
#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nano_gicp/nano_gicp.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <omp.h>

using PointType = pcl::PointXYZ;

namespace dlo {

struct OdometryResult {
    std::vector<double> position;      // [x, y, z]
    std::vector<double> orientation;   // [x, y, z, w] quaternion
    std::vector<double> transformation; // 4x4 matrix in row-major order (16 elements)
    double timestamp;
    bool success;
    bool is_keyframe;
    double computation_time;
};

struct Metrics {
    std::vector<double> spaciousness;
    std::vector<double> computation_times;
};

class DirectLidarOdometry {
public:
    // Constructor with configuration parameters
    DirectLidarOdometry();
    ~DirectLidarOdometry();

    // Main processing methods - PYTHON_BIND functions
    OdometryResult track(const pcl::PointCloud<PointType>::Ptr& cloud, double timestamp);
    std::vector<double> getCurrentPose() const;
    std::vector<double> getCurrentTransformation() const;
    std::vector<std::vector<double>> getTrajectoryPoses() const;
    int getNumKeyframes() const;
    void reset();
    void setInitialPose(const std::vector<double>& pose_vec);
    void configure(double keyframe_dist_thresh = 1.0, 
                   double keyframe_rot_thresh = 15.0,
                   bool enable_adaptive = true,
                   bool use_voxel_filter = true,
                   double voxel_size = 0.05);
    pcl::PointCloud<PointType>::Ptr getMap() const;
    std::vector<double> getComputationTimes() const;
    std::pair<std::vector<std::vector<double>>, std::vector<pcl::PointCloud<PointType>::Ptr>>
    refine(const std::vector<pcl::PointCloud<PointType>::Ptr>& source_clouds,
                            const std::vector<pcl::PointCloud<PointType>::Ptr>& target_clouds,
                            int max_iterations = 50,
                            double transformation_epsilon = 1e-6,
                            double euclidean_fitness_epsilon = 1e-6,
                            double max_correspondence_distance = 1.0);
    static std::vector<std::vector<double>> computeNormals(
        const pcl::PointCloud<PointType>::Ptr& cloud,
        int k_neighbors = 10,
        double radius = 0.20
    );
    static std::tuple<std::vector<double>, double, bool> alignPointClouds(
        const pcl::PointCloud<PointType>::Ptr& source_cloud,
        const pcl::PointCloud<PointType>::Ptr& target_cloud,
        const std::vector<double>& initial_guess_vec,
        int max_iterations = 100,
        double transformation_epsilon = 1e-6,
        double euclidean_fitness_epsilon = 1e-6,
        double max_correspondence_distance = 1.0
    );

    // Configuration methods (advanced)
    void setKeyframeThresholds(double distance_thresh, double rotation_thresh);
    void setSubmapParameters(int knn, int kcv, int kcc);
    void setVoxelFilterParameters(bool use_scan_filter, double scan_res, 
                                  bool use_submap_filter, double submap_res);
    void setCropBoxParameters(bool use_crop, double crop_size);
    void setGICPParameters(int min_points, int max_iter_s2s, int max_iter_s2m,
                          double transform_eps, double fitness_eps);
    void setAdaptiveParameters(bool enable);
    void setInitialPose(const Eigen::Vector3f& position, const Eigen::Quaternionf& orientation);

    // Utility methods
    bool isInitialized() const { return initialized_; }
    std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> getTrajectory() const;
    pcl::PointCloud<PointType>::Ptr getKeyframesCloud() const;
    Metrics getMetrics() const { return metrics_; }

private:
    // Initialization
    void initialize();
    void initializeInputTarget();

    // Point cloud processing
    void deskewPointCloud();
    void preprocessPoints();
    void setInputSources();
    void transformCurrentScan();

    // Odometry computation
    void getNextPose();
    void propagateS2S(const Eigen::Matrix4f& T);
    void propagateS2M();

    // Keyframe management
    void updateKeyframes();
    void getSubmapKeyframes();
    Eigen::Matrix4f predictMotion(double dt);
    void updateMotionModel(double dt);
    void configureMotionModel(bool enable, double smoothing_factor);
    void pushSubmapIndices(const std::vector<float>& dists, int k, 
                          const std::vector<int>& frames);

    // Hull computation
    void computeConvexHull();
    void computeConcaveHull();

    // Metrics and adaptive parameters
    void computeMetrics();
    void computeSpaciousness();
    void setAdaptiveParams();

    // Configuration parameters - Based on provided DLO configuration
    double keyframe_thresh_dist_ = 1.0;   // threshD from config
    double keyframe_thresh_rot_ = 15.0;   // threshR from config
    int submap_knn_ = 10;                 // submap keyframe knn
    int submap_kcv_ = 10;                 // submap keyframe kcv
    int submap_kcc_ = 10;                 // submap keyframe kcc
    
    bool vf_scan_use_ = true;             // voxelFilter scan use
    double vf_scan_res_ = 0.1;           // voxelFilter scan res
    bool vf_submap_use_ = true;           // voxelFilter submap use
    double vf_submap_res_ = 0.25;          // voxelFilter submap res
    
    bool crop_use_ = false;                // cropBoxFilter use
    double crop_size_ = 1.0;              // cropBoxFilter size
    
    int gicp_min_num_points_ = 10;        // gicp minNumPoints
    bool adaptive_params_use_ = true;    // Keep adaptive disabled
    
    // GICP parameters - From provided configuration
    int gicps2s_max_iter_ = 64;           // gicp s2s maxIterations
    int gicps2m_max_iter_ = 64;           // gicp s2m maxIterations
    double gicps2s_transformation_ep_ = 0.01;      // gicp s2s transformationEpsilon
    double gicps2m_transformation_ep_ = 0.01;      // gicp s2m transformationEpsilon
    double gicps2s_euclidean_fitness_ep_ = 0.01;   // gicp s2s euclideanFitnessEpsilon
    double gicps2m_euclidean_fitness_ep_ = 0.01;   // gicp s2m euclideanFitnessEpsilon

    // Initialize motion model
    bool motion_model_enabled_ = true;
    double motion_smoothing_factor_ = 0.8f;
    Eigen::Vector3f velocity_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f angular_velocity_ = Eigen::Vector3f::Zero();
    double prev_timestamp_ = 0.0;
    bool motion_initialized_ = false;

    // Initial pose
    bool initial_pose_use_ = false;
    Eigen::Vector3f initial_position_ = Eigen::Vector3f::Zero();
    Eigen::Quaternionf initial_orientation_ = Eigen::Quaternionf::Identity();

    // State variables
    bool initialized_ = false;
    Eigen::Vector3f pose_ = Eigen::Vector3f::Zero();
    Eigen::Quaternionf rotq_ = Eigen::Quaternionf::Identity();
    Eigen::Vector3f pose_s2s_ = Eigen::Vector3f::Zero();
    Eigen::Quaternionf rotq_s2s_ = Eigen::Quaternionf::Identity();
    
    Eigen::Matrix4f T_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_s2s_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_s2s_prev_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f rotSO3_;
    Eigen::Matrix3f rotSO3_s2s_;

    // Point clouds
    pcl::PointCloud<PointType>::Ptr original_scan_;
    pcl::PointCloud<PointType>::Ptr current_scan_;
    pcl::PointCloud<PointType>::Ptr current_scan_t_;
    pcl::PointCloud<PointType>::Ptr source_cloud_;
    pcl::PointCloud<PointType>::Ptr target_cloud_;
    
    pcl::PointCloud<PointType>::Ptr keyframe_cloud_;
    pcl::PointCloud<PointType>::Ptr keyframes_cloud_;
    pcl::PointCloud<PointType>::Ptr submap_cloud_;

    // Keyframes data
    std::vector<std::pair<std::pair<Eigen::Vector3f, Eigen::Quaternionf>, 
                         pcl::PointCloud<PointType>::Ptr>> keyframes_;

    // PCL objects - replaced PCL GICP with nano_gicp
    nano_gicp::NanoGICP<PointType, PointType> gicp_s2s_;
    nano_gicp::NanoGICP<PointType, PointType> gicp_;
    pcl::VoxelGrid<PointType> vf_scan_;
    pcl::VoxelGrid<PointType> vf_submap_;
    pcl::CropBox<PointType> crop_;
    pcl::ConvexHull<PointType> convex_hull_;
    pcl::ConcaveHull<PointType> concave_hull_;

    // Missing member variables
    Metrics metrics_;
    int num_keyframes_ = 0;
    std::vector<std::pair<Eigen::Vector3f, Eigen::Quaternionf>> trajectory_;
    
    // Keyframe hull data
    std::vector<int> keyframe_convex_;
    std::vector<int> keyframe_concave_;
    
    // Normals data
    std::vector<Eigen::Vector3f> keyframe_normals_;
    std::vector<Eigen::Vector3f> submap_normals_;
    
    // Submap indices
    std::vector<int> submap_kf_idx_curr_;
    std::vector<int> submap_kf_idx_prev_;
    
    // Submap state
    bool submap_hasChanged_ = true;

    // Thread safety
    mutable std::mutex mutex_;
};

} // namespace dlo
