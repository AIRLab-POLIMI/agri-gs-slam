#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Include the header file (assuming it's in the same directory or properly configured)
#include <dlo/dlo.hpp>

namespace py = pybind11;

// Helper function to convert numpy array to PCL point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr numpy_to_pcl(py::array_t<float> points) {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    
    auto buf = points.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("Input array must be Nx3 (x,y,z)");
    }
    
    // Check if array is C-contiguous by examining strides
    bool is_c_contiguous = true;
    if (buf.ndim >= 2) {
        // For C-contiguous: stride[i] = stride[i+1] * shape[i+1]
        py::ssize_t expected_stride = sizeof(float);
        for (int i = buf.ndim - 1; i >= 0; --i) {
            if (buf.strides[i] != expected_stride) {
                is_c_contiguous = false;
                break;
            }
            if (i > 0) {
                expected_stride *= buf.shape[i];
            }
        }
    }
    
    if (!is_c_contiguous) {
        throw std::runtime_error("Input array must be C-contiguous. Use numpy.ascontiguousarray()");
    }
    
    float* ptr = static_cast<float*>(buf.ptr);
    size_t num_points = static_cast<size_t>(buf.shape[0]);
    
    // Validate input data
    if (num_points == 0) {
        throw std::runtime_error("Input array cannot be empty");
    }
    
    // Reserve space first, then resize
    cloud->points.reserve(num_points);
    cloud->points.resize(num_points);
    cloud->width = static_cast<uint32_t>(num_points);
    cloud->height = 1;
    cloud->is_dense = true;
    
    // Copy points with bounds checking
    size_t total_elements = static_cast<size_t>(buf.size);
    for (size_t i = 0; i < num_points; ++i) {
        size_t base_idx = i * 3;
        
        // Validate array bounds
        if (base_idx + 2 >= total_elements) {
            throw std::runtime_error("Array index out of bounds during conversion at point " + std::to_string(i));
        }
        
        cloud->points[i].x = ptr[base_idx + 0];
        cloud->points[i].y = ptr[base_idx + 1];
        cloud->points[i].z = ptr[base_idx + 2];
        
        // Check for invalid values
        if (!std::isfinite(cloud->points[i].x) || 
            !std::isfinite(cloud->points[i].y) || 
            !std::isfinite(cloud->points[i].z)) {
            throw std::runtime_error("Invalid point coordinates (NaN or Inf) at index " + std::to_string(i));
        }
    }
    
    return cloud;
}

// Helper function to convert PCL point cloud to numpy array
py::array_t<float> pcl_to_numpy(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    if (!cloud || cloud->empty()) {
        // Create empty array with proper shape
        std::vector<py::ssize_t> shape = {0, 3};
        return py::array_t<float>(shape);
    }
    
    size_t num_points = cloud->size();
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(num_points), 3};
    auto result = py::array_t<float>(shape);
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    for (size_t i = 0; i < num_points; ++i) {
        ptr[i * 3 + 0] = cloud->points[i].x;
        ptr[i * 3 + 1] = cloud->points[i].y;
        ptr[i * 3 + 2] = cloud->points[i].z;
    }
    
    return result;
}

PYBIND11_MODULE(odometry, m) {
    m.doc() = "Direct LiDAR Odometry Python Bindings";
    
    // Bind OdometryResult struct
    py::class_<dlo::OdometryResult>(m, "OdometryResult")
        .def(py::init<>())
        .def_readwrite("position", &dlo::OdometryResult::position)
        .def_readwrite("orientation", &dlo::OdometryResult::orientation)
        .def_readwrite("transformation", &dlo::OdometryResult::transformation)
        .def_readwrite("timestamp", &dlo::OdometryResult::timestamp)
        .def_readwrite("success", &dlo::OdometryResult::success)
        .def_readwrite("is_keyframe", &dlo::OdometryResult::is_keyframe)
        .def_readwrite("computation_time", &dlo::OdometryResult::computation_time);
    
    // Bind Metrics struct
    py::class_<dlo::Metrics>(m, "Metrics")
        .def(py::init<>())
        .def_readwrite("spaciousness", &dlo::Metrics::spaciousness)
        .def_readwrite("computation_times", &dlo::Metrics::computation_times);
    
    // Bind DirectLidarOdometry class
    py::class_<dlo::DirectLidarOdometry>(m, "DirectLidarOdometry")
        .def(py::init<>())
        
        // Main processing methods with error handling
        .def("track", [](dlo::DirectLidarOdometry &self, py::array_t<float> points, double timestamp) {
            try {
                // Validate input
                if (points.size() == 0) {
                    throw std::runtime_error("Empty point cloud provided");
                }
                
                // Convert numpy array to PCL cloud
                auto cloud = numpy_to_pcl(points);
                
                // Additional validation
                if (!cloud || cloud->empty()) {
                    throw std::runtime_error("Failed to convert numpy array to PCL point cloud");
                }
                
                // Call the C++ implementation
                return self.track(cloud, timestamp);
            }
            catch (const std::exception& e) {
                throw std::runtime_error("Error in track(): " + std::string(e.what()));
            }
        }, "Track odometry with an Nx3 float array", py::arg("points"), py::arg("timestamp"))

        .def("get_current_pose", &dlo::DirectLidarOdometry::getCurrentPose,
             "Get current pose as [x, y, z, qx, qy, qz, qw]")
        
        .def("get_current_transformation", &dlo::DirectLidarOdometry::getCurrentTransformation,
             "Get current transformation matrix as flat vector")
        
        .def("get_trajectory_poses", &dlo::DirectLidarOdometry::getTrajectoryPoses,
             "Get all trajectory poses")
        
        .def("get_num_keyframes", &dlo::DirectLidarOdometry::getNumKeyframes,
             "Get number of keyframes")
        
        .def("reset", &dlo::DirectLidarOdometry::reset,
             "Reset the odometry system")
        
        .def("set_initial_pose", py::overload_cast<const std::vector<double>&>(&dlo::DirectLidarOdometry::setInitialPose),
             "Set initial pose from vector [x, y, z, qx, qy, qz, qw]", py::arg("pose_vec"))
        
        .def("configure", &dlo::DirectLidarOdometry::configure,
             "Configure odometry parameters",
             py::arg("keyframe_dist_thresh") = 1.0,
             py::arg("keyframe_rot_thresh") = 15.0,
             py::arg("enable_adaptive") = true,
             py::arg("use_voxel_filter") = true,
             py::arg("voxel_size") = 0.05)
        
        .def("get_map", [](dlo::DirectLidarOdometry& self) {
            try {
                return pcl_to_numpy(self.getMap());
            }
            catch (const std::exception& e) {
                throw std::runtime_error("Error getting map: " + std::string(e.what()));
            }
        }, "Get the current map as numpy array")
        
        .def("get_computation_times", &dlo::DirectLidarOdometry::getComputationTimes,
             "Get computation times")
        
        // Configuration methods
        .def("set_keyframe_thresholds", &dlo::DirectLidarOdometry::setKeyframeThresholds,
             "Set keyframe thresholds", py::arg("distance_thresh"), py::arg("rotation_thresh"))
        
        .def("set_submap_parameters", &dlo::DirectLidarOdometry::setSubmapParameters,
             "Set submap parameters", py::arg("knn"), py::arg("kcv"), py::arg("kcc"))
        
        .def("set_voxel_filter_parameters", &dlo::DirectLidarOdometry::setVoxelFilterParameters,
             "Set voxel filter parameters",
             py::arg("use_scan_filter"), py::arg("scan_res"),
             py::arg("use_submap_filter"), py::arg("submap_res"))
        
        .def("set_crop_box_parameters", &dlo::DirectLidarOdometry::setCropBoxParameters,
             "Set crop box parameters", py::arg("use_crop"), py::arg("crop_size"))
        
        .def("set_gicp_parameters", &dlo::DirectLidarOdometry::setGICPParameters,
             "Set GICP parameters",
             py::arg("min_points"), py::arg("max_iter_s2s"), py::arg("max_iter_s2m"),
             py::arg("transform_eps"), py::arg("fitness_eps"))
        
        .def("set_adaptive_parameters", &dlo::DirectLidarOdometry::setAdaptiveParameters,
             "Set adaptive parameters", py::arg("enable"))
        
        // Utility methods
        .def("is_initialized", &dlo::DirectLidarOdometry::isInitialized,
             "Check if odometry is initialized")
        
        .def("get_trajectory", &dlo::DirectLidarOdometry::getTrajectory,
             "Get trajectory as pairs of position and orientation")
        
        .def("get_keyframes_cloud", [](dlo::DirectLidarOdometry& self) {
            try {
            return pcl_to_numpy(self.getKeyframesCloud());
            }
            catch (const std::exception& e) {
            throw std::runtime_error("Error getting keyframes cloud: " + std::string(e.what()));
            }
        }, "Get keyframes cloud as numpy array")
        
        .def("get_metrics", &dlo::DirectLidarOdometry::getMetrics,
             "Get odometry metrics")
        
        .def("get_current_transformation_matrix", &dlo::DirectLidarOdometry::getCurrentTransformation,
             "Get current transformation matrix")
        
        .def_static("compute_normals", [](py::array_t<float> points, int k_neighbors = 10, double radius = 0.20) -> py::list {
            try {
            // Convert numpy array to PCL cloud
            auto cloud = numpy_to_pcl(points);
            
            if (!cloud || cloud->empty()) {
                throw std::runtime_error("Input point cloud is empty or invalid");
            }
            
            // Call the static C++ function
            auto normals = dlo::DirectLidarOdometry::computeNormals(cloud, k_neighbors, radius);
            
            // Convert result to Python list
            py::list result;
            for (const auto& normal : normals) {
                py::list normal_vec;
                for (double val : normal) {
                normal_vec.append(val);
                }
                result.append(normal_vec);
            }
            
            return result;
            }
            catch (const std::exception& e) {
                throw std::runtime_error("Error in compute_normals(): " + std::string(e.what()));
            }
        }, "Compute normals for point cloud", 
        py::arg("points"), py::arg("k_neighbors") = 10, py::arg("radius") = 0.20)
        
        .def("refine", [](dlo::DirectLidarOdometry& self, py::list source_clouds_list, py::list target_clouds_list,
                int max_iterations = 50,
                double transformation_epsilon = 1e-6,
                double euclidean_fitness_epsilon = 1e-6,
                double max_correspondence_distance = 1.0) -> py::tuple {
            try {
                // Convert list of numpy arrays to vector of PCL clouds for source
                std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> source_clouds;
                source_clouds.reserve(source_clouds_list.size());
                
                for (size_t i = 0; i < source_clouds_list.size(); ++i) {
                    py::array_t<float> points = source_clouds_list[i].cast<py::array_t<float>>();
                    auto cloud = numpy_to_pcl(points);
                    if (!cloud || cloud->empty()) {
                        throw std::runtime_error("Source cloud at index " + std::to_string(i) + " is empty or invalid");
                    }
                    source_clouds.push_back(cloud);
                }
                
                // Convert list of numpy arrays to vector of PCL clouds for target
                std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> target_clouds;
                target_clouds.reserve(target_clouds_list.size());
                
                for (size_t i = 0; i < target_clouds_list.size(); ++i) {
                    py::array_t<float> points = target_clouds_list[i].cast<py::array_t<float>>();
                    auto cloud = numpy_to_pcl(points);
                    if (!cloud || cloud->empty()) {
                        throw std::runtime_error("Target cloud at index " + std::to_string(i) + " is empty or invalid");
                    }
                    target_clouds.push_back(cloud);
                }
                
                if (source_clouds.empty() || target_clouds.empty()) {
                    throw std::runtime_error("Source and target clouds lists cannot be empty");
                }
                
                if (source_clouds.size() != target_clouds.size()) {
                    throw std::runtime_error("Source and target clouds lists must have the same size");
                }
                
                // Call the C++ refine method with initial guess
                auto result = self.refine(source_clouds, target_clouds,  
                                        max_iterations, transformation_epsilon, 
                                        euclidean_fitness_epsilon, max_correspondence_distance);
                
                // Convert the result to Python types
                // First element: vector of transformations (vector<vector<double>>)
                py::list transformations;
                for (const auto& transform : result.first) {
                    py::list transform_list;
                    for (double val : transform) {
                        transform_list.append(val);
                    }
                    transformations.append(transform_list);
                }
                
                // Second element: vector of refined clouds
                py::list refined_clouds;
                for (const auto& cloud : result.second) {
                    refined_clouds.append(pcl_to_numpy(cloud));
                }
                
                return py::make_tuple(transformations, refined_clouds);
            }
            catch (const std::exception& e) {
                throw std::runtime_error("Error in refine(): " + std::string(e.what()));
            }
        }, "Refine point clouds using iterative registration between source and target clouds with initial guess",
        py::arg("source_clouds"), 
        py::arg("target_clouds"),
        py::arg("max_iterations") = 50,
        py::arg("transformation_epsilon") = 1e-6,
        py::arg("euclidean_fitness_epsilon") = 1e-6,
        py::arg("max_correspondence_distance") = 1.0)
        
        .def_static("align_point_clouds", [](py::array_t<float> source_points, py::array_t<float> target_points,
                const std::vector<double>& initial_guess_vec,
                int max_iterations = 100,
                double transformation_epsilon = 1e-6,
                double euclidean_fitness_epsilon = 1e-6,
                double max_correspondence_distance = 1.0) -> py::tuple {
            try {
                auto source_cloud = numpy_to_pcl(source_points);
                auto target_cloud = numpy_to_pcl(target_points);
                
                if (!source_cloud || source_cloud->empty()) {
                    throw std::runtime_error("Source point cloud is empty or invalid");
                }
                
                if (!target_cloud || target_cloud->empty()) {
                    throw std::runtime_error("Target point cloud is empty or invalid");
                }
                
                auto result = dlo::DirectLidarOdometry::alignPointClouds(
                    source_cloud, target_cloud, initial_guess_vec,
                    max_iterations, transformation_epsilon,
                    euclidean_fitness_epsilon, max_correspondence_distance
                );
                
                py::list transform_matrix;
                for (double val : std::get<0>(result)) {
                    transform_matrix.append(val);
                }
                
                return py::make_tuple(transform_matrix, std::get<1>(result), std::get<2>(result));
            }
            catch (const std::exception& e) {
                throw std::runtime_error("Error in align_point_clouds(): " + std::string(e.what()));
            }
        }, "Fast multiscale robust point cloud registration between source and target clouds",
        py::arg("source_points"), py::arg("target_points"), py::arg("initial_guess_vec"),
        py::arg("max_iterations") = 100,
        py::arg("transformation_epsilon") = 1e-6,
        py::arg("euclidean_fitness_epsilon") = 1e-6,
        py::arg("max_correspondence_distance") = 1.0);
}
