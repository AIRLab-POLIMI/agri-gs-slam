#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Include the original header file
#include <scancontext/scancontext.hpp>

namespace py = pybind11;

// Helper function to convert numpy array to PCL point cloud
pcl::PointCloud<SCPointType> numpy_to_pcl(py::array_t<float> points) {
     auto buf = points.request();
     if (buf.ndim != 2 || buf.shape[1] < 3) {
          throw std::runtime_error("Input array must be N x 3 or N x 4 (x, y, z, [intensity])");
     }
     
     pcl::PointCloud<SCPointType> cloud;
     cloud.width = buf.shape[0];
     cloud.height = 1;
     cloud.is_dense = true;
     cloud.points.resize(cloud.width * cloud.height);
     
     float* ptr = (float*)buf.ptr;
     for (int i = 0; i < cloud.width; ++i) {
          cloud.points[i].x = ptr[i * buf.shape[1] + 0];
          cloud.points[i].y = ptr[i * buf.shape[1] + 1];
          cloud.points[i].z = ptr[i * buf.shape[1] + 2];
          cloud.points[i].intensity = (buf.shape[1] > 3) ? ptr[i * buf.shape[1] + 3] : 0.0f;
     }
     
     return cloud;
}

PYBIND11_MODULE(scancontext, m) {
     m.doc() = "ScanContext loop closure detection";
     
     // Bind utility functions
     m.def("xy_to_theta", &xy2theta, "Convert x, y coordinates to theta angle");
     m.def("circular_shift", &circshift, "Circular shift matrix");
     m.def("matrix_to_std_vector", &mat2stdvec, "Convert matrix to std::vector");
     m.def("core_import_test", &coreImportTest, "Test core imports");
     m.def("rad_to_deg", &rad2deg, "Convert radians to degrees");
     m.def("deg_to_rad", &deg2rad, "Convert degrees to radians");
     
     // Bind ScanData structure
     py::class_<ScanData>(m, "ScanData")
          .def(py::init<>())
          .def_readwrite("id", &ScanData::id)
          .def_readwrite("pose", &ScanData::pose)
          .def_readwrite("scancontext", &ScanData::scancontext)
          .def_readwrite("ringkey", &ScanData::ringkey)
          .def_readwrite("sectorkey", &ScanData::sectorkey)
          .def_readwrite("invkey_vec", &ScanData::invkey_vec)
          .def_readwrite("timestamp", &ScanData::timestamp);
     
     // Bind PCL PointXYZ for pose
     py::class_<pcl::PointXYZ>(m, "PointXYZ")
          .def(py::init<>())
          .def(py::init<float, float, float>())
          .def_readwrite("x", &pcl::PointXYZ::x)
          .def_readwrite("y", &pcl::PointXYZ::y)
          .def_readwrite("z", &pcl::PointXYZ::z);
     
     // Bind main SCManager class
     py::class_<SCManager>(m, "SCManager")
          .def(py::init<>())
          
          // Core functionality
          .def("make_scan_context", [](SCManager& self, py::array_t<float> points) {
               auto cloud = numpy_to_pcl(points);
               return self.makeScancontext(cloud);
          }, "Create scan context from point cloud")
          
          .def("make_ring_key_from_scan_context", &SCManager::makeRingkeyFromScancontext,
                "Create ring key from scan context")
          
          .def("make_sector_key_from_scan_context", &SCManager::makeSectorkeyFromScancontext,
                "Create sector key from scan context")
          
          .def("fast_align_using_vkey", &SCManager::fastAlignUsingVkey,
                "Fast alignment using vertical key")
          
          .def("direct_distance_scan_context", &SCManager::distDirectSC,
                "Direct distance between scan contexts")
          
          .def("distance_between_scan_contexts", &SCManager::distanceBtnScanContext,
                "Distance between scan contexts with alignment")
          
          // New optimized API
          .def("add_scan_data", [](SCManager& self, py::array_t<float> points, int id, const pcl::PointXYZ& pose) {
               auto cloud = numpy_to_pcl(points);
               self.addScanData(cloud, id, pose);
        }, "Add scan data to database")
        
        .def("detect_loop_closure", [](SCManager& self, py::array_t<float> points, int id, const pcl::PointXYZ& pose) {
               auto cloud = numpy_to_pcl(points);
               auto result = self.detectLoopClosure(cloud, id, pose);
               return py::make_tuple(
                        result.first,
                        py::make_tuple(
                               std::get<0>(result.second),
                               std::get<1>(result.second),
                               std::get<2>(result.second)
                        )
               );
        }, "Detect loop closure with new optimized API")
          
          // Configuration setters
          .def("set_position_search_radius", &SCManager::setPositionSearchRadius,
                "Set position search radius")
          .def("set_position_search_min_candidates", &SCManager::setPositionSearchMinCandidates,
                "Set minimum number of position candidates")
          .def("set_position_search_max_candidates", &SCManager::setPositionSearchMaxCandidates,
                "Set maximum number of position candidates")
          .def("set_time_exclusion_window", &SCManager::setTimeExclusionWindow,
                "Set time exclusion window")
          
          // Getters/Setters
          .def("get_lidar_height", &SCManager::getLidarHeight, "Get LiDAR height")
          .def("set_lidar_height", &SCManager::setLidarHeight, "Set LiDAR height")
          .def("get_num_rings", &SCManager::getPCNumRing, "Get number of rings")
          .def("get_num_sectors", &SCManager::getPCNumSector, "Get number of sectors")
          .def("get_max_radius", &SCManager::getPCMaxRadius, "Get max radius")
          .def("get_distance_threshold", &SCManager::getSCDistThres, "Get distance threshold")
          
          // Data access
          .def("get_scan_database", &SCManager::getScanDatabase,
                py::return_value_policy::reference_internal,
                "Get scan database")
          .def("get_database_size", &SCManager::getDatabaseSize,
                "Get database size");
}
