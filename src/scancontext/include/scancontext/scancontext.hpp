#pragma once

#include <ctime>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm> 
#include <cstdlib>
#include <memory>
#include <iostream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.hpp"

using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using std::atan2;
using std::cos;
using std::sin;

using SCPointType = pcl::PointXYZI;
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

// Structure to hold scan data
struct ScanData {
    int id;
    pcl::PointXYZ pose;
    std::vector<std::vector<double>> scancontext;
    std::vector<std::vector<double>> ringkey;
    std::vector<std::vector<double>> sectorkey;
    std::vector<float> invkey_vec;
    double timestamp;
};

void coreImportTest();

// Helper functions 
float xy2theta(const float& _x, const float& _y);
std::vector<std::vector<double>> circshift(std::vector<std::vector<double>>& _mat, int _num_shift);
std::vector<float> mat2stdvec(const std::vector<std::vector<double>>& _mat);
float rad2deg(float radians);
float deg2rad(float degrees);

class SCManager
{
public: 
    SCManager() = default;
    
    // Core functionality
    std::vector<std::vector<double>> makeScancontext(pcl::PointCloud<SCPointType>& _scan_down);
    std::vector<std::vector<double>> makeRingkeyFromScancontext(std::vector<std::vector<double>>& _desc);
    std::vector<std::vector<double>> makeSectorkeyFromScancontext(std::vector<std::vector<double>>& _desc);

    int fastAlignUsingVkey(std::vector<std::vector<double>>& _vkey1, std::vector<std::vector<double>>& _vkey2); 
    double distDirectSC(std::vector<std::vector<double>>& _sc1, std::vector<std::vector<double>>& _sc2);
    std::pair<double, int> distanceBtnScanContext(std::vector<std::vector<double>>& _sc1, std::vector<std::vector<double>>& _sc2);

    // New optimized API
    void addScanData(pcl::PointCloud<SCPointType>& _scan_down, int _id, const pcl::PointXYZ& _pose);
    std::pair<bool, std::tuple<std::vector<int>, std::vector<float>, int>> detectLoopClosure(pcl::PointCloud<SCPointType>& _scan_down, int _id, const pcl::PointXYZ& _pose);
    
    // Configuration setters
    void setPositionSearchRadius(double radius) { POSITION_SEARCH_RADIUS = radius; }
    void setPositionSearchMinCandidates(int min_candidates) { POSITION_SEARCH_MIN_CANDIDATES = min_candidates; }
    void setPositionSearchMaxCandidates(int max_candidates) { POSITION_SEARCH_MAX_CANDIDATES = max_candidates; }
    void setTimeExclusionWindow(double window) { TIME_EXCLUSION_WINDOW = window; }
    
    // Getters
    double getLidarHeight() const { return LIDAR_HEIGHT; }
    void setLidarHeight(double height) { LIDAR_HEIGHT = height; }
    
    int getPCNumRing() const { return PC_NUM_RING; }
    int getPCNumSector() const { return PC_NUM_SECTOR; }
    double getPCMaxRadius() const { return PC_MAX_RADIUS; }
    double getSCDistThres() const { return SC_DIST_THRES; }
    
    // Access to internal data
    const std::vector<ScanData>& getScanDatabase() const { return scan_database_; }
    size_t getDatabaseSize() const { return scan_database_.size(); }

private:
    // Core scancontext parameters
    double LIDAR_HEIGHT = 2.0;
    const int PC_NUM_RING = 20;
    const int PC_NUM_SECTOR = 60;
    const double PC_MAX_RADIUS = 80.0;
    const double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
    const double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

    // Position-based filtering parameters
    double POSITION_SEARCH_RADIUS = 50.0;  // meters
    int POSITION_SEARCH_MIN_CANDIDATES = 1;
    int POSITION_SEARCH_MAX_CANDIDATES = 20;
    double TIME_EXCLUSION_WINDOW = 30.0;  // seconds

    // Loop thresholds
    const double SEARCH_RATIO = 0.1;
    const double SC_DIST_THRES = 0.13;

    // Data storage
    std::vector<ScanData> scan_database_;
    
    // KD-tree for position-based search
    pcl::KdTreeFLANN<pcl::PointXYZ> position_kdtree_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr position_cloud_;
    
    // Helper methods
    std::vector<int> findPositionCandidates(const pcl::PointXYZ& query_pose, double current_time);
    void updatePositionTree();
};
