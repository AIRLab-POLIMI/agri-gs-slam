#include <scancontext/scancontext.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

void coreImportTest() {
    // Test function - implementation can be empty or return status
}

float rad2deg(float radians) {
    return radians * 180.0f / M_PI;
}

float deg2rad(float degrees) {
    return degrees * M_PI / 180.0f;
}

float xy2theta(const float& _x, const float& _y) {
    if (_x >= 0 && _y >= 0) 
        return (180.0f / M_PI) * atan(_y / _x);
    
    if (_x < 0 && _y >= 0) 
        return 180.0f - ((180.0f / M_PI) * atan(_y / (-_x)));
    
    if (_x < 0 && _y < 0) 
        return 180.0f + ((180.0f / M_PI) * atan(_y / _x));
    
    if (_x >= 0 && _y < 0)
        return 360.0f - ((180.0f / M_PI) * atan((-_y) / _x));
    
    return 0.0f; // fallback
}

std::vector<std::vector<double>> circshift(std::vector<std::vector<double>>& _mat, int _num_shift) {
    assert(_num_shift >= 0);
    
    if (_num_shift == 0 || _mat.empty()) {
        return _mat;
    }
    
    int rows = _mat.size();
    int cols = _mat[0].size();
    
    std::vector<std::vector<double>> shifted_mat(rows, std::vector<double>(cols, 0.0));
    
    for (int row_idx = 0; row_idx < rows; row_idx++) {
        for (int col_idx = 0; col_idx < cols; col_idx++) {
            int new_location = (col_idx + _num_shift) % cols;
            shifted_mat[row_idx][new_location] = _mat[row_idx][col_idx];
        }
    }
    
    return shifted_mat;
}

std::vector<float> mat2stdvec(const std::vector<std::vector<double>>& _mat) {
    std::vector<float> vec;
    for (const auto& row : _mat) {
        for (const auto& val : row) {
            vec.push_back(static_cast<float>(val));
        }
    }
    return vec;
}

double SCManager::distDirectSC(std::vector<std::vector<double>>& _sc1, std::vector<std::vector<double>>& _sc2) {
    int num_eff_cols = 0;
    double sum_sector_similarity = 0.0;
    
    int rows = _sc1.size();
    int cols = _sc1[0].size();
    
    for (int col_idx = 0; col_idx < cols; col_idx++) {
        // Extract columns
        std::vector<double> col_sc1(rows), col_sc2(rows);
        for (int row_idx = 0; row_idx < rows; row_idx++) {
            col_sc1[row_idx] = _sc1[row_idx][col_idx];
            col_sc2[row_idx] = _sc2[row_idx][col_idx];
        }
        
        // Calculate norms
        double norm1 = 0.0, norm2 = 0.0, dot_product = 0.0;
        for (int i = 0; i < rows; i++) {
            norm1 += col_sc1[i] * col_sc1[i];
            norm2 += col_sc2[i] * col_sc2[i];
            dot_product += col_sc1[i] * col_sc2[i];
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        
        if (norm1 == 0 || norm2 == 0)
            continue;
        
        double sector_similarity = dot_product / (norm1 * norm2);
        sum_sector_similarity += sector_similarity;
        num_eff_cols++;
    }
    
    if (num_eff_cols == 0) {
        return 1.0; // maximum distance if no valid sectors
    }
    
    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;
}

int SCManager::fastAlignUsingVkey(std::vector<std::vector<double>>& _vkey1, std::vector<std::vector<double>>& _vkey2) {
    int argmin_vkey_shift = 0;
    double min_vkey_diff_norm = std::numeric_limits<double>::max();
    
    int cols = _vkey1[0].size();
    
    for (int shift_idx = 0; shift_idx < cols; shift_idx++) {
        std::vector<std::vector<double>> vkey2_shifted = circshift(_vkey2, shift_idx);
        
        // Calculate difference norm
        double diff_norm = 0.0;
        for (int row_idx = 0; row_idx < _vkey1.size(); row_idx++) {
            for (int col_idx = 0; col_idx < cols; col_idx++) {
                double diff = _vkey1[row_idx][col_idx] - vkey2_shifted[row_idx][col_idx];
                diff_norm += diff * diff;
            }
        }
        diff_norm = sqrt(diff_norm);
        
        if (diff_norm < min_vkey_diff_norm) {
            argmin_vkey_shift = shift_idx;
            min_vkey_diff_norm = diff_norm;
        }
    }
    
    return argmin_vkey_shift;
}

std::pair<double, int> SCManager::distanceBtnScanContext(std::vector<std::vector<double>>& _sc1, std::vector<std::vector<double>>& _sc2) {
    // Fast align using variant key
    std::vector<std::vector<double>> vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
    std::vector<std::vector<double>> vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
    int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);
    
    const int SEARCH_RADIUS = static_cast<int>(round(0.5 * SEARCH_RATIO * _sc1[0].size()));
    std::vector<int> shift_idx_search_space{argmin_vkey_shift};
    
    int cols = _sc1[0].size();
    for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++) {
        shift_idx_search_space.push_back((argmin_vkey_shift + ii + cols) % cols);
        shift_idx_search_space.push_back((argmin_vkey_shift - ii + cols) % cols);
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());
    
    // Find optimal shift
    int argmin_shift = 0;
    double min_sc_dist = std::numeric_limits<double>::max();
    
    for (int num_shift : shift_idx_search_space) {
        std::vector<std::vector<double>> sc2_shifted = circshift(_sc2, num_shift);
        double cur_sc_dist = distDirectSC(_sc1, sc2_shifted);
        
        if (cur_sc_dist < min_sc_dist) {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }
    
    return std::make_pair(min_sc_dist, argmin_shift);
}

std::vector<std::vector<double>> SCManager::makeScancontext(pcl::PointCloud<SCPointType>& _scan_down) {
    const double NO_POINT = -1000.0;
    std::vector<std::vector<double>> desc(PC_NUM_RING, std::vector<double>(PC_NUM_SECTOR, NO_POINT));
    
    for (const auto& point : _scan_down.points) {
        float pt_x = point.x;
        float pt_y = point.y;
        float pt_z = point.z + LIDAR_HEIGHT;
        
        // Convert to polar coordinates
        float azim_range = sqrt(pt_x * pt_x + pt_y * pt_y);
        float azim_angle = xy2theta(pt_x, pt_y);
        
        // Skip if out of range
        if (azim_range > PC_MAX_RADIUS)
            continue;
        
        // Calculate indices
        int ring_idx = std::max(std::min(PC_NUM_RING, 
                                       static_cast<int>(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
        int sector_idx = std::max(std::min(PC_NUM_SECTOR, 
                                         static_cast<int>(ceil((azim_angle / 360.0f) * PC_NUM_SECTOR))), 1);
        
        // Update maximum height at this bin
        if (desc[ring_idx - 1][sector_idx - 1] < pt_z) {
            desc[ring_idx - 1][sector_idx - 1] = pt_z;
        }
    }
    
    // Reset empty bins to zero
    for (int row_idx = 0; row_idx < desc.size(); row_idx++) {
        for (int col_idx = 0; col_idx < desc[0].size(); col_idx++) {
            if (desc[row_idx][col_idx] == NO_POINT) {
                desc[row_idx][col_idx] = 0.0;
            }
        }
    }
    
    return desc;
}

std::vector<std::vector<double>> SCManager::makeRingkeyFromScancontext(std::vector<std::vector<double>>& _desc) {
    std::vector<std::vector<double>> invariant_key(_desc.size(), std::vector<double>(1));
    
    for (int row_idx = 0; row_idx < _desc.size(); row_idx++) {
        double sum = 0.0;
        for (int col_idx = 0; col_idx < _desc[row_idx].size(); col_idx++) {
            sum += _desc[row_idx][col_idx];
        }
        invariant_key[row_idx][0] = sum / _desc[row_idx].size();
    }
    
    return invariant_key;
}

std::vector<std::vector<double>> SCManager::makeSectorkeyFromScancontext(std::vector<std::vector<double>>& _desc) {
    std::vector<std::vector<double>> variant_key(1, std::vector<double>(_desc[0].size()));
    
    for (int col_idx = 0; col_idx < _desc[0].size(); col_idx++) {
        double sum = 0.0;
        for (int row_idx = 0; row_idx < _desc.size(); row_idx++) {
            sum += _desc[row_idx][col_idx];
        }
        variant_key[0][col_idx] = sum / _desc.size();
    }
    
    return variant_key;
}

void SCManager::addScanData(pcl::PointCloud<SCPointType>& _scan_down, int _id, const pcl::PointXYZ& _pose) {
    ScanData scan_data;
    scan_data.id = _id;
    scan_data.pose = _pose;
    scan_data.timestamp = static_cast<double>(std::time(nullptr));
    
    // Generate scan context and keys
    scan_data.scancontext = makeScancontext(_scan_down);
    scan_data.ringkey = makeRingkeyFromScancontext(scan_data.scancontext);
    scan_data.sectorkey = makeSectorkeyFromScancontext(scan_data.scancontext);
    scan_data.invkey_vec = mat2stdvec(scan_data.ringkey);
    
    // Add to database
    scan_database_.push_back(scan_data);
    
    // Update position tree
    updatePositionTree();
}

std::vector<int> SCManager::findPositionCandidates(const pcl::PointXYZ& query_pose, double current_time) {
    std::vector<int> candidates;
    
    if (scan_database_.empty()) {
        return candidates;
    }
    
    // Perform radius search
    std::vector<int> indices;
    std::vector<float> distances;
    
    int found = position_kdtree_.radiusSearch(query_pose, POSITION_SEARCH_RADIUS, indices, distances);
    
    // Filter by time exclusion and limit candidates
    for (int i = 0; i < found && candidates.size() < POSITION_SEARCH_MAX_CANDIDATES; ++i) {
        int idx = indices[i];
        if (current_time - scan_database_[idx].timestamp > TIME_EXCLUSION_WINDOW) {
            candidates.push_back(idx);
        }
    }
    
    return candidates;
}

void SCManager::updatePositionTree() {
    position_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    for (const auto& scan_data : scan_database_) {
        position_cloud_->points.push_back(scan_data.pose);
    }
    
    position_cloud_->width = position_cloud_->points.size();
    position_cloud_->height = 1;
    position_cloud_->is_dense = true;
    
    if (!position_cloud_->empty()) {
        position_kdtree_.setInputCloud(position_cloud_);
    }
}

std::pair<bool, std::tuple<std::vector<int>, std::vector<float>, int>> SCManager::detectLoopClosure(pcl::PointCloud<SCPointType>& _scan_down, int _id, const pcl::PointXYZ& _pose) {
    double current_time = static_cast<double>(std::time(nullptr));
    
    // Find position-based candidates
    std::vector<int> position_candidates = findPositionCandidates(_pose, current_time);
    int num_candidates = position_candidates.size() - 1; // Exclude the current scan
    
    if (num_candidates < POSITION_SEARCH_MIN_CANDIDATES) {
        return std::make_pair(false, std::make_tuple(std::vector<int>{-1, -1, -1}, std::vector<float>{-1.0f, -1.0f, -1.0f}, num_candidates));
    }
    
    // Generate scan context for query
    std::vector<std::vector<double>> query_sc = makeScancontext(_scan_down);
    
    // Store all valid matches
    std::vector<std::tuple<double, int, int>> matches; // (distance, candidate_idx, yaw_shift)
    
    for (int candidate_idx : position_candidates) {
        // Skip if the candidate has the same _id as the query
        if (scan_database_[candidate_idx].id == _id) {
            continue;
        }
        
        std::vector<std::vector<double>> candidate_sc = scan_database_[candidate_idx].scancontext;
        std::pair<double, int> sc_result = distanceBtnScanContext(query_sc, candidate_sc);
        
        double sc_dist = sc_result.first;
        int yaw_shift = sc_result.second;
        
        if (sc_dist < SC_DIST_THRES) {
            matches.push_back(std::make_tuple(sc_dist, candidate_idx, yaw_shift));
        }
    }
    
    // Prepare default return values
    std::vector<int> top_ids{-1, -1, -1};
    std::vector<float> top_yaw_diffs{-1.0f, -1.0f, -1.0f};
    
    if (matches.empty()) {
        return std::make_pair(false, std::make_tuple(top_ids, top_yaw_diffs, num_candidates));
    }
    
    // Sort matches by distance (ascending)
    std::sort(matches.begin(), matches.end());
    
    // Get top 3 (or fewer if less available)
    int top_count = std::min(static_cast<int>(matches.size()), 3);
    for (int i = 0; i < top_count; ++i) {
        int matched_id = scan_database_[std::get<1>(matches[i])].id;
        float yaw_diff_rad = deg2rad(std::get<2>(matches[i]) * PC_UNIT_SECTORANGLE);
        top_ids[i] = matched_id;
        top_yaw_diffs[i] = yaw_diff_rad;
    }
    
    return std::make_pair(true, std::make_tuple(top_ids, top_yaw_diffs, num_candidates));
}
