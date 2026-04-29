import numpy as np
import scancontext

def main():
    # Create an instance of SCManager
    sc_manager = scancontext.SCManager()

    # Example LiDAR point cloud data (N x 4: x, y, z, intensity)
    points = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [4.0, 5.0, 6.0, 0.8],
        [7.0, 8.0, 9.0, 0.3],
        [2.0, 3.0, 1.0, 0.7],
        [5.0, 6.0, 2.0, 0.4]
    ], dtype=np.float32)

    # Create a pose for the scan
    pose = scancontext.PointXYZ(0.0, 0.0, 0.0)
    
    # Add scan data to the database
    scan_id = 0
    sc_manager.add_scan_data(points, scan_id, pose)
    print(f"Added scan data with ID: {scan_id}")

    # Add more scan data for loop closure detection
    points2 = np.array([
        [1.1, 2.1, 3.1, 0.6],
        [4.1, 5.1, 6.1, 0.9],
        [7.1, 8.1, 9.1, 0.4],
        [2.1, 3.1, 1.1, 0.8],
        [5.1, 6.1, 2.1, 0.5]
    ], dtype=np.float32)
    
    pose2 = scancontext.PointXYZ(0.1, 0.1, 0.0)
    scan_id2 = 1
    sc_manager.add_scan_data(points2, scan_id2, pose2)
    print(f"Added scan data with ID: {scan_id2}")

    # Detect loop closure with the new optimized API
    points3 = np.array([
        [1.05, 2.05, 3.05, 0.55],
        [4.05, 5.05, 6.05, 0.85],
        [7.05, 8.05, 9.05, 0.35]
    ], dtype=np.float32)
    
    pose3 = scancontext.PointXYZ(0.05, 0.05, 0.0)
    scan_id3 = 2
    loop_result = sc_manager.detect_loop_closure(points3, scan_id3, pose3)
    print("Loop closure detection result:", loop_result)

    # Access and modify SCManager properties
    lidar_height = sc_manager.get_lidar_height()
    print("LiDAR Height:", lidar_height)

    sc_manager.set_lidar_height(2.0)
    print("Updated LiDAR Height:", sc_manager.get_lidar_height())

    # Get configuration parameters
    num_rings = sc_manager.get_num_rings()
    num_sectors = sc_manager.get_num_sectors()
    max_radius = sc_manager.get_max_radius()
    distance_threshold = sc_manager.get_distance_threshold()
    
    print(f"Number of rings: {num_rings}")
    print(f"Number of sectors: {num_sectors}")
    print(f"Max radius: {max_radius}")
    print(f"Distance threshold: {distance_threshold}")

    # Set configuration parameters
    sc_manager.set_position_search_radius(10.0)
    sc_manager.set_position_search_min_candidates(5)
    sc_manager.set_position_search_max_candidates(20)
    sc_manager.set_time_exclusion_window(30)
    print("Updated configuration parameters")

    # Get database information
    database_size = sc_manager.get_database_size()
    print(f"Database size: {database_size}")

    # Get scan database (returns reference to internal data)
    scan_database = sc_manager.get_scan_database()
    print(f"Scan database type: {type(scan_database)}")

    # Test utility functions
    theta = scancontext.xy_to_theta(1.0, 1.0)
    print(f"Angle from xy(1,1): {theta}")
    
    radians = scancontext.deg_to_rad(45.0)
    degrees = scancontext.rad_to_deg(radians)
    print(f"45 degrees = {radians} radians = {degrees} degrees")

    # Test core import
    scancontext.core_import_test()
    print("Core import test completed")

if __name__ == "__main__":
    main()