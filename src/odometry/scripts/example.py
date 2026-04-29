#!/usr/bin/env python3
"""
Example usage of Direct LiDAR Odometry Python Bindings (odometry_py)
Place this script in odometry_py/scripts/example_usage.py and run:

    cd odometry_py/scripts
    python3 example_usage.py
"""
import time
import numpy as np

# Import bindings
import odometry as dlo

def main():
    # Fix random seed for reproducibility
    np.random.seed(42)
    
    # Create odometry object
    odom = dlo.DirectLidarOdometry()
    print("[Python] Initialized DirectLidarOdometry?", odom.is_initialized())

    # Configure odometry parameters
    odom.configure(
        keyframe_dist_thresh=0.5,
        keyframe_rot_thresh=10.0,
        enable_adaptive=False,
        use_voxel_filter=True,
        voxel_size=0.1
    )
    print("[Python] Configuration set.")

    # Set additional parameters
    odom.set_keyframe_thresholds(distance_thresh=0.6, rotation_thresh=12.0)
    odom.set_submap_parameters(knn=20, kcv=5, kcc=10)
    odom.set_voxel_filter_parameters(use_scan_filter=True, scan_res=0.2,
                                     use_submap_filter=True, submap_res=0.2)
    odom.set_gicp_parameters(min_points=50, max_iter_s2s=30, max_iter_s2m=30,
                              transform_eps=1e-6, fitness_eps=1e-6)
    odom.set_adaptive_parameters(enable=True)
    print("[Python] All parameters configured.")

    # Optionally set an initial pose
    init_pose = [0.0, 0.0, 0.0,   # position x,y,z
                 0.0, 0.0, 0.0, 1.0]  # orientation quaternion x,y,z,w
    odom.set_initial_pose(init_pose)
    print("[Python] Initial pose set.")

    # Simulate feeding a sequence of point clouds
    trajectory = []
    metrics_list = []
    inference_times = []
    
    for i in range(5):
        # Generate dummy point cloud (1000 points)
        points = np.random.uniform(-5, 5, size=(1000, 3)).astype(np.float32)
        timestamp = time.time()

        # Track odometry for this scan with timing
        start_time = time.time()
        result = odom.track(points, timestamp)
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        hz = 1.0 / inference_time if inference_time > 0 else 0.0

        # Collect and print odometry result
        print(f"[Python] Frame {i}: success={result.success}, timestamp={result.timestamp}")
        print(f"[Python]   Position: {result.position}")
        print(f"[Python]   Orientation: {result.orientation}")
        print(f"[Python]   Num keyframes: {result.num_keyframes}")
        print(f"[Python]   Computation time: {result.computation_time} s")
        print(f"[Python]   Inference time: {inference_time:.4f} s ({hz:.2f} Hz)")
        
        # Convert to lists first to avoid potential indexing issues
        pos_list = list(result.position) if hasattr(result.position, '__iter__') else [result.position]
        orient_list = list(result.orientation) if hasattr(result.orientation, '__iter__') else [result.orientation]
        trajectory.append(pos_list + orient_list)

        # Retrieve metrics
        met = odom.get_metrics()
        print(f"[Python]   Spaciousness: {met.spaciousness}")
        metrics_list.append(met.computation_times)

    # Print average Hz
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        avg_hz = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        print(f"[Python]\nAverage inference time: {avg_inference_time:.4f} s ({avg_hz:.2f} Hz)")

    # Access other information
    current_pose = odom.get_current_pose()
    print("[Python] Current pose [x,y,z,qx,qy,qz,qw]:", current_pose)

    current_tf = odom.get_current_transformation()
    print("[Python] Current flat transformation matrix:", current_tf)

    traj_poses = odom.get_trajectory_poses()
    print(f"[Python] Trajectory poses (count={len(traj_poses)}):", traj_poses)

    num_kf = odom.get_num_keyframes()
    print("[Python] Number of keyframes:", num_kf)

    # Retrieve the map and keyframes cloud as numpy arrays
    map_pts = odom.get_map()
    kf_cloud = odom.get_keyframes_cloud()
    print(f"[Python] Map points shape: {map_pts.shape}")
    print(f"[Python] Keyframes cloud shape: {kf_cloud.shape}")

    # Get computation times and reset
    comp_times = odom.get_computation_times()
    print("[Python] Computation times for each module:", comp_times)

    odom.reset()
    print("[Python] Odometry system reset. Initialized?", odom.is_initialized())

if __name__ == '__main__':
    main()
