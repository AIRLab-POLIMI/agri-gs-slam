#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import odometry as dlo
from dataloader import DataloaderAgriGS
import torch
from keyframe import KeyframeAgriGS


class TrackerAgriGS:
    """Lightweight wrapper class for Direct LiDAR Odometry"""
    
    def __init__(self, config=None):
        # Initialize Direct LiDAR Odometry
        self.odom = dlo.DirectLidarOdometry()
        
        # Store config
        self.config = config
        self.keyframes_counter = 0
        
        # Configure with provided config or defaults
        self.configure()
    
    def configure(self):
        """Configure DLO parameters from config"""
        if self.config is None:
            # Use default values
            keyframe_dist_thresh = 1.0
            keyframe_rot_thresh = 15.0
            enable_adaptive = True
            use_voxel_filter = True
            voxel_size = 0.1
            
            # Submap parameters
            knn = 10
            kcv = 10
            kcc = 10
            
            # Voxel filter parameters
            use_scan_filter = True
            scan_res = 0.1
            use_submap_filter = True
            submap_res = 0.25
            
            # GICP parameters
            min_points = 10
            max_iter_s2s = 64
            max_iter_s2m = 64
            transform_eps = 0.01
            fitness_eps = 0.01
            
            # Normal computation parameters
            self.normals_k_neighbors = 50
            self.normals_radius = 0.15
            
            # Initial pose
            init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        else:
            # Extract parameters from config
            keyframe_dist_thresh = self.config.get('keyframe_dist_thresh', 1.0)
            keyframe_rot_thresh = self.config.get('keyframe_rot_thresh', 15.0)
            enable_adaptive = self.config.get('enable_adaptive', True)
            use_voxel_filter = self.config.get('use_voxel_filter', True)
            voxel_size = self.config.get('voxel_size', 0.1)
            
            # Submap parameters
            submap_config = self.config.get('submap', {})
            knn = submap_config.get('knn', 10)
            kcv = submap_config.get('kcv', 10)
            kcc = submap_config.get('kcc', 10)
            
            # Voxel filter parameters
            voxel_filter_config = self.config.get('voxel_filter', {})
            use_scan_filter = voxel_filter_config.get('use_scan_filter', True)
            scan_res = voxel_filter_config.get('scan_res', 0.1)
            use_submap_filter = voxel_filter_config.get('use_submap_filter', True)
            submap_res = voxel_filter_config.get('submap_res', 0.25)
            
            # GICP parameters
            gicp_config = self.config.get('gicp', {})
            min_points = gicp_config.get('min_points', 10)
            max_iter_s2s = gicp_config.get('max_iter_s2s', 64)
            max_iter_s2m = gicp_config.get('max_iter_s2m', 64)
            transform_eps = gicp_config.get('transform_eps', 0.01)
            fitness_eps = gicp_config.get('fitness_eps', 0.01)
            
            # Normal computation parameters
            normals_config = self.config.get('normals', {})
            self.normals_k_neighbors = normals_config.get('k_neighbors', 10)
            self.normals_radius = normals_config.get('radius', 0.20)
            
            # Initial pose
            init_pose = self.config.get('initial_pose', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        # Configure DLO with extracted parameters
        self.odom.configure(
            keyframe_dist_thresh=keyframe_dist_thresh,
            keyframe_rot_thresh=keyframe_rot_thresh,
            enable_adaptive=enable_adaptive,
            use_voxel_filter=use_voxel_filter,
            voxel_size=voxel_size
        )
        
        # Set additional parameters
        self.odom.set_keyframe_thresholds(distance_thresh=keyframe_dist_thresh, 
                                         rotation_thresh=keyframe_rot_thresh)
        self.odom.set_submap_parameters(knn=knn, kcv=kcv, kcc=kcc)
        self.odom.set_voxel_filter_parameters(use_scan_filter=use_scan_filter, scan_res=scan_res,
                                             use_submap_filter=use_submap_filter, submap_res=submap_res)
        self.odom.set_gicp_parameters(min_points=min_points, max_iter_s2s=max_iter_s2s, max_iter_s2m=max_iter_s2m,
                                      transform_eps=transform_eps, fitness_eps=fitness_eps)
        self.odom.set_adaptive_parameters(enable=enable_adaptive)
        
        # Set initial pose
        self.odom.set_initial_pose(init_pose)
    
    def track(self, data_item, verbose=False):
        """
        Track a single frame and return KeyframeAgriGS if it's a keyframe
        
        Args:
            data_item: Dataset item containing lidar, points, colors, robot2camera, Ks, ts
            verbose: Enable verbose output
            
        Returns:
            KeyframeAgriGS object if this frame is a keyframe, otherwise None
        """
        # Get LiDAR data for tracking
        lidar_tensor = data_item['lidar'].cpu().numpy()
        lidar_points = lidar_tensor.astype(np.float32)
        
        # Track using LiDAR data
        result = self.odom.track(lidar_points, data_item['ts'].item())
        
        if not result.success:
            return None
            
        # Only create KeyframeAgriGS if this is a keyframe
        if result.is_keyframe:
            # Get pose
            pos_list = list(result.position) if hasattr(result.position, '__iter__') else [result.position]
            orient_list = list(result.orientation) if hasattr(result.orientation, '__iter__') else [result.orientation]
            
            # Create world2robot transform
            world2robot = np.eye(4)
            world2robot[:3, 3] = pos_list
            qx, qy, qz, qw = orient_list
            R_mat = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
            world2robot[:3, :3] = R_mat
            
            # Get RGB data
            points_tensor = data_item['points'].cpu().numpy()
            colors_tensor = data_item['colors'].cpu().numpy()
            robot2camera = data_item['robot2camera'].cpu().numpy()
            
            # Calculate world2cameras
            world2cameras = []
            for c in range(robot2camera.shape[0]):
                robot2cam = robot2camera[c]
                world2cam = world2robot @ robot2cam
                world2cameras.append(world2cam)
            
            # Transform LiDAR points to world coordinates
            lidar_pc = o3d.geometry.PointCloud()
            lidar_pc.points = o3d.utility.Vector3dVector(lidar_points)
            lidar_pc.transform(world2robot)
            lidar_world_points = np.asarray(lidar_pc.points)
            
            # Get min/max z values from LiDAR pointcloud
            lidar_z_min = np.min(lidar_world_points[:, 2])
            lidar_z_max = np.max(lidar_world_points[:, 2])
            
            # Transform RGB points to world coordinates
            all_rgb_points = []
            all_rgb_colors = []
            
            for c in range(points_tensor.shape[0]):
                pts = points_tensor[c]
                clr = colors_tensor[c]
                
                # Create point cloud
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(pts)
                pc.colors = o3d.utility.Vector3dVector(clr)
                
                # Transform to world coordinates
                pc.transform(world2robot)
                
                # Get transformed RGB points
                rgb_world_points = np.asarray(pc.points)
                rgb_colors = np.asarray(pc.colors)
                
                all_rgb_points.append(rgb_world_points)
                all_rgb_colors.append(rgb_colors)
            
            # Merge RGB points
            merged_points = np.vstack(all_rgb_points) if all_rgb_points else np.empty((0, 3))
            merged_colors = np.vstack(all_rgb_colors) if all_rgb_colors else np.empty((0, 3))
            
            # Filter merged pointcloud based on LiDAR z range
            if len(merged_points) > 0:
                z_mask = (merged_points[:, 2] >= lidar_z_min) & (merged_points[:, 2] <= lidar_z_max)
                merged_points = merged_points[z_mask]
                merged_colors = merged_colors[z_mask]
                
                if verbose:
                    print(f"Filtered RGB points by LiDAR z-range [{lidar_z_min:.2f}, {lidar_z_max:.2f}]: "
                          f"{np.sum(z_mask)}/{len(z_mask)} points kept")
            
            # Create KeyframeAgriGS object
            try:
                # Convert numpy arrays to CPU tensors
                lidar_tensor_torch = torch.from_numpy(lidar_world_points).float().cpu()
                world2robot_tensor = torch.from_numpy(world2robot).float().cpu()
                
                # Create camera poses dictionary with string keys
                world2cams_dict = {}
                for c, world2cam in enumerate(world2cameras):
                    world2cams_dict[f'cam_{c}'] = torch.from_numpy(world2cam).float().cpu()
                
                points_tensor_torch = torch.from_numpy(merged_points).float().cpu()
                colors_tensor_torch = torch.from_numpy(merged_colors).float().cpu()
                
                # Compute LiDAR normals using DLO with config parameters
                try:
                    lidar_normals_list = dlo.DirectLidarOdometry.compute_normals(
                        lidar_world_points, k_neighbors=self.normals_k_neighbors, radius=self.normals_radius
                    )
                    # Convert list of lists to numpy array
                    lidar_normals = np.array(lidar_normals_list, dtype=np.float32)
                    lidar_normals_tensor = torch.from_numpy(lidar_normals).float().cpu()
                except Exception as e:
                    if verbose:
                        print(f"⚠️ Failed to compute LiDAR normals: {e}")
                    # Use zero normals as fallback
                    lidar_normals_tensor = torch.zeros((len(lidar_world_points), 3), dtype=torch.float32).cpu()
                
                # Create camera IDs tensor
                camera_ids = torch.arange(robot2camera.shape[0], dtype=torch.long).cpu()
                
                # Convert robot2camera and Ks to tensors
                robot2camera_tensor = torch.from_numpy(robot2camera).float().cpu()
                Ks_tensor = data_item['Ks'].cpu() if 'Ks' in data_item else None
                
                # Get optional data if available
                images_tensor = data_item.get('image', None)
                if images_tensor is not None:
                    images_tensor = images_tensor.cpu()
                
                depths_tensor = data_item.get('depth', None)
                if depths_tensor is not None:
                    depths_tensor = depths_tensor.cpu()
                
                masks_tensor = data_item.get('mask', None)
                if masks_tensor is not None:
                    masks_tensor = masks_tensor.cpu()
                
                semantic_tensor = data_item.get('semantic', None)
                if semantic_tensor is not None:
                    semantic_tensor = semantic_tensor.cpu()
                
                # Create KeyframeAgriGS object
                keyframe_lots = KeyframeAgriGS(
                    id=self.keyframes_counter,
                    timestamp=data_item['ts'].item(),
                    lidar_depth=data_item.get('lidar_depth', None),
                    lidar_pointcloud=lidar_tensor_torch,
                    lidar_colors=data_item.get('lidar_colors', None),
                    world2robot_pose=world2robot_tensor,
                    world2cams_poses=world2cams_dict,
                    points=points_tensor_torch,
                    colors=colors_tensor_torch,
                    splat_indices=None,
                    lidar_normals=lidar_normals_tensor,
                    camera_ids=camera_ids,
                    robot2camera=robot2camera_tensor,
                    Ks=Ks_tensor,
                    images=images_tensor,
                    depths=depths_tensor,
                    masks=masks_tensor,
                    semantic=semantic_tensor
                ).to_device(torch.device('cpu'))

                self.keyframes_counter += 1

                return keyframe_lots
                
            except Exception as e:
                if verbose:
                    print(f"⚠️ Failed to create KeyframeAgriGS object: {e}")
                return None
        
        # Not a keyframe
        return None


def create_camera_frustum(world2camera, K, scale=0.1, color=[1.0, 0.4, 0.0]):
    """Create a camera frustum visualization"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    near_plane = 0.02 * scale
    far_plane = 0.2 * scale
    
    x_scale = near_plane / fx
    y_scale = near_plane / fy
    
    near_corners = np.array([
        [-cx * x_scale, -cy * y_scale, near_plane],
        [cx * x_scale, -cy * y_scale, near_plane],
        [cx * x_scale, cy * y_scale, near_plane],
        [-cx * x_scale, cy * y_scale, near_plane]
    ])
    
    x_scale_far = far_plane / fx
    y_scale_far = far_plane / fy
    
    far_corners = np.array([
        [-cx * x_scale_far, -cy * y_scale_far, far_plane],
        [cx * x_scale_far, -cy * y_scale_far, far_plane],
        [cx * x_scale_far, cy * y_scale_far, far_plane],
        [-cx * x_scale_far, cy * y_scale_far, far_plane]
    ])
    
    camera_center = np.array([0, 0, 0])
    points = np.vstack([camera_center[np.newaxis, :], near_corners, far_corners])
    
    points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
    world_points = (world2camera @ points_hom.T).T[:, :3]
    
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
        [1, 5], [2, 6], [3, 7], [4, 8],
        [5, 6], [6, 7], [7, 8], [8, 5]
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    
    return line_set


def create_visualization_from_keyframes(keyframes_lots):
    """Create visualization from list of KeyframeAgriGS objects"""
    vis_elements = []
    
    # Collect data from keyframes
    all_rgb_points = []
    all_rgb_colors = []
    keyframe_positions = []
    
    for keyframe in keyframes_lots:
        # Get RGB points and colors
        rgb_points = keyframe.get_points().cpu().numpy()
        rgb_colors = keyframe.get_colors().cpu().numpy()
        
        if len(rgb_points) > 0:
            all_rgb_points.append(rgb_points)
            all_rgb_colors.append(rgb_colors)
        
        # Get robot pose
        world2robot = keyframe.world2robot_pose.cpu().numpy()
        robot_pos = world2robot[:3, 3]
        keyframe_positions.append(robot_pos)
        
        # Add keyframe marker
        keyframe_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        keyframe_sphere.paint_uniform_color([0.5, 1, 0.5])
        keyframe_sphere.translate(robot_pos)
        vis_elements.append(keyframe_sphere)
        
        # Add coordinate frame for keyframe
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
        coord_frame.transform(world2robot)
        vis_elements.append(coord_frame)
        
        # Add camera frustums
        camera_poses = keyframe.get_camera_poses()
        camera_positions = []
        
        for cam_name, world2camera_tensor in camera_poses.items():
            world2camera = world2camera_tensor.cpu().numpy()
            camera_pos = world2camera[:3, 3]
            camera_positions.append(camera_pos)
            
            # Use default intrinsics for visualization
            K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
            
            # Create camera frustum
            camera_frustum = create_camera_frustum(
                world2camera, K, 
                scale=0.15, 
                color=[0, 0.7, 1]
            )
            vis_elements.append(camera_frustum)
            
            # Add camera coordinate frame
            camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            camera_axes.transform(world2camera)
            vis_elements.append(camera_axes)
            
            # Connect camera to keyframe robot position
            line_points = [camera_pos, robot_pos]
            lines = [[0, 1]]
            
            camera_to_robot_link = o3d.geometry.LineSet()
            camera_to_robot_link.points = o3d.utility.Vector3dVector(line_points)
            camera_to_robot_link.lines = o3d.utility.Vector2iVector(lines)
            camera_to_robot_link.colors = o3d.utility.Vector3dVector([[1.0, 0.4, 0.0]])
            
            vis_elements.append(camera_to_robot_link)
        
        # Link cameras with lines for each keyframe
        if len(camera_positions) > 1:
            for j in range(len(camera_positions)):
                for k in range(j + 1, len(camera_positions)):
                    line_points = [camera_positions[j], camera_positions[k]]
                    lines = [[0, 1]]
                    
                    camera_link = o3d.geometry.LineSet()
                    camera_link.points = o3d.utility.Vector3dVector(line_points)
                    camera_link.lines = o3d.utility.Vector2iVector(lines)
                    camera_link.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
                    
                    vis_elements.append(camera_link)
    
    # Add RGB point cloud
    if all_rgb_points:
        merged_points_np = np.vstack(all_rgb_points)
        merged_colors_np = np.vstack(all_rgb_colors)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(merged_points_np)
        pc.colors = o3d.utility.Vector3dVector(merged_colors_np)
        vis_elements.append(pc)
        print(f"✓ Added RGB point cloud: {len(merged_points_np)} points")
    
    # Draw keyframe trajectory path
    if len(keyframe_positions) > 1:
        for i in range(len(keyframe_positions) - 1):
            start_pos = keyframe_positions[i]
            end_pos = keyframe_positions[i + 1]
            
            line_points = [start_pos, end_pos]
            lines = [[0, 1]]
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
            
            vis_elements.append(line_set)
        
        print(f"✓ Added keyframe trajectory: {len(keyframe_positions)} keyframes")
    
    # Enhanced origin marker
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    origin_sphere.paint_uniform_color([1, 0.8, 0])
    vis_elements.append(origin_sphere)
    
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis_elements.append(world_axes)
    
    return vis_elements


def print_statistics(keyframes_lots):
    """Print comprehensive statistics from KeyframeAgriGS objects"""
    if not keyframes_lots:
        print("No keyframes to analyze.")
        return
    
    total_rgb_points = 0
    total_lidar_points = 0
    total_cameras = 0
    
    for keyframe in keyframes_lots:
        total_rgb_points += len(keyframe.get_points())
        total_lidar_points += len(keyframe.get_lidar_points())
        total_cameras += len(keyframe.get_camera_poses())
    
    print(f"\n📊 KEYFRAME STATISTICS:")
    print(f"  Total keyframes: {len(keyframes_lots)}")
    print(f"  Total RGB points: {total_rgb_points:,}")
    print(f"  Total LiDAR points: {total_lidar_points:,}")
    print(f"  Total camera poses: {total_cameras}")
    print(f"  Average RGB points per keyframe: {total_rgb_points/len(keyframes_lots):.0f}")
    print(f"  Average LiDAR points per keyframe: {total_lidar_points/len(keyframes_lots):.0f}")
    print(f"  Average cameras per keyframe: {total_cameras/len(keyframes_lots):.1f}")


def print_visualization_summary(vis_elements, keyframes_lots):
    """Print visualization summary"""
    if not keyframes_lots:
        print("No keyframes for visualization.")
        return
    
    total_rgb_points = sum(len(kf.get_points()) for kf in keyframes_lots)
    total_lidar_points = sum(len(kf.get_lidar_points()) for kf in keyframes_lots)
    keyframe_count = len(keyframes_lots)
    
    print(f"\n🎨 Visualization Summary:")
    print(f"  🌈 RGB points: Colored point cloud ({total_rgb_points:,})")
    print(f"  🟢 Green spheres: Keyframe poses ({keyframe_count})")
    print(f"  🔵 Blue frustums: Camera poses")
    print(f"  🟠 Orange lines: Camera-to-robot links")
    print(f"  ⚫ Black lines: Inter-camera links")
    print(f"  🟢 Green lines: Keyframe trajectory path")
    print(f"  🟡 Gold sphere: World origin")
    print(f"  📊 Total elements: {len(vis_elements)}")


if __name__ == "__main__":
    import yaml
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Direct LiDAR Odometry with RGB Visualization')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--config', '-c', default='/agrigs_slam/config/default.yaml', help='Config file path')
    parser.add_argument('--start', '-s', type=int, default=300, help='Start frame index')
    parser.add_argument('--end', '-e', type=int, default=350, help='End frame index')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    try:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)["agrigs_slam"]
            data_configs = config_dict["dataloader"]
            tracker_configs = config_dict["slam"]["frontend"]
    except Exception as e:
        print(f"Config load error ({e}), using dummy config.")
        exit(1)
    
    # Initialize dataset
    print(f"Initializing DataloaderAgriGS with storage: {data_configs['path']}")
    dataset = DataloaderAgriGS(data_configs)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        exit(1)
    
    # Initialize TrackerAgriGS
    tracker = TrackerAgriGS(tracker_configs)
    
    print(f"\n🚀 Starting Direct LiDAR Odometry processing with TrackerAgriGS...")
    print(f"Processing frames {args.start} to {args.end-1}")
    
    # Store keyframes for visualization and statistics
    keyframes_lots = []
    
    # Process frames
    for i in range(args.start, args.end):
        data_item = dataset[i]
        
        # Track frame using TrackerAgriGS
        keyframe_lots = tracker.track(data_item, verbose=args.verbose)
        
        # Check if a KeyframeAgriGS object was created
        if keyframe_lots is not None:
            keyframes_lots.append(keyframe_lots)
            print(f"🔑 Keyframe {i}: KeyframeAgriGS created with {len(keyframe_lots.get_lidar_points())} LiDAR points, "
                  f"{len(keyframe_lots.get_points())} RGB points, {len(keyframe_lots.get_camera_poses())} cameras")
        else:
            if args.verbose:
                print(f"Frame {i}: Not a keyframe")
    
    # Print keyframe summary
    print(f"\n🔑 Total KeyframeAgriGS objects created: {len(keyframes_lots)}")
    
    # Print comprehensive statistics
    print_statistics(keyframes_lots)
    
    # Create and display visualization
    if keyframes_lots:
        vis_elements = create_visualization_from_keyframes(keyframes_lots)
        
        # Print visualization summary
        print_visualization_summary(vis_elements, keyframes_lots)
        
        # Launch Open3D visualization
        if vis_elements:
            window_title = "Direct LiDAR Odometry with RGB Visualization"
            o3d.visualization.draw_geometries(vis_elements, window_name=window_title, width=1280, height=720)
        else:
            print("No visual elements to display. Exiting visualization.")
    else:
        print("No keyframes created. Nothing to visualize.")
    
    print("👋 Goodbye!")
