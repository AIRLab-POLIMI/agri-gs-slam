import os
import glob
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
import re
import bisect
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from camera import CameraAgriGS
import multiprocessing as mp
import gc

class DataloaderAgriGS(Dataset):
    class Mode(Enum):
        TRAINING = "train"
        VALIDATION = "val"

    def __init__(self, config: Dict, mode: 'DataloaderAgriGS.Mode'):
        """
        Initialize the AgriGS dataset from a configuration dictionary,
        adapted for the new directory structure and timestamp-based synchronization.

        Args:
            config (Dict): Configuration dictionary with at least the "path" key.
            mode (DataloaderAgriGS.Mode): Specifies whether to load training or validation data.
        """
        base_path = config.get("path", '')
        if not base_path or not os.path.isdir(base_path):
            raise ValueError(f"Base path '{base_path}' not found or not a directory.")

        self.storage = os.path.join(base_path, mode.value)
        if not os.path.isdir(self.storage):
            raise ValueError(f"Data path '{self.storage}' not found or not a directory.")
        self.mode = mode

        # Initialize cameras from config, filtering by `enabled_cameras` (default: all)
        enabled = config.get("enabled_cameras")
        all_ids = [c["camera_id"] for c in config["cameras"]]
        if enabled is None:
            enabled_ids = all_ids
        else:
            unknown = [c for c in enabled if c not in all_ids]
            if unknown:
                raise ValueError(f"enabled_cameras references unknown camera(s) {unknown}; defined: {all_ids}")
            enabled_ids = list(enabled)
        if not enabled_ids:
            raise ValueError("enabled_cameras is empty; specify at least one camera_id or remove the key to use all.")
        print(f"Enabled cameras: {enabled_ids}")

        self.cameras = {}
        for cam_cfg in config["cameras"]:
            if cam_cfg["camera_id"] not in enabled_ids:
                continue
            self.cameras[cam_cfg["camera_id"]] = CameraAgriGS(
                camera_id=cam_cfg["camera_id"],
                extrinsic=cam_cfg["extrinsic"],
                intrinsic=cam_cfg["intrinsic"],
                distortion=cam_cfg.get("distortion", None)
            )

        if not self.cameras:
            raise ValueError("No cameras defined in the configuration.")

        # Pre-compute all camera matrices and transformations
        self.camera_list = list(self.cameras.values())
        self.camera_ids = list(self.cameras.keys())
        self.Ks = np.stack([cam.get_camera_matrix() for cam in self.camera_list], axis=0)
        self.robot2visions = np.stack([cam.extrinsic for cam in self.camera_list], axis=0)
        self.vision2robots = np.linalg.inv(self.robot2visions)

        # Pre-compute transformations
        self.sensor2vision = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.vision2sensor = np.linalg.inv(self.sensor2vision)

        # Initialize lidar transformation
        self._init_lidar_transform(config)
        
        self._first_pose_matrix = None
        self._first_pose_inv = None

        # Pre-compute camera transformations
        self.robot2sensors = self.robot2visions @ self.vision2sensor
        self.robot2cameras = self.robot2sensors @ self.sensor2vision

        # Use optimal number of workers
        self.max_workers = min(mp.cpu_count(), 8)
        
        # Pre-compile regex patterns
        self.ts_pattern_sec_nano = re.compile(r'(\d+)-(\d+)\..+')
        self.ts_pattern_float = re.compile(r'(\d+\.\d+)\.[^.]+$')
        
        self.samples: List[Dict] = []
        self._populate_samples()

    def _init_lidar_transform(self, config: Dict):
        """Initialize lidar to robot transformation matrix."""
        lidar_extrinsic = np.array(config["lidar"]["extrinsic"], dtype=np.float32)
        
        if lidar_extrinsic.shape == (7,):
            x, y, z, qx, qy, qz, qw = lidar_extrinsic
            rotation = Rotation.from_quat([qx, qy, qz, qw])
            self.lidar2robot = np.eye(4, dtype=np.float32)
            self.lidar2robot[:3, :3] = rotation.as_matrix()
            self.lidar2robot[:3, 3] = [x, y, z]
        elif lidar_extrinsic.shape == (4, 4):
            self.lidar2robot = lidar_extrinsic
        else:
            self.lidar2robot = None
        
        # Clean up temporary variable
        del lidar_extrinsic

    @lru_cache(maxsize=2000)
    def _parse_timestamp_from_filename(self, filename: str) -> Optional[float]:
        """Parse timestamp from filename with pre-compiled regex."""
        base = os.path.basename(filename)
        
        # Try sec-nanosec format first (more common)
        match = self.ts_pattern_sec_nano.match(base)
        if match:
            secs = int(match.group(1))
            nanosecs = int(match.group(2))
            return secs + nanosecs / 1e9

        # Try float format
        match = self.ts_pattern_float.search(base)
        if match:
            return float(match.group(1))
        
        return None

    def _get_files_with_timestamps(self, directory: str, pattern: str) -> List[Tuple[str, float]]:
        """Get files with timestamps using os.scandir for better performance."""
        if not os.path.isdir(directory):
            return []
        
        files_with_ts = []
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(pattern.replace('*', '')):
                        ts = self._parse_timestamp_from_filename(entry.name)
                        if ts is not None:
                            files_with_ts.append((entry.path, ts))
        except OSError:
            return []
        
        files_with_ts.sort(key=lambda x: x[1])
        return files_with_ts

    def _find_closest_file_from_list(self, candidate_files: List[Tuple[str, float]], target_ts: float) -> Tuple[Optional[str], Optional[float]]:
        """Find closest file using binary search for O(log n) complexity."""
        if not candidate_files:
            return None, None

        timestamps = [ts for _, ts in candidate_files]
        idx = bisect.bisect_left(timestamps, target_ts)

        if idx == 0:
            return candidate_files[0]
        elif idx == len(timestamps):
            return candidate_files[-1]
        else:
            path_before, ts_before = candidate_files[idx-1]
            path_after, ts_after = candidate_files[idx]
            if target_ts - ts_before <= ts_after - target_ts:
                return path_before, ts_before
            else:
                return path_after, ts_after

    def _populate_samples(self):
        """Populate samples with optimized sequential file loading."""
        self.samples = []

        # Define base paths
        pose_base_path = os.path.join(self.storage, "fixposition", "odometry", "poses")
        ouster_base_path = os.path.join(self.storage, "ouster", "points")

        # Load files with timestamps sequentially
        pose_files_with_ts = self._get_files_with_timestamps(pose_base_path, ".csv")
        ouster_files_with_ts = self._get_files_with_timestamps(ouster_base_path, ".ply")
        
        camera_data_files_with_ts = {}
        for camera_id in self.camera_ids:
            zed_rgb_dir = os.path.join(self.storage, "zed_multi", camera_id, "rgb")
            zed_pc_dir = os.path.join(self.storage, "zed_multi", camera_id, "point_cloud")
            zed_depth_dir = os.path.join(self.storage, "zed_multi", camera_id, "depth_anything")
            
            camera_data_files_with_ts[camera_id] = {
                'rgb': self._get_files_with_timestamps(zed_rgb_dir, ".jpg"),
                'depth': self._get_files_with_timestamps(zed_depth_dir, ".png"),
                'point_cloud': self._get_files_with_timestamps(zed_pc_dir, ".ply")
            }

        if not pose_files_with_ts:
            print("Warning: No pose files found. Cannot populate samples.")
            return

        # Process samples with early termination checks
        for pose_path, pose_ts in pose_files_with_ts:
            # Find closest Ouster cloud
            closest_ouster_path, _ = self._find_closest_file_from_list(ouster_files_with_ts, pose_ts)
            if not closest_ouster_path:
                continue

            # Check all cameras have required data
            temp_camera_info = {}
            all_cameras_ok = True
            
            for camera_id in self.camera_ids:
                cam_data = camera_data_files_with_ts[camera_id]
                
                closest_rgb_path, _ = self._find_closest_file_from_list(cam_data['rgb'], pose_ts)
                closest_depth_path, _ = self._find_closest_file_from_list(cam_data['depth'], pose_ts)
                closest_pc_path, _ = self._find_closest_file_from_list(cam_data['point_cloud'], pose_ts)

                if not closest_rgb_path or not closest_depth_path:
                    all_cameras_ok = False
                    break
                
                temp_camera_info[camera_id] = {
                    'image': closest_rgb_path,
                    'depth': closest_depth_path,
                    'point_cloud': closest_pc_path,
                    'semantic': None
                }
            
            if all_cameras_ok:
                self.samples.append({
                    'pose': pose_path,
                    'cloud': closest_ouster_path,
                    'cameras': temp_camera_info
                })
            
            # Clean up temporary variables
            del temp_camera_info

        # Clean up temporary variables
        del pose_files_with_ts
        del ouster_files_with_ts
        del camera_data_files_with_ts
        gc.collect()

    def __len__(self) -> int:
        return len(self.samples)

    def _process_camera_data(self, file_info: Dict[str, Optional[str]], cam_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Process single camera data with optimized operations."""
        img, dep, m, p, c = self._process_image_data(file_info)
        
        # Apply sky removal with pre-computed matrices
        if len(p) > 0:
            p, c = self._optimized_sky_removal(p, c, dep, 
                                             self.Ks[cam_idx], 
                                             self.robot2sensors[cam_idx])
        
        return img, dep, m, p, c

    def _project_lidar_to_cameras_vectorized(self, lidar_points: np.ndarray, camera_files: Dict) -> np.ndarray:
        """Vectorized lidar projection to all cameras."""
        if len(lidar_points) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        # Initialize colors array
        colors = np.zeros((len(lidar_points), 3), dtype=np.float32)
        valid_count = np.zeros(len(lidar_points), dtype=np.int32)
        
        # Create homogeneous coordinates once
        lidar_homogeneous = np.column_stack((lidar_points, np.ones(len(lidar_points), dtype=np.float32)))
        
        for i, cam_id in enumerate(self.camera_ids):
            if cam_id not in camera_files or not camera_files[cam_id]['image']:
                continue
                
            try:
                # Load image
                image = self.load_rgb_image(camera_files[cam_id]['image'])
                h, w = image.shape[:2]
                
                # Transform to vision frame using pre-computed matrix
                points_vision = (self.vision2robots[i] @ lidar_homogeneous.T).T[:, :3]
                
                # Project to image plane using pre-computed K
                proj = (self.Ks[i] @ points_vision.T).T
                z = proj[:, 2]
                
                # Filter and project in one step
                valid_mask = z > 0.1
                if not np.any(valid_mask):
                    continue
                
                z_valid = z[valid_mask]
                u = np.round(proj[valid_mask, 0] / z_valid).astype(np.int32)
                v = np.round(proj[valid_mask, 1] / z_valid).astype(np.int32)
                
                # Bounds check
                bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if not np.any(bounds_mask):
                    continue
                
                # Get valid indices and colors
                valid_indices = np.where(valid_mask)[0]
                bounds_indices = valid_indices[bounds_mask]
                
                u_bounds = u[bounds_mask]
                v_bounds = v[bounds_mask]
                pixel_colors = image[v_bounds, u_bounds].astype(np.float32) / 255.0
                
                # Accumulate colors
                colors[bounds_indices] += pixel_colors
                valid_count[bounds_indices] += 1
                
                # Clean up temporary variables
                del image
                del points_vision
                del proj
                del z
                del valid_mask
                del z_valid
                del u
                del v
                del bounds_mask
                del valid_indices
                del bounds_indices
                del u_bounds
                del v_bounds
                del pixel_colors
                
            except Exception as e:
                print(f"Error processing camera {cam_id}: {e}")
                continue
        
        # Average colors
        valid_mask = valid_count > 0
        colors[valid_mask] /= valid_count[valid_mask][:, np.newaxis]
        
        # Clean up temporary variables
        del lidar_homogeneous
        del valid_count
        del valid_mask
        
        return colors
    
    def _optimized_sky_removal(self, p: np.ndarray, c: np.ndarray, dep: np.ndarray, 
                              K: np.ndarray, robot2sensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized sky removal with vectorized operations."""
        if len(p) == 0:
            return p, c
            
        # Transform points (vectorized)
        p_homogeneous = np.column_stack((p, np.ones(len(p), dtype=np.float32)))
        p_vision = (self.vision2sensor @ p_homogeneous.T).T[:, :3]

        # Project to image plane (vectorized)
        proj = (K @ p_vision.T).T
        z = proj[:, 2]
        
        # Avoid division by zero
        valid_z = np.abs(z) > 1e-10
        if not np.any(valid_z):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
            
        # Filter valid z values
        p_vision_valid = p_vision[valid_z]
        c_valid = c[valid_z]
        proj_valid = proj[valid_z]
        z_valid = proj_valid[:, 2]
        
        # Compute pixel coordinates
        u = (proj_valid[:, 0] / z_valid).round().astype(np.int32)
        v = (proj_valid[:, 1] / z_valid).round().astype(np.int32)

        # Check bounds
        h, w = dep.shape[:2]
        bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        if not np.any(bounds_mask):
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        # Check depth validity
        u_bounds = u[bounds_mask]
        v_bounds = v[bounds_mask]
        depth_values = dep[v_bounds, u_bounds].squeeze()
        depth_valid = depth_values > 0.0
        
        # Combine all masks
        final_mask = np.zeros(len(p_vision_valid), dtype=bool)
        final_mask[bounds_mask] = depth_valid
        
        # Apply final filtering and transform back
        p_filtered = p_vision_valid[final_mask]
        c_filtered = c_valid[final_mask]
        
        if len(p_filtered) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
            
        p_homogeneous_filtered = np.column_stack((p_filtered, np.ones(len(p_filtered), dtype=np.float32)))
        p_final = (robot2sensor @ self.sensor2vision @ p_homogeneous_filtered.T).T[:, :3]

        # Clean up temporary variables
        del p_homogeneous
        del p_vision
        del proj
        del z
        del valid_z
        del p_vision_valid
        del c_valid
        del proj_valid
        del z_valid
        del u
        del v
        del bounds_mask
        del u_bounds
        del v_bounds
        del depth_values
        del depth_valid
        del final_mask
        del p_filtered
        del p_homogeneous_filtered

        return p_final, c_filtered
    
    def _process_image_data(self, file_info: Dict[str, Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Process image data with optimized tensor operations."""
        image_path = file_info['image']
        depth_path = file_info['depth']
        pc_path = file_info['point_cloud']

        if not image_path or not depth_path:
            raise ValueError("Image or depth path is missing in file_info.")

        # Load images
        image = self.load_rgb_image(image_path)
        depth = self.load_depth_image(depth_path)
        pc = self.load_camera_cloud(pc_path)

        # Optimized tensor conversion
        image_tensor = torch.from_numpy(image).float().div_(255.0)
        
        # Handle depth efficiently
        if depth.ndim == 3:
            if depth.shape[2] == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            elif depth.shape[2] == 4:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGRA2GRAY)
            else:
                depth = depth[:,:,0]
        
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(-1)
        valid_depth_mask = depth_tensor.gt(1e-6).squeeze()

        # Process point cloud
        if pc is not None and pc.has_points():
            points = np.asarray(pc.points, dtype=np.float32)
            colors = np.asarray(pc.colors, dtype=np.float32)
            
            if colors.size == 0 and points.size > 0:
                colors = np.full_like(points, 0.5, dtype=np.float32)
                
            # Fast filtering
            valid_mask = ~(np.isnan(points) | np.isinf(points)).any(axis=1)
            if np.any(valid_mask):
                points = points[valid_mask]
                colors = colors[valid_mask]
                
                # Normalize colors
                if colors.size > 0 and colors.max() > 1.0:
                    colors = np.clip(colors / 255.0, 0.0, 1.0)
            else:
                points = np.empty((0, 3), dtype=np.float32)
                colors = np.empty((0, 3), dtype=np.float32)
        else:
            points = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.float32)

        # Clean up temporary variables
        del image
        del depth
        if pc is not None:
            del pc

        return image_tensor, depth_tensor, valid_depth_mask, points, colors

    @staticmethod
    def load_rgb_image(path: str) -> np.ndarray:
        """Load RGB image with optimized error handling."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to load RGB image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_depth_image(path: str, dilate_zeros: int = 2) -> np.ndarray:
        """Load depth image with optimized processing."""
        depth_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise IOError(f"Failed to load depth image: {path}")
        
        # Convert to grayscale if needed
        if depth_img.ndim == 3:
            if depth_img.shape[2] == 3:
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
            elif depth_img.shape[2] == 4:
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGRA2GRAY)
            else:
                depth_img = depth_img[:,:,0]
        
        # Optimized dilation
        if dilate_zeros > 0:
            zero_mask = (depth_img == 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (2 * dilate_zeros + 1, 2 * dilate_zeros + 1))
            dilated_mask = cv2.dilate(zero_mask.astype(np.uint8), kernel, iterations=1)
            depth_img[dilated_mask > 0] = 0.0
            
            # Clean up temporary variables
            del zero_mask
            del kernel
            del dilated_mask

        return depth_img

    def load_lidar_cloud(self, path: str) -> np.ndarray:
        """Load and filter lidar point cloud with maximum optimization."""
        cloud = o3d.io.read_point_cloud(path, remove_nan_points=True, remove_infinite_points=True)
        points = np.asarray(cloud.points, dtype=np.float32)

        # Clean up cloud immediately
        del cloud

        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Transform to robot frame
        if self.lidar2robot is not None:
            points_homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
            points_robot = (self.lidar2robot @ points_homogeneous.T).T[:, :3]
            del points_homogeneous
        else:
            points_robot = points

        # Combined vectorized filtering
        abs_x = np.abs(points_robot[:, 0])
        abs_y = np.abs(points_robot[:, 1])
        abs_z = np.abs(points_robot[:, 2])
        
        box_mask = (abs_x > 3.0) | (abs_y > 1.0) | (abs_z > 2.0)
        distances_sq = np.sum(points_robot * points_robot, axis=1)
        distance_mask = distances_sq <= 400.0  # 20.0^2
        
        result = points_robot[box_mask & distance_mask]
        
        # Clean up temporary variables
        del points
        del points_robot
        del abs_x
        del abs_y
        del abs_z
        del box_mask
        del distances_sq
        del distance_mask
        
        return result
    
    def load_camera_cloud(self, path: str) -> Optional[o3d.geometry.PointCloud]:
        """Load camera point cloud with optimized filtering."""
        if not os.path.exists(path):
            return None
            
        try:
            cloud = o3d.io.read_point_cloud(path, remove_nan_points=True, remove_infinite_points=True)
            
            if not cloud.has_points():
                return None

            points = np.asarray(cloud.points, dtype=np.float32)
            colors = np.asarray(cloud.colors, dtype=np.float32)
            
            # Vectorized filtering
            height_mask = points[:, 2] <= 2.0
            distances_sq = np.sum(points * points, axis=1)
            distance_mask = distances_sq <= 100.0  # 10.0^2
            combined_mask = height_mask & distance_mask
            
            if not np.any(combined_mask):
                return None
            
            # Apply filter
            filtered_cloud = o3d.geometry.PointCloud()
            filtered_cloud.points = o3d.utility.Vector3dVector(points[combined_mask])
            filtered_cloud.colors = o3d.utility.Vector3dVector(colors[combined_mask])
            
            # Clean up temporary variables
            del points
            del colors
            del height_mask
            del distances_sq
            del distance_mask
            del combined_mask
            del cloud
            
            # Statistical outlier removal
            if len(filtered_cloud.points) > 40:  # Only if we have enough points
                filtered_cloud, _ = filtered_cloud.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
            
            return filtered_cloud if filtered_cloud.has_points() else None
            
        except Exception as e:
            print(f"Error loading point cloud {path}: {e}")
            return None
    
    def load_pose(self, path: str) -> Tuple[np.ndarray, float]:
        """Load pose with optimized parsing."""
        with open(path, 'r') as f:
            f.readline()  # Skip header
            data_line = f.readline().strip()

        parts = data_line.split(',')
        if len(parts) < 8:
            raise ValueError(f"Expected at least 8 columns, got {len(parts)} in {path}")

        # Parse timestamp
        ts_str = parts[0]
        if '-' in ts_str:
            sec, nano = ts_str.split('-', 1)
            pose_ts = int(sec) + int(nano) / 1e9
        else:
            pose_ts = float(ts_str)

        # Parse pose - convert directly to avoid intermediate variables
        translation = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
        quat = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float32)

        # Build transformation matrix
        rotation = Rotation.from_quat(quat)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation.as_matrix()
        T[:3, 3] = translation

        # Handle first pose
        if self._first_pose_matrix is None:
            self._first_pose_matrix = T.copy()
            self._first_pose_inv = np.linalg.inv(T)
            T_out = np.eye(4, dtype=np.float32)
        else:
            T_out = self._first_pose_inv @ T

        # Clean up temporary variables
        del parts
        del ts_str
        del translation
        del quat
        del rotation
        del T

        return T_out, pose_ts
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample = self.samples[idx]
        world2robot, ts = self.load_pose(sample['pose'])
        ts = int(ts)
        
        # Load lidar points once
        lidar_points = self.load_lidar_cloud(sample['cloud'])
        
        # Compute lidar normals
        lidar_normals = np.zeros((len(lidar_points), 3), dtype=np.float32)
        if len(lidar_points) > 0:
            # Create point cloud for normal estimation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_points)
            
            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50))
            lidar_normals = np.asarray(pcd.normals, dtype=np.float32)
        
        # Process cameras sequentially to minimize memory usage
        images = []
        depths = []
        masks = []
        points = []
        colors = []

        for i, cam_id in enumerate(self.camera_ids):
            if cam_id in sample['cameras']:
                img, dep, m, p, c = self._process_camera_data(sample['cameras'][cam_id], i)
                images.append(img)
                depths.append(dep)
                masks.append(m)
                points.append(p)
                colors.append(c)

        # Project lidar points to get colors
        lidar_colors = self._project_lidar_to_cameras_vectorized(lidar_points, sample['cameras'])

        # Compute lidar depth and normals for each camera
        lidar_depths = []
        lidar_normals_projected = []
        for cam_idx in range(len(images)):
            img = images[cam_idx]
            K = self.Ks[cam_idx]
            cam2robot = np.linalg.inv(self.robot2cameras[cam_idx])
            
            # Create depth map and normal map initialized to zeros
            h, w = img.shape[:2]
            depth_map = np.zeros((h, w), dtype=np.float32)
            normal_map = np.zeros((h, w, 3), dtype=np.float32)
            
            if len(lidar_points) > 0:
                # Transform lidar points to camera frame
                lidar_homogeneous = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
                lidar_cam = (cam2robot @ lidar_homogeneous.T).T[:, :3]
                
                # Transform normals to camera frame (only rotation, no translation)
                lidar_normals_cam = (cam2robot[:3, :3] @ lidar_normals.T).T
                
                # Project to image plane
                proj = (K @ lidar_cam.T).T
                z = proj[:, 2]
                
                # Filter points in front of camera
                valid_mask = z > 0.1
                if np.any(valid_mask):
                    z_valid = z[valid_mask]
                    u = proj[valid_mask, 0] / z_valid
                    v = proj[valid_mask, 1] / z_valid
                    
                    # Filter points within image bounds
                    bounds_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                    if np.any(bounds_mask):
                        u_bounds = np.clip(np.round(u[bounds_mask]).astype(np.int32), 0, w-1)
                        v_bounds = np.clip(np.round(v[bounds_mask]).astype(np.int32), 0, h-1)
                        # Set depth values in the depth map
                        depth_map[v_bounds, u_bounds] = z_valid[bounds_mask]
                        # Set normal values in the normal map
                        normal_map[v_bounds, u_bounds] = lidar_normals_cam[valid_mask][bounds_mask]
            
            lidar_depths.append(torch.from_numpy(depth_map).float().unsqueeze(-1))
            lidar_normals_projected.append(torch.from_numpy(normal_map).float())

        # Step 1: Find minimum number of points across all point clouds
        min_N = min(len(p) for p in points)

        # Step 2: Randomly sample min_N points using the same indices for points and colors
        indices = [np.random.choice(len(p), min_N, replace=False) for p in points]
        points_sampled = [p[idx] for p, idx in zip(points, indices)]
        colors_sampled = [c[idx] for c, idx in zip(colors, indices)]

        # Convert to tensors efficiently
        pixels = torch.stack(images).float() if images else torch.empty(0, 0, 0, 3)
        masks_tensor = torch.stack(masks).float() if masks else torch.empty(0, 0, 0)
        if images:
            pixels = pixels * masks_tensor.unsqueeze(-1)

        result = {
            'camera_id': torch.arange(len(images), dtype=torch.long),
            'robot2camera': torch.from_numpy(self.robot2cameras[:len(images)]).float(),
            'world2robot_gps': torch.from_numpy(world2robot).float(),
            'Ks': torch.from_numpy(self.Ks[:len(images)]).float(),
            'lidar': torch.from_numpy(lidar_points).float(),
            'lidar_colors': torch.from_numpy(lidar_colors).float(),
            'lidar_normals': torch.from_numpy(lidar_normals).float(),
            'lidar_depth': torch.stack(lidar_depths).float() if lidar_depths else torch.empty(0, 0, 0, 1),
            'lidar_normals_projected': torch.stack(lidar_normals_projected).float() if lidar_normals_projected else torch.empty(0, 0, 0, 3),
            'points': torch.from_numpy(np.stack(points_sampled, axis=0)).float(),
            'colors': torch.from_numpy(np.stack(colors_sampled, axis=0)).float(),
            'image': pixels,
            'depth': torch.stack(depths).float() if depths else torch.empty(0, 0, 0, 1),
            'mask': masks_tensor,
            'semantic': torch.zeros_like(pixels) if images else torch.empty(0, 0, 0, 3),
            'ts': torch.tensor([ts], dtype=torch.long),
        }

        # Clean up temporary variables
        del lidar_points
        del lidar_normals
        del lidar_colors
        del lidar_depths
        del lidar_normals_projected
        del points_sampled
        del colors_sampled
        del images
        del depths
        del masks
        del points
        del colors
        del pixels
        del masks_tensor
        
        return result


# Keep the main block unchanged for compatibility
if __name__ == "__main__":
    import yaml
    import time
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    config_path = "/agrigs_slam/config/default.yaml"
    try:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
            data_configs = config_dict["agrigs_slam"]["dataloader"]
    except Exception as e:
        raise(f"Config load error ({e}), using dummy config.")

    # Dataset loader
    print(f"Initializing DataloaderAgriGS with path: {data_configs['path']}")
    dataset = DataloaderAgriGS(data_configs, mode=DataloaderAgriGS.Mode.TRAINING)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        exit(1)

    # Merge all point clouds
    merged_points = []
    merged_colors = []
    robot_positions = []

    start = 1
    step = 1
    for i in range(start, start + step):
        start_time = time.time()
        data_item = dataset[i]
        end_time = time.time()
        frequency = 1.0 / (end_time - start_time)
        print(f"Processing item {i} at frequency: {frequency:.2f} Hz")
        lidar = data_item['lidar'].cpu().numpy()  # Points in robot frame
        lidar_colors = data_item['lidar_colors'].cpu().numpy()  # Colors for lidar points
        world2robot = data_item['world2robot_gps'].cpu().numpy()  # 4x4 transformation matrix
        lidar_depths = data_item['lidar_depth'].cpu().numpy()  # Precomputed lidar depths
        lidar_normals_projected = data_item['lidar_normals_projected'].cpu().numpy()  # Precomputed lidar normals

        # Transform lidar points from robot frame to world frame
        merged_points.append(lidar)
        merged_colors.append(lidar_colors)

        # Display RGB images with precomputed lidar depth maps and normals
        images = data_item['image'].cpu().numpy()
        
        if len(images) > 0:
            fig, axes = plt.subplots(2, len(images), figsize=(15, 10))  # 2 rows for depth and normals

            if len(images) == 1:
                axes = axes.reshape(2, 1)  # Ensure consistent indexing

            for cam_idx in range(len(images)):
                img = images[cam_idx]
                lidar_depth = lidar_depths[cam_idx].squeeze()
                lidar_normal = lidar_normals_projected[cam_idx]  # Shape: (1200, 1920, 3)

                # Top row: RGB image with depth overlay
                ax_depth = axes[0, cam_idx] if len(images) > 1 else axes[0, 0]
                ax_depth.imshow(img)

                # Extract valid pixels (non-zero depth values)
                valid_mask = lidar_depth > 0
                if np.any(valid_mask):
                    y_coords, x_coords = np.where(valid_mask)
                    valid_depths = lidar_depth[valid_mask]

                    # Normalize depth values for colormap
                    if len(valid_depths) > 0:
                        depth_min, depth_max = valid_depths.min(), valid_depths.max()
                        if depth_max > depth_min:
                            normalized_depths = (valid_depths - depth_min) / (depth_max - depth_min)
                        else:
                            normalized_depths = np.zeros_like(valid_depths)

                        # Apply rainbow colormap
                        rainbow_colors = cm.rainbow(normalized_depths)

                        # Overlay valid pixels with rainbow colors
                        ax_depth.scatter(x_coords, y_coords, c=rainbow_colors, s=1, alpha=0.8)

                ax_depth.set_title(f'Camera {cam_idx} RGB + LiDAR Depth - Frame {i}')
                ax_depth.axis('off')

                # Bottom row: RGB image with normal vectors overlay
                ax_normal = axes[1, cam_idx] if len(images) > 1 else axes[1, 0]
                ax_normal.imshow(img)

                # Extract valid normal pixels (non-zero normal vectors)
                normal_magnitude = np.linalg.norm(lidar_normal, axis=2)
                valid_normal_mask = normal_magnitude > 0
                
                if np.any(valid_normal_mask):
                    y_coords, x_coords = np.where(valid_normal_mask)
                    valid_normals = lidar_normal[valid_normal_mask]
                    
                    # Convert normals to RGB colors (normalize from [-1,1] to [0,1])
                    normal_colors = (valid_normals + 1) / 2
                    
                    # Overlay normal pixels with normal-based colors
                    ax_normal.scatter(x_coords, y_coords, c=normal_colors, s=1, alpha=0.8)

                ax_normal.set_title(f'Camera {cam_idx} RGB + LiDAR Normals - Frame {i}')
                ax_normal.axis('off')

            plt.tight_layout()
            plt.show()

    merged_points_np = np.vstack(merged_points)
    merged_colors_np = np.vstack(merged_colors)

    # Create point cloud from transformed lidar points with colors
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(merged_points_np)
    pc.colors = o3d.utility.Vector3dVector(merged_colors_np)

    # Create robot spheres at each robot position
    robot_spheres = []
    for pos in robot_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.7)
        sphere.paint_uniform_color([1, 0, 0])  # Red color for robot
        sphere.translate(pos)
        robot_spheres.append(sphere)

    # Add the world origin as a blue marker and its own axes
    origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.7)
    origin_sphere.paint_uniform_color([0, 0, 1])  # Blue color for world origin
    origin_sphere.translate([0, 0, 0])

    origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    # Visualize point cloud and robot positions in world frame
    # o3d.visualization.draw_geometries(
    #     [pc, origin_sphere, origin_axes] + robot_spheres,
    #     window_name="Lidar Point Cloud and Robot Positions in World Frame"
    # )

# if __name__ == "__main__":
#     import yaml
#     import time
#     import matplotlib.pyplot as plt
#     import matplotlib.cm as cm
    
#     config_path = "/agrigs_slam/config/default.yaml"
#     try:
#         with open(config_path, "r") as file:
#             config_dict = yaml.safe_load(file)
#             data_configs = config_dict["agrigs_slam"]["dataloader"]
#     except Exception as e:
#         raise(f"Config load error ({e}), using dummy config.")

#     # Dataset loader
#     print(f"Initializing DataloaderAgriGS with path: {data_configs['path']}")
#     dataset = DataloaderAgriGS(data_configs, mode=DataloaderAgriGS.Mode.TRAINING)
#     if len(dataset) == 0:
#         print("Dataset is empty. Exiting.")
#         exit(1)

#     # Merge all point clouds
#     merged_points = []
#     merged_colors = []
#     robot_positions = []

#     start = 200
#     step = 1
#     for i in range(start, start + step):
#         start_time = time.time()
#         data_item = dataset[i]
#         end_time = time.time()
#         frequency = 1.0 / (end_time - start_time)
#         print(f"Processing item {i} at frequency: {frequency:.2f} Hz")
#         lidar = data_item['lidar'].cpu().numpy()  # Points in robot frame
#         lidar_colors = data_item['lidar_colors'].cpu().numpy()  # Colors for lidar points
#         world2robot = data_item['world2robot_gps'].cpu().numpy()  # 4x4 transformation matrix
#         lidar_depths = data_item['lidar_depth'].cpu().numpy()  # Precomputed lidar depths

#         # Transform lidar points from robot frame to world frame
#         merged_points.append(lidar)
#         merged_colors.append(lidar_colors)

#         # Display RGB images with precomputed lidar depth maps
#         images = data_item['image'].cpu().numpy()
        
#         if len(images) > 0:
#             fig, axes = plt.subplots(1, len(images), figsize=(15, 5))  # Only 1 row

#             if len(images) == 1:
#                 axes = np.array([axes])  # Ensure consistent indexing

#             for cam_idx in range(len(images)):
#                 img = images[cam_idx]
#                 lidar_depth = lidar_depths[cam_idx].squeeze()

#                 # Display RGB image
#                 ax = axes[cam_idx] if len(images) > 1 else axes[0]
#                 ax.imshow(img)

#                 # Extract valid pixels (non-zero depth values)
#                 valid_mask = lidar_depth > 0
#                 if np.any(valid_mask):
#                     y_coords, x_coords = np.where(valid_mask)
#                     valid_depths = lidar_depth[valid_mask]

#                     # Normalize depth values for colormap
#                     if len(valid_depths) > 0:
#                         depth_min, depth_max = valid_depths.min(), valid_depths.max()
#                         if depth_max > depth_min:
#                             normalized_depths = (valid_depths - depth_min) / (depth_max - depth_min)
#                         else:
#                             normalized_depths = np.zeros_like(valid_depths)

#                         # Apply rainbow colormap
#                         rainbow_colors = cm.rainbow(normalized_depths)

#                         # Overlay valid pixels with rainbow colors
#                         ax.scatter(x_coords, y_coords, c=rainbow_colors, s=1, alpha=0.8)

#                 ax.set_title(f'Camera {cam_idx} RGB + LiDAR Overlay - Frame {i}')
#                 ax.axis('off')

#             plt.tight_layout()
#             plt.show()

#     merged_points_np = np.vstack(merged_points)
#     merged_colors_np = np.vstack(merged_colors)

#     # Create point cloud from transformed lidar points with colors
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(merged_points_np)
#     pc.colors = o3d.utility.Vector3dVector(merged_colors_np)

#     # Create robot spheres at each robot position
#     robot_spheres = []
#     for pos in robot_positions:
#         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.7)
#         sphere.paint_uniform_color([1, 0, 0])  # Red color for robot
#         sphere.translate(pos)
#         robot_spheres.append(sphere)

#     # Add the world origin as a blue marker and its own axes
#     origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.7)
#     origin_sphere.paint_uniform_color([0, 0, 1])  # Blue color for world origin
#     origin_sphere.translate([0, 0, 0])

#     origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
