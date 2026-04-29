import viser
from pathlib import Path
from typing import Literal, Tuple, Callable, Optional, List, Dict, Any
from nerfview import Viewer, RenderTabState
import argparse
import math
import os
import time
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from nerfview import CameraState, apply_float_colormap
import threading
import pickle
import json

# Add the KeyframeAgriGS import (assuming it's in a separate file)
from keyframe import KeyframeAgriGS


class AgriGSRenderTabState(RenderTabState):
    """State for AgriGS rendering tab with gsplat-specific parameters."""
    
    # non-controllable parameters
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # controllable parameters
    max_sh_degree: int = 5
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb", "depth(accumulated)", "depth(expected)", "alpha"
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    
    # online mode parameters
    online_mode: bool = False
    refresh_rate: float = 1.0
    last_update_time: float = 0.0


class ViewerAgriGS(Viewer):
    """
    Advanced viewer for gsplat with enhanced functionality and improved UI.
    """

    def __init__(self, config: Dict):
        """
        Initialize viewer from configuration dictionary.
        
        Args:
            config: Configuration dictionary containing viewer settings
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Online mode attributes
        self.online_mode = True
        self.refresh_rate = config.get("refresh_rate", 1.0)
        self.online_thread = None
        self.stop_online_thread = threading.Event()
        self.data_lock = threading.Lock()
        
        # Current gaussian data
        self.means = None
        self.quats = None
        self.scales = None
        self.opacities = None
        self.colors = None
        self.sh_degree = None
        
        # Camera frustums for visualization
        self.camera_frustums = []
        
        # Trajectory visualization
        self.keyframes = []  # Store keyframes for trajectory
        self.trajectory_lines = []  # Store trajectory line objects
        self.robot_trajectory_line = None  # Main robot trajectory
        self.loop_closure_lines = []  # Store loop closure connections
        self.robot2camera_lines = []  # Store robot-to-camera connections
        
        # Extract basic config
        port = config.get("port", 8080)
        output_dir = Path(config.get("output_dir", "results/"))
        title = "AgriGS-3DGS-SLAM Viewer"
        
        # Initialize server and parent
        server = viser.ViserServer(port=port, verbose=False)
        
        # Initialize gaussian data and create render function
        self._initialize_gaussian_data()
        render_fn = self._create_render_function()
        
        super().__init__(server, render_fn, output_dir, "rendering")
        server.gui.configure_theme(
            titlebar_content=None,
            control_layout="collapsible",
            dark_mode=True,
        )
        server.gui.set_panel_label(title)
        self._setup_viewer()
        
        # Start online mode if enabled
        if self.online_mode:
            self._start_online_mode()

    def display_keyframe(self, keyframe: KeyframeAgriGS) -> bool:
        """
        Display a KeyframeAgriGS by updating splat attributes and adding camera frustums.
        
        Args:
            keyframe: KeyframeAgriGS object to display
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use a timeout to prevent deadlocks
            if not self.data_lock.acquire(timeout=5.0):
                print("Warning: Could not acquire data lock, skipping keyframe display")
                return False
            
            try:
                # Clean up previous keyframe's splats to save GPU memory
                if hasattr(self, 'last_keyframe') and self.last_keyframe is not None:
                    try:
                        self.last_keyframe.clean_splats()
                    except Exception as e:
                        print(f"Warning: Could not clean previous keyframe splats: {e}")
                
                # Get splats from keyframe
                splats = keyframe.get_splats()
                
                if not splats:
                    print("Warning: Keyframe has no splats to display")
                    return False
                
                # Update gaussian data from keyframe splats
                if 'means' in splats:
                    self.means = splats['means'].to(self.device)
                if 'quats' in splats:
                    self.quats = F.normalize(splats['quats'], p=2, dim=-1).to(self.device)
                if 'scales' in splats:
                    # Check if scales are already exponential or log space
                    scales_tensor = splats['scales'].to(self.device)
                    if torch.all(scales_tensor > 0):
                        self.scales = scales_tensor  # Already exponential
                    else:
                        self.scales = torch.exp(scales_tensor)  # Convert from log space
                if 'opacities' in splats:
                    # Check if opacities are already sigmoid or logit space
                    opacities_tensor = splats['opacities'].to(self.device)
                    if torch.all((opacities_tensor >= 0) & (opacities_tensor <= 1)):
                        self.opacities = opacities_tensor  # Already sigmoid
                    else:
                        self.opacities = torch.sigmoid(opacities_tensor)  # Convert from logit space
                
                # Handle spherical harmonics colors
                if 'sh0' in splats and 'shN' in splats:
                    sh0 = splats['sh0'].to(self.device)
                    shN = splats['shN'].to(self.device)
                    self.colors = torch.cat([sh0, shN], dim=-2)
                    self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)
                elif 'colors' in splats:
                    self.colors = splats['colors'].to(self.device)
                    # Determine SH degree from color tensor shape
                    if self.colors.dim() > 2:
                        self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)
                    else:
                        self.sh_degree = None
                else:
                    # Fallback to default colors if no color information available
                    if self.means is not None:
                        self.colors = torch.ones(len(self.means), 1, 3, device=self.device) * 0.5
                        self.sh_degree = None
                
                # Add keyframe to trajectory and update connections
                self._add_keyframe_to_trajectory(keyframe)
                
                # Clear existing camera frustums
                # self._clear_camera_frustums()
                
                # Add camera frustums from keyframe
                self._add_camera_frustums(keyframe)
                
                # Update render function with new data
                self.render_fn = self._create_render_function()
                
                # Store reference to current keyframe for cleanup next time
                self.last_keyframe = keyframe
                
                # Force a rerender
                if hasattr(self, 'render_tab_state'):
                    self.render_tab_state.last_update_time = time.time()
                
                return True
                
            finally:
                self.data_lock.release()
                
        except Exception as e:
            print(f"Error displaying keyframe: {e}")
            return False

    def _add_keyframe_to_trajectory(self, keyframe: KeyframeAgriGS):
        """
        Add keyframe to trajectory and update visualization with loop edges and robot2camera connections.
        
        Args:
            keyframe: KeyframeAgriGS object to add to trajectory
        """
        try:
            # Add keyframe to list
            self.keyframes.append(keyframe)
            
            # Update robot trajectory
            self._update_robot_trajectory()
            
            # Add robot pose marker
            self._add_robot_pose_marker(keyframe)
            
            # Update loop closure edges
            #self._update_loop_closure_edges()
            
            # Update robot-to-camera connections
            #self._update_robot2camera_connections(keyframe)
            
            # Update trajectory statistics if GUI is available
            self._update_trajectory_stats()
            
        except Exception as e:
            print(f"Warning: Could not add keyframe to trajectory: {e}")

    def _update_loop_closure_edges(self):
        """Update loop closure edges between robot poses based on pose similarity."""
        if len(self.keyframes) < 2:
            return
            
        try:
            # Clear existing loop closure lines
            for line_name in self.loop_closure_lines:
                try:
                    self.server.scene[line_name].remove()
                except:
                    pass
            self.loop_closure_lines.clear()
            
            # Extract all robot positions
            robot_positions = []
            for kf in self.keyframes:
                try:
                    robot2world = kf.world2robot_pose
                    robot_pos = robot2world[:3, 3].cpu().numpy()
                    robot_positions.append(robot_pos)
                except Exception as e:
                    print(f"Warning: Could not extract robot position from keyframe: {e}")
                    continue
            
            if len(robot_positions) < 2:
                return
                
            robot_positions = np.array(robot_positions)
            
            # Create loop closure edges based on distance threshold
            loop_threshold = 2.0  # meters - adjust based on your scene scale
            edge_points = []
            edge_colors = []
            
            for i in range(len(robot_positions)):
                for j in range(i + 10, len(robot_positions)):  # Skip nearby keyframes to avoid clutter
                    distance = np.linalg.norm(robot_positions[i] - robot_positions[j])
                    if distance < loop_threshold:
                        # Add loop closure edge
                        edge_points.extend([robot_positions[i], robot_positions[j]])
                        edge_colors.extend([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])  # Magenta for loop closures
            
            # Create loop closure line segments if any exist
            if edge_points:
                edge_points = np.array(edge_points).reshape(-1, 3)
                edge_colors = np.array(edge_colors).reshape(-1, 3)
                
                loop_line_name = f"loop_closures_{int(time.time())}"
                self.server.scene.add_line_segments(
                    name=loop_line_name,
                    points=edge_points,
                    colors=edge_colors,
                    line_width=2.0
                )
                self.loop_closure_lines.append(loop_line_name)
                
        except Exception as e:
            print(f"Warning: Could not update loop closure edges: {e}")

    def _update_robot2camera_connections(self, keyframe: KeyframeAgriGS):
        """
        Update robot-to-camera connections using robot2camera transform.
        
        Args:
            keyframe: Current KeyframeAgriGS object
        """
        try:
            # Get robot position in world coordinates
            robot2world = torch.inverse(keyframe.world2robot_pose)
            robot_pos = robot2world[:3, 3].cpu().numpy()
            
            # Get robot2camera transform if available
            if not hasattr(keyframe, 'robot2camera_poses') or not keyframe.robot2camera_poses:
                return
                
            # Connect robot to each camera using robot2camera transform
            connection_points = []
            connection_colors = []
            
            for camera_name, robot2cam in keyframe.robot2camera_poses.items():
                try:
                    # Transform robot2camera to world coordinates
                    robot2cam_world = robot2world @ robot2cam.cpu()
                    camera_pos = robot2cam_world[:3, 3].numpy()
                    
                    # Add connection line from robot to camera
                    connection_points.extend([robot_pos, camera_pos])
                    connection_colors.extend([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])  # Cyan for robot2camera
                    
                except Exception as e:
                    print(f"Warning: Could not process robot2camera connection for {camera_name}: {e}")
                    continue
            
            # Create robot2camera connection lines if any exist
            if connection_points:
                connection_points = np.array(connection_points).reshape(-1, 3)
                connection_colors = np.array(connection_colors).reshape(-1, 3)
                
                connection_line_name = f"robot2camera_{keyframe.timestamp}"
                self.server.scene.add_line_segments(
                    name=connection_line_name,
                    points=connection_points,
                    colors=connection_colors,
                    line_width=1.5
                )
                self.robot2camera_lines.append(connection_line_name)
                
        except Exception as e:
            print(f"Warning: Could not update robot2camera connections: {e}")

    def _update_robot_trajectory(self):
        """Update the main robot trajectory line."""
        if len(self.keyframes) < 2:
            return
            
        try:
            # Extract robot positions from all keyframes
            positions = []
            for kf in self.keyframes:
                # world2robot_pose is [4,4], we want world position of robot
                # So we need robot2world, which is the inverse
                robot2world = torch.inverse(kf.world2robot_pose)
                robot_pos = robot2world[:3, 3].cpu().numpy()
                positions.append(robot_pos)
            
            positions = np.array(positions)
            
            # Remove existing trajectory line
            if self.robot_trajectory_line is not None:
                try:
                    self.server.scene[self.robot_trajectory_line].remove()
                except:
                    pass
            
            # Create trajectory line
            trajectory_name = f"robot_trajectory_{int(time.time())}"
            self.server.scene.add_line_segments(
                name=trajectory_name,
                points=positions,
                colors=np.array([[0.0, 1.0, 0.0]] * len(positions)),  # Green trajectory
                line_width=3.0
            )
            
            self.robot_trajectory_line = trajectory_name
            
        except Exception as e:
            print(f"Warning: Could not update robot trajectory: {e}")

    def _add_robot_pose_marker(self, keyframe: KeyframeAgriGS):
        """
        Add a pose marker for the robot at the keyframe location.
        
        Args:
            keyframe: KeyframeAgriGS object
        """
        try:
            # Get robot position and orientation in world coordinates
            robot2world = keyframe.world2robot_pose
            robot_pos = robot2world[:3, 3].cpu().numpy()
            robot_rot = robot2world[:3, :3].cpu().numpy()
            
            # Create pose marker name
            marker_name = f"robot_pose_{keyframe.timestamp}"
            
            # Add coordinate frame to show robot pose
            self.server.scene.add_frame(
                name=marker_name,
                wxyz=self._matrix_to_quaternion(robot_rot),
                position=robot_pos,
                axes_length=0.3,
                axes_radius=0.02
            )
            
            # Store marker name for cleanup
            if not hasattr(self, 'robot_pose_markers'):
                self.robot_pose_markers = []
            self.robot_pose_markers.append(marker_name)
            
        except Exception as e:
            print(f"Warning: Could not add robot pose marker: {e}")

    def _clear_trajectory(self):
        """Clear all trajectory visualization elements."""
        # Clear robot trajectory line
        if self.robot_trajectory_line is not None:
            try:
                self.server.scene[self.robot_trajectory_line].remove()
            except:
                pass
            self.robot_trajectory_line = None
        
        # Clear loop closure lines
        for line_name in self.loop_closure_lines:
            try:
                self.server.scene[line_name].remove()
            except:
                pass
        self.loop_closure_lines.clear()
        
        # Clear robot2camera connection lines
        for line_name in self.robot2camera_lines:
            try:
                self.server.scene[line_name].remove()
            except:
                pass
        self.robot2camera_lines.clear()
        
        # Clear robot pose markers
        if hasattr(self, 'robot_pose_markers'):
            for marker_name in self.robot_pose_markers:
                try:
                    self.server.scene[marker_name].remove()
                except:
                    pass
            self.robot_pose_markers.clear()
        
        # Clear keyframes list
        self.keyframes.clear()

    def _update_trajectory_stats(self):
        """Update trajectory statistics in GUI if available."""
        if hasattr(self, '_rendering_tab_handles') and "trajectory_length_text" in self._rendering_tab_handles:
            total_distance = self._calculate_trajectory_distance()
            self._rendering_tab_handles["trajectory_length_text"].value = f"{total_distance:.2f}m"
            
        if hasattr(self, '_rendering_tab_handles') and "keyframe_count_text" in self._rendering_tab_handles:
            self._rendering_tab_handles["keyframe_count_text"].value = str(len(self.keyframes))
            
        if hasattr(self, '_rendering_tab_handles') and "loop_count_text" in self._rendering_tab_handles:
            loop_count = len(self.loop_closure_lines)
            self._rendering_tab_handles["loop_count_text"].value = str(loop_count)

    def _calculate_trajectory_distance(self) -> float:
        """Calculate total distance traveled along trajectory."""
        if len(self.keyframes) < 2:
            return 0.0
            
        total_distance = 0.0
        prev_pos = None
        
        for kf in self.keyframes:
            robot2world = torch.inverse(kf.world2robot_pose)
            robot_pos = robot2world[:3, 3].cpu().numpy()
            
            if prev_pos is not None:
                distance = np.linalg.norm(robot_pos - prev_pos)
                total_distance += distance
            
            prev_pos = robot_pos
            
        return total_distance

    def _clear_camera_frustums(self):
        """Clear all existing camera frustums from the scene."""
        # Remove existing frustum objects from the server
        for frustum_name in self.camera_frustums:
            try:
                if hasattr(self.server.scene, frustum_name):
                    self.server.scene[frustum_name].remove()
            except:
                pass
        self.camera_frustums.clear()

    def _add_camera_frustums(self, keyframe):
        """
        Add camera frustums to the scene based on keyframe camera poses and intrinsics.
        
        Args:
            keyframe: KeyframeAgriGS object containing world2cams_poses dict, Ks tensor, and images tensor
        """
        # Extract camera-to-world poses
        camera_poses = keyframe.world2cams_poses  # Dict[str, torch.Tensor]

        # Build per-camera intrinsics map if provided
        K_map = {}
        if keyframe.Ks is not None:
            Ks_np = keyframe.Ks.cpu().numpy()
            for name, K in zip(camera_poses.keys(), Ks_np):
                K_map[name] = K

        # Build per-camera image size map if images tensor available
        size_map = {}
        if keyframe.images is not None:
            imgs_np = keyframe.images.cpu().numpy()
            for name, img in zip(camera_poses.keys(), imgs_np):
                h, w = img.shape[0], img.shape[1]
                size_map[name] = (w, h)

        # Define a palette of distinct colors
        palette = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 1.0, 0.0),  # green
            (0.0, 0.0, 1.0),  # blue
            (1.0, 1.0, 0.0),  # yellow
            (1.0, 0.0, 1.0),  # magenta
            (0.0, 1.0, 1.0),  # cyan
            (0.5, 0.5, 0.5),  # grey
        ]
        if not hasattr(self, 'camera_color_map'):
            self.camera_color_map = {}
        color_idx = 0

        # Iterate through each camera
        for camera_name, world2cam in camera_poses.items():
            try:
                cam2world = world2cam.detach().cpu().numpy()

                # Retrieve intrinsics and image size
                K = K_map.get(camera_name)
                img_w, img_h = size_map.get(camera_name, (640, 480))

                # Compute FOV if intrinsics available
                if K is not None and img_h > 0:
                    fy = K[1, 1]
                    fov_y = np.degrees(2 * np.arctan2(img_h / 2.0, fy))
                    aspect = float(img_w / img_h)
                else:
                    fov_y = 60.0
                    aspect = 4.0/3.0

                # Use a fixed, reasonable scale for frustum visualization
                scale = 0.2

                # Assign a unique color from palette
                if camera_name not in self.camera_color_map:
                    self.camera_color_map[camera_name] = palette[color_idx % len(palette)]
                    color_idx += 1
                color = self.camera_color_map[camera_name]

                # Name and add frustum
                frustum_name = f"camera_frustum_{camera_name}_{keyframe.timestamp}"
                self.server.scene.add_camera_frustum(
                    name=frustum_name,
                    fov=fov_y,
                    aspect=aspect,
                    scale=scale,
                    wxyz=self._matrix_to_quaternion(cam2world[:3, :3]),
                    position=cam2world[:3, 3],
                    color=color
                )
                self.camera_frustums.append(frustum_name)
            except Exception as e:
                print(f"Warning: Could not add frustum for camera {camera_name}: {e}")


    def _matrix_to_quaternion(self, rotation_matrix):
        """
        Convert rotation matrix to quaternion in wxyz format.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            numpy array: quaternion in wxyz format
        """
        # Simple conversion from rotation matrix to quaternion
        # This is a basic implementation - you might want to use a more robust version
        trace = np.trace(rotation_matrix)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s=4*qw 
            qw = 0.25 * s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                qx = 0.25 * s
                qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                qy = 0.25 * s
                qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                qz = 0.25 * s
        
        return np.array([qw, qx, qy, qz])

    def _initialize_gaussian_data(self):
        """Initialize gaussian data from config."""
        torch.manual_seed(42)
        
        ckpt_paths = self.config.get("ckpt")
        scene_grid = self.config.get("scene_grid", 1)
        
        if len(ckpt_paths) == 0:
            if self.online_mode:
                # In online mode without checkpoint, initialize with empty/placeholder data
                # Data will be set continuously by the online thread or display_keyframe
                self.means = torch.empty(0, 3, device=self.device)
                self.quats = torch.empty(0, 4, device=self.device)
                self.scales = torch.empty(0, 3, device=self.device)
                self.opacities = torch.empty(0, 1, device=self.device)
                self.colors = torch.empty(0, 1, 3, device=self.device)
                self.sh_degree = None
            else:
                # Fallback to test data for non-online mode
                (
                    self.means, self.quats, self.scales, self.opacities, colors,
                    viewmats, Ks, width, height,
                ) = load_test_data(device=self.device, scene_grid=scene_grid)
                
                self.means.requires_grad = True
                self.quats.requires_grad = True
                self.scales.requires_grad = True
                self.opacities.requires_grad = True
                colors.requires_grad = True
                self.colors = colors
                self.sh_degree = None
        else:
            # Load from checkpoint
            self.means, self.quats, self.scales, self.opacities, sh0, shN = load_checkpoint_data(ckpt_paths, self.device)
            self.colors = torch.cat([sh0, shN], dim=-2)
            self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)

    def set_map(self, map_data: Dict) -> bool:
        """Set gaussian data from map dictionary."""
        if not map_data:
            return False
            
        try:
            return self._update_gaussian_data_from_map(map_data)
            
        except Exception as e:
            return False

    def _update_gaussian_data_from_map(self, map_data: Dict) -> bool:
        """Update gaussian data from map dictionary."""
        try:
            # Use timeout to prevent deadlocks
            if not self.data_lock.acquire(timeout=5.0):
                print("Warning: Could not acquire data lock for map update")
                return False
            
            try:
                # Handle different map structures
                if 'splats' in map_data:
                    splats = map_data['splats']
                    self.means = splats['means'].to(self.device)
                    self.quats = F.normalize(splats['quats'], p=2, dim=-1).to(self.device)
                    self.scales = torch.exp(splats['scales']).to(self.device)
                    self.opacities = torch.sigmoid(splats['opacities']).to(self.device)
                    sh0 = splats['sh0'].to(self.device)
                    shN = splats['shN'].to(self.device)
                    self.colors = torch.cat([sh0, shN], dim=-2)
                    self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)
                else:
                    # Direct structure
                    self.means = map_data['means'].to(self.device)
                    self.quats = F.normalize(map_data['quats'], p=2, dim=-1).to(self.device)
                    self.scales = torch.exp(map_data['scales']).to(self.device)
                    self.opacities = torch.sigmoid(map_data['opacities']).to(self.device)
                    
                    if 'colors' in map_data:
                        self.colors = map_data['colors'].to(self.device)
                        self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1) if self.colors.dim() > 2 else None
                    elif 'sh0' in map_data and 'shN' in map_data:
                        sh0 = map_data['sh0'].to(self.device)
                        shN = map_data['shN'].to(self.device)
                        self.colors = torch.cat([sh0, shN], dim=-2)
                        self.sh_degree = int(math.sqrt(self.colors.shape[-2]) - 1)
                    else:
                        # Fallback to simple RGB
                        self.colors = torch.ones(len(self.means), 1, 3, device=self.device) * 0.5
                        self.sh_degree = None
                        
                return True
                
            finally:
                self.data_lock.release()
                
        except Exception as e:
            return False

    def _start_online_mode(self):
        """Start the online refreshing thread."""
        if not hasattr(self, 'map_path') or not self.map_path:
            return
            
        self.online_thread = threading.Thread(target=self._online_refresh_loop, daemon=True)
        self.online_thread.start()

    def _online_refresh_loop(self):
        """Main loop for online refreshing."""
        last_modified = 0
        
        while not self.stop_online_thread.is_set():
            try:
                if hasattr(self, 'map_path') and os.path.exists(self.map_path):
                    current_modified = os.path.getmtime(self.map_path)
                    
                    # Check if file has been modified
                    if current_modified > last_modified:
                        if self._load_map_data():
                            last_modified = current_modified
                            # Force a rerender
                            if hasattr(self, 'render_tab_state'):
                                self.render_tab_state.last_update_time = time.time()                            
                time.sleep(self.refresh_rate)
                
            except Exception as e:
                time.sleep(self.refresh_rate)

    def _create_render_function(self) -> Callable:
        """Create render function using current gaussian data."""
        return create_render_function(
            self.means, self.quats, self.scales, self.opacities, 
            self.colors, self.sh_degree, self.device, self.config, self.data_lock
        )

    def _setup_viewer(self) -> None:
        """Initialize viewer-specific settings."""
        pass  # Additional setup can be added here

    def _init_rendering_tab(self) -> None:
        """Initialize the rendering tab with AgriGS-specific state."""
        self.render_tab_state = AgriGSRenderTabState()
        self.render_tab_state.online_mode = self.online_mode
        self.render_tab_state.refresh_rate = self.refresh_rate
        
        # Apply config defaults to render state
        render_state_config = self.config.get("render_state", {})
        for key, value in render_state_config.items():
            if hasattr(self.render_tab_state, key):
                if key == "backgrounds" and isinstance(value, list):
                    setattr(self.render_tab_state, key, tuple(value))
                else:
                    setattr(self.render_tab_state, key, value)
        
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self) -> None:
        """Populate the rendering tab with controls."""
        server = self.server
        
        with self._rendering_folder:
            self._add_online_controls(server)
            self._add_trajectory_controls(server)
            self._add_gaussian_controls(server)
            self._add_rendering_controls(server)
            self._add_display_controls(server)
            
        super()._populate_rendering_tab()

    def _add_online_controls(self, server: viser.ViserServer) -> None:
        """Add online mode controls."""
        if self.online_mode:
            with server.gui.add_folder("Online Mode"):
                online_status = server.gui.add_text(
                    "Status",
                    initial_value="Online" if self.online_mode else "Offline",
                    disabled=True,
                    hint="Current online mode status.",
                )
                
                refresh_rate_slider = server.gui.add_number(
                    "Refresh Rate (s)",
                    initial_value=self.refresh_rate,
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    hint="How often to check for map updates.",
                )
                
                last_update_text = server.gui.add_text(
                    "Last Update",
                    initial_value="Never",
                    disabled=True,
                    hint="Time of last successful map update.",
                )
                
                @refresh_rate_slider.on_update
                def _(_) -> None:
                    self.refresh_rate = refresh_rate_slider.value
                    self.render_tab_state.refresh_rate = refresh_rate_slider.value
                
                self._rendering_tab_handles.update({
                    "online_status": online_status,
                    "refresh_rate_slider": refresh_rate_slider,
                    "last_update_text": last_update_text,
                })

    def _add_trajectory_controls(self, server: viser.ViserServer) -> None:
        """Add trajectory visualization controls."""
        with server.gui.add_folder("Trajectory"):
            keyframe_count_text = server.gui.add_text(
                "Keyframes",
                initial_value="0",
                disabled=True,
                hint="Number of keyframes in trajectory.",
            )
            
            trajectory_length_text = server.gui.add_text(
                "Distance",
                initial_value="0.00m",
                disabled=True,
                hint="Total distance traveled along trajectory.",
            )
            
            loop_count_text = server.gui.add_text(
                "Loop Closures",
                initial_value="0",
                disabled=True,
                hint="Number of detected loop closure connections.",
            )
            
            clear_trajectory_button = server.gui.add_button(
                "Clear Trajectory",
                hint="Clear all trajectory visualization elements.",
            )
            
            @clear_trajectory_button.on_click
            def _(_) -> None:
                self._clear_trajectory()
            
            self._rendering_tab_handles.update({
                "keyframe_count_text": keyframe_count_text,
                "trajectory_length_text": trajectory_length_text,
                "loop_count_text": loop_count_text,
                "clear_trajectory_button": clear_trajectory_button,
            })

    def _add_gaussian_controls(self, server: viser.ViserServer) -> None:
        """Add Gaussian splat-specific controls."""
        with server.gui.add_folder("Gaussian Splats"):
            total_gs_count_number = server.gui.add_number(
                "Total Splats",
                initial_value=self.render_tab_state.total_gs_count,
                disabled=True,
                hint="Total number of Gaussian splats in the scene.",
            )
            
            rendered_gs_count_number = server.gui.add_number(
                "Rendered Splats",
                initial_value=self.render_tab_state.rendered_gs_count,
                disabled=True,
                hint="Number of splats currently being rendered.",
            )

            max_sh_degree_number = server.gui.add_number(
                "Max SH Degree",
                initial_value=self.render_tab_state.max_sh_degree,
                min=0,
                max=5,
                step=1,
                hint="Maximum spherical harmonics degree for color representation.",
            )

            @max_sh_degree_number.on_update
            def _(_) -> None:
                self.render_tab_state.max_sh_degree = int(max_sh_degree_number.value)
                self.rerender(_)

            self._rendering_tab_handles.update({
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
                "max_sh_degree_number": max_sh_degree_number,
            })

    def _add_rendering_controls(self, server: viser.ViserServer) -> None:
        """Add rendering-specific controls."""
        with server.gui.add_folder("Rendering Parameters"):
            near_far_plane_vec2 = server.gui.add_vector2(
                "Near/Far Planes",
                initial_value=(
                    self.render_tab_state.near_plane,
                    self.render_tab_state.far_plane,
                ),
                min=(1e-3, 1e1),
                max=(1e1, 1e3),
                step=1e-3,
                hint="Near and far clipping planes for rendering.",
            )

            @near_far_plane_vec2.on_update
            def _(_) -> None:
                self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                self.rerender(_)

            radius_clip_slider = server.gui.add_number(
                "Radius Clip",
                initial_value=self.render_tab_state.radius_clip,
                min=0.0,
                max=100.0,
                step=1.0,
                hint="2D radius clipping threshold for splat culling.",
            )

            @radius_clip_slider.on_update
            def _(_) -> None:
                self.render_tab_state.radius_clip = radius_clip_slider.value
                self.rerender(_)

            eps2d_slider = server.gui.add_number(
                "2D Epsilon",
                initial_value=self.render_tab_state.eps2d,
                min=0.0,
                max=1.0,
                step=0.01,
                hint="Regularization epsilon for 2D covariance matrices.",
            )

            @eps2d_slider.on_update
            def _(_) -> None:
                self.render_tab_state.eps2d = eps2d_slider.value
                self.rerender(_)

            rasterize_mode_dropdown = server.gui.add_dropdown(
                "Anti-Aliasing",
                ("classic", "antialiased"),
                initial_value=self.render_tab_state.rasterize_mode,
                hint="Rasterization mode: classic or antialiased.",
            )

            @rasterize_mode_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                self.rerender(_)

            camera_model_dropdown = server.gui.add_dropdown(
                "Camera Model",
                ("pinhole", "ortho", "fisheye"),
                initial_value=self.render_tab_state.camera_model,
                hint="Camera projection model for rendering.",
            )

            @camera_model_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.camera_model = camera_model_dropdown.value
                self.rerender(_)

            self._rendering_tab_handles.update({
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
            })

    def _add_display_controls(self, server: viser.ViserServer) -> None:
        """Add display and visualization controls."""
        with server.gui.add_folder("Display Options"):
            backgrounds_slider = server.gui.add_rgb(
                "Background Color",
                initial_value=self.render_tab_state.backgrounds,
                hint="Background color for rendering.",
            )

            @backgrounds_slider.on_update
            def _(_) -> None:
                self.render_tab_state.backgrounds = backgrounds_slider.value
                self.rerender(_)

            render_mode_dropdown = server.gui.add_dropdown(
                "Render Mode",
                ("rgb", "depth(accumulated)", "depth(expected)", "alpha"),
                initial_value=self.render_tab_state.render_mode,
                hint="Select what to visualize: colors, depth, or transparency.",
            )

            normalize_nearfar_checkbox = server.gui.add_checkbox(
                "Normalize Depth",
                initial_value=self.render_tab_state.normalize_nearfar,
                disabled=True,
                hint="Normalize depth values using near/far planes.",
            )

            inverse_checkbox = server.gui.add_checkbox(
                "Inverse Depth",
                initial_value=self.render_tab_state.inverse,
                disabled=True,
                hint="Invert depth values for visualization.",
            )

            colormap_dropdown = server.gui.add_dropdown(
                "Colormap",
                ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                initial_value=self.render_tab_state.colormap,
                hint="Colormap for depth and alpha visualization.",
            )

            @render_mode_dropdown.on_update
            def _(_) -> None:
                is_depth_mode = "depth" in render_mode_dropdown.value
                normalize_nearfar_checkbox.disabled = not is_depth_mode
                inverse_checkbox.disabled = not is_depth_mode
                self.render_tab_state.render_mode = render_mode_dropdown.value
                self.rerender(_)

            @normalize_nearfar_checkbox.on_update
            def _(_) -> None:
                self.render_tab_state.normalize_nearfar = normalize_nearfar_checkbox.value
                self.rerender(_)

            @inverse_checkbox.on_update
            def _(_) -> None:
                self.render_tab_state.inverse = inverse_checkbox.value
                self.rerender(_)

            @colormap_dropdown.on_update
            def _(_) -> None:
                self.render_tab_state.colormap = colormap_dropdown.value
                self.rerender(_)

            self._rendering_tab_handles.update({
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
            })

    def _after_render(self) -> None:
        """Update GUI elements after rendering."""
        if "total_gs_count_number" in self._rendering_tab_handles:
            self._rendering_tab_handles["total_gs_count_number"].value = \
                self.render_tab_state.total_gs_count
        if "rendered_gs_count_number" in self._rendering_tab_handles:
            self._rendering_tab_handles["rendered_gs_count_number"].value = \
                self.render_tab_state.rendered_gs_count
                
        # Update online mode status
        if self.online_mode and "last_update_text" in self._rendering_tab_handles:
            if self.render_tab_state.last_update_time > 0:
                last_update_str = time.strftime(
                    "%H:%M:%S", time.localtime(self.render_tab_state.last_update_time)
                )
                self._rendering_tab_handles["last_update_text"].value = last_update_str

    def __del__(self):
        """Cleanup when viewer is destroyed."""
        if hasattr(self, 'stop_online_thread'):
            self.stop_online_thread.set()
        self._clear_camera_frustums()
        self._clear_trajectory()


def load_checkpoint_data(ckpt_paths: List[str], device: torch.device) -> tuple:
    """Load and concatenate data from multiple checkpoint files."""
    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    
    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
            
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        shN.append(ckpt["shN"])
    
    return (
        torch.cat(means, dim=0),
        torch.cat(quats, dim=0),
        torch.cat(scales, dim=0),
        torch.cat(opacities, dim=0),
        torch.cat(sh0, dim=0),
        torch.cat(shN, dim=0),
    )


def create_render_function(means, quats, scales, opacities, colors, sh_degree, device, config, data_lock=None):
    """Create the rendering function for the viewer."""
    
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, AgriGSRenderTabState)
        
        # Use data lock if in online mode
        if data_lock is not None:
            data_lock.acquire()
        
        try:
            # Check if we have valid data to render
            if means is None or len(means) == 0:
                # Return empty/black image if no data
                if render_tab_state.preview_render:
                    width = render_tab_state.render_width
                    height = render_tab_state.render_height
                else:
                    width = render_tab_state.viewer_width
                    height = render_tab_state.viewer_height
                return np.zeros((height, width, 3), dtype=np.float32)
            
            # Determine render dimensions
            if render_tab_state.preview_render:
                width = render_tab_state.render_width
                height = render_tab_state.render_height
            else:
                width = render_tab_state.viewer_width
                height = render_tab_state.viewer_height
                
            # Setup camera parameters
            c2w = torch.from_numpy(camera_state.c2w).float().to(device)
            K = torch.from_numpy(camera_state.get_K((width, height))).float().to(device)
            viewmat = c2w.inverse()

            # Render mode mapping
            RENDER_MODE_MAP = {
                "rgb": "RGB",
                "depth(accumulated)": "D",
                "depth(expected)": "ED",
                "alpha": "RGB",
            }

            # Perform rendering
            render_colors, render_alphas, info = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmat[None],
                K[None],
                width,
                height,
                sh_degree=(
                    min(render_tab_state.max_sh_degree, sh_degree)
                    if sh_degree is not None
                    else None
                ),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=torch.tensor([render_tab_state.backgrounds], device=device) / 255.0,
                render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
                packed=False,
                with_ut=config.get('with_ut', False),
                with_eval3d=config.get('with_eval3d', False),
            )
            
            # Update statistics
            render_tab_state.total_gs_count = len(means)
            render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

            # Process output based on render mode
            result = process_render_output(
                render_colors, render_alphas, render_tab_state
            )
            
        finally:
            if data_lock is not None:
                data_lock.release()
                
        return result
    
    return viewer_render_fn


def process_render_output(render_colors, render_alphas, render_tab_state):
    """Process rendering output based on the selected mode."""
    if render_tab_state.render_mode == "rgb":
        render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
        return render_colors.cpu().numpy()
        
    elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
        depth = render_colors[0, ..., 0:1]
        
        if render_tab_state.normalize_nearfar:
            near_plane = render_tab_state.near_plane
            far_plane = render_tab_state.far_plane
        else:
            near_plane = depth.min()
            far_plane = depth.max()
            
        depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
        depth_norm = torch.clip(depth_norm, 0, 1)
        
        if render_tab_state.inverse:
            depth_norm = 1 - depth_norm
            
        return apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        
    elif render_tab_state.render_mode == "alpha":
        alpha = render_alphas[0, ..., 0:1]
        return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()


def create_viewer_from_config(config: Dict[str, Any]) -> ViewerAgriGS:
    """
    Create and return a ViewerAgriGS instance from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with viewer settings
        
    Returns:
        ViewerAgriGS instance ready to run
        
    Example:
        config = {
            "port": 8080,
            "output_dir": "/path/to/output",
            "ckpt": ["/path/to/checkpoint.pt"],
            "online_mode": True,
            "map_path": "/path/to/map.pt",
            "refresh_rate": 2.0,
            "render_state": {
                "max_sh_degree": 5,
                "backgrounds": [0.0, 0.0, 0.0]
            }
        }
        viewer = create_viewer_from_config(config)
        viewer.run()
    """
    return ViewerAgriGS(config)


# Keep the original main function for backward compatibility
def main(local_rank: int, world_rank: int, world_size: int, args):
    """Main function for running the AgriGS viewer (backward compatibility)."""
    config = {
        "port": args.port,
        "output_dir": args.output_dir,
        "scene_grid": args.scene_grid,
        "ckpt": args.ckpt,
        "with_ut": args.with_ut,
        "with_eval3d": args.with_eval3d,
        "title": "AgriGS Gaussian Splat Viewer",
        "online_mode": args.online_mode,
        "map_path": args.map_path,
        "refresh_rate": args.refresh_rate,
    }
    
    viewer = ViewerAgriGS(config)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down viewer...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriGS Gaussian Splat Viewer")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/", 
        help="Output directory for results"
    )
    parser.add_argument(
        "--scene_grid", 
        type=int, 
        default=1, 
        help="Repeat scene in NxN grid (must be odd)"
    )
    parser.add_argument(
        "--ckpt", 
        type=str, 
        nargs="+", 
        default=["/agrigs_slam/pippo/ckpts/ckpt.pt"],
        help="Path(s) to checkpoint .pt files"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Port for viewer server"
    )
    parser.add_argument(
        "--with_ut", 
        action="store_true", 
        help="Use uncentered transform"
    )
    parser.add_argument(
        "--with_eval3d", 
        action="store_true", 
        help="Use eval 3D"
    )
    parser.add_argument(
        "--online_mode",
        action="store_true",
        help="Enable online mode for live map updates"
    )
    parser.add_argument(
        "--map_path",
        type=str,
        help="Path to map file for online mode (.pt, .pkl, or .json)"
    )
    parser.add_argument(
        "--refresh_rate",
        type=float,
        default=1.0,
        help="Refresh rate in seconds for online mode"
    )
    
    args = parser.parse_args()
    
    if args.scene_grid % 2 == 0:
        raise ValueError("scene_grid must be odd")
    
    cli(main, args, verbose=True)

    # import yaml
    # # Load configuration from YAML file
    # with open("/agrigs_slam/config/default.yaml", "r") as f:
    #     config = yaml.safe_load(f)["agrigs_slam"]["viewer"]
    # viewer = ViewerAgriGS(config)

    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\nShutting down viewer...")
    #     viewer.stop_online_thread.set()
    #     viewer.server.close()
    #     viewer.__del__()
    #     viewer = None  # Clean up viewer reference to allow garbage collection
    #     print("Viewer shutdown complete.")  