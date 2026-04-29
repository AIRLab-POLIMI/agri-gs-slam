from typing import Tuple, Optional, Dict
import torch

class KeyframeAgriGS:
    """
    Encapsulates a keyframe containing lidar pointcloud, poses, and associated Gaussian splats.
    """
    
    def __init__(self, 
        id: int,
        timestamp: float,
        lidar_pointcloud: torch.Tensor,
        lidar_colors: torch.Tensor,
        lidar_depth: torch.Tensor,
        world2robot_pose: torch.Tensor,
        world2cams_poses: Dict[str, torch.Tensor],
        points: torch.Tensor,
        colors: torch.Tensor,
        splat_indices: Optional[torch.Tensor] = None,
        lidar_normals: Optional[torch.Tensor] = None,
        camera_ids: Optional[torch.Tensor] = None,
        robot2camera: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        depths: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        semantic: Optional[torch.Tensor] = None):
        """
        Initialize a AgriGS keyframe.
        
        Args:
            id: Unique identifier for the keyframe
            timestamp: Timestamp of the keyframe
            lidar_pointcloud: [N, 3] tensor of lidar points in world coordinates
            world2robot_pose: [4, 4] transformation matrix from world to robot
            world2cams_poses: Dictionary mapping camera names to [4, 4] world2cam transforms
            points: [N, 3] tensor of 3D points
            colors: [N, 3] tensor of RGB colors for the points
            splat_indices: Optional [M,] tensor of indices referencing splats in the gaussian model
            lidar_normals: Optional [N, 3] tensor of normal vectors for lidar points
            camera_ids: Optional tensor of camera IDs
            robot2camera: Optional tensor of robot to camera transforms
            Ks: Optional tensor of camera intrinsics
            images: Optional tensor of camera images
            depths: Optional tensor of depth images
            masks: Optional tensor of masks
            semantic: Optional tensor of semantic data
        """
        self.id = id
        self.timestamp = timestamp
        self.trainable = True  # Indicates if this keyframe is trainable
        self.lidar_depth_image = lidar_depth.clone() if lidar_depth is not None else None
        self.lidar_pointcloud = lidar_pointcloud.clone()
        self.lidar_colors = lidar_colors.clone() if lidar_colors is not None else None
        self.world2robot_pose = world2robot_pose.clone()
        self.world2cams_poses = {k: v.clone() for k, v in world2cams_poses.items()}
        self.points = points.clone()
        self.colors = colors.clone()
        self.splat_indices = splat_indices.clone() if splat_indices is not None else None
        self.lidar_normals = lidar_normals.clone() if lidar_normals is not None else None
        self.splats = None
        self.cam2world_optimizable = {}  # Dictionary to hold optimizable camera poses
        
        # New fields for camera data
        self.camera_ids = camera_ids.clone() if camera_ids is not None else None
        self.robot2camera = robot2camera.clone() if robot2camera is not None else None
        self.Ks = Ks.clone() if Ks is not None else None
        self.images = images.clone() if images is not None else None
        self.depths = depths.clone() if depths is not None else None
        self.masks = masks.clone() if masks is not None else None
        self.semantic = semantic.clone() if semantic is not None else None
        
        # Rendered data attributes
        self.render_colors = None
        self.render_depths = None
        self.render_alphas = None
        self.render_points = None
        self.render_normals = None
        self.rasterization_info = None
        self.statistics_dict = {
            "ID": self.id,
            "N. OPT": 0,
        }

    def set_splats(self, splats: Dict[str, torch.nn.Parameter]):
        """
        Set the splats for this keyframe.
        
        Args:
            splats: Dictionary of splat names to torch.nn.Parameter objects
        """
        self.splats = splats
    
    def get_splats(self) -> Dict[str, torch.nn.Parameter]:
        """
        Return the splats associated with this keyframe.
        
        Returns:
            Dictionary of splat names to torch.nn.Parameter objects
        """
        return self.splats
    
    def clean_splats(self):
        """
        Remove all splats from this keyframe cleaning the gpu memory.
        """
        self.splats = None
        torch.cuda.empty_cache()

    def get_lidar_points(self) -> torch.Tensor:
        """Return the lidar pointcloud."""
        return self.lidar_pointcloud
    
    def get_lidar_normals(self) -> Optional[torch.Tensor]:
        """Return the lidar normals."""
        return self.lidar_normals
    
    def set_lidar_normals(self, normals: torch.Tensor):
        """Set the lidar normals for this keyframe."""
        self.lidar_normals = normals.clone()
    
    def get_robot_pose(self) -> torch.Tensor:
        """Return the world2robot transformation matrix."""
        return self.world2robot_pose
    
    def get_camera_poses(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of camera poses."""
        return self.world2cams_poses
    
    def get_camera_pose(self, camera_name: str) -> Optional[torch.Tensor]:
        """Return pose for a specific camera."""
        return self.world2cams_poses.get(camera_name)
    
    def get_points(self, lidar_mode: bool = False) -> torch.Tensor:
        """
        Return the 3D points. If lidar_mode is True, return lidar points where the colors are not black.
        
        Args:
            lidar_mode: Whether to filter points based on lidar colors
        
        Returns:
            Tensor of 3D points
        """
        if lidar_mode and self.lidar_colors is not None:
            mask = (self.lidar_colors != 0).any(dim=1)
            return self.lidar_pointcloud[mask]
        return self.points
    
    def get_colors(self, lidar_mode: bool = False) -> torch.Tensor:
        """
        Return the RGB colors. If lidar_mode is True, return lidar colors where they are not black.
        
        Args:
            lidar_mode: Whether to filter colors based on lidar colors
        
        Returns:
            Tensor of RGB colors
        """
        if lidar_mode and self.lidar_colors is not None:
            mask = (self.lidar_colors != 0).any(dim=1)
            return self.lidar_colors[mask]
        return self.colors
    
    def get_splat_indices(self) -> Optional[torch.Tensor]:
        """Return indices of associated splats."""
        return self.splat_indices
    
    def set_splat_indices(self, indices: torch.Tensor):
        """Set the splat indices for this keyframe."""
        self.splat_indices = indices.clone()
    
    def has_splats(self) -> bool:
        """Check if this keyframe has associated splats."""
        return self.splat_indices is not None and len(self.splat_indices) > 0
    
    def get_camera_data(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return all camera-related data."""
        return {
            'camera_ids': self.camera_ids,
            'robot2camera': self.robot2camera,
            'Ks': self.Ks,
            'images': self.images,
            'depths': self.depths,
            'masks': self.masks,
            'semantic': self.semantic
        }
    
    def offload_images(self):
        """
        Remove images, depth data, and rendered data from the keyframe to free memory.
        """
        # Clear the images and depths from the keyframe
        self.images = None
        self.depths = None
        self.lidar_depth_image = None
        self.lidar_colors = None
        self.masks = None
        self.semantic = None
        
        # Clear rendered data
        self.render_colors = None
        self.render_depths = None
        self.render_alphas = None
        self.rasterization_info = None
    
    def to_device(self, device: torch.device):
        """Move all tensors to specified device."""
        self.lidar_pointcloud = self.lidar_pointcloud.to(device)
        self.world2robot_pose = self.world2robot_pose.to(device)
        self.world2cams_poses = {k: v.to(device) for k, v in self.world2cams_poses.items()}
        self.points = self.points.to(device)
        self.colors = self.colors.to(device)
        if self.splat_indices is not None:
            self.splat_indices = self.splat_indices.to(device)
        if self.lidar_normals is not None:
            self.lidar_normals = self.lidar_normals.to(device)
        if self.lidar_depth_image is not None:
            self.lidar_depth_image = self.lidar_depth_image.to(device)
        
        # Move new camera data fields
        if self.camera_ids is not None:
            self.camera_ids = self.camera_ids.to(device)
        if self.robot2camera is not None:
            self.robot2camera = self.robot2camera.to(device)
        if self.Ks is not None:
            self.Ks = self.Ks.to(device)
        if self.images is not None:
            self.images = self.images.to(device)
        if self.depths is not None:
            self.depths = self.depths.to(device)
        if self.masks is not None:
            self.masks = self.masks.to(device)
        if self.semantic is not None:
            self.semantic = self.semantic.to(device)
        
        # Move rendered data
        if self.render_colors is not None:
            self.render_colors = self.render_colors.to(device)
        if self.render_depths is not None:
            self.render_depths = self.render_depths.to(device)
        if self.render_alphas is not None:
            self.render_alphas = self.render_alphas.to(device)
        
        return self
    
    def flush(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return points and colors, then remove them from the keyframe.
        Maintains the lidar pointcloud.
        
        Returns:
            Tuple of (points, colors) tensors before they are cleared
        """
        # Get copies of the data to return
        flushed_points = self.points.clone()
        flushed_colors = self.colors.clone()
        
        # Clear the points and colors from the keyframe
        # Create empty tensors with same device and dtype
        device = self.points.device
        dtype_points = self.points.dtype
        dtype_colors = self.colors.dtype
        
        self.points = torch.empty((0, 3), device=device, dtype=dtype_points)
        self.colors = torch.empty((0, 3), device=device, dtype=dtype_colors)
        
        # Also clear splat indices since they reference the points
        self.splat_indices = None
        
        return flushed_points, flushed_colors

    def clone(self) -> 'KeyframeAgriGS':
        """Create a deep copy of the keyframe."""
        return KeyframeAgriGS(
            timestamp=self.timestamp,
            lidar_pointcloud=self.lidar_pointcloud.clone(),
            world2robot_pose=self.world2robot_pose.clone(),
            world2cams_poses={k: v.clone() for k, v in self.world2cams_poses.items()},
            points=self.points.clone(),
            colors=self.colors.clone(),
            splat_indices=self.splat_indices.clone() if self.splat_indices is not None else None,
            lidar_normals=self.lidar_normals.clone() if self.lidar_normals is not None else None,
            camera_ids=self.camera_ids.clone() if self.camera_ids is not None else None,
            robot2camera=self.robot2camera.clone() if self.robot2camera is not None else None,
            Ks=self.Ks.clone() if self.Ks is not None else None,
            images=self.images.clone() if self.images is not None else None,
            depths=self.depths.clone() if self.depths is not None else None,
            masks=self.masks.clone() if self.masks is not None else None,
            semantic=self.semantic.clone() if self.semantic is not None else None
        )

    def get_rasterization_params(self) -> Dict:
        """
        Get all parameters needed for rasterization.
        
        Returns:
            Dictionary containing rasterization parameters
        """
        # Convert world2cam poses to camtoworld poses
        camtoworlds = {}
        for cam_name, world2cam in self.world2cams_poses.items():
            # Inverse of world2cam gives camtoworld
            camtoworlds[cam_name] = torch.inverse(world2cam)
        
        return {
            'splats': self.splats,
            'camtoworlds': camtoworlds,
            'Ks': self.Ks,
            'masks': self.masks
        }

    def set_image_dimensions(self, width: int, height: int):
        """
        Set the image dimensions for rasterization.
        
        Args:
            width: Image width
            height: Image height
        """
        self.width = width
        self.height = height

    def get_image_dimensions(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the image dimensions. If not explicitly set, tries to infer from images.
        
        Returns:
            Tuple of (width, height). Returns (None, None) if dimensions cannot be determined.
        """
        # First check if dimensions are explicitly set
        if hasattr(self, 'width') and hasattr(self, 'height'):
            return self.width, self.height
        
        # Try to infer from images tensor
        if self.images is not None and self.images.numel() > 0:
            # Assuming images shape is [C, H, W, 3] or [C, 3, H, W]
            if len(self.images.shape) == 4:
                if self.images.shape[-1] == 3:  # [C, H, W, 3]
                    self.height, self.width = self.images.shape[1], self.images.shape[2]
                elif self.images.shape[1] == 3:  # [C, 3, H, W]
                    self.height, self.width = self.images.shape[2], self.images.shape[3]
                else:
                    return None, None
                return self.width, self.height
        
        return None, None

    def get_camtoworld_poses(self) -> Dict[str, torch.Tensor]:
        """
        Get camera-to-world transformation matrices.
        
        Returns:
            Dictionary mapping camera names to camtoworld transforms
        """
        camtoworlds = {}
        with torch.no_grad():
            for cam_name, world2cam in self.world2cams_poses.items():
                camtoworlds[cam_name] = torch.inverse(world2cam)
        return camtoworlds

    def get_camtoworld_tensor(self) -> torch.Tensor:
        """
        Get camera-to-world poses as a stacked tensor.
        
        Returns:
            Tensor of shape [C, 4, 4] where C is number of cameras
        """
        if self.trainable:
            return torch.stack(list(self.cam2world_optimizable.values()))
        else:
             return torch.stack([torch.inverse(world2cam) for world2cam in self.world2cams_poses.values()])

    def get_intrinsics_tensor(self) -> torch.Tensor:
        """
        Get camera intrinsics as a tensor.
        
        Returns:
            Tensor of shape [C, 3, 3] where C is number of cameras
        """
        if self.Ks is None:
            raise ValueError("Camera intrinsics (Ks) not set")
        return self.Ks

    def set_rasterization_result(self, render_colors: Optional[torch.Tensor], render_depths: Optional[torch.Tensor], 
                                render_alphas: Optional[torch.Tensor], render_points: Optional[torch.Tensor],
                                render_normals: Optional[torch.Tensor], info: Dict):
        """
        Store the rasterization result in the keyframe.
        
        Args:
            render_colors: Optional tensor of rendered colors
            render_depths: Optional tensor of rendered depths
            render_alphas: Optional tensor of rendered alpha values
            render_points: Optional tensor of rendered 3D points
            render_normals: Optional tensor of rendered surface normals
            info: Dictionary containing additional rasterization information
        """
        self.render_colors = render_colors.clone() if render_colors is not None else None
        self.render_depths = render_depths.clone() if render_depths is not None else None
        self.render_alphas = render_alphas.clone() if render_alphas is not None else None
        self.render_points = render_points.clone() if render_points is not None else None
        self.render_normals = render_normals.clone() if render_normals is not None else None
        self.rasterization_info = info
