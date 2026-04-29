# Standard library imports
import os
from typing import Dict, Tuple, Any
import cv2
import math

# Third-party imports
import imageio
import numpy as np
import torch
from torch import Tensor
import matplotlib.cm as cm

# Local/project imports
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from keyframe import KeyframeAgriGS
from monitor import MonitorAgriGS
from scipy.ndimage import binary_dilation, distance_transform_edt


class SplatterAgriGS:
    """Engine for rasterizing Gaussian Splatting with AgriGS enhancements."""

    def __init__(self, config: Dict[str, Any], monitor: MonitorAgriGS) -> None:
        """
        Initialize the SplatterAgriGS engine.
        
        Args:
            config: Configuration dictionary containing optimization and rendering settings
            monitor: MonitorAgriGS instance for tracking and visualizing metrics
        """
        self.config = config
        self.monitor = monitor
        self._setup_directories()
        self._setup_training_parameters()
        self._setup_optimization_parameters()

    def _setup_directories(self) -> None:
        """Create necessary directories for outputs."""
        os.makedirs(self.config["results"], exist_ok=True)
        
        self.ckpt_dir = os.path.join(self.config["results"], "ckpts")
        self.stats_dir = os.path.join(self.config["results"], "stats")
        self.render_dir = os.path.join(self.config["results"], "renders")
        
        for dir_path in [self.ckpt_dir, self.stats_dir, self.render_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def _setup_training_parameters(self) -> None:
        """Setup training-related parameters."""
        self.image_every = self.config.get("image_every", 100)

    def _setup_optimization_parameters(self) -> None:
        """Setup optimization-related parameters."""
        self.packed = self.config.get("packed", False)
        self.sparse_grad = self.config.get("sparse_grad", False)
        self.antialiased = self.config.get("antialiased", False)
        self.visible_adam = self.config.get("optimization", {}).get("visible_adam", False)

    def save_canvas(self, iter, mode: str = "train") -> None:
        """
        Save a concatenated canvas of ground truth and rendered images.
        
        Args:
            keyframe: KeyframeAgriGS object containing ground truth and rendered images
            step: Current training step for naming the saved image
        """
        with torch.no_grad():
            if self.last_keyframe.id % self.image_every != 0:
                return
            
            # Clone tensors and immediately delete references
            pixels = self.last_keyframe.images.clone()
            colors = self.last_keyframe.render_colors.clone()

            # Convert to numpy for processing
            pixels_np = (pixels.detach().cpu().numpy() * 255).astype(np.uint8)
            colors_np = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # Delete tensor versions immediately
            del pixels, colors
            
            # Calculate font scale based on image height (10% of subimage height)
            font_scale = pixels_np.shape[1] * 0.1 / 30  # 30 is approximate height of default font
            thickness = max(1, int(font_scale * 4))
            
            # Calculate text position based on font size to avoid cutting
            text_y = max(int(font_scale * 30 + 10), 40)  # Ensure text is below top edge
            
            # Add text labels to each image
            for i in range(pixels_np.shape[0]):
                # Add "GT" label to ground truth images with black contour
                text = f"GT CAM {i + 1}"
                cv2.putText(pixels_np[i], text, (30, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 20, cv2.LINE_AA)  # Black contour
                cv2.putText(pixels_np[i], text, (30, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)  # White text
                
                # Add "RENDER" label to rendered images with black contour
                text = f"RENDER CAM {i + 1}"
                cv2.putText(colors_np[i], text, (30, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 20, cv2.LINE_AA)  # Black contour
                cv2.putText(colors_np[i], text, (30, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)  # White text

            # Concatenate ground truth and rendered images
            canvas = np.concatenate([pixels_np, colors_np], axis=2)
            
            # Delete numpy arrays after concatenation
            del pixels_np, colors_np
            
            canvas = canvas.reshape(-1, *canvas.shape[2:])
            
            # Reduce canvas size for storage efficiency
            canvas_resized = cv2.resize(
                canvas, 
                (canvas.shape[1] // 4, canvas.shape[0] // 4), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Delete original canvas after resizing
            del canvas
            
            # Save the image
            output_path = os.path.join(self.render_dir, f"{mode}_keyframe_{self.last_keyframe.id}_nopt_{iter}.png")
            imageio.imwrite(output_path, canvas_resized)

            # Delete final canvas
            del canvas_resized
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_splat_parameters(self, splats: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Extract and transform splat parameters from raw dictionary.
        
        Args:
            splats: Dictionary containing raw splat parameters
            
        Returns:
            Tuple of (means, quats, scales, opacities)
        """
        means = splats["means"]  # [N, 3] - 3D positions
        quats = splats["quats"]  # [N, 4] - Rotation quaternions
        
        # Process scales and opacities
        scales = torch.exp(splats["scales"])  # [N, 3] - Convert log scale to actual scale
        opacities = torch.sigmoid(splats["opacities"])  # [N,] - Convert logits to [0,1]
        
        return means, quats, scales, opacities

    def _compute_colors(self, splats: Dict[str, Tensor]) -> Tensor:
        """
        Compute colors from splat features and camera information.
        
        Args:
            splats: Dictionary containing splat parameters
            
        Returns:
            Computed colors tensor
        """
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
        return colors

    def _get_rasterization_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare keyword arguments for rasterization.
        
        Returns:
            Dictionary of rasterization parameters
        """
        raster_kwargs = {
            "packed": self.packed,
            "sparse_grad": self.sparse_grad,
            "rasterize_mode": "antialiased" if self.antialiased else "classic",
            "distributed": getattr(self, 'world_size', 1) > 1,
            "camera_model": "pinhole",
            "render_mode": "RGB+ED"
        }
        
        # Add absgrad if using DefaultStrategy
        if hasattr(self, 'strategy') and isinstance(self.strategy, DefaultStrategy):
            raster_kwargs["absgrad"] = self.strategy.absgrad
        else:
            raster_kwargs["absgrad"] = False
            
        # Add any additional kwargs
        raster_kwargs.update(kwargs)
        
        return raster_kwargs

    def rasterize(self, keyframe: KeyframeAgriGS) -> KeyframeAgriGS:
        """
        Rasterize Gaussian splats and update the given keyframe with the results.

        Args:
            keyframe: KeyframeAgriGS object containing splats and camera data.

        Returns:
            Updated KeyframeAgriGS object with rasterization results.
        """
        # Store the last keyframe that went into rasterize
        self.last_keyframe = keyframe
        
        # Extract rasterization parameters from the keyframe
        splats = keyframe.splats
        camtoworlds = keyframe.get_camtoworld_tensor()
        Ks = keyframe.get_intrinsics_tensor()
        width, height = keyframe.get_image_dimensions()
        
        # Check if we need gradients
        if not keyframe.trainable:
            # For non-trainable keyframes, use no_grad for efficiency
            with torch.no_grad():
                # Extract splat parameters
                means = splats["means"]  # [N, 3] - 3D positions of the splats
                quats = splats["quats"]  # [N, 4] - Rotation as quaternions
                scales = torch.exp(splats["scales"])  # [N, 3] - Convert log scale to actual scale
                opacities = torch.sigmoid(splats["opacities"])  # [N,] - Convert logits to [0,1] opacity

                # Compute colors
                colors = self._compute_colors(splats)

                # Prepare rasterization arguments
                raster_kwargs = self._get_rasterization_kwargs()

                # Perform core rasterization
                render_output, render_alphas, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=3,
                    near_plane=0.01,
                    far_plane=10.0,
                    **raster_kwargs,
                )

                # Clear intermediate variables immediately after rasterization
                del colors, raster_kwargs

                # Split renders into colors and depths if needed
                if render_output.shape[-1] == 4:
                    render_colors = render_output[..., 0:3].contiguous()
                    render_depths = render_output[..., 3:4].contiguous()
                    # Delete render_output after splitting to free memory
                    del render_output
                else:
                    render_colors = render_output
                    render_depths = None

                # For non-trainable keyframes, we only need render_colors and render_depths
                render_points = None
                render_normals = None
                
                # Clean up camera parameters
                del camtoworlds, Ks, scales, opacities, quats, means
        else:
            # For trainable keyframes, compute everything with gradients
            # Extract splat parameters
            means = splats["means"]  # [N, 3] - 3D positions of the splats
            quats = splats["quats"]  # [N, 4] - Rotation as quaternions
            scales = torch.exp(splats["scales"])  # [N, 3] - Convert log scale to actual scale
            opacities = torch.sigmoid(splats["opacities"])  # [N,] - Convert logits to [0,1] opacity

            # Compute colors
            colors = self._compute_colors(splats)

            # Prepare rasterization arguments
            raster_kwargs = self._get_rasterization_kwargs()

            # Perform core rasterization
            render_output, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=3,
                near_plane=0.01,
                far_plane=10.0,
                **raster_kwargs,
            )

            # Clear intermediate variables immediately after rasterization
            del colors, raster_kwargs

            # Split renders into colors and depths if needed
            if render_output.shape[-1] == 4:
                render_colors = render_output[..., 0:3].contiguous()
                render_depths = render_output[..., 3:4].contiguous()
                # Delete render_output after splitting to free memory
                del render_output
            else:
                render_colors = render_output
                render_depths = None

            # Assign splat means to render_points
            render_points = means
            
            # Compute rendered normals from quaternions and scales
            render_normals = None
            if render_points is not None:
                # Normalize quaternions to ensure they're unit quaternions
                quats_norm = torch.nn.functional.normalize(quats, dim=-1)
                
                # Extract quaternion components (assuming w, x, y, z format)
                w, x, y, z = quats_norm[:, 0], quats_norm[:, 1], quats_norm[:, 2], quats_norm[:, 3]
                
                # Convert quaternion to rotation matrix
                # This creates the rotation matrix from quaternion
                R = torch.stack([
                    torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
                    torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1),
                    torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1)
                ], dim=-2)  # [N, 3, 3]
                
                # Find the axis with minimum scale (the "thin" direction)
                min_scale_idx = torch.argmin(scales, dim=-1)  # [N]
                
                # Create one-hot encoding for the minimum scale axis
                basis_vectors = torch.eye(3, device=quats.device, dtype=quats.dtype)[min_scale_idx]  # [N, 3]
                
                # Transform the basis vector by the rotation matrix to get the normal
                # This gives us the direction of the smallest scale axis in world coordinates
                render_normals = torch.bmm(R, basis_vectors.unsqueeze(-1)).squeeze(-1)  # [N, 3]

            # Clean up camera parameters after computing points
            del camtoworlds, Ks, scales, opacities, quats

        # Convert lidar depth image to numpy and remove channel dimension
        # 1) pull your RGB frames (float32 in [0,1]) into numpy and clone them
        # images_float = keyframe.images.detach().cpu().numpy().copy()        # shape: (B, H, W, 3), values in [0,1]

        # # 2) make an 8‑bit version for display/overlay
        # images_uint8 = (images_float * 255).astype(np.uint8)               # now in 0–255
        # overlay_images = images_uint8.copy()                              # we'll paint into this

        # # 3) grab & squeeze the lidar depth: (B, H, W)
        # lidar_depth_np = keyframe.lidar_depth_image.detach().cpu().numpy().squeeze(-1)

        # # 4) mask of valid lidar
        # valid_mask = lidar_depth_np > 0

        # if np.any(valid_mask):
        #     # global min/max over *all* valid pixels
        #     dmin = lidar_depth_np[valid_mask].min()
        #     dmax = lidar_depth_np[valid_mask].max()
        #     drange = dmax - dmin + 1e-8

        #     # circular kernel for dilation
        #     radius = 4
        #     yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
        #     circular_kernel = (yy**2 + xx**2) <= radius**2

        #     for b in range(lidar_depth_np.shape[0]):
        #         bm = valid_mask[b]
        #         if not bm.any():
        #             continue  # keep original pixels as they are

        #         # normalized depth [0,1]
        #         nd = (lidar_depth_np[b] - dmin) / drange

        #         # colormap → uint8
        #         color_map = (cm.viridis(nd)[..., :3] * 255).astype(np.uint8)

        #         # dilate + distance transform
        #         dilated = binary_dilation(bm, structure=circular_kernel)
        #         dist, idxs = distance_transform_edt(~bm, return_indices=True)
        #         within = (dist <= radius) & dilated

        #         ys, xs = np.nonzero(within)
        #         ny = idxs[0, ys, xs]
        #         nx = idxs[1, ys, xs]

        #         # paint the overlay
        #         overlay_images[b, ys, xs] = color_map[ny, nx]

        # pack up your results
        image_data = {
            "image": np.stack([img.detach().cpu().numpy() for img in keyframe.images]),
            "rendered_image": np.stack([r.detach().cpu().numpy() for r in render_colors]),
            "rendered_depth": np.stack([d.detach().cpu().numpy().squeeze() for d in render_depths]),
        }
        self.monitor.update_dashboard_images(image_data)

        # del overlay_images, lidar_depth_np

        # Update keyframe with rasterization results
        keyframe.set_rasterization_result(render_colors, render_depths, render_alphas, render_points, render_normals, info)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return keyframe
