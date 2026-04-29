import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from keyframe import KeyframeAgriGS
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

class LossAgriGS:
    """
    Memory-optimized comprehensive loss manager for AgriGS (LiDAR Odometry and Tracking with Splats)
    Uses PyTorch3D chamfer_distance for geometric losses
    """
    
    def __init__(self, config: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Loss component flags
        self.enable_color_loss = self.config.get('enable_color_loss', True)
        self.enable_ssim_loss = self.config.get('enable_ssim_loss', True)
        self.enable_depth_loss = self.config.get('enable_depth_loss', False)
        self.enable_chamfer_loss = self.config.get('enable_chamfer_loss', False)
        self.enable_normal_loss = self.config.get('enable_normal_loss', False)
        self.enable_raydrop_loss = self.config.get('enable_raydrop_loss', False)
        self.enable_line_of_sight_loss = self.config.get('enable_line_of_sight_loss', False)
        

        # Loss weights (lambdas) - all read from config with sensible defaults
        self.l1_lambda = self.config.get('l1_lambda', 0.8)
        self.ssim_lambda = self.config.get('ssim_lambda', 0.2)
        self.depth_lambda = self.config.get('depth_lambda', 0.1)
        self.chamfer_lambda = self.config.get('chamfer_lambda', 0.1)
        self.normal_lambda = self.config.get('normal_lambda', 0.1)
        self.raydrop_lambda = self.config.get('raydrop_lambda', 0.01)
        self.los_lambda = self.config.get('los_lambda', 0.1)
        
        # Light enhancement loss weight
        # Light enhancement loss flag
        self.enable_light_loss = self.config.get('enable_light_loss', False)
        self.light_lambda = self.config.get('light_lambda', 0.1)
        
        # Raydrop loss parameters
        self.raydrop_distance_threshold = self.config.get('raydrop_distance_threshold', 0.1)
        
        # Robust kernel settings
        self.robust_kernel = self.config.get('robust_kernel', 'huber')
        self.huber_delta = self.config.get('huber_delta', 0.1)
        self.geman_epsilon = self.config.get('geman_epsilon', 0.1)

        # KDTree settings
        self.kdtree_radius = self.config.get('kdtree_radius', 0.2)

        # Chamfer distance settings
        self.chamfer_norm = self.config.get('chamfer_norm', 1)  # L1 norm by default
        self.chamfer_single_directional = self.config.get('chamfer_single_directional', False)
        self.chamfer_abs_cosine = self.config.get('chamfer_abs_cosine', True)
        self.chamfer_point_reduction = self.config.get('chamfer_point_reduction', 'mean')
        self.chamfer_batch_reduction = self.config.get('chamfer_batch_reduction', 'mean')
    
    def cleanup_memory(self):
        """Force garbage collection and clear GPU cache"""
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()
    
    def compute_raydrop_loss(self, render_depth, lidar_depth_image):
        """
        Simple raydrop loss with per-pixel target masks
        
        Args:
            render_depth: Rendered depth map [B, H, W, 1]
            render_alpha: Rendered alpha/opacity map [B, H, W, 1]  
            lidar_depth_image: LiDAR depth image [B, H, W, 1]
        """
        if not self.enable_raydrop_loss:
            return torch.tensor(0.0, device=self.device)
        
        # Squeeze channel dimension
        render_depth = render_depth.squeeze(-1)  # [B, H, W]
        lidar_depth_image = lidar_depth_image.squeeze(-1)  # [B, H, W]
        
        # Create simple per-pixel target mask
        # Raydrop where: low alpha OR depth mismatch OR missing lidar
        depth_diff = torch.abs(render_depth - lidar_depth_image)
        depth_mask = depth_diff > 0.2  # Significant depth difference
        missing_mask = lidar_depth_image < 0.01  # Missing lidar data
        
        # Combine masks - pixel is raydrop if ANY condition is true
        target_raydrop = (depth_mask | missing_mask).float()
        
        # Simple visibility model: just use depth ratio
        depth_ratio = render_depth / (lidar_depth_image + 1e-6)
        predicted_raydrop = torch.sigmoid(torch.abs(torch.log(depth_ratio + 1e-6)))
        
        # Only compute loss where we have valid data
        valid_mask = (lidar_depth_image > 0) & (render_depth > 0)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Simple binary cross entropy
        loss = F.binary_cross_entropy(
            predicted_raydrop[valid_mask],
            target_raydrop[valid_mask]
        )
        
        return loss

    # Light enhancement loss method (simplified to just tone mapping)
    def compute_light_loss(self, rendered_colors, target_colors):
        """
        Compute light enhancement loss using tone mapping
        
        Args:
            rendered_colors: Rendered colors from splats
            target_colors: Target ground truth colors
        """
        if not self.enable_light_loss:
            return torch.tensor(0.0, device=self.device)
        eta = 1e-4
        x = torch.clip(rendered_colors, min=eta, max=1-eta)
        y = torch.clip(target_colors, min=eta, max=1-eta)
        f = lambda x: 0.5 - torch.sin(torch.asin(1.0 - 2.0 * x) / 3.0)
        return torch.mean((f(x) - f(y)) ** 2)

    # Robust loss kernels
    def geman_mcclure_loss(self, residuals, epsilon=None):
        if epsilon is None:
            epsilon = self.geman_epsilon
        r_squared = residuals ** 2
        result = r_squared / (r_squared + epsilon ** 2)
        del r_squared
        return result

    def huber_loss(self, residuals, delta=None):
        if delta is None:
            delta = self.huber_delta
        abs_residuals = torch.abs(residuals)
        result = torch.where(
            abs_residuals <= delta,
            0.5 * residuals ** 2,
            delta * (abs_residuals - 0.5 * delta)
        )
        del abs_residuals
        return result
    
    def apply_robust_kernel(self, residuals):
        if self.robust_kernel == 'huber':
            return self.huber_loss(residuals)
        elif self.robust_kernel == 'geman_mcclure':
            return self.geman_mcclure_loss(residuals)
        else:
            return residuals ** 2
    
    # Individual loss components
    def compute_color_loss(self, colors, target_rgb):
        """Compute color loss using ALL splats (no subsampling)"""
        if not self.enable_color_loss:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        l1loss = F.l1_loss(colors, target_rgb)
        
        ssimloss = torch.tensor(0.0, device=self.device)
        if self.enable_ssim_loss:
            if colors.dim() == 3:
                colors_ssim = colors.permute(2, 0, 1).unsqueeze(0)
                target_ssim = target_rgb.permute(2, 0, 1).unsqueeze(0)
            else:
                colors_ssim = colors
                target_ssim = target_rgb
             
            ssimloss = 1.0 - fused_ssim(colors_ssim, target_ssim)
            
            # Clean up SSIM intermediate tensors
            if colors.dim() == 3:
                del colors_ssim, target_ssim

        return l1loss, ssimloss

    def compute_depth_loss(self, render_depth, lidar_projected_depth):
        """
        Compute depth loss for pose optimization using sparse LiDAR supervision with KL divergence.
        render_depth: dense depth from rendered pose
        lidar_projected_depth: sparse LiDAR depth projected to image
        """
        if not self.enable_depth_loss:
            return torch.tensor(0.0, device=self.device)
        
        # Create valid mask for pixels with LiDAR measurements
        valid_mask = (lidar_projected_depth > 0) & (render_depth > 0)
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Extract valid pixels for direct comparison
        render_valid = render_depth[valid_mask]
        lidar_valid = lidar_projected_depth[valid_mask]
        
        # Convert depths to probability distributions using softmax
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        render_probs = F.softmax(render_valid.unsqueeze(0), dim=1).squeeze(0) + eps
        lidar_probs = F.softmax(lidar_valid.unsqueeze(0), dim=1).squeeze(0) + eps
        
        # Compute KL divergence: KL(lidar || render)
        kl_loss = F.kl_div(torch.log(render_probs), lidar_probs, reduction='batchmean')
        
        return kl_loss
    
    def compute_chamfer_loss(self, render_points, render_normals, lidar_points, lidar_normals):
        """
        Compute Chamfer Distance loss using PyTorch3D's chamfer_distance
        Only uses LiDAR points that overlap with render points based on knn_points
        
        Args:
            render_points: Rendered/splat points [N, 3]
            render_normals: Rendered/splat normals [N, 3] (optional)
            lidar_points: LiDAR points [M, 3]
            lidar_normals: LiDAR normals [M, 3] (optional)
        
        Returns:
            tuple: (chamfer_loss, normal_loss) where normal_loss is 0 if normals not provided
        """
        # Reshape points to batch format for knn_points
        render_points_batch = render_points.unsqueeze(0)  # [1, N, 3]
        lidar_points_batch = lidar_points.unsqueeze(0)    # [1, M, 3]
        
        # Find overlapping LiDAR points using knn_points
        # Find nearest render point for each LiDAR point
        knn_result = knn_points(
            lidar_points_batch, 
            render_points_batch, 
            K=1,  # Only need closest point
            return_nn=False,
            return_sorted=False
        )
        
        # Extract distances - shape [1, M, 1]
        knn_distances = knn_result.dists.squeeze(0).squeeze(1)  # [M]
        
        # Create overlap mask: True where LiDAR point has a nearby render point
        overlap_mask = knn_distances < (self.kdtree_radius ** 2)  # knn_points returns squared distances
        
        # Filter LiDAR points and normals to only overlapping ones
        if overlap_mask.sum() == 0:
            # No overlapping points, return zero loss
            del render_points_batch, lidar_points_batch, knn_result, knn_distances, overlap_mask
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        
        # Extract overlapping LiDAR points
        overlapping_lidar_points = lidar_points[overlap_mask]  # [K, 3] where K <= M
        overlapping_lidar_points_batch = overlapping_lidar_points.unsqueeze(0)  # [1, K, 3]
        
        # Prepare normals if available
        render_normals_batch = None
        overlapping_lidar_normals_batch = None
        
        if render_normals is not None:
            render_normals_batch = render_normals.unsqueeze(0)  # [1, N, 3]
        if lidar_normals is not None:
            overlapping_lidar_normals = lidar_normals[overlap_mask]  # [K, 3]
            overlapping_lidar_normals_batch = overlapping_lidar_normals.unsqueeze(0)  # [1, K, 3]
        
        # Compute chamfer distance using only overlapping LiDAR points
        loss_chamfer, loss_normals = chamfer_distance(
            render_points_batch, 
            overlapping_lidar_points_batch,
            x_normals=render_normals_batch,
            y_normals=overlapping_lidar_normals_batch,
            batch_reduction=self.chamfer_batch_reduction,
            point_reduction=self.chamfer_point_reduction,
            norm=self.chamfer_norm,
            single_directional=self.chamfer_single_directional,
            abs_cosine=self.chamfer_abs_cosine
        )
        
        # Handle case where normals are not provided
        if loss_normals is None:
            loss_normals = torch.tensor(0.0, device=self.device)
        
        # Clean up intermediate tensors
        del render_points_batch, lidar_points_batch, knn_result, knn_distances, overlap_mask
        del overlapping_lidar_points, overlapping_lidar_points_batch
        if render_normals_batch is not None:
            del render_normals_batch
        if overlapping_lidar_normals_batch is not None:
            del overlapping_lidar_normals, overlapping_lidar_normals_batch
        
        return loss_chamfer, loss_normals

    def compute_loss(self, keyframe: KeyframeAgriGS):
        """
        Compute comprehensive loss using PyTorch3D chamfer distance
        
        Args:
            keyframe: KeyframeAgriGS object containing all necessary data
        """
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        
        # Extract data from keyframe
        target_rgb = keyframe.images
        colors = keyframe.render_colors
        depths = keyframe.render_depths
        render_points = keyframe.render_points
        render_normals = keyframe.render_normals
        render_opacities = keyframe.render_alphas
        lidar_depth = keyframe.lidar_depth_image
        lidar_pts = keyframe.lidar_pointcloud
        lidar_normals = keyframe.get_lidar_normals()

        # Compute and accumulate color loss (apply weights)
        if self.enable_color_loss:
            l1_loss, ssim_loss = self.compute_color_loss(colors, target_rgb)
            color_loss = self.l1_lambda * l1_loss + self.ssim_lambda * ssim_loss
            total_loss += color_loss
            
            # Store scalar values
            loss_dict['⬇ COLOR'] = color_loss.detach().item()

            # Clean up color loss tensors
            del l1_loss, ssim_loss, color_loss

        # Compute and accumulate light enhancement loss
        if self.enable_light_loss:
            light_loss = self.compute_light_loss(colors, target_rgb)
            weighted_light_loss = self.light_lambda * light_loss
            total_loss += weighted_light_loss
            loss_dict['⬇ LIGHT'] = light_loss.detach().item()
            del light_loss, weighted_light_loss

        # Compute and accumulate depth loss (apply weight)
        if self.enable_depth_loss:
            depth_loss = self.compute_depth_loss(depths, lidar_depth)
            weighted_depth_loss = self.depth_lambda * depth_loss
            total_loss += weighted_depth_loss
            loss_dict['⬇ DEPTH'] = depth_loss.detach().item()
            del depth_loss, weighted_depth_loss

        # Compute and accumulate line-of-sight loss
        if self.enable_line_of_sight_loss and render_opacities is not None:
            los_loss = self.los_loss(render_opacities, depths, lidar_depth)
            weighted_los_loss = self.los_lambda * los_loss
            total_loss += weighted_los_loss
            loss_dict['⬇ LOS'] = los_loss.detach().item()
            del los_loss, weighted_los_loss

        # Compute and accumulate raydrop loss
        if self.enable_raydrop_loss:
            raydrop_loss = self.compute_raydrop_loss(
                render_depth=depths,
                lidar_depth_image=lidar_depth
            )
            total_loss += self.raydrop_lambda * raydrop_loss
            loss_dict['⬇ RAYDROP'] = raydrop_loss.detach().item()
            del raydrop_loss

        # Compute and accumulate chamfer loss using PyTorch3D
        if self.enable_chamfer_loss and lidar_pts is not None and render_points is not None:
            # If normal loss is disabled, set normals to None
            use_render_normals = render_normals if self.enable_normal_loss else None
            use_lidar_normals = lidar_normals if self.enable_normal_loss else None
            
            chamfer_loss, normal_loss = self.compute_chamfer_loss(
            render_points, use_render_normals, lidar_pts, use_lidar_normals
            )
            weighted_chamfer_loss = self.chamfer_lambda * chamfer_loss
            total_loss += weighted_chamfer_loss
            loss_dict['⬇ CHAMFER'] = chamfer_loss.detach().item()
            
            # Add normal loss if enabled and available
            if self.enable_normal_loss and normal_loss.item() > 0:
                weighted_normal_loss = self.normal_lambda * normal_loss
                total_loss += weighted_normal_loss
                loss_dict['⬇ NORMAL'] = normal_loss.detach().item()
                del weighted_normal_loss
            
            del chamfer_loss, normal_loss, weighted_chamfer_loss, use_render_normals, use_lidar_normals
        
        # Compute normal loss separately if enabled but chamfer is disabled
        elif self.enable_normal_loss and not self.enable_chamfer_loss and lidar_pts is not None and render_points is not None and render_normals is not None and lidar_normals is not None:
            # Only compute normal loss without chamfer
            _, normal_loss = self.compute_chamfer_loss(
            render_points, render_normals, lidar_pts, lidar_normals
            )
            
            if normal_loss.item() > 0:
                weighted_normal_loss = self.normal_lambda * normal_loss
                total_loss += weighted_normal_loss
                loss_dict['⬇ NORMAL'] = normal_loss.detach().item()
                del weighted_normal_loss
            
            del normal_loss
        
        # Clean up extracted data tensors
        del target_rgb, colors, depths, render_points, render_normals, lidar_pts, lidar_normals
        if render_opacities is not None:
            del render_opacities
        
        # Aggressive final cleanup
        self.cleanup_memory()
        
        # Store total loss as scalar
        loss_dict['⬇ TOTAL'] = total_loss.detach().item()
        
        return total_loss, loss_dict

    def los_loss(self, render_alpha, render_depth, render_lidar_depth):
        """
        Compute line-of-sight loss for rendered Gaussians vs LiDAR depth.
        Optimized for sparse LiDAR data with softer constraints to avoid artifacts.

        Args:
            render_alpha (Tensor): Rendered alpha values, shape (B, H, W, 1)
            render_depth (Tensor): Rendered depth values, shape (B, H, W, 1)
            render_lidar_depth (Tensor): LiDAR depth values, shape (B, H, W, 1)

        Returns:
            Tensor: Scalar loss value averaged across valid LiDAR measurements
        """
        # Create valid LiDAR mask (non-zero depths)
        valid_lidar_mask = render_lidar_depth > 0
        
        if valid_lidar_mask.sum() == 0:
            return torch.tensor(0.0, device=render_alpha.device)
        
        # Use much larger and adaptive safety margins to avoid artifacts
        # Scale epsilon based on depth to be more permissive at greater distances
        epsilon = torch.clamp(0.25 * render_lidar_depth, min=0.5, max=2.0)

        # Only penalize severe violations (render depth much closer than LiDAR)
        violation_threshold = render_lidar_depth - 4.0 * epsilon
        
        # Use much softer sigmoid with wider transition zone
        depth_diff = render_depth - violation_threshold
        los_violation_soft = torch.sigmoid(-depth_diff * 1.0)  # Reduced steepness from 5.0 to 1.0
        
        # Apply spatial smoothing to reduce sharp boundaries
        kernel_size = 3
        padding = kernel_size // 2
        smoothed_violation = F.avg_pool2d(
            los_violation_soft.permute(0, 3, 1, 2), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        ).permute(0, 2, 3, 1)
        
        # Apply only where LiDAR is valid, with reduced alpha penalty
        masked_alpha = render_alpha * smoothed_violation * valid_lidar_mask.float() * 0.5  # Reduce impact by 50%
        
        # Sum violations per batch and normalize by number of valid LiDAR pixels
        violations_per_batch = masked_alpha.sum(dim=(1, 2, 3))  # Shape: (B,)
        valid_pixels_per_batch = valid_lidar_mask.sum(dim=(1, 2, 3)).float()  # Shape: (B,)
        
        # Normalize by valid LiDAR pixels to account for sparsity
        normalized_loss_per_batch = violations_per_batch / (valid_pixels_per_batch + 1e-8)
        
        return normalized_loss_per_batch.mean()
    
    def enable_mapping_mode(self):
        """Enable losses that primarily affect mapping (splat structure)"""
        self.enable_chamfer_loss = True
        self.enable_line_of_sight_loss = True
        
    def enable_full_mode(self):
        """Enable all losses for joint optimization"""
        self.enable_color_loss = True
        self.enable_depth_loss = True
        self.enable_chamfer_loss = True
        self.enable_raydrop_loss = True
        self.enable_line_of_sight_loss = True
        self.enable_light_loss = True

    def get_loss_weights_summary(self):
        """Return a summary of all loss weights for logging/debugging"""
        weights = {
            'l1_lambda': self.l1_lambda,
            'ssim_lambda': self.ssim_lambda,
            'depth_lambda': self.depth_lambda,
            'chamfer_lambda': self.chamfer_lambda,
            'raydrop_lambda': self.raydrop_lambda,
            'los_lambda': self.los_lambda,
            'light_lambda': self.light_lambda
        }
        
        return weights
