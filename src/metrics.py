import torch
import time
import json
from typing import Dict, Optional, Any
import os
import gc
from keyframe import KeyframeAgriGS
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points


class MetricsAgriGS:
    """Metrics calculator for AgriGS system to compute image quality metrics and Chamfer Distance."""
    
    def __init__(self):
        """
        Initialize the MetricsAgriGS calculator.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize image quality metrics (SSIM, PSNR, LPIPS)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
        
    def cleanup_gpu_memory(self):
        """Clean up GPU memory by clearing cache and running garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def compute_chamfer_distance(self, predicted_points: torch.Tensor, ground_truth_points: torch.Tensor) -> Dict[str, float]:
        """
        Compute Chamfer Distance between predicted and ground truth point clouds,
        using filtered knn points.
        
        Args:
            predicted_points: Predicted point cloud tensor of shape [B, N, 3]
            ground_truth_points: Ground truth point cloud tensor of shape [B, M, 3]
            
        Returns:
            Dictionary containing Chamfer Distance metrics
        """
        with torch.no_grad():
            # Ensure tensors are on the correct device
            predicted_points = predicted_points.to(self.device)
            ground_truth_points = ground_truth_points.to(self.device)
            
            # Reshape points to batch format for knn_points
            render_points_batch = predicted_points.unsqueeze(0)  # [1, N, 3]
            lidar_points_batch = ground_truth_points.unsqueeze(0)  # [1, M, 3]
            
            # Find nearest render point for each LiDAR point
            knn_result = knn_points(
                lidar_points_batch, 
                render_points_batch, 
                K=1,  # Only need closest point
                return_nn=True,
                return_sorted=False
            )
            
            # Extract distances and indices of nearest neighbors
            knn_distances = knn_result.dists.squeeze(0).squeeze(-1)  # [M]
            knn_indices = knn_result.idx.squeeze(0).squeeze(-1)  # [M]
            
            # Filter points based on a distance threshold (e.g., 0.1)
            distance_threshold = 0.1
            valid_mask = knn_distances < distance_threshold  # [M]
            
            # Filter ground truth and predicted points
            filtered_gt_points = ground_truth_points[valid_mask]  # [M_filtered, 3]
            valid_indices = knn_indices[valid_mask]  # [M_filtered]
            filtered_predicted_points = predicted_points[valid_indices]  # [M_filtered, 3]
            
            # Compute Chamfer Distance on filtered points
            chamfer_dist, _ = chamfer_distance(
                filtered_predicted_points.unsqueeze(0), 
                filtered_gt_points.unsqueeze(0),
                point_reduction="mean",
                batch_reduction="mean"
            )
            
            # Extract individual components
            metrics = {
                '⬇ CHAMFER': chamfer_dist.item(),
            }
            
            return metrics

    def compute(self, keyframe: KeyframeAgriGS, step: Optional[int] = None) -> Dict[str, float]:
        """
        Compute image quality metrics (PSNR, SSIM, LPIPS) and Chamfer Distance.
        
        Args:
            keyframe: KeyframeAgriGS object containing rendered colors, ground truth images,
                     and point clouds for Chamfer Distance computation
            step: Optional step number for logging

        Returns:
            Dictionary containing computed metrics
        """
        with torch.no_grad():
            # Clone tensors to avoid modifying original data
            render_colors = keyframe.render_colors.clone()
            images = keyframe.images.clone()
            
            # Ensure tensors are in the right format [B, 3, H, W]
            if render_colors.shape[-1] == 3:  # [B, H, W, 3]
                render_colors = render_colors.permute(0, 3, 1, 2)
            if images.shape[-1] == 3:  # [B, H, W, 3]
                images = images.permute(0, 3, 1, 2)

            # Clamp values to valid range
            render_colors.clamp_(0.0, 1.0)
            images.clamp_(0.0, 1.0)

            # Create mask based on 0.0 pixel values in render_colors
            # Mask is True where render_colors > 0.0 (valid pixels)
            mask = (render_colors > 0.0).all(dim=1, keepdim=True)  # [B, 1, H, W]
            
            # Apply mask to both render_colors and ground truth images
            render_colors *= mask
            images *= mask

            # Calculate image quality metrics
            psnr_value = self.psnr(render_colors, images).mean().item()
            ssim_value = self.ssim(render_colors, images).mean().item()
            lpips_value = self.lpips(render_colors, images).mean().item()

            metrics = {
                '⬆ PSNR ': psnr_value,
                '⬆ SSIM ': ssim_value,
                '⬇ LPIPS ': lpips_value
            }

            # Compute Chamfer Distance if point clouds are available
            # if hasattr(keyframe, 'lidar_pointcloud') and hasattr(keyframe, 'render_points'):
            #     chamfer_metrics = self.compute_chamfer_distance(
            #         keyframe.render_points, 
            #         keyframe.lidar_pointcloud
            #     )
            #     metrics.update(chamfer_metrics)

            # Store last computed metrics and keyframe for logging
            self.last_metrics = metrics
            self.last_keyframe = keyframe
            self.last_step = step

            # Clean up temporary tensors
            del render_colors, images, mask
            
            # Clean up GPU memory
            self.cleanup_gpu_memory()

            return metrics

    def save_metrics(self, metrics: Dict[str, Any], filepath: str):
        """
        Save metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            filepath: Path to save the metrics file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert any tensor values to float for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cleanup_gpu_memory()
