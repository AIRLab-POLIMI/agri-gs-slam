from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union, Optional
import gc
import math

import torch
from typing_extensions import Literal

from gsplat.strategy.base import Strategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split
from pytorch3d.ops import knn_points


@dataclass
class StrategyAgriGS(Strategy):
    """SLAM strategy optimized for maximum scene overfitting.

    Designed for perfect reconstruction at the cost of generalization:
    - Minimal pruning to retain maximum detail
    - Aggressive densification for fine features
    - Extended parameter ranges for complex geometry
    - Maximum Gaussian density allowance
    - Efficient pruning with sensitivity scores

    Args:
        prune_opa (float): Very low opacity threshold. Default is 0.01.
        grow_grad2d (float): Aggressive growth threshold. Default is 0.00015.
        max_scale_3d (float): Maximum scale. Default is 0.3.
        min_scale_3d (float): Minimum scale. Default is 0.00005.
        initial_scale_3d (float): Initial size. Default is 0.0003.
        aspect_ratio_threshold (float): Allow extreme elongation. Default is 50.0.
        refine_every (int): Refine every iteration. Default is 1.
        prune_every (int): Prune frequency. Default is 3.
        max_gaussians (int): Maximum Gaussian limit. Default is 2000000.
        split_threshold_multiplier (float): Multiplier for split thresholds. Default is 0.1.
        densification_boost (float): Boost for densification. Default is 3.0.
        reset_opa_every (int): Opacity reset frequency. Default is 5.
        reset_opa_threshold (float): Reset opacity threshold. Default is 0.05.
        reset_opa_value (float): Reset opacity value. Default is 0.8.
        learning_rate_boost (float): Learning rate boost. Default is 3.0.
        memory_efficient_mode (bool): Enable memory efficiency. Default is True.
        batch_size_limit (int): Batch size limit. Default is 50000.
        
        # Efficient pruning parameters
        enable_efficient_pruning (bool): Enable Speedy-Splat pruning. Default is True.
        soft_pruning_ratio (float): Ratio for soft pruning. Default is 0.2.
        hard_pruning_ratio (float): Ratio for hard pruning. Default is 0.1.
        soft_pruning_iterations (list): Iterations for soft pruning. Default is [1, 3, 5].
        hard_pruning_start (int): Start iteration for hard pruning. Default is 15.
        hard_pruning_interval (int): Interval for hard pruning. Default is 5.
        
        verbose (bool): Whether to print information. Default is False.
        key_for_gradient (str): Gradient key. Default is "means2d".
    """

    # Core parameters
    prune_opa: float = 0.01
    grow_grad2d: float = 0.00015
    max_scale_3d: float = 0.3
    min_scale_3d: float = 0.00005
    initial_scale_3d: float = 0.0003
    aspect_ratio_threshold: float = 50.0
    refine_every: int = 1
    prune_every: int = 3
    max_gaussians: int = 2000000
    
    # Densification parameters
    split_threshold_multiplier: float = 0.1
    densification_boost: float = 3.0
    
    # Opacity reset parameters
    reset_opa_every: int = 5
    reset_opa_threshold: float = 0.05
    reset_opa_value: float = 0.8
    reset_opa_enabled: bool = True
    
    # Learning and memory parameters
    learning_rate_boost: float = 3.0
    memory_efficient_mode: bool = True
    batch_size_limit: int = 50000
    
    # Efficient pruning parameters
    enable_efficient_pruning: bool = True
    soft_pruning_ratio: float = 0.2
    hard_pruning_ratio: float = 0.1
    soft_pruning_iterations: list = None
    hard_pruning_start: int = 15
    hard_pruning_interval: int = 5
    
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def __post_init__(self):
        if self.soft_pruning_iterations is None:
            self.soft_pruning_iterations = [1, 3, 5]

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize state for maximum overfitting."""
        state = {
            "grad2d": None,
            "count": None,
            "radii": None,
            "step": 0,
            "total_gaussians_created": 0,
            "total_gaussians_pruned": 0,
            "current_submap_step": 0,
            "opacity_resets": 0,
            "memory_pressure": 0,
            "last_densification_step": 0,
            
            # Efficient pruning state
            "pruning_scores": None,
            "pruning_score_count": None,
            "last_soft_pruning": -1,
            "last_hard_pruning": -1,
            "total_soft_pruned": 0,
            "total_hard_pruned": 0,
            "l1_loss_history": [],
        }
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Standard sanity check."""
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Pre-backward step - boost learning rates for overfitting."""
        # Detect new submap
        if step == 0 and state["step"] != 0:
            if self.verbose:
                print(f"New submap detected! Previous submap: {state['total_gaussians_created']} created, "
                      f"{state['total_gaussians_pruned']} pruned")
            state["current_submap_step"] = 0
        
        # Boost learning rates for faster overfitting
        if self.learning_rate_boost != 1.0:
            self._boost_learning_rates(optimizers, self.learning_rate_boost)
        
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()
        
        # Store rendered image gradient for pruning score computation
        if self.enable_efficient_pruning and "rendered_image" in info:
            info["rendered_image"].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False
    ):
        """GPU-friendly densification and pruning for overfitting."""
        state["step"] = step
        state["current_submap_step"] += 1

        self._update_state(params, state, info, packed=packed)
        
        # Update pruning scores if enabled
        if self.enable_efficient_pruning:
            self._update_pruning_scores(params, state, info, packed=packed)

        # Check memory pressure
        current_gaussians = len(params['means'])
        memory_pressure = current_gaussians / self.max_gaussians
        state["memory_pressure"] = memory_pressure

        # Check for soft pruning opportunities
        if self.enable_efficient_pruning and self._should_soft_prune(state):
            n_soft_pruned = self._soft_prune(params, optimizers, state)
            state["total_soft_pruned"] += n_soft_pruned
            state["last_soft_pruning"] = state["current_submap_step"]
            if self.verbose and n_soft_pruned > 0:
                print(f"Soft pruning at step {state['current_submap_step']}: "
                      f"Removed {n_soft_pruned} Gaussians. Total: {len(params['means'])}")

        # Check for hard pruning opportunities
        if self.enable_efficient_pruning and self._should_hard_prune(state):
            n_hard_pruned = self._hard_prune(params, optimizers, state)
            state["total_hard_pruned"] += n_hard_pruned
            state["last_hard_pruning"] = state["current_submap_step"]
            if self.verbose and n_hard_pruned > 0:
                print(f"Hard pruning at step {state['current_submap_step']}: "
                      f"Removed {n_hard_pruned} Gaussians. Total: {len(params['means'])}")
            # Statistical outlier removal pruning - called every step
            # try:
            #     n_pruned_outliers = self._statistical_outlier_pruning(params, optimizers, state)
            #     state["total_gaussians_pruned"] += n_pruned_outliers
            #     if self.verbose and n_pruned_outliers > 0:
            #         print(f"Step {state['current_submap_step']}: Statistical outlier pruning removed {n_pruned_outliers} Gaussians. "
            #             f"Total: {len(params['means'])}")
            # except ImportError:
            #     if self.verbose:
            #         print("Warning: pytorch3d not available for statistical outlier removal")


        # GPU-friendly densification with memory awareness
        can_densify = (
            state["current_submap_step"] % self.refine_every == 0 and 
            state["current_submap_step"] >= 1 and 
            len(params['means']) < self.max_gaussians * 0.9
        )
        
        if can_densify:
            max_new_gaussians = min(
                self.batch_size_limit,
                int(self.max_gaussians * 0.1),
                self.max_gaussians - len(params['means'])
            )
            
            n_created = self._gpu_friendly_densification(
                params, optimizers, state, step, max_new_gaussians
            )
            state["total_gaussians_created"] += n_created
            state["last_densification_step"] = state["current_submap_step"]
            
            if self.verbose and n_created > 0:
                print(f"Step {state['current_submap_step']}: Created {n_created} Gaussians. "
                      f"Total: {len(params['means'])} (Memory: {memory_pressure:.1%})")

        # Reset opacity before pruning to give Gaussians a second chance
        if (self.reset_opa_enabled and 
            state["current_submap_step"] % self.reset_opa_every == 0 and 
            state["current_submap_step"] > 0):
            n_reset = self._reset_dim_gaussians(params, optimizers, state, step)
            state["opacity_resets"] += n_reset
            if self.verbose and n_reset > 0:
                print(f"Step {state['current_submap_step']}: Reset opacity of {n_reset} dim Gaussians. "
                      f"Total resets: {state['opacity_resets']}")

        # More frequent pruning when memory pressure is high
        prune_frequency = self.prune_every
        if memory_pressure > 0.8:
            prune_frequency = max(2, prune_frequency // 2)
        
        if state["current_submap_step"] % prune_frequency == 0:
            n_pruned = self._memory_aware_pruning(params, optimizers, state, step, memory_pressure)
            state["total_gaussians_pruned"] += n_pruned
            if self.verbose and n_pruned > 0:
                print(f"Step {state['current_submap_step']}: Pruned {n_pruned} Gaussians. "
                      f"Total: {len(params['means'])} (Memory: {memory_pressure:.1%})")

        # Clean up GPU memory periodically
        if state["current_submap_step"] % 10 == 0:
            self._cleanup_gpu_memory()

    @torch.no_grad()
    def _statistical_outlier_pruning(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        k: int = 16,
        radius_thresh: float = 0.10,
    ) -> int:
        """
        Statistical outlier removal based on k-nearest neighbor distances using pytorch3d.
        
        Args:
            params: Gaussian parameters
            optimizers: Optimizers dict
            state: Current state
            k: Number of nearest neighbors to consider
            radius_thresh: Radius threshold for outlier detection
            
        Returns:
            Number of Gaussians pruned
        """
        try:
            n_gaussians = len(params['means'])
            
            # Need enough points for meaningful statistics
            if n_gaussians < k + 10:
                return 0
            
            means = params['means'].detach()  # (N, 3)
            
            # Add batch dimension as required by knn_points
            points = means.unsqueeze(0)  # (1, N, 3)
            
            # Find k-nearest neighbors
            knn_result = knn_points(points, points, K=k+1)
            
            # Calculate distances (excluding self-distance at index 0)
            dists = knn_result.dists[..., 1:]  # (1, N, k)
            mean_dists = dists.mean(dim=-1).squeeze(0)  # (N,)
            
            # Create mask for inliers (points within radius threshold)
            inlier_mask = mean_dists < radius_thresh
            outlier_mask = ~inlier_mask
            
            # Additional safety checks to avoid over-pruning
            n_outliers = outlier_mask.sum().item()
            max_prune = max(1, int(n_gaussians * 0.05))  # Don't prune more than 5% at once
            
            if n_outliers > max_prune:
                # Keep only the worst outliers
                outlier_distances = mean_dists[outlier_mask]
                _, worst_indices_in_outliers = torch.topk(outlier_distances, max_prune)
                
                # Map back to original indices
                outlier_positions = torch.where(outlier_mask)[0]
                worst_positions = outlier_positions[worst_indices_in_outliers]
                
                # Create new mask with only the worst outliers
                new_outlier_mask = torch.zeros_like(outlier_mask)
                new_outlier_mask[worst_positions] = True
                outlier_mask = new_outlier_mask
            
            n_pruned = outlier_mask.sum().item()
            
            # Perform the actual pruning
            if n_pruned > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=outlier_mask)
                
                # Update pruning scores if they exist
                if (self.enable_efficient_pruning and 
                    state.get("pruning_scores") is not None and 
                    len(state["pruning_scores"]) >= len(outlier_mask)):
                    remaining_indices = ~outlier_mask
                    state["pruning_scores"] = state["pruning_scores"][remaining_indices]
                    state["pruning_score_count"] = state["pruning_score_count"][remaining_indices]
            
            return n_pruned
            
        except ImportError:
            # pytorch3d not available
            raise ImportError("pytorch3d is required for statistical outlier removal")
        except Exception as e:
            if self.verbose:
                print(f"Error in statistical outlier pruning: {e}")
            self._cleanup_gpu_memory()
            return 0

    @torch.no_grad()
    def _update_pruning_scores(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Update efficient pruning scores based on Speedy-Splat method."""
        try:
            if "rendered_image" not in info or info["rendered_image"].grad is None:
                return
            
            rendered_grad = info["rendered_image"].grad
            n_gaussians = len(params["means"])
            
            # Initialize pruning score arrays if needed
            if state["pruning_scores"] is None:
                device = rendered_grad.device
                state["pruning_scores"] = torch.zeros(n_gaussians, device=device)
                state["pruning_score_count"] = torch.zeros(n_gaussians, device=device)
            
            # Expand arrays if new Gaussians were added
            if len(state["pruning_scores"]) < n_gaussians:
                device = rendered_grad.device
                old_size = len(state["pruning_scores"])
                
                new_scores = torch.zeros(n_gaussians, device=device)
                new_counts = torch.zeros(n_gaussians, device=device)
                
                new_scores[:old_size] = state["pruning_scores"]
                new_counts[:old_size] = state["pruning_score_count"]
                
                state["pruning_scores"] = new_scores
                state["pruning_score_count"] = new_counts
            
            # Compute simplified pruning scores
            rendered_grad_magnitude = rendered_grad.norm()
            opacities = torch.sigmoid(params["opacities"].flatten())
            current_scores = opacities * rendered_grad_magnitude
            
            # Accumulate scores
            state["pruning_scores"] += current_scores
            state["pruning_score_count"] += 1
            
            # Store L1 loss for pruning timing decisions
            if "l1_loss" in info:
                state["l1_loss_history"].append(info["l1_loss"].item())
                if len(state["l1_loss_history"]) > 1000:
                    state["l1_loss_history"].pop(0)
            
        except Exception as e:
            if self.verbose:
                print(f"Error updating pruning scores: {e}")

    def _should_soft_prune(self, state: Dict[str, Any]) -> bool:
        """Check if soft pruning should be performed."""
        if not self.enable_efficient_pruning:
            return False
        
        current_step = state["current_submap_step"]
        
        for prune_iter in self.soft_pruning_iterations:
            if current_step == prune_iter and state["last_soft_pruning"] < prune_iter:
                if len(state["l1_loss_history"]) > 10:
                    recent_loss = sum(state["l1_loss_history"][-10:]) / 10
                    if recent_loss < 0.1:
                        return True
                else:
                    return current_step >= 6000
        
        return False

    def _should_hard_prune(self, state: Dict[str, Any]) -> bool:
        """Check if hard pruning should be performed."""
        if not self.enable_efficient_pruning:
            return False
        
        current_step = state["current_submap_step"]
        
        if current_step < self.hard_pruning_start:
            return False
        
        if current_step - state["last_hard_pruning"] >= self.hard_pruning_interval:
            return True
        
        return False

    @torch.no_grad()
    def _soft_prune(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
        """Perform soft pruning based on sensitivity scores."""
        try:
            if state["pruning_scores"] is None or state["pruning_score_count"] is None:
                return 0
            
            score_counts = state["pruning_score_count"].clamp_min(1)
            avg_scores = state["pruning_scores"] / score_counts
            
            n_gaussians = len(avg_scores)
            n_to_prune = int(n_gaussians * self.soft_pruning_ratio)
            
            if n_to_prune == 0:
                return 0
            
            _, prune_indices = torch.topk(avg_scores, n_to_prune, largest=False)
            prune_mask = torch.zeros(n_gaussians, dtype=torch.bool, device=avg_scores.device)
            prune_mask[prune_indices] = True
            
            n_actual_prune = prune_mask.sum().item()
            if n_actual_prune > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)
                
                remaining_indices = ~prune_mask
                state["pruning_scores"] = state["pruning_scores"][remaining_indices]
                state["pruning_score_count"] = state["pruning_score_count"][remaining_indices]
            
            return n_actual_prune
            
        except Exception as e:
            if self.verbose:
                print(f"Error in soft pruning: {e}")
            return 0

    @torch.no_grad()
    def _hard_prune(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
        """Perform hard pruning based on sensitivity scores."""
        try:
            if state["pruning_scores"] is None or state["pruning_score_count"] is None:
                return 0
            
            score_counts = state["pruning_score_count"].clamp_min(1)
            avg_scores = state["pruning_scores"] / score_counts
            
            n_gaussians = len(avg_scores)
            n_to_prune = int(n_gaussians * self.hard_pruning_ratio)
            
            if n_to_prune == 0:
                return 0
            
            _, prune_indices = torch.topk(avg_scores, n_to_prune, largest=False)
            prune_mask = torch.zeros(n_gaussians, dtype=torch.bool, device=avg_scores.device)
            prune_mask[prune_indices] = True
            
            n_actual_prune = prune_mask.sum().item()
            if n_actual_prune > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)
                
                remaining_indices = ~prune_mask
                state["pruning_scores"] = state["pruning_scores"][remaining_indices]
                state["pruning_score_count"] = state["pruning_score_count"][remaining_indices]
            
            return n_actual_prune
            
        except Exception as e:
            if self.verbose:
                print(f"Error in hard pruning: {e}")
            return 0

    @torch.no_grad()
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent accumulation."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def _boost_learning_rates(self, optimizers: Dict[str, torch.optim.Optimizer], multiplier: float):
        """Boost learning rates for faster overfitting convergence."""
        for optimizer_name, optimizer in optimizers.items():
            if hasattr(optimizer, 'param_groups'):
                for param_group in optimizer.param_groups:
                    if 'original_lr' not in param_group:
                        param_group['original_lr'] = param_group['lr']
                    param_group['lr'] = param_group['original_lr'] * multiplier

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Enhanced state update with gradient tracking."""
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        try:
            grads = info[self.key_for_gradient].grad.clone()
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

            n_gaussian = len(list(params.values())[0])
            if state["grad2d"] is None or len(state["grad2d"]) < n_gaussian:
                device = grads.device
                new_grad2d = torch.zeros(n_gaussian, device=device)
                new_count = torch.zeros(n_gaussian, device=device)
                new_radii = torch.zeros(n_gaussian, device=device)
                
                if state["grad2d"] is not None:
                    old_size = len(state["grad2d"])
                    new_grad2d[:old_size] = state["grad2d"]
                    new_count[:old_size] = state["count"]
                    new_radii[:old_size] = state["radii"]
                    
                    del state["grad2d"], state["count"], state["radii"]
                
                state["grad2d"] = new_grad2d
                state["count"] = new_count
                state["radii"] = new_radii

            if packed:
                gs_ids = info["gaussian_ids"]
                radii = info["radii"].max(dim=-1).values
            else:
                sel = (info["radii"] > 0.0).all(dim=-1)
                gs_ids = torch.where(sel)[1]
                grads = grads[sel]
                radii = info["radii"][sel].max(dim=-1).values

            grad_norms = grads.norm(dim=-1)
            
            if self.densification_boost != 1.0:
                high_detail_mask = grad_norms > grad_norms.median()
                grad_norms[high_detail_mask] *= self.densification_boost
            
            state["grad2d"].index_add_(0, gs_ids, grad_norms)
            state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
            
            normalized_radii = radii / float(max(info["width"], info["height"]))
            state["radii"][gs_ids] = torch.maximum(state["radii"][gs_ids], normalized_radii)

            del grad_norms, normalized_radii
            if not packed:
                del sel
                
        except Exception as e:
            self._cleanup_gpu_memory()
            raise e

    @torch.no_grad()
    def _gpu_friendly_densification(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        max_new_gaussians: int,
    ) -> int:
        """GPU-friendly densification with memory limits."""
        try:
            count = state["count"]
            grads = state["grad2d"] / count.clamp_min(1)
            
            is_high_gradient = grads > self.grow_grad2d
            
            scales = torch.exp(params["scales"])
            max_scales = scales.max(dim=-1).values
            
            is_large = max_scales > self.initial_scale_3d * self.split_threshold_multiplier
            is_split = is_high_gradient & is_large
            
            is_small = max_scales <= self.initial_scale_3d * 2.0
            is_dupli = is_high_gradient & is_small
            
            n_split_candidates = is_split.sum().item()
            n_dupli_candidates = is_dupli.sum().item()
            total_candidates = n_split_candidates + n_dupli_candidates
            
            if total_candidates > max_new_gaussians:
                all_candidates = is_split | is_dupli
                candidate_gradients = grads[all_candidates]
                
                if total_candidates > 0 and len(candidate_gradients) > 0:
                    _, top_indices = torch.topk(
                        candidate_gradients, 
                        min(max_new_gaussians, len(candidate_gradients))
                    )
                    
                    selected_mask = torch.zeros_like(all_candidates)
                    candidate_positions = torch.where(all_candidates)[0]
                    selected_positions = candidate_positions[top_indices]
                    selected_mask[selected_positions] = True
                    
                    is_split = is_split & selected_mask
                    is_dupli = is_dupli & selected_mask & ~is_split
                    
                    del candidate_gradients, top_indices, selected_mask, candidate_positions, selected_positions
            
            n_split = is_split.sum().item()
            n_dupli = is_dupli.sum().item()
            
            if self.memory_efficient_mode and (n_split + n_dupli) > self.batch_size_limit:
                total_created = self._batched_densification(
                    params, optimizers, state, is_split, is_dupli
                )
            else:
                if n_split > 0:
                    split(params=params, optimizers=optimizers, state=state, mask=is_split)
                
                if n_dupli > 0:
                    duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
                
                total_created = n_split + n_dupli
            
            del grads, scales, max_scales, is_high_gradient, is_large, is_small, is_split, is_dupli
            
            return total_created
            
        except Exception as e:
            self._cleanup_gpu_memory()
            raise e

    @torch.no_grad()
    def _batched_densification(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        is_split: torch.Tensor,
        is_dupli: torch.Tensor,
    ) -> int:
        """Process densification in batches to manage memory."""
        try:
            total_created = 0
            
            split_indices = torch.where(is_split)[0]
            for i in range(0, len(split_indices), self.batch_size_limit):
                batch_indices = split_indices[i:i + self.batch_size_limit]
                batch_mask = torch.zeros_like(is_split)
                batch_mask[batch_indices] = True
                
                split(params=params, optimizers=optimizers, state=state, mask=batch_mask)
                total_created += len(batch_indices)
                
                del batch_mask
            
            dupli_indices = torch.where(is_dupli)[0]
            for i in range(0, len(dupli_indices), self.batch_size_limit):
                batch_indices = dupli_indices[i:i + self.batch_size_limit]
                batch_mask = torch.zeros_like(is_dupli)
                batch_mask[batch_indices] = True
                
                duplicate(params=params, optimizers=optimizers, state=state, mask=batch_mask)
                total_created += len(batch_indices)
                
                del batch_mask
            
            del split_indices, dupli_indices
            
            return total_created
            
        except Exception as e:
            self._cleanup_gpu_memory()
            raise e

    @torch.no_grad()
    def _memory_aware_pruning(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        memory_pressure: float,
    ) -> int:
        """Memory-aware pruning with adaptive thresholds."""
        try:
            opacities = torch.sigmoid(params["opacities"].flatten())
            scales = torch.exp(params["scales"])
            max_scales = scales.max(dim=-1).values
            
            adaptive_opa_threshold = self.prune_opa * (1 + memory_pressure * 10)
            
            is_invisible = opacities < adaptive_opa_threshold
            is_extremely_large = max_scales > self.max_scale_3d
            is_extreme_noise = max_scales < self.min_scale_3d
            
            is_prune = is_invisible | is_extremely_large | is_extreme_noise
            
            if memory_pressure > 0.8:
                is_large_dim = (opacities < 0.1) & (max_scales > self.initial_scale_3d * 10)
                is_prune = is_prune | is_large_dim
                del is_large_dim
            
            n_prune = is_prune.sum().item()
            if n_prune > 0:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

            del opacities, scales, max_scales, is_invisible, is_extremely_large, is_extreme_noise, is_prune

            return n_prune
            
        except Exception as e:
            self._cleanup_gpu_memory()
            raise e

    @torch.no_grad()
    def _reset_dim_gaussians(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Reset opacity of dim but geometrically valid Gaussians for overfitting."""
        try:
            opacities = torch.sigmoid(params["opacities"].flatten())
            scales = torch.exp(params["scales"])
            max_scales = scales.max(dim=-1).values
            min_scales = scales.min(dim=-1).values
            
            is_dim = (opacities < self.reset_opa_threshold) & (opacities > self.prune_opa)
            is_reasonable_size = (max_scales < self.max_scale_3d) & (max_scales > self.min_scale_3d)
            
            aspect_ratios = max_scales / (min_scales + 1e-8)
            is_not_extreme = aspect_ratios < (self.aspect_ratio_threshold * 0.5)
            
            has_gradient_info = state["grad2d"] is not None and state["count"] is not None
            is_high_gradient = torch.zeros_like(is_dim)
            
            if has_gradient_info:
                count = state["count"]
                grads = state["grad2d"] / count.clamp_min(1)
                gradient_threshold = self.grow_grad2d * 0.5
                is_high_gradient = grads > gradient_threshold
                del grads
            
            is_reset = is_dim & is_reasonable_size & (is_not_extreme | is_high_gradient)
            
            memory_pressure = state.get("memory_pressure", 0)
            if memory_pressure > 0.8:
                is_reset = is_reset & is_high_gradient
            
            n_reset = is_reset.sum().item()
            if n_reset > 0:
                reset_opa(
                    params=params, 
                    optimizers=optimizers, 
                    state=state, 
                    value=self.reset_opa_value
                )

            del opacities, scales, max_scales, min_scales, aspect_ratios
            del is_dim, is_reasonable_size, is_not_extreme, is_high_gradient, is_reset
            
            return n_reset
            
        except Exception as e:
            self._cleanup_gpu_memory()
            raise e
