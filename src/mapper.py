import math
import torch
import torch.nn.functional as F
from simple_knn._C import distCUDA2
import gc
from keyframe import KeyframeAgriGS
from utils.utils import rgb_to_sh
from monitor import MonitorAgriGS

class MapperAgriGS:
    def __init__(self, config: dict, monitor: MonitorAgriGS = None):
        self.splats = torch.nn.ParameterDict()      # CPU storage (non-trainable)
        self.active_splats = torch.nn.ParameterDict()  # GPU storage (trainable)
        self.active_points = None                # Active points
        self.optimizers = {}                       # Persistent optimizers
        self.schedulers = {}                       # Learning rate schedulers
        self.param_state_cache = {}                # For preserving optimizer state
        self.last_timestamp = None                  # Last pose used for splat transfer
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = monitor

        # Hash voxels by combining coords using large primes
        self.PRIME1 = torch.tensor(73856093, dtype=torch.int64, device=self.device)
        self.PRIME2 = torch.tensor(19349663, dtype=torch.int64, device=self.device)
        self.PRIME3 = torch.tensor(83492791, dtype=torch.int64, device=self.device)

        # Set default values and update with config
        default_config = {
            "memory_threshold": 0.8,
            "adaptive_voxel_size": True,
            "base_voxel_size": 0.25,
            "min_scale": 1e-6,
            "max_scale": 10.0,
            "opacity_threshold": 0.005,
            "lr_decay_factor": 0.99,
            "lr_decay_steps": 100,
            "batch_size": 1,
            "world_size": 1  # Default world size if not provided
        }
        
        # Update defaults with provided config
        for key, value in config.items():
            default_config[key] = value
        self.config = default_config

        self.optim_cfg = {
            "betas": (0.9, 0.999),
            "eps": 1e-15,
            "lr_scales": {
                "means": 1.6e-4,
                "scales": 5e-3,
                "quats": 1e-3,
                "opacities": 5e-2,
                "sh0": 2.5e-3,
                "shN": 2.5e-3 / 20
            }
        }

    def quaternion_to_rotation_matrix(self, quaternions):
        """Convert quaternions to rotation matrices"""
        # Normalize quaternions
        quaternions = F.normalize(quaternions, dim=1)
        
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        
        # Compute rotation matrix elements
        xx, yy, zz = x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        
        # Create rotation matrices [N, 3, 3]
        rotation_matrices = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device, dtype=quaternions.dtype)
        
        rotation_matrices[:, 0, 0] = 1 - 2*(yy + zz)
        rotation_matrices[:, 0, 1] = 2*(xy - wz)
        rotation_matrices[:, 0, 2] = 2*(xz + wy)
        
        rotation_matrices[:, 1, 0] = 2*(xy + wz)
        rotation_matrices[:, 1, 1] = 1 - 2*(xx + zz)
        rotation_matrices[:, 1, 2] = 2*(yz - wx)
        
        rotation_matrices[:, 2, 0] = 2*(xz - wy)
        rotation_matrices[:, 2, 1] = 2*(yz + wx)
        rotation_matrices[:, 2, 2] = 1 - 2*(xx + yy)
        
        # Clean up temporary variables
        del xx, yy, zz, wx, wy, wz, xy, xz, yz, w, x, y, z
        
        return rotation_matrices

    def upload_splats(self, keyframe: KeyframeAgriGS):
        """
        Update splats based on keyframe data and maintain splat indices reference.
        Also setup camera pose optimization for the keyframe if trainable.
        
        Args:
            keyframe: KeyframeAgriGS object containing points, colors, poses, and timestamp
        """
        if keyframe.timestamp is None:
            return keyframe
        self.last_timestamp = keyframe.timestamp

        # Get points and colors from keyframe (keep on CPU initially)
        input_xyz_points = keyframe.get_points()
        input_rgb_points = keyframe.get_colors()
        
        # Early return if no points
        if input_xyz_points.numel() == 0:
            return keyframe

        # 1. Prepare query points (move to GPU for computations)
        self.active_points = input_xyz_points.squeeze().float().to(self.device)

        # Use adaptive voxel size if enabled
        voxel_size = self._get_adaptive_voxel_size() if self.config["adaptive_voxel_size"] else self.config["base_voxel_size"]

        # Handle validation mode (non-trainable) - only retrieve overlapping splats
        if not keyframe.trainable:
            with torch.no_grad():
                # 2. Transfer out splats no longer overlapping active ones
                if len(self.active_splats) > 0:
                    current_active_query_mask, active_target_mask = self._get_overlapping_splats(
                        self.active_points, self.active_splats["means"], voxel_size
                    )
                    
                    if isinstance(current_active_query_mask, torch.Tensor) and isinstance(active_target_mask, torch.Tensor):
                        # Update our tracking mask
                        active_query_mask = current_active_query_mask
                        
                        out_target_mask = ~active_target_mask
                        if out_target_mask.any():
                            self._transfer_between_dicts(
                                self.active_splats,
                                self.splats,
                                out_target_mask,
                                requires_grad=False,
                                to_device="cpu"
                            )
                        
                        # Clean up temporary masks
                        del out_target_mask, active_target_mask
                    
                    # Clean up query mask
                    del current_active_query_mask

                # 3. Transfer in new overlaps from splats into active_splats
                if len(self.splats) > 0:
                    _, target_mask = self._get_overlapping_splats(
                        self.active_points, self.splats["means"], voxel_size
                    )
                    
                    if isinstance(target_mask, torch.Tensor) and target_mask.any():
                        self._transfer_between_dicts(
                            self.splats,
                            self.active_splats,
                            target_mask,
                            requires_grad=True,
                            to_device=self.device
                        )
                                
                    # Clean up temporary masks
                    if isinstance(target_mask, torch.Tensor):
                        del target_mask

                # set splats
                keyframe.set_splats(self.active_splats)
                        
                # Force garbage collection to clean up any remaining temporary tensors
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                        
                return keyframe

        # Training mode (trainable=True) - original logic
        # Initialize the query mask - this will track which points already have active splats
        active_query_mask = torch.zeros(self.active_points.shape[0], dtype=torch.bool, device=self.device)

        # 2. Transfer out splats no longer overlapping active ones
        if len(self.active_splats) > 0:
            current_active_query_mask, active_target_mask = self._get_overlapping_splats(
                self.active_points, self.active_splats["means"], voxel_size
            )
            
            if isinstance(current_active_query_mask, torch.Tensor) and isinstance(active_target_mask, torch.Tensor):
                # Update our tracking mask
                active_query_mask = current_active_query_mask
                
                out_target_mask = ~active_target_mask
                if out_target_mask.any():
                    # Handle index tracking
                    if hasattr(self, '_active_splat_global_indices'):
                        if len(self._active_splat_global_indices) == len(self.active_splats["means"]):
                            out_indices = self._active_splat_global_indices[out_target_mask]
                            remaining_indices = self._active_splat_global_indices[~out_target_mask]
                            del self._active_splat_global_indices
                            self._active_splat_global_indices = remaining_indices
                            
                            # Store the outgoing indices
                            if not hasattr(self, '_splat_global_indices'):
                                self._splat_global_indices = out_indices
                            else:
                                if hasattr(self, '_splat_global_indices') and len(self._splat_global_indices) > 0:
                                    temp_indices = torch.cat([self._splat_global_indices, out_indices])
                                    del self._splat_global_indices
                                    self._splat_global_indices = temp_indices
                                else:
                                    self._splat_global_indices = out_indices
                        else:
                            # Indices are out of sync - reset them
                            print(f"Warning: Active splat indices out of sync. Resetting.")
                            if not hasattr(self, '_next_global_index'):
                                self._next_global_index = 0
                            num_active = len(self.active_splats["means"])
                            new_indices = torch.arange(
                                self._next_global_index, 
                                self._next_global_index + num_active,
                                device=self.device
                            )
                            if hasattr(self, '_active_splat_global_indices'):
                                del self._active_splat_global_indices
                            self._active_splat_global_indices = new_indices
                            self._next_global_index += num_active
                    
                    self._transfer_between_dicts(
                        self.active_splats,
                        self.splats,
                        out_target_mask,
                        requires_grad=False,
                        to_device="cpu"
                    )
                
                # Clean up temporary masks
                del out_target_mask, active_target_mask
            
            # Clean up query mask
            del current_active_query_mask

        # 3. Transfer in new overlaps from splats into active_splats
        if len(self.active_splats) != 0:
            if len(self.splats) > 0:
                _, target_mask = self._get_overlapping_splats(
                    self.active_splats["means"], self.splats["means"], voxel_size
                )
                
                if isinstance(target_mask, torch.Tensor) and target_mask.any():
                    # Handle index tracking for incoming splats
                    if hasattr(self, '_splat_global_indices') and len(self._splat_global_indices) > 0:
                        if len(self._splat_global_indices) == len(self.splats["means"]):
                            splats_to_add_indices = self._splat_global_indices[target_mask]
                            remaining_storage_indices = self._splat_global_indices[~target_mask]
                            del self._splat_global_indices
                            self._splat_global_indices = remaining_storage_indices
                            
                            if not hasattr(self, '_active_splat_global_indices'):
                                self._active_splat_global_indices = splats_to_add_indices
                            else:
                                temp_indices = torch.cat([
                                    self._active_splat_global_indices,
                                    splats_to_add_indices
                                ])
                                del self._active_splat_global_indices
                                self._active_splat_global_indices = temp_indices
                        else:
                            # Storage indices out of sync - generate new ones
                            if not hasattr(self, '_next_global_index'):
                                self._next_global_index = 0
                            num_incoming = int(target_mask.sum().item())
                            new_indices = torch.arange(
                                self._next_global_index,
                                self._next_global_index + num_incoming,
                                device=self.device
                            )
                            self._next_global_index += num_incoming
                            
                            if not hasattr(self, '_active_splat_global_indices'):
                                self._active_splat_global_indices = new_indices
                            else:
                                temp_indices = torch.cat([
                                    self._active_splat_global_indices,
                                    new_indices
                                ])
                                del self._active_splat_global_indices
                                self._active_splat_global_indices = temp_indices
                    
                    self._transfer_between_dicts(
                        self.splats,
                        self.active_splats,
                        target_mask,
                        requires_grad=True,
                        to_device=self.device
                    )
                            
                    # Update the query mask
                    if len(self.active_splats) > 0:
                        updated_query_mask, updated_target_mask = self._get_overlapping_splats(
                            self.active_points, self.active_splats["means"], voxel_size
                        )
                        if isinstance(updated_query_mask, torch.Tensor):
                            del active_query_mask
                            active_query_mask = updated_query_mask
                        del updated_target_mask
                
                # Clean up temporary masks
                if isinstance(target_mask, torch.Tensor):
                    del target_mask
        else:
            if len(self.splats) > 0:
                _, target_mask = self._get_overlapping_splats(
                    self.active_points, self.splats["means"], voxel_size
                )
                
                if isinstance(target_mask, torch.Tensor) and target_mask.any():
                    # Handle index tracking for first-time activation
                    if hasattr(self, '_splat_global_indices') and len(self._splat_global_indices) > 0:
                        if len(self._splat_global_indices) == len(self.splats["means"]):
                            active_indices = self._splat_global_indices[target_mask]
                            remaining_indices = self._splat_global_indices[~target_mask]
                            del self._splat_global_indices
                            self._active_splat_global_indices = active_indices
                            self._splat_global_indices = remaining_indices
                        else:
                            # Generate new indices if out of sync
                            if not hasattr(self, '_next_global_index'):
                                self._next_global_index = 0
                            num_activating = int(target_mask.sum().item())
                            new_indices = torch.arange(
                                self._next_global_index,
                                self._next_global_index + num_activating,
                                device=self.device
                            )
                            self._active_splat_global_indices = new_indices
                            self._next_global_index += num_activating
                    
                    self._transfer_between_dicts(
                        self.splats,
                        self.active_splats,
                        target_mask,
                        requires_grad=True,
                        to_device=self.device
                    )
                        
                    # Update the query mask
                    if len(self.active_splats) > 0:
                        updated_query_mask, updated_target_mask = self._get_overlapping_splats(
                            self.active_points, self.active_splats["means"], voxel_size
                        )
                        if isinstance(updated_query_mask, torch.Tensor):
                            del active_query_mask
                            active_query_mask = updated_query_mask
                        del updated_target_mask
                    
                    # Clean up temporary masks
                    del target_mask

        # 4. Initialize new splats for points not yet active
        new_mask = ~active_query_mask
        
        if new_mask.any():
            # Get next available global indices for new splats
            if not hasattr(self, '_next_global_index'):
                self._next_global_index = 0
            
            num_new_splats = int(new_mask.sum().item())
            new_splat_indices = torch.arange(
                self._next_global_index, 
                self._next_global_index + num_new_splats,
                device=self.device
            )
            self._next_global_index += num_new_splats
            
            # Move mask to CPU to index CPU tensors
            new_mask_cpu = new_mask.cpu()
            new_points = input_xyz_points[new_mask_cpu]
            new_colors = input_rgb_points[new_mask_cpu]
            del new_mask_cpu
            
            self._initialize_new_splats(new_points, new_colors)
            
            # Update active splat indices tracking
            if not hasattr(self, '_active_splat_global_indices'):
                self._active_splat_global_indices = new_splat_indices
            else:
                temp_indices = torch.cat([
                    self._active_splat_global_indices,
                    new_splat_indices
                ])
                del self._active_splat_global_indices
                self._active_splat_global_indices = temp_indices
            
            # Clean up temporary data
            del new_points, new_colors

        # Clean up temporary masks
        del new_mask, active_query_mask

        # 5. Setup camera pose optimization for new keyframe
        self._setup_camera_pose_optimization(keyframe)

        # 6. Final synchronization check and keyframe update
        if hasattr(self, '_active_splat_global_indices') and len(self._active_splat_global_indices) > 0:
            # Ensure indices match active splats count
            if len(self.active_splats) > 0:
                expected_count = len(self.active_splats["means"])
                if len(self._active_splat_global_indices) != expected_count:
                    print(f"Warning: Final sync mismatch. Regenerating indices.")
                    if not hasattr(self, '_next_global_index'):
                        self._next_global_index = 0
                    new_indices = torch.arange(
                        self._next_global_index,
                        self._next_global_index + expected_count,
                        device=self.device
                    )
                    if hasattr(self, '_active_splat_global_indices'):
                        del self._active_splat_global_indices
                    self._active_splat_global_indices = new_indices
                    self._next_global_index += expected_count
            
            keyframe.set_splat_indices(self._active_splat_global_indices.clone())
        else:
            empty_indices = torch.empty(0, dtype=torch.long, device=self.device)
            keyframe.set_splat_indices(empty_indices)
            del empty_indices

        # 7. Update optimizers while preserving state
        self._update_optimizers()
        keyframe.set_splats(self.active_splats)
        self.monitor.update_dashboard_splats(
            len(self.active_splats["means"]) if len(self.active_splats) > 0 else 0,
            len(self.splats["means"]) if len(self.splats) > 0 else 0,
        )
        
        # Force garbage collection to clean up any remaining temporary tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return keyframe

    def _setup_camera_pose_optimization(self, keyframe: KeyframeAgriGS):
        """Setup camera pose optimization for the keyframe"""
        # Clean up existing pose optimizers/schedulers for new keyframe
        if hasattr(self, 'pose_optimizers'):
            del self.pose_optimizers
        if hasattr(self, 'pose_schedulers'):
            del self.pose_schedulers
        
        self.pose_optimizers = {}
        self.pose_schedulers = {}
        
        # Convert world2cam poses to cam2world poses for optimization
        if keyframe.world2cams_poses:
            for cam_name, world2cam in keyframe.world2cams_poses.items():
                # Convert world2cam to cam2world (inverse transformation)
                with torch.no_grad():
                    cam2world = torch.inverse(world2cam)
                    temp_cam2world = cam2world.detach().clone().requires_grad_(True)
                    del cam2world
                    cam2world = temp_cam2world
                
                # Get config values with defaults
                pose_lr = 5e-6  # Slightly reduced learning rate for finer adjustments
                pose_reg = 5e-8  # Reduced weight decay for less regularization
                
                # Use Adam optimizer for camera pose
                pose_optimizer = torch.optim.Adam(
                    [cam2world],
                    lr=pose_lr,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=pose_reg
                )

                # Use StepLR with a smoother decay rate
                pose_scheduler = torch.optim.lr_scheduler.StepLR(
                    pose_optimizer,
                    step_size=5,  # Increased step size for slower decay
                    gamma=0.85  # Slightly higher gamma for smoother decay
                )
                # Store optimizers and schedulers
                self.pose_optimizers[cam_name] = pose_optimizer
                self.pose_schedulers[cam_name] = pose_scheduler

                if not hasattr(self, 'optimizable_cam2worlds'):
                    self.optimizable_cam2worlds = {}
                    self.optimizable_cam2worlds[cam_name] = cam2world
                
                # Store the learnable cam2world back to keyframe
                keyframe.cam2world_optimizable[cam_name] = cam2world

    def _get_adaptive_voxel_size(self):
        """Calculate adaptive voxel size based on point density"""
        if self.active_points is None or len(self.active_points) < 2:
            return self.config["base_voxel_size"]
        
        # Calculate mean distance to nearest neighbor as proxy for density
        with torch.no_grad():
            if len(self.active_points) >= 4:
                dist2_avg = distCUDA2(self.active_points)
                sqrt_dist = torch.sqrt(dist2_avg)
                mean_dist = sqrt_dist.mean().item()
                # Clean up temporary tensors
                del dist2_avg, sqrt_dist
                
                # Scale voxel size based on point density
                adaptive_size = max(self.config["base_voxel_size"] * 0.5, 
                                  min(self.config["base_voxel_size"] * 2.0, mean_dist * 0.5))
                return adaptive_size
            else:
                return self.config["base_voxel_size"]

    def _transfer_between_dicts(self, src_dict, dst_dict, mask, requires_grad, to_device):
        """Transfer parameters between dicts with device change, ensuring all tensors are on the same device"""
        with torch.no_grad():
            for name in list(src_dict.keys()):
                param = src_dict[name]
                # Ensure mask is on the same device as the parameter data
                mask_device = param.data.device
                mask_on_device = mask.to(mask_device)

                # Select and move data for masked indices
                selected = param.data[mask_on_device]
                data = selected.to(to_device)
                del selected

                # Preserve existing or create new parameter in destination
                if name in dst_dict:
                    dst_data_device = dst_dict[name].data.device
                    dst_data = dst_dict[name].data.to(to_device) if dst_data_device != to_device else dst_dict[name].data
                    new_data = torch.cat([dst_data, data])
                    del dst_dict[name]  # Clean up old parameter
                    dst_dict[name] = torch.nn.Parameter(
                        new_data,
                        requires_grad=requires_grad
                    )
                else:
                    dst_dict[name] = torch.nn.Parameter(
                        data,
                        requires_grad=requires_grad
                    )

                # Update or remove source parameter based on remaining indices
                remaining_mask = ~mask_on_device
                remaining = param.data[remaining_mask]
                del mask_on_device, remaining_mask
                
                if remaining.numel() > 0:
                    del src_dict[name]  # Clean up old parameter
                    src_dict[name] = torch.nn.Parameter(
                        remaining.to(mask_device),
                        requires_grad=param.requires_grad
                    )
                else:
                    del src_dict[name]

    def _update_optimizers(self):
        """Maintain optimizers with persistent state and add schedulers."""
        BS = self.config["batch_size"] * self.config.get("world_size", 1)
        sqrt_BS = math.sqrt(BS)
        
        for param_name in self.active_splats.keys():
            param = self.active_splats[param_name]
            lr = self.optim_cfg["lr_scales"][param_name] * sqrt_BS * self.config["scene_scale"]

            if param_name not in self.optimizers:
                # Initialize new optimizer
                opt_class = torch.optim.SparseAdam if self.config["sparse_grad"] else torch.optim.Adam
                self.optimizers[param_name] = opt_class(
                    [{"params": param, "lr": lr}],
                    eps=self.optim_cfg["eps"]/sqrt_BS,
                    betas=self.optim_cfg["betas"]
                )
                if param_name in self.param_state_cache:
                    self.optimizers[param_name].load_state_dict(self.param_state_cache[param_name])
                
                # Add a scheduler for the optimizer
                scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizers[param_name], 
                    step_size=self.config["lr_decay_steps"], 
                    gamma=self.config["lr_decay_factor"]
                )
                self.schedulers[param_name] = scheduler
            else:
                # Update existing optimizer
                self._update_optimizer_params(param_name, param, lr)

    def _update_optimizer_params(self, param_name, param, lr):
        """Smart parameter-group updating with state preservation."""
        opt = self.optimizers[param_name]

        # 1) Find all existing parameters in the optimizer
        existing = {id(p) for group in opt.param_groups for p in group['params']}

        # 2) If our new param isn't in there yet, add it as its own group
        if id(param) not in existing:
            opt.add_param_group({
                "params": [param],
                "lr": lr
            })

        # 3) Remove parameter groups that are no longer active
        active_ids = {id(self.active_splats[param_name])}
        new_groups = []
        for group in opt.param_groups:
            p = group['params'][0]
            if id(p) in active_ids:
                group['lr'] = lr
                new_groups.append(group)
            else:
                # Cache state before removal
                self.param_state_cache.setdefault(param_name, {})[id(p)] = opt.state.pop(p, None)
        opt.param_groups = new_groups

        # 4) Save optimizer state
        self.param_state_cache[param_name] = opt.state_dict()

    def _initialize_new_splats(self, points, rgbs):
        """Initialize new splats on GPU with proper parameter initialization"""
        points_gpu = points.to(self.device)
        rgbs_gpu = rgbs.to(self.device)
        
        # Calculate scales from nearest neighbors
        if len(points_gpu) >= 4:
            dist2_avg = distCUDA2(points_gpu)
            sqrt_dist = torch.sqrt(dist2_avg)
            init_scale_tensor = torch.full_like(sqrt_dist, self.config["init_scale"])
            scales = torch.log(sqrt_dist * init_scale_tensor).unsqueeze(-1).repeat(1, 3)
            # Clean up temporary tensors
            del dist2_avg, sqrt_dist, init_scale_tensor
        else:
            scale_value = torch.full((len(points_gpu),), self.config["init_scale"], device=self.device)
            scales = torch.log(scale_value).unsqueeze(-1).repeat(1, 3)
            del scale_value

        # Clamp scales to configured limits
        min_scale_log = math.log(self.config["min_scale"])
        max_scale_log = math.log(self.config["max_scale"])
        scales = torch.clamp(scales, min_scale_log, max_scale_log)

        # Initialize quaternions (normalized random quaternions)
        quats = torch.rand((len(points_gpu), 4), device=self.device, dtype=points_gpu.dtype)
        quats = F.normalize(quats, dim=1)
        
        # Create new parameters
        opacity_tensor = torch.full((len(points_gpu),), self.config["init_opacity"], device=self.device)
        opacities = torch.logit(opacity_tensor)
        del opacity_tensor
        
        sh0_data = rgb_to_sh(rgbs_gpu).unsqueeze(1)
        shN_data = torch.zeros((len(points_gpu), (self.config["sh_degree"]+1)**2-1, 3), device=self.device)
        
        new_params = {
            "means": points_gpu,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": sh0_data,
            "shN": shN_data
        }

        # Add to active splats
        for name, data in new_params.items():
            if name in self.active_splats:
                concatenated_data = torch.cat([self.active_splats[name].data, data])
                del self.active_splats[name]  # Clean up old parameter
                self.active_splats[name] = torch.nn.Parameter(
                    concatenated_data,
                    requires_grad=True
                )
            else:
                self.active_splats[name] = torch.nn.Parameter(data, requires_grad=True)
        
        # Clean up temporary data
        del points_gpu, rgbs_gpu, scales, quats, opacities, sh0_data, shN_data

    def get_trainable_params(self):
        """Return active parameters for rendering/training"""
        return self.active_splats

    def cache_optimizer_states(self):
        """Preserve optimizer states before potential parameter removal"""
        for name, opt in self.optimizers.items():
            self.param_state_cache[name] = opt.state_dict()

    @torch.no_grad()
    def _get_overlapping_splats(self, query_points, target_splats, voxel_size=None):
        if voxel_size is None:
            voxel_size = self.config["base_voxel_size"]
            
        target_splats_gpu = target_splats.to(self.device)

        # Quantize to voxel coordinates
        voxel_scale = 1.0 / voxel_size
        A_voxels = (query_points * voxel_scale).floor().long()
        B_voxels = (target_splats_gpu * voxel_scale).floor().long()

        # Hash using prime multiplication
        keys_A = (A_voxels[:, 0] * self.PRIME1 + A_voxels[:, 1] * self.PRIME2 + A_voxels[:, 2] * self.PRIME3)
        keys_B = (B_voxels[:, 0] * self.PRIME1 + B_voxels[:, 1] * self.PRIME2 + B_voxels[:, 2] * self.PRIME3)

        # Clean up temporary voxel coordinates
        del A_voxels, B_voxels

        # GPU-based intersection
        unique_keys_A = torch.unique(keys_A)
        unique_keys_B = torch.unique(keys_B)

        # Concatenate and find intersection
        concatenated_uniques = torch.cat((unique_keys_A, unique_keys_B))
        del unique_keys_A, unique_keys_B

        # Find common keys
        u, counts = torch.unique(concatenated_uniques, return_counts=True)
        common_keys = u[counts > 1]
        del concatenated_uniques, u, counts

        # Create masks using the identified common keys
        mask_A = torch.isin(keys_A, common_keys)
        mask_B = torch.isin(keys_B, common_keys)
        
        # Clean up temporary tensors
        del keys_A, keys_B, common_keys

        return mask_A, mask_B
    
    def save_checkpoint(self, filename: str):
        # Temporarily merge active splats and stored splats
        merged_splats = torch.nn.ParameterDict()
        
        # Copy stored splats to merged dict
        for name, param in self.splats.items():
            merged_splats[name] = torch.nn.Parameter(param.data.clone())
        
        # Add active splats to merged dict
        for name, param in self.active_splats.items():
            if name in merged_splats:
                # Concatenate with existing stored splats
                merged_data = torch.cat([merged_splats[name].data, param.data.cpu()])
                del merged_splats[name]  # Clean up old parameter
                merged_splats[name] = torch.nn.Parameter(merged_data)
            else:
                # Add new parameter
                merged_splats[name] = torch.nn.Parameter(param.data.cpu().clone())
        
        # Save the checkpoint with merged splats
        checkpoint_data = {"step": 0, "splats": merged_splats.state_dict()}
        torch.save(checkpoint_data, f"{filename}/ckpt.pt")
        del checkpoint_data
        
        # Free the temporary merged splats memory
        del merged_splats
        
        # Force garbage collection and GPU memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def optimize(self, steps: int = 1):
        for step in range(steps):
            # 1) take an optimizer step for splat parameters
            for _, optimizer in self.optimizers.items():
                optimizer.step()
                if step == steps - 1:
                    optimizer.zero_grad(set_to_none=True)
            
            # 2) then step each scheduler for splat parameters
            for _, scheduler in self.schedulers.items():
                scheduler.step()

        # 1b) take an optimizer step for pose parameters
        if hasattr(self, 'pose_optimizers'):
            for _, pose_optimizer in self.pose_optimizers.items():
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)
        
        # 2b) then step each scheduler for pose parameters
        if hasattr(self, 'pose_schedulers'):
            for _, pose_scheduler in self.pose_schedulers.items():
                pose_scheduler.step()