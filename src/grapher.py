from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
import yaml
import gtsam
from gtsam import ISAM2, ISAM2Params, NonlinearFactorGraph, Values, Pose3, Point3
from gtsam import PriorFactorPose3, BetweenFactorPose3
from keyframe import KeyframeAgriGS
from monitor import MonitorAgriGS

# Import DLO for point cloud alignment
try:
    import odometry as dlo
    DLO_AVAILABLE = True
    print("✅ DLO (Direct Lidar Odometry) available for loop closure refinement")
except ImportError:
    print("❌ DLO module not found. Loop closure refinement disabled.")
    DLO_AVAILABLE = False

# ScanContext for loop closure detection
try:
    import scancontext
except ImportError:
    print("Warning: scancontext module not found. Loop closure detection disabled.")
    scancontext = None

import time
from datetime import datetime


class MinimalKeyframe:
    """Minimal keyframe storage with only essential data."""
    
    def __init__(self, keyframe_id: int, timestamp: float, pose: torch.Tensor, pointcloud: Optional[np.ndarray] = None):
        self.id = keyframe_id
        self.timestamp = timestamp
        self.world2robot_pose = pose
        self.lidar_pointcloud = pointcloud


class GrapherAgriGS:
    """
    GTSAM ISAM2-based LIDAR SLAM system for AgriGS keyframes.
    Now includes ScanContext-based loop closure detection with DLO refinement.
    """

    def __init__(self, config: dict = None, monitor: MonitorAgriGS = None):
        """
        Initialize the GrapherAgriGS SLAM system.
        
        Args:
            config: Configuration dictionary
            monitor: DashboardAgriGS instance for visualization
        """
        # Load configuration
        self.config = config if config is not None else {}
        self.setup_default_config()

        # ISAM2 parameters
        params = ISAM2Params()
        params.setRelinearizeThreshold(self.config.get('isam2', {}).get('relinearize_threshold', 0.01))
        self.isam = ISAM2(params)
        
        # Graph and values
        self.graph = NonlinearFactorGraph()
        self.initial_estimate = Values()
        self.current_estimate = Values()
        
        # Minimal keyframes storage - only essential data
        self.keyframes: List[MinimalKeyframe] = []
        self.keyframe_poses: Dict[int, gtsam.Pose3] = {}
        
        # Loop closure detection and refinement
        self.sc_manager = None
        self.loop_closures: List[Tuple[int, int, gtsam.Pose3]] = []  # (from_id, to_id, relative_pose)
        self.loop_closure_candidates: List[Tuple[int, float]] = []  # Store recent candidates for visualization
        self.dlo_alignment_results: List[Dict] = []  # Store DLO alignment results for visualization
        
        if scancontext is not None:
            self.sc_manager = scancontext.SCManager()
            self.update_scancontext_params()
            print("✅ ScanContext loop closure detection enabled")
        else:
            print("❌ ScanContext not available - loop closure detection disabled")
        
        # State
        self.pose_count = 0
        self.last_pose_key = None
        self.monitor = monitor
    
    def update_scancontext_params(self):
        """Update ScanContext parameters from config."""
        if self.sc_manager is not None:
            sc_config = self.config.get('scancontext', {})
            self.sc_manager.set_position_search_radius(sc_config.get('position_search_radius', 2.0))
            self.sc_manager.set_position_search_min_candidates(sc_config.get('min_candidates', 1))
            self.sc_manager.set_position_search_max_candidates(sc_config.get('max_candidates', 10))
            self.sc_manager.set_time_exclusion_window(sc_config.get('time_exclusion_window', 30))
            self.sc_manager.set_lidar_height(sc_config.get('lidar_height', 2.0))
    
    def load_scancontext_params_from_dashboard(self):
        """Load updated ScanContext parameters from dashboard."""
        updated_params = self.monitor.get_updated_scancontext_params()
        if updated_params:
            # Update config
            self.config['scancontext'].update(updated_params)
            
            # Update ScanContext manager
            self.update_scancontext_params()
            
            # Update dashboard
            self.monitor.update_scancontext_params(self.config['scancontext'])
            
            print(f"📊 Updated ScanContext parameters: {updated_params}")
    
    def _extract_point_cloud_from_keyframe(self, keyframe: MinimalKeyframe) -> Optional[np.ndarray]:
        """
        Extract point cloud from minimal keyframe for ScanContext processing.
        
        Args:
            keyframe: MinimalKeyframe object
            
        Returns:
            numpy array of shape (N, 3) or (N, 4) with x, y, z, [intensity]
        """
        try:
            # Extract point cloud from minimal keyframe
            if keyframe.lidar_pointcloud is not None:
                # If point cloud is stored as tensor
                if isinstance(keyframe.lidar_pointcloud, torch.Tensor):
                    points = keyframe.lidar_pointcloud.cpu().numpy()
                else:
                    points = np.array(keyframe.lidar_pointcloud)
                
                # Ensure proper shape (N, 3) or (N, 4)
                if points.ndim == 2 and points.shape[1] >= 3:
                    return points.astype(np.float32)
            
            return None
            
        except Exception as e:
            print(f"Error extracting point cloud from keyframe {keyframe.id}: {e}")
            return None
    
    def _gtsam_pose_to_pcl_point(self, pose: gtsam.Pose3):
        """
        Convert GTSAM Pose3 to PCL PointXYZ for ScanContext.
        
        Args:
            pose: GTSAM Pose3 object
            
        Returns:
            PCL PointXYZ object
        """
        translation = pose.translation()
        return scancontext.PointXYZ(float(translation[0]), float(translation[1]), float(translation[2]))
    
    def _refine_loop_closure_with_dlo(self, source_keyframe: MinimalKeyframe, target_keyframe: MinimalKeyframe) -> Optional[Tuple[gtsam.Pose3, Dict]]:
        """
        Refine loop closure using DLO point cloud alignment.
        
        Args:
            source_keyframe: Source keyframe for alignment
            target_keyframe: Target keyframe for alignment
            
        Returns:
            Tuple of (refined_relative_pose, alignment_info) if successful, None otherwise
        """
        if not DLO_AVAILABLE:
            return None
            
        try:
            # Extract point clouds
            source_points = self._extract_point_cloud_from_keyframe(source_keyframe)
            target_points = self._extract_point_cloud_from_keyframe(target_keyframe)
            
            if source_points is None or target_points is None:
                print(f"❌ Could not extract point clouds for DLO alignment")
                return None
            
            # Ensure point clouds have sufficient points
            min_points = self.config['dlo_alignment']['min_points']
            if len(source_points) < min_points or len(target_points) < min_points:
                print(f"❌ Insufficient points for DLO alignment (need {min_points}, got {len(source_points)} and {len(target_points)})")
                return None
            
            # Get poses from graph for initial guess
            source_pose = self.keyframe_poses[source_keyframe.id]
            target_pose = self.keyframe_poses[target_keyframe.id]
            
            # Calculate initial guess as transform from source to target
            relative_pose = source_pose.inverse().compose(target_pose)
            initial_guess_matrix = relative_pose.matrix()
            
            # Convert to flat vector for DLO (row-major order)
            initial_guess_vec = initial_guess_matrix.flatten().tolist()
            
            # DLO alignment parameters
            dlo_config = self.config['dlo_alignment']
            
            print(f"🔄 DLO alignment: {len(source_points)} -> {len(target_points)} points (with initial guess)")
            
            # Call DLO alignment with initial guess
            result = dlo.DirectLidarOdometry.align_point_clouds(
                source_points,
                target_points,
                initial_guess_vec,
                max_iterations=dlo_config['max_iterations'],
                transformation_epsilon=dlo_config['transformation_epsilon'],
                euclidean_fitness_epsilon=dlo_config['euclidean_fitness_epsilon'],
                max_correspondence_distance=dlo_config['max_correspondence_distance']
            )
            
            # Parse result tuple: (transformation_matrix, fitness_score, has_converged)
            transform_matrix, fitness_score, has_converged = result
            
            # Convert transformation matrix list to numpy array
            transform_np = np.array(transform_matrix).reshape(4, 4)
            
            # Calculate translation magnitude
            translation_magnitude = np.linalg.norm(transform_np[:3, 3])
            
            # Store alignment results for visualization
            alignment_info = {
                'source_id': source_keyframe.id,
                'target_id': target_keyframe.id,
                'fitness_score': fitness_score,
                'has_converged': has_converged,
                'translation_magnitude': translation_magnitude,
                'transformation_matrix': transform_np.tolist(),
                'initial_guess_matrix': initial_guess_matrix.tolist(),
                'accepted': False,
                'timestamp': datetime.now()
            }
            
            # Check convergence and translation constraints
            max_translation = self.config['dlo_alignment']['max_translation']
            
            if not has_converged:
                print(f"❌ DLO alignment did not converge (fitness: {fitness_score:.4f})")
                alignment_info['rejection_reason'] = "No convergence"
                self.dlo_alignment_results.append(alignment_info)
                return None
            
            if translation_magnitude > max_translation:
                print(f"❌ DLO alignment translation too large: {translation_magnitude:.2f}m > {max_translation}m")
                alignment_info['rejection_reason'] = f"Translation too large: {translation_magnitude:.2f}m"
                self.dlo_alignment_results.append(alignment_info)
                return None
            
            # Convert to GTSAM Pose3
            rotation_matrix = transform_np[:3, :3]
            translation_vector = transform_np[:3, 3]
            
            gtsam_rotation = gtsam.Rot3(rotation_matrix)
            gtsam_translation = gtsam.Point3(translation_vector)
            refined_pose = gtsam.Pose3(gtsam_rotation, gtsam_translation)
            
            # Mark as accepted
            alignment_info['accepted'] = True
            alignment_info['rejection_reason'] = None
            self.dlo_alignment_results.append(alignment_info)
            
            print(f"✅ DLO alignment successful: translation={translation_magnitude:.2f}m, fitness={fitness_score:.4f}")
            
            return refined_pose, alignment_info
            
        except Exception as e:
            print(f"❌ Error in DLO alignment: {e}")
            return None

    def _detect_loop_closure(self, current_keyframe: MinimalKeyframe) -> Optional[Tuple[int, gtsam.Pose3]]:
        """
        Detect loop closure using ScanContext with DLO refinement.
        Now handles maximum 3 candidates and selects the best one based on fitness score.

        Args:
            current_keyframe: Current keyframe to check for loop closure

        Returns:
            Tuple of (matched_keyframe_id, refined_relative_pose) if loop closure detected, None otherwise
        """
        if self.sc_manager is None:
            return None
            
        try:
            # Load updated parameters from dashboard
            self.load_scancontext_params_from_dashboard()
            
            # Extract point cloud from current keyframe
            points = self._extract_point_cloud_from_keyframe(current_keyframe)
            if points is None:
                return None

            # Convert current keyframe pose to PCL point
            current_pose = self.keyframe_poses[current_keyframe.id]
            pcl_pose = self._gtsam_pose_to_pcl_point(current_pose)

            # Use the detect_loop_closure API
            loop_result = self.sc_manager.detect_loop_closure(points, current_keyframe.id, pcl_pose)
            
            # Clear previous candidates
            self.loop_closure_candidates = []
            
            # Parse the result
            if loop_result is not None:
                detected, result_tuple = loop_result
                if detected:
                    # Extract the three candidate IDs from the result tuple
                    candidates, _, _ = result_tuple
                    
                    # Filter valid candidates (not -1)
                    valid_candidates = [c for c in [candidates[0], candidates[1], candidates[2]] if c != -1]
                    
                    if not valid_candidates:
                        print("🔍 No valid loop closure candidates found")
                        return None
                    
                    print(f"🔍 Found {len(valid_candidates)} loop closure candidates: {valid_candidates}")
                    
                    # For visualization, store candidates with their distances
                    current_pos = current_pose.translation()
                    for candidate_id in valid_candidates:
                        if candidate_id < len(self.keyframes) and candidate_id in self.keyframe_poses:
                            candidate_pose = self.keyframe_poses[candidate_id]
                            candidate_pos = candidate_pose.translation()
                            distance = np.linalg.norm(np.array(current_pos) - np.array(candidate_pos))
                            self.loop_closure_candidates.append((candidate_id, distance))
                    
                    # Process each valid candidate with DLO alignment
                    best_candidate = None
                    best_fitness = -1.0
                    best_pose = None
                    best_alignment_info = None
                    
                    for candidate_id in valid_candidates:
                        # Validate candidate_id and check minimum distance requirement
                        if candidate_id < 0 or candidate_id >= len(self.keyframes):
                            print(f"⚠️ Invalid candidate ID: {candidate_id}")
                            continue
                            
                        # Check if keyframes are far enough apart in sequence
                        min_id_distance = self.config['loop_closure']['min_id_distance']
                        if abs(current_keyframe.id - candidate_id) < min_id_distance:
                            print(f"⚠️ Loop closure candidate {candidate_id} too close in sequence (distance: {abs(current_keyframe.id - candidate_id)})")
                            continue
                        
                        # Get candidate keyframe
                        candidate_keyframe = self.keyframes[candidate_id]
                        
                        # Refine with DLO alignment
                        refinement_result = self._refine_loop_closure_with_dlo(current_keyframe, candidate_keyframe)
                        
                        if refinement_result is not None:
                            refined_pose, alignment_info = refinement_result
                            fitness_score = alignment_info['fitness_score']
                            
                            print(f"✅ Candidate {candidate_id} aligned successfully: fitness={fitness_score:.4f}")
                            
                            # Check if this is the best candidate so far
                            if fitness_score > best_fitness and fitness_score <= 1.00:
                                best_candidate = candidate_id
                                best_fitness = fitness_score
                                best_pose = refined_pose
                                best_alignment_info = alignment_info
                        else:
                            print(f"❌ Candidate {candidate_id} failed DLO alignment")
                    
                    # Return the best candidate if found
                    if best_candidate is not None:
                        print(f"🎯 Selected best loop closure candidate: {best_candidate} (fitness: {best_fitness:.4f})")
                        return (best_candidate, best_pose)
                    else:
                        print("❌ No candidates passed DLO alignment")

        except Exception as e:
            print(f"Error in loop closure detection: {e}")

        return None

    def _create_minimal_keyframe(self, keyframe: KeyframeAgriGS) -> MinimalKeyframe:
        """
        Create a minimal keyframe with only essential data.
        
        Args:
            keyframe: Original KeyframeAgriGS object
            
        Returns:
            MinimalKeyframe with only essential data
        """
        # Extract point cloud if available
        pointcloud = self._extract_point_cloud_from_keyframe(keyframe)
        
        return MinimalKeyframe(
            keyframe_id=keyframe.id,
            timestamp=keyframe.timestamp,
            pose=keyframe.world2robot_pose,
            pointcloud=pointcloud
        )

    def process(self, keyframe: KeyframeAgriGS) -> KeyframeAgriGS:
        """
        Main processing function with loop closure detection, DLO refinement, and real-time visualization.
        
        Args:
            keyframe: KeyframeAgriGS to process
            
        Returns:
            KeyframeAgriGS: Processed keyframe
        """
        # Step 1: Convert keyframe pose to GTSAM pose
        current_pose = self._tensor_to_pose3(keyframe.world2robot_pose)
        pose_key = gtsam.symbol('x', keyframe.id)
        
        # Step 2: Add prior for first pose or odometry factor
        if self.pose_count == 0:
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(self.config['noise_models']['prior_pose'])
            )
            self.graph.add(PriorFactorPose3(pose_key, current_pose, prior_noise))
        else:
            self._add_odometry_factor(pose_key, current_pose)
        
        # Step 3: Add initial estimate and store minimal keyframe
        self.initial_estimate.insert(pose_key, current_pose)
        minimal_keyframe = self._create_minimal_keyframe(keyframe)
        self.keyframes.append(minimal_keyframe)
        self.keyframe_poses[keyframe.id] = current_pose
        
        # Step 4: Add scan to ScanContext database
        self._add_scan_to_database(minimal_keyframe)
        
        # Step 5: Loop closure detection with DLO refinement (only after minimum keyframes)
        loop_result = None
        if (self.config['loop_closure']['enable'] and 
            self.pose_count >= self.config['loop_closure']['min_keyframes']):
            loop_result = self._detect_loop_closure(minimal_keyframe)
            if loop_result is not None:
                matched_id, refined_relative_pose = loop_result
                self._add_loop_closure_factor(keyframe.id, matched_id, refined_relative_pose)
        
        # Step 6: Update visualization data
        self.update_visualization_data(keyframe)
        
        # Step 7: Update state
        self.last_pose_key = pose_key
        self.pose_count += 1
        
        status = "🔄 LOOP+DLO" if loop_result is not None else "➡️"
        print(f"{status} Processed keyframe {keyframe.id} (timestamp: {keyframe.timestamp:.2f})")
        
        return keyframe
    
    def _add_scan_to_database(self, keyframe: MinimalKeyframe):
        """
        Add scan data to ScanContext database using new API.
        
        Args:
            keyframe: MinimalKeyframe object
        """
        if self.sc_manager is None:
            return
            
        try:
            # Extract point cloud
            points = self._extract_point_cloud_from_keyframe(keyframe)
            if points is None:
                return
                
            # Convert pose to PCL point
            current_pose = self.keyframe_poses[keyframe.id]
            pcl_pose = self._gtsam_pose_to_pcl_point(current_pose)
            
            # Add scan data to database
            self.sc_manager.add_scan_data(points, keyframe.id, pcl_pose)
            
        except Exception as e:
            print(f"Error adding scan to database: {e}")
    
    def _add_loop_closure_factor(self, from_id: int, to_id: int, relative_pose: gtsam.Pose3):
        """
        Add loop closure factor to the graph and optimize it with ISAM2.
        
        Args:
            from_id: Source keyframe ID
            to_id: Target keyframe ID
            relative_pose: Relative pose between keyframes
        """
        try:
            from_key = gtsam.symbol('x', from_id)
            to_key = gtsam.symbol('x', to_id)
            
            # Use more conservative noise model for loop closures
            loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(self.config['noise_models']['loop_closure'])
            )
            
            # Add between factor for loop closure
            self.graph.add(BetweenFactorPose3(from_key, to_key, relative_pose, loop_noise))
            
            # Store loop closure for visualization
            self.loop_closures.append((from_id, to_id, relative_pose))
            
            print(f"➕ Added loop closure factor: {from_id} -> {to_id}")
            
            # Optimize the graph with ISAM2
            self._perform_optimization()
            print("✅ Graph optimized after adding loop closure factor")
            
        except Exception as e:
            print(f"Error adding loop closure factor: {e}")
    
    def update_visualization_data(self, keyframe: KeyframeAgriGS):
        """Update visualization data for dashboard."""
        try:
            # Update trajectory with training/validation status
            trajectory_data = {'x': [], 'y': [], 'z': [], 'timestamps': [], 'trainable': []}
            
            for i, minimal_kf in enumerate(self.keyframes):
                if i in self.keyframe_poses:
                    pose = self.keyframe_poses[i]
                    translation = pose.translation()
                    trajectory_data['x'].append(float(translation[0]))
                    trajectory_data['y'].append(float(translation[1]))
                    trajectory_data['z'].append(float(translation[2]))
                    trajectory_data['timestamps'].append(float(minimal_kf.timestamp))
                    # Use original keyframe's trainable status for visualization
                    trajectory_data['trainable'].append(keyframe.trainable)
            
            # Current pose
            current_pose_tensor = keyframe.world2robot_pose
            current_pos = current_pose_tensor[:3, 3].cpu().numpy()
            
            # Determine current status based on trainable flag
            current_status = "Training" if keyframe.trainable else "Validation"
            
            # Prepare loop closure data for visualization
            loop_closures_viz = []
            for from_id, to_id, rel_pose in self.loop_closures:
                loop_closures_viz.append({
                    'from_id': from_id,
                    'to_id': to_id,
                    'distance': np.linalg.norm(rel_pose.translation())
                })
            
            # Prepare loop closure candidates for visualization
            candidates_viz = []
            for candidate_id, distance in self.loop_closure_candidates:
                candidates_viz.append({
                    'candidate_id': candidate_id,
                    'distance': distance
                })
            
            # Update dashboard with all data using new API methods
            self.monitor.update_dashboard_trajectory(trajectory_data)
            self.monitor.update_dashboard_pose({
                'x': float(current_pos[0]),
                'y': float(current_pos[1]),
                'z': float(current_pos[2])
            })
            self.monitor.update_dashboard_keyframe_count(len(self.keyframes))
            self.monitor.update_dashboard_loop_closures(loop_closures_viz)
            self.monitor.update_dashboard_candidates(candidates_viz)
            self.monitor.update_dashboard_dlo_results(self.dlo_alignment_results)
            self.monitor.update_dashboard_scancontext_params(self.config['scancontext'])
            self.monitor.update_dashboard_status(current_status)
            
        except Exception as e:
            print(f"Error updating visualization data: {e}")

    
    def setup_default_config(self):
        """Set up default configuration values."""
        default_config = {
            'isam2': {
                'relinearize_threshold': 0.01,
                'relinearize_skip': 1
            },
            'noise_models': {
                'prior_pose': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'odometry': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'loop_closure': [0.5, 0.5, 0.5, 0.3, 0.3, 0.3]  # More conservative for loop closures
            },
            'optimization': {
                'optimize_every_n_keyframes': 10
            },
            'loop_closure': {
                'enable': False,
                'min_keyframes': 5,  # Minimum keyframes before detecting loops
                'min_id_distance': 15,  # Minimum distance between keyframe IDs for loop closure
                'distance_threshold': 1.0  # Minimum distance for loop closure
            },
            'scancontext': {
                'position_search_radius': 2.0,
                'min_candidates': 1,
                'max_candidates': 10,
                'time_exclusion_window': 30,
                'lidar_height': 2.0
            },
            'dlo_alignment': {
                'enable': True,
                'min_points': 100,  # Minimum points required for alignment
                'max_iterations': 100,
                'transformation_epsilon': 1e-6,
                'euclidean_fitness_epsilon': 1e-6,
                'max_correspondence_distance': 1.0,
                'max_translation': 5.0  # Maximum allowed translation in meters
            }
        }
        
        # Merge with provided config
        self.config = self._merge_configs(default_config, self.config)
    
    def _merge_configs(self, default: dict, override: dict) -> dict:
        """Recursively merge configuration dictionaries."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _add_odometry_factor(self, current_key: int, current_pose: gtsam.Pose3):
        """Add odometry factor between consecutive poses."""
        if self.last_pose_key is None:
            return
        
        last_pose = self.keyframe_poses[self.pose_count - 1]
        relative_pose = last_pose.inverse().compose(current_pose)
        
        odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(self.config['noise_models']['odometry'])
        )
        self.graph.add(BetweenFactorPose3(self.last_pose_key, current_key, relative_pose, odometry_noise))
    
    def _tensor_to_pose3(self, pose_tensor: torch.Tensor) -> gtsam.Pose3:
        """Convert 4x4 tensor to gtsam.Pose3."""
        pose_np = pose_tensor.cpu().numpy()
        # Since world2robot_pose is world to robot, we need to invert to get robot pose in world
        rotation = gtsam.Rot3(pose_np[:3, :3])
        translation = gtsam.Point3(pose_np[:3, 3])
        return gtsam.Pose3(rotation, translation)
    
    def _pose3_to_tensor(self, pose: gtsam.Pose3) -> torch.Tensor:
        """Convert gtsam.Pose3 to 4x4 tensor."""
        matrix = pose.matrix()
        return torch.from_numpy(matrix).float()
    
    def _perform_optimization(self):
        """Perform ISAM2 optimization and update poses."""
        if self.graph.size() > 0:
            self.isam.update(self.graph, self.initial_estimate)
            self.current_estimate = self.isam.calculateEstimate()
            
            # Update stored poses with optimized values
            self._update_optimized_poses()
            
            # Clear for next iteration
            self.graph = NonlinearFactorGraph()
            self.initial_estimate = Values()
    
    def _update_optimized_poses(self):
        """Update stored poses with optimized values from ISAM2."""
        for i in range(len(self.keyframes)):
            key = gtsam.symbol('x', i)
            if self.current_estimate.exists(key):
                optimized_pose = self.current_estimate.atPose3(key)
                self.keyframe_poses[i] = optimized_pose
                
                # Convert back to world2robot_pose format (inverted)
                pose_tensor = self._pose3_to_tensor(optimized_pose)
                self.keyframes[i].world2robot_pose = torch.linalg.inv(pose_tensor)
    
    def get_keyframes(self) -> List[MinimalKeyframe]:
        """Return all minimal keyframes."""
        return self.keyframes
    
    def get_optimized_trajectory(self) -> List[torch.Tensor]:
        """Return optimized trajectory as list of 4x4 pose matrices."""
        trajectory = []
        for i in range(len(self.keyframes)):
            if i in self.keyframe_poses:
                pose_tensor = self._pose3_to_tensor(self.keyframe_poses[i])
                trajectory.append(pose_tensor)
        return trajectory
    
    def get_loop_closures(self) -> List[Tuple[int, int, gtsam.Pose3]]:
        """Return detected loop closures."""
        return self.loop_closures
    
    def get_dlo_alignment_results(self) -> List[Dict]:
        """Return DLO alignment results."""
        return self.dlo_alignment_results
    
    def optimize(self):
        """Force optimization of the current graph."""
        self._perform_optimization()
    
    def stop_dashboard(self):
        """Stop the dashboard."""
        if self.monitor:
            self.monitor.stop_dashboard()
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return self.monitor.get_dashboard_url() if self.monitor else None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_dashboard()


# Example usage
if __name__ == "__main__":
    # Configuration with loop closure and DLO refinement enabled
    config = {
        'loop_closure': {
            'enable': False,
            'min_keyframes': 10,
            'min_id_distance': 15,  # Minimum distance between keyframe IDs
            'distance_threshold': 1.0
        },
        'scancontext': {
            'position_search_radius': 2.0,
            'min_candidates': 10,
            'max_candidates': 50,
            'time_exclusion_window': 50,
            'lidar_height': 2.5
        },
        'dlo_alignment': {
            'enable': True,
            'min_points': 100,
            'max_iterations': 100,
            'transformation_epsilon': 1e-6,
            'euclidean_fitness_epsilon': 1e-6,
            'max_correspondence_distance': 1.0,
            'max_translation': 5.0  # Maximum allowed translation in meters
        }
    }
    
    # Create grapher - dashboard will start automatically
    dashboard = DashboardAgriGS()
    grapher = GrapherAgriGS(config=config, dashboard=dashboard)
    print(f"Dashboard available at: {grapher.get_dashboard_url()}")
    
    # Keep the process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        grapher.stop_dashboard()
