from dataloader import DataloaderAgriGS
from tracker import TrackerAgriGS
from mapper import MapperAgriGS
from strategy import StrategyAgriGS
from metrics import MetricsAgriGS
from grapher import GrapherAgriGS
from loss import LossAgriGS
from monitor import MonitorAgriGS, ThreadStatus, ThreadName
from dashboard import DashboardAgriGS
from splatter import SplatterAgriGS
import threading
import queue
import time
import sys
import torch
import argparse
import json
from gsplat.strategy import DefaultStrategy


class PipelineAgriGS:
    def __init__(self, config, modality="gs-slam"):
        """Initialize the AgriGS-SLAM pipeline with configuration"""
        self.config = config
        self.modality = modality

        # DATALOADER
        self.train_dataloader = DataloaderAgriGS(config=self.config["dataloader"], mode=DataloaderAgriGS.Mode.TRAINING)
        self.val_dataloader = DataloaderAgriGS(config=self.config["dataloader"], mode=DataloaderAgriGS.Mode.VALIDATION)

        # SLAM
        self.monitor = MonitorAgriGS(
            config=self.config,
            train_frames=len(self.train_dataloader),
            val_frames=len(self.val_dataloader)
        )
        self.tracker = TrackerAgriGS(config=self.config["slam"]["frontend"])
        self.grapher = GrapherAgriGS(config=self.config["slam"]["backend"], monitor=self.monitor)

        # GAUSSIAN SPLATTING (only if needed)
        if self.modality in ["gs-odom", "gs-slam"]:
            self.mapper = MapperAgriGS(config=self.config["gaussian_splatting"]["mapper"], monitor=self.monitor)
            strategy_name = str(self.config["gaussian_splatting"].get("strategy", "agri-gs")).lower()
            if strategy_name in ("default", "kerbl", "kerbl3dgs", "3dgs"):
                self.strategy = DefaultStrategy()
            elif strategy_name in ("agri-gs", "agrigs", "agri_gs"):
                self.strategy = StrategyAgriGS()
            else:
                raise ValueError(f"Unknown gaussian_splatting.strategy: {strategy_name!r} (expected 'agri-gs' or 'default')")
            self.splatter = SplatterAgriGS(config=self.config["gaussian_splatting"]["splatter"], monitor=self.monitor)
            self.losser = LossAgriGS(config=self.config["gaussian_splatting"]["loss"])
            self.evaluator = MetricsAgriGS()
        else:
            self.mapper = None
            self.strategy = None
            self.splatter = None
            self.losser = None
            self.evaluator = None
        
        # Thread communication
        self.dataloader_queue = queue.Queue()
        self.slam_frontend_queue = queue.Queue()
        self.slam_backend_queue = queue.Queue()
        self.gs_viewer_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.phase_complete_event = threading.Event()  # Signal phase completion
        self.switch_to_validation_event = threading.Event()  # Signal to switch to validation
        
        # Rich monitoring with mapper reference for splat counting (only if mapper exists)
        self.processed_count = {
            "dataloader": 0,
            "frontend": 0,
            "backend": 0,
            "gaussian": 0,
            "viewer": 0
        }

    def reset_processed_counts(self):
        """Reset processed counts for validation phase"""
        self.processed_count = {
            "frontend": 0,
            "backend": 0,
            "gaussian": 0,
            "viewer": 0
        }
 
    def _append_stats(self, keyframe, mode_text, metrics_dict, loss_dict):
        """Append a JSONL row with the per-iteration stats for this keyframe."""
        if self.splatter is None:
            return
        def _to_py(v):
            if hasattr(v, "item"):
                try:
                    return v.item()
                except Exception:
                    pass
            if isinstance(v, dict):
                return {k: _to_py(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_to_py(x) for x in v]
            return v
        row = {
            "mode": mode_text,
            "keyframe_id": getattr(keyframe, "id", None),
            "n_processed": self.processed_count.get("gaussian"),
            "wall_time": time.time(),
            "statistics": _to_py(keyframe.statistics_dict),
            "metrics": _to_py(metrics_dict),
            "loss": _to_py(loss_dict),
        }
        path = f"{self.splatter.stats_dir}/stats.jsonl"
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:
            print(f"⚠ failed to append stats: {e}")

    def slam_frontend_thread(self):
        """SLAM Frontend: Odometry and keyframe extraction with integrated dataloader"""
        thread_name = ThreadName.ODOMETRY.value
        self.monitor.register_thread(thread_name)
        
        try:
            self.monitor.update_thread(thread_name, ThreadStatus.STARTING, "Starting training phase")
            
            # Training phase
            for i, data in enumerate(self.train_dataloader):
                if self.stop_event.is_set():
                    break
                
                # Update status before processing
                self.monitor.update_thread(
                    thread_name, ThreadStatus.PROCESSING,
                    f"Processing training frame #{i + 1}",
                    processed_items=self.processed_count["frontend"],
                    input_queue_size=1,
                    output_queue_size=self.slam_frontend_queue.qsize()
                )
                
                # Track the frame
                keyframe = self.tracker.track(data)
                
                # Always increment processed count after tracking
                self.processed_count["frontend"] += 1
                
                if keyframe is not None:
                    keyframe.trainable = True  # Training mode
                    self.slam_frontend_queue.put(keyframe)
                    
                    # Update after processing keyframe
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.RUNNING,
                        f"Training keyframe #{self.processed_count['frontend']} extracted",
                        processed_items=self.processed_count["frontend"],
                        input_queue_size=0,
                        output_queue_size=self.slam_frontend_queue.qsize()
                    )
                else:
                    # Update after processing even if no keyframe
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.RUNNING,
                        f"No keyframe from training frame #{i + 1}",
                        processed_items=self.processed_count["frontend"],
                        input_queue_size=0,
                        output_queue_size=self.slam_frontend_queue.qsize()
                    )
            
            # Signal switch to validation
            self.switch_to_validation_event.set()
            self.monitor.set_mode("validation")
            self.reset_processed_counts()
            
            self.monitor.update_thread(thread_name, ThreadStatus.STARTING, "Starting validation phase")
            
            # Validation phase
            for i, data in enumerate(self.val_dataloader):
                if self.stop_event.is_set():
                    break
                
                # Update status before processing
                self.monitor.update_thread(
                    thread_name, ThreadStatus.PROCESSING,
                    f"Processing validation frame #{i + 1}",
                    processed_items=self.processed_count["frontend"],
                    input_queue_size=1,
                    output_queue_size=self.slam_frontend_queue.qsize()
                )
                
                # Track the frame
                keyframe = self.tracker.track(data)
                
                # Always increment processed count after tracking
                self.processed_count["frontend"] += 1
                
                if keyframe is not None:
                    keyframe.trainable = False  # Validation mode
                    self.slam_frontend_queue.put(keyframe)
                    
                    # Update after processing keyframe
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.RUNNING,
                        f"Validation keyframe #{self.processed_count['frontend']} extracted",
                        processed_items=self.processed_count["frontend"],
                        input_queue_size=0,
                        output_queue_size=self.slam_frontend_queue.qsize()
                    )
                else:
                    # Update after processing even if no keyframe
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.RUNNING,
                        f"No keyframe from validation frame #{i + 1}",
                        processed_items=self.processed_count["frontend"],
                        input_queue_size=0,
                        output_queue_size=self.slam_frontend_queue.qsize()
                    )
            
            # Signal completion
            self.slam_frontend_queue.put(None)
            self.monitor.update_thread(thread_name, ThreadStatus.STOPPED, "Completed")
            
        except Exception as e:
            self.monitor.update_thread(thread_name, ThreadStatus.ERROR, str(e)[:30])
            raise

    def slam_backend_thread(self):
        """SLAM Backend: Loop closure detection and pose graph optimization"""
        thread_name = ThreadName.SLAM.value
        self.monitor.register_thread(thread_name)
        
        try:
            self.monitor.update_thread(thread_name, ThreadStatus.RUNNING, "Waiting for keyframes")
            
            while not self.stop_event.is_set():
                try:
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.WAITING, 
                        "Waiting for keyframes",
                        processed_items=self.processed_count["backend"],
                        input_queue_size=self.slam_frontend_queue.qsize(),
                        output_queue_size=self.slam_backend_queue.qsize()
                    )
                    
                    keyframe = self.slam_frontend_queue.get(timeout=1.0)
                    
                    if keyframe is None or self.stop_event.is_set():
                        break
                    
                    # Check if we need to reset counts for validation phase
                    if self.switch_to_validation_event.is_set() and keyframe.trainable == False:
                        # First validation keyframe - reset backend count
                        if self.processed_count["backend"] > 0:
                            self.processed_count["backend"] = 0
                    
                    mode_text = "Training" if keyframe.trainable else "Validation"
                    
                    # Full SLAM with loop closure detection
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.PROCESSING, 
                        f"{mode_text} loop closure detection",
                        processed_items=self.processed_count["backend"],
                        input_queue_size=self.slam_frontend_queue.qsize() + 1,
                        output_queue_size=self.slam_backend_queue.qsize()
                    )
                    
                    keyframe = self.grapher.process(keyframe)
                    
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.RUNNING,
                        f"{mode_text} pose optimized",
                        processed_items=self.processed_count["backend"],
                        input_queue_size=self.slam_frontend_queue.qsize(),
                        output_queue_size=self.slam_backend_queue.qsize()
                    )

                    # Send to next stage based on modality
                    if self.modality in ["gs-odom", "gs-slam"]:
                        self.slam_backend_queue.put(keyframe)
                    # For odom/slam without GS, keyframe is processed but not passed further

                    # keyframe.offload_images()
                    # keyframe.clean_splats()
                    # keyframe.to_device(torch.device('cpu'))
                    # del keyframe
                    
                    self.processed_count["backend"] += 1
                    self.slam_frontend_queue.task_done()
                    
                except queue.Empty:
                    continue
            
            # Signal completion for GS thread if it exists
            if self.modality in ["gs-odom", "gs-slam"]:
                self.slam_backend_queue.put(None)
            
            self.monitor.update_thread(thread_name, ThreadStatus.STOPPED, "Completed")
            
        except Exception as e:
            self.monitor.update_thread(thread_name, ThreadStatus.ERROR, str(e)[:30])
            raise

    def gaussian_splatting_thread(self):
        """Gaussian Splatting: 3D reconstruction and rendering"""
        thread_name = ThreadName.GAUSSIAN_SPLATTING.value
        self.monitor.register_thread(thread_name)
        
        try:
            self.monitor.update_thread(thread_name, ThreadStatus.RUNNING, "Waiting Keyframes")
            
            current_keyframe = None
            validation_started = False
            keyframe_processed = False  # Track if current keyframe has been processed
            
            while not self.stop_event.is_set():
                try:
                    # Check for new keyframes with short timeout
                    try:
                        new_keyframe = self.slam_backend_queue.get(timeout=0.1)
                        
                        if new_keyframe is None or self.stop_event.is_set():
                            break
                        
                        # Check if we're switching to validation mode
                        if not validation_started and not new_keyframe.trainable:
                            validation_started = True
                            self.processed_count["gaussian"] = 0
                        
                        # Send previous keyframe to viewer if we have one
                        if current_keyframe is not None:
                            if not self.config.get("viewer", {}).get("enabled", False):
                                # Periodic artefacts: canvas snapshot every 5 keyframes,
                                # checkpoint every 10. Counter is 1-based at this point
                                # (incremented when the keyframe was received).
                                n_done = self.processed_count["gaussian"]
                                if n_done % 5 == 0:
                                    self.splatter.save_canvas(iter=current_keyframe.statistics_dict["N. OPT"])
                                if n_done % 10 == 0:
                                    self.mapper.save_checkpoint(self.splatter.ckpt_dir)
                                current_keyframe.offload_images()
                                current_keyframe.clean_splats()
                                current_keyframe.to_device(torch.device('cpu'))
                                del current_keyframe
                        
                        # Update current keyframe
                        current_keyframe = new_keyframe
                        current_keyframe.to_device(torch.cuda.current_device())
                        current_keyframe = self.mapper.upload_splats(current_keyframe)
                        self.strategy_state = self.strategy.initialize_state()
                        self.strategy.verbose = False
                        keyframe_processed = False  # Reset processing flag for new keyframe
                        
                        self.processed_count["gaussian"] += 1
                    except queue.Empty:
                        # No new keyframe, continue processing current one if available
                        pass
                    
                    # Process current keyframe if we have one
                    if current_keyframe is not None:
                        # In validation mode, only process once per keyframe
                        if not current_keyframe.trainable and keyframe_processed:
                            self.monitor.update_thread(
                                thread_name, ThreadStatus.WAITING,
                                "Waiting for next validation keyframe",
                                processed_items=self.processed_count["gaussian"],
                                input_queue_size=self.slam_backend_queue.qsize(),
                                output_queue_size=self.gs_viewer_queue.qsize()
                            )
                            continue
                        
                        mode_text = "Training" if current_keyframe.trainable else "Validation"
                        self.monitor.update_thread(
                            thread_name, ThreadStatus.PROCESSING, 
                            f"{mode_text} rasterizing...",
                            processed_items=self.processed_count["gaussian"],
                            input_queue_size=self.slam_backend_queue.qsize(),
                            output_queue_size=self.gs_viewer_queue.qsize()
                        )

                        start_optimization = time.time()
                        current_keyframe = self.splatter.rasterize(current_keyframe)

                        if current_keyframe.trainable:
                            # Training mode - full optimization
                            self.strategy.step_pre_backward(
                                params=self.mapper.active_splats,
                                optimizers=self.mapper.optimizers,
                                state=self.strategy_state,
                                step=current_keyframe.statistics_dict["N. OPT"],
                                info=current_keyframe.rasterization_info,
                            )

                            loss, loss_dict = self.losser.compute_loss(current_keyframe)
                            loss.backward()
                            self.mapper.optimize(steps=3)

                            self.strategy.step_post_backward(
                                params=self.mapper.active_splats,
                                optimizers=self.mapper.optimizers,
                                state=self.strategy_state,
                                step=current_keyframe.statistics_dict["N. OPT"],
                                info=current_keyframe.rasterization_info,
                                packed=self.splatter.packed
                            )
                        else:
                            # Validation mode - no optimization, just compute loss
                            with torch.no_grad():
                                _, loss_dict = self.losser.compute_loss(current_keyframe)
                            keyframe_processed = True  # Mark as processed in validation mode
                        
                        end_optimization = time.time()
                        optimization_duration = end_optimization - start_optimization
                        optimization_frequency = 1.0 / optimization_duration if optimization_duration > 0 else float('inf')
                        current_keyframe.statistics_dict["Hz"] = optimization_frequency

                        current_keyframe.statistics_dict["N. OPT"] += 1
                        metrics_dict = self.evaluator.compute(current_keyframe)
                        self.monitor.update_statistics(metrics_dict, loss_dict, current_keyframe.statistics_dict)
                        self._append_stats(current_keyframe, mode_text, metrics_dict, loss_dict)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                        self.monitor.update_thread(
                            thread_name, ThreadStatus.RUNNING,
                            f"{mode_text} #{current_keyframe.statistics_dict['N. OPT']}",
                            processed_items=self.processed_count["gaussian"],
                            input_queue_size=self.slam_backend_queue.qsize(),
                            output_queue_size=self.gs_viewer_queue.qsize()
                        )
                    else:
                        # No keyframe to process, wait
                        self.monitor.update_thread(
                            thread_name, ThreadStatus.WAITING,
                            "Waiting for keyframes",
                            processed_items=self.processed_count["gaussian"],
                            input_queue_size=self.slam_backend_queue.qsize(),
                            output_queue_size=self.gs_viewer_queue.qsize()
                        )
                except Exception as e:
                    self.monitor.update_thread(thread_name, ThreadStatus.ERROR, str(e)[:30])
                    # Continue processing despite errors
                    continue
            
            # Clean up final keyframe if we have one
            if current_keyframe is not None:
                if not self.config.get("viewer", {}).get("enabled", False):
                    # Always flush a final canvas + checkpoint at end-of-run.
                    self.splatter.save_canvas(iter=current_keyframe.statistics_dict["N. OPT"])
                    self.mapper.save_checkpoint(self.splatter.ckpt_dir)
                    current_keyframe.offload_images()
                    current_keyframe.clean_splats()
                    current_keyframe.to_device(torch.device('cpu'))
                    del current_keyframe
            
            # Signal completion
            self.gs_viewer_queue.put(None)
            self.monitor.update_thread(thread_name, ThreadStatus.STOPPED, "Completed")
            
        except Exception as e:
            self.monitor.update_thread(thread_name, ThreadStatus.ERROR, str(e)[:30])
            raise

    def viewer_thread(self):
        """Viewer thread for visualization"""
        thread_name = ThreadName.VIEWER.value
        self.monitor.register_thread(thread_name)
        
        try:
            self.monitor.update_thread(thread_name, ThreadStatus.RUNNING, "Waiting for frames")
            
            while not self.stop_event.is_set():
                try:
                    keyframe = self.gs_viewer_queue.get(timeout=1.0)
                    
                    if keyframe is None or self.stop_event.is_set():
                        break
                    
                    # Process keyframe for viewing
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.PROCESSING,
                        "Rendering view",
                        processed_items=self.processed_count["viewer"],
                        input_queue_size=self.gs_viewer_queue.qsize()
                    )
                    
                    # Add viewer-specific processing here
                    self.processed_count["viewer"] += 1
                    
                    self.monitor.update_thread(
                        thread_name, ThreadStatus.RUNNING,
                        "View updated",
                        processed_items=self.processed_count["viewer"],
                        input_queue_size=self.gs_viewer_queue.qsize()
                    )
                    
                except queue.Empty:
                    continue
            
            self.monitor.update_thread(thread_name, ThreadStatus.STOPPED, "Completed")
            
        except Exception as e:
            self.monitor.update_thread(thread_name, ThreadStatus.ERROR, str(e)[:30])
            raise

    def start(self):
        """Run the pipeline with continuous threads"""
        # Validate modality
        valid_modalities = ["odom", "slam", "gs-odom", "gs-slam"]
        if self.modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{self.modality}'. Must be one of: {valid_modalities}")
        
        # Start Rich monitoring
        self.monitor.start_display()
        
        try:
            # Setup progress bars
            train_frames = len(self.train_dataloader) if hasattr(self.train_dataloader, '__len__') else 0
            val_frames = len(self.val_dataloader) if hasattr(self.val_dataloader, '__len__') else 0
            self.monitor.setup_progress_bars(train_frames, val_frames)
            
            # Set initial mode to training
            self.monitor.set_mode("train")
            
            # Start worker threads based on modality
            threads = []
            
            # SLAM Frontend (always runs)
            frontend_worker = threading.Thread(
                target=self.slam_frontend_thread,
                name="SLAM-Frontend"
            )
            threads.append(frontend_worker)
            
            # SLAM Backend (always runs for visualization)
            backend_worker = threading.Thread(
                target=self.slam_backend_thread,
                name="SLAM-Backend"
            )
            threads.append(backend_worker)
            
            # Gaussian Splatting (only for gs-odom and gs-slam)
            if self.modality in ["gs-odom", "gs-slam"]:
                gs_thread = threading.Thread(
                    target=self.gaussian_splatting_thread,
                    name="Gaussian Splatting"
                )
                threads.append(gs_thread)

            # Viewer thread (only if GS is enabled and viewer is configured)
            if (self.modality in ["gs-odom", "gs-slam"] and 
                self.config.get("viewer", {}).get("enabled", False)):
                viewer_thread = threading.Thread(
                    target=self.viewer_thread,
                    name="Viewer"
                )
                threads.append(viewer_thread)
            
            print(f"🚀 Starting pipeline in '{self.modality}' mode with {len(threads)} threads")
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
                
        except KeyboardInterrupt:
            self.stop()
            for thread in threads:
                thread.join(timeout=2.0)
            
        finally:
            # Stop monitoring
            self.monitor.stop_display()
            
        # Final status
        self.monitor.console.print("\n" + "="*60)
        if self.stop_event.is_set():
            self.monitor.console.print(f"🚜 [yellow]AgriGS-SLAM Pipeline ({self.modality}) stopped by user[/yellow]")
        else:
            self.monitor.console.print(f"🚜 [bold green]AgriGS-SLAM Pipeline ({self.modality}) completed successfully![/bold green]")
            self.monitor.console.print(f"   🔥 Training frames processed: {self.monitor.completed_train}")
            self.monitor.console.print(f"   🧊 Validation frames processed: {self.monitor.completed_val}")
        self.monitor.console.print("="*60)

    def stop(self):
        """Gracefully stop all pipeline components"""
        self.stop_event.set()
        self.phase_complete_event.set()
        self.switch_to_validation_event.set()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AgriGS-SLAM Pipeline')
    parser.add_argument('--odom', action='store_true',
                        help='Run odometry only (no loop closure detection)')
    parser.add_argument('--slam', action='store_true',
                        help='Run SLAM with loop closure detection')
    parser.add_argument('--gs-odom', action='store_true',
                        help='Run odometry with Gaussian Splatting')
    parser.add_argument('--gs-slam', action='store_true',
                        help='Run SLAM with Gaussian Splatting')
    parser.add_argument('--gs-viewer', action='store_true',
                        help='Enable viewer (only works with --gs-odom or --gs-slam)')
    
    return parser.parse_args()


if __name__ == "__main__":
    import yaml

    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Determine modality
        modality_count = sum([args.odom, args.slam, args.gs_odom, args.gs_slam])
        
        if modality_count == 0:
            modality = "gs-slam"  # Default
            print("🔧 No modality specified, using default: gs-slam")
        elif modality_count > 1:
            print("❌ [bold red]Error: Please specify only one modality[/bold red]")
            sys.exit(1)
        else:
            if args.odom:
                modality = "odom"
            elif args.slam:
                modality = "slam"
            elif args.gs_odom:
                modality = "gs-odom"
            elif args.gs_slam:
                modality = "gs-slam"
        
        # Load configuration
        with open("/agri_gs_slam/config/default.yaml", "r") as file:
            config = yaml.safe_load(file)["agrigs_slam"]
        
        # Handle viewer configuration
        if args.gs_viewer:
            if modality not in ["gs-odom", "gs-slam"]:
                print("❌ [bold red]Error: --gs-viewer can only be used with --gs-odom or --gs-slam[/bold red]")
                sys.exit(1)
            else:
                config["viewer"]["enabled"] = True
        
        print(f"🎯 Running in '{modality}' mode")
        
        # Create and run pipeline
        pipeline = PipelineAgriGS(config, modality=modality)
        
        try:
            pipeline.start()
        except KeyboardInterrupt:
            pipeline.stop()
            print("\n👋 [yellow]Pipeline stopped by user[/yellow]")

    except Exception as e:
        print(f"❌ [bold red]Error: {e}[/bold red]")
        sys.exit(1)