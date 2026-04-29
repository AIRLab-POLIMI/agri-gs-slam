import threading
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
from enum import Enum
import subprocess
import sys
import os
import pickle
import numpy as np

# Rich imports
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.align import Align

# GPU monitoring imports
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

class ThreadName(Enum):
    ODOMETRY = "🧭 Odometry"
    SLAM = "🗺️  SLAM"
    GAUSSIAN_SPLATTING = "🎨 Gaussian Splatting"
    VIEWER = "🌳 Viewer"


class ColorPalette:
    """Centralized color palette for consistent UI styling"""
    
    # Status colors
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "blue"
    ACCENT = "cyan"
    SECONDARY = "magenta"
    
    # UI element colors
    HEADER = "bright_green"
    BORDER = "dim white"
    TEXT_PRIMARY = "white"
    TEXT_SECONDARY = "dim white"
    TEXT_MUTED = "dim"
    
    # Progress and metrics
    TRAIN_COLOR = "dark_orange"
    VAL_COLOR = "cyan"
    PROGRESS_BAR = "blue"
    
    # Resource monitoring
    GPU_LOW = "green"
    GPU_MED = "yellow"
    GPU_HIGH = "red"
    SPLAT_CPU = "cyan"
    SPLAT_GPU = "green"
    SPLAT_TOTAL = "yellow"
    
    # Queue status thresholds
    QUEUE_NORMAL = "white"
    QUEUE_WARNING = "yellow"
    QUEUE_CRITICAL = "red"
    
    # Links and interactive elements
    LINK = "bold dark_orange link"
    HIGHLIGHT = "bold"
    
    # Brand colors
    BRAND_PRIMARY = "green"
    BRAND_ACCENT = "yellow"
    
    @classmethod
    def get_queue_color(cls, size: int) -> str:
        """Get color based on queue size thresholds"""
        if size > 10:
            return cls.QUEUE_CRITICAL
        elif size > 5:
            return cls.QUEUE_WARNING
        return cls.QUEUE_NORMAL
    
    @classmethod
    def get_gpu_color(cls, usage_percent: float) -> str:
        """Get color based on GPU usage percentage"""
        if usage_percent > 90:
            return cls.GPU_HIGH
        elif usage_percent > 70:
            return cls.GPU_MED
        return cls.GPU_LOW


class ThreadStatus(Enum):
    STARTING = ("🔄", f"[{ColorPalette.WARNING}]STARTING[/{ColorPalette.WARNING}]")
    RUNNING = ("✅", f"[{ColorPalette.SUCCESS}]RUNNING[/{ColorPalette.SUCCESS}]")
    WAITING = ("⏳", f"[{ColorPalette.ACCENT}]WAITING[/{ColorPalette.ACCENT}]")
    PROCESSING = ("⚙️", f"[{ColorPalette.INFO}]PROCESSING[/{ColorPalette.INFO}]")
    STOPPED = ("🛑", f"[{ColorPalette.ERROR}]STOPPED[/{ColorPalette.ERROR}]")
    ERROR = ("❌", f"[bold {ColorPalette.ERROR}]ERROR[/bold {ColorPalette.ERROR}]")
    
    def __init__(self, icon, formatted):
        self.icon = icon
        self.formatted = formatted


@dataclass
class ThreadInfo:
    name: str
    status: ThreadStatus
    processed_items: int = 0
    input_queue_size: int = 0
    output_queue_size: int = 0
    last_activity: Optional[datetime] = None
    message: str = ""
    frequency_hz: float = 0.0
    last_update_time: Optional[datetime] = None
    last_processed_count: int = 0


@dataclass
class GPUMemoryInfo:
    used_mb: float = 0.0
    total_mb: float = 0.0
    used_percent: float = 0.0
    free_mb: float = 0.0
    reserved_mb: float = 0.0
    peak_mb: float = 0.0


@dataclass
class SplatInfo:
    cpu_splats: int = 0
    gpu_splats: int = 0
    total_splats: int = 0


@dataclass
class DashboardData:
    """Data structure for dashboard visualization."""
    
    def __init__(self):
        self.trajectory = {'x': [], 'y': [], 'z': [], 'timestamps': [], 'trainable': []}
        self.current_pose = {'x': 0, 'y': 0, 'z': 0}
        self.latest_images = []
        self.rendered_images = []
        self.rendered_depth = []
        self.keyframe_count = 0
        self.loop_closures = []
        self.loop_closure_candidates = []
        self.dlo_alignment_results = []
        self.scancontext_params = {}
        self.last_update = datetime.now()
        self.current_status = "Unknown"


class MonitorAgriGS:
    def __init__(self, config=None, train_frames=None, val_frames=None, enable_dashboard=True):
        self.console = Console()
        self.thread_info: Dict[str, ThreadInfo] = {}
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        self.live = None
        self.running = False
        self.gpu_info = GPUMemoryInfo()
        self.splat_info = SplatInfo()
        self.completed_train = 0
        self.completed_val = 0
        self.current_mode = "train"
        self.config = config if config is not None else {}
        
        # Dashboard components
        self.enable_dashboard = enable_dashboard
        self.dashboard_data = DashboardData()
        self.streamlit_process = None
        self.dashboard_url = None
        
        # Progress tracking
        self.train_frames = train_frames
        self.val_frames = val_frames
        
        # Separate progress bars for each mode
        self.train_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            console=self.console,
            expand=True
        )
        self.val_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            console=self.console,
            expand=True
        )
        
        self.train_progress_task = None
        self.val_progress_task = None
        
        # Metrics dictionaries
        self.train_metrics = {}
        self.val_metrics = {}
        self.train_losses = {}
        self.val_losses = {}
        self.train_kf_stats = {}
        self.val_kf_stats = {}
        
        # Initialize dashboard if enabled
        if self.enable_dashboard:
            self.start_streamlit_dashboard()

    def register_thread(self, name: str):
        with self.lock:
            self.thread_info[name] = ThreadInfo(name, ThreadStatus.STARTING)
            self.thread_info[name].last_update_time = datetime.now()
    
    def setup_progress_bars(self, train_frames: int, val_frames: int):
        """Initialize both progress bars"""
        self.train_frames = train_frames
        self.val_frames = val_frames
        
        self.train_progress_task = self.train_progress.add_task(
            f"[{ColorPalette.TRAIN_COLOR}] WAITING", 
            total=train_frames
        )
        
        self.val_progress_task = self.val_progress.add_task(
            f"[{ColorPalette.VAL_COLOR}] WAITING", 
            total=val_frames
        )
        
        # Save to monitor data file for dashboard
        self._save_monitor_data()
    
    def set_mode(self, mode: str):
        """Set current pipeline mode"""
        self.current_mode = mode
        self.update_dashboard_status(mode)
    
    def update_progress(self, current_frame: int, mode: str = None):
        """Update the appropriate progress bar"""
        if mode is None:
            mode = self.current_mode
            
        if mode == "train" and self.train_progress_task is not None:
            self.completed_train = current_frame
            self.train_progress.update(
                self.train_progress_task, 
                completed=self.completed_train,
                description=f"[{ColorPalette.TRAIN_COLOR}] RUNNING"
            )
        elif mode == "validation" and self.val_progress_task is not None:
            self.completed_val = current_frame
            self.val_progress.update(
                self.val_progress_task, 
                completed=self.completed_val,
                description=f"[{ColorPalette.VAL_COLOR}] RUNNING"
            )
        
        # Save to monitor data file for dashboard
        self._save_monitor_data()
    
    def update_thread(self, name: str, status: ThreadStatus, message: str = "", 
                     processed_items: int = None, input_queue_size: int = None, 
                     output_queue_size: int = None):
        with self.lock:
            if name in self.thread_info:
                info = self.thread_info[name]
                current_time = datetime.now()
                
                # Calculate frequency based on processed items
                if processed_items is not None and info.last_update_time is not None:
                    time_diff = (current_time - info.last_update_time).total_seconds()
                    items_diff = processed_items - info.last_processed_count
                    
                    if time_diff > 0 and items_diff > 0:
                        info.frequency_hz = items_diff / time_diff
                    
                    info.last_processed_count = processed_items
                
                info.status = status
                info.message = message
                info.last_activity = current_time
                info.last_update_time = current_time

                if processed_items is not None:
                    info.processed_items = processed_items
                    if name == ThreadName.ODOMETRY.value:
                        self.update_progress(processed_items, self.current_mode)
                
                if input_queue_size is not None:
                    info.input_queue_size = input_queue_size
                    
                if output_queue_size is not None:
                    info.output_queue_size = output_queue_size
        
        # Save to monitor data file for dashboard
        self._save_monitor_data()

    def update_gpu_memory(self):
        """Update GPU memory information"""
        try:
            if GPU_AVAILABLE:
                device = torch.cuda.current_device()
                mem_allocated = torch.cuda.memory_allocated(device) / (1024**2)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                max_mem_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
                max_mem = torch.cuda.get_device_properties(device).total_memory / (1024**2)
                usage_percentage = (mem_allocated / max_mem) * 100 if max_mem > 0 else 0.0

                self.gpu_info.used_mb = mem_allocated
                self.gpu_info.total_mb = max_mem
                self.gpu_info.free_mb = max_mem - mem_allocated
                self.gpu_info.used_percent = usage_percentage
                self.gpu_info.reserved_mb = mem_reserved
                self.gpu_info.peak_mb = max_mem_allocated
        except Exception:
            pass

    def update_dashboard_splats(self, gpu_splats: int, cpu_splats: int) -> None:
        """Update splat count information"""

        self.splat_info.cpu_splats = cpu_splats
        self.splat_info.gpu_splats = gpu_splats
        self.splat_info.total_splats = self.splat_info.cpu_splats + self.splat_info.gpu_splats
        self._save_monitor_data()

    # Dashboard methods
    def update_dashboard_trajectory(self, trajectory_data: Dict[str, List]) -> None:
        """Update trajectory data for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.trajectory = trajectory_data
        self.dashboard_data.last_update = datetime.now()
        self._save_viz_data()
    
    def update_dashboard_pose(self, pose: Dict[str, float]) -> None:
        """Update current pose for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.current_pose = pose
        self.dashboard_data.last_update = datetime.now()
        self._save_viz_data()
    
    def update_dashboard_images(self, image_data: Dict[str, np.ndarray]) -> None:
        """Update keyframe images, rendered images, and depth maps for dashboard."""
        if not self.enable_dashboard:
            return
        
        # Process images, rendered images, and depth maps in a single loop
        processed_images = []
        processed_rendered_images = []
        processed_rendered_depth = []
        
        num_images = len(image_data.get("image", []))
        
        for i in range(num_images):
            # Process original images
            img_array = image_data["image"][i]
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img_array = img_array[::2, ::2]  # Downsample by 2 times
            processed_images.append(img_array)
            
            # Process rendered images
            rendered_img_array = image_data["rendered_image"][i]
            rendered_img_array = (rendered_img_array * 255).astype(np.uint8)
            rendered_img_array = rendered_img_array[::2, ::2]  # Downsample by 2 times
            processed_rendered_images.append(rendered_img_array)
            
            # Process rendered depth maps
            depth_array = image_data["rendered_depth"][i]
            depth_min = np.min(depth_array)
            depth_max = np.max(depth_array)
            if depth_max > depth_min:
                depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
                depth_normalized = (depth_normalized * 255).astype(np.uint8)
                depth_normalized = depth_normalized[::2, ::2]  # Downsample by 2 times
                processed_rendered_depth.append(depth_normalized)
        
        self.dashboard_data.latest_images = processed_images
        self.dashboard_data.rendered_images = processed_rendered_images
        self.dashboard_data.rendered_depth = processed_rendered_depth

        self._save_viz_data()
    
    def update_dashboard_loop_closures(self, loop_closures: List[Dict]) -> None:
        """Update loop closure data for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.loop_closures = loop_closures
        self._save_viz_data()
    
    def update_dashboard_candidates(self, candidates: List[Dict]) -> None:
        """Update loop closure candidates for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.loop_closure_candidates = candidates
        self._save_viz_data()
    
    def update_dashboard_dlo_results(self, results: List[Dict]) -> None:
        """Update DLO alignment results for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.dlo_alignment_results = results
        self._save_viz_data()
    
    def update_dashboard_scancontext_params(self, params: Dict) -> None:
        """Update ScanContext parameters for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.scancontext_params = params.copy()
        self._save_viz_data()
    
    def update_dashboard_status(self, status: str) -> None:
        """Update current status for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.current_status = status
        self._save_viz_data()
    
    def update_dashboard_keyframe_count(self, count: int) -> None:
        """Update keyframe count for dashboard."""
        if not self.enable_dashboard:
            return
        
        self.dashboard_data.keyframe_count = count
        self._save_viz_data()
    
    def get_updated_scancontext_params(self) -> Optional[Dict]:
        """Load updated ScanContext parameters from dashboard."""
        if not self.enable_dashboard:
            return None
        
        try:
            params_file = '/tmp/grapher_sc_params.pkl'
            if os.path.exists(params_file):
                with open(params_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading ScanContext parameters: {e}")
        
        return None
    
    def _save_monitor_data(self):
        """Save monitor data to temporary file for dashboard."""
        if not self.enable_dashboard:
            return
        
        try:
            # Convert ThreadInfo objects to dictionaries
            thread_data = {}
            for name, info in self.thread_info.items():
                thread_data[name] = {
                    'status': info.status.name,
                    'processed_items': info.processed_items,
                    'input_queue_size': info.input_queue_size,
                    'output_queue_size': info.output_queue_size,
                    'frequency_hz': info.frequency_hz,
                    'message': info.message
                }
            
            # Convert GPU and Splat info to dictionaries
            gpu_data = {
                'used_mb': self.gpu_info.used_mb,
                'total_mb': self.gpu_info.total_mb,
                'used_percent': self.gpu_info.used_percent,
                'free_mb': self.gpu_info.free_mb,
                'reserved_mb': self.gpu_info.reserved_mb,
                'peak_mb': self.gpu_info.peak_mb
            }
            
            splat_data = {
                'cpu_splats': self.splat_info.cpu_splats,
                'gpu_splats': self.splat_info.gpu_splats,
                'total_splats': self.splat_info.total_splats
            }
            
            monitor_data = {
                'thread_info': thread_data,
                'gpu_info': gpu_data,
                'splat_info': splat_data,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_kf_stats': self.train_kf_stats,
                'val_kf_stats': self.val_kf_stats,
                'train_frames': self.train_frames,
                'val_frames': self.val_frames,
                'completed_train': self.completed_train,
                'completed_val': self.completed_val,
                'start_time': self.start_time,
                'config': self.config
            }
            
            data_file = '/tmp/grapher_monitor_data.pkl'
            with open(data_file, 'wb') as f:
                pickle.dump(monitor_data, f)
        except Exception as e:
            print(f"Error saving monitor data: {e}")
    
    def _save_viz_data(self):
        """Save visualization data to temporary file for dashboard."""
        if not self.enable_dashboard:
            return
        
        try:
            viz_data = {
                'trajectory': self.dashboard_data.trajectory,
                'current_pose': self.dashboard_data.current_pose,
                'latest_images': self.dashboard_data.latest_images,
                'rendered_images': self.dashboard_data.rendered_images,
                'rendered_depth': self.dashboard_data.rendered_depth,
                'keyframe_count': self.dashboard_data.keyframe_count,
                'loop_closures': self.dashboard_data.loop_closures,
                'loop_closure_candidates': self.dashboard_data.loop_closure_candidates,
                'dlo_alignment_results': self.dashboard_data.dlo_alignment_results,
                'scancontext_params': self.dashboard_data.scancontext_params,
                'last_update': self.dashboard_data.last_update,
                'current_status': self.dashboard_data.current_status
            }
            
            data_file = '/tmp/grapher_viz_data.pkl'
            with open(data_file, 'wb') as f:
                pickle.dump(viz_data, f)
        except Exception as e:
            print(f"Error saving visualization data: {e}")
    
    def start_streamlit_dashboard(self):
        """Start Streamlit dashboard in a separate process."""
        if not self.enable_dashboard:
            return
        
        try:
            # Create a temporary dashboard file
            dashboard_file = self._create_dashboard_file()
            
            # Start Streamlit server
            port = 8501
            cmd = [
                sys.executable, "-m", "streamlit", "run", 
                dashboard_file, 
                "--server.port", str(port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"
            ]
            
            self.streamlit_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Set dashboard URL
            self.dashboard_url = f"http://localhost:{port}"
            
            # Open browser automatically
            self._open_browser()
            
            print(f"✅ Streamlit dashboard started at: {self.dashboard_url}")
            
        except Exception as e:
            print(f"❌ Failed to start Streamlit dashboard: {e}")
    
    def stop_dashboard(self):
        """Stop the Streamlit dashboard."""
        if self.streamlit_process:
            self.streamlit_process.terminate()
            self.streamlit_process = None
            print("🛑 Streamlit dashboard stopped")
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return self.dashboard_url
    
    def _open_browser(self):
        """Open the dashboard in the default browser."""
        try:
            if self.dashboard_url:
                browser = os.environ.get("BROWSER", "").strip()
                if browser:
                    os.system(f'{browser} {self.dashboard_url}')
        except Exception as e:
            print(f"Could not open browser automatically: {e}")

    def create_header_panel(self):
        """Create the header panel with runtime info"""
        viewer_enabled = self.config.get("viewer", {}).get("enabled", False)
        port = self.config.get("viewer", {}).get("port", 8080) if viewer_enabled else None
        elapsed_time = datetime.now() - self.start_time
        formatted_time = str(elapsed_time).split('.')[0]  # Remove microseconds for cleaner display
        
        segments = [
            ("🚜 AgriGS-SLAM", f"bold {ColorPalette.BRAND_PRIMARY} link {self.dashboard_url}"),
            (" dev by ", f"italic {ColorPalette.TEXT_PRIMARY}"),
            ("Mirko Usuelli", f"italic {ColorPalette.BRAND_PRIMARY}"),
            (" • ⏳ Time Elapsed: ", ColorPalette.TEXT_PRIMARY),
            (formatted_time, f"bold {ColorPalette.BRAND_ACCENT}"),
        ]
        
        if viewer_enabled and port:
            segments.extend([
                (" • 🌳 NeRF Studio: ", ColorPalette.TEXT_PRIMARY),
                (f"Viewer", f"{ColorPalette.LINK} http://localhost:{port}")
            ])

        header_text = Text.assemble(*segments)

        return Panel(
            Align.center(header_text),
            style=ColorPalette.HEADER,
            padding=(0, 1)
        )

    def create_status_table(self):
        """Create the main status table"""
        table = Table(show_header=True, header_style=f"bold {ColorPalette.SECONDARY}", box=None)
        table.add_column("Thread", style=ColorPalette.TEXT_PRIMARY, width=25)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Processed", justify="right", style=ColorPalette.SUCCESS, width=10)
        table.add_column("IN Queue", justify="right", style=ColorPalette.WARNING, width=10)
        table.add_column("OUT Queue", justify="right", style=ColorPalette.WARNING, width=11)
        table.add_column("Freq (Hz)", justify="right", style=ColorPalette.INFO, width=10)
        table.add_column("Message", style=ColorPalette.TEXT_PRIMARY, width=35)

        with self.lock:
            for name, info in self.thread_info.items():
                status_display = info.status.formatted
                processed_display = f"{info.processed_items:,}"
                
                # Apply color coding for queue sizes
                in_queue_display = f"{info.input_queue_size}" if info.input_queue_size > 0 else f"[{ColorPalette.TEXT_MUTED}]-[/{ColorPalette.TEXT_MUTED}]"
                out_queue_display = f"{info.output_queue_size}" if info.output_queue_size > 0 else f"[{ColorPalette.TEXT_MUTED}]-[/{ColorPalette.TEXT_MUTED}]"
                
                if info.input_queue_size > 0:
                    queue_color = ColorPalette.get_queue_color(info.input_queue_size)
                    in_queue_display = f"[{queue_color}]{info.input_queue_size}[/{queue_color}]"
                
                if info.output_queue_size > 0:
                    queue_color = ColorPalette.get_queue_color(info.output_queue_size)
                    out_queue_display = f"[{queue_color}]{info.output_queue_size}[/{queue_color}]"
                
                frequency_display = f"{info.frequency_hz:.2f}" if info.frequency_hz > 0 else f"[{ColorPalette.TEXT_MUTED}]-[/{ColorPalette.TEXT_MUTED}]"
                message_display = info.message[:34] if info.message else f"[{ColorPalette.TEXT_MUTED}]-[/{ColorPalette.TEXT_MUTED}]"
                
                table.add_row(
                    name,
                    status_display,
                    processed_display,
                    in_queue_display,
                    out_queue_display,
                    frequency_display,
                    message_display
                )

        return table

    def create_resource_panel(self):
        """Create GPU memory and splat count panel"""
        self.update_gpu_memory()
        
        # GPU Memory section
        if GPU_AVAILABLE and self.gpu_info.total_mb > 0:
            gpu_color = ColorPalette.get_gpu_color(self.gpu_info.used_percent)
            gpu_text = Text.assemble(
                ("🧠 GPU Memory: ", ColorPalette.TEXT_PRIMARY),
                (f"{self.gpu_info.used_mb:.2f}", f"bold {gpu_color}"),
                ("/", ColorPalette.TEXT_SECONDARY),
                (f"{self.gpu_info.total_mb:.2f} MiB", ColorPalette.TEXT_PRIMARY),
                (" (", ColorPalette.TEXT_SECONDARY),
                (f"{self.gpu_info.used_percent:.2f}%", f"bold {gpu_color}"),
                (")", ColorPalette.TEXT_SECONDARY)
            )
        else:
            gpu_text = Text("🧠 GPU Memory: N/A", style=ColorPalette.TEXT_SECONDARY)
        
        # Splat counts section
        if self.splat_info.total_splats > 0:
            splat_text = Text.assemble(
                ("  •  🫧  Splats: ", ColorPalette.TEXT_PRIMARY),
                (f"{self.splat_info.cpu_splats:,}", f"bold {ColorPalette.SPLAT_CPU}"),
                (" CPU + ", ColorPalette.TEXT_SECONDARY),
                (f"{self.splat_info.gpu_splats:,}", f"bold {ColorPalette.SPLAT_GPU}"),
                (" GPU = ", ColorPalette.TEXT_SECONDARY),
                (f"{self.splat_info.total_splats:,}", f"bold {ColorPalette.SPLAT_TOTAL}"),
                (" total", ColorPalette.TEXT_PRIMARY)
            )
        else:
            splat_text = Text("  •  🫧  Splats: Initializing...", style=ColorPalette.TEXT_SECONDARY)
        
        combined_text = Text.assemble(gpu_text, splat_text)
        
        # Save to monitor data file for dashboard
        self._save_monitor_data()
        
        return Panel(
            Align.center(combined_text),
            title="🛠️  Resources",
            style=ColorPalette.BORDER,
            padding=(0, 1)
        )
    
    def update_statistics(self, metrics_dict: Dict = None, loss_dict: Dict = None, keyframe_stats: Dict = None, mode: str = None):
        """Update statistics (metrics, losses, and keyframe stats) for the specified mode"""
        if mode is None:
            mode = self.current_mode
            
        with self.lock:
            if mode == "train":
                if metrics_dict:
                    self.train_metrics.update(metrics_dict)
                if loss_dict:
                    self.train_losses.update(loss_dict)
                if keyframe_stats:
                    self.train_kf_stats.update(keyframe_stats)
            elif mode == "validation":
                if metrics_dict:
                    self.val_metrics.update(metrics_dict)
                if loss_dict:
                    self.val_losses.update(loss_dict)
                if keyframe_stats:
                    self.val_kf_stats.update(keyframe_stats)
        
        # Save to monitor data file for dashboard
        self._save_monitor_data()

    def create_train_panel(self):
        """Create training panel with progress bar and metrics"""
        # Create metrics table
        metrics_table = Table(show_header=True, box=None, padding=(0, 1))
        metrics_table.add_column("", style=ColorPalette.TRAIN_COLOR)
        metrics_table.add_column("Metric", style=ColorPalette.TEXT_PRIMARY, justify="right")
        
        # Create losses table
        losses_table = Table(show_header=True, box=None, padding=(0, 1))
        losses_table.add_column("", style=ColorPalette.TRAIN_COLOR)
        losses_table.add_column("Loss", style=ColorPalette.TEXT_PRIMARY, justify="right")
        
        # Create keyframe stats table
        kf_table = Table(show_header=True, box=None, padding=(0, 1))
        kf_table.add_column("", style=ColorPalette.TRAIN_COLOR)
        kf_table.add_column("Keyframe", style=ColorPalette.TEXT_PRIMARY, justify="right")
        
        with self.lock:
            # Populate metrics table
            if self.train_metrics:
                for key, value in self.train_metrics.items():
                    if 'loss' not in key.lower():
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                        metrics_table.add_row(key, formatted_value)
            else:
                metrics_table.add_row("Waiting...", f"[{ColorPalette.TEXT_MUTED}]--[/{ColorPalette.TEXT_MUTED}]")

            # Populate losses table
            if self.train_losses:
                for key, value in self.train_losses.items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    losses_table.add_row(key, formatted_value)
            else:
                losses_table.add_row("Waiting...", f"[{ColorPalette.TEXT_MUTED}]--[/{ColorPalette.TEXT_MUTED}]")

            # Populate keyframe stats table
            if self.train_kf_stats:
                for key, value in self.train_kf_stats.items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    kf_table.add_row(key, formatted_value)
            else:
                kf_table.add_row("Waiting...", f"[{ColorPalette.TEXT_MUTED}]--[/{ColorPalette.TEXT_MUTED}]")

        # Create layout with separate sections
        content = Layout()
        tables_layout = Layout()
        tables_layout.split_row(
            Layout(metrics_table, name="metrics"),
            Layout(losses_table, name="losses"),
            Layout(kf_table, name="keyframes")
        )
        
        content.split_column(
            Layout(self.train_progress, size=2),
            Layout(tables_layout)
        )
        
        return Panel(
            content,
            title="🔥 Training",
            style=ColorPalette.TRAIN_COLOR,
            padding=(0, 1)
        )

    def create_val_panel(self):
        """Create validation panel with progress bar and metrics"""
        # Create metrics table
        metrics_table = Table(show_header=True, box=None, padding=(0, 1))
        metrics_table.add_column("", style=ColorPalette.VAL_COLOR)
        metrics_table.add_column("Metric", style=ColorPalette.TEXT_PRIMARY, justify="right")
        
        # Create losses table
        losses_table = Table(show_header=True, box=None, padding=(0, 1))
        losses_table.add_column("", style=ColorPalette.VAL_COLOR)
        losses_table.add_column("Loss", style=ColorPalette.TEXT_PRIMARY, justify="right")
        
        # Create keyframe stats table
        kf_table = Table(show_header=True, box=None, padding=(0, 1))
        kf_table.add_column("", style=ColorPalette.VAL_COLOR)
        kf_table.add_column("Stats", style=ColorPalette.TEXT_PRIMARY, justify="right")
        
        with self.lock:            
            # Populate metrics table
            if self.val_metrics:
                for key, value in self.val_metrics.items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    metrics_table.add_row(key, formatted_value)
            else:
                metrics_table.add_row("Waiting...", f"[{ColorPalette.TEXT_MUTED}]--[/{ColorPalette.TEXT_MUTED}]")
            
            # Populate losses table
            if self.val_losses:
                for key, value in self.val_losses.items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    losses_table.add_row(key, formatted_value)
            else:
                losses_table.add_row("Waiting...", f"[{ColorPalette.TEXT_MUTED}]--[/{ColorPalette.TEXT_MUTED}]")

            # Populate keyframe stats table
            if self.val_kf_stats:
                for key, value in self.val_kf_stats.items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    kf_table.add_row(key, formatted_value)
            else:
                kf_table.add_row("Waiting...", f"[{ColorPalette.TEXT_MUTED}]--[/{ColorPalette.TEXT_MUTED}]")

        # Create layout with separate sections
        content = Layout()
        tables_layout = Layout()
        tables_layout.split_row(
            Layout(metrics_table, name="metrics"),
            Layout(losses_table, name="losses"),
            Layout(kf_table, name="keyframes")
        )
        
        content.split_column(
            Layout(self.val_progress, size=2),
            Layout(tables_layout)
        )
        
        return Panel(
            content,
            title="🧊 Validation",
            style=ColorPalette.VAL_COLOR,
            padding=(0, 1)
        )

    def create_layout(self):
        """Create the complete layout"""
        layout = Layout()
        
        # Calculate dynamic size for bottom layout
        max_train_rows = max(len(self.train_metrics), len(self.train_losses), len(self.train_kf_stats))
        max_val_rows = max(len(self.val_metrics), len(self.val_losses), len(self.val_kf_stats))
        bottom_size = max(3, max_train_rows, max_val_rows)

        bottom_layout = Layout(size=bottom_size)
        bottom_layout.split_row(
            Layout(self.create_train_panel(), name="train"),
            Layout(self.create_val_panel(), name="validation")
        )
        
        layout.split_column(
            Layout(self.create_header_panel(), size=3, name="header"),
            Layout(self.create_status_table(), name="main"),
            Layout(bottom_layout, name="metrics"),
            Layout(self.create_resource_panel(), size=3, name="resources"),
        )
        
        return layout

    def start_display(self):
        """Start the live display"""
        self.running = True
        self.live = Live(
            self.create_layout(),
            console=self.console,
            refresh_per_second=10,
            screen=True
        )
        
        def update_display():
            with self.live:
                while self.running:
                    self.live.update(self.create_layout())
                    time.sleep(0.1)
        
        display_thread = threading.Thread(target=update_display, daemon=True)
        display_thread.start()
        return display_thread

    def stop_display(self):
        """Stop the live display"""
        self.running = False
        if self.live:
            self.live.stop()

    def _create_dashboard_file(self) -> str:
        """Create a temporary dashboard file for Streamlit."""
        dashboard_code = '''
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import pickle
import os
from datetime import datetime
from PIL import Image
import math

# Global data storage
if 'grapher_data' not in st.session_state:
    st.session_state.grapher_data = {
        'trajectory': {'x': [], 'y': [], 'z': [], 'timestamps': [], 'trainable': []},
        'current_pose': {'x': 0, 'y': 0, 'z': 0},
        'latest_images': [],
        'rendered_images': [],
        'rendered_depth': [],
        'keyframe_count': 0,
        'loop_closures': [],
        'loop_closure_candidates': [],
        'dlo_alignment_results': [],
        'scancontext_params': {},
        'last_update': datetime.now(),
        'current_status': 'Unknown',
        'thread_info': {},
        'gpu_info': {},
        'splat_info': {},
        'train_metrics': {},
        'val_metrics': {},
        'train_losses': {},
        'val_losses': {},
        'train_kf_stats': {},
        'val_kf_stats': {},
        'train_frames': 0,
        'val_frames': 0,
        'completed_train': 0,
        'completed_val': 0,
        'start_time': datetime.now(),
        'config': {}
    }

def load_monitor_data():
    """Load monitor data from temporary file."""
    try:
        data_file = '/tmp/grapher_monitor_data.pkl'
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return st.session_state.grapher_data

def load_viz_data():
    """Load visualization data from temporary file."""
    try:
        data_file = '/tmp/grapher_viz_data.pkl'
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return st.session_state.grapher_data

def save_scancontext_params(params):
    """Save ScanContext parameters to temporary file."""
    try:
        params_file = '/tmp/grapher_sc_params.pkl'
        with open(params_file, 'wb') as f:
            pickle.dump(params, f)
    except Exception as e:
        st.error(f"Error saving parameters: {e}")

def create_search_radius_circle(center_x, center_y, radius, num_points=100):
    """Create circle points for search radius visualization."""
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y

def format_elapsed_time(start_time):
    """Format elapsed time."""
    elapsed = datetime.now() - start_time
    return str(elapsed).split('.')[0]

def create_dashboard():
    """Create Streamlit dashboard for visualization."""
    st.set_page_config(
        page_title="AgriGS-SLAM",
        page_icon="🚜",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load data
    monitor_data = load_monitor_data()
    viz_data = load_viz_data()
    
    # Header
    st.title("🚜 AgriGS-SLAM")
    st.set_page_config(layout="wide")
    
    # Header info
    elapsed_time = format_elapsed_time(monitor_data.get('start_time', datetime.now()))
    config = monitor_data.get('config', {})
    
    # Create header with key info
    header_cols = st.columns([3, 1, 1])
    with header_cols[0]:
        st.write(f"🧑🏼‍🌾 **Developer:** Mirko Usuelli   |   ⏳ **Elapsed Time:** {elapsed_time}")
    
    with header_cols[1]:
        pass

    with header_cols[2]:
        viewer_enabled = config.get("viewer", {}).get("enabled", False)
        if viewer_enabled:
            port = config.get("viewer", {}).get("port", 8080)
            st.markdown(f"🌳 **[NeRF Studio](http://localhost:{port})**")
    
    st.divider()
    
    # Sidebar for Settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load current parameters
        sc_params = viz_data.get('scancontext_params', {})
        
        # ScanContext Parameters
        st.subheader("🔍 Parameters")
        
        position_search_radius = st.slider(
            "Search Radius (m)",
            min_value=1.0,
            max_value=50.0,
            value=sc_params.get('position_search_radius', 2.0),
            step=0.5,
            help="Search radius for loop closure candidates"
        )
        
        min_candidates = st.slider(
            "Min Candidates",
            min_value=1,
            max_value=20,
            value=sc_params.get('min_candidates', 5),
            step=1
        )
        
        max_candidates = st.slider(
            "Max Candidates",
            min_value=10,
            max_value=100,
            value=sc_params.get('max_candidates', 30),
            step=5
        )
        
        time_exclusion_window = st.slider(
            "Time Exclusion Window",
            min_value=10,
            max_value=100,
            value=sc_params.get('time_exclusion_window', 30),
            step=5
        )
        
        lidar_height = st.slider(
            "LiDAR Height (m)",
            min_value=0.5,
            max_value=5.0,
            value=sc_params.get('lidar_height', 2.0),
            step=0.1
        )
        
        if st.button("💾 Apply Changes", type="primary"):
            updated_params = {
                'position_search_radius': position_search_radius,
                'min_candidates': min_candidates,
                'max_candidates': max_candidates,
                'time_exclusion_window': time_exclusion_window,
                'lidar_height': lidar_height
            }
            save_scancontext_params(updated_params)
            st.success("✅ Parameters updated!")
        
        st.divider()
        
        # System Status
        st.subheader("🔧 System Status")
        current_status = viz_data.get('current_status', 'Unknown')
        status_color = "🟢" if current_status == "RUNNING" else "🟡"
        st.write(f"{status_color} **Status:** {current_status}")
        
        # Quick Stats
        st.subheader("📊 Quick Stats")
        trajectory_x = viz_data.get('trajectory', {}).get('x', [])
        st.metric("Trajectory Points", len(trajectory_x))
        st.metric("Keyframes", viz_data.get('keyframe_count', 0))
        st.metric("Loop Closures", len(viz_data.get('loop_closures', [])))
    
    # Auto-refresh container
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            # Reload data
            monitor_data = load_monitor_data()
            viz_data = load_viz_data()
            
            # MAIN CONTENT AREA
            main_tabs = st.tabs(["🗺️ Trajectory", "🧵 System Monitor", "📸 Rendering", "📈 Progress"])
            
            # TAB 1: TRAJECTORY
            with main_tabs[0]:
                trajectory_x = viz_data.get('trajectory', {}).get('x', [])
                trajectory_y = viz_data.get('trajectory', {}).get('y', [])
                trajectory_z = viz_data.get('trajectory', {}).get('z', [])
                
                if len(trajectory_x) > 0:
                    # Create trajectory plot
                    fig_2d = go.Figure()
                    
                    # Color by trainable status
                    trainable_data = viz_data.get('trajectory', {}).get('trainable', [True] * len(trajectory_x))
                    
                    # Training points
                    train_x = [x for i, x in enumerate(trajectory_x) if i < len(trainable_data) and trainable_data[i]]
                    train_y = [y for i, y in enumerate(trajectory_y) if i < len(trainable_data) and trainable_data[i]]
                    
                    # Validation points
                    val_x = [x for i, x in enumerate(trajectory_x) if i < len(trainable_data) and not trainable_data[i]]
                    val_y = [y for i, y in enumerate(trajectory_y) if i < len(trainable_data) and not trainable_data[i]]
                    
                    # Add training trajectory
                    if train_x:
                        fig_2d.add_trace(go.Scatter(
                            x=train_x, y=train_y,
                            mode='lines+markers',
                            name='Training Path',
                            line=dict(color='#2E86AB', width=3),
                            marker=dict(color='#2E86AB', size=6)
                        ))
                    
                    # Add validation trajectory
                    if val_x:
                        fig_2d.add_trace(go.Scatter(
                            x=val_x, y=val_y,
                            mode='lines+markers',
                            name='Validation Path',
                            line=dict(color='#F24236', width=3),
                            marker=dict(color='#F24236', size=6)
                        ))
                    
                    # Current position
                    current_pos = viz_data.get('current_pose', {'x': 0, 'y': 0, 'z': 0})
                    current_x, current_y = current_pos['x'], current_pos['y']
                    
                    fig_2d.add_trace(go.Scatter(
                        x=[current_x], y=[current_y],
                        mode='markers',
                        name='Current Position',
                        marker=dict(color='#00C851', size=15, symbol='star')
                    ))
                    
                    # Search radius
                    search_radius = viz_data.get('scancontext_params', {}).get('position_search_radius', 2.0)
                    if search_radius > 0:
                        circle_x, circle_y = create_search_radius_circle(current_x, current_y, search_radius)
                        fig_2d.add_trace(go.Scatter(
                            x=circle_x, y=circle_y,
                            mode='lines',
                            name=f'Search Radius ({search_radius:.1f}m)',
                            line=dict(color='#00C851', width=2, dash='dot'),
                            fill='toself',
                            fillcolor='rgba(0, 200, 81, 0.1)'
                        ))
                    
                    # Loop closures
                    loop_closures = viz_data.get('loop_closures', [])
                    for i, loop in enumerate(loop_closures):
                        from_idx = loop.get('from_id', -1)
                        to_idx = loop.get('to_id', -1)
                        
                        if (0 <= from_idx < len(trajectory_x) and 0 <= to_idx < len(trajectory_x)):
                            fig_2d.add_trace(go.Scatter(
                                x=[trajectory_x[from_idx], trajectory_x[to_idx]],
                                y=[trajectory_y[from_idx], trajectory_y[to_idx]],
                                mode='lines',
                                name='Loop Closure' if i == 0 else '',
                                line=dict(color='#FF6B35', width=3, dash='dash'),
                                showlegend=(i == 0)
                            ))
                    
                    # Loop closure candidates
                    candidates = viz_data.get('loop_closure_candidates', [])
                    if candidates:
                        candidate_x, candidate_y, candidate_text = [], [], []
                        for candidate_data in candidates:
                            if isinstance(candidate_data, dict):
                                candidate_id = candidate_data.get('candidate_id', -1)
                                distance = candidate_data.get('distance', 0.0)
                            else:
                                candidate_id, distance = candidate_data
                            
                            if 0 <= candidate_id < len(trajectory_x):
                                candidate_x.append(trajectory_x[candidate_id])
                                candidate_y.append(trajectory_y[candidate_id])
                                candidate_text.append(f"ID: {candidate_id}, Dist: {distance:.1f}m")
                        
                        if candidate_x:
                            fig_2d.add_trace(go.Scatter(
                                x=candidate_x, y=candidate_y,
                                mode='markers',
                                name='Loop Candidates',
                                marker=dict(color='#A041C1', size=8, symbol='circle-open', line=dict(width=2)),
                                text=candidate_text,
                                hovertemplate='%{text}<extra></extra>'
                            ))
                    
                    fig_2d.update_layout(
                        title="Real-time Trajectory Visualization",
                        xaxis_title="X Position (m)",
                        yaxis_title="Y Position (m)",
                        yaxis=dict(scaleanchor="x", scaleratio=1),
                        height=600,
                        showlegend=True,
                        legend=dict(
                            yanchor="top", y=0.99,
                            xanchor="left", x=0.01,
                            bgcolor="rgba(255,255,255,0.0)"
                        )
                    )
                    
                    st.plotly_chart(fig_2d, use_container_width=True)
                    
                    # Trajectory summary
                    traj_cols = st.columns(4)
                    with traj_cols[0]:
                        distance = np.sqrt(current_x**2 + current_y**2)
                        st.metric("Distance Traveled", f"{distance:.2f} m")
                    with traj_cols[1]:
                        st.metric("Active Candidates", len(candidates))
                    with traj_cols[2]:
                        st.metric("Total Loop Closures", len(loop_closures))
                    with traj_cols[3]:
                        st.metric("Search Radius", f"{search_radius:.1f} m")
                    
                    # Loop closure details
                    if loop_closures:
                        st.subheader("🔄 Loop Closure Details")
                        with st.expander("View Loop Closures", expanded=False):
                            loop_df = pd.DataFrame(loop_closures)
                            st.dataframe(loop_df, use_container_width=True)
                    
                    # DLO alignment results
                    dlo_results = viz_data.get('dlo_alignment_results', [])
                    if dlo_results:
                        st.subheader("🎯 DLO Alignment Results")
                        with st.expander("View Alignment Results", expanded=False):
                            dlo_df = pd.DataFrame(dlo_results)
                            recent_results = dlo_df.tail(10)
                            st.dataframe(recent_results[['source_id', 'target_id', 'has_converged', 'translation_magnitude', 'fitness_score', 'accepted']], use_container_width=True)
                            
                            # DLO statistics
                            dlo_cols = st.columns(4)
                            with dlo_cols[0]:
                                st.metric("Total Alignments", len(dlo_results))
                            with dlo_cols[1]:
                                accepted = len([r for r in dlo_results if r.get('accepted', False)])
                                st.metric("Accepted", accepted)
                            with dlo_cols[2]:
                                rejected = len([r for r in dlo_results if not r.get('accepted', False)])
                                st.metric("Rejected", rejected)
                            with dlo_cols[3]:
                                acceptance_rate = (accepted / len(dlo_results) * 100) if dlo_results else 0
                                st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
                else:
                    st.info("⏳ Waiting for trajectory data...")
            
            # TAB 2: SYSTEM MONITOR
            with main_tabs[1]:
                st.subheader("🧵 Thread Status")
                thread_info = monitor_data.get('thread_info', {})
                
                if thread_info:
                    thread_data = []
                    for name, info in thread_info.items():
                        thread_data.append({
                            'Thread': name,
                            'Status': info.get('status', 'Unknown'),
                            'Processed': f"{info.get('processed_items', 0):,}",
                            'IN Queue': info.get('input_queue_size', 0),
                            'OUT Queue': info.get('output_queue_size', 0),
                            'Frequency': f"{info.get('frequency_hz', 0):.2f} Hz",
                            'Message': info.get('message', '-')[:50] + ('...' if len(info.get('message', '')) > 50 else '')
                        })
                    
                    thread_df = pd.DataFrame(thread_data)
                    st.dataframe(thread_df, use_container_width=True)
                else:
                    st.info("⏳ No thread information available")
                
                st.divider()
                
                # Resource Usage
                st.subheader("🛠️ Resource Usage")
                resource_cols = st.columns(2)
                
                with resource_cols[0]:
                    st.write("**🧠 GPU Memory**")
                    gpu_info = monitor_data.get('gpu_info', {})
                    if gpu_info and gpu_info.get('total_mb', 0) > 0:
                        used_mb = gpu_info.get('used_mb', 0)
                        total_mb = gpu_info.get('total_mb', 0)
                        used_percent = gpu_info.get('used_percent', 0)
                        
                        st.metric("GPU Memory", f"{used_mb:.0f} / {total_mb:.0f} MiB")
                        st.metric("Usage", f"{used_percent:.1f}%")
                        st.metric("Peak Usage", f"{gpu_info.get('peak_mb', 0):.0f} MiB")
                    else:
                        st.info("GPU information not available")
                
                with resource_cols[1]:
                    st.write("**🫧 Gaussian Splats**")
                    splat_info = monitor_data.get('splat_info', {})
                    if splat_info and splat_info.get('total_splats', 0) > 0:
                        cpu_splats = splat_info.get('cpu_splats', 0)
                        gpu_splats = splat_info.get('gpu_splats', 0)
                        total_splats = splat_info.get('total_splats', 0)
                        
                        st.metric("Total Splats", f"{total_splats:,}")
                        st.metric("CPU Splats", f"{cpu_splats:,}")
                        st.metric("GPU Splats", f"{gpu_splats:,}")
                    else:
                        st.info("Splat information not available")
            
            # TAB 3: VISUAL DATA
            with main_tabs[2]:
                st.subheader("📸 Rendering")
                
                # Image columns
                img_cols = st.columns(3)
                
                with img_cols[0]:
                    st.write("**🏞️ Original Images**")
                    latest_images = viz_data.get('latest_images', [])
                    if latest_images:
                        for i, img_array in enumerate(latest_images):
                            if img_array is not None:
                                try:
                                    if img_array.dtype != np.uint8:
                                        img_array = (img_array * 255).astype(np.uint8)
                                    img = Image.fromarray(img_array)
                                    st.image(img, caption=f"Frame {i+1}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying image {i+1}: {e}")
                    else:
                        st.info("⏳ Waiting for original images...")
                
                with img_cols[1]:
                    st.write("**🎨 Rendered Images**")
                    rendered_images = viz_data.get('rendered_images', [])
                    if rendered_images:
                        for i, img_array in enumerate(rendered_images):
                            if img_array is not None:
                                try:
                                    if img_array.dtype != np.uint8:
                                        img_array = (img_array * 255).astype(np.uint8)
                                    img = Image.fromarray(img_array)
                                    st.image(img, caption=f"Rendered {i+1}", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying rendered image {i+1}: {e}")
                    else:
                        st.info("⏳ Waiting for rendered images...")
                
                with img_cols[2]:
                    st.write("**📏 Rendered Depth**")
                    rendered_depth = viz_data.get('rendered_depth', [])
                    if rendered_depth:
                        for i, depth_array in enumerate(rendered_depth):
                            if depth_array is not None:
                                try:
                                    # Normalize depth for visualization
                                    depth_normalized = depth_array.copy()
                                    depth_min = np.min(depth_normalized)
                                    depth_max = np.max(depth_normalized)
                                    
                                    if depth_max > depth_min:
                                        depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min)
                                        depth_normalized = (depth_normalized * 255).astype(np.uint8)
                                        
                                        # Apply colormap
                                        import matplotlib.cm as cm
                                        colored_depth = cm.viridis(depth_normalized / 255.0)
                                        colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)
                                        
                                        img = Image.fromarray(colored_depth)
                                        st.image(img, caption=f"Depth {i+1}", use_container_width=True)
                                    else:
                                        st.warning(f"Invalid depth data for image {i+1}")
                                except Exception as e:
                                    st.error(f"Error displaying depth image {i+1}: {e}")
                    else:
                        st.info("⏳ Waiting for depth images...")
            
            # TAB 4: TRAINING PROGRESS
            with main_tabs[3]:
                st.subheader("📈 Progress")

                # Split screen: Training (left) and Validation (right)
                train_col, val_col = st.columns(2)

                # --- Training Panel ---
                with train_col:
                    st.write("## 🔥 Training")
                    total_train = monitor_data.get("train_frames", 0)
                    done_train = monitor_data.get("completed_train", 0)
                    if total_train:
                        st.write(f"{done_train}/{total_train} frames")
                        st.progress(done_train / total_train)
                    else:
                        st.write("Waiting to start...")
                        st.progress(0)

                    # Training Metrics
                    train_metrics = monitor_data.get("train_metrics", {}) or {}
                    if train_metrics:
                        st.write("**Metrics**")
                        df_train_metrics = pd.DataFrame({
                            'Metric': list(train_metrics.keys()),
                            'Value': [f"{v:.6f}" if isinstance(v, float) else str(v) for v in train_metrics.values()]
                        })
                        st.dataframe(df_train_metrics, use_container_width=True)
                    else:
                        st.info("_No training metrics available_")

                    # Training Losses
                    train_losses = monitor_data.get("train_losses", {}) or {}
                    if train_losses:
                        st.write("**Losses**")
                        df_train_losses = pd.DataFrame({
                            'Loss': list(train_losses.keys()),
                            'Value': [f"{v:.6f}" if isinstance(v, float) else str(v) for v in train_losses.values()]
                        })
                        st.dataframe(df_train_losses, use_container_width=True)
                    else:
                        st.info("_No training losses available_")

                # --- Validation Panel ---
                with val_col:
                    st.write("## 🧊 Validation")
                    total_val = monitor_data.get("val_frames", 0)
                    done_val = monitor_data.get("completed_val", 0)
                    if total_val:
                        st.write(f"{done_val}/{total_val} frames")
                        st.progress(done_val / total_val)
                    else:
                        st.write("Waiting to start...")
                        st.progress(0)

                    # Validation Metrics
                    val_metrics = monitor_data.get("val_metrics", {}) or {}
                    if val_metrics:
                        st.write("**Metrics**")
                        df_val_metrics = pd.DataFrame({
                            'Metric': list(val_metrics.keys()),
                            'Value': [f"{v:.6f}" if isinstance(v, float) else str(v) for v in val_metrics.values()]
                        })
                        st.dataframe(df_val_metrics, use_container_width=True)
                    else:
                        st.info("_No validation metrics available_")

                    # Validation Losses
                    val_losses = monitor_data.get("val_losses", {}) or {}
                    if val_losses:
                        st.write("**Losses**")
                        df_val_losses = pd.DataFrame({
                            'Loss': list(val_losses.keys()),
                            'Value': [f"{v:.6f}" if isinstance(v, float) else str(v) for v in val_losses.values()]
                        })
                        st.dataframe(df_val_losses, use_container_width=True)
                    else:
                        st.info("_No validation losses available_")

        # Refresh rate
        time.sleep(0.5)
        st.rerun()

if __name__ == "__main__":
    create_dashboard()
        '''
            
        # Write to temporary file
        dashboard_file = '/tmp/grapher_dashboard.py'
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_code)
        
        return dashboard_file