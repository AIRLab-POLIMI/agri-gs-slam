# dashboard_lots.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import pickle
import os
import sys
import subprocess
import threading
from datetime import datetime
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod


class DashboardData:
    """Data structure for dashboard visualization."""
    
    def __init__(self):
        self.trajectory = {'x': [], 'y': [], 'z': [], 'timestamps': [], 'trainable': []}
        self.current_pose = {'x': 0, 'y': 0, 'z': 0}
        self.latest_images = []
        self.keyframe_count = 0
        self.loop_closures = []
        self.loop_closure_candidates = []
        self.dlo_alignment_results = []
        self.scancontext_params = {}
        self.last_update = datetime.now()
        self.current_status = "Unknown"


class DashboardInterface(ABC):
    """Interface for dashboard data updates."""
    
    @abstractmethod
    def update_trajectory(self, trajectory_data: Dict[str, List]) -> None:
        """Update trajectory data."""
        pass
    
    @abstractmethod
    def update_current_pose(self, pose: Dict[str, float]) -> None:
        """Update current pose."""
        pass
    
    @abstractmethod
    def update_keyframe_images(self, images: List[np.ndarray]) -> None:
        """Update keyframe images."""
        pass
    
    @abstractmethod
    def update_loop_closures(self, loop_closures: List[Dict]) -> None:
        """Update loop closure data."""
        pass
    
    @abstractmethod
    def update_loop_closure_candidates(self, candidates: List[Dict]) -> None:
        """Update loop closure candidates."""
        pass
    
    @abstractmethod
    def update_dlo_results(self, results: List[Dict]) -> None:
        """Update DLO alignment results."""
        pass
    
    @abstractmethod
    def update_scancontext_params(self, params: Dict) -> None:
        """Update ScanContext parameters."""
        pass
    
    @abstractmethod
    def update_status(self, status: str) -> None:
        """Update current status."""
        pass


class DashboardAgriGS(DashboardInterface):
    """
    Dashboard visualization system for AgriGS with Streamlit interface.
    Separated from GrapherAgriGS to provide clean interface-based updates.
    """
    
    def __init__(self, enable_dashboard: bool = True):
        """
        Initialize the dashboard.
        
        Args:
            enable_dashboard: Whether to enable dashboard visualization
        """
        self.enable_dashboard = enable_dashboard
        self.data = DashboardData()
        self.streamlit_process = None
        self.dashboard_url = None
        
        if enable_dashboard:
            self.start_streamlit_dashboard()
    
    def update_trajectory(self, trajectory_data: Dict[str, List]) -> None:
        """Update trajectory data."""
        if not self.enable_dashboard:
            return
        
        self.data.trajectory = trajectory_data
        self.data.last_update = datetime.now()
        self._save_viz_data()
    
    def update_current_pose(self, pose: Dict[str, float]) -> None:
        """Update current pose."""
        if not self.enable_dashboard:
            return
        
        self.data.current_pose = pose
        self.data.last_update = datetime.now()
        self._save_viz_data()
    
    def update_keyframe_images(self, images: List[np.ndarray]) -> None:
        """Update keyframe images."""
        if not self.enable_dashboard:
            return
        
        # Process images for display
        processed_images = []
        for img_array in images:
            if img_array is not None:
                # Ensure values are in [0, 1] range, then convert to [0, 255]
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                processed_images.append(img_array)
        
        self.data.latest_images = processed_images
        self._save_viz_data()
    
    def update_loop_closures(self, loop_closures: List[Dict]) -> None:
        """Update loop closure data."""
        if not self.enable_dashboard:
            return
        
        self.data.loop_closures = loop_closures
        self._save_viz_data()
    
    def update_loop_closure_candidates(self, candidates: List[Dict]) -> None:
        """Update loop closure candidates."""
        if not self.enable_dashboard:
            return
        
        self.data.loop_closure_candidates = candidates
        self._save_viz_data()
    
    def update_dlo_results(self, results: List[Dict]) -> None:
        """Update DLO alignment results."""
        if not self.enable_dashboard:
            return
        
        self.data.dlo_alignment_results = results
        self._save_viz_data()
    
    def update_scancontext_params(self, params: Dict) -> None:
        """Update ScanContext parameters."""
        if not self.enable_dashboard:
            return
        
        self.data.scancontext_params = params.copy()
        self._save_viz_data()
    
    def update_status(self, status: str) -> None:
        """Update current status."""
        if not self.enable_dashboard:
            return
        
        self.data.current_status = status
        self._save_viz_data()
    
    def update_keyframe_count(self, count: int) -> None:
        """Update keyframe count."""
        if not self.enable_dashboard:
            return
        
        self.data.keyframe_count = count
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
    
    def _save_viz_data(self):
        """Save visualization data to temporary file for dashboard."""
        if not self.enable_dashboard:
            return
        
        try:
            viz_data = {
                'trajectory': self.data.trajectory,
                'current_pose': self.data.current_pose,
                'latest_images': self.data.latest_images,
                'keyframe_count': self.data.keyframe_count,
                'loop_closures': self.data.loop_closures,
                'loop_closure_candidates': self.data.loop_closure_candidates,
                'dlo_alignment_results': self.data.dlo_alignment_results,
                'scancontext_params': self.data.scancontext_params,
                'last_update': self.data.last_update,
                'current_status': self.data.current_status
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
        'keyframe_count': 0,
        'loop_closures': [],
        'loop_closure_candidates': [],
        'dlo_alignment_results': [],
        'scancontext_params': {},
        'last_update': datetime.now(),
        'current_status': 'Unknown'
    }

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

def create_dashboard():
    """Create Streamlit dashboard for visualization."""
    st.set_page_config(
        page_title="AgriGS-SLAM",
        page_icon="🚜",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚜 AgriGS-SLAM")
    
    # Sidebar for ScanContext parameters
    with st.sidebar:
        st.header("🧑🏼‍🌾 Settings")
        
        # Load current parameters
        viz_data = load_viz_data()
        sc_params = viz_data.get('scancontext_params', {})
        
        # Parameter inputs
        position_search_radius = st.slider(
            "Position Search Radius (m)",
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
            step=1,
            help="Minimum number of candidates for loop closure"
        )
        
        max_candidates = st.slider(
            "Max Candidates",
            min_value=10,
            max_value=100,
            value=sc_params.get('max_candidates', 30),
            step=5,
            help="Maximum number of candidates for loop closure"
        )
        
        time_exclusion_window = st.slider(
            "Time Exclusion Window",
            min_value=10,
            max_value=100,
            value=sc_params.get('time_exclusion_window', 30),
            step=5,
            help="Minimum time gap between keyframes for loop closure"
        )
        
        lidar_height = st.slider(
            "LiDAR Height (m)",
            min_value=0.5,
            max_value=5.0,
            value=sc_params.get('lidar_height', 2.0),
            step=0.1,
            help="Height of LiDAR sensor above ground"
        )
        
        # Save button
        if st.button("💾 Apply Parameters"):
            updated_params = {
                'position_search_radius': position_search_radius,
                'min_candidates': min_candidates,
                'max_candidates': max_candidates,
                'time_exclusion_window': time_exclusion_window,
                'lidar_height': lidar_height
            }
            save_scancontext_params(updated_params)
            st.success("Parameters updated!")
        
        # Display current values
        st.subheader("📊 Current Values")
        st.write(f"Search Radius: {position_search_radius:.1f} m")
        st.write(f"Candidates: {min_candidates}-{max_candidates}")
        st.write(f"Time Window: {time_exclusion_window}")
        st.write(f"LiDAR Height: {lidar_height:.1f} m")
        
        # Display current status
        st.subheader("🔧 Status")
        current_status = viz_data.get('current_status', 'Unknown')
        st.write(f"Mode: {current_status}")
    
    # Auto-refresh every 0.5 seconds
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            # Load latest data
            viz_data = load_viz_data()
            
            # Validate trajectory data
            trajectory_x = viz_data.get('trajectory', {}).get('x', [])
            trajectory_y = viz_data.get('trajectory', {}).get('y', [])
            trajectory_z = viz_data.get('trajectory', {}).get('z', [])
            
            # DLO alignment results
            dlo_results = viz_data.get('dlo_alignment_results', [])
            
            # Metrics row
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("Keyframes", viz_data.get('keyframe_count', 0))
            
            with col2:
                st.metric("Loop Closures", len(viz_data.get('loop_closures', [])))
            
            with col3:
                candidates = len(viz_data.get('loop_closure_candidates', []))
                st.metric("LC Candidates", candidates)
            
            with col4:
                accepted_alignments = len([r for r in dlo_results if r.get('accepted', False)])
                st.metric("DLO Accepted", accepted_alignments)
            
            with col5:
                rejected_alignments = len([r for r in dlo_results if not r.get('accepted', False)])
                st.metric("DLO Rejected", rejected_alignments)
            
            with col6:
                current_pos = viz_data.get('current_pose', {'x': 0, 'y': 0, 'z': 0})
                distance = np.sqrt(current_pos['x']**2 + current_pos['y']**2 + current_pos['z']**2)
                st.metric("Distance", f"{distance:.2f} m")
            
            # Main content in two columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🗺️ Trajectory")
                if len(trajectory_x) > 0:
                    # Create 2D plot
                    fig_2d = go.Figure()
                    
                    # Color trajectory by trainable status
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
                            x=train_x,
                            y=train_y,
                            mode='lines+markers',
                            name='Training',
                            line=dict(color='blue', width=3),
                            marker=dict(color='blue', size=6)
                        ))
                    
                    # Add validation trajectory
                    if val_x:
                        fig_2d.add_trace(go.Scatter(
                            x=val_x,
                            y=val_y,
                            mode='lines+markers',
                            name='Validation',
                            line=dict(color='orange', width=3),
                            marker=dict(color='orange', size=6)
                        ))
                    
                    # Add loop closures
                    loop_closures = viz_data.get('loop_closures', [])
                    for i, loop in enumerate(loop_closures):
                        from_idx = loop.get('from_id', -1)
                        to_idx = loop.get('to_id', -1)
                        
                        # Validate indices
                        if (0 <= from_idx < len(trajectory_x) and 
                            0 <= to_idx < len(trajectory_x)):
                            fig_2d.add_trace(go.Scatter(
                                x=[trajectory_x[from_idx], trajectory_x[to_idx]],
                                y=[trajectory_y[from_idx], trajectory_y[to_idx]],
                                mode='lines',
                                name=f'Loop {from_idx}→{to_idx}',
                                line=dict(color='red', width=3, dash='dash'),
                                showlegend=(i == 0)  # Only show legend for first loop closure
                            ))
                    
                    # Current position
                    current_x = current_pos['x']
                    current_y = current_pos['y']
                    
                    fig_2d.add_trace(go.Scatter(
                        x=[current_x],
                        y=[current_y],
                        mode='markers',
                        name='Current Position',
                        marker=dict(color='green', size=15, symbol='star')
                    ))
                    
                    # Add search radius circle around current position
                    search_radius = viz_data.get('scancontext_params', {}).get('position_search_radius', 2.0)
                    if search_radius > 0:
                        circle_x, circle_y = create_search_radius_circle(current_x, current_y, search_radius)
                        fig_2d.add_trace(go.Scatter(
                            x=circle_x,
                            y=circle_y,
                            mode='lines',
                            name=f'Search Radius ({search_radius:.1f}m)',
                            line=dict(color='lightgreen', width=2, dash='dot'),
                            fill='toself',
                            fillcolor='rgba(144, 238, 144, 0.1)',
                            opacity=0.7
                        ))
                    
                    # Add loop closure candidates with bounds checking
                    candidates = viz_data.get('loop_closure_candidates', [])
                    if candidates:
                        candidate_x = []
                        candidate_y = []
                        candidate_text = []
                        
                        for candidate_data in candidates:
                            if isinstance(candidate_data, dict):
                                candidate_id = candidate_data.get('candidate_id', -1)
                                distance = candidate_data.get('distance', 0.0)
                            else:
                                # Handle old format (tuple)
                                candidate_id, distance = candidate_data
                            
                            # Validate candidate_id bounds
                            if 0 <= candidate_id < len(trajectory_x):
                                candidate_x.append(trajectory_x[candidate_id])
                                candidate_y.append(trajectory_y[candidate_id])
                                candidate_text.append(f"ID: {candidate_id}, Dist: {distance:.1f}m")
                        
                        if candidate_x:
                            fig_2d.add_trace(go.Scatter(
                                x=candidate_x,
                                y=candidate_y,
                                mode='markers',
                                name='LC Candidates',
                                marker=dict(color='purple', size=8, symbol='circle-open', line=dict(width=2)),
                                text=candidate_text,
                                hovertemplate='%{text}<extra></extra>'
                            ))
                    
                    fig_2d.update_layout(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        yaxis=dict(scaleanchor="x", scaleratio=1),
                        height=700,
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                    
                    # Statistics
                    col1a, col2a, col3a = st.columns(3)
                    with col1a:
                        st.metric("Total Trajectory Points", len(trajectory_x))
                    with col2a:
                        st.metric("Active Candidates", len(candidates))
                    with col3a:
                        st.metric("Search Radius", f"{search_radius:.1f} m")
                    
                    # Loop closure summary
                    if loop_closures:
                        st.subheader("🔄 Loop Closures")
                        loop_df = pd.DataFrame(loop_closures)
                        st.dataframe(loop_df, use_container_width=True)
                    
                    # DLO alignment results
                    if dlo_results:
                        st.subheader("🎯 DLO Alignment Results")
                        dlo_df = pd.DataFrame(dlo_results)
                        
                        # Display recent results
                        recent_results = dlo_df.tail(10)
                        st.dataframe(recent_results[['source_id', 'target_id', 'has_converged', 'translation_magnitude', 'fitness_score', 'accepted']], use_container_width=True)
                        
                        # Statistics
                        col1b, col2b, col3b, col4b = st.columns(4)
                        with col1b:
                            st.metric("Total Alignments", len(dlo_results))
                        with col2b:
                            st.metric("Accepted", len([r for r in dlo_results if r.get('accepted', False)]))
                        with col3b:
                            st.metric("Rejected", len([r for r in dlo_results if not r.get('accepted', False)]))
                        with col4b:
                            acceptance_rate = (len([r for r in dlo_results if r.get('accepted', False)]) / len(dlo_results) * 100) if dlo_results else 0
                            st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
                    
                    # Candidates summary
                    if candidates:
                        st.subheader("🎯 Current Loop Closure Candidates")
                        candidates_data = []
                        for candidate_data in candidates:
                            if isinstance(candidate_data, dict):
                                candidates_data.append([
                                    candidate_data.get('candidate_id', -1),
                                    candidate_data.get('distance', 0.0)
                                ])
                            else:
                                # Handle old format (tuple)
                                candidates_data.append(list(candidate_data))
                        
                        if candidates_data:
                            candidates_df = pd.DataFrame(candidates_data, columns=['Keyframe ID', 'Distance (m)'])
                            candidates_df = candidates_df.sort_values('Distance (m)')
                            st.dataframe(candidates_df, use_container_width=True)
                        
                else:
                    st.info("Waiting for trajectory data...")
            
            with col2:
                st.subheader("📸 Keyframe")
                latest_images = viz_data.get('latest_images', [])
                if len(latest_images) > 0:
                    # Display latest images
                    for i, img_array in enumerate(latest_images):
                        if img_array is not None:
                            try:
                                # Convert numpy array to PIL Image
                                if img_array.dtype != np.uint8:
                                    img_array = (img_array * 255).astype(np.uint8)
                                
                                img = Image.fromarray(img_array)
                                st.image(img, caption=f"Image {i+1}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image {i+1}: {e}")
                else:
                    st.info("Waiting for keyframe images...")
        
        # Refresh every 0.5 seconds
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
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_dashboard()
