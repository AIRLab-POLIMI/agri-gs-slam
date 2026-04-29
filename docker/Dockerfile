# ==============================================================================
# STAGE 1: Heavy base image with all dependencies
# ==============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda-11.8 \
    PATH=/usr/local/cuda-11.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    build-essential \
    cmake \
    ninja-build \
    libopencv-dev \
    libpcl-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libgeographic-dev \
    libceres-dev \
    unzip \
    sudo \
    gdb \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Upgrade pip and install wheel for better package installation
RUN pip3 install --upgrade pip wheel setuptools

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies in logical groups for better caching
# Core scientific computing packages
RUN pip3 install \
    "numpy<2" \
    opencv-python \
    imageio \
    scikit-learn \
    numba \
    pandas \
    pillow

# 3D processing and visualization
RUN pip3 install \
    open3d \
    transforms3d \
    viser \
    rich \
    plotly \
    streamlit

# PyTorch ecosystem packages
RUN pip3 install \
    torchmetrics \
    torchpq \
    tensorboard \
    tensorly

# Computer vision and SLAM packages
RUN pip3 install \
    pycolmap \
    gtsam

# Utilities and CLI tools
RUN pip3 install \
    tqdm \
    tyro \
    splines \
    nerfview \
    pybind11 \
    cupy-cuda11x \
    huggingface_hub

# Install packages from Git repositories (split for better error handling)
# Install pytorch3d separately as it's more complex
# pytorch3d requires --no-build-isolation so its build backend can see the already-installed torch
RUN pip install fvcore iopath && \
    pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d

# Install nerfstudio packages (need --no-build-isolation so torch is visible at build time)
RUN pip install --no-build-isolation \
    git+https://github.com/nerfstudio-project/nerfacc \
    git+https://github.com/nerfstudio-project/gsplat@v1.4.0

# Install remaining git packages (same reason: CUDA extensions require torch at build time)
RUN pip install --no-build-isolation \
    git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5 \
    git+https://github.com/fraunhoferhhi/PLAS/ \
    git+https://github.com/camenduru/simple-knn

# ==============================================================================
# STAGE 2: Lightweight runtime image
# ==============================================================================
FROM base AS runtime

# Accept build arguments for user/group IDs and username
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=mirko

# Create group and user with specified IDs for consistency with host
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install runtime dependencies and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libopencv-dev \
    libpcl-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libgeographic-dev \
    libceres-dev \
    sudo \
    pybind11-dev \
    curl && \
    # Clean up APT caches
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create application directory and set ownership
WORKDIR /agri_gs_slam
RUN chown -R ${USER_NAME}:${USER_NAME} /agri_gs_slam

# Switch to non-root user
USER ${USER_NAME}

# Expose necessary ports
EXPOSE 8080 8500

# Default shell
CMD ["/bin/bash"]
