#!/bin/bash

# Semantics-Map-Omnigibson Environment Setup Script
# This script installs ROS 2 Humble and all required dependencies
# Excludes: CUDA, Docker images, OmniGibson, and Rerun (as specified)

set -e  # Exit on error

echo "=========================================="
echo "Semantics-Map-Omnigibson Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu
if [ ! -f /etc/os-release ]; then
    print_error "Cannot determine OS. This script is designed for Ubuntu."
    exit 1
fi

. /etc/os-release
if [ "$ID" != "ubuntu" ]; then
    print_error "This script is designed for Ubuntu. Detected: $ID"
    exit 1
fi

print_status "Detected Ubuntu $VERSION_ID"

# Update system packages
print_status "Updating system packages..."
apt-get update

# Install basic dependencies
print_status "Installing basic dependencies..."
apt-get install -y \
    software-properties-common \
    curl \
    gnupg \
    lsb-release \
    wget \
    git \
    build-essential \
    cmake \
    python3-pip \
    python3-dev

#==========================================
# ROS 2 Humble Installation
#==========================================
print_status "Installing ROS 2 Humble..."

# Add ROS 2 repository
print_status "Adding ROS 2 repository..."
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update after adding ROS 2 repo
apt-get update

# Install ROS 2 Humble Desktop (includes rviz, demos, tutorials)
print_status "Installing ROS 2 Humble Desktop..."
apt-get install -y ros-humble-desktop

# Install ROS 2 development tools
print_status "Installing ROS 2 development tools..."
apt-get install -y \
    ros-dev-tools \
    python3-colcon-common-extensions \
    python3-rosdep

# Install ROS 2 packages needed for the code
print_status "Installing ROS 2 packages..."
apt-get install -y \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-tf2-ros \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-message-filters

# Initialize rosdep
print_status "Initializing rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    rosdep init
fi
rosdep update

#==========================================
# Python Dependencies
#==========================================
print_status "Installing Python dependencies..."

# Upgrade pip
print_status "Upgrading pip..."
python3 -m pip install --upgrade pip

# Core Python packages for computer vision and robotics
print_status "Installing numpy, scipy, opencv..."
pip3 install numpy scipy opencv-python opencv-contrib-python

# PyTorch (CPU version as CUDA is excluded)
print_status "Installing PyTorch (CPU version)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Note: ROS 2 Python packages (rclpy, sensor-msgs, etc.) are already installed
# with ros-humble-desktop and should be used from there, not from pip

# Open3D for point cloud processing
print_status "Installing Open3D..."
pip3 install open3d

# fast_gicp for point cloud registration (GICP/FastGICP)
print_status "Installing fast_gicp..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/SMRT-AIST/fast_gicp --recursive
cd fast_gicp
python3 setup.py install --user
cd -
rm -rf "$TEMP_DIR"
print_status "fast_gicp installed successfully"

# Note: message_filters is included in ros-humble-message-filters package

#==========================================
# Additional System Dependencies
#==========================================
print_status "Installing additional system dependencies..."

# OpenCV dependencies
apt-get install -y \
    libopencv-dev \
    python3-opencv

# Eigen3 for linear algebra (used by many robotics libraries)
apt-get install -y \
    libeigen3-dev

# PCL (Point Cloud Library) - optional but useful
print_status "Installing Point Cloud Library (PCL)..."
apt-get install -y \
    libpcl-dev \
    pcl-tools

# GTSAM or similar SLAM libraries (optional)
print_status "Installing additional SLAM dependencies..."
apt-get install -y \
    libboost-all-dev \
    libtbb-dev

#==========================================
# Setup ROS 2 Environment
#==========================================
print_status "Setting up ROS 2 environment..."

# Add ROS 2 sourcing to bashrc if not already present
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# ROS 2 Humble" >> ~/.bashrc
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
    print_status "Added ROS 2 sourcing to ~/.bashrc"
else
    print_warning "ROS 2 already sourced in ~/.bashrc"
fi

# Source ROS 2 for current session
source /opt/ros/humble/setup.bash

#==========================================
# Verify Installation
#==========================================
print_status "Verifying installation..."

echo ""
echo "=========================================="
echo "Checking installed versions:"
echo "=========================================="

# Check ROS 2
if [ -f /opt/ros/humble/setup.bash ]; then
    print_status "ROS 2 Humble: Installed ✓"
    ros2 --version || print_warning "Could not verify ROS 2 version"
else
    print_error "ROS 2 Humble: Not found ✗"
fi

# Check Python packages
echo ""
print_status "Python package versions:"
python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null || print_warning "  numpy: Not installed"
python3 -c "import cv2; print(f'  opencv: {cv2.__version__}')" 2>/dev/null || print_warning "  opencv: Not installed"
python3 -c "import torch; print(f'  torch: {torch.__version__}')" 2>/dev/null || print_warning "  torch: Not installed"
python3 -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>/dev/null || print_warning "  scipy: Not installed"
python3 -c "import open3d; print(f'  open3d: {open3d.__version__}')" 2>/dev/null || print_warning "  open3d: Not installed"
python3 -c "import pygicp; print(f'  pygicp/fast_gicp: Installed')" 2>/dev/null || print_warning "  pygicp/fast_gicp: Not installed"

#==========================================
# Create workspace directory
#==========================================
print_status "Creating workspace directory..."
WORKSPACE_DIR=~/semantics_map_ws
if [ ! -d "$WORKSPACE_DIR" ]; then
    mkdir -p "$WORKSPACE_DIR/src"
    print_status "Created workspace at $WORKSPACE_DIR"
else
    print_warning "Workspace already exists at $WORKSPACE_DIR"
fi
sudo apt-get install gnome-terminal

git clone https://github.com/Oliverbihop/Semantics-Map-Omnigibson.git
cd Semantics-Map-Omnigibson

#==========================================
# Final Instructions
#==========================================
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
print_status "Setup completed successfully!"
