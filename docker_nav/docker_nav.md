# Object Navigation Omnigibson - Docker Setup

Simple Docker setup for Semantics-Map-Omnigibson Navigation with ROS 2 Humble.


## System Prerequisites
- Ubuntu 22.04 (host machine)
- NVIDIA GPU with CUDA support
- Docker installed
- NVIDIA Docker runtime installed
- At least 28GB free disk space

## 📦 Initial Setup
Clone the repository with sparse checkout:

```bash
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K/OmniGibson/
git remote add origin https://github.com/Oliverbihop/Semantics-Map-Omnigibson.git
git sparse-checkout init --cone
git sparse-checkout set docker_nav
git pull origin main
cd docker_ssmap
```
## Implementation

### 1. Build the Docker Image

First time setup (takes 10-20 minutes):

```bash
chmod +x docker.sh
./docker.sh build
```

This will:
- Build the Docker image with all dependencies
- Install ROS 2 Humble
- Clone and build the ros2_nav workspace
- Set up the environment

### 2. Run the Container

**With GUI (default)**:
```bash
./docker.sh run
```

**Headless mode** (no GUI):
```bash
./docker.sh run --headless
```

### 3. Inside the Container

Once inside the container, you'll be in the `/omnigibson-src` directory with the following structure:

```
/omnigibson-src/
├── ros2_nav/              # ROS 2 workspace
│   ├── src/
│   │   └── behavior/      # Navigation behavior package
│   ├── install/           # Built packages
│   └── build/
└── omnigibson_data/       # Persistent data (mounted from host)
```
