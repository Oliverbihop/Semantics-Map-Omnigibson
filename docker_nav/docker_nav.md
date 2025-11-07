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
