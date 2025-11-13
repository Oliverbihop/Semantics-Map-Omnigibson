# Semantics-Map-Omnigibson - Docker Setup

Simple Docker setup for Semantics-Map-Omnigibson with ROS 2 Humble.

## 📦 Initial Setup

Clone the repository with sparse checkout:
```bash
# 1️⃣ Clone BEHAVIOR-1K v3.7.0
git clone -b v3.7.0 https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K/OmniGibson/docker

# 2️⃣ Go to a temporary directory and clone only the Semantics-Map-Omnigibson repo
cd /tmp
git clone --depth 1 --filter=blob:none --sparse https://github.com/Oliverbihop/Semantics-Map-Omnigibson.git
cd Semantics-Map-Omnigibson

# 3️⃣ Enable sparse checkout and select only docker_ssmap
git sparse-checkout init --cone
git sparse-checkout set docker_ssmap

# 4️⃣ Copy the folder into your target BEHAVIOR-1K docker directory
cp -r docker_ssmap "$HOME/BEHAVIOR-1K/OmniGibson/docker/"

# 5️⃣ Clean up the temporary clone
cd ..
rm -rf Semantics-Map-Omnigibson

# ✅ Now the folder is available
cd "$HOME/BEHAVIOR-1K/OmniGibson/docker/docker_ssmap"
ls
```

## 📁 Files

- `Dockerfile` - Builds your custom image
- `docker.sh` - Build and run script
- `set_up_env_docker.sh` - Your setup script (place this in same directory)

## 🚀 Quick Start

### Step 1: Build (First Time Only)

```bash
./docker.sh build
```

This takes 10-20 minutes. It will:
- Build from `stanfordvl/omnigibson:latest`
- Install ROS 2 Humble and all dependencies
- Clone Semantics-Map-Omnigibson to `/omnigibson-src/`

### Step 2: Run

```bash
# With GUI
./docker.sh run

# Without GUI (headless)
./docker.sh run --headless

#In this step, we need to accept the "BEHAVIOR DATA BUNDLE END USER LICENSE AGREEMENT"
```

You'll start in: `/omnigibson-src`

### Step 3: Run Main Script 

```bash
# Inside the container
cd /Semantics-Map-Omnigibson
./run_robot_micromamba.sh
```

That's it! 🎉

## 📂 Directory Structure

Inside the container:
```
/omnigibson-src/                      # Main workspace (like the base image)
├── Semantics-Map-Omnigibson/         # Your project (you start here)
└── ...                                # OmniGibson files

/root/semantics_map_ws/                # ROS 2 workspace
└── src/

/opt/ros/humble/                       # ROS 2 installation
```

## 💻 Usage

After running the container, you're automatically in your project directory:

```bash
# You're already here: /omnigibson-src/Semantics-Map-Omnigibson

# Run your main script
./run_robot_micromamba.sh

# Or run other Python scripts
python your_script.py

# ROS 2 is already sourced
ros2 --version

# Access ROS workspace
cd ~/semantics_map_ws
```

## 📝 Commands

```bash
# Build (first time only)
./docker.sh build

# Run with GUI
./docker.sh run

# Run headless
./docker.sh run --headless

# Help
./docker.sh --help
```

## 🔧 What's Installed

✅ **OmniGibson** (from base image)  
✅ **ROS 2 Humble Desktop**  
✅ **Python packages**: numpy, scipy, opencv, torch (CPU), open3d  
✅ **Point Cloud Library (PCL)**  
✅ **fast_gicp** for registration  
✅ **Your project**: Semantics-Map-Omnigibson  

## 🗂️ Data Storage

OmniGibson data is saved to `./omnigibson_data/` in your host directory:
```
omnigibson_data/
├── datasets/           # Mounted to /data
└── isaac-sim/          # Cache, logs, config
```

This persists between container runs!

## 📊 Size Info

- Base OmniGibson: ~8GB
- With ROS 2 + deps: ~12-15GB  
- Total with data: ~20-25GB

Ensure you have enough disk space!

## 🎯 Tips

- First build takes time - be patient! ☕
- Use `--headless` for servers without display
- Data persists in `./omnigibson_data/`
- Project is at `/omnigibson-src/Semantics-Map-Omnigibson`
- ROS 2 is automatically sourced
- Mount your code as volume for live development

## 🔗 Resources

- [OmniGibson Docs](https://behavior.stanford.edu/omnigibson/)
- [ROS 2 Humble Docs](https://docs.ros.org/en/humble/)
- [Docker Docs](https://docs.docker.com/)

---

**Questions?** Open an issue on GitHub!
