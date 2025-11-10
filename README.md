<h1 align="center">Construct Semantic Map on Omnigibson Simulation and Navigation</h1>
<div align="center">

[![Watch the video](https://github.com/Oliverbihop/Semantics-Map-Omnigibson/blob/main/assets/Screenshot%202025-10-15%20162153.png)](https://www.youtube.com/watch?v=19Ay2go7jEI)
<br>
<a href="https://www.youtube.com/watch?v=19Ay2go7jEI">▶️ Watch demo video on YouTube</a>

</div>


<p align="center">
  <img src="https://raw.githubusercontent.com/Oliverbihop/Semantics-Map-Omnigibson/main/assets/map_capture.png" alt="Project Banner" width="800"/>
</p>



<p align="center">
  A semantic mapping framework built on top of Omnigibson and fast_gicp.
</p>

---

## 🚀 Installation Guide

### Prerequissite
* Ubuntu 22.04
* ROS2 Humble
### Docker Instruction
* Please refer to the readme file in docker_nav and docker_ssmap

### 1️⃣ Clone the Repository

```bash
git clone --recursive https://github.com/Oliverbihop/Semantics-Map-Omnigibson.git
cd Semantics-Map-Omnigibson 
```
### 2️⃣ Install GICP (Python bindings only)
```bash
cd submodules/fast_gicp

# Remove any previous build
rm -rf build

# Install Python bindings
python3 setup.py install --user

cd ../..
```
### 3️⃣ Install Omnigibson (BEHAVIOR-1K)
```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
cd BEHAVIOR-1K

# Full installation with environment + dataset
./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval --primitives
```
## 💾 Datasets Bag Files
Google Drive link: https://drive.google.com/drive/folders/1LHEkOGGjcsvM41ybrqrOnuIwgvTVOZE_?usp=sharing

## Run
### Note: You should choose to run with recorded data or realtime-robot.
### Launch the Simulation and Control the Robot (If you want to run the robot in real-time)
```bash
conda activate behavior
python3 ros2_publisher_fetch.py
```
### Download the dataset and unzip (If you want to run the bag file)
```bash
ros2 bag play <Bag_file>.bag
```
### Run SLAM with semantic mapping
```bash
python3 unify_process_run.py
```
## 🤖 Navigation 
Please see in docker_nav
