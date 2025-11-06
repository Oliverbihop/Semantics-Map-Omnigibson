#!/bin/bash
source /opt/ros/humble/setup.bash

# Define your Conda environment name
CONDA_ENV="omnigibson"

# Open first terminal → activate conda env → run ros2_publisher_fetch.py
gnome-terminal -- bash -c "
source ~/anaconda3/etc/profile.d/conda.sh;
conda activate $CONDA_ENV;
echo 'Activated conda environment: $CONDA_ENV';
echo 'Starting ros2_publisher_fetch.py...';
python3 ros2_publisher_fetch.py;
exec bash
"& disown

# Wait 10 seconds before launching the second one
sleep 150

# Open second terminal → activate conda env → run unify_process_run.py
gnome-terminal -- bash -c "
source ~/anaconda3/etc/profile.d/conda.sh;
conda activate $CONDA_ENV;
echo 'Activated conda environment: $CONDA_ENV';
echo 'Starting unify_process_run.py...';
python3 unify_process_run.py;
exec bash
"& disown

