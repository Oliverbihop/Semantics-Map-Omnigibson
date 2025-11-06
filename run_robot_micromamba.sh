#!/bin/bash
source /opt/ros/humble/setup.bash

# Define your micromamba environment name
export NO_AT_BRIDGE=1
MAMBA_ENV="omnigibson"

# Path to micromamba initialization script
MAMBA_INIT="$HOME/micromamba/etc/profile.d/micromamba.sh"

# Open first terminal → activate micromamba env → run ros2_publisher_fetch.py
gnome-terminal -- bash -c "
source $MAMBA_INIT;
micromamba activate $MAMBA_ENV;
echo 'Activated micromamba environment: $MAMBA_ENV';
echo 'Starting ros2_publisher_fetch.py...';
python3 ros2_publisher_fetch.py;
exec bash
"

# Wait 150 seconds before launching the second one
sleep 60

# Open second terminal → activate micromamba env → run unify_process_run.py
gnome-terminal -- bash -c "
source $MAMBA_INIT;
micromamba activate $MAMBA_ENV;
echo 'Activated micromamba environment: $MAMBA_ENV';
echo 'Starting unify_process_run.py...';
python3 unify_process_run.py;
exec bash
"
