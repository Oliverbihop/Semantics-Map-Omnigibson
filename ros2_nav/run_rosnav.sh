#!/bin/bash
export NO_AT_BRIDGE=1
# Terminal 1: Run bridge.py and wait 60s
gnome-terminal -- bash -c "
echo '=== Terminal 1: Bridge ===';
cd src/behavior/;
echo 'Changed directory to: \$(pwd)';
echo 'Starting bridge.py...';
python bridge.py;
exec bash
"

sleep 90

# Terminal 2: Run navigation_launch
gnome-terminal -- bash -c "
echo '=== Terminal 2: Navigation Launch ===';
source install/setup.bash;
echo 'Starting navigation launch...';
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=false map:=/omnigibson-src/ros2_nav/src/behavior/maps/trav_map.yaml;
exec bash
"

# Wait a moment before launching next terminal
sleep 2

# Terminal 3: Run tf_publisher
gnome-terminal -- bash -c "
echo '=== Terminal 3: Navigate TF_Publisher ===';
source install/setup.bash;
echo 'Starting navigate_node...';
ros2 run behavior tf_publisher;
exec bash
"

# Wait a moment before launching next terminal
sleep 2

# Terminal 4: Run navigate_node
gnome-terminal -- bash -c "
echo '=== Terminal 4: Navigate Node===';
source install/setup.bash;
echo 'Starting navigate_node...';
ros2 run behavior navigate_node;
exec bash
"

# Wait a moment before launching next terminal
sleep 2

# Terminal 5: Loop goal_publisher every 5 seconds
gnome-terminal -- bash -c "
echo '=== Terminal 5: Goal Publisher (loops every 5s) ===';
sleep 1;
source install/setup.bash;
echo 'Starting goal_publisher loop...';
while true; do
    echo '--- Publishing goal ---';
    ros2 run behavior goal_publisher 4;
    echo 'Waiting 5 seconds before next publish...';
    sleep 5;
done;
exec bash
"

echo "All terminals launched!"
