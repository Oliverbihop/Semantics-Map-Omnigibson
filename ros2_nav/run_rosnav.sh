#!/bin/bash

# Terminal 1: Run bridge.py, wait 60s, then run navigation_launch
gnome-terminal -- bash -c "
echo '=== Terminal 1: Bridge and Navigation Launch ===';
source install/setup.bash;
cd /src/behavior/;
echo 'Changed directory to: \$(pwd)';
echo 'Starting bridge.py...';
python3 bridge.py &
BRIDGE_PID=\$!;
echo 'Bridge running with PID: \$BRIDGE_PID';
echo 'Waiting 60 seconds...';
sleep 60;
echo 'Starting navigation launch...';
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=false map:=/omnigibson-src/ros2_nav/src/behavior/maps/trav_map.yaml;
exec bash
"

# Wait a moment before launching next terminal
sleep 2

# Terminal 2: Run first navigate_node
gnome-terminal -- bash -c "
echo '=== Terminal 2: Navigate Node 1 ===';
source install/setup.bash;
echo 'Starting navigate_node...';
ros2 run behavior navigate_node;
exec bash
"

# Wait a moment before launching next terminal
sleep 2

# Terminal 3: Run second navigate_node
gnome-terminal -- bash -c "
echo '=== Terminal 3: Navigate Node 2 ===';
source install/setup.bash;
echo 'Starting navigate_node...';
ros2 run behavior navigate_node;
exec bash
"

# Wait a moment before launching next terminal
sleep 2

# Terminal 4: Wait 1 second, then run goal_publisher once
gnome-terminal -- bash -c "
echo '=== Terminal 4: Goal Publisher (delayed 1s) ===';
sleep 1;
source install/setup.bash;
echo 'Running goal_publisher...';
ros2 run behavior goal_publisher;
exec bash
"

# Terminal 5: Empty terminal for monitoring/debugging
gnome-terminal -- bash -c "
echo '=== Terminal 5: Monitoring Terminal ===';
echo 'This terminal is available for monitoring or additional commands.';
source install/setup.bash;
exec bash
"

echo "All terminals launched!"
