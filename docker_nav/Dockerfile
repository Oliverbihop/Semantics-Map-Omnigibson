FROM stanfordvl/omnigibson:latest

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Configure timezone non-interactively
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set workspace to /omnigibson-src (matching the base image)
WORKDIR /omnigibson-src

# Install additional dependencies
RUN apt-get update && apt-get install -y python3-pip libgtk-3-dev libxkbcommon-x11-0 vulkan-tools
RUN python3 -m pip install rerun-sdk==0.22.1

# Copy and run setup script (this will clone and build ros2_nav)
COPY set_up_env_docker.sh /omnigibson-src/
RUN chmod +x /omnigibson-src/set_up_env_docker.sh && \
    bash /omnigibson-src/set_up_env_docker.sh

# Source ROS 2 in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set environment variables
ENV ROS_DISTRO=humble

# Working directory will be in /omnigibson-src
WORKDIR /omnigibson-src/

# Run bash when container starts
CMD ["bash"]
