FROM stanfordvl/omnigibson:latest

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Configure timezone non-interactively
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set workspace to /omnigibson-src (matching the base image)
WORKDIR /omnigibson-src

# Copy and run setup script
COPY set_up_env_docker.sh /omnigibson-src/
RUN chmod +x /omnigibson-src/set_up_env_docker.sh && \
    bash /omnigibson-src/set_up_env_docker.sh

# Source ROS 2 in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set environment variables
ENV ROS_DISTRO=humble

# Working directory will be in /omnigibson-src/Semantics-Map-Omnigibson
WORKDIR /omnigibson-src

CMD ["bash"]