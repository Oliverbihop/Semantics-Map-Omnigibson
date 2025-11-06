#!/usr/bin/env bash
set -e -o pipefail

GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

# Configuration
IMAGE_NAME="semantics-map-omnigibson"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="semantics-map-container"
DATA_DIR="./omnigibson_data"

# Parse arguments
ACTION=""
GUI=true

while [[ $# -gt 0 ]]; do
    case $1 in
        build)
            ACTION="build"
            shift
            ;;
        run)
            ACTION="run"
            shift
            ;;
        -h|--headless)
            GUI=false
            shift
            ;;
        --help)
            echo "Usage: ./docker.sh [build|run] [OPTIONS]"
            echo ""
            echo "Commands:"
            echo "  build             Build the Docker image"
            echo "  run               Run the container"
            echo ""
            echo "Options:"
            echo "  -h, --headless    Run without GUI (for 'run' command)"
            echo "  --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  ./docker.sh build              # Build the image (first time)"
            echo "  ./docker.sh run                # Run with GUI"
            echo "  ./docker.sh run --headless     # Run without GUI"
            echo ""
            echo "After running, your project will be at:"
            echo "  /omnigibson-src/Semantics-Map-Omnigibson"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no action specified, show help
if [ -z "$ACTION" ]; then
    echo "Usage: ./docker.sh [build|run] [OPTIONS]"
    echo "Use --help for more information"
    exit 1
fi

#==========================================
# BUILD COMMAND
#==========================================
if [ "$ACTION" = "build" ]; then
    echo -e "${BLUE}=========================================="
    echo "Building Semantics-Map-Omnigibson Image"
    echo -e "==========================================${NC}"
    echo ""
    
    # Check if setup script exists
    if [ ! -f "set_up_env_docker.sh" ]; then
        echo -e "${YELLOW}Error: set_up_env_docker.sh not found!${NC}"
        echo "Please place the setup script in the current directory"
        exit 1
    fi
    
    echo -e "${GREEN}Building image: ${FULL_IMAGE}${NC}"
    echo "This will take 10-20 minutes..."
    echo ""
    
    # Build the image
    docker build -t ${FULL_IMAGE} -f Dockerfile .
    
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Build Complete!"
    echo -e "==========================================${NC}"
    echo ""
    echo "Image: ${FULL_IMAGE}"
    echo ""
    echo "Your project will be located at:"
    echo "  /omnigibson-src/Semantics-Map-Omnigibson"
    echo ""
    echo "To run the container:"
    echo "  ./docker.sh run              # With GUI"
    echo "  ./docker.sh run --headless   # Without GUI"
    
    exit 0
fi

#==========================================
# RUN COMMAND
#==========================================
if [ "$ACTION" = "run" ]; then
    echo -e "${BLUE}=========================================="
    echo "Running Semantics-Map-Omnigibson"
    echo -e "==========================================${NC}"
    echo ""
    
    # Check if image exists
    if ! docker image inspect ${FULL_IMAGE} >/dev/null 2>&1; then
        echo -e "${YELLOW}Error: Image ${FULL_IMAGE} not found!${NC}"
        echo "Please build the image first:"
        echo "  ./docker.sh build"
        exit 1
    fi
    
    # Get absolute path for data directory
    DATA_PATH=$(cd "$(dirname "${DATA_DIR}")" && pwd)/$(basename "${DATA_DIR}")
    
    # Create data directory if it doesn't exist
    mkdir -p "${DATA_PATH}/datasets"
    mkdir -p "${DATA_PATH}/isaac-sim/cache/kit"
    mkdir -p "${DATA_PATH}/isaac-sim/cache/ov"
    mkdir -p "${DATA_PATH}/isaac-sim/cache/pip"
    mkdir -p "${DATA_PATH}/isaac-sim/cache/glcache"
    mkdir -p "${DATA_PATH}/isaac-sim/cache/computecache"
    mkdir -p "${DATA_PATH}/isaac-sim/logs"
    mkdir -p "${DATA_PATH}/isaac-sim/config"
    mkdir -p "${DATA_PATH}/isaac-sim/data"
    mkdir -p "${DATA_PATH}/isaac-sim/documents"
    
    echo -e "${GREEN}Data will be saved at: ${DATA_PATH}${NC}"
    echo ""
    
    # EULA acceptance
    echo "The NVIDIA Omniverse License Agreement (EULA) must be accepted"
    echo "before Omniverse Kit can start. License terms:"
    echo "https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html"
    echo ""
    
    while true; do
        read -p "Do you accept the Omniverse EULA? [y/n] " yn
        case $yn in
            [Yy]* ) break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
    
    echo ""
    
    # Setup display variables
    DOCKER_DISPLAY=""
    OMNIGIBSON_HEADLESS=1
    
    if [ "$GUI" = true ]; then
        echo -e "${GREEN}Running with GUI enabled${NC}"
        xhost +local:root
        DOCKER_DISPLAY=$DISPLAY
        OMNIGIBSON_HEADLESS=0
    else
        echo -e "${YELLOW}Running in headless mode${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}Starting container...${NC}"
    echo "Container name: ${CONTAINER_NAME}"
    echo "Working directory: /omnigibson-src/Semantics-Map-Omnigibson"
    echo ""
    
    # Run the container
    docker run \
        --gpus all \
        --privileged \
        --name ${CONTAINER_NAME} \
        -e DISPLAY=${DOCKER_DISPLAY} \
        -e OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS} \
        -v ${DATA_PATH}/datasets:/data \
        -v ${DATA_PATH}/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
        -v ${DATA_PATH}/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ${DATA_PATH}/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ${DATA_PATH}/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ${DATA_PATH}/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ${DATA_PATH}/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ${DATA_PATH}/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
        -v ${DATA_PATH}/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ${DATA_PATH}/isaac-sim/documents:/root/Documents:rw \
        --network=host \
        --rm \
        -it \
        ${FULL_IMAGE}
    
    # Cleanup xhost
    if [ "$GUI" = true ]; then
        xhost -local:root
    fi
    
    echo ""
    echo -e "${GREEN}Container stopped${NC}"
fi
