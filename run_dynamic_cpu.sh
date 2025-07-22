#!/bin/bash
# This script dynamically calculates 70% of available CPU cores and
# allocates them to the whisper-diarize container.

# Check for nproc or getconf to find the number of cores
if command -v nproc &> /dev/null; then
    CORES=$(nproc)
elif command -v getconf &> /dev/null; then
    CORES=$(getconf _NPROCESSORS_ONLN)
else
    echo "Warning: Could not determine number of CPU cores. Defaulting to 1."
    CORES=1
fi

echo "Host has $CORES CPU cores."

# Calculate 70% of the cores using bc (or awk as a fallback)
if command -v bc &> /dev/null; then
    CPU_LIMIT=$(echo "$CORES * 0.7" | bc)
else
    echo "Warning: 'bc' command not found. Using 'awk' for calculation."
    CPU_LIMIT=$(awk -v cores="$CORES" 'BEGIN {printf "%.2f", cores * 0.7}')
fi

echo "Setting CPU limit for Docker to $CPU_LIMIT cores."

# Export the variable so docker-compose can use it
export CPU_LIMIT

# Edit the .env file to replace the CPU_LIMIT variable with the calculated value
sed -i "s/^CPU_LIMIT=.*/CPU_LIMIT=$CPU_LIMIT/" .env

# List the GPU devices if there are any
if nvidia-smi &> /dev/null; then
    echo "GPU devices:"
    nvidia-smi
    # If there is a GPU, set the GPU_LIMIT to 1
    GPU_LIMIT=1
    sed -i "s/^GPU_LIMIT=.*/GPU_LIMIT=$GPU_LIMIT/" .env

    # Replace ARG WHISPER_VER=latest with ARG WHISPER_VER=gpu
    sed -i "s/^ARG WHISPER_VER=latest/ARG WHISPER_VER=gpu/" Dockerfile
fi

# Run the docker-compose command, passing all script arguments to it
# docker compose run --rm whisper-diarize "$@"
docker compose build --no-cache whisper-diarize