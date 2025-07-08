#!/bin/bash

# Print system info
echo "=== System Information ==="
uname -a
echo

# Check if Docker is running
echo "=== Docker Status ==="
docker info | head -n 10
echo

# Check the data directory
echo "=== Data Directory Contents ==="
ls -la ./data
echo

# Check file permissions
echo "=== File Permissions ==="
stat -f "%p %u %g" ./data/processed_input.wav
echo

# Try running a simple command in the container
echo "=== Testing Container Basic Functionality ==="
docker compose run --rm whisper-diarize echo "Container is working"
echo

# Try listing the /data directory in the container
echo "=== Container Data Directory ==="
docker run --rm -v $(pwd)/data:/data whisper_testing-whisper-diarize ls -la /data
echo

# Try running a simple command with the audio file
echo "=== Testing Audio File Access ==="
docker run --rm -v $(pwd)/data:/data whisper_testing-whisper-diarize file /data/processed_input.wav
echo

echo "=== Debug Complete ===" 