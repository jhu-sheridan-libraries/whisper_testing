#!/bin/bash

# Create directory structure
mkdir -p src include third_party models build bin lib

# Download dr_wav
echo "Downloading dr_wav.h..."
curl -L https://raw.githubusercontent.com/mackron/dr_libs/master/dr_wav.h -o third_party/dr_wav.h

# Clone whisper.cpp if not already present
if [ ! -d "whisper.cpp" ]; then
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggerganov/whisper.cpp.git
fi

# Create models symlink
if [ ! -L "models" ] && [ ! -d "models" ]; then
    ln -s whisper.cpp/models models
fi

echo "Setup complete!" 