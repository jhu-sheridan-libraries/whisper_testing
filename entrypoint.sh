#!/bin/bash

# At the start of the script, add:
echo "Environment variables at startup:"
env | grep -i hf_

# Check for cached models
echo "Checking for cached models..."
WHISPER_CACHE="/app/models/whisper"
if [ -d "$WHISPER_CACHE" ]; then
    echo "Found Whisper cache directory:"
    echo "Contents of $WHISPER_CACHE:"
    ls -lh "$WHISPER_CACHE"
    if [ -f "$WHISPER_CACHE/tiny.pt" ]; then
        echo "Found cached tiny model"
    fi
    if [ -f "$WHISPER_CACHE/base.pt" ]; then
        echo "Found cached base model"
    fi
    if [ -f "$WHISPER_CACHE/small.pt" ]; then
        echo "Found cached small model"
    fi
    if [ -f "$WHISPER_CACHE/medium.pt" ]; then
        echo "Found cached medium model"
    fi
    if [ -f "$WHISPER_CACHE/large.pt" ]; then
        echo "Found cached large model"
    fi
else
    echo "No cached Whisper models found. First run will download models."
    mkdir -p "$WHISPER_CACHE"
fi

# Configure Hugging Face credentials
if [ -n "$HF_TOKEN" ]; then
    echo "Configuring Hugging Face credentials..."
    
    # Create necessary directories
    mkdir -p "$HOME/.cache/huggingface"
    mkdir -p "$HOME/.huggingface"
    chmod 700 "$HOME/.cache/huggingface" "$HOME/.huggingface"
    
    # Set token in all possible locations
    echo -n "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
    echo -n "$HF_TOKEN" > "$HOME/.huggingface/token"
    
    # Set environment variables
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export HF_HUB_TOKEN="$HF_TOKEN"
    export HF_TOKEN="$HF_TOKEN"
    export HF_HOME="$HOME/.cache/huggingface"
    
    # Test token
    echo "Testing Hugging Face token..."
    python /app/download_models.py --token-test-only
    TOKEN_TEST_STATUS=$?
    
    if [ $TOKEN_TEST_STATUS -ne 0 ]; then
        echo "WARNING: Token test failed. Continuing anyway, but model downloads may fail."
    else
        echo "Token verified successfully."
    fi
fi

# Run the original command
exec python transcribe_diarize.py "$@" 