#!/bin/bash

# At the start of the script, add:
echo "Environment variables at startup:"
env | grep -i hf_

# Debug information
if [ "$1" == "--debug" ]; then
    echo "=== Debug Mode ==="
    echo "User: $(whoami)"
    echo "Current directory: $(pwd)"
    echo "Contents of /app:"
    ls -la /app
    echo "Contents of /app/models:"
    ls -la /app/models
    exit 0
fi

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
    mkdir -p "$WHISPER_CACHE" || echo "Warning: Could not create $WHISPER_CACHE directory"
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

# Check if the first argument is a direct command
if [[ "$1" == "ls" || "$1" == "file" || "$1" == "echo" || "$1" == "cat" || "$1" == "find" ]]; then
    # Execute the command directly
    exec "$@"
elif [[ "$1" == "--" ]]; then
    # If -- is provided, shift it out and execute the command directly
    shift
    exec "$@"
elif [[ "$1" == "--shell" ]]; then
    # If --shell is provided, start a shell
    shift
    exec /bin/bash
else
    # Run the transcribe_diarize.py script with all arguments
    exec python /app/transcribe_diarize.py "$@"
fi 