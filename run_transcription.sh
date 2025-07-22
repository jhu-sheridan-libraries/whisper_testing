#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <audio_filename> [--model <model_name>] [--format <output_format>]"
  echo "Example: $0 data/Trump_Wants_a_3rd_Term.mp3 --model medium --format vtt"
  exit 1
fi

AUDIO_FILE="$1"
shift

# Set default parameters
MODEL="medium"
FORMAT="vtt"

# Parse optional arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift ;;
    --format) FORMAT="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Get the filename only
FILENAME=$(basename "$AUDIO_FILE")
BASENAME="${FILENAME%.*}"
OUTPUT_FILE="${BASENAME}.${FORMAT}"

# First make sure the image is built
echo "Building Docker image..."
docker compose build

# Create a temporary Dockerfile for processing
TEMP_DIR="$(mktemp -d)"
mkdir -p "$TEMP_DIR/input"

echo "Creating temporary image with the audio file..."
cp "$AUDIO_FILE" "$TEMP_DIR/input/$FILENAME"

# Create a simple Dockerfile that extends the main one
cat > "$TEMP_DIR/Dockerfile" << EOF
FROM whisper_testing-whisper-diarize

WORKDIR /app
COPY input/$FILENAME /app/$FILENAME

# When container starts, process the audio file
CMD ["/app/$FILENAME", "--model", "$MODEL", "--format", "$FORMAT", "--output", "/app/$OUTPUT_FILE"]
EOF

# Build the temporary image
docker build -t whisper-process-image "$TEMP_DIR"

echo "Running transcription for: $FILENAME"
echo "Using model: $MODEL, format: $FORMAT"

# Run the container and capture the output
docker run --rm -it whisper-process-image

# Create a container just to extract the output file
CONTAINER_ID=$(docker create whisper-process-image)
mkdir -p output
docker cp "$CONTAINER_ID:/app/$OUTPUT_FILE" "output/$OUTPUT_FILE" || echo "Failed to copy output file from /app/$OUTPUT_FILE"
docker rm "$CONTAINER_ID"

# Clean up
rm -rf "$TEMP_DIR"

echo "Transcription complete. Check output directory for results." 