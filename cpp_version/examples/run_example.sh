#!/bin/bash

# Run from the cpp_version directory
# bash ./examples/run_example.sh

CURRENT_DIR=$(pwd)
PARENT_DIR=$(dirname $CURRENT_DIR)

# Find first mp3 file in data directory
MP3_FILE=$(find $PARENT_DIR/data/ -name "*.mp3" -print -quit)
MODEL_NAME="tiny"

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        --file)
        MP3_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        --model)
        MODEL_NAME="$2"
        shift # past argument
        shift # past value
        ;;
        --techniques)
        # Collect all techniques until the next flag or end of arguments
        TECHNIQUES=()
        shift # past --techniques
        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
            TECHNIQUES+=("$1")
            shift
        done
        ;;
        --speakers)
        NUM_SPEAKERS="$2"
        shift # past argument
        shift # past value
        ;;
        --help)
        echo "Usage: $0 [options]"
        echo "  --model: tiny, base, small, medium, large"
        echo "  --file: specify the MP3 file to transcribe"
        echo "  --speakers: number of speakers or 'auto' for automatic detection"
        echo "  --techniques: space-separated list of audio preprocessing techniques:"
        echo "      noise_reduction volume_normalization dynamic_range_compression"
        echo "      high_pass_filtering de_essing combine"
        echo "      sample_rate_standardization audio_channel_management"
        echo "Example: $0 --file audio.mp3 --model tiny --techniques noise_reduction high_pass_filtering"
        exit 0
        ;;
        *)    # unknown option
        MODEL_NAME="$1"
        shift # past argument
        ;;
    esac
done

if [ -f "run_example.sh" ]; then
    cd ..
fi

# Download tiny model if not exists
if [ ! -f "whisper.cpp/models/ggml-${MODEL_NAME}.bin" ]; then
    echo "Downloading ${MODEL_NAME} model..."
    cd whisper.cpp
    bash ./models/download-ggml-model.sh ${MODEL_NAME}
    cd ..
fi

# Get absolute path
MP3_FILE=$(realpath $MP3_FILE)

if [ -z "$MP3_FILE" ]; then
    echo "Error: No MP3 files found in ./data directory"
    exit 1
fi

echo "Using audio file: $MP3_FILE"

# Create temp directory if it doesn't exist
TEMP_DIR="$PARENT_DIR/temp"
mkdir -p "$TEMP_DIR"

# Convert audio to correct format (16kHz mono WAV)
TEMP_WAV="$TEMP_DIR/processed_input.wav"

# If no techniques specified, use default audio_channel_management
if [ ${#TECHNIQUES[@]} -eq 0 ]; then
    TECHNIQUES=("audio_channel_management")
fi

echo "Applying audio preprocessing techniques: ${TECHNIQUES[*]}"
# Pass all techniques to preprocess_audio.sh
bash ../preprocess_audio.sh "$MP3_FILE" "${TECHNIQUES[@]}"

# Check if the file exists
if [ ! -f "$TEMP_WAV" ]; then
    echo "Error: WAV file not found"
    exit 1
fi

# Verify the WAV file
echo "Verifying WAV file..."
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$TEMP_WAV"
file "$TEMP_WAV"
ls -l "$TEMP_WAV"

# Output file name is the same as the mp3 file but with .vtt extension
OUTPUT_FILE="${MP3_FILE%.mp3}_cpp_${MODEL_NAME}.vtt"

# Create models directory symlink if it doesn't exist
if [ ! -L "models" ] && [ ! -d "models" ]; then
    ln -s whisper.cpp/models models
fi

# Get absolute path to model
MODEL_PATH=$(realpath whisper.cpp/models/ggml-$MODEL_NAME.bin)
echo "Using model file: $MODEL_PATH"

# If the file doesn't exist, download it
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model..."
    cd whisper.cpp
    bash ./models/download-ggml-model.sh $MODEL_NAME
    cd ..
fi

# Before running transcription, handle speaker detection
if [ -z "$NUM_SPEAKERS" ]; then
    # Default to 1 speaker if not specified
    NUM_SPEAKERS="1"
fi

if [ "$NUM_SPEAKERS" = "auto" ]; then
    echo "Counting speakers in audio file..."
    NUM_SPEAKERS=$(pipenv run python ${PARENT_DIR}/count_speakers_in_audio.py "$TEMP_WAV")
    echo "Number of speakers: $NUM_SPEAKERS"
fi

# Validate that NUM_SPEAKERS is a positive integer
if ! [[ "$NUM_SPEAKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of speakers must be a positive integer. Got: $NUM_SPEAKERS"
    echo "Defaulting to 1 speaker."
    NUM_SPEAKERS="1"
fi

# Run transcription with speaker count
echo "Running transcription with $NUM_SPEAKERS speakers..."
# Create a temporary file for stats
STATS_FILE=$(mktemp)

# Update whisper_diarize command to include speaker count
./build/whisper_diarize \
    -a "$TEMP_WAV" \
    -o "$OUTPUT_FILE" \
    -m "$MODEL_NAME" \
    -f vtt \
    -s "$NUM_SPEAKERS" &
WHISPER_PID=$!

# Monitor CPU and memory usage every second while whisper_diarize runs
while kill -0 $WHISPER_PID 2>/dev/null; do
    # Get CPU usage (both percentage and number of cores)
    CPU_STATS=$(ps -p $WHISPER_PID -o %cpu=,nlwp=)
    CPU_PERCENT=$(echo $CPU_STATS | awk '{print $1}')
    CPU_CORES=$(echo $CPU_STATS | awk '{print $2}')
    
    # Get memory percentage
    MEM_PERCENT=$(ps -p $WHISPER_PID -o %mem=)
    
    echo "$CPU_PERCENT,$CPU_CORES,$MEM_PERCENT" >> $STATS_FILE
    sleep 1
done

# Calculate average CPU and memory usage
AVG_STATS=$(awk -F',' '{ 
    cpu_sum+=$1; 
    cores_sum+=$2;
    mem_sum+=$3; 
    count++ 
} END { 
    printf "%.1f,%.0f,%.1f", 
    cpu_sum/count, 
    cores_sum/count, 
    mem_sum/count 
}' $STATS_FILE)

# Parse the averages
CPU_PERCENT=$(echo $AVG_STATS | cut -d',' -f1)
CPU_CORES=$(echo $AVG_STATS | cut -d',' -f2)
MEM_PERCENT=$(echo $AVG_STATS | cut -d',' -f3)

# Clean up stats file
rm $STATS_FILE

wait $WHISPER_PID
PROCESSING_TIME=$SECONDS

# Clean up temp files
rm -f "$TEMP_WAV"

# If output file is empty, exit
if [ ! -s "$OUTPUT_FILE" ]; then
    echo "Error: Output file is empty"
    exit 1
else
    # If the file's content is only one line, exit
    if [ $(wc -l < "$OUTPUT_FILE") -eq 2 ]; then
        echo "Error: Output file is only one line"
        exit 1
    else
        echo -e "\nTranscription output:"
        head -n 10 $OUTPUT_FILE
    fi
fi

echo -e "\nRunning fix_benchmark.sh to ensure benchmark file is in the correct format"
bash ./fix_benchmark.sh

# Record benchmark data directly
if [ -f "$OUTPUT_FILE" ]; then
    # Get audio duration in seconds
    DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$MP3_FILE")
    # Round to 1 decimal place
    DURATION=$(printf "%.1f" "$DURATION")
    
    # Get just the filename
    FILENAME=$(basename "$MP3_FILE")
    
    # Calculate real-time factor
    RTF=$(echo "$PROCESSING_TIME / $DURATION" | bc -l)
    RTF=$(printf "%.3f" "$RTF")
    
    echo "Adding benchmark entry:"
    echo "  File: $FILENAME"
    echo "  Duration: $DURATION seconds"
    echo "  Model: $MODEL_NAME"
    echo "  Processing time: $PROCESSING_TIME seconds"
    echo "  RTF: $RTF"
    
    # Add entry directly to benchmark file
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$TIMESTAMP,$FILENAME,$DURATION,$NUM_SPEAKERS,$MODEL_NAME,$PROCESSING_TIME,$RTF,$CPU_PERCENT,$CPU_CORES,$MEM_PERCENT" >> whisper_benchmarks.csv
    
    echo "Benchmark entry added!"
fi

# Clean up temp files
rm -f "$TEMP_WAV"
echo -e "\nDone!"