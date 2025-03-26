#! /bin/bash

# Accepts a filepath and which normalize technique to use.
# Noise Reduction: Apply adaptive noise reduction algorithms to remove background noise while preserving speech characteristics.
# Volume Normalization: Normalize to around -16 to -14 dB LUFS (Loudness Units relative to Full Scale) for consistent volume levels.
# Dynamic Range Compression: Apply light compression (2:1 or 3:1 ratio) to reduce the difference between quiet and loud sections.
# High-Pass Filtering: Remove frequencies below 80-100 Hz which typically contain room rumble and other low-frequency noise.
# De-essing: Selectively reduce sibilance (harsh "s" and "sh" sounds) that can confuse speech models.
# Dereverberation: Apply algorithms to reduce echo and reverberation effects, especially for recordings in large rooms.
# Voice Activity Detection (VAD): Segment audio to only process parts containing speech, reducing processing of silence or noise-only sections.
# Sample Rate Standardization: Convert to 16kHz sampling rate (Whisper's native rate) to avoid potential resampling artifacts.
# Audio Channel Management: Convert stereo to mono by averaging channels, as Whisper works best with single-channel audio.
# Audio Segmentation: Break long recordings into 30-second chunks for optimal performance, as Whisper works best with shorter segments.
# Source Separation: For multi-speaker scenarios, apply source separation techniques to isolate individual voices.

# Usage: ./preprocess_audio.sh <filepath> <technique1> [technique2] [technique3] ...
# Example: ./preprocess_audio.sh audio.mp3 noise_reduction

if [ -z "$1" ]; then
    echo "Error: No filepath provided"
    echo "Usage: ./preprocess_audio.sh <filepath> <technique1> [technique2] [technique3] ..."
    echo "Technique options: noise_reduction, volume_normalization, dynamic_range_compression, high_pass_filtering, de_essing, dereverberation, voice_activity_detection, sample_rate_standardization, audio_channel_management, audio_segmentation, source_separation"
    echo "Example: ./preprocess_audio.sh audio.mp3 noise_reduction"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/temp"

# Create temp directory if it doesn't exist
mkdir -p "$TEMP_DIR"

# Get the input file
INPUT_FILE="$1"
shift  # Remove the input file from arguments

# If no techniques specified, use default
if [ $# -eq 0 ]; then
    TECHNIQUES=("audio_channel_management")
else
    TECHNIQUES=("$@")
fi

# FFmpeg commands for each technique
# Modified to ensure consistent audio processing
noise_reduction='anlmdn=s=0.001:p=0.003:r=0.01'
volume_normalization='loudnorm=I=-16:TP=-2:LRA=11'  # Fixed TP value to be in valid range
dynamic_range_compression='acompressor=threshold=-12dB:ratio=2:attack=200:release=1000'
high_pass_filtering='highpass=f=100'
de_essing='highpass=f=6000,lowpass=f=10000,acompressor=threshold=-12dB:ratio=2:attack=200:release=1000'
dereverberation='arnndn=m=./rnnoise-models/sh.rnnn'
sample_rate_standardization='-ar 16000'
audio_channel_management='-ac 1'
audio_segmentation='-filter_complex "compand=gain=-10dB,loudnorm=I=-16:TP=-1.5dB:LRA=11dB:measured_i=1:measured_lra=1:measured_thresh=-20dB:linear=true:print_format=json"'
source_separation='-filter_complex "compand=gain=-10dB,loudnorm=I=-16:TP=-1.5dB:LRA=11dB:measured_i=1:measured_lra=1:measured_thresh=-20dB:linear=true:print_format=json"'

CURRENT_FILE="$INPUT_FILE"

# Process each technique in sequence
for technique in "${TECHNIQUES[@]}"; do
    OUTPUT_FILE="${CURRENT_FILE%.*}_${technique}.wav"
    
    case $technique in
        noise_reduction|volume_normalization|dynamic_range_compression|high_pass_filtering|de_essing)
            FILTER="${!technique}"  # This gets the value of the variable named in $technique
            # Convert to mono first, then apply filter
            ffmpeg -y -i "$CURRENT_FILE" -ac 1 -af "$FILTER" -ar 16000 -c:a pcm_s16le "$OUTPUT_FILE"
            CURRENT_FILE="$OUTPUT_FILE"
            ;;
        sample_rate_standardization)
            ffmpeg -y -i "$CURRENT_FILE" -ar 16000 -ac 1 -c:a pcm_s16le "$OUTPUT_FILE"
            CURRENT_FILE="$OUTPUT_FILE"
            ;;
        audio_channel_management)
            ffmpeg -y -i "$CURRENT_FILE" -ac 1 -ar 16000 -c:a pcm_s16le "$OUTPUT_FILE"
            CURRENT_FILE="$OUTPUT_FILE"
            ;;
        combine)
            # Special case that combines compatible filters into one pass
            FILTERS="${noise_reduction},${high_pass_filtering},${dynamic_range_compression},${volume_normalization}"
            ffmpeg -y -i "$CURRENT_FILE" -ac 1 -af "$FILTERS" -ar 16000 -c:a pcm_s16le "$OUTPUT_FILE"
            CURRENT_FILE="$OUTPUT_FILE"
            ;;
        audio_segmentation)
            ffmpeg -i "$CURRENT_FILE" -filter_complex "$audio_segmentation" -c:a pcm_s16le "$OUTPUT_FILE"
            CURRENT_FILE="$OUTPUT_FILE"
            ;;
        source_separation)
            ffmpeg -i "$CURRENT_FILE" -filter_complex "$source_separation" -c:a pcm_s16le "$OUTPUT_FILE"
            CURRENT_FILE="$OUTPUT_FILE"
            ;;
        *)
            echo "Invalid technique: $technique"
            exit 1
            ;;
    esac
    
    # Update current file for next iteration
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "Error processing $technique"
        exit 1
    fi
done

# Move the final output to temp directory
mv "$OUTPUT_FILE" "$TEMP_DIR/processed_input.wav"

# Print the final output file name
echo "Final output: $TEMP_DIR/processed_input.wav"
