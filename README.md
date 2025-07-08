# Whisper Testing Project

## Overview

This project provides a containerized solution for automatic speech transcription and speaker diarization. It uses `faster-whisper` for efficient transcription and `pyannote.audio` for speaker diarization. The project is designed to be easy to use and to provide a consistent environment for batch processing audio files. It also includes tools for benchmarking the performance of different models and configurations.

## Getting Started

### Prerequisites

- Docker and Docker Compose installed on your system
- A Hugging Face account and API token (for accessing the diarization models)
- Audio files you want to transcribe (supported formats: mp3, wav, m4a, etc.)

### Setup

1. **Clone this repository:**
   ```bash
   git clone https://github.com/DonRichards/whisper_testing
   cd whisper_testing
   ```

2. **Get a Hugging Face API token:**
   - Create an account on [Hugging Face](https://huggingface.co/)
   - Go to your profile settings and create an access token
   - The token must have at least READ permission

3. **Accept the model license agreements:**
   - Visit both:
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
   - You may need to first log in to your Hugging Face account
   - Click the "Access repository" button or "Files and versions"
   - Read and accept the terms of use for both models

4. **Configure your environment:**
   - The application uses the `HF_TOKEN` environment variable. You can set it in your shell profile or pass it directly to the `docker-compose` command.

## Usage

1. **Place your audio files in the `data` directory:**
   ```bash
   mkdir -p data
   cp path/to/your/audio.mp3 data/
   ```

2. **Run the transcription:**
   ```bash
   # Basic usage (output will be in the output/ directory)
   docker compose run --rm -e HF_TOKEN=$HF_TOKEN whisper-diarize /data/audio.mp3 --model tiny

   # Specify number of speakers
   docker compose run --rm -e HF_TOKEN=$HF_TOKEN whisper-diarize /data/audio.mp3 --model tiny --num-speakers 2

   # Choose output format
   docker compose run --rm -e HF_TOKEN=$HF_TOKEN whisper-diarize /data/audio.mp3 --model tiny --format txt
   ```

   The output file will be placed in the `output` directory, and will automatically use the same name as the input file but with the appropriate extension (e.g., audio.mp3 → audio.vtt)

## Direct Command Execution

The `entrypoint.sh` script allows for direct command execution inside the container. This is useful for debugging and exploring the container's environment.

- **Debug:**
  ```bash
  docker compose run --rm whisper-diarize --debug
  ```
- **Shell:**
  ```bash
  docker compose run --rm whisper-diarize --shell
  ```
- **Other commands:**
  ```bash
  docker compose run --rm whisper-diarize ls -l /app
  ```

## Model Options

Available Whisper model sizes:
- `tiny` (fastest, least accurate)
- `base`
- `small`
- `medium` (default)
- `large`
- `large-v2` (slowest, most accurate)

## Pre-downloading Models

To avoid downloading models every time you run the container, you can pre-download them:

```bash
# Create models directory
mkdir -p models

# Download a specific model size
# Options: "tiny", "base", "small", "medium", "large", "large-v2"
python download_whisper_models.py --model tiny
```

## Configuration Options

Command line arguments:
- `--model` - Whisper model size (default: medium)
- `--output` - Custom output file path (optional)
- `--format` - Output format: vtt, srt, or txt (default: vtt)
- `--num-speakers` - Number of speakers expected in the audio
- `--language` - Language code for transcription
- `--task` - Choose between "transcribe" or "translate" (to English)

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Make sure your audio file is in the ./data directory
   - Use the correct path (/data/filename.mp3 inside the container)

2. **Speaker Diarization Not Working**
   - Verify your Hugging Face token is correct
   - Ensure you've accepted the terms for both required models:
     - pyannote/speaker-diarization-3.1
     - pyannote/segmentation-3.0

3. **Model Download Issues**
   - Check your internet connection
   - Verify the models directory has correct permissions
   - Try pre-downloading the model using download_whisper_models.py

## Cleaning Up

To avoid accumulating orphan containers:
```bash
# Use --rm flag when running
docker compose run --rm whisper-diarize ...

# Or clean up manually
docker compose down --remove-orphans
docker container prune
```

## Output Formats

1. **VTT (default)**
   - WebVTT format with speaker labels
   - Compatible with most video players

2. **TXT**
   - Simple text format
   - Each line prefixed with speaker label

3. **SRT**
   - SubRip format (currently outputs as VTT)
   - Planned for future implementation

## Benchmarking

The system automatically logs performance metrics to `/output/whisper_benchmarks.csv`, including:
- Timestamp of the run
- Audio filename and duration
- Model used and number of speakers
- Processing time and real-time factor
- CPU and memory usage

Expected performance for a 5-minute audio file on a typical CPU:
- tiny: ~1-2x real-time (5-10 minutes)
- base: ~2-3x real-time (10-15 minutes)
- small: ~3-4x real-time (15-20 minutes)
- medium: ~4-6x real-time (20-30 minutes)
- large: ~6-8x real-time (30-40 minutes)

Note: Performance can vary significantly based on:
- CPU speed and number of cores
- Audio quality and complexity
- Number of speakers
- Whether diarization is enabled

## Processing Time Estimates

The system will provide an estimate of processing time based on:
- Audio file duration
- Selected model size
- Whether speaker diarization is enabled

Approximate processing speeds (on a typical CPU):
1. **Transcription Only:**
   - tiny: ~1.2x real-time
   - base: ~2.0x real-time
   - small: ~3.0x real-time
   - medium: ~4.5x real-time
   - large: ~6.0x real-time
   - large-v2: ~7.0x real-time

2. **With Speaker Diarization:**
   - Add approximately 1.5x real-time

Example:
- 10-minute audio file
- Using 'medium' model
- With speaker diarization
- Expected time: (10 min × 4.5) + (10 min × 1.5) = 60 minutes

Note: Actual processing times may vary based on:
- CPU speed and number of cores
- Available memory
- Audio quality and complexity
- Number of speakers
- Background noise levels

## Solution

The project has been updated to use a more streamlined Docker setup, which resolves previous issues with file permissions and volume mounts. The key changes are:

- **Base Image**: The Dockerfile now uses the `linuxserver/faster-whisper:latest` base image, which comes with many of the required dependencies pre-installed.
- **Simplified Dockerfile**: The Dockerfile is now much shorter and easier to understand.
- **Simplified `docker-compose.yml`**: The `docker-compose.yml` file has been simplified to remove unnecessary volume mounts and configurations.
- **Output Directory**: A new `output/` directory is used for all output files, including transcripts and benchmark logs.

These changes make the project easier to use and more reliable, especially on macOS.

## Project Structure
- **`data/`**: Input audio files.
- **`output/`**: Output files (transcripts, benchmarks).
- **`models/`**: Cached models.
- **`cpp_version/`**: C++ implementation of the project.
- **`Dockerfile`**: Docker configuration.
- **`docker-compose.yml`**: Docker Compose configuration.
- **`transcribe_diarize.py`**: Main Python script for transcription and diarization.
- **`entrypoint.sh`**: Entrypoint script for the Docker container.
