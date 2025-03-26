# Whisper Diarize C++

A C++ implementation of the Whisper Diarization tool using [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for efficient CPU-based transcription.

## Prerequisites

### Required Tools
- CMake (>= 3.10)
- C++ compiler with C++17 support (gcc/g++ >= 7 or clang >= 5)
- Git
- Make
- FFmpeg (for audio processing)

### Installing Prerequisites

#### Ubuntu/Debian:
```bash
# Install build essentials and CMake
sudo apt-get update
sudo apt-get install -y build-essential cmake git ffmpeg

# Verify installations
gcc --version
g++ --version
cmake --version
```

#### macOS:
```bash
# Using Homebrew
brew install cmake gcc make git ffmpeg

# Verify installations
gcc --version
g++ --version
cmake --version
```

#### Windows:
1. Install Visual Studio Build Tools or MinGW
2. Install CMake from https://cmake.org/download/
3. Add CMake to your PATH
4. Install Git from https://git-scm.com/download/win
5. Install FFmpeg from https://www.ffmpeg.org/download.html

## Building from Source

1. Clone this repository and enter the cpp_version directory:
```bash
cd cpp_version
```

2. Clone and build whisper.cpp:
```bash
# Clone the repository
git clone https://github.com/ggerganov/whisper.cpp.git

# Enter directory and build
cd whisper.cpp
make
cd ..
```

3. Download a Whisper model:
```bash
# From the whisper.cpp directory
cd whisper.cpp
bash ./models/download-ggml-model.sh tiny  # or base, small, medium, large
cd ..
```

4. Build the project:
```bash
# Create and enter build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
make

# If using Windows with MSVC:
cmake --build . --config Release
```

## Quick Start

1. Build the project:
```bash
mkdir -p build
cd build
cmake ..
make
cd ..
```

2. Download a model:
```bash
cd whisper.cpp
bash ./models/download-ggml-model.sh tiny  # or base, small, medium, large
cd ..
```

3. Run transcription:
```bash
./build/whisper_diarize -a input.mp3 -o output.vtt
```

## Command Line Options

```
Usage: whisper_diarize [OPTIONS]

Options:
  -a, --audio     Audio file path (required)
  -o, --output    Output file path (default: output.vtt)
  -m, --model     Model size (tiny/base/small/medium/large) [default: medium]
  -f, --format    Output format (vtt/txt) [default: vtt]
  -s, --speakers  Number of speakers [default: 0 (auto)]
  -l, --language  Language code [default: auto]
  -h, --help      Print usage
```

## Examples

1. Basic transcription with tiny model:
```bash
./build/whisper_diarize -a input.mp3 -m tiny
```

2. Save as text format:
```bash
./build/whisper_diarize -a input.mp3 -f txt -o output.txt
```

3. Specify number of speakers:
```bash
./build/whisper_diarize -a input.mp3 -s 2
```

4. Specify language:
```bash
./build/whisper_diarize -a input.mp3 -l en
```

## Supported Audio Formats

Currently supports:
- WAV (16kHz, mono)
- MP3 (will be converted to WAV)
- Other formats supported by FFmpeg (will be converted)

## Output Formats

1. VTT format (default):
```vtt
WEBVTT

1
00:00:00.000 --> 00:00:02.400
<v SPEAKER_01> Hello, this is a test.

2
00:00:02.400 --> 00:00:04.800
<v SPEAKER_02> This is another speaker.
```

2. TXT format:
```
[SPEAKER_01] Hello, this is a test.
[SPEAKER_02] This is another speaker.
```

## Performance

The C++ implementation offers several advantages over the Python version:
- Lower memory usage
- Faster processing times
- No Python runtime dependency
- Efficient CPU-based inference

Approximate processing speeds on CPU:
- tiny: ~0.8x real-time
- base: ~1.5x real-time
- small: ~2.5x real-time
- medium: ~3.5x real-time
- large: ~5.0x real-time

Note: Actual speeds may vary based on your CPU and audio complexity.

## Benchmarking

The tool automatically logs performance metrics to `whisper_benchmarks.csv`, including:
- Processing time
- CPU usage
- Memory consumption
- Real-time factor

## Model Files

Models are stored in the `models` directory with the naming format:
```
models/ggml-{size}.bin
```

Available sizes:
- ggml-tiny.bin (~75MB)
- ggml-base.bin (~142MB)
- ggml-small.bin (~466MB)
- ggml-medium.bin (~1.5GB)
- ggml-large.bin (~3GB)

## Limitations

Current limitations compared to the Python version:
- Speaker diarization is more basic
- No GPU acceleration (CPU-only)
- Fewer language options
- Limited audio format support

## Troubleshooting

1. **Model not found**
   ```bash
   # Download the model manually
   cd whisper.cpp
   bash ./models/download-ggml-model.sh tiny
   ```

2. **Build errors**
   ```bash
   # Make sure you have all dependencies
   sudo apt-get install build-essential cmake
   ```

3. **Audio loading fails**
   ```bash
   # Convert audio to supported format
   ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 