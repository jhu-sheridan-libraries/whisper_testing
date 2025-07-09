# Audio Transcription and Speaker Identification Tool

## What This Tool Does

This tool automatically converts audio recordings (like meetings, interviews, or lectures) into text and identifies who said what. It's like having an AI assistant that:

- **Listens** to your audio files and writes down everything that's said
- **Identifies speakers** and labels each part of the conversation (Speaker 1, Speaker 2, etc.)
- **Works with any audio format** (MP3, WAV, M4A, and more)
- **Runs on your computer** - your audio files never leave your machine

**Perfect for:** Meeting notes, interview transcripts, lecture notes, podcast transcriptions, research interviews, and any situation where you need to convert speech to text with speaker identification.

## Before You Start

### What You'll Need

**For Everyone:**
- A computer with internet connection
- Audio files you want to transcribe
- About 30 minutes for initial setup

**Technical Requirements:**
- Docker (we'll help you install this - it's like a virtual computer that runs our tool)
- A free Hugging Face account (this gives us access to the AI models)

### How Long Does It Take?

The tool processes audio slower than real-time. Here's what to expect:

- **5-minute audio file:** Takes about 15-30 minutes to process
- **1-hour meeting:** Takes about 3-6 hours to process
- **Factors that affect speed:** Audio quality, number of speakers, background noise

*Don't worry - you can start the process and walk away. It runs automatically.*

## Quick Start Guide

### Step 1: Get Your Computer Ready

**Install Docker (The Virtual Environment)**

Docker is like a virtual computer that ensures our tool works the same way on every machine. It's free and safe to install.

- **Windows/Mac:** Download from [docker.com](https://www.docker.com/products/docker-desktop/)
- **Follow the installer** - it's straightforward
- **Restart your computer** when installation finishes

**Get the Tool:**

**Option 1 - Download ZIP (Easier):**
1. Download this project from GitHub (green "Code" button → "Download ZIP")
2. Extract the ZIP file to a folder you'll remember (like your Desktop)
3. Remember where you put it - you'll need to find this folder later

**Option 2 - Using Git (If you have it installed):**
```bash
git clone https://github.com/DonRichards/whisper_testing
cd whisper_testing
```

### Step 2: Set Up AI Access

**Create a Hugging Face Account:**
1. Go to [huggingface.co](https://huggingface.co/) and create a free account
2. Once logged in, go to your profile settings
3. Create an "Access Token" with READ permission
4. **Save this token** - you'll need it every time you use the tool

**Accept AI Model Licenses:**
You need to accept terms for the AI models we use:
1. Visit [Speaker Diarization Model](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. Visit [Speaker Segmentation Model](https://huggingface.co/pyannote/segmentation-3.0)
3. For each one, click "Access repository" and accept the terms

### Step 3: Prepare Your Audio

1. **Create a 'data' folder** inside the project folder you downloaded
2. **Copy your audio files** into this 'data' folder
3. **Supported formats:** MP3, WAV, M4A, and most common audio formats

## Using the Tool

### Basic Transcription

**Open your computer's terminal/command prompt:**
- **Windows:** Press Windows key, type "cmd", press Enter
- **Mac:** Press Cmd+Space, type "terminal", press Enter

**Navigate to your project folder:**
```bash
cd path/to/your/whisper_testing/folder
```
*(Replace "path/to/your/whisper_testing/folder" with the actual location)*

**Run your first transcription:**
```bash
docker compose run --rm -e HF_TOKEN=your_token_here whisper-diarize /data/your_audio_file.mp3 --model tiny
```

**Replace:**
- `your_token_here` with your Hugging Face token
- `your_audio_file.mp3` with your actual filename

**What happens:**
1. The tool downloads the AI models (only happens once)
2. It processes your audio file
3. Creates a transcript in the 'output' folder
4. Shows progress as it works

### Choosing Quality vs Speed

**Model Sizes (Pick One):**
- `tiny` - **Fastest** (good for testing, less accurate)
- `base` - **Fast** (better than tiny)
- `small` - **Good balance** (recommended for most users)
- `medium` - **Better accuracy** (default, slower)
- `large` - **High accuracy** (slower)
- `large-v2` - **Best accuracy** (slowest, for important transcriptions)

**Example with better quality:**
```bash
docker compose run --rm -e HF_TOKEN=your_token_here whisper-diarize /data/meeting.mp3 --model small
```

### Advanced Options

**If you know how many people are speaking:**
```bash
docker compose run --rm -e HF_TOKEN=your_token_here whisper-diarize /data/interview.mp3 --model small --num-speakers 2
```

**Choose output format:**
```bash
# For video subtitles (VTT format)
docker compose run --rm -e HF_TOKEN=your_token_here whisper-diarize /data/lecture.mp3 --model small --format vtt

# For simple text file
docker compose run --rm -e HF_TOKEN=your_token_here whisper-diarize /data/meeting.mp3 --model small --format txt
```

## Understanding Your Results

**Your transcript appears in the 'output' folder with the same name as your audio file.**

**Example output formats:**

**Text Format (.txt):**
```
Speaker 1: Welcome everyone to today's meeting.
Speaker 2: Thank you for having me. I'm excited to discuss the project.
Speaker 1: Let's start with the budget overview.
```

**Video Subtitle Format (.vtt):**
```
00:00:00.000 --> 00:00:03.840
Speaker 1: Welcome everyone to today's meeting.

00:00:03.840 --> 00:00:07.200
Speaker 2: Thank you for having me. I'm excited to discuss the project.
```

**Output Format Details:**
1. **VTT (default)** - WebVTT format with speaker labels, compatible with most video players
2. **TXT** - Simple text format, each line prefixed with speaker label  
3. **SRT** - SubRip format (currently outputs as VTT, planned for future implementation)

## When Things Go Wrong

### "File Not Found"
- **Check:** Is your audio file in the 'data' folder?
- **Check:** Are you using the correct filename?
- **Try:** List files in data folder: `ls data/` (Mac) or `dir data\` (Windows)

### "Token Invalid" or Diarization Errors
- **Check:** Did you copy your Hugging Face token correctly?
- **Check:** Did you accept both model licenses?
- **Try:** Create a new token from your Hugging Face profile

### "Docker Not Found"
- **Solution:** Docker isn't installed or running
- **Fix:** Install Docker Desktop and make sure it's running

### Very Slow Processing
- **Normal:** Processing takes much longer than your audio length
- **Speed up:** Use 'tiny' or 'small' model for faster results
- **Be patient:** Large files with multiple speakers take hours

### Poor Transcription Quality
- **Try:** Use a larger model (small → medium → large)
- **Check:** Is your audio clear? Background noise affects accuracy
- **Consider:** Cleaning up audio first (removing background noise)

## Making It Easier Next Time

### Save Your Token
**Mac/Linux users** can save their token to avoid typing it every time:
```bash
echo 'export HF_TOKEN=your_token_here' >> ~/.zshrc
source ~/.zshrc
```

**Windows users** can set it in their session:
```cmd
set HF_TOKEN=your_token_here
```

Then you can run commands without typing the token:
```bash
docker compose run --rm whisper-diarize /data/audio.mp3 --model small
```

**Alternative:** You can also set the HF_TOKEN environment variable in your shell profile or pass it directly to the docker compose command as shown in the examples above.

### Pre-download Models
Speed up future runs by downloading models ahead of time:
```bash
python download_whisper_models.py --model small
```

### Batch Processing
Process multiple files by running the command multiple times with different filenames.

## Understanding the Technology

### What's Happening Behind the Scenes

**Whisper AI:** Created by OpenAI, this converts speech to text with high accuracy across many languages.

**Speaker Diarization:** AI technology that identifies different speakers in audio by analyzing voice characteristics.

**Docker:** Creates a consistent environment so the tool works the same on every computer, regardless of what other software you have installed.

**Why These Choices:**
- **Accuracy:** These are some of the best open-source tools available
- **Privacy:** Everything runs on your computer - no audio leaves your machine
- **Reliability:** Docker ensures consistent performance across different computers

## Technical Reference

### Command Line Options
```bash
docker compose run --rm -e HF_TOKEN=$HF_TOKEN whisper-diarize [audio_file] [options]
```

**Available Options:**
- `--model tiny|base|small|medium|large|large-v2` - Model size (default: medium)
- `--output /path/to/output.txt` - Custom output file path
- `--format vtt|srt|txt` - Output format: vtt (default), srt (currently outputs as vtt), txt
- `--num-speakers N` - Expected number of speakers
- `--language en` - Language code for transcription
- `--task transcribe|translate` - Transcribe or translate to English

### Performance Benchmarks

**Processing Time Multipliers (vs real-time):**
| Model | Transcription Only | With Speaker ID |
|-------|-------------------|-----------------|
| tiny | 1.2x | 2.7x |
| base | 2.0x | 3.5x |
| small | 3.0x | 4.5x |
| medium | 4.5x | 6.0x |
| large | 6.0x | 7.5x |
| large-v2 | 7.0x | 8.5x |

*Example: 10-minute audio with 'small' model = ~30-45 minutes processing*

Expected performance for a 5-minute audio file on a typical CPU:
- tiny: ~1-2x real-time (5-10 minutes)
- base: ~2-3x real-time (10-15 minutes)  
- small: ~3-4x real-time (15-20 minutes)
- medium: ~4-6x real-time (20-30 minutes)
- large: ~6-8x real-time (30-40 minutes)

### System Requirements
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB for models, plus space for audio/transcripts
- **CPU:** Any modern processor (faster = quicker processing)

### Docker Commands Reference

**Development and Debugging:**
```bash
# Access container shell for debugging
docker compose run --rm whisper-diarize --shell

# Run in debug mode
docker compose run --rm whisper-diarize --debug

# Clean up containers
docker compose down --remove-orphans
docker container prune
```

### File Structure
```
whisper_testing/
├── data/           # Put your audio files here
├── output/         # Transcripts appear here
├── models/         # Downloaded AI models (auto-created)
│   └── whisper/    # Whisper model cache
├── cpp_version/    # C++ implementation of the project
├── temp/           # Temporary files
├── docker-compose.yml
├── Dockerfile
├── transcribe_diarize.py
├── entrypoint.sh   # Docker entrypoint script
├── download_whisper_models.py
├── Pipfile         # Python dependencies
├── Pipfile.lock    # Locked dependency versions
└── README.md
```

### Benchmark Logging
The system automatically logs performance metrics to `output/whisper_benchmarks.csv` including:
- Processing time and efficiency
- Model used and configuration
- Audio duration and file details
- System resource usage

## Getting Help

**For Non-Technical Users:**
- Read the "When Things Go Wrong" section first
- Check that all files are in the right folders
- Verify your Hugging Face token is working
- Try with a shorter audio file first

**For Developers:**
- Check the Docker logs for detailed error messages
- Use `--debug` flag for verbose output
- Examine the benchmark CSV for performance insights
- Review the source code in `transcribe_diarize.py`

**Community Support:**
- Create an issue on the GitHub repository
- Include your error message and what you were trying to do
- Mention your operating system and audio file type

---

*This tool is designed to work offline after initial setup. Your audio files are processed locally and never uploaded to external servers.*
