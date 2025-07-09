# Cross-Platform Whisper Transcription

This setup provides Linux and Windows container compatibility for the whisper transcription service.

## Key Changes Made

### 1. New Cross-Platform Files
- `Dockerfile.cross-platform` - Uses `python:3.11-slim` base image
- `entrypoint.py` - Python replacement for bash script
- `docker-compose.cross-platform.yml` - Updated compose file
- `requirements.txt` - Consolidated dependencies

### 2. Path Handling Updates
- Added `pathlib.Path` imports for cross-platform paths
- Replaced hardcoded Unix paths with Path objects
- Updated cache directory creation logic

### 3. Docker Changes
- **Base Image**: `python:3.11-slim` (works on Linux/Windows)
- **Entrypoint**: Python script instead of bash
- **Dependencies**: Standard pip packages instead of Linux-specific ones

## Usage

### Linux/macOS
```bash
docker compose -f docker-compose.cross-platform.yml build whisper-diarize
docker compose -f docker-compose.cross-platform.yml run --rm whisper-diarize "/path/to/audio.wav"
```

### Windows
```powershell
docker compose -f docker-compose.cross-platform.yml build whisper-diarize
docker compose -f docker-compose.cross-platform.yml run --rm whisper-diarize "C:\path\to\audio.wav"
```

### Google Colab
```python
# In Colab cell:
!docker compose -f docker-compose.cross-platform.yml build whisper-diarize
!docker compose -f docker-compose.cross-platform.yml run --rm whisper-diarize "/content/audio.wav"
```

## Environment Variables
- `HF_TOKEN` - Required for diarization models
- `PYTHONUNBUFFERED=1` - For real-time output
- `PYTHONPATH=/app` - Python module resolution

## Whisper Options
The setup supports both:
- **faster-whisper** (primary, CPU optimized)
- **openai-whisper** (fallback, original implementation)

## Testing
```bash
# Debug mode
docker compose -f docker-compose.cross-platform.yml run --rm whisper-diarize --debug

# Help
docker compose -f docker-compose.cross-platform.yml run --rm whisper-diarize --help
```

## Compatibility
✅ Linux containers
✅ Windows containers  
✅ Google Colab
✅ Docker Desktop (all platforms)
✅ Cloud platforms (AWS, GCP, Azure)