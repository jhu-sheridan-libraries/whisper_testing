FROM python:3.10-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install required Python packages
RUN pip install --no-cache-dir \
    faster-whisper==0.9.0 \
    pyannote.audio==3.1.1 \
    webvtt-py==0.4.6 \
    python-dotenv==1.0.0 \
    pydub==0.25.1 \
    psutil==5.9.8 \
    tqdm==4.66.1 \
    numpy==1.26.0 \
    huggingface_hub

# Install OpenAI's whisper as a fallback
RUN pip install openai-whisper

# Install faster-whisper from GitHub to get the latest fixes
RUN pip install git+https://github.com/guillaumekln/faster-whisper.git@v0.9.0

# Create a non-root user with the same UID/GID as the host user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/bash -m appuser

# Setup HuggingFace credentials directory for appuser
RUN mkdir -p /home/appuser/.cache/huggingface && \
    mkdir -p /home/appuser/.huggingface && \
    chown -R appuser:appuser /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser/.huggingface

# Set up app directories
RUN mkdir -p /app/models
RUN mkdir -p /data
RUN chown -R appuser:appuser /app /data

# Copy our scripts
COPY transcribe_diarize.py /app/
COPY .env /app/
COPY download_models.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
RUN chmod +x /app/download_models.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/appuser

# Switch to the non-root user
USER appuser

# Command to run when container starts
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]