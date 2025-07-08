FROM linuxserver/faster-whisper:latest

# docker compose build --no-cache whisper-diarize

WORKDIR /app

COPY . .

# Install build dependencies and ffmpeg
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages we need
RUN pip install --no-cache-dir \
    pyannote.audio==3.1.1 \
    webvtt-py==0.4.6 \
    python-dotenv==1.0.0 \
    pydub==0.25.1 \
    psutil==5.9.8 \
    tqdm==4.66.1 \
    numpy==1.26.0

# Setup HuggingFace credentials directory for appuser
RUN mkdir -p ~/.cache/huggingface && \
    mkdir -p ~/.huggingface

# Copy entrypoint script to /usr/local/bin and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Bind mount the app directory
VOLUME /app
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run when container starts
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]