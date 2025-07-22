# Docker does not natively support reading variables from .env in the FROM instruction.
# However, you can use build arguments to pass the value from .env at build time.

# In your .env file, you have:
# WHISPER_VER=latest

# In your Dockerfile, use an ARG and substitute it in the FROM line:
ARG WHISPER_VER=latest
FROM linuxserver/faster-whisper:${WHISPER_VER}

# docker compose build --no-cache whisper-diarize

WORKDIR /app

COPY . .

# Install build dependencies and ffmpeg
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

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