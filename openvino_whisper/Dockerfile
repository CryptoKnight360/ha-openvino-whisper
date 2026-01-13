FROM python:3.11-slim-bookworm

# Use root for setup
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# We install torch CPU-only to save massive amounts of space
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    transformers \
    soundfile \
    wyoming==1.5.2

WORKDIR /app
COPY server.py run.sh ./
RUN chmod +x run.sh

# Environment variables
ENV MODEL_ID="OpenVINO/whisper-small-int8-ov"
ENV DEVICE="AUTO"

ENTRYPOINT ["/app/run.sh"]
