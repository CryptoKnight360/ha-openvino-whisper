# Use the official Intel OpenVINO image (Standard Docker Hub, No 403)
FROM intel/openvino_runtime:2024.3.0

USER root

# Install system dependencies for audio and building Wyoming
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the STT stack
# We use the OpenVINO-provided Python environment
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    "transformers>=4.45.0" \
    "soundfile" \
    "wyoming==1.5.2" \
    "numpy<2.0.0"

WORKDIR /app
COPY server.py run.sh ./
RUN chmod +x run.sh

# The OpenVINO image uses a specific user, we stay root to ensure HA compatibility
ENTRYPOINT ["/app/run.sh"]
