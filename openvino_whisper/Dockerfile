FROM ghcr.io/home-assistant/amd64-base-debian:bookworm

# 1. Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    pkg-config \
    patchelf \
    git \
    libsndfile1 \
    libvulkan1 \
    libgbm1 \
    libusb-1.0-0-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install optimized Whisper libraries
RUN pip3 install --no-cache-dir --upgrade pip --break-system-packages && \
    pip3 install --no-cache-dir --upgrade setuptools wheel --break-system-packages && \
    pip3 install --no-cache-dir --break-system-packages \
    wyoming==1.5.2 \
    "optimum[openvino]" \
    openvino-genai==2025.0.0 \
    openvino-tokenizers==2025.0.0 \
    transformers \
    librosa \
    numpy

WORKDIR /app

# 3. Setup the Wyoming Server script
COPY wyoming_server.py .

# 4. Setup the Service (This replaces CMD/ENTRYPOINT)
# We create a folder for the service and copy run.sh there as 'run'
RUN mkdir -p /etc/services.d/openvino_whisper
COPY run.sh /etc/services.d/openvino_whisper/run
RUN chmod a+x /etc/services.d/openvino_whisper/run
