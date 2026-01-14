FROM ghcr.io/home-assistant/amd64-base-debian:bookworm

# Install system dependencies
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

# Install OpenVINO and Wyoming stack
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
COPY wyoming_server.py .
COPY run.sh /
RUN chmod a+x /run.sh

# This is the standard HA init path
ENTRYPOINT ["/init"]
CMD ["/run.sh"]
