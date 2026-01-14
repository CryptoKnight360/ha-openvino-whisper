FROM ghcr.io/home-assistant/amd64-base-debian:bookworm

# Install Python and system dependencies required for OpenVINO and Audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libsndfile1 \
    libvulkan1 \
    libgbm1 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Wyoming, Faster-Whisper, and OpenVINO runtime
# --break-system-packages is required for Debian Bookworm based images
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --break-system-packages \
    wyoming-faster-whisper \
    openvino==2024.5.0 \
    ctranslate2>=4.4.0

WORKDIR /data

# Copy the entrypoint script
COPY run.sh /
RUN chmod a+x /run.sh

ENTRYPOINT ["/run.sh"]
