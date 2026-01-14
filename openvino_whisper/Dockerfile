FROM ghcr.io/home-assistant/amd64-base-debian:bookworm

# 1. Install system requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake \
    libsndfile1 libvulkan1 libgbm1 libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install OpenVINO stack
RUN pip3 install --no-cache-dir --upgrade pip --break-system-packages && \
    pip3 install --no-cache-dir --break-system-packages \
    wyoming==1.5.2 \
    "optimum[openvino]" \
    openvino-genai==2025.0.0 \
    numpy

WORKDIR /app

# 3. Copy only the necessary files
COPY wyoming_server.py .
COPY run.sh /

# 4. CRITICAL: Automatically wipe any broken folders/services from previous attempts
RUN rm -rf /etc/s6-overlay/s6-rc.d/ov_* /etc/services.d/ov_* && chmod a+x /run.sh

ENTRYPOINT ["/init"]
CMD ["/run.sh"]
