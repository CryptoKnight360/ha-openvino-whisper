FROM ghcr.io/home-assistant/amd64-base-debian:bookworm

# Install requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential cmake \
    libsndfile1 libvulkan1 libgbm1 libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python stack
RUN pip3 install --no-cache-dir --upgrade pip --break-system-packages && \
    pip3 install --no-cache-dir --break-system-packages \
    wyoming==1.5.2 \
    "optimum[openvino]" \
    openvino-genai==2025.0.0 \
    numpy

WORKDIR /app
COPY wyoming_server.py .
COPY run.sh /
RUN chmod a+x /run.sh

# This satisfies the s6-overlay/PID 1 requirement
ENTRYPOINT ["/init"]
CMD ["/run.sh"]
