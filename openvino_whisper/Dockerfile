FROM ghcr.io/home-assistant/amd64-base-debian:bookworm

# Install Python and dependencies via apt-get (Confirmed working on Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install requirements as per the OpenVINO report
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    "transformers>=4.45.0" \
    "wyoming==1.8.0" \
    "numpy<2.0.0"

WORKDIR /app
COPY server.py /app/server.py
COPY rootfs /

# Force executable permissions for S6 services
RUN chmod -R 755 /etc/s6-overlay/s6-rc.d/ && \
    chmod a+x /app/server.py
