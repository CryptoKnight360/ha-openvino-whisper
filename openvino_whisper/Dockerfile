FROM ghcr.io/home-assistant/amd64-base-python:3.12-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg intel-opencl-icd && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install Wyoming and OpenVINO
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    "transformers>=4.45.0" \
    "wyoming==1.8.0" \
    "numpy<2.0.0"

WORKDIR /app
COPY server.py /app/server.py
COPY rootfs /

# Force executable permissions on all s6 scripts
RUN chmod -R 755 /etc/s6-overlay/s6-rc.d/ && \
    chmod a+x /app/server.py
