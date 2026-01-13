FROM ghcr.io/home-assistant/amd64-base-python:3.12-bookworm

# System dependencies for OpenVINO and Audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg intel-opencl-icd && \
    rm -rf /var/lib/apt/lists/*

# Install specific versions as per report
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    "transformers>=4.45.0" \
    "wyoming==1.8.0" \
    "numpy<2.0.0"

# Copy the server script
COPY server.py /app/server.py
WORKDIR /app

# Copy the S6-overlay configuration
COPY rootfs /
RUN chmod a+x /app/server.py
