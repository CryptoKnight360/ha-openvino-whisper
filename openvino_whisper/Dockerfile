# This allows the Supervisor to inject the correct base image automatically
ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.12-bookworm
FROM ${BUILD_FROM}

# Install system dependencies for OpenVINO and Audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Install requirements exactly as the report specifies
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    "transformers>=4.45.0" \
    "wyoming==1.8.0" \
    "numpy<2.0.0"

# Set up the application
WORKDIR /app
COPY server.py /app/server.py
COPY rootfs /

# Ensure the server script is executable
RUN chmod a+x /app/server.py
