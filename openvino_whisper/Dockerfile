FROM python:3.11-slim-bookworm

# Install system dependencies
# libgl1-mesa-glx and libglib2.0-0 are required for OpenVINO/OpenCV
# jq is installed to help with parsing config in shell scripts if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# optimum-intel[openvino] installs the necessary Intel backends
# wyoming is the protocol library
# zeroconf is for auto-discovery
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "optimum-intel[openvino,nncf]" \
    "transformers>=4.36.0" \
    "wyoming>=1.5.0" \
    "zeroconf>=0.131.0" \
    "scipy"

# Copy application files from the local rootfs/app folder to the container /app folder
COPY rootfs/app /app
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD [ "/run.sh" ]
