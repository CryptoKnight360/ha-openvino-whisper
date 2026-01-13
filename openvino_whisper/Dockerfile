FROM ghcr.io/intel/openvino/runtime:latest

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python requirements
# We use the CPU version of torch to keep the image size down
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    wyoming==1.5.2 \
    openvino \
    "optimum[openvino]" \
    transformers \
    soundfile \
    numpy \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Copy scripts
COPY server.py run.sh ./
RUN chmod +x run.sh

# Set environment variables (can be overridden at runtime)
ENV MODEL_ID="OpenVINO/whisper-small-int8-ov"
ENV DEVICE="AUTO"

EXPOSE 10300

ENTRYPOINT ["/app/run.sh"]
