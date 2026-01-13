FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
USER root

# Install Intel GPU Drivers for 13th Gen (Iris Xe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gpg ca-certificates \
    && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" > /etc/apt/sources.list.d/intel-gpu-jammy.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3-pip \
    python3-dev \
    build-essential \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    && rm -rf /var/lib/apt/lists/*

# Install Python stack
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    "transformers>=4.45.0" \
    "soundfile" \
    "wyoming==1.5.2" \
    "numpy<2.0.0"

WORKDIR /app
COPY server.py run.sh ./
RUN chmod +x run.sh

ENTRYPOINT ["/app/run.sh"]
