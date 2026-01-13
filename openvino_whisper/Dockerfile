FROM python:3.11-slim-bookworm

USER root

# Install tools to add Intel's official repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gpg ca-certificates \
    && wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy main" > /etc/apt/sources.list.d/intel.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential python3-dev \
    intel-opencl-icd intel-level-zero-gpu level-zero \
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
