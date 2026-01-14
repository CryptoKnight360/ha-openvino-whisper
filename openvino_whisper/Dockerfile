FROM python:3.11-slim-bookworm

# 1. Sources (Verified Working for Intel Drivers)
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free non-free-firmware" > /etc/apt/sources.list.d/backports.list

# 2. Drivers & System Dependencies
# CRITICAL FIX: Added 'ffmpeg' (Required for audio processing)
# CRITICAL FIX: Added 'git' (Required for some pip installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    clinfo \
    libnuma1 \
    ffmpeg \
    git \
    && apt-get install -y --no-install-recommends -t bookworm-backports \
    intel-opencl-icd \
    intel-level-zero-gpu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Python Dependencies
# FIX: We let pip resolve dependency hell. 
# optimum-intel>=1.21.0 is required for proper Transformers >4.44 support and Turbo models.
# onnxruntime-openvino is often required for specific export operations.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "optimum-intel[openvino,nncf]>=1.21.0" \
    "transformers>=4.46.0" \
    "torch>=2.4.0" \
    "wyoming>=1.5.0" \
    "zeroconf>=0.131.0" \
    "scipy" \
    "openvino-tokenizers"

COPY rootfs/app /app
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD [ "/run.sh" ]
