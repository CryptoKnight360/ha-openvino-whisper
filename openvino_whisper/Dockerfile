FROM python:3.11-slim-bookworm

# 1. Configure APT Repositories
# We remove the default source file and create a new one listing 'non-free' and 'non-free-firmware'.
# This allows apt to find the proprietary Intel drivers.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free non-free-firmware" > /etc/apt/sources.list.d/backports.list

# 2. Install System Dependencies & Drivers
# CRITICAL FIX: We install 'intel-opencl-icd' from backports (it supports 13th Gen).
# We DO NOT install 'intel-level-zero-gpu' because it is missing from backports and caused Exit 100.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    clinfo \
    libnuma1 \
    && apt-get install -y --no-install-recommends -t bookworm-backports \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# 3. Install Python Dependencies
# Pinned versions prevent the "Unsupported attribute type" XML error.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "transformers==4.45.2" \
    "openvino==2024.4.0" \
    "optimum-intel[openvino,nncf]==1.20.1" \
    "wyoming>=1.5.0" \
    "zeroconf>=0.131.0" \
    "scipy"

# Copy application files
COPY rootfs/app /app
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD [ "/run.sh" ]
