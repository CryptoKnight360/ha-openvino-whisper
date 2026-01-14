FROM python:3.11-slim-bookworm

# 1. Configure APT Repositories (CRITICAL FIX)
# We remove the default .sources file and write a standard sources.list.
# We explicitly include 'non-free' and 'non-free-firmware' so apt can find the Intel drivers.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free non-free-firmware" > /etc/apt/sources.list.d/backports.list

# 2. Install System Dependencies & Drivers
# We pull intel-opencl-icd and intel-level-zero-gpu from backports.
# This provides the Compute Runtime required for 13th Gen (Raptor Lake).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    clinfo \
    libnuma1 \
    && apt-get install -y --no-install-recommends -t bookworm-backports \
    intel-opencl-icd \
    intel-level-zero-gpu \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# 3. Install Python Dependencies
# PINNED VERSIONS: Prevents "Unsupported attribute type: inputs" error.
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
