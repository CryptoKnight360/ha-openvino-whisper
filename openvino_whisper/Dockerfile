FROM python:3.11-slim-bookworm

# 1. Configure APT Repositories
# Remove default source files to avoid conflicts/duplicates
# Add 'non-free' and 'non-free-firmware' to both standard and backports
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free non-free-firmware" > /etc/apt/sources.list.d/backports.list

# 2. Install System Dependencies & Drivers
# Now that 'non-free' is enabled, apt will successfully find intel-opencl-icd
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
# Pinned versions to prevent XML Serialization errors (XML bug in 2024.5+)
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
