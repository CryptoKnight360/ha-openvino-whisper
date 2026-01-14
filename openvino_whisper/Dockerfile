FROM python:3.11-slim-bookworm

# 1. Add Debian Bookworm Backports (Official source for newer kernels/drivers)
RUN echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free-firmware" > /etc/apt/sources.list.d/backports.list

# 2. Install System Dependencies & Drivers
# We pull intel-opencl-icd and intel-level-zero-gpu from backports to get 13th Gen support
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

# 3. Install Python Dependencies (PINNED STABLE VERSIONS)
# This fixes the "Unsupported attribute type for serialization" error
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
