FROM python:3.11-slim-bookworm

# 1. Sources
# Switch to the new Debian sources format (Deb822) handling or standard list.
# We enable 'contrib', 'non-free', and 'non-free-firmware' for Intel drivers.
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
    echo "deb http://deb.debian.org/debian bookworm-backports main contrib non-free non-free-firmware" > /etc/apt/sources.list.d/backports.list

# 2. Drivers & System Dependencies
# FIX: 'intel-level-zero-gpu' does not exist in Debian repos. 
# We install 'intel-opencl-icd' (Driver) and 'libze1' (Level Zero Loader).
# We also install 'intel-media-va-driver-non-free' for HW media decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    jq \
    clinfo \
    libnuma1 \
    ffmpeg \
    git \
    vim \
    procps \
    libze1 \
    && apt-get install -y --no-install-recommends -t bookworm-backports \
    intel-opencl-icd \
    intel-media-va-driver-non-free \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Python Dependencies
# We use >= to allow the resolver to find compatible versions.
# optimum-intel[openvino,nncf] pulls in the OpenVINO runtime.
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
