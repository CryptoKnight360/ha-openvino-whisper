FROM python:3.11-slim-bookworm

# Install system dependencies
# wget, ca-certificates: Required for downloading drivers securely
# clinfo: Hardware verification
# jq: Config parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    wget \
    clinfo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# INSTALL INTEL COMPUTE RUNTIME (NEO) - GEN 2 (RAPTOR LAKE SUPPORT)
# -----------------------------------------------------------------------------
WORKDIR /tmp/intel-drivers

# 1. Download IGC Gen2 Compiler (Verified Active URLs)
RUN wget -q https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb

# 2. Download Compute Runtime & GMM (Matches the Compiler above)
RUN wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb

# 3. Install all packages
RUN dpkg -i *.deb && rm *.deb

# -----------------------------------------------------------------------------

# Grant permissions to 'root' for render/video groups
RUN getent group video || groupadd video && \
    getent group render || groupadd render && \
    usermod -aG video,render root

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "optimum-intel[openvino,nncf]>=1.14.0" \
    "transformers>=4.36.0" \
    "wyoming>=1.5.0" \
    "zeroconf>=0.131.0" \
    "scipy"

# Copy application files
COPY rootfs/app /app
COPY run.sh /run.sh

RUN chmod +x /run.sh

CMD [ "/run.sh" ]
