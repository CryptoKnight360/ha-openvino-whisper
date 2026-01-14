FROM python:3.11-slim-bookworm

# Install system dependencies
# wget/ca-certificates: To download drivers
# clinfo: To verify hardware
# jq: To parse config
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
# INSTALL INTEL COMPUTE RUNTIME (NEO) - MATCHED SET
# Release: 23.43.27642.40 (Stable, supports 13th Gen Raptor Lake)
# -----------------------------------------------------------------------------
WORKDIR /tmp/intel-drivers

# 1. Download compatible Graphics Compiler (IGC)
RUN wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15193.4/intel-igc-core_1.0.15193.4_amd64.deb && \
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15193.4/intel-igc-opencl_1.0.15193.4_amd64.deb

# 2. Download Compute Runtime and Memory Manager
RUN wget -q https://github.com/intel/compute-runtime/releases/download/23.43.27642.40/intel-level-zero-gpu_1.3.27642.40_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/23.43.27642.40/intel-opencl-icd_23.43.27642.40_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/23.43.27642.40/libigdgmm12_22.3.0_amd64.deb

# 3. Install all packages together to resolve dependencies
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
