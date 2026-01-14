FROM python:3.11-slim-bookworm

# Install basic tools required to download drivers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    wget \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# INSTALL LATEST INTEL COMPUTE RUNTIME (NEO) FOR RAPTOR LAKE (13th Gen)
# Debian Bookworm default drivers are too old for LattePanda Sigma.
# We download the official releases from Intel GitHub.
# -----------------------------------------------------------------------------
WORKDIR /tmp/intel-drivers

# Download specific compatible versions of Compute Runtime, Graphics Compiler, and Level Zero
# Versions: 24.52.32224.5 (Latest Stable as of late 2025/early 2026 era context)
RUN wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-core_1.0.17791.9_amd64.deb && \
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-opencl_1.0.17791.9_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget -q https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb

# Install the downloaded packages
RUN dpkg -i *.deb && rm *.deb

# -----------------------------------------------------------------------------

# Grant permissions to 'root' for render/video groups (just in case)
RUN getent group video || groupadd video && \
    getent group render || groupadd render && \
    usermod -aG video,render root

# Reset working directory
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
