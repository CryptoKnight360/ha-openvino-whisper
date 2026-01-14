FROM python:3.11-slim-bookworm

# Install system dependencies
# intel-opencl-icd: Drivers for Iris Xe iGPU
# clinfo: Verification tool
# jq: Config parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq \
    ocl-icd-libopencl1 \
    intel-opencl-icd \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# Grant OpenCL permissions to root user by adding to video/render groups
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
