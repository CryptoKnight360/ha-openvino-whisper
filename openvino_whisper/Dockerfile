FROM ghcr.io/home-assistant/amd64-base:3.19

# Install OpenVINO and Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Wyoming and Optimum-Intel (OpenVINO backend for Transformers)
RUN pip3 install --no-cache-dir \
    wyoming==1.5.2 \
    "optimum[openvino]" \
    transformers \
    accelerate \
    librosa

WORKDIR /app
COPY wyoming_server.py .
COPY run.sh /
RUN chmod a+x /run.sh

ENTRYPOINT ["/run.sh"]
