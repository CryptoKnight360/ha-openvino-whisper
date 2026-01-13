FROM ghcr.io/intel/openvino/runtime:latest

USER root
RUN apt-get update && apt-get install -y \
    python3-pip ffmpeg sox build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Force update pip
RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir \
    wyoming==1.5.2 \
    openvino \
    "optimum[openvino]" \
    transformers \
    soundfile \
    webrtcvad \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY . .
RUN chmod +x run.sh

# Switch back to openvino user if desired, or stay root for simple setups
ENTRYPOINT ["/bin/bash", "/app/run.sh"]
