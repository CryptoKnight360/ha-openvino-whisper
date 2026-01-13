FROM ghcr.io/intel/openvino/runtime:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y     python3 python3-pip ffmpeg sox     && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir     wyoming     openvino     optimum[openvino]     transformers     soundfile     numpy     webrtcvad     torch

WORKDIR /app
COPY run.sh server.py benchmark.py ./
RUN chmod +x run.sh

CMD ["./run.sh"]
