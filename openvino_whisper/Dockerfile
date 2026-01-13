FROM python:3.11-slim-bookworm

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir "optimum-intel[openvino]" transformers soundfile wyoming==1.5.2

WORKDIR /app
COPY server.py run.sh ./
RUN chmod +x run.sh

ENTRYPOINT ["/app/run.sh"]
