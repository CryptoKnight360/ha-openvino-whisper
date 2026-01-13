FROM python:3.11-slim-bookworm

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential python3-dev intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    "optimum-intel[openvino]" \
    "transformers>=4.45.0" \
    "soundfile" \
    "wyoming==1.5.2" \
    "numpy<2.0.0"

WORKDIR /app
COPY server.py ./
RUN chmod +x server.py

ENTRYPOINT ["python3", "/app/server.py"]
