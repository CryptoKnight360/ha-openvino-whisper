#!/bin/bash
set -e

CONFIG_PATH="/data/options.json"

if [ -f "$CONFIG_PATH" ]; then
    MODEL=$(jq --raw-output '.model // "openai/whisper-large-v3-turbo"' "$CONFIG_PATH")
    DEVICE=$(jq --raw-output '.device // "GPU"' "$CONFIG_PATH")
    LANGUAGE=$(jq --raw-output '.language // "en"' "$CONFIG_PATH")
    BEAM_SIZE=$(jq --raw-output '.beam_size // 1' "$CONFIG_PATH")
else
    MODEL="openai/whisper-large-v3-turbo"
    DEVICE="GPU"
    LANGUAGE="en"
    BEAM_SIZE=1
fi

echo "Starting OpenVINO Whisper..."
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Port: 10555 (Host Network)"

# Ensure Cache Exists
export HF_HOME="/data/model_cache"
mkdir -p "$HF_HOME"

# Permission Fix
RENDER_NODE="/dev/dri/renderD128"
if [ -e "$RENDER_NODE" ]; then
    RENDER_GID=$(stat -c '%g' "$RENDER_NODE")
    if ! getent group "$RENDER_GID" > /dev/null; then
        groupadd -g "$RENDER_GID" render_custom || true
    fi
    usermod -aG "$RENDER_GID" root || true
fi

# Launch on Port 10555
exec python3 /app/main.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --language "$LANGUAGE" \
    --beam-size "$BEAM_SIZE" \
    --uri "tcp://0.0.0.0:10555"
