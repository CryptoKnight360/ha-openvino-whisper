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

echo "---------------------------------------------------"
echo "Checking Intel Graphics Status (clinfo)..."
if command -v clinfo &> /dev/null; then
    clinfo | grep -E "Platform Name|Device Name" || echo "WARNING: clinfo returned no devices."
else
    echo "clinfo not installed."
fi
echo "---------------------------------------------------"

mkdir -p /data/model_cache

exec python3 /app/main.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --language "$LANGUAGE" \
    --beam-size "$BEAM_SIZE"
