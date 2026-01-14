#!/bin/bash
set -e

# Config path
CONFIG_PATH="/data/options.json"

# Parse configuration
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

# DEBUG: Check if the system sees the iGPU
# We expect to see "Intel(R) Iris(R) Xe Graphics"
echo "Checking OpenCL devices..."
if command -v clinfo &> /dev/null; then
    clinfo | grep -E "Platform Name|Device Name" || echo "No OpenCL devices found."
else
    echo "clinfo not found."
fi

# Ensure cache directory exists
mkdir -p /data/model_cache

# Start Python App
exec python3 /app/main.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --language "$LANGUAGE" \
    --beam-size "$BEAM_SIZE"
