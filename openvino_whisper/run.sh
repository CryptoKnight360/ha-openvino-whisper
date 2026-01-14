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

# ---------------------------------------------------------------------
# CLEANUP
# Wiping cache is MANDATORY because previous crashes left broken files.
# ---------------------------------------------------------------------
if [ -d "/data/model_cache" ]; then
    echo "Wiping model cache to ensure clean conversion..."
    rm -rf /data/model_cache
fi
mkdir -p /data/model_cache

# ---------------------------------------------------------------------
# PERMISSION FIXER
# ---------------------------------------------------------------------
RENDER_NODE="/dev/dri/renderD128"
if [ -e "$RENDER_NODE" ]; then
    RENDER_GID=$(stat -c '%g' "$RENDER_NODE")
    if ! getent group "$RENDER_GID" > /dev/null; then
        groupadd -g "$RENDER_GID" render_custom
    fi
    usermod -aG "$RENDER_GID" root
fi

# ---------------------------------------------------------------------
echo "Checking Intel Graphics Status (clinfo)..."
if command -v clinfo &> /dev/null; then
    # We expect to see "Intel(R) Iris(R) Xe Graphics"
    clinfo | grep -E "Platform Name|Device Name" || echo "ERROR: clinfo found 0 devices."
else
    echo "clinfo not installed."
fi
echo "---------------------------------------------------"

exec python3 /app/main.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --language "$LANGUAGE" \
    --beam-size "$BEAM_SIZE"
