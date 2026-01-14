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
echo "Language: $LANGUAGE"

# ---------------------------------------------------------------------
# CACHE MANAGEMENT
# We map /data/model_cache to the Hugging Face cache.
# We DO NOT wipe this every time, or startup will take 10 minutes.
# ---------------------------------------------------------------------
export HF_HOME="/data/model_cache"
mkdir -p "$HF_HOME"

# Check if cache is suspiciously empty (permissions check)
if [ ! -w "$HF_HOME" ]; then
    echo "ERROR: /data/model_cache is not writable. Attempting fix..."
    chmod 777 "$HF_HOME"
fi

# ---------------------------------------------------------------------
# PERMISSION FIXER
# ---------------------------------------------------------------------
RENDER_NODE="/dev/dri/renderD128"
if [ -e "$RENDER_NODE" ]; then
    echo "Found render node at $RENDER_NODE"
    RENDER_GID=$(stat -c '%g' "$RENDER_NODE")
    
    # Create group if it doesn't exist
    if ! getent group "$RENDER_GID" > /dev/null; then
        groupadd -g "$RENDER_GID" render_custom || true
    fi
    
    # Add root (current user) to that group
    usermod -aG "$RENDER_GID" root || true
else
    echo "WARNING: GPU Render node not found! Performance will be low."
fi

# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------
echo "Checking Intel Graphics Status (clinfo)..."
if command -v clinfo &> /dev/null; then
    # Output only the device count to keep logs clean, unless 0
    DEVICE_COUNT=$(clinfo | grep "Number of devices" | head -n 1 | awk '{print $4}')
    echo "OpenCL Devices Found: ${DEVICE_COUNT:-0}"
else
    echo "clinfo not installed."
fi
echo "---------------------------------------------------"

# Launch the Python Application
exec python3 /app/main.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --language "$LANGUAGE" \
    --beam-size "$BEAM_SIZE"
