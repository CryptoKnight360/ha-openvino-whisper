#!/bin/bash

# Enable strict error handling
set -e

echo "Starting Wyoming OpenVINO Whisper..."

CONFIG_PATH="/data/options.json"

# Check if config exists, otherwise use defaults (useful for local testing outside HA)
if [ -f "$CONFIG_PATH" ]; then
    MODEL=$(jq --raw-output '.model // "openai/whisper-large-v3-turbo"' $CONFIG_PATH)
    DEVICE=$(jq --raw-output '.device // "GPU"' $CONFIG_PATH)
    LANGUAGE=$(jq --raw-output '.language // "en"' $CONFIG_PATH)
    BEAM_SIZE=$(jq --raw-output '.beam_size // 1' $CONFIG_PATH)
else
    echo "Config file not found, using defaults."
    MODEL="openai/whisper-large-v3-turbo"
    DEVICE="GPU"
    LANGUAGE="en"
    BEAM_SIZE=1
fi

echo "--------------------------------"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Language: $LANGUAGE"
echo "  Beam Size: $BEAM_SIZE"
echo "--------------------------------"

# Start the application
# We use exec so the python process becomes PID 1 (receives signals correctly)
exec python3 /app/main.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --language "$LANGUAGE" \
    --beam-size "$BEAM_SIZE"
