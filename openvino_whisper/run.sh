#!/usr/bin/with-contenv bashio

echo "Starting Wyoming OpenVINO Whisper..."

# Read configuration from Home Assistant's options.json (mapped by bashio or standard locations)
CONFIG_PATH="/data/options.json"

# Defaults
MODEL="distil-whisper/distil-small.en"
DEVICE="GPU"
LANGUAGE="en"
BEAM_SIZE=1

if [ -f "$CONFIG_PATH" ]; then
    # Parse JSON using python one-liner if jq is not available, or simple grep for simple setup
    # Using python to parse options safely
    MODEL=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['model'])")
    DEVICE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['device'])")
    LANGUAGE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['language'])")
    BEAM_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['beam_size'])")
fi

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Language: $LANGUAGE"
echo "  Beam Size: $BEAM_SIZE"

# Start the application
exec python3 /app/main.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --language "$LANGUAGE" \
    --beam-size "$BEAM_SIZE"
