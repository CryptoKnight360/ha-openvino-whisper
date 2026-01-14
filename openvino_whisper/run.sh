#!/usr/bin/env bashio

MODEL=$(bashio::config 'model' 'openai/whisper-large-v3-turbo')
DEVICE="CPU"

bashio::log.info "Starting Wyoming OpenVINO Server for LattePanda..."

# Run the python script from the /app directory
python3 /app/wyoming_server.py \
    --model "$MODEL" \
    --device "$DEVICE"
