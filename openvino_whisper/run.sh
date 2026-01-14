#!/usr/bin/with-contenv bashio

MODEL=$(bashio::config 'model' 'openai/whisper-large-v3-turbo')
DEVICE=$(bashio::config 'device' 'CPU')

bashio::log.info "Starting OpenVINO Whisper on $DEVICE..."

python3 /app/wyoming_server.py \
    --model "$MODEL" \
    --device "$DEVICE"
