#!/usr/bin/with-contenv bashio

MODEL=$(bashio::config 'model')
DEVICE=$(bashio::config 'device')

bashio::log.info "Starting OpenVINO Whisper on $DEVICE..."

python3 /app/wyoming_server.py \
    --model "$MODEL" \
    --device "$DEVICE"
