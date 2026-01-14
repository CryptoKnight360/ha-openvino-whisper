#!/usr/bin/with-contenv bashio

# Fetch config values from Home Assistant
MODEL=$(bashio::config 'model' 'large-v3-turbo')
LANGUAGE=$(bashio::config 'language' 'en')

bashio::log.info "Starting Wyoming OpenVINO Server..."
bashio::log.info "Model: $MODEL"
bashio::log.info "Hardware Device: OpenVINO"

# Run the Wyoming server
# Port 10300 is required for Home Assistant discovery
python3 -m wyoming_faster_whisper \
    --uri 'tcp://0.0.0.0:10300' \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --device openvino \
    --data-dir /data \
    --download-dir /data
