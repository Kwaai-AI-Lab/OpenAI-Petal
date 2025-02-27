#!/bin/bash
set -e

# Ensure /root/.cache/temp exists
if [ ! -d "/root/.cache/temp" ]; then
    echo "Creating /root/.cache/temp directory..."
    mkdir -p /root/.cache/temp
fi

echo "Checking if model is already downloaded..."
MODEL_PATH="/root/.cache/huggingface/hub/models--${KWAAINET_MODEL//\//--}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH.. Downloading..."
    huggingface-cli download "$KWAAINET_MODEL" --cache-dir /root/.cache/huggingface
else
    echo "Model already exists. Skipping download."
fi

echo "Starting Petals server..."
CMD="python3 -m petals.cli.run_server $KWAAINET_MODEL --num_blocks $KWAAINET_BLOCKS"

if [ -n "$ANNOUNCE_ADDR" ]; then
    CMD+=" --announce_maddrs $ANNOUNCE_ADDR"
fi

if [ -z "$ANNOUNCE_ADDR" ] && [ -n "$PUBLIC_IP" ]; then
    CMD+=" --public_ip $PUBLIC_IP"
fi

if [ -n "$INITIAL_PEERS" ]; then
    CMD+=" --initial_peers $INITIAL_PEERS"
fi

if [ -n "$NORELAY" ]; then
    CMD+=" --no_auto_relay"
fi

if [ -n "$PUBLIC_NAME" ]; then
    CMD+=" --public_name $PUBLIC_NAME"
fi

CMD+=" --port 8080"  

exec $CMD
