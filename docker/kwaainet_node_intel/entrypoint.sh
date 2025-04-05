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

# Apply XPU patches if running on Intel GPU
if python -c "import torch; import sys; sys.exit(0 if hasattr(torch, 'xpu') and torch.xpu.is_available() else 1)" 2>/dev/null; then
    echo "Intel XPU device detected! Applying XPU support patches..."
    # Assuming patch_xpu_support.py is in the same directory as this script
    SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
    python "${SCRIPT_DIR}/patch_xpu_support.py" /opt/conda/lib/python3.10/site-packages/petals/
    echo "XPU patches applied successfully."
else
    echo "No Intel XPU device detected. Continuing without patches."
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

if [ -n "$PORT" ] && [[ "$PORT" =~ ^[0-9]+$ ]]; then
    CMD+=" --port $PORT"
else
    CMD+=" --port 8080"
fi

if [ -n "$DEVICE" ]; then
    CMD+=" --device $DEVICE"
elif python -c "import torch; import sys; sys.exit(0 if hasattr(torch, 'xpu') and torch.xpu.is_available() else 1)" 2>/dev/null; then
    # Automatically use XPU if available
    CMD+=" --device xpu"
fi

exec $CMD