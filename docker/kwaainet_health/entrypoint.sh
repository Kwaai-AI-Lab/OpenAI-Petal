#!/bin/bash

# Wait for Petals DHT to start (ensure the peer ID is available)
sleep 5

# Extract public key (Multihash Peer ID)
MULTIHASH_PEER_ID=$(python3 /app/check_identity.py get-public-secret "$KWAAI_SECRET_KEY")
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract public key (Multihash Peer ID) from secret"
    exit 1
fi

echo "✅ Multihash Peer ID (Base58): $MULTIHASH_PEER_ID"

# Generate the config.py file dynamically with Pydantic v1 & v2 support
cat <<EOF > /app/health_service/config.py
import pydantic

# Detect Pydantic version
PYDANTIC_VERSION = int(pydantic.__version__.split(".")[0])

if PYDANTIC_VERSION >= 2:
    from pydantic import BaseModel
else:
    from pydantic.dataclasses import dataclass

# Define models dynamically
if PYDANTIC_VERSION >= 2:
    class ModelInfo(BaseModel):
        dht_prefix: str
        repository: str
        num_blocks: int
else:
    @dataclass
    class ModelInfo:
        dht_prefix: str
        repository: str
        num_blocks: int

# Set initial peers dynamically
INITIAL_PEERS = ["/ip4/127.0.0.1/tcp/8000/p2p/$MULTIHASH_PEER_ID"]

# Define available models
MODELS = [
    ModelInfo(
        dht_prefix="Meta-Llama-3-1-405B-Instruct-hf",
        repository="meta-llama/Meta-Llama-3.1-405B-Instruct",
        num_blocks=126,
    ),
    ModelInfo(
        dht_prefix="mistralai/Mixtral-8x22B-Instruct-v0-1",
        repository="mistralai/Mixtral-8x22B-Instruct-v0.1",
        num_blocks=56,
    ),
]

UPDATE_PERIOD = 60
EOF

echo "✅ Generated config.py with INITIAL_PEERS: /ip4/127.0.0.1/tcp/8000/p2p/$MULTIHASH_PEER_ID"

# Start Gunicorn Health API on port 8080
exec gunicorn app:app --bind 0.0.0.0:8000 --worker-class gthread --threads 10 --timeout 120
