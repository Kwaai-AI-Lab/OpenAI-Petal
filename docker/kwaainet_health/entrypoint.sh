#!/bin/bash

# Ensure INITIAL_PEERS is set
if [ -z "$INITIAL_PEERS" ]; then
    echo "ERROR: Environment variable INITIAL_PEERS is not set!"    
    exit 0
fi


# Generate the config.py file dynamically with Pydantic v1 & v2 support
cat <<EOF > /app/health_service/config.py
# Set initial peers dynamically
# Convert comma-separated string to proper Python array with quotes
INITIAL_PEERS = [$(echo "$INITIAL_PEERS" | awk -F, '{for(i=1;i<=NF;i++) printf "\"%s\"%s", $i, (i==NF?"":", ")}')]

#Define available models
MODELS = [ 
]

UPDATE_PERIOD = 60
EOF

echo "✅ Starting Kwaainet health site"

# Start Gunicorn Health API on port 8000
exec gunicorn app:app --bind 0.0.0.0:8000 --worker-class gthread --threads 10 --timeout 120
