#!/bin/bash

# Generate the config.py file dynamically for local development
cat <<EOF > /app/health_service/config.py
# Empty peers for local testing (avoids DHT connection errors)
INITIAL_PEERS = []

# Define available models (empty for local testing)
MODELS = []

# Faster update period for local development
UPDATE_PERIOD = 30

# Local development settings
DEBUG = True
TESTING = True
EOF

echo "✅ Starting KwaaiNet health site (Local Development Mode)"
echo "📍 Using mock data (no DHT connection required)"
echo "🔧 Running in development mode with debug enabled"
echo "📝 Logs will be available in /app/logs/"

# Create log file for debugging
LOG_FILE="/app/logs/health_$(date +%Y%m%d_%H%M%S).log"
mkdir -p /app/logs

# Use our simple Flask app instead of the complex Petals health service
echo "🚀 Starting simple Flask app for local testing..."
exec python3 /app/simple_app.py