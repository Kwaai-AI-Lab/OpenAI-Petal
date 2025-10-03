#!/usr/bin/env python3
"""
Simple Flask app for local KwaaiNet health dashboard testing
Avoids DHT connections and uses mock data
"""

from flask import Flask, render_template_string
from flask_cors import CORS
import json
import time

app = Flask(__name__)
CORS(app)

# Mock data for local testing
MOCK_NODES = [
    {
        "peer_id": "QmLocalDev1",
        "location": {"latitude": 37.7749, "longitude": -122.4194},
        "status": "online",
        "blocks": 2,
        "model": "test-model",
        "last_seen": time.time()
    },
    {
        "peer_id": "QmLocalDev2",
        "location": {"latitude": 40.7128, "longitude": -74.0060},
        "status": "online",
        "blocks": 1,
        "model": "test-model",
        "last_seen": time.time()
    }
]

@app.route('/')
def index():
    """Serve the local development HTML"""
    try:
        with open('/app/health_service/templates/index.html', 'r') as f:
            return f.read()
    except:
        return """
        <html>
        <body>
        <h1>KwaaiNet Local Health Dashboard</h1>
        <p>✅ Container is running successfully!</p>
        <p>📊 Mock data mode for local testing</p>
        <p>🔧 No DHT connection required</p>
        </body>
        </html>
        """

@app.route('/api/stats')
def stats():
    """API endpoint with mock statistics"""
    return json.dumps({
        "total_nodes": len(MOCK_NODES),
        "online_nodes": len([n for n in MOCK_NODES if n["status"] == "online"]),
        "total_blocks": sum(n["blocks"] for n in MOCK_NODES),
        "models": list(set(n["model"] for n in MOCK_NODES)),
        "last_updated": time.time()
    })

@app.route('/api/nodes')
def nodes():
    """API endpoint with mock node data"""
    return json.dumps(MOCK_NODES)

if __name__ == '__main__':
    print("🔬 Starting KwaaiNet Local Health Dashboard")
    print("📍 Running on http://0.0.0.0:8000")
    print("🔧 Mock data mode - no DHT connection required")
    app.run(host='0.0.0.0', port=8000, debug=True)