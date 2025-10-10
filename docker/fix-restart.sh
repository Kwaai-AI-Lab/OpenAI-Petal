#!/bin/bash
# Fix KwaaiNet containers to auto-restart after reboot

set -e

echo "🔧 Enabling Podman auto-restart on boot"
echo "========================================"
echo ""

# Enable podman-restart service (makes restart policies work across reboots)
echo "Enabling podman-restart.service..."
sudo systemctl enable podman-restart.service

echo "✓ Podman restart service enabled"
echo ""

# Start containers
echo "Starting KwaaiNet containers..."
cd /home/metro/Source/OpenAI-Petal/docker
sudo podman compose -f compose.yml up -d

echo ""
echo "✓ Containers started"
echo ""

# Show status
echo "Container status:"
sudo podman ps

echo ""
echo "✅ Done! Containers will now auto-restart after system reboots."
echo ""
echo "Useful commands:"
echo "  sudo podman ps                          # Check running containers"
echo "  sudo podman logs -f kwaainet-node       # Follow node logs"
echo "  sudo podman logs -f kwaainet-api        # Follow API logs"
echo "  sudo podman compose -f compose.yml down # Stop containers"
