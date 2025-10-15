#!/bin/bash
# KwaaiNet Docker Installation Script
# For fresh installation on a new machine

set -e

echo "🚀 KwaaiNet Docker Installation"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}ERROR: Please run as a regular user (not root)${NC}"
    echo "This script will use sudo when needed"
    exit 1
fi

# Step 1: Check/Install Podman or Docker
echo -e "${YELLOW}Step 1: Checking for container runtime${NC}"
if command -v podman &> /dev/null; then
    echo -e "${GREEN}✓ Podman is installed${NC}"
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed${NC}"
    CONTAINER_CMD="docker"
else
    echo "Container runtime not found. Installing Podman..."

    # Detect OS and install podman
    if [ -f /etc/redhat-release ]; then
        # RHEL/CentOS/Fedora
        sudo dnf install -y podman podman-compose
    elif [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y podman podman-compose
    else
        echo -e "${RED}ERROR: Unsupported OS. Please install podman or docker manually.${NC}"
        exit 1
    fi

    CONTAINER_CMD="podman"
    echo -e "${GREEN}✓ Podman installed${NC}"
fi
echo ""

# Step 2: Check for NVIDIA GPU
echo -e "${YELLOW}Step 2: Checking for NVIDIA GPU${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1

    # Check for nvidia-container-toolkit (needed for GPU access in containers)
    if [ "$CONTAINER_CMD" = "podman" ]; then
        if [ ! -f /usr/share/containers/oci/hooks.d/oci-nvidia-hook.json ] && \
           [ ! -f /usr/share/containers/oci/hooks.d/nvidia-container-toolkit.json ]; then
            echo -e "${YELLOW}⚠ NVIDIA container toolkit not found${NC}"
            echo "Installing nvidia-container-toolkit for GPU access..."

            if [ -f /etc/redhat-release ]; then
                sudo dnf install -y nvidia-container-toolkit
            elif [ -f /etc/debian_version ]; then
                sudo apt-get install -y nvidia-container-toolkit
            fi

            # Configure podman to use nvidia hook
            sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml || true
            echo -e "${GREEN}✓ NVIDIA container toolkit installed${NC}"
        else
            echo -e "${GREEN}✓ NVIDIA container toolkit is configured${NC}"
        fi
    fi

    # Enable nvidia-persistenced for reliable device creation at boot
    echo "Enabling NVIDIA persistence daemon..."
    if systemctl list-unit-files | grep -q nvidia-persistenced.service; then
        sudo systemctl enable nvidia-persistenced.service 2>/dev/null || true
        sudo systemctl start nvidia-persistenced.service 2>/dev/null || true
        echo -e "${GREEN}✓ NVIDIA persistence daemon enabled${NC}"
    else
        echo -e "${YELLOW}⚠ nvidia-persistenced not available (may need manual setup)${NC}"
    fi

    USE_GPU=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected (CPU-only mode)${NC}"
    USE_GPU=false
fi
echo ""

# Step 3: Create installation directory
echo -e "${YELLOW}Step 3: Setting up installation directory${NC}"
INSTALL_DIR="${HOME}/kwaainet"
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"
echo -e "${GREEN}✓ Created ${INSTALL_DIR}${NC}"
echo ""

# Step 4: Download compose file
echo -e "${YELLOW}Step 4: Creating Docker Compose configuration${NC}"

if [ "$USE_GPU" = true ]; then
    # GPU-enabled compose file with CDI notation
    cat > compose.yml << 'EOF'
version: "3.9"

services:
  kwaainet-node:
    image: kwaailab/kwaainet-node:latest
    container_name: kwaainet-node
    restart: unless-stopped
    environment:
      - PUBLIC_NAME=${USER:-kwaainet_user}
      - KWAAINET_BLOCKS=4
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8081:8080"
    security_opt:
      - label=disable
    devices:
      - nvidia.com/gpu=all

  kwaainet-api:
    image: kwaailab/kwaainet-api:latest
    container_name: kwaainet-api
    restart: unless-stopped
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    security_opt:
      - label=disable
EOF
else
    # CPU-only compose file
    cat > compose.yml << 'EOF'
version: "3.9"

services:
  kwaainet-node:
    image: kwaailab/kwaainet-node:latest
    container_name: kwaainet-node
    restart: unless-stopped
    environment:
      - PUBLIC_NAME=${USER:-kwaainet_user}
      - KWAAINET_BLOCKS=4
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8081:8080"
    security_opt:
      - label=disable

  kwaainet-api:
    image: kwaailab/kwaainet-api:latest
    container_name: kwaainet-api
    restart: unless-stopped
    volumes:
      - ${HOME}/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    security_opt:
      - label=disable
EOF
fi

echo -e "${GREEN}✓ Compose file created${NC}"
echo ""

# Step 5: Configure rootless podman for privileged ports
if [ "$CONTAINER_CMD" = "podman" ]; then
    echo -e "${YELLOW}Step 5: Configuring rootless podman${NC}"

    # Enable unprivileged port binding (for port 80)
    if ! grep -q "net.ipv4.ip_unprivileged_port_start" /etc/sysctl.conf 2>/dev/null; then
        echo "Enabling unprivileged port binding (port 80)..."
        echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee -a /etc/sysctl.conf
        sudo sysctl -w net.ipv4.ip_unprivileged_port_start=80
        echo -e "${GREEN}✓ Unprivileged port 80 enabled${NC}"
    else
        echo -e "${GREEN}✓ Unprivileged ports already configured${NC}"
    fi

    # Enable user lingering (containers survive logout)
    echo "Enabling user lingering for auto-start..."
    sudo loginctl enable-linger $(whoami)
    echo -e "${GREEN}✓ User lingering enabled${NC}"

    # Create systemd user service for auto-start
    echo "Creating systemd user service for auto-start..."
    mkdir -p "${HOME}/.config/systemd/user"
    cat > "${HOME}/.config/systemd/user/kwaainet-compose.service" << SERVICEEOF
[Unit]
Description=KwaaiNet Docker Compose Services
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${INSTALL_DIR}
ExecStartPre=/bin/bash -c 'for i in {1..30}; do [ -c /dev/nvidia-uvm ] && break; sleep 1; done'
ExecStart=/usr/bin/podman compose -f ${INSTALL_DIR}/compose.yml up -d
ExecStop=/usr/bin/podman compose -f ${INSTALL_DIR}/compose.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
SERVICEEOF

    # Enable and start the service
    systemctl --user daemon-reload
    systemctl --user enable kwaainet-compose.service
    echo -e "${GREEN}✓ Auto-start systemd service created${NC}"
    echo ""
fi

# Step 6: Pull container images
echo -e "${YELLOW}Step 6: Pulling container images (this may take a while)${NC}"
${CONTAINER_CMD} compose -f compose.yml pull
echo -e "${GREEN}✓ Images downloaded${NC}"
echo ""

# Step 7: Start containers
echo -e "${YELLOW}Step 7: Starting containers${NC}"
${CONTAINER_CMD} compose -f compose.yml up -d
echo -e "${GREEN}✓ Containers started${NC}"
echo ""

# Step 8: Wait for initialization and show status
echo -e "${YELLOW}Step 8: Waiting for services to initialize...${NC}"
sleep 5

echo ""
echo "Container status:"
${CONTAINER_CMD} ps

echo ""
echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Service Information:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  • API Service:  http://localhost:8000"
echo "  • Node Status:  http://localhost:8081"
echo "  • Installation: ${INSTALL_DIR}"
echo ""
echo "📝 Useful Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Check status:       ${CONTAINER_CMD} ps"
echo "  View node logs:     ${CONTAINER_CMD} logs -f kwaainet-node"
echo "  View API logs:      ${CONTAINER_CMD} logs -f kwaainet-api"
echo "  Stop services:      cd ${INSTALL_DIR} && ${CONTAINER_CMD} compose down"
echo "  Start services:     cd ${INSTALL_DIR} && ${CONTAINER_CMD} compose up -d"
echo "  Restart services:   cd ${INSTALL_DIR} && ${CONTAINER_CMD} compose restart"
echo ""
echo "🧪 Testing:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Test API:           curl http://localhost:8000/v1/models"
echo ""
echo "🌐 Network:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Your node will appear on the Petals network map shortly."
echo "  Monitor at: https://health.petals.dev/"
echo ""
echo "⚙️  Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Compose file:       ${INSTALL_DIR}/compose.yml"
echo "  Model cache:        ${HOME}/.cache/huggingface"
echo ""
echo "💡 The containers will automatically restart after system reboots."
echo ""
