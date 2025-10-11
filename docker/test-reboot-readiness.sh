#!/bin/bash
# Test script to verify system is ready for reboot auto-start
# Run this before rebooting to ensure containers will auto-start

set -e

echo "🔍 Checking Reboot Readiness for KwaaiNet Rootless Deployment"
echo "=============================================================="
echo ""

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: User lingering
echo "1. Checking user lingering..."
LINGER=$(loginctl show-user $USER --property=Linger | cut -d= -f2)
if [ "$LINGER" = "yes" ]; then
    echo -e "${GREEN}✓ User lingering enabled${NC}"
else
    echo -e "${RED}✗ User lingering NOT enabled${NC}"
    echo "  Fix: sudo loginctl enable-linger $USER"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check 2: Unprivileged port binding
echo "2. Checking unprivileged port binding..."
PORT_START=$(sysctl -n net.ipv4.ip_unprivileged_port_start 2>/dev/null || echo "1024")
if [ "$PORT_START" -le 80 ]; then
    echo -e "${GREEN}✓ Unprivileged port 80 enabled (start=$PORT_START)${NC}"
else
    echo -e "${RED}✗ Unprivileged port 80 NOT enabled (start=$PORT_START)${NC}"
    echo "  Fix: echo 'net.ipv4.ip_unprivileged_port_start=80' | sudo tee -a /etc/sysctl.conf"
    echo "       sudo sysctl -w net.ipv4.ip_unprivileged_port_start=80"
    ERRORS=$((ERRORS + 1))
fi

# Check if persistent in sysctl.conf
if grep -q "net.ipv4.ip_unprivileged_port_start" /etc/sysctl.conf; then
    echo -e "${GREEN}✓ Setting is persistent in /etc/sysctl.conf${NC}"
else
    echo -e "${YELLOW}⚠ Setting not in /etc/sysctl.conf (won't survive reboot)${NC}"
    echo "  Fix: echo 'net.ipv4.ip_unprivileged_port_start=80' | sudo tee -a /etc/sysctl.conf"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check 3: Podman restart service
echo "3. Checking podman-restart service..."
if systemctl --user is-enabled podman-restart.service &>/dev/null; then
    echo -e "${GREEN}✓ podman-restart.service is enabled${NC}"
else
    echo -e "${YELLOW}⚠ podman-restart.service not enabled${NC}"
    echo "  Fix: systemctl --user enable podman-restart.service"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check 4: Running containers with restart policy
echo "4. Checking running containers..."
CONTAINERS=$(podman ps --format "{{.Names}}" 2>/dev/null || echo "")
if [ -z "$CONTAINERS" ]; then
    echo -e "${YELLOW}⚠ No containers currently running${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo "Found containers:"
    for container in $CONTAINERS; do
        RESTART_POLICY=$(podman inspect $container --format '{{.HostConfig.RestartPolicy.Name}}' 2>/dev/null || echo "unknown")
        PORTS=$(podman inspect $container --format '{{range $p, $conf := .NetworkSettings.Ports}}{{$p}} {{end}}' 2>/dev/null || echo "")

        if [ "$RESTART_POLICY" = "unless-stopped" ] || [ "$RESTART_POLICY" = "always" ]; then
            echo -e "  ${GREEN}✓ $container${NC} (restart: $RESTART_POLICY, ports: $PORTS)"
        else
            echo -e "  ${RED}✗ $container${NC} (restart: $RESTART_POLICY - should be 'unless-stopped')"
            echo "     Fix: Recreate container with --restart unless-stopped"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi
echo ""

# Check 5: Port bindings
echo "5. Checking critical port bindings..."
if ss -tlnp 2>/dev/null | grep -q ":80 "; then
    PORT_80_OWNER=$(ss -tlnp 2>/dev/null | grep ":80 " | head -1)
    echo -e "${GREEN}✓ Port 80 is bound${NC}"
    echo "  $PORT_80_OWNER"
else
    echo -e "${YELLOW}⚠ Port 80 is not bound${NC}"
    echo "  Expected: API container should be listening on port 80"
    WARNINGS=$((WARNINGS + 1))
fi

if ss -tlnp 2>/dev/null | grep -q ":8080 "; then
    PORT_8080_OWNER=$(ss -tlnp 2>/dev/null | grep ":8080 " | head -1)
    echo -e "${GREEN}✓ Port 8080 is bound${NC}"
    echo "  $PORT_8080_OWNER"
else
    echo -e "${YELLOW}⚠ Port 8080 is not bound${NC}"
    echo "  Expected: Node container should be listening on port 8080"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# Check 6: GPU access (if applicable)
echo "6. Checking GPU configuration..."
if command -v nvidia-smi &> /dev/null; then
    if [ -f /etc/cdi/nvidia.yaml ]; then
        echo -e "${GREEN}✓ NVIDIA GPU detected with CDI configuration${NC}"
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    else
        echo -e "${YELLOW}⚠ NVIDIA GPU detected but no CDI configuration${NC}"
        echo "  Fix: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected (CPU-only mode)${NC}"
fi
echo ""

# Summary
echo "=============================================================="
echo "Summary:"
echo "=============================================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! System is ready for reboot test.${NC}"
    echo ""
    echo "Expected behavior after reboot:"
    echo "  1. Containers will auto-start within ~30 seconds"
    echo "  2. API will be accessible on port 80"
    echo "  3. Node will be accessible on port 8080"
    echo "  4. Both will appear on public network map"
    echo ""
    echo "To reboot now: sudo reboot"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  ${WARNINGS} warnings found. System may work but not optimally.${NC}"
    echo ""
    echo "You can proceed with reboot, but some features may not work as expected."
    exit 1
else
    echo -e "${RED}❌ ${ERRORS} critical errors found. Fix these before rebooting!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}   Also ${WARNINGS} warnings to address.${NC}"
    fi
    echo ""
    echo "Containers may NOT auto-start after reboot until errors are fixed."
    exit 2
fi
