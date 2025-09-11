#!/bin/bash

# Quick KwaaiNet Container Test (Non-interactive)
# Tests the current running containers

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

NODE_PORT=8081
API_PORT=8000

success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo -e "${CYAN}🧪 Quick Container Test${NC}"

# Test API endpoint
info "Testing API endpoint..."
if curl -s "http://localhost:$API_PORT/v1/models" | grep -q "object\|data"; then
    success "API server responding correctly"
else
    error "API server not responding"
fi

# Test Node endpoint  
info "Testing Node endpoint..."
if curl -s "http://localhost:$NODE_PORT" >/dev/null 2>&1; then
    success "Node server responding"
else
    error "Node server not responding"
fi

# Check container status
info "Checking container status..."
podman-compose ps

echo -e "${GREEN}🎉 Quick test completed!${NC}"