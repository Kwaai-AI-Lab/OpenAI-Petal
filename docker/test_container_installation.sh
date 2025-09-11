#!/bin/bash

# KwaaiNet Container Installation Test Script
# Tests both Docker and Podman installations with comprehensive validation
# Author: KwaaiNet Team
# Version: 1.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="compose.yml"
COMPOSE_URL="https://raw.githubusercontent.com/Kwaai-AI-Lab/OpenAI-Petal/main/docker/compose.yml"
TEST_TIMEOUT=120  # seconds to wait for services to be ready
NODE_PORT=8081
API_PORT=8000
LOG_FILE="/tmp/kwaainet_container_test.log"

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
CONTAINER_ENGINE=""

# Logging functions
log() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
    ((TESTS_PASSED++))
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
    ((TESTS_FAILED++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}" | tee -a "$LOG_FILE"
}

header() {
    echo -e "${PURPLE}╭─────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${PURPLE}│$(printf "%69s" "$1" | sed 's/^/                    /; s/                    \(.*\)/                    \1/')│${NC}"
    echo -e "${PURPLE}╰─────────────────────────────────────────────────────────────────────╯${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for service to be ready
wait_for_service() {
    local service_name="$1"
    local port="$2"
    local endpoint="$3"
    local timeout="$4"
    
    info "Waiting for $service_name to be ready on port $port..."
    
    local count=0
    while [ $count -lt $timeout ]; do
        if curl -s "http://localhost:$port$endpoint" >/dev/null 2>&1; then
            success "$service_name is ready on port $port"
            return 0
        fi
        sleep 1
        ((count++))
        
        # Show progress every 10 seconds
        if [ $((count % 10)) -eq 0 ]; then
            info "Still waiting for $service_name... ($count/${timeout}s)"
        fi
    done
    
    error "$service_name failed to start within ${timeout}s"
    return 1
}

# Test container engine availability
test_container_engine() {
    header "Container Engine Detection"
    
    if command_exists podman && command_exists podman-compose; then
        CONTAINER_ENGINE="podman"
        success "Podman and podman-compose detected"
        info "Podman version: $(podman --version)"
        info "Podman-compose version: $(podman-compose --version | head -1)"
    elif command_exists docker && command_exists docker-compose; then
        CONTAINER_ENGINE="docker"
        success "Docker and docker-compose detected"
        info "Docker version: $(docker --version)"
        info "Docker-compose version: $(docker-compose --version)"
    else
        error "Neither Podman nor Docker with compose support found"
        echo -e "${YELLOW}Please install one of the following:${NC}"
        echo "  - Podman: podman + podman-compose"
        echo "  - Docker: docker + docker-compose"
        exit 1
    fi
}

# Download compose file if not present
download_compose_file() {
    header "Compose File Setup"
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        info "Downloading compose.yml from repository..."
        if curl -fsSL "$COMPOSE_URL" -o "$COMPOSE_FILE"; then
            success "Downloaded compose.yml"
        else
            error "Failed to download compose.yml"
            exit 1
        fi
    else
        success "Using existing compose.yml file"
    fi
    
    # Validate compose file
    if [ "$CONTAINER_ENGINE" = "podman" ]; then
        if podman-compose config >/dev/null 2>&1; then
            success "Compose file validation passed (Podman)"
        else
            error "Compose file validation failed (Podman)"
            exit 1
        fi
    else
        if docker-compose config >/dev/null 2>&1; then
            success "Compose file validation passed (Docker)"
        else
            error "Compose file validation failed (Docker)"
            exit 1
        fi
    fi
}

# Stop and clean existing containers
cleanup_containers() {
    header "Container Cleanup"
    
    info "Stopping and removing existing containers..."
    
    if [ "$CONTAINER_ENGINE" = "podman" ]; then
        podman-compose down >/dev/null 2>&1 || true
        # Remove any dangling containers
        podman container prune -f >/dev/null 2>&1 || true
    else
        docker-compose down >/dev/null 2>&1 || true
        # Remove any dangling containers
        docker container prune -f >/dev/null 2>&1 || true
    fi
    
    success "Container cleanup completed"
}

# Start containers
start_containers() {
    header "Container Startup"
    
    info "Starting KwaaiNet containers with $CONTAINER_ENGINE..."
    
    if [ "$CONTAINER_ENGINE" = "podman" ]; then
        if podman-compose up -d; then
            success "Containers started successfully (Podman)"
        else
            error "Failed to start containers (Podman)"
            return 1
        fi
    else
        if docker-compose up -d; then
            success "Containers started successfully (Docker)"
        else
            error "Failed to start containers (Docker)"
            return 1
        fi
    fi
    
    # Wait a moment for containers to initialize
    sleep 5
}

# Check container status
check_container_status() {
    header "Container Status Check"
    
    if [ "$CONTAINER_ENGINE" = "podman" ]; then
        local status_output
        status_output=$(podman-compose ps)
        echo "$status_output" | tee -a "$LOG_FILE"
        
        # Check if both containers are running
        if echo "$status_output" | grep -q "kwaainet-node.*Up" && echo "$status_output" | grep -q "kwaainet-api.*Up"; then
            success "Both containers are running"
        else
            error "One or more containers failed to start properly"
            return 1
        fi
    else
        local status_output
        status_output=$(docker-compose ps)
        echo "$status_output" | tee -a "$LOG_FILE"
        
        # Check if both containers are running
        if echo "$status_output" | grep -q "kwaainet-node.*Up" && echo "$status_output" | grep -q "kwaainet-api.*Up"; then
            success "Both containers are running"
        else
            error "One or more containers failed to start properly"
            return 1
        fi
    fi
}

# Test API endpoints
test_api_endpoints() {
    header "API Endpoint Testing"
    
    # Test API server
    if wait_for_service "KwaaiNet API" "$API_PORT" "/" "$TEST_TIMEOUT"; then
        # Test models endpoint
        info "Testing /v1/models endpoint..."
        local models_response
        models_response=$(curl -s "http://localhost:$API_PORT/v1/models" 2>/dev/null)
        if echo "$models_response" | grep -q "object.*list" || echo "$models_response" | grep -q "data"; then
            success "API models endpoint responding correctly"
        else
            warning "API models endpoint response format unexpected"
            info "Response: $models_response"
        fi
    else
        error "API server failed to start"
        return 1
    fi
}

# Test node connectivity
test_node_connectivity() {
    header "Node Connectivity Testing"
    
    # Test node health
    if wait_for_service "KwaaiNet Node" "$NODE_PORT" "/" "$TEST_TIMEOUT"; then
        info "Node is responding on port $NODE_PORT"
        
        # Check node logs for network connectivity
        info "Checking node logs for network connectivity..."
        local node_logs
        if [ "$CONTAINER_ENGINE" = "podman" ]; then
            node_logs=$(podman logs kwaainet-node 2>&1 | tail -20)
        else
            node_logs=$(docker logs kwaainet-node 2>&1 | tail -20)
        fi
        
        if echo "$node_logs" | grep -q "bootstrap.*kwaai.ai"; then
            success "Node is connecting to KwaaiNet bootstrap peers"
        else
            warning "Bootstrap peer connection not detected in logs"
        fi
        
        if echo "$node_logs" | grep -q "Announced.*blocks.*joining"; then
            success "Node is announcing blocks to the network"
        else
            warning "Block announcement not detected in logs"
        fi
        
        if echo "$node_logs" | grep -q "Network throughput"; then
            success "Node network throughput measurement completed"
        else
            info "Node still measuring network throughput (this is normal)"
        fi
    else
        error "Node failed to start"
        return 1
    fi
}

# Test GPU access (if available)
test_gpu_access() {
    header "GPU Access Testing"
    
    if command_exists nvidia-smi; then
        info "NVIDIA GPU detected, checking container GPU access..."
        
        local gpu_logs
        if [ "$CONTAINER_ENGINE" = "podman" ]; then
            gpu_logs=$(podman logs kwaainet-node 2>&1)
        else
            gpu_logs=$(docker logs kwaainet-node 2>&1)
        fi
        
        if echo "$gpu_logs" | grep -i "cuda.*available\|gpu.*detected\|using.*gpu"; then
            success "Container has GPU access"
        else
            warning "GPU access not detected in container logs"
            info "This may be normal if the model is running on CPU"
        fi
    else
        info "No NVIDIA GPU detected, skipping GPU tests"
    fi
}

# Performance and resource check
test_performance() {
    header "Performance and Resource Check"
    
    # Check container resource usage
    info "Checking container resource usage..."
    
    if [ "$CONTAINER_ENGINE" = "podman" ]; then
        local stats_output
        stats_output=$(podman stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "Stats not available")
        echo "$stats_output" | tee -a "$LOG_FILE"
    else
        local stats_output
        stats_output=$(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "Stats not available")
        echo "$stats_output" | tee -a "$LOG_FILE"
    fi
    
    success "Resource usage check completed"
}

# Generate test completion report
generate_report() {
    header "Test Results Summary"
    
    local total_tests=$((TESTS_PASSED + TESTS_FAILED))
    
    echo -e "${CYAN}📊 Test Results:${NC}" | tee -a "$LOG_FILE"
    echo -e "   ✅ Passed: ${GREEN}$TESTS_PASSED${NC}" | tee -a "$LOG_FILE"
    echo -e "   ❌ Failed: ${RED}$TESTS_FAILED${NC}" | tee -a "$LOG_FILE"
    echo -e "   📝 Total:  $total_tests" | tee -a "$LOG_FILE"
    echo -e "   🚀 Engine: ${BLUE}$CONTAINER_ENGINE${NC}" | tee -a "$LOG_FILE"
    echo -e "   📋 Log:    $LOG_FILE" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! KwaaiNet container installation is working correctly.${NC}" | tee -a "$LOG_FILE"
        echo -e "${CYAN}🌐 Services accessible at:${NC}" | tee -a "$LOG_FILE"
        echo -e "   • API Server: http://localhost:$API_PORT" | tee -a "$LOG_FILE"
        echo -e "   • Node Health: http://localhost:$NODE_PORT" | tee -a "$LOG_FILE"
        return 0
    else
        echo -e "${RED}❌ Some tests failed. Please check the logs for details.${NC}" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Main test execution
main() {
    # Initialize log file
    echo "KwaaiNet Container Installation Test - $(date)" > "$LOG_FILE"
    
    header "🐳 KwaaiNet Container Installation Test"
    info "Starting comprehensive container installation test..."
    info "Log file: $LOG_FILE"
    
    # Run all test phases
    test_container_engine
    download_compose_file
    cleanup_containers
    start_containers
    
    if check_container_status; then
        test_api_endpoints
        test_node_connectivity
        test_gpu_access
        test_performance
    fi
    
    # Generate final report
    generate_report
    local exit_code=$?
    
    # Cleanup option
    echo ""
    read -p "Do you want to stop the containers? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Stopping containers..."
        cleanup_containers
        success "Containers stopped"
    else
        info "Containers left running for further testing"
        echo -e "${CYAN}To stop containers later, run:${NC}"
        echo "  $CONTAINER_ENGINE-compose down"
    fi
    
    exit $exit_code
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Test interrupted by user${NC}"; cleanup_containers; exit 1' INT

# Run main function
main "$@"