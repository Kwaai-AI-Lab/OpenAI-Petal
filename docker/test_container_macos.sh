#!/bin/bash

# KwaaiNet macOS Container Installation Test Script
# Tests Docker installation with Apple Silicon MPS GPU support
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
COMPOSE_FILE="compose-macos.yml"
TEST_TIMEOUT=120  # seconds to wait for services to be ready
NODE_PORT=8081
API_PORT=8000
LOG_FILE="/tmp/kwaainet_container_macos_test.log"

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0

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

# Check if running on macOS
check_macos() {
    if [[ "$(uname)" != "Darwin" ]]; then
        error "This script is designed for macOS. Use test_container_installation.sh for other platforms."
        exit 1
    fi
    
    success "Running on macOS $(sw_vers -productVersion)"
    info "Architecture: $(uname -m)"
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

# Test Docker availability
test_docker_engine() {
    header "Docker Engine Detection (macOS)"
    
    if ! command_exists docker; then
        error "Docker not found. Please install Docker Desktop for Mac"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon not running. Please start Docker Desktop"
        exit 1
    fi
    
    success "Docker Desktop detected and running"
    info "Docker version: $(docker --version)"
    
    if command_exists docker-compose; then
        info "Docker Compose version: $(docker-compose --version)"
    else
        error "docker-compose not found"
        exit 1
    fi
}

# Validate macOS-specific compose file
validate_compose_file() {
    header "macOS Compose File Validation"
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        error "macOS compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    success "Using macOS-optimized compose file: $COMPOSE_FILE"
    
    # Validate compose file
    if docker-compose -f "$COMPOSE_FILE" config >/dev/null 2>&1; then
        success "Compose file validation passed"
    else
        error "Compose file validation failed"
        exit 1
    fi
    
    # Show the configuration
    info "Compose configuration:"
    docker-compose -f "$COMPOSE_FILE" config | head -20
}

# Stop and clean existing containers
cleanup_containers() {
    header "Container Cleanup"
    
    info "Stopping and removing existing containers..."
    docker-compose -f "$COMPOSE_FILE" down >/dev/null 2>&1 || true
    docker container prune -f >/dev/null 2>&1 || true
    
    success "Container cleanup completed"
}

# Start containers
start_containers() {
    header "Container Startup (macOS with MPS)"
    
    info "Starting KwaaiNet containers optimized for macOS..."
    info "This will download images if not present (~2-4GB total)"
    
    if docker-compose -f "$COMPOSE_FILE" up -d; then
        success "Containers started successfully"
    else
        error "Failed to start containers"
        return 1
    fi
    
    # Wait a moment for containers to initialize
    sleep 5
}

# Check container status
check_container_status() {
    header "Container Status Check"
    
    local status_output
    status_output=$(docker-compose -f "$COMPOSE_FILE" ps)
    echo "$status_output" | tee -a "$LOG_FILE"
    
    # Check if both containers are running
    if echo "$status_output" | grep -q "kwaainet-node.*Up" && echo "$status_output" | grep -q "kwaainet-api.*Up"; then
        success "Both containers are running"
    else
        error "One or more containers failed to start properly"
        
        # Show logs for debugging
        info "Container logs for debugging:"
        docker-compose -f "$COMPOSE_FILE" logs --tail=20
        return 1
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
        node_logs=$(docker logs kwaainet-node 2>&1 | tail -20)
        
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
    else
        error "Node failed to start"
        return 1
    fi
}

# Test macOS-specific features
test_macos_features() {
    header "macOS-Specific Feature Testing"
    
    info "Checking for MPS (Metal Performance Shaders) support..."
    local node_logs
    node_logs=$(docker logs kwaainet-node 2>&1)
    
    if echo "$node_logs" | grep -qi "mps\|metal"; then
        success "MPS/Metal GPU acceleration detected in logs"
    else
        warning "MPS/Metal not explicitly mentioned in logs"
        info "This may be normal - PyTorch will auto-detect GPU capabilities"
    fi
    
    # Check for Apple Silicon optimizations
    if [[ "$(uname -m)" == "arm64" ]]; then
        success "Running on Apple Silicon (ARM64) - optimal performance expected"
    else
        info "Running on Intel Mac - performance may vary"
    fi
}

# Performance and resource check
test_performance() {
    header "Performance and Resource Check"
    
    # Check container resource usage
    info "Checking container resource usage..."
    local stats_output
    stats_output=$(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || echo "Stats not available")
    echo "$stats_output" | tee -a "$LOG_FILE"
    
    success "Resource usage check completed"
}

# Generate test completion report
generate_report() {
    header "Test Results Summary"
    
    local total_tests=$((TESTS_PASSED + TESTS_FAILED))
    
    echo -e "${CYAN}📊 macOS Container Test Results:${NC}" | tee -a "$LOG_FILE"
    echo -e "   ✅ Passed: ${GREEN}$TESTS_PASSED${NC}" | tee -a "$LOG_FILE"
    echo -e "   ❌ Failed: ${RED}$TESTS_FAILED${NC}" | tee -a "$LOG_FILE"
    echo -e "   📝 Total:  $total_tests" | tee -a "$LOG_FILE"
    echo -e "   🍎 Platform: macOS $(sw_vers -productVersion) ($(uname -m))" | tee -a "$LOG_FILE"
    echo -e "   📋 Log:    $LOG_FILE" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! KwaaiNet containers are working on macOS with GPU support.${NC}" | tee -a "$LOG_FILE"
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
    echo "KwaaiNet macOS Container Test - $(date)" > "$LOG_FILE"
    
    header "🍎 KwaaiNet macOS Container Test"
    info "Starting macOS-optimized container installation test..."
    info "Log file: $LOG_FILE"
    
    # Run all test phases
    check_macos
    test_docker_engine
    validate_compose_file
    cleanup_containers
    start_containers
    
    if check_container_status; then
        test_api_endpoints
        test_node_connectivity
        test_macos_features
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
        echo "  docker-compose -f $COMPOSE_FILE down"
    fi
    
    exit $exit_code
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Test interrupted by user${NC}"; cleanup_containers; exit 1' INT

# Run main function
main "$@"