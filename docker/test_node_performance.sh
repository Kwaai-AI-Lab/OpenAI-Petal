#!/bin/bash

# KwaaiNet Node Performance Testing Script
# Tests containerized node performance on macOS

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

NODE_PORT=8081
LOG_FILE="/tmp/kwaainet_node_performance.log"

# Logging functions
success() { echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"; }
error() { echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}" | tee -a "$LOG_FILE"; }
header() {
    echo -e "${PURPLE}╭─────────────────────────────────────────────────────────────────────╮${NC}"
    echo -e "${PURPLE}│$(printf "%69s" "$1" | sed 's/^/                    /; s/                    \(.*\)/                    \1/')│${NC}"
    echo -e "${PURPLE}╰─────────────────────────────────────────────────────────────────────╯${NC}"
}

# Wait for node to be ready
wait_for_node() {
    header "Waiting for Node to be Ready"
    
    local max_wait=300  # 5 minutes
    local count=0
    
    info "Waiting for node to complete initialization..."
    
    while [ $count -lt $max_wait ]; do
        if curl -s "http://localhost:$NODE_PORT/" >/dev/null 2>&1; then
            success "Node is ready and responding"
            return 0
        fi
        
        sleep 5
        ((count += 5))
        
        if [ $((count % 30)) -eq 0 ]; then
            info "Still waiting... ($count/${max_wait}s)"
            # Show latest logs for progress
            echo "Latest node activity:"
            docker logs kwaainet-node --tail 3 | tail -3
        fi
    done
    
    error "Node failed to become ready within ${max_wait} seconds"
    return 1
}

# Test basic node health
test_node_health() {
    header "Node Health Check"
    
    local response
    response=$(curl -s "http://localhost:$NODE_PORT/" -m 10 2>/dev/null)
    
    if [ -n "$response" ]; then
        success "Node health check passed"
        info "Response: ${response:0:100}..."
    else
        error "Node health check failed"
        return 1
    fi
}

# Test node info endpoint
test_node_info() {
    header "Node Information"
    
    info "Fetching node information..."
    
    # Try to get node info (this might vary by node implementation)
    local node_logs
    node_logs=$(docker logs kwaainet-node 2>&1 | grep -E "(throughput|blocks|peers|connections)" | tail -10)
    
    if [ -n "$node_logs" ]; then
        success "Node performance info gathered"
        echo "$node_logs" | while read line; do
            info "$line"
        done
    else
        warning "No specific performance info found in logs"
    fi
}

# Measure node performance
test_node_performance() {
    header "Performance Benchmarks"
    
    info "Gathering performance metrics..."
    
    # Container resource usage
    local stats
    stats=$(docker stats kwaainet-node --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.PIDs}}")
    
    echo -e "${CYAN}Container Resource Usage:${NC}"
    echo "$stats"
    
    # Network throughput from logs
    info "Checking throughput measurements from node logs..."
    local throughput_info
    throughput_info=$(docker logs kwaainet-node 2>&1 | grep -E "throughput|tokens/sec" | tail -5)
    
    if [ -n "$throughput_info" ]; then
        success "Throughput information found"
        echo "$throughput_info" | while read line; do
            info "$line"
        done
    else
        warning "Throughput information not yet available"
    fi
}

# Test network connectivity
test_network_connectivity() {
    header "Network Connectivity"
    
    info "Checking node network status..."
    
    # Check for peer connections in logs
    local network_logs
    network_logs=$(docker logs kwaainet-node 2>&1 | grep -E "(peers|connections|bootstrap|kwaai.ai)" | tail -5)
    
    if [ -n "$network_logs" ]; then
        success "Network activity detected"
        echo "$network_logs" | while read line; do
            info "$line"
        done
    else
        warning "No network activity detected in recent logs"
    fi
}

# Test load simulation (if node is ready)
test_load_simulation() {
    header "Load Simulation (Basic)"
    
    info "Testing node responsiveness under basic load..."
    
    # Simple concurrent requests test
    local success_count=0
    local total_requests=5
    
    for i in $(seq 1 $total_requests); do
        if curl -s "http://localhost:$NODE_PORT/" -m 5 >/dev/null 2>&1; then
            ((success_count++))
        fi
        sleep 1
    done
    
    local success_rate=$((success_count * 100 / total_requests))
    
    if [ $success_rate -ge 80 ]; then
        success "Load test passed: $success_count/$total_requests requests successful ($success_rate%)"
    else
        warning "Load test concerning: only $success_count/$total_requests requests successful ($success_rate%)"
    fi
}

# Generate performance report
generate_report() {
    header "Performance Report Summary"
    
    echo -e "${CYAN}📊 KwaaiNet Node Performance Report${NC}" | tee -a "$LOG_FILE"
    echo -e "   🍎 Platform: macOS $(sw_vers -productVersion) ($(uname -m))" | tee -a "$LOG_FILE"
    echo -e "   🐳 Container: Docker Desktop" | tee -a "$LOG_FILE"
    echo -e "   📋 Log File: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Current resource usage
    echo -e "${CYAN}Current Resource Usage:${NC}" | tee -a "$LOG_FILE"
    docker stats kwaainet-node --no-stream | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Node status
    echo -e "${CYAN}Node Status:${NC}" | tee -a "$LOG_FILE"
    if curl -s "http://localhost:$NODE_PORT/" -m 5 >/dev/null 2>&1; then
        echo "   ✅ Node is responsive" | tee -a "$LOG_FILE"
    else
        echo "   ❌ Node is not responding" | tee -a "$LOG_FILE"
    fi
    
    # Show key performance metrics from logs
    local key_metrics
    key_metrics=$(docker logs kwaainet-node 2>&1 | grep -E "throughput.*tokens/sec|Inference throughput" | tail -3)
    if [ -n "$key_metrics" ]; then
        echo -e "${CYAN}Key Performance Metrics:${NC}" | tee -a "$LOG_FILE"
        echo "$key_metrics" | sed 's/^/   /' | tee -a "$LOG_FILE"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    echo -e "${GREEN}🎯 Performance testing completed!${NC}" | tee -a "$LOG_FILE"
}

# Main execution
main() {
    # Initialize log
    echo "KwaaiNet Node Performance Test - $(date)" > "$LOG_FILE"
    
    header "🚀 KwaaiNet Node Performance Test"
    info "Testing containerized node performance on macOS"
    info "Log file: $LOG_FILE"
    
    # Run all tests
    if wait_for_node; then
        test_node_health
        test_node_info
        test_node_performance
        test_network_connectivity
        test_load_simulation
    else
        error "Node failed to initialize - skipping performance tests"
    fi
    
    generate_report
}

# Run main function
main "$@"