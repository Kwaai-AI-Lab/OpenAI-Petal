#!/bin/bash

# KwaaiNet Services Manager
# Manages node and API services independently or together

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Service definitions
NODE_COMPOSE="node-compose.yml"
API_COMPOSE="api-compose.yml"

success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
header() { echo -e "${CYAN}🚀 $1${NC}"; }

show_usage() {
    echo "KwaaiNet Services Manager"
    echo "========================"
    echo ""
    echo "Usage: $0 <command> [service]"
    echo ""
    echo "Commands:"
    echo "  start [node|api|all]     - Start service(s)"
    echo "  stop [node|api|all]      - Stop service(s)" 
    echo "  restart [node|api|all]   - Restart service(s)"
    echo "  status                   - Show status of all services"
    echo "  logs [node|api]          - Show logs for service"
    echo "  pull                     - Pull latest images"
    echo "  cleanup                  - Stop and remove all containers"
    echo ""
    echo "Examples:"
    echo "  $0 start node            - Start only the node service"
    echo "  $0 start api             - Start only the API service" 
    echo "  $0 start all             - Start both services"
    echo "  $0 status                - Show status of both services"
    echo "  $0 logs node             - Show node logs"
}

check_files() {
    if [ ! -f "$NODE_COMPOSE" ]; then
        error "Node compose file not found: $NODE_COMPOSE"
        exit 1
    fi
    
    if [ ! -f "$API_COMPOSE" ]; then
        error "API compose file not found: $API_COMPOSE"
        exit 1
    fi
}

start_service() {
    local service="$1"
    
    case "$service" in
        "node")
            header "Starting KwaaiNet Node Service"
            docker-compose -f "$NODE_COMPOSE" up -d
            success "Node service started"
            ;;
        "api")
            header "Starting KwaaiNet API Service"  
            docker-compose -f "$API_COMPOSE" up -d
            success "API service started"
            ;;
        "all")
            header "Starting All KwaaiNet Services"
            docker-compose -f "$NODE_COMPOSE" up -d
            docker-compose -f "$API_COMPOSE" up -d
            success "All services started"
            ;;
        *)
            error "Invalid service: $service. Use: node, api, or all"
            exit 1
            ;;
    esac
}

stop_service() {
    local service="$1"
    
    case "$service" in
        "node")
            header "Stopping KwaaiNet Node Service"
            docker-compose -f "$NODE_COMPOSE" down
            success "Node service stopped"
            ;;
        "api")
            header "Stopping KwaaiNet API Service"
            docker-compose -f "$API_COMPOSE" down  
            success "API service stopped"
            ;;
        "all")
            header "Stopping All KwaaiNet Services"
            docker-compose -f "$NODE_COMPOSE" down
            docker-compose -f "$API_COMPOSE" down
            success "All services stopped"
            ;;
        *)
            error "Invalid service: $service. Use: node, api, or all"
            exit 1
            ;;
    esac
}

show_status() {
    header "KwaaiNet Services Status"
    
    echo -e "\n${CYAN}Container Status:${NC}"
    docker ps -a --filter "name=kwaainet" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\n${CYAN}Resource Usage:${NC}"
    if docker ps --filter "name=kwaainet" --filter "status=running" -q | grep -q .; then
        docker stats $(docker ps --filter "name=kwaainet" --filter "status=running" --format "{{.Names}}") --no-stream
    else
        info "No KwaaiNet containers running"
    fi
}

show_logs() {
    local service="$1"
    
    case "$service" in
        "node")
            header "KwaaiNet Node Logs"
            docker-compose -f "$NODE_COMPOSE" logs -f
            ;;
        "api")
            header "KwaaiNet API Logs"
            docker-compose -f "$API_COMPOSE" logs -f
            ;;
        *)
            error "Invalid service: $service. Use: node or api"
            exit 1
            ;;
    esac
}

pull_images() {
    header "Pulling Latest Images"
    docker-compose -f "$NODE_COMPOSE" pull
    docker-compose -f "$API_COMPOSE" pull
    success "Images updated"
}

cleanup() {
    header "Cleaning Up KwaaiNet Services"
    docker-compose -f "$NODE_COMPOSE" down --remove-orphans
    docker-compose -f "$API_COMPOSE" down --remove-orphans
    docker container prune -f --filter "label=com.docker.compose.project=docker"
    success "Cleanup completed"
}

# Main execution
main() {
    local command="$1"
    local service="${2:-all}"
    
    # Check required files exist
    check_files
    
    case "$command" in
        "start")
            start_service "$service"
            ;;
        "stop")
            stop_service "$service"
            ;;
        "restart")
            stop_service "$service"
            sleep 2
            start_service "$service"
            ;;
        "status")
            show_status
            ;;
        "logs")
            if [ "$service" = "all" ]; then
                error "Please specify service for logs: node or api"
                exit 1
            fi
            show_logs "$service"
            ;;
        "pull")
            pull_images
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

main "$@"