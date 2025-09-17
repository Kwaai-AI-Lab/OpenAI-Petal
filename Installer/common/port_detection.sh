#!/bin/bash

# Automatic Port Detection and Selection for KwaaiNet
# Cross-platform port availability checking and intelligent selection

# Function to check if a specific port is available
is_port_available() {
    local port="$1"
    local timeout="${2:-2}"

    # Validate port number
    if [ "$port" -lt 1024 ] || [ "$port" -gt 65535 ]; then
        return 1
    fi

    # Cross-platform port checking
    if command -v nc >/dev/null 2>&1; then
        # Use netcat if available (most reliable)
        if ! nc -z localhost "$port" 2>/dev/null; then
            return 0  # Port is available
        fi
    elif command -v telnet >/dev/null 2>&1; then
        # Fallback to telnet
        if ! timeout "$timeout" telnet localhost "$port" </dev/null >/dev/null 2>&1; then
            return 0  # Port is available
        fi
    else
        # Fallback using /dev/tcp (bash built-in, Linux/macOS)
        if ! timeout "$timeout" bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
            return 0  # Port is available
        fi
    fi

    return 1  # Port is not available
}

# Function to get detailed port information
get_port_info() {
    local port="$1"
    local info=""

    # Try to identify what's using the port
    if command -v lsof >/dev/null 2>&1; then
        # Use lsof for detailed process information
        info=$(lsof -i ":$port" -n 2>/dev/null | grep -v COMMAND)
        if [ -n "$info" ]; then
            echo "Port $port in use:"
            echo "$info" | while read -r line; do
                local process=$(echo "$line" | awk '{print $1}')
                local pid=$(echo "$line" | awk '{print $2}')
                echo "  - Process: $process (PID: $pid)"
            done
            return 1
        fi
    elif command -v netstat >/dev/null 2>&1; then
        # Fallback to netstat
        info=$(netstat -an 2>/dev/null | grep ":$port ")
        if [ -n "$info" ]; then
            echo "Port $port in use (netstat): $info"
            return 1
        fi
    fi

    return 0  # Port appears available
}

# Function to find an available port in a range
find_available_port_in_range() {
    local start_port="${1:-8080}"
    local end_port="${2:-8130}"
    local preferred_ports="${3:-}"

    echo "🔍 Searching for available port in range $start_port-$end_port..."

    # First try preferred ports if specified
    if [ -n "$preferred_ports" ]; then
        for port in $preferred_ports; do
            if [ "$port" -ge "$start_port" ] && [ "$port" -le "$end_port" ]; then
                if is_port_available "$port"; then
                    echo "✅ Found preferred port: $port"
                    echo "$port"
                    return 0
                fi
            fi
        done
    fi

    # Then try the default starting port
    if is_port_available "$start_port"; then
        echo "✅ Default port available: $start_port"
        echo "$start_port"
        return 0
    fi

    # Search through the range
    for port in $(seq $((start_port + 1)) "$end_port"); do
        if is_port_available "$port"; then
            echo "✅ Found available port: $port"
            echo "$port"
            return 0
        fi
    done

    echo "❌ No available ports found in range $start_port-$end_port"
    return 1
}

# Function to suggest best port for KwaaiNet
suggest_kwaainet_port() {
    local preferred_port="${1:-8080}"
    local avoid_dev_ports="${2:-true}"

    echo "🎯 Finding optimal port for KwaaiNet..."

    # Define port ranges and preferences
    local kwaainet_range_start=8080
    local kwaainet_range_end=8089
    local fallback_range_start=8090
    local fallback_range_end=8199

    # Ports to avoid (common development services)
    local dev_ports=""
    if [ "$avoid_dev_ports" = "true" ]; then
        dev_ports="3000 3001 5000 8000 8888"
        echo "ℹ️ Avoiding common development ports: $dev_ports"
    fi

    # Check if preferred port is available
    if is_port_available "$preferred_port"; then
        local conflict=false
        for dev_port in $dev_ports; do
            if [ "$preferred_port" = "$dev_port" ]; then
                echo "⚠️ Warning: Port $preferred_port is commonly used for development"
                conflict=true
                break
            fi
        done

        if [ "$conflict" = "false" ]; then
            echo "✅ Preferred port available: $preferred_port"
            echo "$preferred_port"
            return 0
        fi
    else
        echo "⚠️ Preferred port $preferred_port is already in use"
        get_port_info "$preferred_port" >/dev/null
    fi

    # Try KwaaiNet-optimized range first
    echo "🔍 Searching KwaaiNet-optimized range ($kwaainet_range_start-$kwaainet_range_end)..."
    local result
    result=$(find_available_port_in_range "$kwaainet_range_start" "$kwaainet_range_end")
    if [ $? -eq 0 ]; then
        echo "$result"
        return 0
    fi

    # Try fallback range
    echo "🔍 Searching fallback range ($fallback_range_start-$fallback_range_end)..."
    result=$(find_available_port_in_range "$fallback_range_start" "$fallback_range_end")
    if [ $? -eq 0 ]; then
        echo "$result"
        return 0
    fi

    # Last resort: let OS pick a random port
    echo "🎲 Using system-assigned port..."
    local random_port
    random_port=$(get_system_assigned_port)
    if [ $? -eq 0 ]; then
        echo "✅ System assigned port: $random_port"
        echo "$random_port"
        return 0
    fi

    echo "❌ Unable to find any available port"
    return 1
}

# Function to get a system-assigned available port
get_system_assigned_port() {
    # Try using Python if available
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
" 2>/dev/null
        return $?
    elif command -v python >/dev/null 2>&1; then
        python -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
" 2>/dev/null
        return $?
    fi

    return 1
}

# Function to validate and suggest port configuration
validate_port_config() {
    local current_port="$1"
    local suggest_alternative="${2:-true}"

    echo "🔍 Validating port configuration..."
    echo "Current port: $current_port"

    if is_port_available "$current_port"; then
        echo "✅ Port $current_port is available"
        return 0
    else
        echo "❌ Port $current_port is not available"
        get_port_info "$current_port"

        if [ "$suggest_alternative" = "true" ]; then
            echo ""
            echo "💡 Suggesting alternative port..."
            local alternative
            alternative=$(suggest_kwaainet_port "$current_port")
            if [ $? -eq 0 ]; then
                echo "🔧 Recommended alternative: $alternative"
                echo "   To use this port, run:"
                echo "   kwaainet start --port $alternative"
                echo "   Or set in config: kwaainet config --set port $alternative"
            fi
        fi

        return 1
    fi
}

# Function to perform comprehensive port analysis
analyze_port_environment() {
    echo "🔍 Analyzing port environment..."
    echo ""

    # Check common service ports
    local common_ports="80 443 3000 5000 8000 8080 8888"
    echo "Common service port status:"
    for port in $common_ports; do
        if is_port_available "$port"; then
            echo "  Port $port: ✅ Available"
        else
            echo "  Port $port: ❌ In use"
        fi
    done

    echo ""

    # Suggest optimal KwaaiNet port
    echo "🎯 KwaaiNet port recommendation:"
    local recommended
    recommended=$(suggest_kwaainet_port)
    if [ $? -eq 0 ]; then
        echo "  Recommended port: $recommended"
        echo "  Ready to use: kwaainet start --port $recommended"
    else
        echo "  ⚠️ No optimal port found - system may be heavily loaded"
    fi

    echo ""

    # Show port usage summary if lsof is available
    if command -v lsof >/dev/null 2>&1; then
        echo "📊 Active network services:"
        lsof -i -n 2>/dev/null | grep LISTEN | head -10 | while read -r line; do
            local process=$(echo "$line" | awk '{print $1}')
            local port=$(echo "$line" | awk -F: '{print $NF}' | awk '{print $1}')
            echo "  $process listening on port $port"
        done
    fi
}

# Function to update KwaaiNet config with optimal port
update_kwaainet_port_config() {
    local config_file="$HOME/.kwaainet/config.yaml"
    local new_port="$1"

    if [ -z "$new_port" ]; then
        # Auto-detect optimal port
        new_port=$(suggest_kwaainet_port)
        if [ $? -ne 0 ]; then
            echo "❌ Unable to find suitable port"
            return 1
        fi
    fi

    echo "🔧 Updating KwaaiNet configuration..."
    echo "Setting port to: $new_port"

    # Create config directory if it doesn't exist
    mkdir -p "$(dirname "$config_file")"

    # Update config file
    if [ -f "$config_file" ]; then
        # Update existing config
        if command -v sed >/dev/null 2>&1; then
            # Backup original
            cp "$config_file" "${config_file}.backup"
            # Update port
            sed -i.tmp "s/^port: .*/port: $new_port/" "$config_file"
            rm -f "${config_file}.tmp"
        fi
    else
        # Create new config with optimal port
        cat > "$config_file" << EOF
# KwaaiNet Configuration (Auto-generated)
model: unsloth/Llama-3.1-8B-Instruct
blocks: 1
port: $new_port
use_gpu: true
log_level: INFO
initial_peers:
- /dns/bootstrap-1.kwaai.ai/tcp/8000/p2p/QmQhRuheeCLEsVD3RsnknM75gPDDqxAb8DhnWgro7KhaJc
- /dns/bootstrap-2.kwaai.ai/tcp/8000/p2p/Qmd3A8N5aQBATe2SYvNikaeCS9CAKN4E86jdCPacZ6RZJY
EOF
    fi

    echo "✅ Configuration updated successfully"
    echo "   Port set to: $new_port"
    echo "   Config file: $config_file"

    return 0
}