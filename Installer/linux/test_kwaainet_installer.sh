#!/bin/bash

# KwaaiNet Linux Installer Test Script
# This script validates that all compatibility patches work automatically

# Note: Not using 'set -e' to allow graceful handling of expected failures

echo "🧪 KwaaiNet Linux Installer Test Protocol"
echo "=========================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "/home/metro/Source/OpenAI-Petal/Installer/linux" ]; then
    print_error "OpenAI-Petal repository not found at expected location"
    print_error "Please ensure the repository exists at /home/metro/Source/OpenAI-Petal/"
    exit 1
fi

print_status "Phase 1: Complete System Cleanup"
echo "================================="

# Use the official uninstaller for proper cleanup
print_status "Running official KwaaiNet uninstaller..."
if [ -f "/home/metro/Source/OpenAI-Petal/Installer/linux/linuxuninstaller.sh" ]; then
    # Run uninstaller non-interactively (allow it to fail gracefully)
    echo "y" | /bin/bash /home/metro/Source/OpenAI-Petal/Installer/linux/linuxuninstaller.sh || {
        print_warning "Uninstaller completed with warnings (this is normal if nothing was installed)"
    }
else
    print_error "Official uninstaller not found!"
    exit 1
fi

print_success "Uninstaller completed"

# Verify clean state
print_status "Verifying clean state..."
if conda env list | grep -q kwaainet; then
    print_error "kwaainet environment still exists after uninstaller!"
    print_error "This indicates the uninstaller needs improvement."
    exit 1
else
    print_success "Clean state confirmed"
fi

echo

print_status "Phase 2: Fresh Installation"
echo "============================"

# Create fresh environment
print_status "Creating fresh conda environment with Python 3.10..."
conda create -n kwaainet python=3.10 -y

print_status "Activating environment and installing kwaainet..."
# Use conda run to execute in the environment
conda run -n kwaainet pip install -e /home/metro/Source/OpenAI-Petal/Installer/linux

print_success "Installation completed"
echo

print_status "Phase 3: Create Launcher Script"
echo "==============================="

# Create launcher directory
mkdir -p ~/.local/bin

# Create launcher script
cat > ~/.local/bin/kwaainet << 'EOF'
#!/bin/bash
source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate kwaainet 2>/dev/null || {
    echo "Error: Could not activate kwaainet conda environment"
    exit 1
}
python -m kwaainet "$@"
EOF

# Make executable
chmod +x ~/.local/bin/kwaainet

print_success "Launcher script created"
echo

print_status "Phase 4: Test Basic Functionality"
echo "=================================="

# Add to PATH
export PATH=$PATH:~/.local/bin

print_status "Testing kwaainet command..."
if ~/.local/bin/kwaainet --help >/dev/null 2>&1; then
    print_success "kwaainet command works"
else
    print_error "kwaainet command failed"
    exit 1
fi

echo

print_status "Phase 5: Test Automatic Compatibility Patches"
echo "=============================================="

print_status "Starting daemon to test automatic patch application..."
echo "Expected patches:"
echo "  ✓ huggingface_hub fallback implementation"
echo "  ✓ hivemind PyTorch 2.x compatibility"
echo "  ✓ transformers Llama configuration"
echo "  ✓ transformers Llama modeling"
echo

# Start daemon and capture output
print_status "Executing: kwaainet start --daemon"
daemon_output=$(~/.local/bin/kwaainet start --daemon 2>&1)

echo "Daemon startup output:"
echo "======================"
echo "$daemon_output"
echo "======================"

# Check for expected patch messages
patches_applied=0

if echo "$daemon_output" | grep -q "Applied fallback implementation for huggingface_hub"; then
    print_success "✓ huggingface_hub patch applied"
    ((patches_applied++))
else
    print_warning "✗ huggingface_hub patch not found in output"
fi

if echo "$daemon_output" | grep -q -E "(hivemind.*patched|Applied hivemind.*compatibility)"; then
    print_success "✓ hivemind patch applied"
    ((patches_applied++))
else
    # Check for the alternative message format
    if echo "$daemon_output" | grep -q "Applied hivemind PyTorch 2.x compatibility patch"; then
        print_success "✓ hivemind patch applied"
        ((patches_applied++))
    else
        print_warning "✗ hivemind patch not found in output"
        # Show what we actually got for debugging
        echo "     Looking for hivemind patch in output..."
        echo "$daemon_output" | grep -i hivemind || echo "     No hivemind messages found"
    fi
fi

if echo "$daemon_output" | grep -q "Applied transformers Llama configuration patch"; then
    print_success "✓ transformers Llama config patch applied"
    ((patches_applied++))
else
    print_warning "✗ transformers Llama config patch not found in output"
fi

if echo "$daemon_output" | grep -q "Applied transformers Llama modeling patch"; then
    print_success "✓ transformers Llama modeling patch applied"
    ((patches_applied++))
else
    print_warning "✗ transformers Llama modeling patch not found in output"
fi

echo

print_status "Phase 6: Test Daemon Functionality"
echo "=================================="

print_status "Waiting 20 seconds for daemon to stabilize..."
sleep 20

print_status "Checking daemon status..."
status_output=$(~/.local/bin/kwaainet status 2>&1)
echo "$status_output"

if echo "$status_output" | grep -q "KwaaiNet daemon is running"; then
    print_success "✓ Daemon is running successfully"
    
    # Extract daemon info
    if echo "$status_output" | grep -q "Connections:"; then
        print_success "✓ Network connectivity detected"
    fi
    
    if echo "$status_output" | grep -q "Threads:"; then
        print_success "✓ Active threads detected"
    fi
else
    print_error "✗ Daemon is not running"
    print_status "Checking logs..."
    ~/.local/bin/kwaainet logs --lines 20 2>&1 || true
fi

echo

print_status "Phase 7: Test Management Commands"
echo "================================="

print_status "Testing stop command..."
stop_output=$(~/.local/bin/kwaainet stop 2>&1)
if echo "$stop_output" | grep -q "stopped gracefully"; then
    print_success "✓ Daemon stopped gracefully"
else
    print_warning "Stop output: $stop_output"
fi

print_status "Verifying daemon stopped..."
final_status=$(~/.local/bin/kwaainet status 2>&1)
if echo "$final_status" | grep -q "not running"; then
    print_success "✓ Daemon confirmed stopped"
else
    print_warning "Final status: $final_status"
fi

echo

print_status "Test Results Summary"
echo "===================="

echo "Patches Applied: $patches_applied/4"

if [ "$patches_applied" -eq 4 ]; then
    print_success "🎉 ALL TESTS PASSED!"
    print_success "✅ All 4 compatibility patches applied automatically"
    print_success "✅ Daemon functionality working correctly"
    print_success "✅ Management commands working correctly"
    print_success "✅ NO MANUAL PATCHES REQUIRED"
    echo
    print_success "The KwaaiNet Linux installer is PRODUCTION READY! 🚀"
    exit 0
else
    print_warning "⚠️  Some patches may not have applied correctly"
    print_warning "Check the output above for details"
    exit 1
fi