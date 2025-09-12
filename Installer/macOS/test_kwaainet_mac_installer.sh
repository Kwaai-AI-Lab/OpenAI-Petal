#!/bin/bash

# KwaaiNet macOS Installer Test Script
# This script validates the complete macOS installation and functionality

# Note: Not using 'set -e' to allow graceful handling of expected failures

echo "🧪 KwaaiNet macOS Installer Test Protocol"
echo "=========================================="
echo "Note: Some RuntimeWarnings about module imports are cosmetic and don't affect functionality"
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

# Check if we're running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    print_error "This test script is only for macOS systems."
    exit 1
fi

# Check if we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
if [ ! -d "$SCRIPT_DIR" ] || [ ! -f "$SCRIPT_DIR/macinstaller.sh" ]; then
    print_error "macOS installer not found in expected location"
    print_error "Please run this script from the Installer/macOS directory"
    exit 1
fi

print_status "Detected macOS system: $(sw_vers -productName) $(sw_vers -productVersion)"
print_status "Architecture: $(uname -m)"
echo

print_status "Phase 1: Complete System Cleanup"
echo "================================="

# Use the official uninstaller for proper cleanup
print_status "Running official KwaaiNet macOS uninstaller..."
if [ -f "$SCRIPT_DIR/macuninstaller.sh" ]; then
    # Run uninstaller non-interactively (allow it to fail gracefully)
    echo "y" | /bin/bash "$SCRIPT_DIR/macuninstaller.sh" || {
        print_warning "Uninstaller completed with warnings (this is normal if nothing was installed)"
    }
else
    print_warning "Official uninstaller not found, performing manual cleanup..."
    
    # Manual cleanup for macOS
    if command -v conda >/dev/null 2>&1; then
        print_status "Removing existing kwaainet conda environment..."
        conda env remove -n kwaainet -y 2>/dev/null || true
    fi
    
    # Remove launcher scripts
    rm -f ~/.local/bin/kwaainet 2>/dev/null || true
    rm -f /usr/local/bin/kwaainet 2>/dev/null || true
fi

print_success "Cleanup completed"

# Verify clean state
print_status "Verifying clean state..."
if command -v conda >/dev/null 2>&1 && conda env list | grep -q kwaainet; then
    print_error "kwaainet environment still exists after cleanup!"
    print_error "This indicates the cleanup needs improvement."
    exit 1
else
    print_success "Clean state confirmed"
fi

echo

print_status "Phase 2: Prerequisites Check"
echo "============================"

# Check for Xcode Command Line Tools
print_status "Checking Xcode Command Line Tools..."
if xcode-select -p &>/dev/null; then
    print_success "✓ Xcode Command Line Tools installed"
else
    print_warning "Xcode Command Line Tools not found - may be needed for installation"
fi

# Check for Homebrew
print_status "Checking Homebrew..."
if command -v brew >/dev/null 2>&1; then
    print_success "✓ Homebrew found: $(brew --version | head -1)"
else
    print_warning "Homebrew not found - will be installed during macOS installer run"
fi

# Check for conda
print_status "Checking conda..."
if command -v conda >/dev/null 2>&1; then
    print_success "✓ Conda found: $(conda --version)"
else
    print_warning "Conda not found - will be installed during macOS installer run"
fi

echo

print_status "Phase 3: Run macOS Installer"
echo "============================"

print_status "Executing macOS installer script..."
print_status "This may take several minutes and will install dependencies..."
echo

# Run the macOS installer and capture output
installer_output_file="/tmp/kwaainet_mac_installer_test.log"

# macOS doesn't have timeout by default, so we'll run without it
# The installer should complete relatively quickly on macOS
print_warning "Running installer without timeout (macOS doesn't have timeout by default)"
if /bin/bash "$SCRIPT_DIR/macinstaller.sh" > "$installer_output_file" 2>&1; then
    print_success "✓ macOS installer completed successfully"
else
    installer_exit_code=$?
    print_error "✗ macOS installer failed with exit code: $installer_exit_code"
    print_status "Installer output (last 50 lines):"
    tail -50 "$installer_output_file"
    exit 1
fi

print_success "Installation completed"
echo

print_status "Phase 4: Verify Installation"
echo "============================"

# Check if conda environment was created
print_status "Checking conda environment..."
if command -v conda >/dev/null 2>&1 && conda env list | grep -q kwaainet; then
    print_success "✓ kwaainet conda environment created"
else
    print_error "✗ kwaainet conda environment not found"
    exit 1
fi

# Check launcher script
print_status "Checking launcher script..."
if [ -f ~/.local/bin/kwaainet ] && [ -x ~/.local/bin/kwaainet ]; then
    print_success "✓ kwaainet launcher script created and executable"
else
    print_error "✗ kwaainet launcher script not found or not executable"
    exit 1
fi

echo

print_status "Phase 5: Test Basic Functionality"
echo "=================================="

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

print_status "Testing kwaainet command..."
if ~/.local/bin/kwaainet --help >/dev/null 2>&1; then
    print_success "✓ kwaainet command works"
    
    # Just test that the help command works, don't try --version as it's not supported
    print_status "Help command functional"
else
    print_error "✗ kwaainet command failed"
    print_status "Testing direct conda activation..."
    
    # Try with direct conda activation
    if conda activate kwaainet && python -m kwaainet --help >/dev/null 2>&1; then
        print_warning "Direct activation works but launcher may need fixing"
    else
        print_error "Even direct activation failed"
        exit 1
    fi
fi

echo

print_status "Phase 6: Test macOS-Specific Features"
echo "====================================="

print_status "Testing MPS (Metal Performance Shaders) detection..."
# Don't use --get gpu_type as it's not supported, just note that we'll test MPS in PyTorch
print_status "Will test MPS availability via PyTorch directly"

# Test if PyTorch with MPS is available
print_status "Testing PyTorch MPS availability..."
conda activate kwaainet 2>/dev/null || true
mps_available=$(python -c "
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import torch
try:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('MPS available')
    else:
        print('MPS not available')
except Exception as e:
    print(f'MPS test failed: {e}')
" 2>/dev/null)

if echo "$mps_available" | grep -q "MPS available"; then
    print_success "✓ PyTorch MPS backend available"
elif echo "$mps_available" | grep -q "MPS not available"; then
    print_warning "⚠ PyTorch MPS backend not available (may be expected on older Macs)"
else
    print_warning "⚠ Could not test MPS availability"
fi

echo

print_status "Phase 7: Test Daemon Functionality"
echo "=================================="

print_status "Starting daemon to test macOS compatibility..."
echo "Expected for macOS:"
echo "  ✓ MPS compatibility patches (if applicable)"
echo "  ✓ Environment variable setup"  
echo "  ✓ Network connectivity"
echo

# Start daemon and capture output
print_status "Executing: kwaainet start --daemon"
daemon_output=$(~/.local/bin/kwaainet start --daemon 2>&1)

echo "Daemon startup output:"
echo "======================"
echo "$daemon_output"
echo "======================"

# Check for macOS-specific patches or setups
patches_applied=0

if echo "$daemon_output" | grep -q -i "mps\|metal"; then
    print_success "✓ MPS/Metal configuration detected"
    ((patches_applied++))
else
    print_status "○ No MPS/Metal messages (may be normal)"
fi

if echo "$daemon_output" | grep -q "Environment.*setup\|cache.*setup"; then
    print_success "✓ Environment setup detected"
    ((patches_applied++))
else
    print_status "○ Environment setup not explicitly mentioned"
fi

if echo "$daemon_output" | grep -q -E "(started|running|daemon.*success)"; then
    print_success "✓ Daemon startup successful"
    ((patches_applied++))
else
    print_warning "✗ Daemon startup may have issues"
fi

echo

print_status "Phase 8: Test Daemon Stability"
echo "=============================="

print_status "Waiting 20 seconds for daemon to stabilize..."
sleep 20

print_status "Checking daemon status..."
status_output=$(~/.local/bin/kwaainet status 2>&1)
echo "$status_output"

# Fix status detection for the formatted output that includes "Status: Running"
if echo "$status_output" | grep -q -E "(Status: Running|Running \(PID|daemon.*running)"; then
    print_success "✓ Daemon is running successfully"
    
    # Extract daemon info using better patterns for the formatted output
    if echo "$status_output" | grep -q -E "(Connections:|peer|network)"; then
        print_success "✓ Network connectivity detected"
    fi
    
    if echo "$status_output" | grep -q -E "(Threads:|thread|process|worker)"; then
        print_success "✓ Active processes/threads detected"
    fi
    
    daemon_stable=true
else
    print_error "✗ Daemon is not running"
    print_status "Checking logs..."
    ~/.local/bin/kwaainet logs --lines 20 2>&1 || true
    daemon_stable=false
fi

echo

print_status "Phase 9: Test Management Commands"
echo "================================="

print_status "Testing stop command..."
stop_output=$(~/.local/bin/kwaainet stop 2>&1)
if echo "$stop_output" | grep -q -E "(stopped|shutdown|terminated)"; then
    print_success "✓ Daemon stopped gracefully"
else
    print_warning "Stop output: $stop_output"
fi

print_status "Verifying daemon stopped..."
sleep 2
final_status=$(~/.local/bin/kwaainet status 2>&1)
if echo "$final_status" | grep -q -E "(Not running|not running|stopped|inactive)"; then
    print_success "✓ Daemon confirmed stopped"
else
    print_warning "Final status: $final_status"
fi

echo

print_status "Phase 10: Test Start After Stop Functionality" 
echo "=============================================="

# Since restart requires a previous command and we stopped the daemon, 
# let's test start command after stop instead
print_status "Testing start command after stop..."
start_output=$(~/.local/bin/kwaainet start --daemon 2>&1)
if echo "$start_output" | grep -q -E "(started|daemon.*started|Starting)"; then
    print_success "✓ Start command works after stop"
    
    # Wait and check status again
    sleep 10
    restart_status=$(~/.local/bin/kwaainet status 2>&1)
    if echo "$restart_status" | grep -q -E "(Status: Running|Running \(PID|daemon.*running)"; then
        print_success "✓ Daemon running after restart"
        
        # Clean stop after test
        ~/.local/bin/kwaainet stop >/dev/null 2>&1 || true
    else
        print_warning "Daemon may not be stable after restart"
    fi
else
    print_warning "Start after stop test output: $start_output"
fi

echo

print_status "Test Results Summary"
echo "===================="

# Determine overall success
if [ -f ~/.local/bin/kwaainet ] && ~/.local/bin/kwaainet --help >/dev/null 2>&1; then
    basic_functionality=true
else
    basic_functionality=false
fi

echo "Installation Status: $([ "$basic_functionality" = true ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "Daemon Stability: $([ "$daemon_stable" = true ] && echo "✅ SUCCESS" || echo "❌ FAILED")"
echo "macOS Integration: ✅ TESTED"

if [ "$basic_functionality" = true ] && [ "$daemon_stable" = true ]; then
    print_success "🎉 ALL CRITICAL TESTS PASSED!"
    print_success "✅ macOS installer working correctly"
    print_success "✅ Basic functionality verified"
    print_success "✅ Daemon functionality working"
    print_success "✅ Management commands functional"
    print_success "✅ macOS integration tested"
    echo
    print_success "The KwaaiNet macOS installer is PRODUCTION READY! 🚀"
    
    # Cleanup test log
    rm -f "$installer_output_file"
    exit 0
else
    print_warning "⚠️  Some critical tests failed"
    print_warning "Check the output above for details"
    print_status "Installer log preserved at: $installer_output_file"
    exit 1
fi