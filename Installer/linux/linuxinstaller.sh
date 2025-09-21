#!/bin/bash

# KwaaiNet for Linux - One-Step Installer v0.2.18
# This script handles the entire installation process for KwaaiNet on Linux

set -e  # Exit on error

# Installer version
INSTALLER_VERSION="0.2.23"

# Set up logging
LOG_FILE="$HOME/kwaainet_install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo "=== KwaaiNet Linux Installer v$INSTALLER_VERSION ==="
echo "Installation started at: $(date)"
echo "Log file: $LOG_FILE"
echo "System: $(uname -a)"
echo ""

# Parse command line arguments
SKIP_SYSTEM_PACKAGES=false
FORCE_CONDA=false
FORCE_VENV=false
NO_BUILD_TOOLS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-system-packages)
            SKIP_SYSTEM_PACKAGES=true
            shift
            ;;
        --force-conda)
            FORCE_CONDA=true
            shift
            ;;
        --force-venv)
            FORCE_VENV=true
            shift
            ;;
        --no-build-tools)
            NO_BUILD_TOOLS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --no-system-packages  Skip system package installation (assumes all dependencies are available)"
            echo "  --force-conda         Force using conda environment instead of auto-detection"
            echo "  --force-venv          Force using virtual environment instead of auto-detection"
            echo "  --no-build-tools      Use pre-built wheels only (recommended - saves ~5GB disk space)"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================================="
echo "KwaaiNet for Linux - One-Step Installer v$INSTALLER_VERSION"
echo "=========================================================="
echo "This installer will set up KwaaiNet for sharing compute on Linux"
echo "It includes Python setup, dependencies, and environment configuration"
if [ "$SKIP_SYSTEM_PACKAGES" = true ]; then
    echo "⚠️ Skipping system package installation (--no-system-packages)"
fi
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get the appropriate pip command for the current environment
get_pip_command() {
    # If we're in a conda environment, use pip directly
    if [ "${CONDA_DEFAULT_ENV:-}" = "kwaainet" ] || [ "${PYTHON_METHOD:-}" = "conda" ]; then
        echo "pip"
    # If we're in a venv and it has its own pip, use it
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/pip" ]; then
        echo "${VIRTUAL_ENV}/bin/pip"
    # Otherwise, try to find the best system pip
    elif command_exists pip3; then
        echo "pip3"
    elif command_exists pip && ${PYTHON_CMD:-python3} -c "import sys; exit(0 if sys.version_info[0] == 3 else 1)" 2>/dev/null; then
        echo "pip"
    else
        echo "${PYTHON_CMD:-python3} -m pip"
    fi
}

# Function to show a spinner while a command runs
show_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    echo -n " "
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
        DISTRO_FAMILY=""
        
        case $DISTRO in
            ubuntu|debian|linuxmint|pop|elementary)
                DISTRO_FAMILY="debian"
                PKG_MANAGER="apt"
                PKG_UPDATE="apt update"
                PKG_INSTALL="apt install -y"
                ;;
            fedora|centos|rhel|rocky|almalinux)
                DISTRO_FAMILY="redhat"
                if command_exists dnf; then
                    PKG_MANAGER="dnf"
                    PKG_UPDATE="dnf check-update || true"
                    PKG_INSTALL="dnf install -y"
                else
                    PKG_MANAGER="yum"
                    PKG_UPDATE="yum check-update || true"
                    PKG_INSTALL="yum install -y"
                fi
                ;;
            arch|manjaro|endeavouros)
                DISTRO_FAMILY="arch"
                PKG_MANAGER="pacman"
                PKG_UPDATE="pacman -Sy"
                PKG_INSTALL="pacman -S --noconfirm"
                ;;
            opensuse*|sles)
                DISTRO_FAMILY="suse"
                PKG_MANAGER="zypper"
                PKG_UPDATE="zypper refresh"
                PKG_INSTALL="zypper install -y"
                ;;
            *)
                DISTRO_FAMILY="unknown"
                echo "⚠️ Unknown distribution: $DISTRO"
                echo "Attempting to use generic commands..."
                ;;
        esac
        
        echo "✅ Detected: $DISTRO $DISTRO_VERSION ($DISTRO_FAMILY family)"
        echo "📦 Package manager: $PKG_MANAGER"
    else
        echo "❌ Error: Cannot detect Linux distribution"
        exit 1
    fi
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        echo "⚠️ Warning: Running as root. This installer should be run as a regular user."
        echo "Some operations will use sudo when needed."
        USE_SUDO=""
    else
        USE_SUDO="sudo"
    fi
}

# Function to apply PyTorch 2.3+ compatibility patch for hivemind
apply_hivemind_pytorch_patch() {
    echo "🔧 Checking for hivemind PyTorch compatibility issues..."
    
    # Ensure we're in the right environment
    if [[ -n "$CONDA_BASE" ]]; then
        # Activate conda environment explicitly
        source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate kwaainet 2>/dev/null || true
    fi
    
    # Locate hivemind installation - try multiple Python executables
    local hivemind_path=""
    local python_candidates=("$PYTHON_EXEC" "python3" "python" "/home/metro/.conda/envs/kwaainet/bin/python")
    
    for python_cmd in "${python_candidates[@]}"; do
        if [[ -n "$python_cmd" ]] && command -v "$python_cmd" >/dev/null 2>&1; then
            echo "   Trying Python: $python_cmd"
            hivemind_path=$($python_cmd -c "
try:
    import hivemind
    import os
    print(os.path.dirname(hivemind.__file__))
except ImportError:
    print('NOT_FOUND')
except Exception as e:
    print('ERROR: ' + str(e))
" 2>/dev/null)
            echo "   Result: $hivemind_path"
            if [[ "$hivemind_path" != "NOT_FOUND" && "$hivemind_path" != ERROR* && -n "$hivemind_path" ]]; then
                echo "   ✅ Found hivemind at: $hivemind_path"
                break
            fi
        else
            echo "   Skipping: $python_cmd (not available)"
        fi
    done
    
    if [[ "$hivemind_path" == "NOT_FOUND" || -z "$hivemind_path" ]]; then
        echo "⚠️ Could not locate hivemind installation for patching"
        return 1
    fi
    
    local grad_scaler_file="$hivemind_path/optim/grad_scaler.py"
    if [[ ! -f "$grad_scaler_file" ]]; then
        echo "⚠️ Could not locate hivemind grad_scaler.py file"
        return 1
    fi
    
    # Check PyTorch version to determine if patch needed
    local pytorch_version=""
    for python_cmd in "$PYTHON_EXEC" "python3" "python" "/home/metro/.conda/envs/kwaainet/bin/python"; do
        if [[ -n "$python_cmd" ]] && command -v "$python_cmd" >/dev/null 2>&1; then
            pytorch_version=$($python_cmd -c "
try:
    import torch
    print(torch.__version__.split('+')[0])
except ImportError:
    print('NOT_FOUND')
" 2>/dev/null)
            if [[ "$pytorch_version" != "NOT_FOUND" && -n "$pytorch_version" ]]; then
                break
            fi
        fi
    done
    
    if [[ "$pytorch_version" == "NOT_FOUND" ]]; then
        echo "⚠️ PyTorch not found for compatibility check"
        return 1
    fi
    
    # Check if PyTorch 2.3+ (needs patch) - use same fallback logic
    local needs_patch=""
    for python_cmd in "$PYTHON_EXEC" "python3" "python" "/home/metro/.conda/envs/kwaainet/bin/python"; do
        if [[ -n "$python_cmd" ]] && command -v "$python_cmd" >/dev/null 2>&1; then
            needs_patch=$($python_cmd -c "
try:
    from packaging import version
    torch_ver = '$pytorch_version'
    needs = version.parse(torch_ver) >= version.parse('2.3.0')
    print('YES' if needs else 'NO')
except:
    print('UNKNOWN')
" 2>/dev/null)
            if [[ "$needs_patch" != "UNKNOWN" && -n "$needs_patch" ]]; then
                break
            fi
        fi
    done
    
    if [[ "$needs_patch" == "YES" ]]; then
        echo "   PyTorch $pytorch_version detected - applying compatibility patch..."
        
        # Create backup
        cp "$grad_scaler_file" "$grad_scaler_file.backup" 2>/dev/null
        
        # Apply patches for PyTorch 2.3+ import locations
        sed -i 's/from torch\.cuda\.amp import GradScaler as TorchGradScaler/from torch.amp import GradScaler as TorchGradScaler/' "$grad_scaler_file"
        sed -i 's/from torch\.cuda\.amp\.grad_scaler import OptState, _refresh_per_optimizer_state/from torch.amp.grad_scaler import OptState, _refresh_per_optimizer_state/' "$grad_scaler_file"
        
        # Verify patch applied correctly - use fallback logic
        local patch_success=false
        for python_cmd in "$PYTHON_EXEC" "python3" "python" "/home/metro/.conda/envs/kwaainet/bin/python"; do
            if [[ -n "$python_cmd" ]] && command -v "$python_cmd" >/dev/null 2>&1; then
                if $python_cmd -c "import hivemind; print('✅ hivemind imports successfully')" 2>/dev/null >/dev/null; then
                    patch_success=true
                    break
                fi
            fi
        done
        
        if [[ "$patch_success" == "true" ]]; then
            echo "   ✅ Compatibility patch applied successfully"
            rm -f "$grad_scaler_file.backup"  # Clean up backup
            return 0
        else
            echo "   ❌ Patch failed, restoring backup..."
            if [[ -f "$grad_scaler_file.backup" ]]; then
                mv "$grad_scaler_file.backup" "$grad_scaler_file"
            fi
            return 1
        fi
    elif [[ "$needs_patch" == "NO" ]]; then
        echo "   ✅ PyTorch $pytorch_version - no patch needed"
        return 0
    else
        echo "   ⚠️ Cannot determine if patch needed - applying anyway..."
        # Apply patch as a safeguard
        sed -i 's/from torch\.cuda\.amp import GradScaler as TorchGradScaler/from torch.amp import GradScaler as TorchGradScaler/' "$grad_scaler_file" 2>/dev/null || true
        sed -i 's/from torch\.cuda\.amp\.grad_scaler import OptState, _refresh_per_optimizer_state/from torch.amp.grad_scaler import OptState, _refresh_per_optimizer_state/' "$grad_scaler_file" 2>/dev/null || true
        return 0
    fi
}

# Function to verify package versions are correct
verify_package_versions() {
    echo "🔍 Verifying package versions..."
    
    local expected_versions=(
        "torch:2.3.1"
        "hivemind:1.1.10.post2"  
        "petals:2.2.0.post1"
        "transformers:4.43.1"
    )
    
    local all_good=true
    for package_version in "${expected_versions[@]}"; do
        local package=$(echo $package_version | cut -d: -f1)
        local expected=$(echo $package_version | cut -d: -f2)
        
        local actual=$($PYTHON_EXEC -c "
try:
    import $package
    print($package.__version__.split('+')[0])
except ImportError:
    print('NOT_FOUND')
except AttributeError:
    print('NO_VERSION')
" 2>/dev/null)
        
        if [[ "$actual" == "NOT_FOUND" ]]; then
            echo "   ❌ $package: not installed"
            all_good=false
        elif [[ "$actual" == "NO_VERSION" ]]; then
            echo "   ⚠️ $package: installed but version unknown"
        elif [[ "$actual" != "$expected"* ]]; then
            echo "   ❌ $package: expected $expected, got $actual"
            all_good=false
        else
            echo "   ✅ $package: $actual"
        fi
    done
    
    if [[ "$all_good" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to verify import compatibility
verify_import_compatibility() {
    echo "🔍 Testing import compatibility..."
    
    # Test critical imports that were failing
    local imports=(
        "torch:import torch; print(f'PyTorch {torch.__version__}')"
        "hivemind:import hivemind; print(f'hivemind {hivemind.__version__}')"
        "petals:import petals; print('petals imported successfully')"
        "transformers:from transformers import AutoModel; print('transformers imports working')"
    )
    
    local all_imports_good=true
    for import_test in "${imports[@]}"; do
        local package=$(echo "$import_test" | cut -d: -f1)
        local test_code=$(echo "$import_test" | cut -d: -f2-)
        
        if $PYTHON_EXEC -c "$test_code" 2>/dev/null >/dev/null; then
            echo "   ✅ $package imports successfully"
        else
            echo "   ❌ $package import failed"
            all_imports_good=false
        fi
    done
    
    if [[ "$all_imports_good" == "true" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to verify KwaaiNet functionality
verify_kwaainet_functionality() {
    echo "🔍 Testing KwaaiNet functionality..."
    
    # Test basic command availability
    if ! command_exists kwaainet; then
        echo "   ❌ kwaainet command not found"
        return 1
    fi
    
    # Test help command
    if kwaainet --help >/dev/null 2>&1; then
        echo "   ✅ kwaainet command accessible"
    else
        echo "   ❌ kwaainet command failed"
        return 1
    fi
    
    # Test configuration system (without starting daemon)
    if $PYTHON_EXEC -c "
import sys
sys.path.insert(0, '/home/metro/Source/OpenAI-Petal')
try:
    from kwaainet.config import load_config
    config = load_config()
    print('✅ Configuration system working')
except Exception as e:
    print(f'❌ Configuration failed: {e}')
    exit(1)
" 2>/dev/null >/dev/null; then
        echo "   ✅ Configuration system functional"
        return 0
    else
        echo "   ❌ Configuration system failed"
        return 1
    fi
}

# Master verification function
run_comprehensive_verification() {
    echo "🧪 Running comprehensive installation verification..."
    
    local tests=(
        "verify_package_versions"
        "verify_import_compatibility" 
        "verify_kwaainet_functionality"
    )
    
    local all_tests_passed=true
    for test in "${tests[@]}"; do
        if ! $test; then
            echo "❌ Verification failed at: $test"
            all_tests_passed=false
        fi
    done
    
    if [[ "$all_tests_passed" == "true" ]]; then
        echo "✅ All verification tests passed!"
        echo "🎉 Installation completed successfully and is ready for daemon startup"
        return 0
    else
        echo "⚠️ Some verification tests failed. Installation may have issues."
        return 1
    fi
}

# Function to check if system dependencies are available
check_system_deps() {
    echo "🔍 Checking system dependencies..."
    
    local missing_essential=()
    local missing_optional=()
    local missing_build=()
    
    # Check essential commands
    if ! command_exists curl; then
        missing_essential+=("curl")
    fi
    
    if ! command_exists wget; then
        missing_essential+=("wget") 
    fi
    
    if ! command_exists git; then
        missing_essential+=("git")
    fi
    
    # Find best Python version
    PYTHON_CMD=""
    PYTHON_VERSION=""
    
    # Try to find the newest Python version (check 3.12 down to 3.7)
    for ver in 3.12 3.11 3.10 3.9 3.8 3.7; do
        if command_exists "python$ver"; then
            PYTHON_CMD="python$ver"
            PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo "0.0")
            echo "✅ Found Python $PYTHON_VERSION at $PYTHON_CMD"
            break
        fi
    done
    
    # Fallback to python3 if specific versions not found
    if [ -z "$PYTHON_CMD" ] && command_exists python3; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo "0.0")
        echo "✅ Found Python $PYTHON_VERSION at python3"
    fi
    
    if [ -z "$PYTHON_CMD" ]; then
        missing_essential+=("python3")
    else
        # Check Python version
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
            echo "⚠️ Python $PYTHON_VERSION found, but Python 3.7+ is required"
            missing_essential+=("python3 (3.7+)")
        elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 7 ]; then
            echo "⚠️ Python 3.7 detected. This may not be compatible with all dependencies."
            echo "   Modern ML libraries (transformers, pytorch) typically require Python 3.8+."
            echo "   The installer will try to continue but may fail during package installation."
            echo ""
            read -p "   Continue anyway? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Installation cancelled. Please upgrade to Python 3.8+ for best compatibility."
                exit 1
            fi
        fi
        
        # Export the best Python command for later use
        export PYTHON_CMD
    fi
    
    # Check pip - any working pip interface is acceptable
    # On Ubuntu 24.04+, pip might only be available via 'python3 -m pip'
    local pip_available=false
    if command_exists pip3; then
        pip_available=true
    elif command_exists pip; then
        pip_available=true  
    elif ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
        pip_available=true
    elif ${PYTHON_CMD:-python3} -c "import ensurepip" >/dev/null 2>&1; then
        # ensurepip is available, we can bootstrap pip
        pip_available=true
    fi
    
    if [ "$pip_available" = false ]; then
        missing_essential+=("pip (python package manager)")
    fi
    
    # Check build tools (can potentially be handled by conda or skipped)
    if [ "$NO_BUILD_TOOLS" = true ]; then
        echo "ℹ️ Skipping build tools check (--no-build-tools flag)"
        echo "   Will attempt to use pre-built wheels only"
    else
        echo "🔍 Checking build tools for compiling Python packages..."
        
        # Check C/C++ compiler
        if ! command_exists gcc && ! command_exists clang; then
            missing_build+=("build tools (gcc/clang)")
            echo "   ❌ No C/C++ compiler found (needed for some Python packages)"
        else
            echo "   ✅ C/C++ compiler available"
        fi
        
        # Note: Rust compiler check removed since source compilation is disabled
        # All packages will use pre-built wheels only, eliminating need for Rust
        
        # Give specific guidance if build tools are missing
        if [ ${#missing_build[@]} -gt 0 ]; then
            echo ""
            echo "⚠️ BUILD TOOLS MISSING - This will cause compilation failures!"
            echo "   Missing: ${missing_build[*]}"
            echo ""
            echo "   🔧 IMMEDIATE OPTIONS:"
            echo "   1. Install missing tools now (recommended):"
            case $DISTRO_FAMILY in
                debian) echo "      sudo apt update && sudo apt install build-essential curl" ;;
                redhat) echo "      sudo yum groupinstall 'Development Tools' && sudo yum install curl" ;;
                arch) echo "      sudo pacman -S base-devel curl" ;;
                suse) echo "      sudo zypper install -t pattern devel_basis && sudo zypper install curl" ;;
                *) echo "      Install build tools using your distribution's package manager" ;;
            esac
            if [[ " ${missing_build[*]} " =~ "rust compiler" ]]; then
                echo "      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
                echo "      source ~/.cargo/env"
            fi
            echo ""
            echo "   2. Use pre-built packages only (may have limited functionality):"
            echo "      Re-run installer with: --no-build-tools"
            echo ""
            echo "   3. Continue anyway (will fail on packages requiring compilation)"
            echo ""
            read -p "   Continue installation? [y/N]: " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Installation cancelled. Install build tools and try again."
                exit 1
            fi
            echo "   Continuing with missing build tools..."
        fi
    fi
    
    # Check development headers (needed for some Python packages)
    if ! ${PYTHON_CMD:-python3} -c "import sysconfig; import os; print(os.path.exists(sysconfig.get_path('include')))" 2>/dev/null | grep -q True; then
        missing_optional+=("python3-dev headers")
    fi
    
    # Set global variables for different dependency types
    export MISSING_ESSENTIAL=("${missing_essential[@]}")
    export MISSING_BUILD=("${missing_build[@]}")  
    export MISSING_OPTIONAL=("${missing_optional[@]}")
    
    # Summary report
    if [ ${#missing_essential[@]} -eq 0 ]; then
        if [ ${#missing_build[@]} -gt 0 ] || [ ${#missing_optional[@]} -gt 0 ]; then
            echo "✅ Essential dependencies satisfied"
            if [ ${#missing_build[@]} -gt 0 ]; then
                echo "⚠️ Missing build tools: ${missing_build[*]}"
                echo "   💡 These can be provided by conda or you can use --no-build-tools"
            fi
            [ ${#missing_optional[@]} -gt 0 ] && echo "ℹ️ Missing optional: ${missing_optional[*]}"
            return 2  # Partial success - essential OK, build tools missing
        else
            echo "✅ All dependencies available - ready for full installation"
            return 0  # Full success
        fi
    else
        echo "❌ CRITICAL: Missing essential dependencies: ${missing_essential[*]}"
        [ ${#missing_build[@]} -gt 0 ] && echo "❌ Missing build tools: ${missing_build[*]}"
        [ ${#missing_optional[@]} -gt 0 ] && echo "⚠️ Missing optional: ${missing_optional[*]}"
        echo ""
        echo "   Cannot continue without essential dependencies."
        case $DISTRO_FAMILY in
            debian) echo "   Install with: sudo apt update && sudo apt install ${missing_essential[*]}" ;;
            redhat) echo "   Install with: sudo yum install ${missing_essential[*]}" ;;
            arch) echo "   Install with: sudo pacman -S ${missing_essential[*]}" ;;
            suse) echo "   Install with: sudo zypper install ${missing_essential[*]}" ;;
        esac
        return 1  # Failure
    fi
}

# Function to install system dependencies
install_system_deps() {
    echo "📦 Installing system dependencies..."
    echo "🔍 Debug: Starting install_system_deps function"
    
    # Check what we need to install
    set +e  # Temporarily disable exit on error
    check_system_deps
    local dep_status=$?
    set -e  # Re-enable exit on error
    
    echo "🔍 Debug: Dependency check status: $dep_status"
    
    if [ $dep_status -eq 0 ]; then
        echo "✅ System dependencies already satisfied"
        return 0
    elif [ $dep_status -eq 2 ]; then
        echo "ℹ️ Essential dependencies satisfied, build tools missing"
        echo "ℹ️ Will use conda to provide build tools and avoid sudo"
        export FORCE_CONDA_FOR_BUILD_TOOLS=true
        return 0
    fi
    
    # Debug: Show what dependencies are detected as missing
    echo "🔍 Debug: Missing essential dependencies: ${MISSING_ESSENTIAL[*]:-none}"
    echo "🔍 Debug: Number of missing essential: ${#MISSING_ESSENTIAL[@]}"
    
    # Only need sudo if essential packages are missing
    if [ ${#MISSING_ESSENTIAL[@]} -gt 0 ]; then
        if ! command_exists sudo && [ "$EUID" -ne 0 ]; then
            echo "❌ Error: Essential system packages need to be installed but sudo is not available."
            echo "Please install the missing essential dependencies manually or run as root."
            echo "Required: ${MISSING_ESSENTIAL[*]}"
            echo ""
            echo "Alternatively, install these packages manually:"
            case $DISTRO_FAMILY in
                debian) echo "  sudo apt update && sudo apt install ${MISSING_ESSENTIAL[*]// / }" ;;
                redhat) echo "  sudo yum install ${MISSING_ESSENTIAL[*]// / }" ;;
                arch) echo "  sudo pacman -S ${MISSING_ESSENTIAL[*]// / }" ;;
                suse) echo "  sudo zypper install ${MISSING_ESSENTIAL[*]// / }" ;;
            esac
            exit 1
        fi
    else
        echo "ℹ️ Only build tools are missing - will use conda to provide them"
        export FORCE_CONDA_FOR_BUILD_TOOLS=true
        return 0
    fi
    
    case $DISTRO_FAMILY in
        debian)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git build-essential python3 python3-venv python3-dev pciutils
            
            # Handle pip installation for Ubuntu 24.04+
            echo "🔄 Setting up pip for Python package management..."
            if ! $USE_SUDO $PKG_INSTALL python3-pip 2>/dev/null; then
                echo "ℹ️ python3-pip package not available, using ensurepip approach..."
                # Try to install ensurepip package
                $USE_SUDO $PKG_INSTALL python3-ensurepip 2>/dev/null || true
            fi
            
            # Ensure pip is available via some method
            if ! command_exists pip3 && ! command_exists pip && ! ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
                echo "🔄 Bootstrapping pip using ensurepip..."
                ${PYTHON_CMD:-python3} -m ensurepip --upgrade 2>/dev/null || true
                # If that fails, try without --upgrade
                if ! ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
                    echo "🔄 Attempting basic ensurepip bootstrap..."
                    ${PYTHON_CMD:-python3} -m ensurepip 2>/dev/null || true
                fi
            fi
            
            # Final verification
            if ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
                echo "✅ pip is available via '${PYTHON_CMD:-python3} -m pip'"
            elif command_exists pip3; then
                echo "✅ pip3 command is available"
            elif command_exists pip; then
                echo "✅ pip command is available"
            else
                echo "⚠️ pip installation may have failed, but continuing..."
            fi
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL mesa-utils || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                if command_exists snap && $USE_SUDO snap install rustup --classic 2>/dev/null; then
                    echo "✅ Rust installed via snap"
                    export PATH="$PATH:/snap/bin"
                    /snap/bin/rustup default stable 2>/dev/null || true
                    # Verify snap installation worked
                    if ! command_exists rustc && [ -f "/snap/bin/rustc" ]; then
                        export PATH="/snap/bin:$PATH"
                    fi
                else
                    echo "📥 Installing Rust via rustup..."
                    if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable; then
                        if [ -f "$HOME/.cargo/env" ]; then
                            source "$HOME/.cargo/env"
                            echo "✅ Rust compiler installed successfully"
                            # Verify Rust is now available
                            if command_exists rustc; then
                                echo "   Rust version: $(rustc --version 2>/dev/null || echo 'unknown')"
                            fi
                        else
                            echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                            export RUST_INSTALL_FAILED=true
                        fi
                    else
                        echo "⚠️ Rust installation failed. Will use pre-built tokenizers wheels only."
                        export RUST_INSTALL_FAILED=true
                    fi
                fi
            else
                echo "✅ Rust compiler already available: $(rustc --version 2>/dev/null || echo 'unknown version')"
            fi
            ;;
        redhat)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git gcc gcc-c++ make python3 python3-pip python3-devel pciutils
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL mesa-dri-drivers || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                # Try to install rust via package manager first
                if $USE_SUDO $PKG_INSTALL rust cargo 2>/dev/null; then
                    echo "✅ Rust installed via package manager"
                else
                    echo "📥 Installing Rust via rustup..."
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    if [ -f "$HOME/.cargo/env" ]; then
                        source "$HOME/.cargo/env"
                        echo "✅ Rust compiler installed successfully"
                    else
                        echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                    fi
                fi
            fi
            ;;
        arch)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git base-devel python python-pip pciutils
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL mesa || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                if $USE_SUDO $PKG_INSTALL rust 2>/dev/null; then
                    echo "✅ Rust installed via pacman"
                else
                    echo "📥 Installing Rust via rustup..."
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    if [ -f "$HOME/.cargo/env" ]; then
                        source "$HOME/.cargo/env"
                        echo "✅ Rust compiler installed successfully"
                    else
                        echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                    fi
                fi
            fi
            ;;
        suse)
            echo "🔄 Updating package list..."
            if ! $USE_SUDO $PKG_UPDATE 2>/dev/null; then
                echo "⚠️ Failed to update package list. Continuing..."
            fi
            echo "📦 Installing packages..."
            $USE_SUDO $PKG_INSTALL curl wget git gcc gcc-c++ make python3 python3-pip python3-devel pciutils
            # GPU support packages (optional)
            $USE_SUDO $PKG_INSTALL Mesa || true
            
            # Install Rust compiler for tokenizers (if not already installed)
            if ! command_exists rustc; then
                echo "🦀 Installing Rust compiler for tokenizers..."
                if $USE_SUDO $PKG_INSTALL rust cargo 2>/dev/null; then
                    echo "✅ Rust installed via zypper"
                else
                    echo "📥 Installing Rust via rustup..."
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                    if [ -f "$HOME/.cargo/env" ]; then
                        source "$HOME/.cargo/env"
                        echo "✅ Rust compiler installed successfully"
                    else
                        echo "⚠️ Rust installation may have failed. tokenizers might need pre-built wheels."
                    fi
                fi
            fi
            ;;
        *)
            echo "❌ Unknown distribution family. Please install the following manually:"
            echo "  - curl, wget, git"
            echo "  - Python 3.8+ with pip and development headers"
            echo "  - Build tools (gcc, make)"
            exit 1
            ;;
    esac
    
    # Verify installation was successful and fix common pip issues
    echo "🔍 Verifying system dependencies installation..."
    
    # Special handling for pip3 symlink issues
    if ! command_exists pip3 && command_exists pip && ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
        echo "ℹ️ pip3 command not found but pip works with ${PYTHON_CMD:-python3}. This is normal on some systems."
    elif ! command_exists pip3 && ${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1; then
        echo "ℹ️ pip3 command not found but ${PYTHON_CMD:-python3} -m pip works. This is normal on some systems."
    fi
    
    # Final dependency check with more lenient pip detection
    set +e  # Temporarily disable exit on error
    check_system_deps
    local final_status=$?
    set -e  # Re-enable exit on error
    
    if [ $final_status -eq 1 ]; then
        # Still failing - try to provide more helpful error information
        echo "❌ System dependency installation verification failed."
        echo ""
        echo "🔍 Debugging information:"
        echo "   - pip3 command: $(command_exists pip3 && echo "✅ Available" || echo "❌ Missing")"
        echo "   - pip command: $(command_exists pip && echo "✅ Available" || echo "❌ Missing")"
        echo "   - ${PYTHON_CMD:-python3} -m pip: $(${PYTHON_CMD:-python3} -m pip --version >/dev/null 2>&1 && echo "✅ Available" || echo "❌ Missing")"
        echo ""
        echo "If pip is installed but pip3 command is missing, you can create a symlink:"
        echo "   sudo ln -sf \$(which pip) /usr/local/bin/pip3"
        echo ""
        echo "Or the installer will use 'python3 -m pip' instead of 'pip3' command."
        
        # Don't exit - allow installer to continue with python3 -m pip
        echo "⚠️ Continuing installation with available pip interface..."
    elif [ $final_status -eq 2 ]; then
        echo "✅ Essential dependencies verified (build tools will be provided by conda)"
    else
        echo "✅ All system dependencies verified successfully"
    fi
}

# Function to detect GPU
detect_gpu() {
    echo "🔍 Detecting GPU hardware..."
    
    GPU_TYPE="none"
    GPU_INFO=""
    
    # Check if lspci is available
    if ! command_exists lspci; then
        echo "⚠️ lspci command not available. Limited GPU detection."
        return 0
    fi
    
    # Check for NVIDIA GPU
    if command_exists nvidia-smi; then
        GPU_TYPE="nvidia"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "✅ NVIDIA GPU detected: $GPU_INFO"
    elif lspci 2>/dev/null | grep -i nvidia >/dev/null 2>&1; then
        GPU_TYPE="nvidia"
        GPU_INFO=$(lspci 2>/dev/null | grep -i nvidia | head -1)
        echo "✅ NVIDIA GPU detected: $GPU_INFO"
        echo "⚠️ NVIDIA drivers may not be installed. GPU acceleration might not work."
    # Check for AMD GPU
    elif command_exists rocm-smi; then
        GPU_TYPE="amd"
        GPU_INFO=$(rocm-smi --showproductname 2>/dev/null | grep "Card series" | head -1)
        echo "✅ AMD GPU detected: $GPU_INFO"
    elif lspci 2>/dev/null | grep -i amd | grep -i vga >/dev/null 2>&1; then
        GPU_TYPE="amd"
        GPU_INFO=$(lspci 2>/dev/null | grep -i amd | grep -i vga | head -1)
        echo "✅ AMD GPU detected: $GPU_INFO"
        echo "⚠️ ROCm may not be installed. GPU acceleration might not work."
    # Check for Intel GPU
    elif lspci 2>/dev/null | grep -i intel | grep -i vga >/dev/null 2>&1; then
        GPU_TYPE="intel"
        GPU_INFO=$(lspci 2>/dev/null | grep -i intel | grep -i vga | head -1)
        echo "✅ Intel GPU detected: $GPU_INFO"
    else
        echo "ℹ️ No dedicated GPU detected. Using CPU-only mode."
    fi
    
    export GPU_TYPE GPU_INFO
}

# Function to install Miniconda
install_miniconda() {
    echo "🐍 Installing Miniconda..."
    
    # Check if conda is already installed
    if command_exists conda; then
        echo "✅ Conda already installed"
        return 0
    fi
    
    # Determine architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            CONDA_ARCH="x86_64"
            ;;
        aarch64|arm64)
            CONDA_ARCH="aarch64"
            ;;
        *)
            echo "❌ Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    # Download and install Miniconda
    CONDA_INSTALLER="Miniconda3-latest-Linux-${CONDA_ARCH}.sh"
    CONDA_URL="https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}"
    
    echo "📥 Downloading Miniconda..."
    wget -q "$CONDA_URL" -O "/tmp/$CONDA_INSTALLER"
    
    echo "🔧 Installing Miniconda..."
    bash "/tmp/$CONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # Clean up
    rm "/tmp/$CONDA_INSTALLER"
    
    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" init zsh 2>/dev/null || true
    
    # Source conda for current session
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    
    echo "✅ Miniconda installed successfully"

    # Monitor space after Miniconda installation
    if command -v monitor_installation_space >/dev/null 2>&1; then
        monitor_installation_space "$HOME" "After Miniconda installation"
    fi
}

# Function to choose Python environment method
choose_python_method() {
    echo "🐍 Choosing Python environment method..."
    
    PYTHON_METHOD=""
    
    # Check if we need conda for build tools (to avoid sudo)
    if [ "${FORCE_CONDA_FOR_BUILD_TOOLS:-}" = "true" ]; then
        PYTHON_METHOD="conda"
        echo "ℹ️ Using conda to provide build tools and avoid sudo requirements"
    # Check command line flags
    elif [ "$FORCE_VENV" = true ]; then
        PYTHON_METHOD="system"
        echo "ℹ️ Using virtual environment (--force-venv)"
    elif [ "$FORCE_CONDA" = true ]; then
        PYTHON_METHOD="conda"
        echo "ℹ️ Using conda (--force-conda)"
    # Check environment variables
    elif [ "${KWAAINET_USE_SYSTEM_PYTHON:-}" = "true" ]; then
        PYTHON_METHOD="system"
        echo "ℹ️ Using system Python (KWAAINET_USE_SYSTEM_PYTHON=true)"
    elif [ "${KWAAINET_USE_CONDA:-}" = "true" ]; then
        PYTHON_METHOD="conda"
        echo "ℹ️ Using conda (KWAAINET_USE_CONDA=true)"
    else
        # Auto-detect best method
        if command_exists conda; then
            PYTHON_METHOD="conda"
            echo "✅ Using existing conda installation"
        elif ${PYTHON_CMD:-python3} --version >/dev/null 2>&1; then
            PYTHON_VERSION=$(${PYTHON_CMD:-python3} --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            # Use simple version comparison instead of bc
            MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
            MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
            if [ "$MAJOR" -gt 3 ] || [ "$MAJOR" -eq 3 -a "$MINOR" -ge 8 ]; then
                PYTHON_METHOD="system"
                echo "✅ Using system Python $PYTHON_VERSION (${PYTHON_CMD:-python3})"
            else
                echo "⚠️ System Python is too old ($PYTHON_VERSION). Installing conda..."
                PYTHON_METHOD="conda"
            fi
        else
            echo "⚠️ No suitable Python found. Installing conda..."
            PYTHON_METHOD="conda"
        fi
    fi
    
    export PYTHON_METHOD
}

# Function to ensure shell configuration file exists
ensure_shell_config() {
    local shell_name="$(basename "${SHELL}")"
    local rc_file="$HOME/.${shell_name}rc"
    
    # Create the rc file if it doesn't exist
    if [[ ! -f "$rc_file" ]]; then
        echo "Creating shell config file: $rc_file"
        touch "$rc_file"
    fi
    
    echo "$rc_file"
}

# Function to check if a line exists in a file and add it if not
add_line_if_not_exists() {
    local file="$1"
    local line="$2"
    local comment="$3"
    
    # Escape the line for grep
    local escaped_line=$(echo "$line" | sed 's/[]\/$*.^|[]/\\&/g')
    
    if ! grep -q "$escaped_line" "$file"; then
        if [ -n "$comment" ]; then
            echo "" >> "$file"
            echo "$comment" >> "$file"
        fi
        echo "$line" >> "$file"
        return 0
    fi
    return 1
}

# Function to configure CUDA library paths for bitsandbytes
configure_cuda_paths() {
    echo "🔧 Configuring CUDA library paths for bitsandbytes..."
    
    # Only configure for NVIDIA GPUs
    if [ "$GPU_TYPE" != "nvidia" ]; then
        echo "ℹ️ No NVIDIA GPU detected, skipping CUDA configuration"
        return 0
    fi
    
    # Common CUDA library locations to search
    CUDA_SEARCH_PATHS=(
        "/usr/local/cuda*/lib64"
        "/usr/local/cuda/lib64"
        "/opt/cuda*/lib64"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib64"
        "/lib/x86_64-linux-gnu"
        "/lib64"
        "$HOME/.conda/envs/*/lib"
        "$HOME/miniconda3/envs/*/lib"
        "$HOME/anaconda3/envs/*/lib"
    )
    
    CUDA_LIBS_FOUND=()
    echo "🔍 Searching for CUDA libraries..."
    
    # Search for libcudart.so
    for search_path in "${CUDA_SEARCH_PATHS[@]}"; do
        # Handle wildcards in paths
        for expanded_path in $search_path; do
            if [ -d "$expanded_path" ]; then
                if find "$expanded_path" -name "libcudart.so*" -type f 2>/dev/null | head -1 | read -r lib_path; then
                    LIB_DIR=$(dirname "$lib_path")
                    # Add to array if not already present
                    if [[ ! " ${CUDA_LIBS_FOUND[@]} " =~ " ${LIB_DIR} " ]]; then
                        CUDA_LIBS_FOUND+=("$LIB_DIR")
                        echo "✅ Found CUDA libraries in: $LIB_DIR"
                    fi
                fi
            fi
        done
    done
    
    # If no CUDA libraries found, provide guidance
    if [ ${#CUDA_LIBS_FOUND[@]} -eq 0 ]; then
        echo "⚠️ No CUDA libraries found. bitsandbytes may not work with GPU acceleration."
        echo "   To fix this issue:"
        echo "   1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
        echo "   2. Or use conda-forge CUDA packages: 'conda install cuda -c conda-forge'"
        echo "   3. Then run: 'find / -name libcudart.so* 2>/dev/null'"
        echo "   4. Add the directory to LD_LIBRARY_PATH manually"
        return 1
    fi
    
    # Create LD_LIBRARY_PATH export string
    CUDA_PATHS_STR=""
    for lib_path in "${CUDA_LIBS_FOUND[@]}"; do
        if [ -z "$CUDA_PATHS_STR" ]; then
            CUDA_PATHS_STR="$lib_path"
        else
            CUDA_PATHS_STR="$CUDA_PATHS_STR:$lib_path"
        fi
    done
    
    # Update shell configuration files
    echo "📝 Adding CUDA library paths to shell configuration..."
    
    # Get shell configuration files
    SHELL_FILES=(
        "$HOME/.zshrc"
        "$HOME/.bashrc" 
        "$HOME/.bash_profile"
        "$HOME/.profile"
    )
    
    CUDA_EXPORT_LINE="export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:$CUDA_PATHS_STR\""
    CUDA_UPDATED=false
    
    for rc_file in "${SHELL_FILES[@]}"; do
        if [ -f "$rc_file" ]; then
            # Check if any LD_LIBRARY_PATH line contains our CUDA paths
            if ! grep -q "LD_LIBRARY_PATH.*$(echo "$CUDA_PATHS_STR" | head -c 20)" "$rc_file"; then
                echo "📝 Adding CUDA paths to $rc_file"
                add_line_if_not_exists "$rc_file" "$CUDA_EXPORT_LINE" "# CUDA library paths for bitsandbytes (added by KwaaiNet installer)"
                CUDA_UPDATED=true
            else
                echo "✅ CUDA paths already configured in $rc_file"
            fi
        fi
    done
    
    # If path wasn't in any existing file, create/update the default for current shell
    if [ "$CUDA_UPDATED" = false ]; then
        SHELL_RC=$(ensure_shell_config)
        echo "📝 Adding CUDA paths to $SHELL_RC (default)"
        add_line_if_not_exists "$SHELL_RC" "$CUDA_EXPORT_LINE" "# CUDA library paths for bitsandbytes (added by KwaaiNet installer)"
    fi
    
    # Update current session LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATHS_STR"
    
    echo "✅ CUDA library paths configured successfully"
    echo "   Added paths: $CUDA_PATHS_STR"
    
    return 0
}

# Function to install tokenizers with hybrid wheel-first strategy
# This function implements a 6-stage hybrid approach:
# 1. Safe wheel versions (>=0.15.1) - guaranteed to have pre-built wheels
# 2. Specific known versions - confirmed wheel availability
# 3. Dynamic platform detection - check what wheels exist for this platform
# 4. Conda fallback - alternative package source
# 5. Any wheel version - final attempt at any available wheel
# 6. Source build - absolute last resort only
install_tokenizers_with_fallback() {
    echo "🔤 Installing tokenizers with wheel-only strategy (source compilation disabled)..."

    # Only show minimal environment info since source compilation is disabled
    echo "🔍 Environment check:"
    echo "   - Installation mode: Pre-built wheels only (saves ~5GB space)"
    
    # Note: Rust detection and environment setup removed since source compilation is disabled
    # All tokenizer installation will use pre-built wheels only
    
    # Strategy 1: Safe wheel-only versions first (prioritize reliability)
    echo "📦 Strategy 1: Installing tokenizers with guaranteed pre-built wheels..."
    if $PIP_EXEC install --only-binary=tokenizers "tokenizers>=0.15.1" 2>/dev/null; then
        echo "✅ tokenizers installed successfully (safe wheel version >=0.15.1)"
        return 0
    else
        echo "⚠️ Strategy 1 failed: No wheels available for >=0.15.1"
    fi

    # Strategy 2: Try specific known working versions with wheels
    echo "📦 Strategy 2: Trying specific versions with confirmed wheel availability..."
    for version in "0.22.0" "0.21.4" "0.21.2" "0.20.3" "0.19.1" "0.15.2" "0.15.1"; do
        echo "   Trying tokenizers==$version..."
        if $PIP_EXEC install --only-binary=tokenizers "tokenizers==$version" 2>/dev/null; then
            echo "✅ tokenizers $version installed successfully (confirmed wheel)"
            return 0
        fi
    done
    echo "⚠️ Strategy 2 failed: No specific wheel versions worked"

    # Strategy 3: Dynamic platform-specific wheel detection
    echo "📦 Strategy 3: Checking platform-specific wheel availability..."
    # Get available versions and try most recent that work
    if command -v python3 >/dev/null 2>&1; then
        # Try to get available wheel versions for this platform
        available_versions=$(python3 -c "
import subprocess
import sys
try:
    result = subprocess.run([sys.executable, '-m', 'pip', 'index', 'versions', 'tokenizers'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Available versions:' in line:
                versions = line.split('Available versions:')[1].strip()
                # Split and take first 8 versions
                version_list = [v.strip() for v in versions.split(',')][:8]
                print(' '.join(version_list))
                break
except:
    pass
" 2>/dev/null)

        if [ -n "$available_versions" ]; then
            echo "   Found available versions: $available_versions"
            for version in $available_versions; do
                version=$(echo "$version" | tr -d ' ')
                echo "   Trying platform wheel for tokenizers==$version..."
                if $PIP_EXEC install --only-binary=tokenizers "tokenizers==$version" 2>/dev/null; then
                    echo "✅ tokenizers $version installed (platform-specific wheel)"
                    return 0
                fi
            done
        fi
    fi
    echo "⚠️ Strategy 3 failed: No platform-specific wheels worked"

    # Strategy 4: Emergency conda fallback (if conda environment)
    if command -v conda >/dev/null 2>&1 && [ "${PYTHON_METHOD:-}" = "conda" ]; then
        echo "📦 Strategy 4: Emergency conda installation..."
        if conda install -y tokenizers -c conda-forge 2>/dev/null; then
            echo "✅ tokenizers installed via conda-forge"
            return 0
        else
            echo "⚠️ Strategy 4 failed: Conda installation failed"
        fi
    fi

    # Strategy 5: Try any available tokenizers version (final wheel attempt)
    echo "📦 Strategy 5: Installing any available tokenizers version (final wheel attempt)..."
    if $PIP_EXEC install --only-binary=tokenizers tokenizers 2>/dev/null; then
        echo "✅ tokenizers installed (any available version)"
        return 0
    else
        echo "⚠️ Strategy 5 failed: No tokenizers wheels available for this platform"
    fi

    # Strategy 6: Source compilation DISABLED to prevent storage issues and build failures
    echo "ℹ️ Strategy 6 skipped: Source compilation disabled (saves ~5GB disk space and prevents build failures)"
    
    # All strategies failed
    echo "❌ All 5 tokenizers wheel installation strategies failed."
    echo ""
    echo "🔧 RECOMMENDED SOLUTIONS (in order of preference):"
    echo ""
    echo "   1. PLATFORM ISSUE: Your platform may not have pre-built tokenizers wheels"
    echo "      • Check https://pypi.org/project/tokenizers/#files for wheel availability"
    echo "      • Try a different Python version (3.9, 3.10, 3.11 have better wheel support)"
    echo ""
    echo "   2. QUICK RETRY: Run with updated pip (may have better wheel resolution)"
    echo "      pip install --upgrade pip && curl -fsSL \\${INSTALLER_URL} | bash"
    echo ""
    echo "   2. PYTHON VERSION: Try a different Python version with better wheel support:"
    echo "      • Python 3.9, 3.10, or 3.11 typically have more pre-built wheels"
    echo "      • Use pyenv or conda to install an alternative Python version"
    echo ""
    echo "   3. ALTERNATIVE: Use system package manager:"
    case $DISTRO_FAMILY in
        debian) echo "      sudo apt install python3-tokenizers (if available)" ;;
        redhat) echo "      sudo yum install python3-tokenizers (if available)" ;;
        arch) echo "      sudo pacman -S python-tokenizers (if available)" ;;
    esac
    echo ""
    echo "⚠️ Installation will continue, but text processing may not work properly."
    export TOKENIZERS_INSTALL_FAILED=true
    return 1
}

# Main installation flow starts here
echo "🔍 Detecting system configuration..."

# Detect distribution
detect_distro

# Check root privileges
check_root

# Detect GPU
detect_gpu

# Install system dependencies (unless skipped)
if [ "$SKIP_SYSTEM_PACKAGES" = true ]; then
    echo "⏭️ Skipping system package installation"
    # Still check if we have the required dependencies
    if ! check_system_deps; then
        echo "❌ Error: Required system dependencies are missing."
        echo "Install them manually or run without --no-system-packages flag."
        exit 1
    fi
else
    install_system_deps
fi

# Choose Python method
choose_python_method

# Source storage check and error diagnosis functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../common/storage_check.sh" ]; then
    source "$SCRIPT_DIR/../common/storage_check.sh"
else
    echo "⚠️ Warning: Storage check functions not found - continuing without space verification"
fi

if [ -f "$SCRIPT_DIR/../common/error_diagnosis.sh" ]; then
    source "$SCRIPT_DIR/../common/error_diagnosis.sh"
else
    echo "⚠️ Warning: Error diagnosis functions not found - using basic error handling"
fi

# Check storage requirements before installation
echo ""
echo "🔍 Verifying storage space requirements..."
SKIP_STORAGE_CHECK="${SKIP_STORAGE_CHECK:-false}"
if [ "$SKIP_STORAGE_CHECK" != "true" ] && command -v check_storage_requirements >/dev/null 2>&1; then
    # Determine installation parameters for space estimation
    USE_CONDA_FOR_SPACE="true"
    INSTALL_BUILD_TOOLS="true"
    if [ "$PYTHON_METHOD" = "system" ]; then
        USE_CONDA_FOR_SPACE="false"
    fi
    if [ "$NO_BUILD_TOOLS" = "true" ]; then
        INSTALL_BUILD_TOOLS="false"
    fi

    # Check storage and exit if insufficient
    if ! check_storage_requirements "$HOME" "$USE_CONDA_FOR_SPACE" "$INSTALL_BUILD_TOOLS"; then
        echo ""
        echo "❌ Installation cannot proceed due to insufficient storage space."
        echo ""
        echo "💡 Options to continue:"
        echo "   1. Free up space using the suggestions above"
        echo "   2. Use space-saving options:"
        echo "      --no-build-tools     (saves ~1GB)"
        echo "      --force-venv         (saves ~500MB vs conda)"
        echo "   3. Skip storage check: SKIP_STORAGE_CHECK=true bash installer.sh"
        echo ""
        exit 1
    fi

    # Monitor space during installation
    monitor_installation_space "$HOME" "Pre-installation"
else
    echo "ℹ️ Storage space check skipped"
fi
echo ""

# Install Python environment if needed
if [ "$PYTHON_METHOD" = "conda" ]; then
    if ! command_exists conda; then
        install_miniconda
        # Re-source conda after installation
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/miniconda3/etc/profile.d/conda.sh"
        fi
    fi
    
    # Create environment
    echo "⚙️ Setting up KwaaiNet conda environment..."
    if ! conda info --envs | grep -q kwaainet; then
        if [ "${FORCE_CONDA_FOR_BUILD_TOOLS:-}" = "true" ]; then
            echo "📦 Installing build tools via conda to avoid sudo requirements..."
            conda create -y -n kwaainet python=3.10 gcc_linux-64 gxx_linux-64 make
            echo "✅ Created Python 3.10 environment with build tools for KwaaiNet"
        else
            conda create -y -n kwaainet python=3.10
            echo "✅ Created Python 3.10 environment for KwaaiNet"
        fi
    else
        echo "✅ Using existing kwaainet environment"
        # Add build tools if needed and missing
        if [ "${FORCE_CONDA_FOR_BUILD_TOOLS:-}" = "true" ]; then
            echo "📦 Adding build tools to existing conda environment..."
            conda install -y -n kwaainet gcc_linux-64 gxx_linux-64 make || echo "⚠️ Some build tools may already be installed"
        fi
    fi

    # Configure conda channels to avoid Terms of Service issues
    echo "🔧 Configuring conda channels..."
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    # Remove problematic Anaconda commercial channels if they exist
    conda config --env --remove channels https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda config --env --remove channels https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    echo "✅ Configured conda-forge as primary channel (avoids Terms of Service issues)"
    
    # Determine conda installation path
    CONDA_BASE=""
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif command_exists conda; then
        # Try to find conda base from existing installation
        CONDA_BASE=$(conda info --base 2>/dev/null || dirname $(dirname $(which conda)) 2>/dev/null)
    fi
    
    if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        echo "❌ Error: Cannot find conda installation or conda.sh script"
        exit 1
    fi
    
    echo "ℹ️ Using conda installation at: $CONDA_BASE"
    
    # Source conda for current session
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    
    # Activate environment
    if ! conda activate kwaainet 2>/dev/null; then
        echo "⚠️ Failed to activate kwaainet environment. Re-initializing conda..."
        "$CONDA_BASE/bin/conda" init bash
        "$CONDA_BASE/bin/conda" init zsh 2>/dev/null || true
        # Re-source conda
        . "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate kwaainet
    fi
    PYTHON_EXEC="python"
    PIP_EXEC="$(get_pip_command)"
    
elif [ "$PYTHON_METHOD" = "system" ]; then
    # Use system Python with virtual environment
    echo "⚙️ Setting up KwaaiNet virtual environment..."
    
    VENV_PATH="$HOME/.kwaainet-venv"
    if [ ! -d "$VENV_PATH" ]; then
        ${PYTHON_CMD:-python3} -m venv "$VENV_PATH"
        echo "✅ Created virtual environment for KwaaiNet"
    else
        echo "✅ Using existing virtual environment"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    PYTHON_EXEC="$VENV_PATH/bin/python"
    PIP_EXEC="$(get_pip_command)"
fi

# Test Hugging Face connectivity before proceeding
test_huggingface_connectivity() {
    echo "🌐 Testing Hugging Face model download connectivity..."
    
    # Test basic HF connectivity
    if ! curl -s --connect-timeout 10 "https://huggingface.co" > /dev/null; then
        echo "⚠️ Warning: Cannot reach huggingface.co"
        echo "   Model downloads may fail due to network connectivity issues"
        return 1
    fi
    
    # Test model file access (small config file)
    if curl -s --connect-timeout 10 "https://huggingface.co/gpt2/resolve/main/config.json" > /dev/null; then
        echo "✅ Hugging Face model download connectivity verified"
        return 0
    else
        echo "⚠️ Warning: Cannot access Hugging Face model files"
        echo "   This may be due to network restrictions or firewall settings"
        echo "   Model downloads may fail, but installation will continue"
        return 1
    fi
}

# Clear cached versions of the package
echo "🧹 Clearing any cached versions of KwaaiNet..."
$PIP_EXEC cache remove kwaainet &>/dev/null || true
$PIP_EXEC cache remove kwaainet_linux &>/dev/null || true
rm -rf /tmp/pip-* 2>/dev/null || true

# Test connectivity
test_huggingface_connectivity

# Install the package
echo "📦 Installing KwaaiNet for Linux..."

# Install basic dependencies first
$PIP_EXEC install pyyaml &>/dev/null || {
    echo "⚠️ Failed to install pyyaml. Continuing..."
}

# Install Petals
if [ "$NO_BUILD_TOOLS" = true ]; then
    echo "📦 Installing Petals from PyPI (pre-built wheels only)..."
    if $PIP_EXEC install petals --only-binary=all 2>/dev/null; then
        echo "✅ Petals installed from PyPI (pre-built)"
    else
        echo "⚠️ No pre-built Petals available. Trying regular PyPI installation..."
        if $PIP_EXEC install petals 2>/dev/null; then
            echo "✅ Petals installed from PyPI"
        else
            echo "❌ Failed to install Petals without build tools"
            echo "Try running without --no-build-tools or install build dependencies manually"
            exit 1
        fi
    fi
else
    echo "📦 Installing Petals 2.3.0.dev2 with rope_scaling support..."
    echo "   This may take several minutes as it builds from source..."

    # Try to install with progress bar, fallback to verbose if progress bar not supported
    if $PIP_EXEC install git+https://github.com/bigscience-workshop/petals.git 2>/dev/null; then
        echo "✅ Petals installed successfully from git"
    elif $PIP_EXEC install git+https://github.com/bigscience-workshop/petals.git -v 2>/dev/null; then
        echo "✅ Petals installed successfully from git (verbose mode)"
    else
        echo "⚠️ Failed to install petals from git. Trying fallback installation..."
        if $PIP_EXEC install petals 2>/dev/null || $PIP_EXEC install petals -v; then
            echo "✅ Petals installed from PyPI"
        else
            echo "⚠️ Failed to install petals. Continuing with local installation..."
        fi
    fi
fi

# Install tokenizers with comprehensive fallback handling
# Note: tokenizers installation is now handled after transformers to avoid dependency conflicts
if [ "${TOKENIZERS_INSTALL_FAILED:-false}" != true ]; then
    install_tokenizers_with_fallback
else
    echo "⚠️ Skipping separate tokenizers installation due to earlier failure"
fi

# Install compatible versions of transformers and huggingface_hub
echo "📦 Installing compatible transformers and huggingface_hub versions..."

# Check Python version for compatibility
PYTHON_VERSION=$(${PYTHON_CMD:-python3} -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo "0.0")
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

# Add --only-binary flag if no build tools
BINARY_FLAG=""
if [ "$NO_BUILD_TOOLS" = true ]; then
    BINARY_FLAG="--only-binary=all"
    echo "ℹ️ Using pre-built wheels only (--no-build-tools)"
fi

# Ensure Rust environment is available for any compilation
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -eq 7 ]; then
    echo "📦 Using Python 3.7 compatible versions..."
    # Try transformers 4.21.3 which was the last version with good Python 3.7 support
    if $PIP_EXEC install $BINARY_FLAG "transformers==4.21.3" "huggingface_hub>=0.8.0,<0.20.0"; then
        echo "✅ Successfully installed Python 3.7 compatible transformers and huggingface_hub"
    else
        echo "⚠️ Failed to install Python 3.7 compatible versions. Trying default versions..."
        if $PIP_EXEC install $BINARY_FLAG "transformers==4.34.1" "huggingface_hub==0.34.0" "tokenizers==0.14.1"; then
            echo "✅ Successfully installed default transformers and huggingface_hub"
        else
            echo "⚠️ Failed to install transformers/huggingface_hub. May have compatibility issues..."
        fi
    fi
else
    # Python 3.8+ - use fixed versions that resolve the dependency conflict
    echo "📦 Installing transformers with dependency conflict resolution..."
    
    # Use transformers 4.43.1 (Petals requirement) with compatible versions
    # This combination was tested and works (from Sept 10th CDN fix)

    # Strategy 1: Install all compatible versions together
    if $PIP_EXEC install $BINARY_FLAG "transformers==4.43.1" "huggingface_hub>=0.34.0" "tokenizers>=0.15.0"; then
        echo "✅ Strategy 1: Installed transformers 4.43.1 with compatible versions"
    elif $PIP_EXEC install $BINARY_FLAG "transformers==4.43.1" "huggingface_hub>=0.34.0"; then
        echo "✅ Strategy 2: Installed transformers 4.43.1 and huggingface_hub (tokenizers auto-resolve)"
    elif $PIP_EXEC install $BINARY_FLAG "transformers==4.43.1"; then
        echo "✅ Strategy 3: Installed transformers 4.43.1 (dependencies auto-resolve)"
        # Try to install compatible versions separately
        $PIP_EXEC install $BINARY_FLAG "huggingface_hub>=0.34.0" "tokenizers>=0.15.0" 2>/dev/null || echo "⚠️ Some dependencies may need manual resolution"
    else
        echo "❌ Failed to install transformers. This is a critical error."
        echo ""
        echo "🔧 MANUAL RESOLUTION REQUIRED:"
        echo "   The dependency conflict preventing installation is:"
        echo "   - Petals requires transformers==4.43.1"
        echo "   - transformers==4.43.1 is compatible with tokenizers>=0.15.0"
        echo "   - We need huggingface_hub>=0.34.0 for CDN compatibility"
        echo ""
        echo "   Try installing the correct versions manually:"
        echo "   pip install 'transformers==4.43.1' 'huggingface_hub>=0.34.0' 'tokenizers>=0.15.0'"
        echo ""
        echo "⚠️ Installation will continue but may have issues..."
    fi
fi

# Install from the local development version
INSTALLER_DIR="$(dirname "$0")"
if [ -d "$INSTALLER_DIR/linux" ]; then
    echo "📦 Installing from local development version..."
    if $PIP_EXEC install -e "$INSTALLER_DIR/linux/" 2>/dev/null; then
        echo "✅ KwaaiNet Linux package installed successfully (local development version)"
    else
        echo "⚠️ Failed to install local development version. Installing from GitHub..."
        
        # Install PyTorch based on GPU availability (using compatible version)
        if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
            echo "📦 Installing PyTorch 2.3.1+cu121 (CUDA version compatible with hivemind)..."
            echo "   This may take a few minutes to download..."
            if $PIP_EXEC install $BINARY_FLAG "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "torchaudio==2.3.1+cu121" --index-url https://download.pytorch.org/whl/cu121; then
                echo "✅ PyTorch CUDA 2.3.1 installed successfully"
                
                # CRITICAL: Lock PyTorch version to prevent auto-upgrade
                echo "🔒 Locking PyTorch version to prevent dependency conflicts..."
                $PIP_EXEC install --force-reinstall --no-deps "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "torchaudio==2.3.1+cu121"
                
                # Verify version lock
                PYTORCH_VERSION=$($PYTHON_EXEC -c "import torch; print(torch.__version__)" 2>/dev/null || echo "failed")
                if [[ "$PYTORCH_VERSION" == "2.3.1+cu121" ]]; then
                    echo "✅ PyTorch version locked at 2.3.1+cu121"
                else
                    echo "⚠️ PyTorch version lock may have failed: $PYTORCH_VERSION"
                fi
            else
                echo "⚠️ Failed to install CUDA PyTorch 2.3.1. Falling back to CPU version..."
                if $PIP_EXEC install $BINARY_FLAG "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" "torchaudio==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu; then
                    echo "✅ PyTorch CPU 2.3.1 installed successfully"
                    
                    # CRITICAL: Lock PyTorch version to prevent auto-upgrade
                    echo "🔒 Locking PyTorch CPU version to prevent dependency conflicts..."
                    $PIP_EXEC install --force-reinstall --no-deps "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" "torchaudio==2.3.1+cpu"
                else
                    echo "❌ Failed to install PyTorch CPU version"
                    if command -v diagnose_pip_failure >/dev/null 2>&1; then
                        diagnose_pip_failure $? "PyTorch installation failed" "torch" "$HOME"
                    else
                        echo "Please check your internet connection and available disk space."
                    fi
                    exit 1
                fi
            fi
        else
            echo "📦 Installing PyTorch 2.3.1+cpu (compatible with hivemind)..."
            echo "   This may take a few minutes to download..."
            if $PIP_EXEC install $BINARY_FLAG "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" "torchaudio==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu; then
                echo "✅ PyTorch CPU 2.3.1 installed successfully"
                
                # CRITICAL: Lock PyTorch version to prevent auto-upgrade
                echo "🔒 Locking PyTorch CPU version to prevent dependency conflicts..."
                $PIP_EXEC install --force-reinstall --no-deps "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" "torchaudio==2.3.1+cpu"
            else
                echo "❌ Failed to install PyTorch CPU version"
                if command -v diagnose_pip_failure >/dev/null 2>&1; then
                    diagnose_pip_failure $? "PyTorch installation failed" "torch" "$HOME"
                else
                    echo "Please check your internet connection and available disk space."
                fi
                exit 1
            fi
        fi
        
        
        # Install bitsandbytes for quantization support
        echo "📦 Installing bitsandbytes for quantization support..."
        if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
            echo "   Installing CUDA-compatible version for NVIDIA GPU..."
            # First try standard installation which should auto-detect CUDA
            if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
                echo "✅ bitsandbytes CUDA installed successfully"
            else
                echo "⚠️ Failed to install bitsandbytes CUDA. Quantization may not work properly."
            fi
        else
            echo "   Installing CPU version..."
            if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
                echo "✅ bitsandbytes CPU installed successfully"
            else
                echo "⚠️ Failed to install bitsandbytes. Quantization may not work properly."
            fi
        fi
        
        # Install KwaaiNet Linux package from GitHub as fallback
        echo "📦 Installing KwaaiNet Linux package from GitHub..."
        if $PIP_EXEC install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/linux"; then
            echo "✅ KwaaiNet Linux package installed successfully"
        else
            echo "❌ Failed to install KwaaiNet Linux package from GitHub"
            if command -v diagnose_pip_failure >/dev/null 2>&1; then
                diagnose_pip_failure $? "KwaaiNet package installation failed" "kwaainet" "$HOME"
            else
                echo "Please check your internet connection and available disk space."
            fi
            exit 1
        fi
    fi
else
    echo "⚠️ Local development version not found. Installing from GitHub repository..."
    
    # Install PyTorch based on GPU availability (using compatible version)
    if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
        echo "📦 Installing PyTorch 2.3.1+cu121 (CUDA version compatible with hivemind)..."
        echo "   This may take a few minutes to download..."
        if $PIP_EXEC install $BINARY_FLAG "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" "torchaudio==2.3.1+cu121" --index-url https://download.pytorch.org/whl/cu121; then
            echo "✅ PyTorch CUDA 2.3.1 installed successfully"
        else
            echo "⚠️ Failed to install CUDA PyTorch 2.3.1. Falling back to CPU version..."
            if $PIP_EXEC install $BINARY_FLAG "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" "torchaudio==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu; then
                echo "✅ PyTorch CPU 2.3.1 installed successfully"
            else
                echo "❌ Failed to install PyTorch (CUDA and CPU versions both failed)"
                if command -v diagnose_pip_failure >/dev/null 2>&1; then
                    diagnose_pip_failure $? "PyTorch installation failed" "torch" "$HOME"
                else
                    echo "Please check your internet connection and available disk space."
                fi
                exit 1
            fi
        fi
    else
        echo "📦 Installing PyTorch 2.3.1+cpu (compatible with hivemind)..."
        echo "   This may take a few minutes to download..."

        # Use monitored installation if available
        if command -v monitor_package_installation >/dev/null 2>&1; then
            if ! monitor_package_installation "PyTorch CPU" "$PIP_EXEC install $BINARY_FLAG \"torch==2.3.1+cpu\" \"torchvision==0.18.1+cpu\" \"torchaudio==2.3.1+cpu\" --index-url https://download.pytorch.org/whl/cpu" "$HOME"; then
                exit 1
            fi
        else
            # Fallback to standard installation
            if $PIP_EXEC install $BINARY_FLAG "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" "torchaudio==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu; then
                echo "✅ PyTorch CPU 2.3.1 installed successfully"
            else
                echo "❌ Failed to install PyTorch CPU version"
                if command -v diagnose_pip_failure >/dev/null 2>&1; then
                    diagnose_pip_failure $? "PyTorch installation failed" "torch" "$HOME"
                else
                    echo "Please check your internet connection and available disk space."
                fi
                exit 1
            fi
        fi
    fi

    # Monitor space after PyTorch installation
    if command -v monitor_installation_space >/dev/null 2>&1; then
        monitor_installation_space "$HOME" "After PyTorch installation"
    fi

    # Install bitsandbytes for quantization support
    echo "📦 Installing bitsandbytes for quantization support..."
    if [ "$GPU_TYPE" = "nvidia" ] && command_exists nvidia-smi; then
        echo "   Installing CUDA-compatible version for NVIDIA GPU..."
        # First try standard installation which should auto-detect CUDA
        if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
            echo "✅ bitsandbytes CUDA installed successfully"
        else
            echo "⚠️ Failed to install bitsandbytes CUDA. Quantization may not work properly."
        fi
    else
        echo "   Installing CPU version..."
        if $PIP_EXEC install $BINARY_FLAG bitsandbytes; then
            echo "✅ bitsandbytes CPU installed successfully"
        else
            echo "⚠️ Failed to install bitsandbytes. Quantization may not work properly."
        fi
    fi
    
    # Install KwaaiNet Linux package from GitHub
    echo "📦 Installing KwaaiNet Linux package..."
    if $PIP_EXEC install "git+https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git#subdirectory=Installer/linux"; then
        echo "✅ KwaaiNet Linux package installed successfully"
    else
        echo "❌ Failed to install KwaaiNet Linux package from GitHub"
        if command -v diagnose_pip_failure >/dev/null 2>&1; then
            diagnose_pip_failure $? "KwaaiNet package installation failed" "kwaainet" "$HOME"
        else
            echo "Please check your internet connection and available disk space."
        fi
        exit 1
    fi
fi

# Ensure correct hivemind version for petals compatibility
echo "📦 Installing hivemind compatible with petals (v1.1.10.post2)..."
echo "   Constraining PyTorch version to prevent auto-upgrade..."
if $PIP_EXEC install --upgrade --force-reinstall "hivemind==1.1.10.post2" "torch>=2.3.0,<2.4.0"; then
    echo "✅ hivemind 1.1.10.post2 installed (required by petals)"
    
    # Apply PyTorch 2.3+ compatibility patch for hivemind
    echo "🔧 Applying PyTorch compatibility patches for hivemind..."
    if ! apply_hivemind_pytorch_patch; then
        echo "⚠️ Patching failed, but continuing installation..."
        echo "   Manual patching may be required for full functionality"
    fi
else
    echo "⚠️ Failed to install correct hivemind version. Daemon may fail to start."
fi

# Final PyTorch version lock after all installations
echo ""
echo "🔒 Final PyTorch version lock to prevent any auto-upgrades..."
$PIP_EXEC install --force-reinstall --no-deps "torch>=2.3.0,<2.4.0" 2>/dev/null || echo "   ⚠️ Version lock may have failed"

# Final post-installation patch attempt (in case earlier patching failed)
echo ""
echo "🔧 Final compatibility check and patching..."
if ! apply_hivemind_pytorch_patch; then
    echo "⚠️ Final patching failed - manual patch may be needed for optimal performance"
fi

# Run comprehensive verification of the installation
echo ""
echo "🧪 Running installation verification..."
if run_comprehensive_verification; then
    echo "✅ Installation verification completed successfully!"
else
    echo "⚠️ Installation verification found issues - daemon may not work properly"
    echo "   Check the logs above for specific problems"
fi

# Configure CUDA library paths for bitsandbytes (NVIDIA GPUs only)
configure_cuda_paths

# Install KwaaiNet Linux package
echo "📦 Installing KwaaiNet for Linux..."

# Clear cached versions and uninstall existing packages to avoid conflicts
echo "🧹 Clearing any cached versions and removing existing packages..."
$PIP_EXEC cache remove kwaainet &>/dev/null || true
$PIP_EXEC cache remove kwaainet_linux &>/dev/null || true
$PIP_EXEC uninstall -y kwaainet kwaainet_linux &>/dev/null || true

# Download and install the current project code
echo "📦 Downloading KwaaiNet source code..."

# Create permanent installation directory
INSTALL_DIR="$HOME/.kwaainet/source"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Remove any existing installation
rm -rf OpenAI-Petal* 2>/dev/null || true

if command -v git >/dev/null 2>&1; then
    echo "📡 Cloning repository with git..."
    git clone --depth 1 https://github.com/Kwaai-AI-Lab/OpenAI-Petal.git
    PROJECT_PATH="$INSTALL_DIR/OpenAI-Petal"
else
    echo "📡 Downloading repository archive..."
    curl -L https://github.com/Kwaai-AI-Lab/OpenAI-Petal/archive/main.tar.gz -o main.tar.gz
    tar -xzf main.tar.gz
    PROJECT_PATH="$INSTALL_DIR/OpenAI-Petal-main"
fi

# Install the current project in development mode from permanent location
echo "📦 Installing KwaaiNet for Linux in development mode..."
cd "$PROJECT_PATH/Installer/linux"
$PIP_EXEC install -e .

if [ $? -eq 0 ]; then
    echo "✅ KwaaiNet Linux package installed successfully"
else
    echo "❌ Failed to install KwaaiNet Linux package"
    exit 1
fi

# Create launcher script for one-step execution
echo "🚀 Creating launcher script..."
LAUNCHER_PATH="$HOME/.local/bin/kwaainet"
mkdir -p "$HOME/.local/bin"

if [ "$PYTHON_METHOD" = "conda" ]; then
    # Conda-based launcher
    cat > "$LAUNCHER_PATH" << 'EOF'
#!/bin/bash
# KwaaiNet Launcher - Run KwaaiNet without having to activate conda first

# Find conda installation (prioritize user installations over system)
CONDA_PATH=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3"
elif command -v conda >/dev/null 2>&1; then
    # Try to get conda base, but verify it has conda.sh
    POTENTIAL_PATH=$(conda info --base 2>/dev/null)
    if [ -n "$POTENTIAL_PATH" ] && [ -f "$POTENTIAL_PATH/etc/profile.d/conda.sh" ]; then
        CONDA_PATH="$POTENTIAL_PATH"
    else
        # Fallback to directory detection
        CONDA_PATH=$(dirname $(dirname $(which conda)) 2>/dev/null)
    fi
fi

if [ -z "$CONDA_PATH" ] || [ ! -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    echo "❌ Error: Could not find conda installation with conda.sh script."
    echo "Expected locations:"
    echo "  - $HOME/miniconda3/etc/profile.d/conda.sh"
    echo "  - $HOME/anaconda3/etc/profile.d/conda.sh"
    exit 1
fi

# Source conda without changing the prompt
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Configure CUDA library paths for bitsandbytes if not already set
if command -v nvidia-smi >/dev/null 2>&1; then
    # Search for CUDA libraries in common locations
    CUDA_PATHS=""
    for search_path in "/usr/local/cuda*/lib64" "/usr/local/cuda/lib64" "/opt/cuda*/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib64" "$CONDA_PATH/envs/kwaainet/lib"; do
        for expanded_path in $search_path; do
            if [ -d "$expanded_path" ] && find "$expanded_path" -name "libcudart.so*" -type f >/dev/null 2>&1; then
                if [ -z "$CUDA_PATHS" ]; then
                    CUDA_PATHS="$expanded_path"
                else
                    CUDA_PATHS="$CUDA_PATHS:$expanded_path"
                fi
            fi
        done
    done
    
    if [ -n "$CUDA_PATHS" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_PATHS"
    fi
fi

# Activate the environment and run the command
conda activate kwaainet

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate kwaainet conda environment."
    exit 1
fi

# Run kwaainet command (installed as console script)
exec kwaainet "$@"
EOF

else
    # Virtual environment launcher
    VENV_PATH="$HOME/.kwaainet-venv"
    cat > "$LAUNCHER_PATH" << EOF
#!/bin/bash
# KwaaiNet Launcher - Run KwaaiNet from virtual environment

VENV_PATH="$VENV_PATH"

if [ ! -d "\$VENV_PATH" ]; then
    echo "❌ Error: KwaaiNet virtual environment not found at \$VENV_PATH"
    exit 1
fi

# Configure CUDA library paths for bitsandbytes if not already set
if command -v nvidia-smi >/dev/null 2>&1; then
    # Search for CUDA libraries in common locations
    CUDA_PATHS=""
    for search_path in "/usr/local/cuda*/lib64" "/usr/local/cuda/lib64" "/opt/cuda*/lib64" "/usr/lib/x86_64-linux-gnu" "/usr/lib64"; do
        for expanded_path in \$search_path; do
            if [ -d "\$expanded_path" ] && find "\$expanded_path" -name "libcudart.so*" -type f >/dev/null 2>&1; then
                if [ -z "\$CUDA_PATHS" ]; then
                    CUDA_PATHS="\$expanded_path"
                else
                    CUDA_PATHS="\$CUDA_PATHS:\$expanded_path"
                fi
            fi
        done
    done
    
    if [ -n "\$CUDA_PATHS" ]; then
        export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CUDA_PATHS"
    fi
fi

# Activate virtual environment and run the command
source "\$VENV_PATH/bin/activate"

# Check if activation was successful
if [ \$? -ne 0 ]; then
    echo "❌ Error: Failed to activate virtual environment at \$VENV_PATH"
    exit 1
fi

# Run kwaainet command (installed as console script)
exec kwaainet "\$@"
EOF
fi

chmod +x "$LAUNCHER_PATH"

# Get the shell configuration file path
SHELL_RC=$(ensure_shell_config)

# Add to PATH if not already there
echo "📝 Ensuring ~/.local/bin is in PATH..."

# Check and update common shell configuration files
SHELL_FILES=(
    "$HOME/.zshrc"
    "$HOME/.bashrc"
    "$HOME/.bash_profile"
    "$HOME/.profile"
)

PATH_UPDATED=false
for rc_file in "${SHELL_FILES[@]}"; do
    if [ -f "$rc_file" ]; then
        if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" "$rc_file"; then
            echo "📝 Adding ~/.local/bin to PATH in $rc_file"
            add_line_if_not_exists "$rc_file" 'export PATH="$HOME/.local/bin:$PATH"' "# Added by KwaaiNet installer"
            PATH_UPDATED=true
        else
            echo "✅ PATH already configured in $rc_file"
        fi
    fi
done

# If path wasn't in any existing file, create/update the default for current shell
if [ "$PATH_UPDATED" = false ]; then
    SHELL_RC=$(ensure_shell_config)
    echo "📝 Adding ~/.local/bin to PATH in $SHELL_RC (default)"
    add_line_if_not_exists "$SHELL_RC" 'export PATH="$HOME/.local/bin:$PATH"' "# Added by KwaaiNet installer"
fi

# Update current session PATH
export PATH="$HOME/.local/bin:$PATH"

# Also update the parent shell's PATH if running via curl/bash
if [ -n "$BASH_VERSION" ] && [ -f "$HOME/.bashrc" ]; then
    echo "🔄 Reloading shell configuration to update PATH..."
    set +e  # Don't exit on source errors
    source "$HOME/.bashrc" 2>/dev/null || true
    set -e  # Re-enable exit on error
fi

# Verify PATH works immediately after installation
echo "🧪 Testing kwaainet command availability..."
if command -v kwaainet >/dev/null 2>&1; then
    echo "✅ kwaainet command is available in PATH"
else
    echo "⚠️ kwaainet not found in PATH, adding current session..."
    export PATH="$HOME/.local/bin:$PATH"
    if command -v kwaainet >/dev/null 2>&1; then
        echo "✅ kwaainet command now available"
    else
        echo "❌ Failed to add kwaainet to PATH - manual shell restart may be required"
    fi
fi

# Try to create system-wide symlink if possible
if [ -w "/usr/local/bin" ]; then
    echo "📌 Creating system-wide link in /usr/local/bin..."
    ln -sf "$LAUNCHER_PATH" /usr/local/bin/kwaainet
fi

# Run initial setup
echo "⚙️ Running initial setup..."
if [ "$PYTHON_METHOD" = "conda" ]; then
    # Ensure conda is sourced and environment is active
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$CONDA_BASE/etc/profile.d/conda.sh"
        if conda activate kwaainet 2>/dev/null; then
            python -m kwaainet.runner setup 2>/dev/null || {
                echo "⚠️ Initial setup failed. You may need to run 'kwaainet setup' manually."
            }
        else
            "$LAUNCHER_PATH" setup 2>/dev/null || {
                echo "⚠️ Initial setup failed. You may need to run 'kwaainet setup' manually."
            }
        fi
    else
        "$LAUNCHER_PATH" setup 2>/dev/null || {
            echo "⚠️ Initial setup failed. You may need to run 'kwaainet setup' manually."
        }
    fi
else
    "$LAUNCHER_PATH" setup 2>/dev/null || {
        echo "⚠️ Initial setup failed. You may need to run 'kwaainet setup' manually."
    }
fi

# Final space monitoring
if command -v monitor_installation_space >/dev/null 2>&1; then
    monitor_installation_space "$HOME" "Installation completed"
fi

echo ""
echo "=========================================================="
echo "🎉 KwaaiNet for Linux installation COMPLETED!"
echo ""
echo "🔧 Installation Summary:"
echo "   - Distribution: $DISTRO $DISTRO_VERSION"
echo "   - GPU: $GPU_TYPE $([ -n "$GPU_INFO" ] && echo "($GPU_INFO)" || echo "")"
echo "   - Python method: $PYTHON_METHOD"
echo "   - PyTorch: $(conda run -n kwaainet python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'installed')"
echo "   - KwaaiNet: $(conda run -n kwaainet python -c 'import kwaainet; print("ready")' 2>/dev/null || echo 'installed')"
echo ""
echo "🚀 To use KwaaiNet:"
echo "   1. Start KwaaiNet node:      kwaainet start"
echo "   2. Or run in daemon mode:    kwaainet start --daemon"
echo "   3. Check status:             kwaainet status"
echo "   4. View help:                kwaainet --help"
echo ""
echo "💡 Quick start example:"
echo "   kwaainet start --daemon"
echo ""

# Final verification test
echo "🧪 Final verification test..."
if command -v kwaainet >/dev/null 2>&1 && kwaainet --help >/dev/null 2>&1; then
    echo "✅ Installation verification PASSED - kwaainet is ready to use!"
else
    echo "⚠️ Installation verification FAILED"
    echo "   Please restart your terminal or run: source ~/.bashrc"
    echo "   Then test with: kwaainet --help"
fi
echo ""
if [ "$GPU_TYPE" = "nvidia" ]; then
echo "🔧 NVIDIA GPU detected - CUDA acceleration configured."
echo "   CUDA library paths have been added to your shell configuration."
echo "   If you encounter CUDA errors, restart your shell or run: source ~/.bashrc"
fi
echo ""
echo "📚 Documentation: https://github.com/Kwaai-AI-Lab/OpenAI-Petal"
echo "🌐 Network map:   https://map.kwaai.ai"
echo ""
echo "📝 Installation log saved to: $LOG_FILE"
echo "   (Include this file when reporting issues)"
echo ""
echo "Installation completed at: $(date)"
echo "=========================================================="